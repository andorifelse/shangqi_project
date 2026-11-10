"""Shared CLI/Python orchestration; algorithms live in four sibling modules."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

VERSION = "1.0.0"
STAGES = ("prepare", "capture", "compose", "intrinsics", "extrinsics", "bev")


def read_json(value):
    """Read an object or copy an in-memory JSON document."""
    if isinstance(value, dict):
        return json.loads(json.dumps(value, allow_nan=False))
    with Path(value).open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Pipeline:
    """Configure once, then run all configured stages or selected stages.

    Relative paths are resolved against the JSON file's directory. In-memory
    configurations use ``base_dir`` (or the caller's working directory).
    ``execute=False`` returns a plan and never creates files or imports CUDA.
    """

    def __init__(self, config, *, base_dir=None):
        self.config_path = None if isinstance(config, dict) else Path(config).expanduser().resolve()
        self.base_dir = (Path(base_dir).expanduser().resolve() if base_dir is not None
                         else self.config_path.parent if self.config_path else Path.cwd())
        self.config = read_json(self.config_path if self.config_path else config)
        if self.config.get("schema_version") != "calibration-pipeline.v1":
            raise ValueError("Expected schema_version=calibration-pipeline.v1")
        self.output_dir = self.path(self.config.get("output_dir", "output/run"))
        if self.output_dir == self.base_dir or self.output_dir == Path(self.output_dir.anchor):
            raise ValueError("Use a dedicated output directory")
        unknown = set(self.config.get("stages", [])) - set(STAGES)
        if unknown:
            raise ValueError("Unknown configured stages: " + ", ".join(sorted(unknown)))

    def path(self, value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.base_dir / path).resolve()

    def document(self, value):
        return read_json(value if isinstance(value, dict) else self.path(value))

    def _stages(self, stage):
        stages = (self.config.get("stages") or [s for s in STAGES
                  if ("scene" if s == "prepare" else s) in self.config]) if stage == "all" else (
                      stage.split(",") if isinstance(stage, str) else list(stage))
        if not stages or any(s not in STAGES for s in stages) or len(stages) != len(set(stages)):
            raise ValueError("Select distinct stages from: " + ", ".join(STAGES))
        if stages != sorted(stages, key=STAGES.index):
            raise ValueError("Stages must follow pipeline order")
        for name in stages:
            key = "scene" if name == "prepare" else name
            if key not in self.config:
                raise ValueError("Missing configuration section: " + key)
            settings = self.config[key]
            if not isinstance(settings, dict):
                raise ValueError("Configuration section must be an object: " + key)
            for field, choices, default in (
                    ("models", ("pinhole", "fisheye"), ["pinhole", "fisheye"]),
                    ("kinds", ("intrinsics", "extrinsics"), ["intrinsics", "extrinsics"])):
                applies = (name == "compose" or field == "models" and name == "bev" and "spec" not in settings)
                if applies:
                    values = settings.get(field, ["fisheye"] if name == "bev" else default)
                    if (not isinstance(values, list) or not values or len(values) != len(set(values))
                            or any(value not in choices for value in values)):
                        raise ValueError("%s.%s needs a nonempty distinct selection from %s" % (key, field, choices))
        return stages

    def plan(self, stage="all"):
        stages = self._stages(stage)
        inputs = []
        path_keys = {"placement", "asset_root", "repo", "render_python", "spec",
                     "intrinsics_spec", "extrinsics_spec", "base_ply", "static_ply", "board_ply",
                     "images", "intrinsics_dir", "extrinsics_dir", "solver_spec", "corners_world"}
        for section, settings in self.config.items():
            if not isinstance(settings, dict):
                continue
            for key, value in settings.items():
                if key in path_keys and isinstance(value, str):
                    p = self.path(value)
                    inputs.append({"role": section + "." + key, "path": str(p), "exists": p.exists()})
        return {"schema_version": "calibration-pipeline-plan.v1", "version": VERSION,
                "dry_run": True, "base_dir": str(self.base_dir), "output_dir": str(self.output_dir),
                "stages": stages, "inputs": inputs,
                "output_layout": {"prepare": "scene", "capture": "captures/{intrinsics,extrinsics}",
                    "compose": "images/{intrinsics,extrinsics}/{model}", "intrinsics": "intrinsics",
                    "extrinsics": "extrinsics", "bev": "bev/{model}"},
                "checks": "Static configuration only; execute validates numerical/data prerequisites.",
                "calibration_claim": "Execution is not proof of unique or physically valid calibration."}

    def run(self, stage="all", *, execute=False):
        if type(execute) is not bool:
            raise ValueError("execute must be a boolean")
        plan = self.plan(stage)
        if not execute:
            return plan
        if str(self.config.get("execution_status", "")).startswith(("paused", "draft_pending")):
            raise ValueError("Configuration is paused or pending prerequisite evidence")
        stages = plan["stages"]
        # Stage directories are exclusive; completed stages are reused only by
        # selecting downstream stages explicitly, never by silently skipping work.
        run_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "_" + uuid.uuid4().hex[:8]
        log_dir = self.output_dir / "pipeline_runs" / run_id
        log_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = log_dir / "run.json"
        record = {"schema_version": "calibration-pipeline-run.v1", "version": VERSION,
                  "run_id": run_id, "status": "RUNNING", "stages": {}, "plan": plan,
                  "config": self.config, "python": sys.version, "platform": platform.platform(),
                  "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                  "config_sha256": hashlib.sha256(json.dumps(self.config, sort_keys=True,
                      ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest(),
                  "calibration_claim": "Conditional estimates retain degeneracy/ID status; no automatic scientific acceptance."}

        def save():
            manifest_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

        save()
        try:
            for name in stages:
                started = time.time()
                record["stages"][name] = {"status": "RUNNING"}
                save()
                result = getattr(self, "_" + name)()
                record["stages"][name] = {"status": "EXECUTED", "seconds": time.time() - started,
                                           "result": result}
                save()
                if isinstance(result, dict) and result.get("status") in ("COMPLETED_WITH_FAILURES", "FAILED"):
                    raise RuntimeError("Stage %s reported failures; inspect its result and run manifest" % name)
            record["status"] = "EXECUTED"
        except Exception as exc:
            record["status"] = "FAILED"
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if "name" in locals():
                record["stages"][name]["status"] = "FAILED"
            save()
            raise
        save()
        return {"status": record["status"], "output_dir": str(self.output_dir),
                "run_manifest": str(manifest_path), "stages": record["stages"]}

    def _prepare(self):
        from .scene import prepare_scene
        spec = dict(self.config["scene"])
        for key in ("placement", "asset_root"):
            if key in spec and not isinstance(spec[key], dict):
                spec[key] = str(self.path(spec[key]))
        return prepare_scene(spec, self.output_dir / "scene", execute=True)

    def _capture(self):
        from .rendering import capture
        cfg = self.config["capture"]
        results = {}
        for kind, static in (("intrinsics", False), ("extrinsics", True)):
            if kind + "_spec" not in cfg:
                continue
            payload = {"capture_spec": self.document(cfg[kind + "_spec"]),
                       "base_ply": str(self.path(cfg["static_ply" if static else "base_ply"]))
                       if ("static_ply" if static else "base_ply") in cfg else
                       str(self.output_dir / "scene" / ("fused.ply" if static else "base.ply")),
                       "out_dir": str(self.output_dir / "captures" / kind), "static": static,
                       "board_ply": str(self.path(cfg["board_ply"])) if not static else None,
                       "repo": str(self.path(cfg["repo"])) if cfg.get("repo") else None,
                       "channels": cfg.get("channels"), "execute": True,
                       "near_plane": cfg.get("near_plane", .01), "seed": cfg.get("seed", 2026)}
            if cfg.get("render_python"):
                # A legacy GPU interpreter can reuse this exact module without
                # installing the modern CPU dependency set or a second worker file.
                env = dict(os.environ)
                package_parent = str(Path(__file__).resolve().parent.parent)
                env["PYTHONPATH"] = package_parent + os.pathsep + env.get("PYTHONPATH", "")
                code = "import json,sys; from calibration_pipeline.rendering import capture; capture(**json.load(sys.stdin))"
                subprocess.run([str(self.path(cfg["render_python"])), "-c", code],
                               input=json.dumps(payload), universal_newlines=True, env=env, check=True)
                out = Path(payload["out_dir"])
                candidates = list(out.glob("capture_manifest*.json"))
                if len(candidates) != 1:
                    raise RuntimeError("GPU worker must produce exactly one capture manifest")
                results[kind] = {"manifest_path": str(candidates[0]), "output": str(out)}
            else:
                results[kind] = capture(**payload)
        if not results:
            raise ValueError("capture requires intrinsics_spec and/or extrinsics_spec")
        return results

    def _compose(self):
        from .rendering import compose
        cfg = self.config["compose"]
        capture_cfg = self.config.get("capture", {})
        results = {}
        for kind in cfg.get("kinds", ["intrinsics", "extrinsics"]):
            if kind not in ("intrinsics", "extrinsics"):
                raise ValueError("compose.kinds must select intrinsics/extrinsics")
            info = cfg.get(kind, {})
            source = info.get("capture_spec", capture_cfg.get(kind + "_spec"))
            if source is None:
                raise ValueError("Missing capture specification for " + kind)
            if "manifest" in info:
                manifest = self.path(info["manifest"])
            else:
                candidates = list((self.output_dir / "captures" / kind).glob("capture_manifest*.json"))
                if len(candidates) != 1:
                    raise ValueError("Need exactly one capture manifest for " + kind)
                manifest = candidates[0]
            for model in cfg.get("models", ["pinhole", "fisheye"]):
                results[kind + "/" + model] = compose(self.document(source), manifest,
                    self.output_dir / "images" / kind / model, model=model, execute=True)
        return results

    def _intrinsics(self):
        from .calibration import calibrate_intrinsics
        cfg = self.config["intrinsics"]
        return calibrate_intrinsics(self.path(cfg["images"]) if cfg.get("images") else
            self.output_dir / "images" / "intrinsics", self.document(cfg["solver_spec"]),
            self.output_dir / "intrinsics", channels=cfg.get("channels"), execute=True)

    def _extrinsics(self):
        from .calibration import calibrate_extrinsics
        cfg = self.config["extrinsics"]
        return calibrate_extrinsics(self.path(cfg["images"]) if cfg.get("images") else
            self.output_dir / "images" / "extrinsics",
            self.path(cfg["intrinsics_dir"]) if cfg.get("intrinsics_dir") else self.output_dir / "intrinsics",
            self.document(cfg["corners_world"]), self.output_dir / "extrinsics",
            regions=cfg.get("regions"), models=cfg.get("models"), execute=True)

    def _bev(self):
        from .bev import stitch_bev, prepare_bev_spec
        cfg = self.config["bev"]
        results = {}
        if "spec" in cfg:
            source = cfg["spec"] if isinstance(cfg["spec"], dict) else self.path(cfg["spec"])
            return stitch_bev(source, self.output_dir / "bev" / "provided",
                              execute=True, base_dir=self.base_dir if isinstance(source, dict) else None)
        for model in cfg.get("models", ["fisheye"]):
            spec = prepare_bev_spec(
                self.path(cfg["intrinsics_dir"]) if cfg.get("intrinsics_dir") else self.output_dir / "intrinsics",
                self.path(cfg["extrinsics_dir"]) if cfg.get("extrinsics_dir") else self.output_dir / "extrinsics",
                self.path(cfg["images"]) if cfg.get("images") else self.output_dir / "images" / "extrinsics",
                model=model, channels=cfg["channels"], grid=cfg["grid"],
                exclude_rectangles_m=cfg.get("exclude_rectangles_m"),
                evaluation_rectangles_m=cfg.get("evaluation_rectangles_m"),
                seam_band_px=cfg.get("seam_band_px", 12))
            results[model] = stitch_bev(spec, self.output_dir / "bev" / model, execute=True)
        return results


def run_pipeline(config, *, stage="all", execute=False, base_dir=None):
    """Convenience function with exactly the same behavior as ``Pipeline.run``."""
    return Pipeline(config, base_dir=base_dir).run(stage, execute=execute)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compact 3DGS calibration and BEV pipeline")
    parser.add_argument("action", nargs="?", choices=["plan", "run"], default="plan")
    parser.add_argument("--version", action="version", version="calibration-pipeline " + VERSION)
    parser.add_argument("--config", required=True, help="calibration-pipeline.v1 JSON")
    parser.add_argument("--stage", default="all", help="all, one stage, or ordered comma-separated stages")
    parser.add_argument("--execute", action="store_true", help="Allow output writes and GPU work")
    args = parser.parse_args(argv)
    try:
        pipeline = Pipeline(args.config)
        result = pipeline.plan(args.stage) if args.action == "plan" else pipeline.run(args.stage, execute=args.execute)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, "%s: %s\n" % (type(exc).__name__, exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
