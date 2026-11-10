"""Fixed-camera native Graphdeco capture and CPU lens composition (Python 3.7+).

Graphdeco is an explicit external backend. No training repository, CUDA binary,
or experiment asset is bundled here. Importing this module never imports torch.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np

from .scene import rotation_quaternion


FACE_AXES = {"front": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
             "right": ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
             "left": ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
             "up": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
             "down": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
             "back": ((-1, 0, 0), (0, 1, 0), (0, 0, -1))}
PARAMETERS = ("get_xyz", "get_features", "get_scaling", "get_rotation", "get_opacity")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Not JSON serializable: " + type(value).__name__)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _document(value):
    if isinstance(value, dict):
        content = json.dumps(value, sort_keys=True, allow_nan=False, default=_json_default).encode("utf-8")
        data = json.loads(content.decode("utf-8"))
        return data, Path.cwd(), hashlib.sha256(content).hexdigest(), None
    path = Path(value).expanduser().resolve()
    return json.loads(path.read_text(encoding="utf-8")), path.parent, _sha(path), path


def _content_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False,
                                      default=_json_default).encode("utf-8")).hexdigest()


def _write(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default)
        stream.write("\n")


def _name(value):
    if (not isinstance(value, str) or not value or value in (".", "..")
            or "/" in value or "\\" in value):
        raise ValueError("Unsafe camera/view name: %r" % value)
    return value


def _new_output(value):
    path = Path(value).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError("Refusing to overwrite rendering output: " + str(path))
    return path.resolve()


def _input_path(value, base):
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else base/path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def camera_frames(spec, channels=None):
    names = list(spec["cameras"]) if channels is None else (
        [item.strip() for item in channels.split(",")] if isinstance(channels, str) else list(channels))
    if not names or len(set(names)) != len(names):
        raise ValueError("Camera channels must be nonempty and unique")
    frames = {}
    for name in names:
        _name(name)
        if name not in spec["cameras"]:
            raise ValueError("Unknown camera channel: " + name)
        value = spec["cameras"][name]["frame"]
        rotation = np.asarray(value["R_w2c"], dtype=np.float64)
        rotation_quaternion(rotation)
        if "camera_center_world" in value:
            center = np.asarray(value["camera_center_world"], dtype=np.float64)
        else:
            translation = np.asarray(value["tvec"], dtype=np.float64)
            if translation.shape != (3,) or not np.isfinite(translation).all():
                raise ValueError("Invalid camera translation: " + name)
            center = -rotation.T@translation
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("Invalid camera optical center: " + name)
        translation = -rotation@center
        if "tvec" in value and not np.allclose(value["tvec"], translation, atol=1e-7, rtol=0):
            raise ValueError("Camera center and translation disagree: " + name)
        frames[name] = {"R_w2c": rotation, "tvec": translation, "camera_center_world": center}
    return frames


def face_frames(rotation, center):
    result = {}
    for face, axes in FACE_AXES.items():
        r = np.asarray(axes, dtype=np.float64)@rotation
        result[face] = {"R_w2c": r, "tvec": -r@center}
    return result


def validate_transform(value, plane=None):
    matrix = np.asarray(value, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or np.abs(matrix).max() > np.finfo(np.float32).max):
        raise ValueError("Board pose must be a finite float32-representable SE3")
    rotation_quaternion(matrix[:3, :3])
    if plane is not None:
        normal = np.asarray(plane["normal"], dtype=np.float64)
        offset = float(plane["offset"])
        if (normal.shape != (3,) or not np.isfinite(normal).all() or not math.isfinite(offset)
                or not np.isclose(np.linalg.norm(normal), 1, atol=1e-7, rtol=0)
                or not np.allclose(matrix[:3, 2], normal, atol=1e-6, rtol=0)
                or not np.isclose(normal@matrix[:3, 3], offset, atol=1e-6, rtol=0)):
            raise ValueError("Small board must remain flat on the declared plane")
    return matrix


def _plans(spec, frames, static):
    background = _name(spec.get("static_background_record", {}).get(
        "frame_id", "big_board_fixed" if static else "intrinsics_background"))
    plans = {}
    for channel, frame in frames.items():
        views = [{"name": background, "board_id": "large_board" if static else "none",
                  "board_to_world": None, "moving_board_visible": False}]
        if not static:
            configured = spec["cameras"][channel].get("views", [])
            if not configured:
                raise ValueError("Dynamic capture needs small-board poses: " + channel)
            names = {background}
            for view in configured:
                name = _name(view["name"])
                if name in names:
                    raise ValueError("Duplicate view name: " + channel+"/"+name)
                names.add(name)
                if "camera_frame" in view:
                    candidate = camera_frames({"cameras": {channel: {"frame": view["camera_frame"]}}})[channel]
                    if any(not np.allclose(candidate[key], frame[key], atol=1e-7, rtol=0) for key in frame):
                        raise ValueError("Camera must remain fixed between views: " + name)
                views.append({"name": name, "board_id": "small_board", "moving_board_visible": True,
                              "board_to_world": validate_transform(view["board_to_world"], spec.get("rig_ground_plane")),
                              "split": view.get("split")})
        plans[channel] = views
    return plans


def _cube_geometry(spec):
    size, padding = spec["cube_size"], spec.get("cube_face_padding_px", 0)
    if isinstance(size, bool) or not isinstance(size, int) or size < 2:
        raise ValueError("cube_size must be an integer >=2")
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise ValueError("cube_face_padding_px must be a nonnegative integer")
    return size, padding, size+2*padding, size/2.


def _render_state(model, torch):
    values = {name: getattr(model, name).detach() for name in PARAMETERS}
    count = int(values["get_xyz"].shape[0])
    shapes = {"get_xyz": (count, 3), "get_features": (count, 16, 3), "get_scaling": (count, 3),
              "get_rotation": (count, 4), "get_opacity": (count, 1)}
    for name, value in values.items():
        if count < 1 or tuple(value.shape) != shapes[name] or not bool(torch.isfinite(value).all().item()):
            raise ValueError("Invalid native Gaussian parameter: " + name)
    if bool((torch.linalg.norm(values["get_rotation"], dim=1) < .99).any().item()):
        raise ValueError("Gaussian quaternions must be normalized")
    return SimpleNamespace(active_sh_degree=3, max_sh_degree=3, **values)


def _move_board(state, start, xyz, quaternions, matrix, torch):
    rotation = torch.as_tensor(matrix[:3, :3], dtype=xyz.dtype, device=xyz.device)
    translation = torch.as_tensor(matrix[:3, 3], dtype=xyz.dtype, device=xyz.device)
    q = torch.as_tensor(rotation_quaternion(matrix[:3, :3]), dtype=quaternions.dtype, device=quaternions.device)
    w, x, y, z = q.unbind()
    a, b, c, d = quaternions.unbind(dim=1)
    moved = torch.stack((w*a-x*b-y*c-z*d, w*b+x*a+y*d-z*c,
                         w*c-x*d+y*a+z*b, w*d+x*c-y*b+z*a), dim=1)
    moved /= torch.linalg.norm(moved, dim=1, keepdim=True)
    state.get_xyz[start:].copy_(xyz@rotation.T+translation)
    state.get_rotation[start:].copy_(moved)


def _camera(definition, focal, side, name, graphics, torch, near):
    r, t = definition["R_w2c"], definition["tvec"]
    view = torch.tensor(graphics.getWorld2View2(r.T.astype(np.float32), np.asarray(t, np.float32),
                                               np.zeros(3), 1.)).transpose(0, 1).cuda()
    fov = graphics.focal2fov(focal, side)
    projection = graphics.getProjectionMatrix(near, 100., fov, fov).transpose(0, 1).cuda()
    return SimpleNamespace(image_width=side, image_height=side, image_name=name, FoVx=fov, FoVy=fov,
                           znear=near, zfar=100., world_view_transform=view,
                           full_proj_transform=view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0),
                           camera_center=view.inverse()[3, :3])


def capture(capture_spec, base_ply, out_dir, *, repo=None, board_ply=None, channels=None,
            static=False, execute=False, near_plane=.01, seed=2026):
    """Render six padded faces per fixed-camera view; native camera-Z sorting.

    Dynamic mode adds one board-free background and the configured SE3 poses of
    a local DC-only board. Static mode renders the already fused large board.
    The Graphdeco repository is external (--repo), or its modules must be installed.
    """
    if type(execute) is not bool:
        raise ValueError("execute must be a bool")
    spec, spec_base, spec_hash, spec_path = _document(capture_spec)
    out = _new_output(out_dir)
    base_path = _input_path(base_ply, spec_base)
    board_path = _input_path(board_ply, spec_base) if board_ply is not None else None
    if bool(board_path) == bool(static):
        raise ValueError("Dynamic mode needs board_ply; static mode must not append a board")
    if not math.isfinite(float(near_plane)) or not 0 < float(near_plane) < 100:
        raise ValueError("near_plane must be finite in (0,100)")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0,2**32)")
    frames = camera_frames(spec, channels)
    plans = _plans(spec, frames, static)
    size, padding, side, focal = _cube_geometry(spec)
    captures = {}
    for channel, views in plans.items():
        captures[channel] = {}
        for view in views:
            record = dict(view)
            record.update(frame_id=view["name"], board_visible=bool(static or view["moving_board_visible"]),
                          camera_frame={key: value.tolist() for key, value in frames[channel].items()},
                          faces={face: str(out/"cube"/channel/view["name"]/(channel+"__"+face+".png")) for face in FACE_AXES})
            if record["board_to_world"] is not None:
                record["board_to_world"] = record["board_to_world"].tolist()
            captures[channel][view["name"]] = record
    manifest = {"schema_version": "calibration-capture.v1", "dry_run": not execute, "complete": False,
                "manifest_path": str(out/"capture_manifest.json"), "output": str(out), "captures": captures,
                "capture_spec": str(spec_path) if spec_path else "inline", "capture_spec_sha256": spec_hash,
                "capture_spec_content_sha256": _content_hash(spec),
                "base_ply": str(base_path), "board_ply": str(board_path) if board_path else None,
                "cube_core_size_px": [size, size], "cube_face_size_px": [side, side],
                "cube_face_padding_px": padding, "cube_face_focal_px": focal,
                "cube_face_intrinsics": {"fx": focal, "fy": focal, "cx": (side-1)/2., "cy": (side-1)/2.},
                "camera_motion": False, "shared_board_frames": True, "static_scene_only": bool(static),
                "expected_cube_faces": sum(len(v) for v in plans.values())*6,
                "gaussian_sort": "per_face_camera_z", "near_plane_m": float(near_plane), "far_plane_m": 100.,
                "renderer": "Native Graphdeco diff-gaussian-rasterization",
                "renderer_limitation": "No KIRI radial sorting; face seams require image inspection",
                "truth_boundary": "Camera/board poses are capture provenance, not solver observations"}
    if not execute:
        return manifest
    import torch
    import torchvision
    if repo is not None:
        repository = Path(repo).expanduser().resolve()
        if not (repository/"gaussian_renderer/__init__.py").is_file():
            raise ValueError("Invalid external Graphdeco repository")
        if str(repository) not in sys.path:
            sys.path.insert(0, str(repository))
    else:
        repository = None
    try:
        graphics = importlib.import_module("utils.graphics_utils")
        model_module = importlib.import_module("scene.gaussian_model")
        renderer = importlib.import_module("gaussian_renderer")
        rasterizer = importlib.import_module("diff_gaussian_rasterization")
    except ImportError as error:
        raise RuntimeError("Install the compatible native Graphdeco backend or supply repo: %s" % error)
    if repository is not None and repository not in Path(renderer.__file__).resolve().parents:
        raise RuntimeError("A different Graphdeco repository is already imported in this process")
    if not torch.cuda.is_available():
        raise RuntimeError("An executed capture requires CUDA")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    input_hashes = {str(path): _sha(path) for path in [base_path]+([board_path] if board_path else [])}
    model = model_module.GaussianModel(3)
    model.load_ply(str(base_path))
    base = _render_state(model, torch)
    base_count, board_count = len(base.get_xyz), 0
    state = base
    local_xyz = local_rotation = None
    if board_path is not None:
        board_model = model_module.GaussianModel(3)
        board_model.load_ply(str(board_path))
        board = _render_state(board_model, torch)
        if bool((board.get_features[:, 1:, :] != 0).any().item()):
            raise ValueError("Moving board requires exactly zero higher-order SH; its rotation is not ignored")
        board_count = len(board.get_xyz)
        local_xyz, local_rotation = board.get_xyz.clone(), board.get_rotation.clone()
        state = SimpleNamespace(active_sh_degree=3, max_sh_degree=3,
                                **{name: torch.cat((getattr(base, name), getattr(board, name)), dim=0) for name in PARAMETERS})
        base = SimpleNamespace(active_sh_degree=3, max_sh_degree=3,
                               **{name: getattr(state, name)[:base_count] for name in PARAMETERS})
        del board, board_model
    del model
    binary = getattr(getattr(rasterizer, "_C", None), "__file__", None)
    manifest.update(base_ply_sha256=input_hashes[str(base_path)],
                    board_ply_sha256=input_hashes.get(str(board_path)), base_gaussian_count=base_count,
                    board_gaussian_count=board_count, maximum_gaussian_count=base_count+board_count, active_sh_degree=3,
                    source_snapshot_sha256=_sha(__file__),
                    color_pipeline="Native SH3; black background; clamp 0..1; torchvision RGB PNG; no extra tone mapping",
                    runtime={"python": sys.version, "torch": str(torch.__version__), "cuda": torch.version.cuda,
                             "torchvision": str(torchvision.__version__), "numpy": np.__version__,
                             "gpu": torch.cuda.get_device_name(0), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                             "repo": str(repository) if repository else None, "renderer_source": str(renderer.__file__),
                             "renderer_source_sha256": _sha(renderer.__file__),
                             "gaussian_model_source_sha256": _sha(model_module.__file__),
                             "graphics_utils_source_sha256": _sha(graphics.__file__),
                             "rasterizer_binary": binary, "rasterizer_binary_sha256": _sha(binary) if binary else None,
                             "seed": seed, "determinism_scope": "Fixed seed; custom CUDA determinism not assumed"})
    pipe = SimpleNamespace(debug=False, antialiasing=False, compute_cov3D_python=False, convert_SHs_python=False)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    out.mkdir(parents=True, exist_ok=False)
    completed = 0
    for channel, frame in frames.items():
        definitions = face_frames(frame["R_w2c"], frame["camera_center_world"])
        cameras = {face: _camera(definition, focal, side, channel+"__"+face, graphics, torch, near_plane)
                   for face, definition in definitions.items()}
        for view in plans[channel]:
            record = captures[channel][view["name"]]
            directory = out/"cube"/channel/view["name"]
            directory.mkdir(parents=True, exist_ok=False)
            submitted = state if view["moving_board_visible"] else base
            with torch.no_grad():
                if view["moving_board_visible"]:
                    _move_board(state, base_count, local_xyz, local_rotation, view["board_to_world"], torch)
                face_records = {}
                for face, camera in cameras.items():
                    center = camera.camera_center.detach().cpu().numpy()
                    if not np.allclose(center, frame["camera_center_world"], atol=1e-5, rtol=0):
                        raise RuntimeError("Rendered face changed the camera optical center")
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    image = renderer.render(camera, submitted, pipe, background)["render"].clamp(0, 1)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-start
                    image_path = Path(record["faces"][face])
                    torchvision.utils.save_image(image, str(image_path))
                    face_records[face] = {"R_w2c": definitions[face]["R_w2c"].astype(np.float32).tolist(),
                        "tvec": definitions[face]["tvec"].astype(np.float32).tolist(),
                        "camera_center_rendered_world": center.tolist(), "image_sha256": _sha(image_path),
                        "resolution_px": [side, side], "render_seconds": elapsed,
                        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated())}
                    completed += 1
            record.update(complete=True, rendered_faces=list(FACE_AXES), face_records=face_records,
                          gaussian_count=len(submitted.get_xyz), board_in_base_ply=bool(static),
                          board_to_world_applied_float32=(view["board_to_world"].astype(np.float32).tolist()
                                                         if view["board_to_world"] is not None else None))
            view_manifest = {key: value for key, value in manifest.items() if key not in ("captures", "manifest_path")}
            view_manifest.update(complete=True, captures={channel: {view["name"]: record}},
                                 completed_cube_faces=6, expected_cube_faces=6)
            _write(directory/"view_manifest.json", view_manifest)
            print(json.dumps({"camera": channel, "view": view["name"], "cube_faces": 6}), flush=True)
        channel_manifest = {key: value for key, value in manifest.items() if key not in ("captures", "manifest_path")}
        channel_manifest.update(complete=True, captures={channel: captures[channel]},
                                completed_cube_faces=len(plans[channel])*6, expected_cube_faces=len(plans[channel])*6)
        _write(out/"cube"/channel/"channel_manifest.json", channel_manifest)
    for path, digest in input_hashes.items():
        if _sha(path) != digest:
            raise RuntimeError("Input PLY changed during capture: " + path)
    if spec_path is not None and _sha(spec_path) != spec_hash:
        raise RuntimeError("Capture spec changed during capture")
    manifest.update(complete=True, completed_cube_faces=completed, manual_acceptance="pending")
    _write(out/"capture_manifest.json", manifest)
    return manifest


def lens_rays(lens):
    """Return camera-space unit rays, validity and selected cube face indices."""
    import cv2
    width, height = lens["width"], lens["height"]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in (width, height)):
        raise ValueError("Lens image dimensions must be integer >=2")
    model = lens.get("model", "fisheye")
    if model == "pinhole":
        matrix = np.asarray(lens["K"], dtype=np.float64)
        distortion = np.asarray(lens.get("distortion", [0.]*5), dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError("Invalid pinhole camera matrix")
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        pixels = np.stack((xx, yy), axis=-1).astype(np.float64)
        # OpenCV's undistortion assumes zero skew; remove skew in pixel space first.
        pixels[..., 0] -= matrix[0, 1]/matrix[1, 1]*(pixels[..., 1]-matrix[1, 2])
        k0 = matrix.copy()
        k0[0, 1] = 0
        arguments = (pixels.reshape(-1, 1, 2), k0, distortion)
        if hasattr(cv2, "undistortPointsIter"):
            xy = cv2.undistortPointsIter(*arguments, R=None, P=None, criteria=(3, 50, 1e-12))
        else:
            xy = cv2.undistortPoints(*arguments, criteria=(3, 50, 1e-12))
        rays = np.concatenate((xy.reshape(height, width, 2), np.ones((height, width, 1))), axis=-1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        valid = np.isfinite(rays).all(axis=-1)
    elif model == "fisheye":
        distortion = np.asarray(lens["distortion"], dtype=np.float64)
        limit = np.deg2rad(float(lens["field_of_view_deg"])/2)
        if distortion.shape != (4,) or not np.isfinite(distortion).all() or not 0 < limit < np.pi:
            raise ValueError("Need four finite fisheye coefficients and FOV in (0,360)")
        polynomial = 1+sum(distortion[i]*limit**(2*i+2) for i in range(4))
        focal = (min(width, height)/2-6)/(limit*polynomial)
        if not math.isfinite(focal) or focal <= 0:
            raise ValueError("Fisheye FOV/distortion gives an invalid focal length")
        xx, yy = np.meshgrid(np.arange(width)+.5, np.arange(height)+.5)
        x, y = (xx-width/2)/focal, (yy-height/2)/focal
        td = np.hypot(x, y)
        theta = td.copy()
        for _ in range(12):
            poly = 1+sum(distortion[i]*theta**(2*i+2) for i in range(4))
            derivative = 1+sum((2*i+3)*distortion[i]*theta**(2*i+2) for i in range(4))
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                theta -= (theta*poly-td)/derivative
        scale = np.divide(np.sin(theta), td, out=np.ones_like(td), where=td > 1e-10)
        rays = np.stack((x*scale, y*scale, np.cos(theta)), axis=-1)
        valid = (theta >= 0)&(theta <= limit)&np.isfinite(rays).all(axis=-1)
    else:
        raise ValueError("Unknown lens model: " + str(model))
    axes = np.asarray(list(FACE_AXES.values()))
    assignment = np.argmax(np.einsum("hwj,fj->hwf", rays, axes[:, 2]), axis=-1)
    return rays, valid, assignment


def compose(capture_spec, manifest, out_dir, *, model="fisheye", execute=False):
    """Compose padded cube faces into original-resolution, image-only lens PNGs.

    Relative face paths use the manifest file's directory. For a manifest mapping,
    use its explicit output directory, then manifest_path's directory, else cwd.
    """
    if type(execute) is not bool:
        raise ValueError("execute must be a bool")
    if model not in ("pinhole", "fisheye"):
        raise ValueError("model must be pinhole or fisheye")
    spec, _, spec_hash, _ = _document(capture_spec)
    captured, manifest_base, manifest_hash, manifest_path = _document(manifest)
    out = _new_output(out_dir)
    size, padding, side, focal = _cube_geometry(spec)
    if captured.get("cube_face_padding_px", 0) != padding:
        raise ValueError("Cubemap padding differs from the capture specification")
    if captured.get("cube_face_size_px", [side, side]) != [side, side]:
        raise ValueError("Cubemap face dimensions differ from the capture specification")
    expected_content = captured.get("capture_spec_content_sha256")
    if expected_content is not None and expected_content != _content_hash(spec):
        raise ValueError("Capture manifest belongs to a different capture specification")
    if (expected_content is None and captured.get("capture_spec_sha256")
            and captured["capture_spec_sha256"] != spec_hash):
        raise ValueError("Capture manifest belongs to a different capture specification")
    lens = dict(spec[model], model=model)
    width, height = lens["width"], lens["height"]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in (width, height)):
        raise ValueError("Lens image dimensions must be integer >=2")
    if not isinstance(captured.get("captures"), dict) or not captured["captures"]:
        raise ValueError("A capture manifest must contain nonempty camera views")
    image_records, associations = [], []
    for channel, views in captured["captures"].items():
        _name(channel)
        if not isinstance(views, dict) or not views:
            raise ValueError("Each captured camera must contain nonempty views")
        for name, record in views.items():
            _name(name)
            if set(record["faces"]) != set(FACE_AXES):
                raise ValueError("Each captured view needs all six cube faces")
            image_records.append({"camera_id": channel, "frame_id": name,
                "image": str(Path(channel)/(name+".png")), "width_px": width, "height_px": height,
                "camera_model": model})
            associations.append({"camera_id": channel, "frame_id": record.get("frame_id", name),
                                 "board_id": record.get("board_id", "unknown"),
                                 "image": str(Path(channel)/(name+".png"))})
    image_manifest = {"schema_version": "calibration-images.v1", "dry_run": not execute,
                      "camera_model": model, "image_size_px": [width, height], "cube_core_size_px": [size, size],
                      "cube_face_size_px": [side, side], "cube_face_padding_px": padding,
                      "capture_spec_sha256": spec_hash, "capture_manifest_sha256": manifest_hash,
                      "coordinate_convention": "zero-based pixel indices; original PNGs, no resize",
                      "image_count": len(image_records), "images": image_records}
    result = {"dry_run": not execute, "output": str(out), "out_dir": str(out),
              "manifest_path": str(out/"image_manifest.json"), "image_manifest": image_manifest}
    if not execute:
        return result
    import cv2
    if captured.get("complete") is False:
        raise ValueError("Cannot compose an incomplete capture")
    rays, valid, assignment = lens_rays(lens)
    if manifest_path is None and captured.get("output"):
        manifest_base = Path(captured["output"])
    elif manifest_path is None and captured.get("manifest_path"):
        manifest_base = Path(captured["manifest_path"]).expanduser().resolve().parent
    out.mkdir(parents=True, exist_ok=False)
    for image_record in image_records:
        channel, name = image_record["camera_id"], image_record["frame_id"]
        if model == "pinhole" and spec.get("per_camera_pinhole"):
            channel_lens = dict(spec["per_camera_pinhole"][channel], model=model)
            if [channel_lens["width"], channel_lens["height"]] != [width, height]:
                raise ValueError("Per-camera lens resolutions must match")
            rays, valid, assignment = lens_rays(channel_lens)
        record = captured["captures"][channel][name]
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for i, (face, axes) in enumerate(FACE_AXES.items()):
            source_path = _input_path(record["faces"][face], manifest_base)
            source = cv2.imread(str(source_path))
            if source is None or source.shape != (side, side, 3):
                raise ValueError("Missing, invalid or wrong-size cube face: " + str(source_path))
            meta = record.get("face_records", {}).get(face)
            if meta and meta.get("image_sha256") and _sha(source_path) != meta["image_sha256"]:
                raise ValueError("Cube face hash changed: " + str(source_path))
            mask = valid&(assignment == i)
            if not mask.any():
                continue
            xyz = np.einsum("hwj,ij->hwi", rays, np.asarray(axes))
            x = (focal*xyz[..., 0]/np.maximum(xyz[..., 2], 1e-9)+(side-1)/2.).astype(np.float32)
            y = (focal*xyz[..., 1]/np.maximum(xyz[..., 2], 1e-9)+(side-1)/2.).astype(np.float32)
            sampled = cv2.remap(source, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            image[mask] = sampled[mask]
        destination = out/image_record["image"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), image):
            raise RuntimeError("Failed to write PNG: " + str(destination))
        image_record["sha256"] = _sha(destination)
    image_manifest["complete"] = True
    _write(out/"image_manifest.json", image_manifest)
    _write(out/"rig_image_index.json", {"cameras": {ch: {"image_size": [width, height], "model": model}
        for ch in captured["captures"]}, "images": associations,
        "corner_identity_status": "Acquisition identities only; physical directed corner IDs require image association"})
    result.update(complete=True)
    return result
