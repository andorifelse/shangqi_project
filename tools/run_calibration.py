"""Generate a seeded synthetic calibration dataset and compare OpenCV estimates to Ground Truth."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for package in (ROOT / "avm_sim_ws" / "src").iterdir():
    if package.is_dir():
        sys.path.insert(0, str(package))

from scene_manager.ground_truth import GroundTruth
from scene_manager.persistence import load_scene
from gs_renderer.calibration import make_calibration_dataset, calibrate_camera_dataset, save_calibration_report
from avm_bringup.experiments import begin_experiment, append_log, finish_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Explicit experiment directory")
    parser.add_argument("--experiment-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--views", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--cameras", nargs="+", help="Default: all cameras in the scene")
    args = parser.parse_args()
    scene_path = args.scene.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else ROOT/"experiments"/args.experiment_id
    scene = load_scene(scene_path)
    config = {"pipeline":"synthetic_opencv_baseline", "source_scene":str(scene_path),
              "seed":args.seed, "views_per_camera":args.views, "supersample":args.supersample,
              "cameras":args.cameras or sorted(scene.cameras),
              "ROS_DOMAIN_ID":os.environ.get("ROS_DOMAIN_ID","0"),
              "CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES","")}
    output, truth = begin_experiment(scene, output, config, ROOT)
    try:
        _run(scene_path, output, truth, args)
    except Exception as exc:
        append_log(output, f"ERROR {type(exc).__name__}: {exc}")
        finish_experiment(output, {"status":"failed","error":f"{type(exc).__name__}: {exc}"})
        raise


def _run(scene_path, output, truth, args):
    metadata = make_calibration_dataset(truth.scene, output, args.views, args.seed,
                                        args.supersample, args.cameras)
    reports = {}
    for name in metadata["cameras"]:
        reports[name] = calibrate_camera_dataset(output, name, truth)
        append_log(output, f"{name}: views={reports[name]['views_detected']}; reprojection_rms_px={reports[name]['reprojection_rms_px']:.6f}")
    report = {"status":"complete", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_scene": str(scene_path), "seed": args.seed,
              "views_per_camera": args.views, "results": reports}
    save_calibration_report(report, output / "result.json")
    with (output / "metrics.csv").open("w", encoding="utf-8") as stream:
        stream.write("camera,views,reprojection_rms_px,fx_error_px,fy_error_px,cx_error_px,cy_error_px,distortion_rmse,translation_rmse_m,rotation_rmse_deg\n")
        for name, result in reports.items():
            intr = result["intrinsic_error"]
            de = result["distortion_error_est_minus_gt"]
            distortion_rmse = (sum(x*x for x in de)/len(de))**.5 if de else 0.
            ext = result["extrinsics_error"]
            stream.write(f"{name},{result['views_detected']},{result['reprojection_rms_px']:.9g},"
                         f"{intr['fx_px']:.9g},{intr['fy_px']:.9g},{intr['cx_px']:.9g},{intr['cy_px']:.9g},"
                         f"{distortion_rmse:.9g},{ext['translation_rmse_m']:.9g},{ext['rotation_rmse_deg']:.9g}\n")
    finish_experiment(output,report)
    print(json.dumps({n:{"views":r["views_detected"],"reprojection_rms_px":r["reprojection_rms_px"],
                         "translation_rmse_m":r["extrinsics_error"]["translation_rmse_m"],
                         "rotation_rmse_deg":r["extrinsics_error"]["rotation_rmse_deg"]} for n,r in reports.items()},indent=2))
    print(f"Experiment saved to {output}")


if __name__ == "__main__":
    main()
