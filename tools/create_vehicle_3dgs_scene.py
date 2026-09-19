"""Create a runnable scene that composes the real environment and vehicle 3DGS assets."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for package in (ROOT / "avm_sim_ws" / "src").iterdir():
    if package.is_dir():
        sys.path.insert(0, str(package))

from avm_bringup.assets import generate_assets
from gs_renderer.gaussian import GaussianScene
from scene_manager.models import default_scene
from scene_manager.persistence import save_scene
from scene_manager.transforms import pose_matrix


def main() -> None:
    asset_dir = ROOT / "assets" / "vehicle_3dgs_calibration_board_2026-09-19"
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=Path,
                        default=ROOT / "assets" / "point_cloud_pgsr_depth.ply")
    parser.add_argument("--vehicle", type=Path,
                        default=asset_dir / "vehicle_mclaren_A2_256views_60k.ply")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "vehicle_3dgs_scene.yaml")
    parser.add_argument("--vehicle-length", type=float, default=4.2)
    args = parser.parse_args()
    environment, vehicle = args.environment.resolve(), args.vehicle.resolve()
    if not environment.is_file():
        raise FileNotFoundError(f"Environment 3DGS not found: {environment}")
    if not vehicle.is_file():
        raise FileNotFoundError(f"Vehicle 3DGS not found: {vehicle}")
    if not np.isfinite(args.vehicle_length) or args.vehicle_length <= 0:
        raise ValueError("Vehicle length must be positive")

    raw = GaussianScene.load(vehicle)
    bounds_min, bounds_max = raw.means.min(axis=0), raw.means.max(axis=0)
    extent = bounds_max - bounds_min
    scale = float(args.vehicle_length / extent[1])
    # Source +Y is vehicle-forward. Rz(-90 deg) maps it to project +X.
    # Translate after scaling so the lowest Gaussian rests on ground z=0.
    asset_pose = [0., 0., float(-bounds_min[2] * scale), 0., 0., float(-np.pi/2)]
    local = raw.transformed(scale, pose_matrix(asset_pose))

    fixtures = ROOT / "avm_sim_ws" / "src" / "avm_bringup" / "example_assets"
    generate_assets(fixtures)
    scene = default_scene(fixtures)
    scene.gaussian_file = str(environment)
    scene.vehicle.asset = str(vehicle)
    scene.vehicle.asset_pose = asset_pose
    scene.vehicle.asset_scale = scale
    scene.vehicle.dimensions = np.ptp(local.means, axis=0).astype(float).tolist()
    save_scene(scene, args.output)
    print(f"Wrote {args.output.resolve()}")
    print(f"Vehicle asset_scale={scale:.9f}, asset_pose={asset_pose}")
    print(f"Vehicle dimensions XYZ={scene.vehicle.dimensions}")
    print("Real scene exceeds one million Gaussians: use --no-render for UI inspection or backend:=gsplat for sensors.")


if __name__ == "__main__":
    main()
