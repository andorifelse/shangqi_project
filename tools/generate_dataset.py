"""Generate a seeded synthetic calibration dataset from one saved SceneState."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
for package in (ROOT/"avm_sim_ws"/"src").iterdir():
    if package.is_dir(): sys.path.insert(0,str(package))
from scene_manager.persistence import load_scene
from gs_renderer.calibration import make_calibration_dataset


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--scene",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--views",type=int,default=32)
    parser.add_argument("--seed",type=int,default=20261020)
    parser.add_argument("--supersample",type=int,default=2)
    parser.add_argument("--board-backend",choices=("cpu","gsplat"),default="cpu")
    parser.add_argument("--cameras",nargs="+")
    args=parser.parse_args()
    scene=load_scene(args.scene)
    metadata=make_calibration_dataset(scene,args.output,args.views,args.seed,args.supersample,args.cameras,args.board_backend)
    print(f"Generated {len(metadata['views'])} deterministic views in {args.output.resolve()}")
    print(f"Seed: {metadata['seed']}; cameras: {', '.join(metadata['cameras'])}")


if __name__=="__main__": main()
