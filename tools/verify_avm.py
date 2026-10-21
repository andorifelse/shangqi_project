"""Export camera masks/overlap/weights and numerically verify BEV ground geometry."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
for package in (ROOT/"avm_sim_ws"/"src").iterdir():
    if package.is_dir():
        sys.path.insert(0,str(package))
from scene_manager.persistence import load_scene
from scene_manager.models import default_scene
from avm_stitcher.validation import export_avm_validation


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--scene",type=Path)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs"/"avm_validation")
    args=parser.parse_args()
    scene=load_scene(args.scene) if args.scene else default_scene(ROOT/"avm_sim_ws"/"src"/"avm_bringup"/"example_assets")
    metrics=export_avm_validation(scene,args.output)
    print(json.dumps(metrics,indent=2))
    if metrics["ground_bev_roundtrip_max_px"]>1e-8 or not metrics["coverage"] or metrics["ground_color_p99_error"]>4:
        raise RuntimeError("AVM geometry validation failed")


if __name__=="__main__":
    main()
