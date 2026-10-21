"""Re-estimate parameters using saved PNG detections; does not regenerate images."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
for package in (ROOT/"avm_sim_ws"/"src").iterdir():
    if package.is_dir(): sys.path.insert(0,str(package))
from scene_manager.ground_truth import GroundTruth
from gs_renderer.calibration import calibrate_camera_dataset, save_calibration_report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cameras",nargs="+")
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Result already exists: {args.output}")
    truth=GroundTruth.load(args.dataset/"ground_truth.yaml")
    metadata=json.loads((args.dataset/"metadata.json").read_text(encoding="utf-8"))
    names=args.cameras or sorted(metadata["cameras"])
    results={name:calibrate_camera_dataset(args.dataset,name,truth) for name in names}
    save_calibration_report({"status":"complete","replayed_from":str(args.dataset.resolve()),"results":results},args.output)
    print(f"Recalibrated {', '.join(names)} from existing PNGs -> {args.output}")


if __name__=="__main__": main()
