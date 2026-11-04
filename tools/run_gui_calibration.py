"""Isolated worker for editor calibration jobs (also useful for reproducing jobs)."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
for package in (ROOT/"avm_sim_ws/src").iterdir():
    if package.is_dir():sys.path.insert(0,str(package))
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--experiment",type=Path,required=True)
    args=parser.parse_args()
    if (args.experiment/"result.json").exists():parser.error("Job already has a result; never overwrite it")
    from scene_editor.calibration_jobs import run_job,initialize_job
    try:
        initialize_job(args.experiment)
        run_job(args.experiment)
    except Exception as exc:
        if not (args.experiment/"result.json").exists():
            from avm_bringup.experiments import finish_experiment
            finish_experiment(args.experiment,{"status":"failed","error":f"{type(exc).__name__}: {exc}",
                "calibration_accepted":False,"scene_parameters_applied":False})
        raise
