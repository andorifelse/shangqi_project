"""Run multi-tilt intrinsics and known-ID marker extrinsics as a GUI worker job."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for package in (ROOT/'avm_sim_ws/src').iterdir():
    if package.is_dir(): sys.path.insert(0, str(package))

from scene_manager.models import Scene
from scene_manager.persistence import load_scene
from scene_manager.store import SceneStore
from scene_editor.calibration_jobs import CalibrationJobs, MULTITILT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--scene', type=Path)
    source.add_argument('--from-experiment', type=Path, help='Reuse immutable scene/settings; capture new images in a new directory')
    source.add_argument('--export-experiment', type=Path, help='Export a passed result as a separate loadable scene, without recapturing')
    source.add_argument('--bev-experiment', type=Path, help='Stitch saved original images using a passed calibration, without recapturing')
    parser.add_argument('--small-board')
    parser.add_argument('--cameras', nargs='+')
    parser.add_argument('--views', type=int)
    parser.add_argument('--holdout', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--backend', choices=('gsplat', 'cpu'))
    parser.add_argument('--supersample', type=int)
    parser.add_argument('--marker-geometry', default='')
    parser.add_argument('--intrinsics-only', action='store_true')
    parser.add_argument('--experiment-id', default='')
    args = parser.parse_args()
    if args.bev_experiment:
        from scene_editor.calibrated_bev import generate
        directory = args.bev_experiment.resolve()
        result = json.loads((directory/'result.json').read_text(encoding='utf-8'))
        metrics = generate(directory, result)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0
    if args.export_experiment:
        from scene_editor.marker_extrinsics import export_calibrated_scene
        directory = args.export_experiment.resolve()
        result = json.loads((directory/'result.json').read_text(encoding='utf-8'))
        if not result.get('calibration_accepted'):
            parser.error('该实验的内外参尚未全部通过')
        request = json.loads((directory/'request.json').read_text(encoding='utf-8'))
        output = directory/'exports'
        output.mkdir(exist_ok=False)
        files = export_calibrated_scene(Scene.from_dict(request['scene_snapshot']),
            result['results'], result['extrinsics'], output)
        for path in files.values(): print(output/path, flush=True)
        return 0
    if args.from_experiment:
        original = json.loads((args.from_experiment/'request.json').read_text(encoding='utf-8'))
        scene = Scene.from_dict(original.pop('scene_snapshot'))
        request = original
    else:
        scene = load_scene(args.scene)
        request = dict(scene.calibration)
    request.update(algorithm=MULTITILT, run_extrinsics=not args.intrinsics_only,
        experiment_id=args.experiment_id, marker_geometry_file=args.marker_geometry)
    for key, value in [('intrinsic_board', args.small_board), ('cameras', args.cameras),
            ('views', args.views), ('holdout', args.holdout), ('seed', args.seed),
            ('backend', args.backend), ('supersample', args.supersample)]:
        if value is not None: request[key] = value
    jobs = CalibrationJobs()
    try:
        directory = jobs.start(SceneStore(scene), request)
        print(directory, flush=True)
        previous = None
        while True:
            state = jobs.poll()
            phase = state['progress'].get('phase')
            if phase != previous:
                print(phase, flush=True); previous = phase
            if not state['running']:
                result = state['result'] or {}
                print(result.get('error') or result.get('status', 'failed'), flush=True)
                print(directory/'result.json', flush=True)
                return 0 if result.get('status') == 'complete' else 1
            time.sleep(.3)
    except KeyboardInterrupt:
        jobs.cancel()
        return 130
    finally:
        jobs.close()


if __name__ == '__main__':
    sys.exit(main())
