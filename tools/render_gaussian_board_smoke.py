"""Render a saved Gaussian board scene, then verify motion changes camera images."""
from pathlib import Path
import argparse
from copy import deepcopy
import json
import sys
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1]
for package in (ROOT/'avm_sim_ws'/'src').iterdir():
    if package.is_dir(): sys.path.insert(0,str(package))
from scene_manager.persistence import load_scene,save_scene
from scene_manager.ground_truth import GroundTruth
from gs_renderer.pipeline import SensorPipeline
from avm_stitcher.projection import AvmStitcher


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--board',default='board_gaussian')
    parser.add_argument('--backend',choices=('cpu','gsplat'),default='gsplat')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Output exists; choose a new directory')
    s=load_scene(args.scene)
    if not s.boards[args.board].is_gaussian: parser.error('Selected board is not Gaussian')
    args.output.mkdir(parents=True)
    pipeline=SensorPipeline(args.backend)
    before=pipeline.render(s)
    renderer=pipeline.renderer
    stable=pipeline.render(s)
    assert pipeline.renderer is renderer,'Stationary frame rebuilt geometry'
    for name in before: np.testing.assert_array_equal(before[name].rgb,stable[name].rgb)
    moved=deepcopy(s)
    moved.boards[args.board].pose[0]+=.5
    moved.boards[args.board].pose[5]+=.2
    after=pipeline.render(moved)
    differences={}
    for name in before:
        differences[name]=int(np.any(before[name].rgb!=after[name].rgb,axis=-1).sum())
        for label,results in (('before',before),('after',after)):
            target=args.output/f'{name}_{label}.png'
            assert cv2.imwrite(str(target),cv2.cvtColor(results[name].rgb,cv2.COLOR_RGB2BGR))
    assert any(differences.values()),'Moving Gaussian board did not change any camera image'
    rgb={n:r.rgb for n,r in before.items()}
    bev=AvmStitcher().stitch(s,rgb).rgb
    assert cv2.imwrite(str(args.output/'avm.png'),cv2.cvtColor(bev,cv2.COLOR_RGB2BGR))
    GroundTruth.capture(s).save(args.output/'ground_truth.yaml')
    save_scene(moved,args.output/'moved_scene.yaml')
    metrics={'backend':args.backend,'board':args.board,'stationary_geometry_reused':True,
        'changed_pixels':differences,'camera_count':len(before)}
    (args.output/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__': main()
