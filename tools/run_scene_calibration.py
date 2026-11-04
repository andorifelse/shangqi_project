"""Run the exact current-scene GUI calibration job without hand-writing JSON."""
from pathlib import Path
import argparse,sys,time
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/'avm_sim_ws/src').iterdir():
    if p.is_dir():sys.path.insert(0,str(p))
from scene_manager.persistence import load_scene
from scene_manager.store import SceneStore
from scene_editor.calibration_jobs import CalibrationJobs,USER_PIPELINE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--small-board',required=True);p.add_argument('--large-board',required=True)
    p.add_argument('--camera-region',nargs='+',required=True,help='Explicit camera=region entries')
    p.add_argument('--backend',choices=('gsplat','cpu'),default='gsplat')
    p.add_argument('--views',type=int,default=12);p.add_argument('--holdout',type=int,default=2)
    p.add_argument('--seed',type=int,default=20261103)
    p.add_argument('--spread-x',type=float,default=.5);p.add_argument('--spread-y',type=float,default=.5)
    p.add_argument('--yaw-range-deg',type=float,default=30.)
    p.add_argument('--sampling',choices=('visible_plane','around_pose'),default='visible_plane')
    p.add_argument('--experiment-id',default='')
    a=p.parse_args();mapping={}
    for item in a.camera_region:
        parts=item.split('=')
        if len(parts)!=2 or not all(parts) or parts[0] in mapping:p.error('Use unique camera=region entries')
        mapping[parts[0]]=parts[1]
    jobs=CalibrationJobs()
    try:
        out=jobs.start(SceneStore(load_scene(a.scene)),{'algorithm':USER_PIPELINE,'input_mode':'scene',
            'cameras':list(mapping),'intrinsic_board':a.small_board,'extrinsic_board':a.large_board,
            'camera_regions':mapping,'backend':a.backend,'views':a.views,'holdout':a.holdout,
            'seed':a.seed,'sampling':a.sampling,'spread_x':a.spread_x,'spread_y':a.spread_y,
            'yaw_range_deg':a.yaw_range_deg,'experiment_id':a.experiment_id})
        print(out,flush=True);last=None
        while True:
            state=jobs.poll();phase=state['progress'].get('phase')
            if phase!=last:print(phase,flush=True);last=phase
            if not state['running']:
                result=state['result'];print(result.get('error') or result['status'],flush=True)
                return 0 if result['status']=='complete' else 1
            time.sleep(.2)
    except KeyboardInterrupt:
        jobs.cancel();return 130
    finally:jobs.close()


if __name__=='__main__':sys.exit(main())
