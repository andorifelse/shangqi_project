"""Add an independent Gaussian board to a scene without modifying the input YAML."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
for package in (ROOT/'avm_sim_ws'/'src').iterdir():
    if package.is_dir(): sys.path.insert(0,str(package))
from scene_manager.persistence import load_scene,save_scene
from scene_manager.store import SceneStore
from gs_renderer.gaussian import GaussianScene


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--ply',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--name',default='board_gaussian')
    parser.add_argument('--pose',type=float,nargs=6,default=[0.,0.,0.,0.,0.,0.],metavar=('X','Y','Z','ROLL','PITCH','YAW'))
    parser.add_argument('--asset-pose',type=float,nargs=6,default=[0.]*6)
    parser.add_argument('--scale',type=float,default=1.)
    parser.add_argument('--rows',type=int,default=6)
    parser.add_argument('--columns',type=int,default=8)
    parser.add_argument('--square-size',type=float,default=.15)
    args=parser.parse_args()
    if args.output.exists():
        parser.error(f'Output already exists, choose a new path: {args.output}')
    asset=args.ply.expanduser().resolve()
    gs=GaussianScene.load(asset)
    store=SceneStore(load_scene(args.scene))
    store.command({'action':'add_board','name':args.name,'properties':{
        'asset':str(asset),'asset_pose':args.asset_pose,'asset_scale':args.scale,'pose':args.pose,
        'rows':args.rows,'columns':args.columns,'square_size':args.square_size}})
    save_scene(store.snapshot()[1],args.output)
    print(f'Added {args.name}: {len(gs.means)} Gaussians -> {args.output.resolve()}')
    print('No automatic metric calibration: align PLY axes/origin/scale to the board frame before evaluating corner Ground Truth.')


if __name__=='__main__': main()
