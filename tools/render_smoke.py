"""Render a complete standalone frame and write reviewable outputs."""
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[1]
for package in (ROOT/"avm_sim_ws"/"src").iterdir():
    if package.is_dir():
        sys.path.insert(0,str(package))
import cv2
from scene_manager.models import default_scene
from scene_manager.persistence import load_scene
from avm_bringup.assets import generate_assets
from gs_renderer.pipeline import SensorPipeline
from avm_stitcher.projection import AvmStitcher


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--backend",choices=("cpu","gsplat"),default="cpu")
    parser.add_argument("--scene",type=Path,help="Existing scene YAML; coordinates and camera settings are preserved")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs"/"smoke")
    parser.add_argument("--frames",type=int,default=1)
    args=parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    fixtures=ROOT/"avm_sim_ws"/"src"/"avm_bringup"/"example_assets"
    generate_assets(fixtures)
    scene=load_scene(args.scene) if args.scene else default_scene(fixtures)
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    pipeline=SensorPipeline(args.backend)
    stitcher=AvmStitcher()
    batches=[]
    if args.backend=="gsplat":
        import torch
        torch.cuda.reset_peak_memory_stats()
    for _ in range(args.frames):
        batch_start=time.perf_counter()
        results=pipeline.render(scene)
        images={n:r.rgb for n,r in results.items()}
        avm=stitcher.stitch(scene,images)
        batches.append(time.perf_counter()-batch_start)
    elapsed=time.perf_counter()-start
    for name,image in {**images,"avm":avm.rgb}.items():
        cv2.imwrite(str(out/f"{name}.png"),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    stats={"backend":args.backend,"seconds":elapsed,"batch_seconds":batches,
           "avm_coverage":float(avm.coverage.mean()),"cameras":list(images),
           "gaussians":len(pipeline.renderer.gs.means) if args.backend=="cpu" else int(pipeline.renderer.means.shape[0])}
    if args.backend=="gsplat":
        stats.update(gpu=torch.cuda.get_device_name(),
                     peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                     peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    (out/"metrics.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    print(json.dumps(stats,indent=2))


if __name__=="__main__":
    main()
