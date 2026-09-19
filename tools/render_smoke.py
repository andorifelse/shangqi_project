"""Render a complete standalone frame and write reviewable outputs."""
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[1]
for package in (ROOT/"avm_sim_ws"/"src").iterdir():
    if package.is_dir():
        sys.path.insert(0,str(package))
import cv2
from scene_manager.models import default_scene
from scene_manager.persistence import save_scene
from avm_bringup.assets import generate_assets
from gs_renderer.pipeline import SensorPipeline
from avm_stitcher.projection import AvmStitcher


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--backend",choices=("cpu","gsplat"),default="cpu")
    args=parser.parse_args()
    fixtures=ROOT/"avm_sim_ws"/"src"/"avm_bringup"/"example_assets"
    generate_assets(fixtures)
    scene=default_scene(fixtures)
    out=ROOT/"outputs"/"smoke"
    out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    results=SensorPipeline(args.backend).render(scene)
    images={n:r.rgb for n,r in results.items()}
    avm=AvmStitcher().stitch(scene,images)
    elapsed=time.perf_counter()-start
    for name,image in {**images,"avm":avm.rgb}.items():
        cv2.imwrite(str(out/f"{name}.png"),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    save_scene(scene,ROOT/"avm_sim_ws"/"src"/"avm_bringup"/"config"/"scene.yaml")
    stats={"backend":args.backend,"seconds":elapsed,"avm_coverage":float(avm.coverage.mean()),"cameras":list(images)}
    (out/"metrics.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    print(json.dumps(stats,indent=2))


if __name__=="__main__":
    main()
