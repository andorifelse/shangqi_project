"""Analytic ground-color rendering validates the complete ray-to-BEV chain."""
import numpy as np
from scene_manager.models import default_scene
from scene_manager.camera_models import camera_rays
from avm_stitcher.projection import AvmStitcher,ground_points


def test_four_camera_ground_coordinates_reconstruct_bev(tmp_path):
    scene=default_scene(tmp_path)
    scene.vehicle.pose=[.4,-.2,0,0,0,.2]
    images={}
    for name,camera in scene.cameras.items():
        rays,valid=camera_rays(camera)
        t=scene.camera_optical_pose(name)
        directions=rays@t[:3,:3].T
        distance=np.divide(-t[2,3],directions[...,2],
            out=np.zeros(valid.shape),where=np.abs(directions[...,2])>1e-8)
        hit=t[:3,3]+distance[...,None]*directions
        colors=np.stack([(hit[...,0]+7)*15,(hit[...,1]+7)*15,np.full(valid.shape,80)],axis=-1)
        colors[(distance<=0)|~valid]=0
        images[name]=np.clip(colors,0,255).astype(np.uint8)
    avm=AvmStitcher().stitch(scene,images)
    ground=ground_points(scene)
    expected=np.stack([(ground[...,0]+7)*15,(ground[...,1]+7)*15,np.full(ground.shape[:2],80)],axis=-1)
    error=np.abs(avm.rgb.astype(float)-expected)[avm.coverage]
    assert np.median(error)<1.1
    assert np.quantile(error,.99)<4


def test_bad_scene_is_rejected_without_mutating_store(tmp_path):
    import pytest
    from scene_manager.persistence import load_scene
    from scene_manager.store import SceneStore
    store=SceneStore(default_scene(tmp_path))
    before=store.snapshot()
    path=tmp_path/"bad.yaml"
    path.write_text("scene: [wrong]\ncameras: wrong")
    with pytest.raises(ValueError):
        store.command({"action":"load","path":str(path)})
    assert store.snapshot()[0]==before[0]
    assert store.snapshot()[1].to_dict()==before[1].to_dict()
