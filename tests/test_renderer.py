import numpy as np
from avm_bringup.assets import generate_assets
from gs_renderer.gaussian import GaussianScene
from gs_renderer.cpu_renderer import CpuRayRenderer
from scene_manager.models import Camera
from scene_manager.transforms import optical_pose, pose_matrix


def test_pinhole_pose_changes_sensor_image(tmp_path):
    generate_assets(tmp_path)
    renderer=CpuRayRenderer(GaussianScene.load(tmp_path/"test_room.ply"))
    c=Camera(model="pinhole",width=64,height=48,K=[40,0,31.5,0,40,23.5,0,0,1],D=[0]*5)
    a=renderer.render(c,optical_pose(pose_matrix([0,0,1,0,.6,0])))
    b=renderer.render(c,optical_pose(pose_matrix([0,0,1,0,.6,1.2])))
    assert a.rgb.shape==(48,64,3)
    assert np.mean(np.abs(a.rgb.astype(float)-b.rgb.astype(float)))>3
    assert np.isfinite(a.depth).any()
