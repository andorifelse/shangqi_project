import numpy as np
from avm_bringup.assets import generate_assets
from gs_renderer.gaussian import GaussianScene
from gs_renderer.cpu_renderer import CpuRayRenderer
from gs_renderer.pipeline import SensorPipeline
from scene_manager.models import Camera, default_scene
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


def test_vehicle_gaussian_is_composed_and_follows_base_link(tmp_path):
    generate_assets(tmp_path)
    scene = default_scene(tmp_path)
    base_count = len(GaussianScene.load(scene.gaussian_file).means)
    scene.vehicle.asset = scene.gaussian_file
    scene.vehicle.asset_scale = .2
    scene.vehicle.asset_pose = [0, 0, .3, 0, 0, -np.pi/2]
    for camera in scene.cameras.values():
        camera.width, camera.height = 32, 24
        camera.K = [14., 0, 15.5, 0, 14., 11.5, 0, 0, 1]
    pipeline = SensorPipeline("cpu")
    first = pipeline.render(scene)
    assert len(pipeline.renderer.gs.means) == 2 * base_count
    scene.vehicle.pose = [1., 0, 0, 0, 0, .2]
    second = pipeline.render(scene)
    assert len(pipeline.renderer.gs.means) == 2 * base_count
    assert np.mean(np.abs(first["front"].rgb.astype(float)-second["front"].rgb.astype(float))) > .1
