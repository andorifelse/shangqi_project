import numpy as np
from scene_manager.models import default_scene
from scene_manager.transforms import inverse,transform_points
from scene_manager.camera_models import project
from avm_stitcher.projection import AvmStitcher,ground_points,projection_maps


def test_avm_ground_plane_and_pixel_mapping(tmp_path):
    scene=default_scene(tmp_path)
    scene.vehicle.pose=[1,2,.1,.15,-.12,.3]
    points=ground_points(scene)
    np.testing.assert_allclose(points[...,2],scene.coordinate.ground_z,atol=1e-12)
    uv,weight=projection_maps(scene,"front")
    expected,_=project(scene.cameras["front"],transform_points(inverse(scene.camera_optical_pose("front")),points))
    np.testing.assert_allclose(uv,expected,atol=1e-4)
    assert np.any(weight>0)


def test_avm_consumes_camera_pixels(tmp_path):
    scene=default_scene(tmp_path)
    images={n:np.full((c.height,c.width,3),[60,120,180],np.uint8) for n,c in scene.cameras.items()}
    stitcher=AvmStitcher()
    a=stitcher.stitch(scene,images)
    np.testing.assert_allclose(a.rgb[a.coverage],np.tile([60,120,180],(a.coverage.sum(),1)),atol=1)
    images["front"][:]=[240,30,10]
    b=stitcher.stitch(scene,images)
    assert np.any(a.rgb!=b.rgb)
    assert a.coverage.mean()>.5


def test_bev_orientation(tmp_path):
    scene=default_scene(tmp_path)
    p=ground_points(scene)
    assert p[0,0,0]>p[-1,0,0]
    assert p[0,0,1]>p[0,-1,1]
