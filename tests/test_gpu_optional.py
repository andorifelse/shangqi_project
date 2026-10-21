"""Run only when explicitly requested on the target CUDA host."""
import os
import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get("AVM_TEST_GPU")!="1",reason="Set AVM_TEST_GPU=1 on a configured CUDA host")
def test_native_gsplat_four_fisheye(tmp_path):
    import torch
    assert torch.cuda.is_available()
    from avm_bringup.assets import generate_assets
    from scene_manager.models import default_scene
    from gs_renderer.pipeline import SensorPipeline
    generate_assets(tmp_path)
    scene=default_scene(tmp_path)
    results=SensorPipeline("gsplat").render(scene)
    assert set(results)=={"front","rear","left","right"}
    for result in results.values():
        assert result.rgb.shape==(240,320,3)
        assert result.rgb.std()>5 and np.isfinite(result.depth).any()


@pytest.mark.skipif(os.environ.get("AVM_TEST_GPU")!="1",reason="Set AVM_TEST_GPU=1 on a configured CUDA host")
def test_left_camera_edit_changes_only_left_image_with_reordered_cameras(tmp_path):
    from copy import deepcopy
    from avm_bringup.assets import generate_assets
    from scene_manager.models import default_scene
    from gs_renderer.pipeline import SensorPipeline
    generate_assets(tmp_path)
    scene = default_scene(tmp_path)
    scene.cameras = {n:scene.cameras[n] for n in ("rear", "left", "right", "front")}
    scene.boards["board_001"].pose = [0, 3., .25, 0, 0, 0]
    pipeline = SensorPipeline("gsplat")
    initial = pipeline.render(scene)
    without_board = deepcopy(scene)
    without_board.boards = {}
    background = pipeline.render(without_board)
    for name in scene.cameras:
        pixels = np.any(initial[name].rgb != background[name].rgb, axis=-1).sum()
        assert (pixels > 0) == (name == "left"), (name, pixels)
    scene.cameras["left"].pose[0] += .3
    changed = pipeline.render(scene)
    assert np.any(changed["left"].rgb != initial["left"].rgb)
    for name in ("rear", "right", "front"):
        np.testing.assert_array_equal(changed[name].rgb, initial[name].rgb)


@pytest.mark.skipif(os.environ.get("AVM_TEST_GPU")!="1",reason="Set AVM_TEST_GPU=1 on a configured CUDA host")
@pytest.mark.parametrize("model",["pinhole","fisheye"])
def test_native_gaussian_centroid_matches_opencv_world_projection(tmp_path,model):
    """Independent numeric image evidence for optical transforms and K + 0.5."""
    from copy import deepcopy
    from scene_manager.models import default_scene
    from scene_manager.camera_models import world_to_pixel
    from scene_manager.transforms import transform_points
    from gs_renderer.gaussian import GaussianScene
    from gs_renderer.gsplat_renderer import GsplatRenderer
    scene=default_scene(tmp_path)
    scene.vehicle.pose=[.3,-.4,.2,.1,-.15,.6]
    for name,camera in scene.cameras.items():
        camera.model=model
        camera.D=[.02,-.003,.0005,-.0001] if model=="fisheye" else [.02,-.003,.0002,-.0003,.0001]
        camera.width,camera.height=128,96
        camera.K=[60.,0,63.5,0,60.,47.5,0,0,1]
        t=scene.camera_optical_pose(name)
        point=transform_points(t,np.array([[.2,-.1,2.]]))
        expected,valid=world_to_pixel(scene,name,point)
        assert valid.all()
        scale=.025
        gs=GaussianScene(point.astype(np.float32),np.eye(3,dtype=np.float32)[None]*scale**2,
            np.ones((1,3),np.float32),np.array([.95],np.float32),
            np.array([[1.,0,0,0]],np.float32),np.full((1,3),scale,np.float32))
        one=deepcopy(scene)
        one.cameras={name:camera}
        output=GsplatRenderer(gs).render_batch(one)[name]
        yy,xx=np.indices(output.alpha.shape)
        alpha=output.alpha
        assert alpha.sum()>.01
        measured=np.array([(xx*alpha).sum(),(yy*alpha).sum()])/alpha.sum()
        np.testing.assert_allclose(measured,expected[0],atol=.35)
