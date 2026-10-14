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
