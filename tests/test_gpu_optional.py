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
