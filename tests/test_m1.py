import numpy as np
import pytest
from avm_bringup.assets import generate_assets
from gs_renderer.gaussian import GaussianScene
from scene_manager.transforms import pose_matrix, inverse, transform_points, R_LINK_OPTICAL


def test_gaussian_load_sim3(tmp_path):
    generate_assets(tmp_path)
    a = GaussianScene.load(tmp_path / "test_room.ply")
    t = pose_matrix([1, 2, 3, .1, .2, .3])
    b = GaussianScene.load(tmp_path / "test_room.ply", 2., t)
    np.testing.assert_allclose(b.means, transform_points(t, a.means * 2), atol=1e-6)
    np.testing.assert_allclose(b.covariances,
        4 * (t[:3, :3] @ a.covariances @ t[:3, :3].T), atol=1e-6)
    assert (np.linalg.eigvalsh(b.covariances) > 0).all()


def test_transform_conventions():
    np.testing.assert_array_equal(R_LINK_OPTICAL @ [0, 0, 1], [1, 0, 0])
    np.testing.assert_array_equal(R_LINK_OPTICAL @ [1, 0, 0], [0, -1, 0])
    np.testing.assert_array_equal(R_LINK_OPTICAL @ [0, 1, 0], [0, 0, -1])
    t = pose_matrix([1, 2, 3, .3, -.2, 1])
    np.testing.assert_allclose(inverse(t) @ t, np.eye(4), atol=1e-14)
    with pytest.raises(ValueError):
        GaussianScene.load("unused.ply", -1)

