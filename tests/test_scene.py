import numpy as np
import pytest
from scene_manager.models import default_scene
from scene_manager.store import SceneStore
from scene_manager.transforms import pose_matrix


def test_parent_motion_preserves_extrinsic(tmp_path):
    store = SceneStore(default_scene(tmp_path))
    _, initial = store.snapshot()
    extrinsic = list(initial.cameras["front"].pose)
    store.command({"action":"pose","id":"base_link","pose":[1,2,0,0,0,.8]})
    _, scene = store.snapshot()
    assert scene.cameras["front"].pose == extrinsic
    np.testing.assert_allclose(scene.world_pose("front"),
        pose_matrix(scene.vehicle.pose) @ pose_matrix(extrinsic))


def test_crud_atomic_validation(tmp_path):
    store = SceneStore(default_scene(tmp_path))
    store.command({"action":"add_board","name":"board_002"})
    store.command({"action":"pose","id":"board_002","pose":[1,2,3,0,0,1]})
    rev, before = store.snapshot()
    with pytest.raises(ValueError):
        store.command({"action":"properties","id":"front","properties":{"width":-1}})
    assert store.snapshot()[0] == rev
    assert store.snapshot()[1].to_dict() == before.to_dict()
    with pytest.raises(ValueError):
        store.command({"action":"delete","id":"front"}, expected_revision=rev-1)
    store.command({"action":"delete","id":"base_link"})
    assert not store.snapshot()[1].cameras
    assert len(store.snapshot()[1].boards) == 2


def test_six_camera_limit(tmp_path):
    store = SceneStore(default_scene(tmp_path))
    for n in ("extra_left","extra_right"):
        store.command({"action":"add_camera","name":n})
    with pytest.raises(ValueError):
        store.command({"action":"add_camera","name":"seventh"})
