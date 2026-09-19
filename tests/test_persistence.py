import numpy as np
from scene_manager.models import default_scene
from scene_manager.persistence import save_scene, load_scene
from scene_manager.frames import scene_frames
from scene_manager.transforms import pose_matrix
from avm_bringup.assets import generate_assets


def test_scene_save_load_relative_assets(tmp_path):
    assets = tmp_path / "assets"
    generate_assets(assets)
    scene = default_scene(assets)
    scene.vehicle.pose = [1,2,.3,.1,.2,.3]
    scene.cameras["left"].D = [.01,-.002,0,0]
    target = tmp_path / "nested" / "scene.yaml"
    save_scene(scene, target)
    loaded = load_scene(target)
    assert loaded.to_dict() == scene.to_dict()
    assert "../assets/" in target.read_text()
    assert not list(target.parent.glob("*.tmp"))


def test_tf_chain_matches_camera(tmp_path):
    scene = default_scene(tmp_path)
    scene.vehicle.pose = [1,2,0,.1,.2,1.3]
    frames = {f.child:f for f in scene_frames(scene)}
    t = frames["base_link"].transform @ frames["camera_front_link"].transform @ frames["camera_front_optical"].transform
    np.testing.assert_allclose(t, scene.camera_optical_pose("front"), atol=1e-12)
    assert frames["board_001"].parent == "world"


def test_legacy_mesh_transform_names_migrate(tmp_path):
    scene = default_scene(tmp_path)
    data = scene.to_dict()
    data["vehicle"]["mesh_pose"] = data["vehicle"].pop("asset_pose")
    data["vehicle"]["mesh_scale"] = data["vehicle"].pop("asset_scale")
    migrated = type(scene).from_dict(data)
    assert migrated.vehicle.asset_pose == [0.] * 6
    assert migrated.vehicle.asset_scale == 1.
