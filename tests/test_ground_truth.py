from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import yaml
from scene_manager.models import default_scene
from scene_manager.store import SceneStore
from scene_manager.ground_truth import GroundTruth, board_corners
from scene_manager.transforms import inverse
from avm_bringup.assets import generate_assets
from avm_bringup.experiments import begin_experiment, finish_experiment


def test_truth_snapshot_roundtrip_revision_and_tamper_detection(tmp_path):
    generate_assets(tmp_path/"assets")
    scene=default_scene(tmp_path/"assets")
    store=SceneStore(scene)
    store.command({"action":"pose","id":"base_link","pose":[1,2,.5,.1,-.2,.7]})
    truth=store.ground_truth()
    scene.vehicle.pose[0]=99
    leaked=truth.scene
    leaked.cameras["left"].K[0]=999
    assert truth.scene.cameras["left"].K[0]==110.
    path=tmp_path/"export"/"ground_truth.yaml"
    store.command({"action":"export_ground_truth","path":str(path)})
    restored=GroundTruth.load(path)
    assert restored.revision==1
    assert restored.sha256==truth.sha256
    assert restored.scene.to_dict()==truth.scene.to_dict()
    for n,c in truth.to_dict()["derived"]["cameras"].items():
        np.testing.assert_allclose(np.asarray(c["T_camera_optical_from_world"]) @ c["T_world_from_camera_optical"],np.eye(4),atol=1e-12)
    data=yaml.safe_load(path.read_text())
    data["derived"]["cameras"]["left"]["T_world_from_camera_optical"][0][3]+=1
    path.write_text(yaml.safe_dump(data),encoding="utf-8")
    with pytest.raises(ValueError,match="derived"):
        GroundTruth.load(path)


def test_scene_store_exports_revisioned_ground_truth(tmp_path):
    generate_assets(tmp_path/"assets")
    store=SceneStore(default_scene(tmp_path/"assets"))
    store.command({"action":"pose","id":"base_link","pose":[1,2,3,.1,.2,.3]})
    path=tmp_path/"scene"/"ground_truth.yaml"
    assert store.command({"action":"export_ground_truth","path":str(path)})==1
    truth=GroundTruth.load(path)
    assert truth.revision==1
    assert truth.scene.vehicle.pose==[1,2,3,.1,.2,.3]


def test_board_corner_metric_definition(tmp_path):
    b=default_scene(tmp_path).boards["board_001"]
    corners=board_corners(b).reshape(b.rows-1,b.columns-1,3)
    assert corners.shape==(5,7,3)
    np.testing.assert_allclose(corners[0,1]-corners[0,0],[b.square_size,0,0])
    np.testing.assert_allclose(corners[1,0]-corners[0,0],[0,b.square_size,0])
    np.testing.assert_allclose(corners[0,0],[-b.width/2+b.square_size,-b.height/2+b.square_size,0])


def test_experiment_manifest_contains_replay_inputs_and_runtime_hashes(tmp_path):
    assets=tmp_path/"assets";generate_assets(assets)
    scene=default_scene(assets)
    experiment,truth=begin_experiment(scene,tmp_path/"experiments"/"case-1",
        {"seed":42,"views":12},Path(__file__).resolve().parents[1])
    assert {p.name for p in experiment.iterdir()}=={"scene.yaml","ground_truth.yaml","config.yaml","logs"}
    assert GroundTruth.load(experiment/"ground_truth.yaml").sha256==truth.sha256
    manifest=yaml.safe_load((experiment/"config.yaml").read_text())
    assert manifest=={"seed":42,"views":12}
    runtime=__import__("json").loads((experiment/"logs"/"runtime.json").read_text())
    assert runtime["source_sha256"]
    finish_experiment(experiment,{"status":"complete"})
    assert __import__("json").loads((experiment/"result.json").read_text())["status"]=="complete"
    assert (experiment/"logs"/"assets.json").is_file()
