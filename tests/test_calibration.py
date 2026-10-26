from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
from avm_bringup.assets import generate_assets
from scene_manager.models import default_scene
from scene_manager.ground_truth import GroundTruth
from gs_renderer.calibration import make_calibration_dataset,calibrate_camera_dataset,sample_board_poses


@pytest.mark.parametrize("model",["pinhole","fisheye"])
def test_image_detection_opencv_calibration_closed_loop(tmp_path,model):
    generate_assets(tmp_path/"assets")
    s=default_scene(tmp_path/"assets")
    c=s.cameras["left"]
    c.model=model
    if model=="pinhole":
        c.D=[.015,-.003,.0005,-.0002,0]
        c.K=[150.,0,159.5,0,150.,119.5,0,0,1]
    else:
        c.D=[.02,-.004,.0007,-.0001]
    t=GroundTruth.capture(s)
    out=tmp_path/"dataset"
    metadata=make_calibration_dataset(s,out,32,20261020,2,["left"])
    r=calibrate_camera_dataset(out,"left",GroundTruth.load(out/"ground_truth.yaml"))
    assert r["views_detected"]>=20
    assert r["reprojection_rms_px"]<.5
    assert r["intrinsic_error"]["focal_relative_percent"]<2.5
    assert r["extrinsics_error"]["translation_rmse_m"]<.05
    assert r["extrinsics_error"]["rotation_rmse_deg"]<2.
    assert t.scene.to_dict()==s.to_dict()
    assert (out/"scene.yaml").is_file() and (out/"ground_truth.yaml").is_file()
    assert (out/"metadata.json").is_file() and (out/"board_poses.csv").is_file()
    # Changing GT must not be accepted for an existing dataset.
    changed=deepcopy(s);changed.cameras["left"].K[0]+=10
    with pytest.raises(ValueError,match="different Ground Truth"):
        calibrate_camera_dataset(out,"left",GroundTruth.capture(changed))


def test_seed_reproducibility_and_overwrite_guard(tmp_path):
    generate_assets(tmp_path/"assets")
    s=default_scene(tmp_path/"assets")
    a=sample_board_poses(s,"left",12,12)
    assert a==sample_board_poses(s,"left",12,12)
    assert a!=sample_board_poses(s,"left",12,13)
    first=make_calibration_dataset(s,tmp_path/"a",16,42,1,["left"])
    second=make_calibration_dataset(s,tmp_path/"b",16,42,1,["left"])
    assert first==second
    assert (tmp_path/"a"/"board_poses.csv").read_bytes()==(tmp_path/"b"/"board_poses.csv").read_bytes()
    with pytest.raises(FileExistsError):
        make_calibration_dataset(s,tmp_path/"a",16,42,1,["left"])
    # Recalibration verifies the immutable Ground Truth and manifest before reading PNGs.
    result=calibrate_camera_dataset(tmp_path/"a","left",GroundTruth.load(tmp_path/"a"/"ground_truth.yaml"))
    assert result["views_detected"]>=8
    image=tmp_path/"a"/first["views"][0]["image"]
    image.write_bytes((tmp_path/"a"/first["views"][1]["image"]).read_bytes())
    with pytest.raises(ValueError,match="checksum"):
        calibrate_camera_dataset(tmp_path/"a","left",GroundTruth.capture(s))


def test_seeded_samples_keep_all_board_corners_inside_camera_domain(tmp_path):
    """Regression: narrow/offset camera poses must resample clipped checkerboards."""
    from scene_manager.camera_models import project
    from scene_manager.transforms import inverse,transform_points,pose_matrix
    from scene_manager.ground_truth import board_corners
    from scipy.spatial.transform import Rotation

    s=default_scene(tmp_path)
    s.cameras["left"].pose=[.143176585303994,-2.406016372212127,.12080512371045671,
        .13862145450335728,-.016775341856597903,-1.468590312211759]
    s.cameras["rear"].pose=[4.403242713756516,.03251434131360561,.6400000000000001,
        0.,0.,.043012351578261936]
    s.cameras["right"].pose=[.0390352081394618,2.0913951967852955,.6205432859103461,
        -.005652866316666061,.05205411364871981,1.466076571675237]
    s.boards["board_001"].rows=6
    s.boards["board_001"].columns=8
    s.boards["board_001"].square_size=.15
    for name in s.cameras:
        poses=sample_board_poses(s,name,40,202610)
        camera=s.cameras[name]
        T_camera_world=inverse(s.camera_optical_pose(name))
        corners=board_corners(s.boards["board_001"])
        for pose in poses:
            points=transform_points(T_camera_world,transform_points(pose_matrix(pose),corners))
            _,valid=project(camera,points)
            assert valid.all(),(name,pose)
