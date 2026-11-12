"""Known physical IDs resolve external pose without camera-truth initialization."""
from copy import deepcopy
import json
import cv2
import numpy as np
import pytest
from scene_editor.marker_extrinsics import fit_extrinsics, public_geometry, detect_markers
from scene_editor.fisheye_solver import project
from scene_manager.transforms import pose_matrix, transform_points


def marker_observations():
    rng = np.random.default_rng(9)
    k = np.array([[320., 0., 640.], [0., 324., 480.], [0., 0., 1.]])
    d = np.array([.03, .04, -.008, -.001])
    pose = np.array([.4, -.5, .2, .1, -.2, 2.])
    geometry, observations = {}, {}
    for i, (x, y) in enumerate([(-.7, -.5), (.5, -.5), (-.7, .5), (.5, .5)]):
        corners = np.array([[x, y, .003], [x+.2, y, .003], [x+.2, y+.2, .003], [x, y+.2, .003]])
        geometry[str(i)] = corners.tolist()
        observations[str(i)] = (project(corners, pose, k, d)+rng.normal(0, .03, (4, 2))).tolist()
    return {'model': 'fisheye', 'K': k.tolist(), 'D': d.tolist(),
        'geometry': {'markers': geometry}, 'observations': observations}, pose


def test_id_resolved_pose_recovers_world_transform_and_rejects_wrong_id():
    document, pose = marker_observations()
    result = fit_extrinsics(document)
    assert result['extrinsics_accepted'], result
    np.testing.assert_allclose(result['R_w2c'], cv2.Rodrigues(pose[:3])[0], atol=.002)
    np.testing.assert_allclose(result['tvec'], pose[3:], atol=.002)
    assert result['heldout_rmse_px'] < .1
    broken = deepcopy(document)
    broken['observations']['0'], broken['observations']['1'] = broken['observations']['1'], broken['observations']['0']
    assert not fit_extrinsics(broken)['extrinsics_accepted']
    document['camera_pose_truth'] = pose.tolist()
    with pytest.raises(ValueError, match='仅接受'): fit_extrinsics(document)


def test_external_holdout_is_disjoint_and_cannot_be_silently_discarded():
    document, _ = marker_observations()
    for corners in document['observations'].values():
        corners[3][0] += 15.
    result = fit_extrinsics(document)
    assert not result['extrinsics_accepted']
    assert result['heldout_rmse_px'] > 10.
    document['observations'] = {'0': document['observations']['0']}
    assert not fit_extrinsics(document)['extrinsics_accepted']


def test_public_marker_geometry_tracks_asset_and_board_edits(tmp_path):
    from test_scene_pipeline import fixture_scene
    scene = fixture_scene(tmp_path)
    original = deepcopy(scene.boards['board_small'])
    records = []
    for i in range(3):
        b = deepcopy(original); b.id = f'marker_{i}'; b.asset = str(tmp_path/f'marker_{i}.ply')
        b.pose = [1., 2., 3., .1, .2, .3]; b.asset_pose = [.1, 0., 0., 0., 0., .2]; b.asset_scale = 1.5
        scene.boards[b.id] = b
        records.append({'id': i, 'file': f'marker_{i}.ply', 'corners_world_m':
            [[0., 0., 0.], [.2, 0., 0.], [.2, .2, 0.], [0., .2, 0.]]})
    path = tmp_path/'geometry.json'
    path.write_text(json.dumps({'marker_dictionary': 'DICT_4X4_1000', 'markers': records}))
    geometry = public_geometry(scene, str(path))
    expected = transform_points(pose_matrix(b.pose)@pose_matrix(b.asset_pose),
        np.array(records[0]['corners_world_m'])*b.asset_scale)
    np.testing.assert_allclose(geometry['markers']['0'], expected)


def test_marker_rgb_detection_preserves_decoded_corner_order():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    rgb = np.full((600, 800, 3), 255, np.uint8)
    marker = cv2.aruco.generateImageMarker(dictionary, 42, 120)
    rgb[240:360, 340:460] = marker[:, :, None]
    k = np.array([[400., 0., 400.], [0., 400., 300.], [0., 0., 1.]])
    found = detect_markers(rgb, k, np.zeros(4), 'DICT_4X4_1000')
    assert '42' in found
    np.testing.assert_allclose(found['42'], [[340., 240.], [459., 240.], [459., 359.], [340., 359.]], atol=1.)


@pytest.mark.parametrize('parent', ['world', 'base_link'])
def test_export_roundtrips_optical_pose_and_lens_without_changing_input(tmp_path, parent):
    from test_scene_pipeline import fixture_scene
    from scene_editor.marker_extrinsics import export_calibrated_scene
    from scene_manager.persistence import load_scene
    scene = fixture_scene(tmp_path)
    scene.cameras['front'].parent = parent
    scene.cameras['front'].model = 'fisheye'
    scene.cameras['front'].D = [0.]*4
    scene.vehicle.pose = [1., 2., .1, .1, -.1, .3]
    before = scene.to_dict()
    doc, _ = marker_observations()
    external = fit_extrinsics(doc)
    lens = {'K': doc['K'], 'D': doc['D'], 'intrinsics_accepted': True}
    export_calibrated_scene(scene, {'front': lens}, {'front': external}, tmp_path)
    loaded = load_scene(tmp_path/'calibrated_scene.yaml')
    np.testing.assert_allclose(loaded.camera_optical_pose('front'), external['T_world_from_camera_optical'], atol=1e-12)
    np.testing.assert_allclose(loaded.cameras['front'].matrix, doc['K'])
    assert before == scene.to_dict()
    lens['intrinsics_accepted'] = False
    with pytest.raises(ValueError, match='均通过'):
        export_calibrated_scene(scene, {'front': lens}, {'front': external}, tmp_path)
