from copy import deepcopy
import json
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scene_editor.multitilt import ALGORITHM,normalize,fit_intrinsics,detect_target,propose,regular_grid
from scene_manager.transforms import pose_matrix


def synthetic():
    rng=np.random.default_rng(19)
    objects=np.array([[x*.12,y*.12,0.] for y in range(5) for x in range(7)])
    objects-=objects.mean(axis=0)
    k=np.array([[320.,0,640.],[0,324.,480.],[0,0,1.]])
    d=np.array([.03,.04,-.008,-.001])
    rows=[]
    for i in range(28):
        rv=Rotation.from_euler('xyz',rng.uniform([-.7,-.7,-.7],[.7,.7,.7])).as_rotvec()
        tv=np.r_[rng.uniform(-.6,.6,2),rng.uniform(.9,2.)]
        pixels=cv2.fisheye.projectPoints(objects[:,None],rv,tv,k,d)[0].reshape(-1,2)
        rows.append({'image':f'{i}.png','split':'train' if i<24 else 'holdout','corners_px':pixels.tolist()})
    return {'model':'fisheye','resolution_px':[1280,960],'object_points_m':objects.tolist(),'observations':rows},k,d


def test_independent_pose_solver_recovers_lens_without_truth():
    public,k,d=synthetic();result=fit_intrinsics(public)
    assert result['intrinsics_accepted'],result
    np.testing.assert_allclose(result['K'],k,atol=.05)
    np.testing.assert_allclose(result['D'],d,atol=.002)
    assert result['heldout_max_frame_rmse_px']<.01
    assert result['normal_span_deg']>30 and result['distance_ratio']>1.2
    public['K']=k.tolist()
    with pytest.raises(ValueError,match='仅允许'):fit_intrinsics(public)


def test_target_roi_does_not_detect_background_board():
    rgb=np.full((320,600,3),240,np.uint8)
    pitch=25
    for row in range(6):
        for col in range(8):
            rgb[50+row*pitch:50+(row+1)*pitch,30+col*pitch:30+(col+1)*pitch]=12 if (row+col)%2 else 240
    assert detect_target(rgb,[7,5],[[20,40],[240,40],[240,210],[20,210]]) is not None
    assert detect_target(rgb,[7,5],[[330,40],[560,40],[560,240],[330,240]]) is None


def test_outer_border_false_corner_fails_grid_check():
    points=np.array([[x*25., y*25.] for y in range(5) for x in range(7)])
    assert regular_grid(points,[7,5])
    points[-1]+=[-15.,40.]
    assert not regular_grid(points,[7,5])


def test_sampling_changes_normal_distance_preserves_camera_and_other_boards(tmp_path):
    from test_scene_pipeline import fixture_scene
    s=fixture_scene(tmp_path);c=s.cameras['front'];c.model='fisheye';c.D=[0.]*4
    r=normalize({'intrinsic_board':'board_small','views':12,'holdout':3,'backend':'cpu'},s)
    before=s.to_dict();rng=np.random.default_rng(10);accepted=[]
    for _ in range(12):
        candidate,rejected=propose(s,'front',s.boards['board_small'],r,rng,None,accepted)
        assert candidate is not None,rejected
        accepted.append(candidate)
    normals=np.array([a['normal'] for a in accepted]);distances=np.array([a['distance'] for a in accepted])
    assert np.rad2deg(np.arccos(np.clip(normals@normals.T,-1,1))).max()>30
    assert distances.max()/distances.min()>1.2
    assert before==s.to_dict()


def test_ui_saves_and_restores_new_route(tmp_path):
    import viser
    from test_scene_pipeline import fixture_scene
    from scene_manager.store import SceneStore
    from scene_editor.calibration_jobs import CalibrationJobs
    from scene_editor.calibration_ui import CalibrationPanel,ALGORITHMS
    s=fixture_scene(tmp_path);s.cameras['front'].model='fisheye';s.cameras['front'].D=[0.]*4
    jobs=CalibrationJobs(experiments=tmp_path/'jobs');store=SceneStore(s)
    server=viser.ViserServer(host='127.0.0.1',port=0,verbose=False)
    try:
        panel=CalibrationPanel(server,store,jobs=jobs)
        panel.algorithm.value=next(n for n,v in ALGORITHMS.items() if v==ALGORITHM)
        panel.events.put('algorithm');panel.tick()
        assert panel.tilt.visible and not panel.mapping_folder.visible and not panel.input_mode.visible
        panel.intrinsic_board.value='board_small';panel.tilt.value=35.
        panel.events.put('save_settings');panel.tick()
        saved=store.snapshot()[1].calibration
        assert saved['algorithm']==ALGORITHM and saved['tilt_deg']==35.
        panel._load_saved(store.snapshot()[1]);assert panel.tilt.value==35.
    finally:jobs.close();server.stop()


def test_invalid_inputs_and_no_gt_in_observation():
    public,_,_=synthetic()
    public['observations'][0]['board_pose']=[0.]*6
    with pytest.raises(ValueError,match='真值'):fit_intrinsics(public)
    public,_,_=synthetic();public['observations'][0]['corners_px'][0][0]=float('nan')
    with pytest.raises(ValueError,match='角点'):fit_intrinsics(public)


def test_holdout_does_not_change_fitted_intrinsics_but_can_reject():
    public,_,_=synthetic();first=fit_intrinsics(public)
    rng=np.random.default_rng(7)
    for row in public['observations']:
        if row['split']=='holdout':
            row['corners_px']=(np.array(row['corners_px'])+rng.normal(0,8.,(35,2))).tolist()
    second=fit_intrinsics(public)
    np.testing.assert_array_equal(first['K'],second['K'])
    np.testing.assert_array_equal(first['D'],second['D'])
    assert not second['intrinsics_accepted'] and second['status']=='REJECTED'


def test_new_route_invalid_settings_and_cancel(tmp_path):
    from test_scene_pipeline import fixture_scene
    from scene_editor.calibration_jobs import CalibrationJobs,normalize_request,project_root
    from scene_manager.store import SceneStore
    s=fixture_scene(tmp_path);s.cameras['front'].model='fisheye';s.cameras['front'].D=[0.]*4
    base={'algorithm':ALGORITHM,'intrinsic_board':'board_small','backend':'cpu','views':12,'holdout':3}
    for update in [{'views':3},{'tilt_deg':0.},{'distance_min':3.,'distance_max':1.},{'cameras':['missing']}]:
        with pytest.raises(ValueError):normalize_request({**base,**update},s,project_root())
    jobs=CalibrationJobs(experiments=tmp_path/'jobs')
    try:
        p=jobs.start(SceneStore(s),base)
        jobs.cancel();state=jobs.poll()
        assert state['result']['status']=='cancelled'
        assert not state['result']['scene_parameters_applied']
    finally:jobs.close()
