from pathlib import Path
from copy import deepcopy
import json
import numpy as np
import pytest
from avm_bringup.assets import generate_assets
from scene_manager.models import default_scene,Board,Bev
from scene_manager.board_geometry import grid_geometry,board_geometry
from scene_manager.store import SceneStore
from scene_manager.persistence import save_scene,load_scene
from scene_manager.ground_truth import GroundTruth
from scene_manager.transforms import pose_matrix
from scene_editor.scene_pipeline import (normalize_scene_request,public_corners,evaluation_frame,
    configs,capture_scene,run_scene_pipeline,sample_intrinsic_pose)
from scene_editor.calibration_jobs import CalibrationJobs,USER_PIPELINE,pipeline_class,project_root
from test_calibration_jobs import wait_job


def fixture_scene(tmp_path):
    generate_assets(tmp_path)
    s=default_scene(tmp_path)
    c=s.cameras['front'];c.parent='world';c.pose=[0,0,2,0,np.pi/2,0]
    c.model='pinhole';c.width=480;c.height=360;c.K=[340.,0,239.5,0,340.,179.5,0,0,1];c.D=[0.]*5
    s.cameras={'front':c}
    s.vehicle.asset='';s.vehicle.pose=[20.,20.,0,0,0,0]
    s.boards={'board_small':Board(id='board_small',pose=[0,0,.1,0,0,0]),
        'board_large':Board(id='board_large',rows=5,columns=4,square_size=.25,pose=[0,0,.1,0,0,0])}
    s.bev=Bev(x_min=-.4,x_max=.4,y_min=-.4,y_max=.4,width=64,height=64)
    return s


def request():
    return {'algorithm':USER_PIPELINE,'input_mode':'scene','cameras':['front'],
        'intrinsic_board':'board_small','extrinsic_board':'board_large',
        'camera_regions':{'front':'board_large'},'views':6,'holdout':2,'seed':20261103,
        'spread_x':.17,'spread_y':.12,'yaw_range_deg':15.,'max_attempts':5,'backend':'cpu','sampling':'around_pose'}


def test_geometry_persistence_and_missing_gaussian_definitions(tmp_path):
    s=fixture_scene(tmp_path);b=s.boards['board_large'];b.asset=s.gaussian_file
    with pytest.raises(ValueError,match='高斯板未定义'):normalize_scene_request(s,request())
    exported=GroundTruth.capture(s).to_dict()['derived']['boards'][b.id]
    assert exported['inner_corners_board_m'] is None and exported['width_m'] is None
    assert exported['geometry_status']=='MISSING_OR_UNVERIFIED'
    g=grid_geometry('front_strip',3,4,.25,[-.25,-.375,0],source='measured test target')
    store=SceneStore(s)
    store.command({'action':'properties','id':b.id,'properties':{'calibration_geometry':g}})
    settings=request();settings['camera_regions']={'front':'front_strip'}
    store.command({'action':'calibration_settings','settings':settings})
    path=tmp_path/'scene.yaml';save_scene(store.snapshot()[1],path)
    loaded=load_scene(path)
    assert loaded.calibration==settings
    assert loaded.boards[b.id].calibration_geometry==g
    assert normalize_scene_request(loaded,settings)['camera_regions']=={'front':'front_strip'}
    truth=tmp_path/'truth.yaml';store.ground_truth().save(truth)
    assert GroundTruth.load(truth).scene.to_dict()==loaded.to_dict()


def test_public_frame_and_scale_and_no_camera_truth_in_generated_specs(tmp_path):
    s=fixture_scene(tmp_path);s.vehicle.pose=[2.,3.,.4,.1,.2,.7]
    b=s.boards['board_large'];b.asset_scale=2
    g=grid_geometry('strip',3,4,.1,source='CAD original units',frame='asset')
    b.calibration_geometry=g;b.pose=[2.,3.,.1,0,0,.7]
    r=request();r['camera_regions']={'front':'strip'};r=normalize_scene_request(s,r)
    assert np.isclose(np.linalg.norm(np.array(r['extrinsic_geometry']['corners'][1]['xyz_m'])-
        r['extrinsic_geometry']['corners'][0]['xyz_m']),.2)
    public=public_corners(s,r)
    np.testing.assert_allclose(public['corners'][0]['xyz_world_m'],[0.,0.,.1],atol=1e-10)
    folder=tmp_path/'job';folder.mkdir()
    specs=configs(s,r,folder,{'front':[{'image':'a.png','split':'train'},{'image':'b.png','split':'holdout'}]})
    solver=json.loads((folder/'solver_inputs/solver_group_0.json').read_text())
    forbidden={'K','D','R_w2c','tvec','camera_center_world','board_to_world','pose','scene_snapshot'}
    def check(v):
        if isinstance(v,dict):
            assert not set(v)&forbidden
            for x in v.values():check(x)
        elif isinstance(v,list):
            for x in v:check(x)
    check(solver)
    assert specs[0]['stages']==['intrinsics','extrinsics']


@pytest.mark.parametrize('gaussian_small',[False,True])
def test_capture_removes_only_intrinsic_board_after_all_cameras(tmp_path,monkeypatch,gaussian_small):
    from types import SimpleNamespace
    import cv2
    from gs_renderer.pipeline import SensorPipeline
    s=fixture_scene(tmp_path)
    s.boards['board_extra']=Board(id='board_extra',pose=[3,4,.2,0,0,.3])
    if gaussian_small:
        b=s.boards['board_small'];b.asset=s.gaussian_file
        b.calibration_geometry=grid_geometry('small',b.columns-1,b.rows-1,b.square_size,
            [-b.width/2+b.square_size,-b.height/2+b.square_size,0],source='test metric geometry')
    other=deepcopy(s.cameras['front']);other.name='right';s.cameras['right']=other
    r=request();r.update(cameras=['front','right'],camera_regions={'front':'board_large','right':'board_large'},
        views=4,holdout=1,spread_x=0.,spread_y=0.,yaw_range_deg=0.)
    r=normalize_scene_request(s,r);before=s.to_dict();samples=[]
    def render(self,sample):
        samples.append(deepcopy(sample))
        return {name:SimpleNamespace(rgb=np.zeros((c.height,c.width,3),np.uint8))
            for name,c in sample.cameras.items()}
    monkeypatch.setattr(SensorPipeline,'render',render)
    monkeypatch.setattr(cv2,'findChessboardCornersSB',lambda *args,**kwargs:(True,None))
    splits=capture_scene(s,r,tmp_path,lambda *args,**kwargs:None)
    intrinsic_count=(r['views']+r['holdout'])*len(r['cameras'])
    assert len(samples)==intrinsic_count+2*len(r['cameras'])
    # One target-free depth prepass per camera, then final target-free extrinsics.
    intrinsic_samples=[sample for sample in samples if 'board_small' in sample.boards]
    assert len(intrinsic_samples)==intrinsic_count
    for sample in samples:
        expected=set(s.boards) if 'board_small' in sample.boards else set(s.boards)-{'board_small'}
        assert set(sample.boards)==expected
        for name in expected:
            assert sample.boards[name]==s.boards[name]
        assert sample.vehicle==s.vehicle and sample.coordinate==s.coordinate
        assert sample.gaussian_file==s.gaussian_file
    assert s.to_dict()==before
    assert all(len(rows)==5 for rows in splits.values())
    manifest=json.loads((tmp_path/'capture/manifest.json').read_text())
    assert manifest['intrinsic_excluded_board'] is None
    assert manifest['extrinsic_excluded_board']=='board_small'
    assert set(manifest['extrinsic_boards'])=={'board_large','board_extra'}
    assert [row['kind'] for row in manifest['records']]==['intrinsics']*intrinsic_count+['extrinsics']*2


@pytest.mark.parametrize('tilted',[False,True])
def test_current_scene_capture_fixed_plane_and_original_solver_pipeline(tmp_path,tilted):
    s=fixture_scene(tmp_path);s.vehicle.pose=[0.]*6
    if tilted:s.boards['board_small'].pose=[0.,0.,.35,.2,-.15,.1]
    # Keep the retained large target behind the moving intrinsic board.
    s.boards['board_large'].pose[2]=0.
    jobs=CalibrationJobs(experiments=tmp_path/'experiments')
    before=s.to_dict();store=SceneStore(s)
    try:
        out=jobs.start(store,request())
        state=wait_job(jobs,180)
        assert state['exit_code']==0,(out/'logs/calibration.log').read_text()
        result=state['result'];assert result['input_scope']=='current_scene_snapshot'
        assert not result['calibration_accepted'] and not result['scene_parameters_applied']
        assert store.snapshot()[1].to_dict()==before
        intr=json.loads((out/'pipeline/intrinsics/pinhole/front/calibration.json').read_text())
        assert intr['status']=='DEGENERATE_PARALLEL_PLANE_ACQUISITION'
        assert intr['truth_read_by_solver'] is False and intr['training_complete']==6
        assert (out/'pipeline/bev/pinhole/comparison.png').exists()
        attempts=json.loads((out/'capture/sampling_attempts.json').read_text())
        initial=pose_matrix(s.boards['board_small'].pose)
        for row in attempts:
            sampled=pose_matrix(row['pose'])
            np.testing.assert_allclose(sampled[:3,2],initial[:3,2],atol=1e-12)
            assert abs(initial[:3,2]@(sampled[:3,3]-initial[:3,3]))<1e-12
        manifest=json.loads((out/'capture/manifest.json').read_text())
        assert len(manifest['records'])==9 and manifest['camera_motion'] is False
        assert len(result['ground_truth_comparison'])==1
        assert result['ground_truth_comparison'][0]['extrinsic_candidates']
    finally:jobs.close()


def test_protocol_rejects_missing_mapping_and_same_roles_but_accepts_tilt(tmp_path):
    s=fixture_scene(tmp_path)
    for update,match in (({'camera_regions':{}},'region'),({'extrinsic_board':'board_small'},'两个不同对象')):
        r=request();r.update(update)
        with pytest.raises(ValueError,match=match):normalize_scene_request(s,r)
    s.boards['board_small'].pose[3]=.2
    assert normalize_scene_request(s,request())['intrinsic_board']=='board_small'


@pytest.mark.parametrize('angles',[[0.,0.,.4],[.3,-.4,.5],[np.pi/2,0.,.2]])
@pytest.mark.parametrize('sampling',['around_pose','visible_plane'])
def test_intrinsic_sampling_preserves_initial_plane(tmp_path,angles,sampling):
    from scene_manager.transforms import inverse,T_LINK_OPTICAL,matrix_pose
    s=fixture_scene(tmp_path);b=s.boards['board_small'];b.pose=[.2,-.1,.3,*angles]
    initial=pose_matrix(b.pose);normal=initial[:3,2]
    # Face the board from its normal, including the vertical-board case.
    optical=initial.copy();optical[:3,:3]=initial[:3,:3]@np.diag([1.,-1.,-1.])
    optical[:3,3]=initial[:3,3]+2.*normal
    s.cameras['front'].pose=matrix_pose(optical@inverse(T_LINK_OPTICAL))
    r=request();r['sampling']=sampling;r=normalize_scene_request(s,r)
    before=s.to_dict();rng=np.random.default_rng(123)
    for _ in range(12):
        pose=sample_intrinsic_pose(s,'front',b,r,rng)
        assert pose is not None
        sampled=pose_matrix(pose)
        np.testing.assert_allclose(sampled[:3,2],normal,atol=1e-12)
        assert abs(normal@(sampled[:3,3]-initial[:3,3]))<1e-12
        relative=inverse(initial)@sampled
        assert abs(relative[0,3])<=r['spread_x']+1e-12 or sampling=='visible_plane'
        assert abs(relative[1,3])<=r['spread_y']+1e-12 or sampling=='visible_plane'
        yaw=np.arctan2(relative[1,0],relative[0,0])
        assert abs(yaw)<=np.deg2rad(r['yaw_range_deg'])+1e-12
        if sampling=='visible_plane':
            from scene_manager.camera_models import world_to_pixel
            pixel,valid=world_to_pixel(s,'front',sampled[:3,3])
            assert bool(valid)
            c=s.cameras['front']
            assert .25*c.width-1e-9<=pixel[0]<=.75*c.width+1e-9
            assert .3*c.height-1e-9<=pixel[1]<=.85*c.height+1e-9
    assert s.to_dict()==before


def test_intrinsic_sampling_rejects_parallel_camera_rays(tmp_path):
    s=fixture_scene(tmp_path);s.boards['board_small'].pose=[0,0,.1,np.pi/2,0,0]
    r=request();r['sampling']='visible_plane'
    # The camera origin lies in the vertical plane: every forward hit is invalid.
    assert sample_intrinsic_pose(s,'front',s.boards['board_small'],r,np.random.default_rng(123)) is None


def test_direct_mask_preserves_black_pixels_and_rejects_modified_mask(tmp_path):
    import cv2,hashlib
    pipeline_class(project_root())
    from calibration_pipeline.bev import sample_camera
    image=tmp_path/'black.png';cv2.imwrite(str(image),np.zeros((20,20,3),np.uint8))
    mask=tmp_path/'valid.png';valid=np.full((20,20),255,np.uint8);valid[10,10]=0;cv2.imwrite(str(mask),valid)
    camera={'id':'camera','model':'pinhole','image':str(image),'K':[[10.,0,10],[0,10.,10],[0,0,1]],
        'D':[0.]*5,'R_w2c':np.eye(3).tolist(),'tvec':[0,0,0],'image_mask':'full'}
    world=np.array([[[0.,0.,2.]]])
    assert sample_camera(world,camera,tmp_path)[1].all()  # Black RGB alone is valid.
    camera['source_valid_mask']=str(mask)
    camera['metadata']={'source_valid_mask_sha256':hashlib.sha256(mask.read_bytes()).hexdigest()}
    assert not sample_camera(world,camera,tmp_path)[1].any()
    cv2.imwrite(str(mask),np.full((20,20),255,np.uint8))
    with pytest.raises(ValueError,match='hash changed'):sample_camera(world,camera,tmp_path)


def test_ui_current_scene_roles_mappings_and_geometry_are_saved(tmp_path):
    import viser
    from scene_editor.calibration_ui import CalibrationPanel,ALGORITHMS
    s=fixture_scene(tmp_path);store=SceneStore(s)
    jobs=CalibrationJobs(experiments=tmp_path/'experiments')
    server=viser.ViserServer(host='127.0.0.1',port=0,verbose=False)
    try:
        panel=CalibrationPanel(server,store,jobs=jobs)
        panel.algorithm.value=next(n for n,k in ALGORITHMS.items() if k==USER_PIPELINE)
        panel.events.put('algorithm');panel.tick()
        assert panel.intrinsic_board.visible and not panel.pipeline_config.visible
        panel.intrinsic_board.value='board_small';panel.extrinsic_board.value='board_large'
        panel.events.put('regions');panel.tick()
        panel.region_controls['front'].value='board_large'
        panel.events.put('save_settings');panel.tick()
        assert store.snapshot()[1].calibration['camera_regions']=={'front':'board_large'}
        assert store.snapshot()[1].calibration['input_mode']=='scene'
        path=tmp_path/'saved.yaml';save_scene(store.snapshot()[1],path)
        reloaded=load_scene(path);assert reloaded.calibration['intrinsic_board']=='board_small'
    finally:jobs.close();server.stop()


def test_failed_capture_and_cancel_keep_actual_artifacts(tmp_path):
    s=fixture_scene(tmp_path);jobs=CalibrationJobs(experiments=tmp_path/'experiments')
    r=request();r['views']=4;r['holdout']=1;r['max_attempts']=1;r['sampling']='around_pose'
    s.boards['board_small'].pose[0]=100.
    try:
        out=jobs.start(SceneStore(s),r);state=wait_job(jobs,60)
        assert state['result']['status']=='failed'
        assert '仅采到' in state['result']['error']
        assert (out/'capture/board_poses.csv').exists()
        assert json.loads((out/'capture/manifest.json').read_text())['complete'] is False
        attempts=json.loads((out/'capture/sampling_attempts.json').read_text())
        assert attempts and all(x['reason']=='no_feasible_board_pose' for x in attempts)
        assert attempts[0]['prefilter_rejections']['outside_camera_domain']>0
    finally:jobs.close()


def test_gt_rotation_comparison_math_independent_of_fitting(tmp_path):
    from scene_editor.scene_pipeline import evaluate
    s=fixture_scene(tmp_path);s.vehicle.pose=[.4,-.2,.5,.15,-.1,.6]
    root=tmp_path/'job/pipeline'
    from scene_manager.transforms import inverse
    truth=inverse(evaluation_frame(s))@s.camera_optical_pose('front')
    tcw=inverse(truth)
    ip=root/'intrinsics/pinhole/front';ip.mkdir(parents=True)
    intr={'status':'DIAGNOSTIC','camera_id':'front','camera_model':'pinhole','solutions':[
        {'seed':0,'K':s.cameras['front'].matrix.tolist(),'D':[0.]*5}]}
    (ip/'calibration.json').write_text(json.dumps(intr))
    ep=root/'extrinsics/pinhole/front';ep.mkdir(parents=True)
    # A physical, nondegenerate synthetic comparison independent of PnP.
    row={'R_w2c':tcw[:3,:3].tolist(),'tvec':tcw[:3,3].tolist(),'intrinsic_seed':0,
         'all_correspondence_rmse_px':0.}
    (ep/'extrinsics.json').write_text(json.dumps({'status':'DIAGNOSTIC','candidates':[row]}))
    result=evaluate(s,request(),tmp_path/'job')[0]
    assert result['extrinsic_candidates'][0]['translation_error_m']<1e-12
    assert result['extrinsic_candidates'][0]['rotation_error_deg']<1e-5


@pytest.mark.parametrize('model',['pinhole','fisheye'])
def test_direct_lens_projection_matches_original_solver_convention(tmp_path,model):
    pipeline_class(project_root())
    from calibration_pipeline.calibration import pixels
    from scene_manager.camera_models import project
    s=fixture_scene(tmp_path);c=s.cameras['front'];c.model=model
    c.K=[201.,0,183.2,0,194.,161.7,0,0,1]
    c.D=[.02,-.003,.0003,-.0002,.0001] if model=='pinhole' else [.02,-.003,.0003,-.0002]
    points=np.array([[.2,-.3,1.],[-.4,.25,2.],[0.,0.,1.]])
    uv,valid=project(c,points)
    assert valid.all()
    np.testing.assert_allclose(uv,pixels(points,c.matrix,np.array(c.D),model),atol=1e-10)


def test_saved_profile_types_rejected_atomically(tmp_path):
    store=SceneStore(fixture_scene(tmp_path));before=store.snapshot()
    for data in ({'camera_regions':[]},{'intrinsic_board':3},{'views':2.5},{'spread_x':float('nan')}):
        with pytest.raises(ValueError):store.command({'action':'calibration_settings','settings':data})
        assert store.snapshot()[0]==before[0]
        assert store.snapshot()[1].to_dict()==before[1].to_dict()


def gaussian_checker(path,columns,rows,pitch):
    """Known-metric test PLY, not the user's unverified ring asset."""
    from plyfile import PlyData,PlyElement
    # Dense opaque DC-only splats with a known white margin for detection.
    step=pitch/12
    x=np.arange(-columns*pitch/2-.06,columns*pitch/2+.06,step)+step/2
    y=np.arange(-rows*pitch/2-.06,rows*pitch/2+.06,step)+step/2
    xx,yy=np.meshgrid(x,y);means=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]
    col=np.floor((means[:,0]+columns*pitch/2)/pitch).astype(int)
    row=np.floor((means[:,1]+rows*pitch/2)/pitch).astype(int)
    inside=(col>=0)&(col<columns)&(row>=0)&(row<rows)
    gray=np.where(inside&((col+row)%2==1),12.,240.)/255
    names=['x','y','z','opacity',*[f'f_dc_{i}' for i in range(3)],
        *[f'scale_{i}' for i in range(3)],*[f'rot_{i}' for i in range(4)]]
    data=np.zeros(len(means),dtype=[(n,'<f4') for n in names])
    for i,n in enumerate(('x','y','z')):
        data[n]=means[:,i];data[f'f_dc_{i}']=(gray-.5)/.28209479177387814
        data[f'scale_{i}']=np.log(step*.55 if i<2 else .001)
    data['opacity']=10.;data['rot_0']=1.
    PlyData([PlyElement.describe(data,'vertex')],text=False).write(str(path))


@pytest.mark.skipif(__import__('os').environ.get('AVM_TEST_GPU')!='1',reason='CUDA opt-in')
@pytest.mark.parametrize('model',['pinhole','fisheye'])
def test_actual_gaussian_board_gpu_capture_to_original_pipeline(tmp_path,model):
    s=fixture_scene(tmp_path);s.vehicle.pose=[0.]*6
    # Both Gaussian targets must be above the Gaussian room floor; the moving
    # target stays in front of the retained fixed board during intrinsic capture.
    s.boards['board_small'].pose[2]=.3
    s.boards['board_large'].pose[2]=.1
    c=s.cameras['front'];c.model=model
    c.D=[0.]*5 if model=='pinhole' else [0.]*4
    if model=='fisheye':c.K=[240.,0,239.5,0,240.,179.5,0,0,1]
    for name,b in s.boards.items():
        p=tmp_path/f'{name}.ply';gaussian_checker(p,b.columns,b.rows,b.square_size);b.asset=str(p)
        b.calibration_geometry=grid_geometry(name,b.columns-1,b.rows-1,b.square_size,
            [-b.width/2+b.square_size,-b.height/2+b.square_size,0],source='generated fixture exact metric definition')
    r=request();r['backend']='gsplat';r['max_attempts']=8
    jobs=CalibrationJobs(experiments=tmp_path/'experiments')
    try:
        out=jobs.start(SceneStore(s),r);state=wait_job(jobs,240)
        assert state['exit_code']==0,(out/'logs/calibration.log').read_text()
        assert (out/f'pipeline/bev/{model}/comparison.png').exists()
        assert state['result']['calibration_accepted'] is False
        assert len(state['result']['ground_truth_comparison'][0]['extrinsic_candidates'])>0
    finally:jobs.close()
