from copy import deepcopy
import os
import numpy as np
import pytest
from avm_bringup.assets import generate_assets
from scene_manager.models import Camera,Board,default_scene
from scene_manager.transforms import pose_matrix
from gs_renderer.gaussian import GaussianScene


def test_ray_cache_is_pose_independent_but_tracks_optical_changes(monkeypatch):
    import scene_manager.camera_models as models
    models._camera_rays.cache_clear()
    camera=Camera(width=64,height=48,K=[30.,0,31.5,0,30.,23.5,0,0,1])
    real=models.unproject;calls=[]
    def inverse(*args,**kwargs):
        calls.append(1);return real(*args,**kwargs)
    monkeypatch.setattr(models,'unproject',inverse)
    rays,valid=models.camera_rays(camera);count=len(calls)
    camera.pose=[1,2,3,.1,.2,.3]
    again,again_valid=models.camera_rays(camera)
    assert again is rays and again_valid is valid and len(calls)==count
    assert not rays.flags.writeable and not valid.flags.writeable
    for field,value in [('K',[32.,0,31.5,0,30.,23.5,0,0,1]),('D',[.02,0,0,0]),('max_angle',1.4)]:
        setattr(camera,field,value)
        changed,_=models.camera_rays(camera)
        assert changed is not again
        uv,ok=models.project(camera,changed[::8,::8])
        u,v=np.meshgrid(np.arange(0,64,8),np.arange(0,48,8))
        np.testing.assert_allclose(uv[ok],np.stack([u,v],axis=-1)[ok],atol=1e-3)
        again=changed


def test_gpu_slice_updates_equal_full_recomposition_on_cpu(tmp_path):
    torch=pytest.importorskip('torch')
    from gs_renderer.gsplat_renderer import GsplatRenderer
    generate_assets(tmp_path);static=GaussianScene.load(tmp_path/'test_room.ply')
    moving=static.transformed(.1,pose_matrix([0,0,.3,0,0,0]))
    renderer=object.__new__(GsplatRenderer)
    renderer.torch=torch;renderer.tensor=lambda a:torch.tensor(a,dtype=torch.float32)
    renderer._parts={};renderer._layout=None;renderer.fallback=None
    renderer.set_parts({'static':static,'moving':moving})
    buffer=renderer.means
    changed=moving.transformed(transform=pose_matrix([.3,0,0,0,0,.4]))
    renderer.set_parts({'static':static,'moving':changed})
    assert renderer.means is buffer
    assert renderer.upload_stats=={'layout_rebuilt':False,'uploaded_gaussians':len(changed.means)}
    expected=GaussianScene.concatenate(static,changed)
    for field in ('means','quats','scales','opacities','colors'):
        np.testing.assert_allclose(getattr(renderer,field).numpy(),getattr(expected,field),atol=1e-7)
    renderer.set_parts({'static':static,'moving':changed})
    assert renderer.upload_stats['uploaded_gaussians']==0
    renderer.set_parts({'static':static})
    assert renderer.upload_stats['layout_rebuilt']
    np.testing.assert_array_equal(renderer.means.numpy(),static.means)


def test_sampler_filters_tiny_or_occluded_targets_and_avoids_failed_neighbours(tmp_path):
    from test_scene_pipeline import fixture_scene,request
    from scene_editor.scene_pipeline import normalize_scene_request
    from scene_editor.capture_sampling import IntrinsicSampler
    s=fixture_scene(tmp_path);r=normalize_scene_request(s,request())
    b=s.boards['board_small'];camera=s.cameras['front']
    blocked=np.full((camera.height,camera.width),.01,np.float32)
    sampler=IntrinsicSampler(s,'front',b,r,np.random.default_rng(1),blocked)
    candidate,reason=sampler.evaluate(b.pose)
    assert candidate is None and reason=='occluded_by_static_scene'
    sampler.depth=None
    candidate,reason=sampler.evaluate(b.pose);assert candidate is not None
    sampler.observe(candidate,False)
    assert sampler.evaluate(b.pose)[1]=='near_failed_observation'
    sampler.failed=[]
    far=list(b.pose);far[2]=-100.
    assert sampler.evaluate(far)[1]=='projected_squares_too_small'


@pytest.mark.skipif(os.environ.get('AVM_TEST_GPU')!='1',reason='CUDA opt-in')
def test_incremental_gpu_render_matches_fresh_scene_after_board_move_and_removal(tmp_path):
    from gs_renderer.pipeline import SensorPipeline
    generate_assets(tmp_path);s=default_scene(tmp_path)
    s.cameras={'front':s.cameras['front']};c=s.cameras['front']
    c.width,c.height=64,48;c.K=[30.,0,31.5,0,30.,23.5,0,0,1]
    s.boards['board_extra']=Board(id='board_extra',asset=s.gaussian_file,asset_scale=.1)
    pipeline=SensorPipeline('gsplat');pipeline.render(s)
    renderer=pipeline.renderer
    for action in ('move','remove'):
        if action=='move':s.boards['board_extra'].pose[1]+=.2
        else:del s.boards['board_extra']
        incremental=pipeline.render(s)['front']
        assert pipeline.renderer is renderer
        fresh=SensorPipeline('gsplat').render(deepcopy(s))['front']
        np.testing.assert_allclose(incremental.rgb,fresh.rgb,atol=1)
        np.testing.assert_allclose(incremental.depth,fresh.depth,rtol=1e-5,atol=1e-5)


def test_locked_plane_impossible_camera_fails_before_any_scene_render(tmp_path,monkeypatch):
    from test_scene_pipeline import fixture_scene,request
    from scene_editor.scene_pipeline import normalize_scene_request,capture_scene
    from gs_renderer.pipeline import SensorPipeline
    s=fixture_scene(tmp_path);camera=deepcopy(s.cameras['front']);camera.name='right'
    camera.pose[4]=-np.pi/2;s.cameras['right']=camera
    r=request();r.update(cameras=['front','right'],camera_regions={'front':'board_large','right':'board_large'},sampling='around_pose')
    r=normalize_scene_request(s,r)
    def forbidden(*args,**kwargs):raise AssertionError('must fail before GPU/scene load')
    monkeypatch.setattr(SensorPipeline,'render',forbidden)
    with pytest.raises(RuntimeError,match='right.*几何预检'):
        capture_scene(s,r,tmp_path,lambda *args,**kwargs:None)


def test_sampler_public_corner_order_does_not_change_geometry(tmp_path):
    from test_scene_pipeline import fixture_scene,request
    from scene_editor.scene_pipeline import normalize_scene_request
    from scene_editor.capture_sampling import IntrinsicSampler
    s=fixture_scene(tmp_path);r=normalize_scene_request(s,request());b=s.boards['board_small']
    a=IntrinsicSampler(s,'front',b,r,np.random.default_rng(1))
    shuffled=deepcopy(r);shuffled['intrinsic_geometry']['corners'].reverse()
    other=IntrinsicSampler(s,'front',b,shuffled,np.random.default_rng(1))
    np.testing.assert_array_equal(a.grid,other.grid)
    assert a.evaluate(b.pose)[0]['min_square_px']==other.evaluate(b.pose)[0]['min_square_px']


def test_auto_plane_repositions_target_for_other_camera_without_mutating_scene(tmp_path):
    from test_scene_pipeline import fixture_scene,request
    from scene_editor.scene_pipeline import normalize_scene_request
    from scene_editor.capture_sampling import plan_intrinsic_board
    s=fixture_scene(tmp_path);camera=deepcopy(s.cameras['front']);camera.name='left'
    camera.pose=[0,0,2,0,-.4,np.pi/2];s.cameras['left']=camera
    r=request();r.update(cameras=['left'],camera_regions={'left':'board_large'},sampling='visible_plane')
    r=normalize_scene_request(s,r);before=s.to_dict();board=s.boards['board_small']
    planned,source,candidate,_=plan_intrinsic_board(s,'left',board,r,12)
    assert source=='camera_facing_plane' and candidate is not None
    assert candidate['min_square_px']>=16.
    np.testing.assert_allclose(pose_matrix(planned.pose)[:3,2],-s.camera_optical_pose('left')[:3,2],atol=1e-12)
    assert s.to_dict()==before
    r['sampling']='around_pose'
    locked,source,candidate,_=plan_intrinsic_board(s,'left',board,r,12)
    assert source=='scene_plane' and locked.pose==board.pose and candidate is None
