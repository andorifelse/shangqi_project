"""Reconstructed boards reuse the logical board's poses and geometry contracts."""
from copy import deepcopy
from types import SimpleNamespace
import asyncio
import numpy as np
import pytest
import yaml
from avm_bringup.assets import generate_assets
from scene_manager.models import default_scene
from scene_manager.store import SceneStore
from scene_manager.persistence import save_scene, load_scene
from scene_manager.ground_truth import GroundTruth
from scene_manager.transforms import pose_matrix, wxyz_from_matrix
from gs_renderer.gaussian import GaussianScene
from gs_renderer.pipeline import SensorPipeline
from gs_renderer.board import overlay_boards
from gs_renderer.types import RenderResult


def board_scene(tmp_path):
    generate_assets(tmp_path)
    s=default_scene(tmp_path)
    b=s.boards['board_001']
    b.asset=str(tmp_path/'test_room.ply')
    b.asset_scale=.1
    b.asset_pose=[.1,.2,.3,.1,-.2,.3]
    b.pose=[2.,.1,.6,.2,.1,-.3]
    for c in s.cameras.values():
        c.width,c.height=32,24
        c.K=[14.,0,15.5,0,14.,11.5,0,0,1]
    return s


def test_board_asset_roundtrip_commands_tf_and_legacy(tmp_path):
    s=board_scene(tmp_path)
    store=SceneStore(s)
    store.command({'action':'pose','id':'board_001','pose':[1,2,3,.1,.2,.3]})
    store.command({'action':'properties','id':'board_001','properties':{'asset_scale':.3}})
    s=store.snapshot()[1]
    p=tmp_path/'nested'/'scene.yaml';save_scene(s,p)
    assert load_scene(p).to_dict()==s.to_dict()
    assert yaml.safe_load(p.read_text())['boards']['board_001']['asset']=='../test_room.ply'
    gt=tmp_path/'nested'/'truth.yaml';store.ground_truth().save(gt)
    assert GroundTruth.load(gt).scene.to_dict()==s.to_dict()
    from scene_manager.frames import scene_frames
    frame=next(f for f in scene_frames(s) if f.child=='board_001')
    np.testing.assert_allclose(frame.transform,pose_matrix(s.boards['board_001'].pose))
    old=s.to_dict()
    for key in ('asset','asset_pose','asset_scale'): old['boards']['board_001'].pop(key)
    assert not type(s).from_dict(old).boards['board_001'].is_gaussian
    before=store.snapshot()[0]
    for props in ({'asset_scale':0},{'asset':'board.glb'},{'asset_pose':[0,0]}):
        with pytest.raises(ValueError):
            store.command({'action':'properties','id':'board_001','properties':props})
        assert store.snapshot()[0]==before


def test_board_pose_composes_covariances_reuses_ply_and_deletes(tmp_path,monkeypatch):
    s=board_scene(tmp_path);b=s.boards['board_001']
    raw=GaussianScene.load(b.asset)
    pipeline=SensorPipeline('cpu')
    first=pipeline.render(s)
    count=len(raw.means)
    assert len(pipeline.renderer.gs.means)==2*count
    local=raw.transformed(b.asset_scale,pose_matrix(b.asset_pose))
    expected=local.transformed(transform=pose_matrix(b.pose))
    np.testing.assert_allclose(pipeline.renderer.gs.means[-count:],expected.means,atol=2e-6)
    np.testing.assert_allclose(pipeline.renderer.gs.covariances[-count:],expected.covariances,atol=1e-8)
    def unexpected_load(*a,**k): raise AssertionError('Pose-only edit reloaded PLY')
    monkeypatch.setattr(GaussianScene,'load',unexpected_load)
    b.pose=[2.,-1.,1.,.4,.2,1.]
    second=pipeline.render(s)
    assert np.any(first['front'].rgb!=second['front'].rgb)
    expected=local.transformed(transform=pose_matrix(b.pose))
    np.testing.assert_allclose(pipeline.renderer.gs.means[-count:],expected.means,atol=2e-6)
    s.boards={};pipeline.render(s)
    assert len(pipeline.renderer.gs.means)==count
    assert not pipeline._board_local


def test_gaussian_boards_are_not_replaced_by_virtual_overlay(tmp_path):
    s=board_scene(tmp_path);c=s.cameras['front']
    shape=(c.height,c.width)
    output=RenderResult(np.zeros((*shape,3),np.uint8),np.full(shape,np.inf),np.zeros(shape),np.ones(shape,bool))
    overlay_boards(s,'front',output)
    assert not output.rgb.any()
    s.boards['board_001'].asset=''
    s.boards['board_001'].pose=[3.,0.,.025,0.,0.,0.]
    overlay_boards(s,'front',output)
    assert output.rgb.any()


def test_gaussian_board_calibration_renderer_uses_ply(tmp_path):
    from gs_renderer.calibration_renderer import render_board_image
    s=board_scene(tmp_path)
    image=render_board_image(s,'front',1)
    reference=deepcopy(s);reference.gaussian_file='';reference.vehicle.asset=''
    reference.cameras={'front':reference.cameras['front']}
    expected=SensorPipeline('cpu').render(reference)['front'].rgb
    np.testing.assert_array_equal(image,expected)


def test_board_editor_import_drag_scale_switch_and_delete(tmp_path):
    import viser
    from scene_editor.editor import Editor
    s=board_scene(tmp_path);s.boards['board_001'].asset=''
    store=SceneStore(s)
    server=viser.ViserServer(host='127.0.0.1',port=0,verbose=False)
    try:
        editor=Editor(server,store)
        editor._handle('select','board_001')
        editor.board_asset.value=str(tmp_path/'test_room.ply')
        editor.board_asset_scale.value=.2
        editor._handle('board_asset',None);editor.tick()
        assert store.snapshot()[1].boards['board_001'].is_gaussian
        frame=editor.objects.frames['board_001']
        assert frame.name=='/objects/board_001'
        desired=[1.5,-.4,.7,.3,-.2,.4];t=pose_matrix(desired)
        editor.gizmo.position=t[:3,3]+editor.gizmo_offset
        editor.gizmo.wxyz=wxyz_from_matrix(t)
        asyncio.run(editor._drag(SimpleNamespace(client=True,target=editor.gizmo,phase='end')))
        editor.tick()
        np.testing.assert_allclose(store.snapshot()[1].boards['board_001'].pose,desired,atol=1e-10)
        assert editor.objects.frames['board_001'] is frame  # Pose edits do not reload PLY.
        editor._handle('board_virtual',None);editor.tick()
        assert not store.snapshot()[1].boards['board_001'].asset
        assert editor.board_asset.value==''
        # Authoritative ROS echoes can arrive after the Apply handler returns.
        store.command({'action':'properties','id':'board_001','properties':{
            'asset':str(tmp_path/'test_room.ply'),'asset_scale':.4}})
        editor.tick()
        assert editor.board_asset.value==str(tmp_path/'test_room.ply')
        assert editor.board_asset_scale.value==.4
        editor._handle('delete',None);editor.tick()
        assert not store.snapshot()[1].boards
    finally:
        server.stop()


def test_multiple_gaussian_boards_share_asset_but_move_independently(tmp_path):
    s=board_scene(tmp_path)
    other=deepcopy(s.boards['board_001']);other.id='board_002';other.pose[1]+=1
    s.boards[other.id]=other
    p=SensorPipeline();p.render(s)
    assert len(p._board_local)==1
    n=len(p._environment.means)
    assert len(p.renderer.gs.means)==3*n
    first=p.renderer.gs.means[n:2*n].copy()
    other.pose[0]+=1;p.render(s)
    np.testing.assert_array_equal(p.renderer.gs.means[n:2*n],first)
    np.testing.assert_allclose(p.renderer.gs.means[2*n:]-first,np.tile([1.,1.,0.],(n,1)),atol=1e-6)
