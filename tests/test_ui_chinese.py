"""Chinese labels are presentation only; saved models and commands remain stable."""
from pathlib import Path
from types import SimpleNamespace
import viser
import pytest
from avm_bringup.assets import generate_assets
from scene_manager.models import default_scene
from scene_manager.store import SceneStore
from scene_editor.editor import Editor
from scene_editor.calibration_ui import CalibrationPanel,ALGORITHMS
from scene_editor.calibration_jobs import CalibrationJobs,USER_PIPELINE
from scene_editor.geometry_ui import BoardGeometryPanel
from scene_editor.ui_text import display_text,object_label,object_id,BACKEND_LABELS


@pytest.fixture
def ui(tmp_path):
    generate_assets(tmp_path/'assets')
    store=SceneStore(default_scene(tmp_path/'assets'))
    server=viser.ViserServer(host='127.0.0.1',port=0,verbose=False)
    try:yield server,store,tmp_path
    finally:server.stop()


def test_chinese_editor_preserves_model_commands_and_transform_modes(ui):
    server,store,_=ui
    editor=Editor(server,store);editor._handle('select','front')
    assert editor.cam_model.value=='鱼眼'
    assert editor.tree.value=='前相机（front）'
    editor.cam_model.value='针孔';editor.distortion.value='0,0,0,0,0'
    editor._handle('camera',None);editor.tick()
    assert store.snapshot()[1].cameras['front'].model=='pinhole'
    editor.mode.value='旋转';editor._handle('select','front')
    assert editor.gizmo.disable_axes and not editor.gizmo.disable_rotations
    editor.mode.value='平移';editor._handle('select','front')
    assert not editor.gizmo.disable_axes and editor.gizmo.disable_rotations
    assert object_id(object_label('board_001'))=='board_001'


def test_chinese_backend_stages_and_settings_roundtrip(ui):
    server,store,tmp=ui
    jobs=CalibrationJobs(experiments=tmp/'experiments')
    panel=CalibrationPanel(server,store,backend='gsplat',jobs=jobs)
    try:
        assert panel.backend.value==BACKEND_LABELS['gsplat']
        panel.camera.value='左相机（left）'
        assert panel.settings()['cameras']==['left']
        assert panel.settings()['backend']=='gsplat'
        panel.algorithm.value=next(n for n,v in ALGORITHMS.items() if v==USER_PIPELINE)
        panel.stage.value='内参，外参,鸟瞰'
        assert panel.settings()['stage']=='intrinsics,extrinsics,bev'
        panel.stage.value='all';assert panel.settings()['stage']=='all'
        panel._load_saved(SimpleNamespace(calibration={'backend':'cpu'},boards={}))
        assert panel.backend.value==BACKEND_LABELS['cpu']
        assert display_text('DEGENERATE_PARALLEL_PLANE_ACQUISITION')=='固定平面退化（参数不可唯一确定）'
    finally:panel.close()


def test_chinese_geometry_coordinate_system_is_not_serialized(ui):
    server,store,_=ui
    panel=BoardGeometryPanel(server,store)
    panel.frame.value='模型原始坐标';panel.source.value='测试模型已知格长';panel.verified.value=True
    panel.events.put('grid');panel.tick()
    board=store.snapshot()[1].boards['board_001']
    assert board.calibration_geometry['frame']=='asset'
    assert '已保存' in panel.status.content


def test_result_statuses_are_chinese_without_changing_raw_result(ui):
    server,store,tmp=ui
    from scene_editor.calibration_jobs import MULTITILT
    panel=CalibrationPanel(server,store,jobs=CalibrationJobs(experiments=tmp/'experiments'))
    result={'algorithm':MULTITILT,'status':'rejected','intrinsics_accepted':False,
        'results':{'left':{'status':'REJECTED','K':[[1,0,0],[0,1,0],[0,0,1]],'D':[0]*4,
                         'rms_px':2.,'heldout_max_frame_rmse_px':4.,'rejection_reasons':['训练误差过大']}},
        'bev':{'status':'SKIPPED','reason':'内参未通过，未生成鸟瞰图'}}
    try:
        panel._show_result({'directory':str(tmp),'result':result})
        assert '左相机' in panel.results.content and '未通过' in panel.results.content
        assert 'REJECTED' not in panel.results.content and 'SKIPPED' not in panel.results.content
        assert result['results']['left']['status']=='REJECTED'
        result['status']='cancelled'
        panel._show_result({'directory':str(tmp),'result':result})
        assert '已取消' in panel.status.content
    finally:panel.close()
