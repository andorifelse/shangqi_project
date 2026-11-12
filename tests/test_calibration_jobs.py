"""Exercise actual subprocess jobs and the real Viser calibration controls."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import time
import pytest
from avm_bringup.assets import generate_assets
from scene_manager.models import default_scene
from scene_manager.store import SceneStore
from scene_editor.calibration_jobs import (CalibrationJobs,BASELINE,USER_PIPELINE,
    normalize_request,project_root,read_json,pipeline_class)


def fixture_store(tmp_path):
    assets=tmp_path/'assets';generate_assets(assets)
    return SceneStore(default_scene(assets))


def wait_job(jobs,timeout=120):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        state=jobs.poll()
        if state and not state['running']:return state
        time.sleep(.05)
    jobs.cancel()
    pytest.fail(f'Job timed out: {jobs.directory}')


def test_gui_baseline_runs_with_immutable_snapshot_and_no_parameter_apply(tmp_path,monkeypatch):
    store=fixture_store(tmp_path)
    jobs=CalibrationJobs(experiments=tmp_path/'experiments')
    before=store.snapshot()[1].to_dict()
    import avm_bringup.experiments as experiment_module
    def forbidden_ui_archive(*args,**kwargs):
        raise AssertionError('Expensive experiment archive executed on UI thread')
    monkeypatch.setattr(experiment_module,'runtime_manifest',forbidden_ui_archive)
    try:
        output=jobs.start(store,{'algorithm':BASELINE,'cameras':['left'],'board_id':'board_001',
            'views':24,'seed':20261020,'supersample':2,'backend':'cpu'})
        with pytest.raises(ValueError,match='already running'):jobs.start(store,{})
        # A concurrent scene edit is deliberately NOT part of this job.
        store.command({'action':'pose','id':'left','pose':[0,1,1,0,0,0]})
        state=wait_job(jobs)
        assert state['exit_code']==0,(output/'logs/calibration.log').read_text()
        r=state['result'];assert r['status']=='complete'
        assert r['scene_revision']==0 and r['scene_parameters_applied'] is False
        assert r['results']['left']['reprojection_rms_px']<.5
        from scene_manager.ground_truth import GroundTruth
        assert GroundTruth.load(output/'ground_truth.yaml').scene.to_dict()==before
        assert store.snapshot()[1].cameras['left'].pose==[0,1,1,0,0,0]
        assert (output/'metrics.csv').is_file()
        assert r['summary'][0]['kind']=='baseline'
    finally:jobs.close()


def test_supplied_pipeline_config_runs_original_code_and_keeps_diagnostics(tmp_path):
    store=fixture_store(tmp_path)
    jobs=CalibrationJobs(experiments=tmp_path/'experiments')
    root=project_root();config=root/'calibration_pipeline_1.0.0/examples/cpu_demo/demo.json'
    original=config.read_bytes()
    try:
        output=jobs.start(store,{'algorithm':USER_PIPELINE,'pipeline_config':str(config),'stage':'all'})
        state=wait_job(jobs,240)
        assert state['exit_code']==0,(output/'logs/calibration.log').read_text()
        r=state['result'];assert r['status']=='complete'
        assert r['input_scope']=='pipeline_config' and not r['calibration_accepted']
        assert config.read_bytes()==original
        assert (output/'pipeline/bev/pinhole/comparison.png').is_file()
        assert (output/'pipeline/bev/fisheye/comparison.png').is_file()
        assert (output/'logs/imported_pipeline_source.zip').is_file()
        rows=r['summary'];assert len(rows)==6
        pinhole=next(x for x in rows if x['kind']=='intrinsics' and x['model']=='pinhole')
        assert pinhole['status']=='DEGENERATE_PARALLEL_PLANE_ACQUISITION'
        assert pinhole['K'] is not None
        assert next(x for x in rows if x['kind']=='extrinsics')['candidate_count']>0
        assert all(x['calibration_accepted'] is False for x in rows)
        assert pipeline_class(root).__module__=='calibration_pipeline.pipeline'
    finally:jobs.close()


def test_invalid_inputs_do_not_create_experiments(tmp_path):
    store=fixture_store(tmp_path);jobs=CalibrationJobs(experiments=tmp_path/'experiments')
    for payload in ({'algorithm':'unknown'},{'experiment_id':'../oops'},
                    {'cameras':['missing']},{'views':7},
                    {'algorithm':USER_PIPELINE,'pipeline_config':'/no/config.json'}):
        with pytest.raises(ValueError):jobs.start(store,payload)
    assert not jobs.experiments.exists()
    s=store.snapshot()[1];s.boards['board_001'].asset=s.gaussian_file
    with pytest.raises(ValueError,match='Gaussian'):
        normalize_request({'algorithm':BASELINE},s,project_root())
    pending=SimpleNamespace(_inflight=True)
    with pytest.raises(ValueError,match='acknowledged'):jobs.start(pending,{})


def test_failure_records_and_cancel_are_not_marked_complete(tmp_path):
    store=fixture_store(tmp_path);s=store.snapshot()[1]
    # Valid schema but pattern too small to be detected at this resolution.
    for c in s.cameras.values():c.width=c.height=16;c.K=[5.,0,7.5,0,5.,7.5,0,0,1]
    store.replace(s);jobs=CalibrationJobs(experiments=tmp_path/'experiments')
    try:
        out=jobs.start(store,{'views':8,'cameras':['left'],'supersample':1,'experiment_id':'failed'})
        state=wait_job(jobs)
        assert state['result']['status']=='failed' and state['exit_code']!=0
        assert state['result']['error'] and (out/'logs/calibration.log').is_file()
        out=jobs.start(fixture_store(tmp_path),{'experiment_id':'cancelled'})
        jobs.cancel()
        assert jobs.poll()['result']['status']=='cancelled'
        assert not jobs.running
        with pytest.raises(ValueError,match='exists'):jobs.start(store,{'experiment_id':'cancelled'})
    finally:jobs.close()


def test_viser_dropdown_button_and_result_widgets(tmp_path):
    import viser
    from scene_editor.calibration_ui import CalibrationPanel,ALGORITHMS
    store=fixture_store(tmp_path);jobs=CalibrationJobs(experiments=tmp_path/'experiments')
    server=viser.ViserServer(host='127.0.0.1',port=0,verbose=False)
    try:
        panel=CalibrationPanel(server,store,jobs=jobs)
        assert len(panel.algorithm.options)==3
        assert panel.board.value=='board_001'
        panel.camera.value='left';panel.views.value=24
        # Same queued action as the browser's Start button.
        panel.events.put('start');panel.tick()
        assert jobs.running and panel.start_button.disabled
        deadline=time.monotonic()+120
        while jobs.running and time.monotonic()<deadline:
            panel.tick();time.sleep(.05)
        panel._last_poll=0;panel.tick()
        assert not jobs.running and not panel.start_button.disabled
        assert 'ESTIMATED' in panel.results.content and 'K=' in panel.results.content
        assert not panel.download_button.disabled
        downloads=[]
        client=SimpleNamespace(send_file_download=lambda name,content:downloads.append((name,content)))
        panel._download(SimpleNamespace(client=client))
        assert json.loads(downloads[0][1])['status']=='complete'
        panel.algorithm.value=next(n for n,v in ALGORITHMS.items() if v==USER_PIPELINE)
        panel.events.put('algorithm');panel.tick()
        assert not panel.pipeline_config.visible and panel.views.visible
        assert panel.input_mode.visible and not panel.pipeline_config.visible
    finally:jobs.close();server.stop()
