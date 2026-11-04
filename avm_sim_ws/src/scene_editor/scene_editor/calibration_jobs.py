"""Small subprocess jobs for both existing calibration implementations.

The UI thread never runs detection, fitting, or CUDA work. Each job owns an
immutable scene snapshot and a new experiment folder. Imported pipeline inputs
remain relative to their ORIGINAL configuration directory.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime
import csv
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

from scene_manager.ground_truth import GroundTruth

BASELINE = "opencv_baseline"
USER_PIPELINE = "pipeline_1_0_0"


def project_root():
    configured = os.environ.get("AVM_PROJECT_ROOT")
    candidates = [Path(configured)] if configured else list(Path(__file__).resolve().parents)
    for root in candidates:
        if (root / "tools/run_calibration.py").is_file():
            return root.resolve()
    raise ValueError("Cannot locate project root; set AVM_PROJECT_ROOT")


def pipeline_class(root):
    source = Path(root) / "calibration_pipeline_1.0.0"
    if not (source / "calibration_pipeline/pipeline.py").is_file():
        raise ValueError(f"User pipeline source not found: {source}")
    # No pip install, environment replacement, or modification of supplied code.
    sys.path.insert(0, str(source)) if str(source) not in sys.path else None
    from calibration_pipeline import Pipeline, __version__
    import calibration_pipeline
    if __version__ != "1.0.0" or source.resolve() not in Path(calibration_pipeline.__file__).resolve().parents:
        raise ValueError("Imported pipeline is not the supplied local version 1.0.0")
    return Pipeline


def normalize_request(request, scene, root):
    request = deepcopy(request)
    algorithm = request.get("algorithm", BASELINE)
    if algorithm not in (BASELINE, USER_PIPELINE):
        raise ValueError(f"Unknown calibration algorithm: {algorithm}")
    if algorithm == BASELINE:
        if not scene.cameras:
            raise ValueError("Current scene has no cameras")
        names = request.get("cameras") or sorted(scene.cameras)
        if not isinstance(names,list) or not names or set(names) - scene.cameras.keys():
            raise ValueError("Select cameras present in the current scene")
        board_id = request.get("board_id") or next(iter(scene.boards), "")
        if board_id not in scene.boards:
            raise ValueError("Select a checkerboard from the current scene")
        board = scene.boards[board_id]
        if board.rows < 4 or board.columns < 4:
            raise ValueError("Baseline requires at least 4x4 squares")
        if any(scene.cameras[n].model not in ("pinhole","fisheye") for n in names):
            raise ValueError("OpenCV baseline supports pinhole/fisheye, not ftheta")
        if board.is_gaussian and not request.get("confirm_board_geometry",False):
            raise ValueError("Gaussian checkerboard needs verified rectangular pattern, dimensions and asset alignment; "
                             "ring strips/center-hole PLYs are not a complete rectangular checkerboard")
        for key,lo,hi,default in (("views",8,500,40),("seed",0,2**32-1,20261020),("supersample",1,4,2)):
            value=request.get(key,default)
            if isinstance(value,bool) or not isinstance(value,int) or not lo<=value<=hi:
                raise ValueError(f"{key} must be an integer in [{lo},{hi}]")
            request[key]=value
        if request.get("backend","cpu") not in ("cpu","gsplat"):
            raise ValueError("Unknown board renderer")
        request.update(cameras=names,board_id=board_id)
        return request
    mode=request.get('input_mode','manual' if request.get('pipeline_config') else 'scene')
    if mode=='scene':
        from .scene_pipeline import normalize_scene_request
        request=normalize_scene_request(scene,request)
        request['input_mode']='scene'
        pipeline_class(root)  # Verify local version before creating an experiment.
        return request
    if mode not in ('manual','demo'):raise ValueError('未知输入模式：请选择当前场景、高级 JSON 或演示')
    if mode=='demo':
        request['pipeline_config']=str(Path(root)/'calibration_pipeline_1.0.0/examples/cpu_demo/demo.json')
    path=Path(request.get("pipeline_config","")).expanduser().resolve()
    if not path.is_file():
        raise ValueError("Select the pipeline JSON configuration file")
    pipeline=pipeline_class(root)(path)
    # The supplied library validates stage ordering and paused status itself.
    stage=request.get("stage","all")
    plan=pipeline.plan(stage)
    if str(pipeline.config.get("execution_status","")).startswith(("paused","draft_pending")):
        raise ValueError("Pipeline configuration is paused/pending; do not bypass its prerequisites")
    sections={"scene" if s=="prepare" else s for s in plan["stages"]}
    missing=[x["path"] for x in plan["inputs"]
             if x["role"].split(".",1)[0] in sections and not x["exists"]]
    if missing:
        raise ValueError("Missing pipeline inputs: " + "; ".join(missing))
    request.update(pipeline_config=str(path),pipeline_base_dir=str(path.parent),
                   pipeline_document=pipeline.config,stage=stage,input_mode=mode)
    return request


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError,ValueError):
        return default


def write_json(path, document):
    # JSON is required here (not YAML); atomic rename avoids partial UI reads.
    path=Path(path)
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(document,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def job_progress(directory, phase, **details):
    write_json(Path(directory)/"progress.json",{"phase":phase,"updated":time.time(),**details})
    print(phase,flush=True)


def initialize_job(directory):
    from avm_bringup.experiments import initialize_experiment_from_snapshot
    from scene_manager.models import Scene
    directory=Path(directory)
    request=read_json(directory/"request.json")
    truth=GroundTruth.capture(Scene.from_dict(request["scene_snapshot"]),request["scene_revision"])
    config={k:v for k,v in request.items() if k!="scene_snapshot"}
    initialize_experiment_from_snapshot(truth,directory,config,project_root())
    job_progress(directory,"实验快照已保存",scene_revision=truth.revision)


def report_rows(directory):
    """Summarize original statuses; never turn diagnostic output into accepted calibration."""
    directory=Path(directory)
    rows=[]
    base=read_json(directory/"baseline_result.json",{})
    for name,result in base.get("results",{}).items():
        ext=result["extrinsics_error"]
        rows.append({"kind":"baseline","camera":name,"model":result["model"],"status":"ESTIMATED",
            "views":result["views_detected"],"rms_px":result["reprojection_rms_px"],
            "translation_rmse_m":ext["translation_rmse_m"],"rotation_rmse_deg":ext["rotation_rmse_deg"]})
    out=directory/"pipeline"
    for path in sorted((out/"intrinsics").glob("*/*/calibration.json")):
        result=read_json(path,{})
        best=next((s for s in result.get("solutions",[]) if s.get("seed")==result.get("best_seed")),{})
        rows.append({"kind":"intrinsics","camera":path.parent.name,"model":path.parent.parent.name,
            "status":result.get("status"),"rms_px":result.get("best_training_rmse_px"),
            "K":best.get("K"),"D":best.get("D"),"best_seed":result.get("best_seed"),
            "calibration_accepted":False,"report":str(path)})
    for path in sorted((out/"extrinsics").glob("*/*/extrinsics.json")):
        result=read_json(path,{})
        candidates=result.get("candidates",[])
        preview=min(candidates,key=lambda c:c["all_correspondence_rmse_px"]) if candidates else {}
        pose={k:preview[k] for k in ("R_w2c","tvec","camera_center_world_m","intrinsic_seed",
             "row_start","quarter_turns","all_correspondence_rmse_px") if k in preview}
        rows.append({"kind":"extrinsics","camera":path.parent.name,"model":path.parent.parent.name,
            "status":result.get("status"),"rms_px":result.get("minimum_conditional_reprojection_rmse_px"),
            "candidate_count":len(candidates),"pose_candidate_preview":pose,
            "candidate_notice":"Preview only: minimum ALL-correspondence RMSE, not a resolved unique physical pose.",
            "calibration_accepted":False,"report":str(path)})
    for path in sorted((out/"bev").glob("*/metrics.json")):
        result=read_json(path,{})
        rows.append({"kind":"bev","camera":"all","model":path.parent.name,"status":result.get("status"),
            "coverage":result.get("coverage_fraction"),"calibration_accepted":False,"report":str(path)})
    comparison=read_json(directory/'ground_truth_comparison.json',{}).get('results',[])
    for row in rows:
        measured=next((r for r in comparison if r['camera']==row['camera'] and r['model']==row['model']),{})
        if row['kind']=='extrinsics' and measured.get('selected_by_observations'):
            estimate=measured['selected_by_observations']
            row['translation_error_m']=estimate['translation_error_m'];row['rotation_error_deg']=estimate['rotation_error_deg']
        if row['kind']=='intrinsics' and measured.get('intrinsic_candidates'):
            candidate=next((c for c in measured['intrinsic_candidates'] if c['seed']==row.get('best_seed')),None)
            if candidate:row['focal_relative_percent']=candidate['focal_relative_percent']
    return rows


def save_metrics(directory, rows):
    fields=["kind","model","camera","status","views","rms_px","translation_rmse_m","rotation_rmse_deg",
            'translation_error_m','rotation_error_deg','focal_relative_percent',"candidate_count","coverage"]
    with (Path(directory)/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)


def run_job(directory):
    from avm_bringup.experiments import finish_experiment
    directory=Path(directory).resolve();root=project_root()
    request=read_json(directory/"request.json")
    result={"status":"failed","algorithm":request["algorithm"],"scene_revision":request["scene_revision"],
            "calibration_accepted":False,"scene_parameters_applied":False}
    try:
        truth=GroundTruth.load(directory/"ground_truth.yaml")
        if request["algorithm"]==BASELINE:
            from gs_renderer.calibration import make_calibration_dataset,calibrate_camera_dataset,save_calibration_report
            scene=truth.scene
            scene.boards={request["board_id"]:scene.boards[request["board_id"]]}
            job_progress(directory,"生成标定数据集",scene_revision=truth.revision)
            dataset=directory/"dataset"
            meta=make_calibration_dataset(scene,dataset,request["views"],request["seed"],
                request["supersample"],request["cameras"],request.get("backend","cpu"))
            dataset_truth=GroundTruth.load(dataset/"ground_truth.yaml")
            reports={}
            for name in meta["cameras"]:
                job_progress(directory,f"OpenCV 标定 {name}")
                reports[name]=calibrate_camera_dataset(dataset,name,dataset_truth)
            save_calibration_report({"results":reports},directory/"baseline_result.json")
            result.update(status="complete",results=reports,input_scope="current_scene_snapshot",
                          scientific_status="SYNTHETIC_BASELINE_ONLY")
        else:
            Pipeline=pipeline_class(root)
            import zipfile
            package=root/"calibration_pipeline_1.0.0"
            with zipfile.ZipFile(directory/"logs/imported_pipeline_source.zip","w",zipfile.ZIP_DEFLATED) as archive:
                for p in sorted((package/"calibration_pipeline").glob("*.py")):
                    archive.write(p,str(p.relative_to(package)))
            if request.get('input_mode')=='scene':
                from .scene_pipeline import run_scene_pipeline
                progress=lambda phase,**kw:job_progress(directory,phase,**kw)
                payload=run_scene_pipeline(truth.scene,request,directory,Pipeline,progress)
                result.update(status='complete',pipeline=payload,input_scope='current_scene_snapshot',
                    scientific_status='DIAGNOSTIC_NOT_ACCEPTED',
                    ground_truth_comparison=payload['ground_truth_comparison'],
                    input_notice='当前场景直出 PNG（不是 Graphdeco 六面采集）；原版 1.0.0 仅接收图片与公开板几何，真值仅采集和事后评估使用。')
            else:
                document=deepcopy(request["pipeline_document"])
                document["output_dir"]=str(directory/"pipeline")
                pipeline=Pipeline(document,base_dir=request["pipeline_base_dir"])
                write_json(directory/"effective_pipeline.json",document)
                write_json(directory/"pipeline_plan.json",pipeline.plan(request["stage"]))
                job_progress(directory,"执行标定流程 1.0.0（配置输入）")
                payload=pipeline.run(request["stage"],execute=True)
                result.update(status="complete",pipeline=payload,input_scope="pipeline_config",
                    scientific_status="DIAGNOSTIC_NOT_ACCEPTED",
                    input_notice="Images/geometry come from the pipeline JSON; saved GUI scene is context only, not solver input.")
        result["summary"]=report_rows(directory)
        save_metrics(directory,result["summary"])
        finish_experiment(directory,result)
        job_progress(directory,"完成",status=result["status"])
        return result
    except Exception as exc:
        comparison=read_json(directory/'ground_truth_comparison.json',{})
        if comparison:result['ground_truth_comparison']=comparison.get('results',[])
        result.update(error=f"{type(exc).__name__}: {exc}",summary=report_rows(directory))
        save_metrics(directory,result["summary"])
        finish_experiment(directory,result)
        job_progress(directory,"失败",error=result["error"])
        raise


class CalibrationJobs:
    def __init__(self, root=None, experiments=None):
        self.root=Path(root or project_root()).resolve()
        self.experiments=Path(experiments or self.root/"experiments").resolve()
        self.process=None;self.directory=None;self.log=None;self.started=None

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, store, request):
        if self.running:
            raise ValueError("A calibration job is already running")
        if getattr(store,"_inflight",False) or getattr(store,"_queue",None) or getattr(store,"_poses",None):
            raise ValueError("请等待 ROS 编辑回传确认（acknowledged）后再开始标定")
        revision,scene=store.snapshot()
        if revision<0:
            raise ValueError("请等待 ROS 场景状态加载完成")
        request=normalize_request(request,scene,self.root)
        name=request.get("experiment_id","").strip() or datetime.now().strftime("calib_gui_%Y%m%d_%H%M%S_")+uuid.uuid4().hex[:6]
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}",name):
            raise ValueError("实验名称请使用英文字母/数字/下划线，不得包含路径分隔符")
        directory=self.experiments/name
        if directory.exists():
            raise ValueError(f"Experiment exists; choose a new ID: {directory}")
        request.update(algorithm=request.get("algorithm",BASELINE),scene_revision=revision)
        truth=GroundTruth.capture(scene,revision)
        request["scene_snapshot"]=truth.scene.to_dict()
        # Only reserve the folder and serialize small metadata on the UI thread.
        # Asset hashes/source archives/PNGs and all calibration run in the child.
        directory.mkdir(parents=True,exist_ok=False)
        (directory/"logs").mkdir()
        write_json(directory/"request.json",request)
        job_progress(directory,"准备启动",scene_revision=revision)
        self.directory=directory;self.started=time.monotonic()
        if self.log:self.log.close()
        self.log=(directory/"logs/calibration.log").open("w",encoding="utf-8")
        env=dict(os.environ);env["AVM_PROJECT_ROOT"]=str(self.root)
        env["OPENBLAS_NUM_THREADS"]=env["OMP_NUM_THREADS"]=env["MKL_NUM_THREADS"]="1"
        try:
            self.process=subprocess.Popen([sys.executable,"-u",str(self.root/"tools/run_gui_calibration.py"),
                "--experiment",str(directory)],stdout=self.log,stderr=subprocess.STDOUT,
                env=env,start_new_session=True)
        except Exception as exc:
            self.log.close();self.log=None
            from avm_bringup.experiments import finish_experiment
            finish_experiment(directory,{"status":"failed","error":str(exc)})
            raise
        return directory

    def poll(self):
        if self.process is None:return None
        code=self.process.poll()
        result=read_json(self.directory/"result.json")
        if code is not None and self.log is not None:
            self.log.close();self.log=None
        if code is not None and not result:
            from avm_bringup.experiments import finish_experiment
            finish_experiment(self.directory,{"status":"failed","error":f"Worker exited {code}; see logs/calibration.log"})
            result=read_json(self.directory/"result.json")
        progress=read_json(self.directory/"progress.json",{})
        # The original pipeline already publishes atomic-enough stage state.
        manifests=sorted((self.directory/"pipeline").glob("**/pipeline_runs/*/run.json"))
        stages={}
        for manifest in manifests:
            record=read_json(manifest,{})
            for n,r in record.get('stages',{}).items():stages[f'{manifest.parent.parent.parent.name}/{n}']=r.get('status')
        return {"running":code is None,"exit_code":code,"directory":str(self.directory),
            "seconds":time.monotonic()-self.started,"progress":progress,
            "pipeline_stages":stages,"result":result}

    def cancel(self):
        if not self.running:
            if self.log:self.log.close();self.log=None
            return
        try:os.killpg(self.process.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid,signal.SIGKILL);self.process.wait(timeout=3)
        if self.log:self.log.close();self.log=None
        from avm_bringup.experiments import finish_experiment
        finish_experiment(self.directory,{"status":"cancelled","calibration_accepted":False,
            "scene_parameters_applied":False,"error":"Cancelled by user or editor shutdown"})
        job_progress(self.directory,"已取消",status="cancelled")

    close=cancel
