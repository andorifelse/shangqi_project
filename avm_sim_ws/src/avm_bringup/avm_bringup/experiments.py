"""Small filesystem experiment records; no database or new framework."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import zipfile
from scene_manager.ground_truth import GroundTruth
from scene_manager.persistence import atomic_yaml, save_scene


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_manifest(project_root):
    result={"python":sys.version,"executable":sys.executable,"platform":platform.platform(),
            "packages":{}}
    for package in ("numpy","scipy","opencv-python-headless","gsplat","torch","PyYAML"):
        try:
            result["packages"][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package]=None
    for key,args in (("git_commit",["git","rev-parse","HEAD"]),
                     ("git_status",["git","status","--short"]),
                     ("git_diff",["git","diff","--no-ext-diff","--binary"])):
        try:
            p=subprocess.run(args,cwd=project_root,capture_output=True,text=True,timeout=10)
            result[key]=p.stdout if p.returncode==0 else None
        except (OSError,subprocess.TimeoutExpired):
            result[key]=None
    # Hash code, including new untracked experiment code not present in git diff.
    root=Path(project_root)
    result["source_sha256"]={str(p.relative_to(root)):file_sha256(p)
        for folder in ("avm_sim_ws/src","tools")
        for p in sorted((root/folder).rglob("*"))
        if p.is_file() and p.suffix in (".py",".sh",".xml",".txt",".cfg")}
    return result


def begin_experiment(scene, directory, config, project_root, revision=0):
    truth=GroundTruth.capture(scene,revision)
    directory=begin_experiment_from_snapshot(truth,directory,config,project_root)
    return directory,truth


def begin_experiment_from_snapshot(snapshot, directory, config, project_root):
    """Persist an existing immutable GroundTruth snapshot and its exact revision."""
    directory=Path(directory).expanduser().resolve()
    directory.mkdir(parents=True,exist_ok=False)
    (directory/"logs").mkdir()
    scene=snapshot.scene
    save_scene(scene,directory/"scene.yaml")
    snapshot.save(directory/"ground_truth.yaml")
    atomic_yaml(config,directory/"config.yaml")
    (directory/"logs"/"runtime.json").write_text(json.dumps(runtime_manifest(project_root),indent=2),encoding="utf-8")
    root=Path(project_root)
    with zipfile.ZipFile(directory/"logs"/"source.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in ("avm_sim_ws/src","tools","tests"):
            for source in sorted((root/folder).rglob("*")):
                if source.is_file() and source.suffix in (".py",".sh",".xml",".txt",".cfg",".yaml"):
                    archive.write(source,str(source.relative_to(root)))
        for name in ("requirements.txt","requirements-gpu.txt","pytest.ini"):
            if (root/name).is_file():
                archive.write(root/name,name)
    (directory/"logs"/"run.log").write_text(f"Started {datetime.now(timezone.utc).isoformat()}\n",encoding="utf-8")
    assets={}
    for label,path in (("environment",scene.gaussian_file),("vehicle",scene.vehicle.asset if scene.vehicle else "")):
        if path:
            assets[label]={"path":str(Path(path).resolve()),"sha256":file_sha256(path)}
    (directory/"logs"/"assets.json").write_text(json.dumps(assets,indent=2),encoding="utf-8")
    return directory


def append_log(directory, message):
    with (Path(directory)/"logs"/"run.log").open("a",encoding="utf-8") as stream:
        stream.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")


def finish_experiment(directory, result):
    result={"finished_utc":datetime.now(timezone.utc).isoformat(),**result}
    (Path(directory)/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    append_log(directory,f"Finished: {result.get('status','unknown')}")
