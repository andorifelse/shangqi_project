"""Emit reproducible environment evidence without requiring ROS/CUDA imports."""
from __future__ import annotations
from pathlib import Path
import importlib.util
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys


def main() -> None:
    report={"platform":platform.platform(),"python":sys.version,"executable":sys.executable}
    for command in ("nvidia-smi","nvcc","ros2","colcon","cl","g++","ninja","wsl"):
        binary=shutil.which(command)
        report[command]={"path":binary}
        if binary and command in ("nvidia-smi","nvcc"):
            result=subprocess.run([binary,"--version"],capture_output=True,timeout=20)
            report[command]["version"]=result.stdout.decode("utf-8",errors="replace")
    for module,package in [("torch","torch"),("gsplat","gsplat"),("viser","viser"),
                           ("numpy","numpy"),("cv2","opencv-python-headless"),("rclpy","rclpy")]:
        found=importlib.util.find_spec(module) is not None
        try:
            version=importlib.metadata.version(package) if found else None
        except importlib.metadata.PackageNotFoundError:
            version="ROS system package"
        report[module]={"found":found,"version":version}
    if report["torch"]["found"]:
        import torch
        report["torch"]["cuda_available"]=torch.cuda.is_available()
        report["torch"]["cuda_runtime"]=torch.version.cuda
    print(json.dumps(report,ensure_ascii=False,indent=2))
    output=Path(__file__).resolve().parents[1]/"outputs"/"environment.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
