"""Versioned YAML schema, atomic saves, asset paths relative to the scene file."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import os
import tempfile
import yaml
from .models import Scene


def load_scene(path: str | Path) -> Scene:
    path = Path(path).expanduser().resolve()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("scene.yaml must contain a mapping")
    data = deepcopy(data)
    for key in ("scene", "coordinate", "vehicle", "cameras", "boards", "bev"):
        if key in data and data[key] is not None and not isinstance(data[key], dict):
            raise ValueError(f"Scene field '{key}' must be a mapping")
    # A scene section is kept for asset fields; object data is at the top level.
    data["gaussian_file"] = (data.pop("scene", {}) or {}).get("gaussian_file", data.get("gaussian_file", ""))
    for obj, key in [(data, "gaussian_file"), (data.get("vehicle") or {}, "asset")]:
        if obj.get(key):
            obj[key] = str((path.parent / obj[key]).resolve())
    try:
        scene = Scene.from_dict(data)
    except (TypeError, AttributeError, KeyError, IndexError) as exc:
        raise ValueError(f"Invalid scene schema: {exc}") from exc
    for asset in (scene.gaussian_file, scene.vehicle.asset if scene.vehicle else ""):
        if asset and not Path(asset).is_file():
            raise ValueError(f"Asset does not exist: {asset}")
    return scene


def save_scene(scene: Scene, path: str | Path) -> None:
    scene.validate()
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = scene.to_dict()
    for obj, key in [(data, "gaussian_file"), (data.get("vehicle") or {}, "asset")]:
        if obj.get(key):
            try:
                obj[key] = Path(os.path.relpath(Path(obj[key]).resolve(), path.parent)).as_posix()
            except ValueError:  # Different Windows drives: retain absolute asset path.
                obj[key] = str(Path(obj[key]).resolve())
    data["scene"] = {"gaussian_file": data.pop("gaussian_file")}
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
