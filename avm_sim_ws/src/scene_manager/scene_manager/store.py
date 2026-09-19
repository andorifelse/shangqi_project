"""Single scene authority with atomic edits, snapshots and revision checks."""
from __future__ import annotations
from copy import deepcopy
from threading import RLock
from .models import Scene, Vehicle, Camera, Board


class SceneStore:
    def __init__(self, scene: Scene):
        scene.validate()
        self._scene = deepcopy(scene)
        self._revision = 0
        self._lock = RLock()

    def snapshot(self) -> tuple[int, Scene]:
        with self._lock:
            return self._revision, deepcopy(self._scene)

    def replace(self, scene: Scene) -> int:
        scene.validate()
        with self._lock:
            self._scene = deepcopy(scene)
            self._revision += 1
            return self._revision

    def command(self, command: dict, expected_revision: int = -1) -> int:
        with self._lock:
            if expected_revision >= 0 and self._revision != expected_revision:
                raise ValueError("Scene changed; refresh before editing")
            action = command["action"]
            if action == "save":
                from .persistence import save_scene
                save_scene(self._scene, command["path"])
                return self._revision
            if action == "load":
                from .persistence import load_scene
                return self.replace(load_scene(command["path"]))
            scene = deepcopy(self._scene)
            if action == "add_vehicle":
                if scene.vehicle:
                    raise ValueError("MVP allows one vehicle")
                scene.vehicle = Vehicle(**command.get("properties", {}))
            elif action == "add_camera":
                name = command["name"]
                if name in scene.object_ids():
                    raise ValueError("Object already exists")
                scene.cameras[name] = Camera(name=name, **command.get("properties", {}))
            elif action == "add_board":
                name = command["name"]
                if name in scene.object_ids():
                    raise ValueError("Object already exists")
                scene.boards[name] = Board(id=name, **command.get("properties", {}))
            elif action == "delete":
                name = command["id"]
                scene.object(name)
                if name == "base_link":
                    scene.vehicle = None
                    scene.cameras = {n:c for n,c in scene.cameras.items() if c.parent != "base_link"}
                else:
                    scene.cameras.pop(name, None)
                    scene.boards.pop(name, None)
            elif action == "pose":
                scene.object(command["id"]).pose = list(command["pose"])
            elif action == "properties":
                obj = scene.object(command["id"])
                allowed = {
                    Camera: {"model", "width", "height", "K", "D", "max_angle"},
                    Board: {"rows", "columns", "square_size"},
                    Vehicle: {"asset", "dimensions", "asset_pose", "asset_scale"}
                }[type(obj)]
                for key, value in command["properties"].items():
                    if key not in allowed:
                        raise ValueError(f"Property cannot be edited: {key}")
                    setattr(obj, key, value)
            elif action == "gaussian":
                scene.gaussian_file = str(command["path"])
            else:
                raise ValueError(f"Unknown command: {action}")
            scene.validate()
            if action == "properties" and "asset" in command["properties"]:
                from pathlib import Path
                asset = command["properties"]["asset"]
                if asset and not Path(asset).is_file():
                    raise ValueError(f"Vehicle asset does not exist: {asset}")
            self._scene = scene
            self._revision += 1
            return self._revision
