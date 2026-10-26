"""Immutable, revisioned export of the existing Scene authority; no second state model."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import yaml
import numpy as np
from .models import Scene, Board
from .transforms import inverse, pose_matrix
from .persistence import asset_entries

# Scene remains the schema consumed by renderer/GUI/ROS. This name documents
# its role without creating another set of intrinsics, poses or AVM settings.
SceneState = Scene


def board_corners(board: Board) -> np.ndarray:
    """Internal corners, row-major in the board's centered XY coordinates (m)."""
    x = (np.arange(1, board.columns) - board.columns / 2) * board.square_size
    y = (np.arange(1, board.rows) - board.rows / 2) * board.square_size
    xx, yy = np.meshgrid(x, y)
    return np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))


@dataclass(frozen=True)
class GroundTruth:
    _scene_json: str
    revision: int = 0

    @classmethod
    def capture(cls, scene: Scene, revision: int = 0) -> "GroundTruth":
        scene.validate()
        return cls(json.dumps(scene.to_dict(), sort_keys=True, allow_nan=False), int(revision))

    @property
    def scene(self) -> Scene:
        return Scene.from_dict(json.loads(self._scene_json))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self._scene_json.encode()).hexdigest()

    def to_dict(self, asset_base: Path | None = None) -> dict:
        scene = self.scene
        state = scene.to_dict()
        if asset_base is not None:
            for obj, key in asset_entries(state):
                if obj.get(key):
                    try:
                        obj[key] = Path(os.path.relpath(Path(obj[key]).resolve(), asset_base.resolve())).as_posix()
                    except ValueError:
                        obj[key] = str(Path(obj[key]).resolve())
        digest = hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return {
            "ground_truth_version": 1, "revision": self.revision,
            "scene_sha256": digest,
            "conventions": {
                "length": "meter", "pose_angle": "radian", "ui_angle": "degree",
                "pose_order": "x,y,z,roll,pitch,yaw", "euler": "Rz(yaw) Ry(pitch) Rx(roll)",
                "world_base": "X forward, Y left, Z up", "optical": "X right, Y down, Z forward",
                "quaternion": "WXYZ internally; ROS message XYZW", "pixel_center": "integer (OpenCV)",
                "transform": "T_A_from_B maps column vectors from B to A",
                "gsplat_K": "add 0.5 to cx/cy for rasterizer half-pixel centers only",
            },
            "state": state,
            "derived": {
                "cameras": {n: {
                    "T_world_from_camera_link": scene.world_pose(n).tolist(),
                    "T_world_from_camera_optical": scene.camera_optical_pose(n).tolist(),
                    "T_camera_optical_from_world": inverse(scene.camera_optical_pose(n)).tolist(),
                } for n in scene.cameras},
                "boards": {n: {"T_world_from_board": pose_matrix(b.pose).tolist(),
                                  "width_m": b.width, "height_m": b.height,
                                  "inner_corners_board_m": board_corners(b).tolist()}
                           for n, b in scene.boards.items()},
            },
        }

    def save(self, path: str | Path) -> None:
        from .persistence import atomic_yaml
        path = Path(path).expanduser().resolve()
        atomic_yaml(self.to_dict(path.parent), path)

    @classmethod
    def load(cls, path: str | Path) -> "GroundTruth":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("ground_truth_version") != 1:
            raise ValueError("Unsupported Ground Truth file")
        digest = hashlib.sha256(json.dumps(data["state"], sort_keys=True, allow_nan=False).encode()).hexdigest()
        if digest != data["scene_sha256"]:
            raise ValueError("Ground Truth scene checksum mismatch")
        state = data["state"]
        path = Path(path).expanduser().resolve()
        for obj, key in asset_entries(state):
            if obj.get(key) and not Path(obj[key]).is_absolute():
                obj[key] = str((path.parent / obj[key]).resolve())
        truth = cls.capture(Scene.from_dict(state), data["revision"])
        if data["derived"] != truth.to_dict()["derived"]:
            raise ValueError("Ground Truth derived transforms disagree with SceneState")
        return truth
