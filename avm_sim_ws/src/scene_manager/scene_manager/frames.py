"""Transport-neutral TF description, used by ROS and coordinate tests."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .models import Scene
from .transforms import pose_matrix, T_LINK_OPTICAL


@dataclass(frozen=True)
class Frame:
    parent: str
    child: str
    transform: np.ndarray


def scene_frames(scene: Scene) -> list[Frame]:
    frames = []
    if scene.vehicle:
        frames.append(Frame("world", "base_link", pose_matrix(scene.vehicle.pose)))
    for name, camera in scene.cameras.items():
        frames.append(Frame(camera.parent, f"camera_{name}_link", pose_matrix(camera.pose)))
        frames.append(Frame(f"camera_{name}_link", f"camera_{name}_optical", T_LINK_OPTICAL.copy()))
    for name, board in scene.boards.items():
        frames.append(Frame("world", name, pose_matrix(board.pose)))
    return frames
