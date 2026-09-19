"""Validated, transport-independent scene state. Distances: meters, angles: radians."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
import numpy as np
from .transforms import pose_matrix, validate_rigid, optical_pose


@dataclass
class Vehicle:
    asset: str = ""
    pose: list[float] = field(default_factory=lambda: [0.] * 6)
    dimensions: list[float] = field(default_factory=lambda: [4.2, 1.8, 1.6])
    # Asset authoring coordinates -> base_link. Applies to meshes and 3DGS PLY.
    # Private assets are never implicitly recentered or normalized.
    asset_pose: list[float] = field(default_factory=lambda: [0.] * 6)
    asset_scale: float = 1.

    @property
    def is_gaussian(self) -> bool:
        return Path(self.asset).suffix.lower() == ".ply"


@dataclass
class Camera:
    name: str = "front"
    parent: str = "base_link"
    pose: list[float] = field(default_factory=lambda: [2.15, 0, 1., 0, .45, 0])
    model: str = "fisheye"
    width: int = 320
    height: int = 240
    K: list[float] = field(default_factory=lambda: [110., 0, 159.5, 0, 110., 119.5, 0, 0, 1])
    D: list[float] = field(default_factory=lambda: [0.] * 4)
    max_angle: float = 1.55

    @property
    def matrix(self) -> np.ndarray:
        return np.array(self.K, dtype=float).reshape(3, 3)


@dataclass
class Board:
    id: str = "board_001"
    rows: int = 6
    columns: int = 8
    square_size: float = .15
    pose: list[float] = field(default_factory=lambda: [3., 0, .025, 0, 0, 0])

    @property
    def width(self) -> float:
        return self.columns * self.square_size

    @property
    def height(self) -> float:
        return self.rows * self.square_size


@dataclass
class Coordinate:
    scale: float = 1.
    ground_z: float = 0.
    T_world_from_gs: list[list[float]] = field(default_factory=lambda: np.eye(4).tolist())


@dataclass
class Bev:
    x_min: float = -5.
    x_max: float = 5.
    y_min: float = -5.
    y_max: float = 5.
    width: int = 512
    height: int = 512


@dataclass
class Scene:
    gaussian_file: str = ""
    coordinate: Coordinate = field(default_factory=Coordinate)
    vehicle: Vehicle | None = None
    cameras: dict[str, Camera] = field(default_factory=dict)
    boards: dict[str, Board] = field(default_factory=dict)
    bev: Bev = field(default_factory=Bev)
    schema_version: int = 1

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported scene schema_version")
        if not np.isfinite([self.coordinate.scale, self.coordinate.ground_z]).all() or self.coordinate.scale <= 0:
            raise ValueError("Invalid coordinate scale / ground_z")
        validate_rigid(np.asarray(self.coordinate.T_world_from_gs))
        if self.vehicle:
            pose_matrix(self.vehicle.pose)
            pose_matrix(self.vehicle.asset_pose)
            if not np.isfinite([*self.vehicle.dimensions, self.vehicle.asset_scale]).all() or min(*self.vehicle.dimensions, self.vehicle.asset_scale) <= 0:
                raise ValueError("Vehicle dimensions and scale must be positive")
        if len(self.cameras) > 6:
            raise ValueError("MVP supports up to 6 cameras")
        for name, camera in self.cameras.items():
            if name == "base_link" or name in self.boards or name != camera.name or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", name):
                raise ValueError("Invalid camera name")
            if camera.parent not in ("world", "base_link") or (camera.parent == "base_link" and self.vehicle is None):
                raise ValueError("Camera parent must exist")
            pose_matrix(camera.pose)
            if camera.model not in ("pinhole", "fisheye", "ftheta"):
                raise ValueError("Unknown camera model")
            if not all(isinstance(v, int) and 16 <= v <= 4096 for v in (camera.width, camera.height)):
                raise ValueError("Image dimensions must be integers in [16,4096]")
            k = camera.matrix
            if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0:
                raise ValueError("K requires positive fx/fy, zero skew, homogeneous last row")
            expected = 5 if camera.model == "pinhole" else 4
            if len(camera.D) != expected or not np.isfinite(camera.D).all():
                raise ValueError(f"{camera.model} requires {expected} distortion coefficients")
            if not 0.1 < camera.max_angle < (1.570796 if camera.model != "ftheta" else 3.13):
                raise ValueError("Camera max_angle outside supported domain")
            if camera.model != "pinhole":
                theta = np.linspace(0, camera.max_angle, 512)
                derivative = 1 + sum((2*i+3)*d*theta**(2*i+2) for i,d in enumerate(camera.D))
                if np.min(derivative) <= .01:
                    raise ValueError("Fisheye polynomial must be monotonic within max_angle")
        for name, board in self.boards.items():
            if name != board.id or not re.fullmatch(r"board_[a-zA-Z0-9_]+", name):
                raise ValueError("Invalid board id; use board_ prefix")
            pose_matrix(board.pose)
            if not all(isinstance(v, int) and 1 <= v <= 100 for v in (board.rows, board.columns)) or not np.isfinite(board.square_size) or board.square_size <= 0:
                raise ValueError("Invalid board dimensions")
        b = self.bev
        if not np.isfinite([b.x_min, b.x_max, b.y_min, b.y_max]).all() or b.x_min >= b.x_max or b.y_min >= b.y_max:
            raise ValueError("Invalid BEV bounds")
        if not all(isinstance(v, int) and 16 <= v <= 4096 for v in (b.width, b.height)):
            raise ValueError("Invalid BEV output resolution")

    def object_ids(self) -> list[str]:
        return (["base_link"] if self.vehicle else []) + list(self.cameras) + list(self.boards)

    def object(self, object_id: str) -> Vehicle | Camera | Board:
        if object_id == "base_link" and self.vehicle:
            return self.vehicle
        if object_id in self.cameras:
            return self.cameras[object_id]
        if object_id in self.boards:
            return self.boards[object_id]
        raise KeyError(f"Object does not exist: {object_id}")

    def world_pose(self, object_id: str) -> np.ndarray:
        obj = self.object(object_id)
        local = pose_matrix(obj.pose)
        if isinstance(obj, Camera) and obj.parent == "base_link":
            return pose_matrix(self.vehicle.pose) @ local
        return local

    def camera_optical_pose(self, name: str) -> np.ndarray:
        return optical_pose(self.world_pose(name))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        data = dict(data)
        vehicle_data = dict(data["vehicle"]) if data.get("vehicle") else None
        if vehicle_data is not None:
            # Schema v1 compatibility: mesh_* used to describe the only vehicle
            # asset type. They now apply equally to meshes and Gaussian PLY.
            vehicle_data.setdefault("asset_pose", vehicle_data.pop("mesh_pose", [0.] * 6))
            vehicle_data.setdefault("asset_scale", vehicle_data.pop("mesh_scale", 1.))
        scene = cls(
            gaussian_file=data.get("gaussian_file", ""),
            coordinate=Coordinate(**data.get("coordinate", {})),
            vehicle=Vehicle(**vehicle_data) if vehicle_data is not None else None,
            cameras={n: Camera(**c) for n, c in data.get("cameras", {}).items()},
            boards={n: Board(**b) for n, b in data.get("boards", {}).items()},
            bev=Bev(**data.get("bev", {})), schema_version=data.get("schema_version", 1))
        scene.validate()
        return scene


def default_scene(asset_directory: Path) -> Scene:
    cameras = {}
    for name, xyz, yaw in [
        ("front", [2.15, 0, 1], 0), ("rear", [-2.15, 0, 1], np.pi),
        ("left", [0, .95, 1.05], np.pi/2), ("right", [0, -.95, 1.05], -np.pi/2)]:
        cameras[name] = Camera(name=name, pose=[*xyz, 0, .55, yaw])
    scene = Scene(gaussian_file=str(asset_directory / "test_room.ply"),
                  vehicle=Vehicle(asset=str(asset_directory / "vehicle.glb")),
                  cameras=cameras, boards={"board_001": Board()})
    scene.validate()
    return scene
