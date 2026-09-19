"""All rigid/Sim(3) coordinate conventions live here.

Column vectors: p_parent = T_parent_child @ p_child.
World/base: X forward, Y left, Z up. Optical: X right, Y down, Z forward.
RPY = extrinsic XYZ, radians; quaternions are WXYZ internally (ROS uses XYZW).
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

Array = NDArray[np.float64]
R_LINK_OPTICAL = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
T_LINK_OPTICAL = np.eye(4)
T_LINK_OPTICAL[:3, :3] = R_LINK_OPTICAL


def pose_matrix(pose: list[float] | Array) -> Array:
    values = np.asarray(pose, dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("Pose must contain six finite XYZ/RPY values")
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler("xyz", values[3:]).as_matrix()
    out[:3, 3] = values[:3]
    return out


def matrix_pose(matrix: Array) -> list[float]:
    validate_rigid(matrix)
    return [*matrix[:3, 3].tolist(), *Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz").tolist()]


def validate_rigid(matrix: Array) -> None:
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected finite 4x4 rigid transform")
    if not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError("Invalid homogeneous transform")
    r = matrix[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1):
        raise ValueError("Rotation must be orthonormal and right handed")


def inverse(matrix: Array) -> Array:
    out = np.eye(4)
    out[:3, :3] = matrix[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ matrix[:3, 3]
    return out


def transform_points(matrix: Array, points: Array) -> Array:
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def wxyz_from_matrix(matrix: Array) -> tuple[float, float, float, float]:
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return tuple(float(v) for v in q[[3, 0, 1, 2]])


def matrix_from_wxyz_position(wxyz: Array, position: Array) -> Array:
    q = np.asarray(wxyz, dtype=float)
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    out[:3, 3] = position
    validate_rigid(out)
    return out


def optical_pose(link_pose: Array) -> Array:
    return link_pose @ T_LINK_OPTICAL

