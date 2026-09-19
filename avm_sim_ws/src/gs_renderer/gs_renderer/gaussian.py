"""Graphdeco Gaussian PLY loader shared by WebGL and sensor backends."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from plyfile import PlyData
from scipy.special import expit
from scipy.spatial.transform import Rotation
from scene_manager.transforms import transform_points, validate_rigid

SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class GaussianScene:
    means: np.ndarray
    covariances: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray
    quats: np.ndarray
    scales: np.ndarray

    def transformed(self, scale: float = 1.,
                    transform: np.ndarray | None = None) -> "GaussianScene":
        """Return a Sim(3)-transformed copy without re-reading the PLY."""
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Gaussian coordinate scale must be positive")
        t = np.eye(4) if transform is None else np.asarray(transform, dtype=float)
        validate_rigid(t)
        rotations = Rotation.from_quat(self.quats[:, [1, 2, 3, 0]]).as_matrix()
        rotations = t[:3, :3] @ rotations
        covariance = scale**2 * (t[:3, :3] @ self.covariances @ t[:3, :3].T)
        quats = Rotation.from_matrix(rotations).as_quat()[:, [3, 0, 1, 2]]
        arrays = (
            transform_points(t, self.means * scale), covariance,
            self.colors.copy(), self.opacities.copy(), quats, self.scales * scale)
        return GaussianScene(*(np.asarray(a, dtype=np.float32) for a in arrays))

    @classmethod
    def concatenate(cls, *scenes: "GaussianScene") -> "GaussianScene":
        """Combine independently transformed Gaussian objects for one render."""
        scenes = tuple(scene for scene in scenes if len(scene.means))
        if not scenes:
            raise ValueError("At least one non-empty Gaussian scene is required")
        fields = ("means", "covariances", "colors", "opacities", "quats", "scales")
        return cls(*(np.concatenate([getattr(scene, field) for scene in scenes], axis=0)
                     for field in fields))

    @classmethod
    def load(cls, path: str | Path, scale: float = 1.,
             transform: np.ndarray | None = None) -> "GaussianScene":
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Gaussian coordinate scale must be positive")
        vertex = PlyData.read(str(path))["vertex"].data
        required = ["x", "y", "z", "opacity", *[f"f_dc_{i}" for i in range(3)],
                    *[f"scale_{i}" for i in range(3)], *[f"rot_{i}" for i in range(4)]]
        missing = set(required) - set(vertex.dtype.names or ())
        if missing:
            raise ValueError(f"Not a Gaussian PLY; missing properties: {sorted(missing)}")
        get = lambda names: np.column_stack([vertex[n] for n in names]).astype(np.float64)
        means = get(["x", "y", "z"])
        q = get([f"rot_{i}" for i in range(4)])
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        if len(means) == 0 or (norms < 1e-10).any():
            raise ValueError("Empty scene or zero Gaussian quaternion")
        q /= norms
        rotations = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        scales = np.exp(np.clip(get([f"scale_{i}" for i in range(3)]), -20, 10))
        covariance = (rotations * scales[:, None, :] ** 2) @ rotations.transpose(0, 2, 1)
        colors = np.clip(0.5 + SH_C0 * get([f"f_dc_{i}" for i in range(3)]), 0, 1)
        opacity = expit(np.asarray(vertex["opacity"], dtype=float))
        q = Rotation.from_matrix(rotations).as_quat()[:, [3, 0, 1, 2]]
        arrays = [means, covariance, colors, opacity, q, scales]
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("Gaussian PLY contains non-finite data")
        raw = cls(*(a.astype(np.float32) for a in arrays))
        return raw.transformed(scale, transform)
