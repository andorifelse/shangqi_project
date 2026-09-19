"""Deterministic, meter-scale fixtures, generated without private assets."""
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement


def generate_assets(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    ply = directory / "test_room.ply"
    if not ply.exists():
        axis = np.arange(-6., 6.01, .28)
        x, y = np.meshgrid(axis, axis)
        means = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
        checker = ((np.floor(means[:, 0]) + np.floor(means[:, 1])) % 2).astype(bool)
        colors = np.where(checker[:, None], [.32, .38, .43], [.58, .63, .67])
        for side in (-1, 1):
            xx, zz = np.meshgrid(axis, np.arange(.3, 2.7, .3))
            wall = np.column_stack([xx.ravel(), np.full(xx.size, side * 6.), zz.ravel()])
            means = np.vstack([means, wall])
            colors = np.vstack([colors, np.tile([.30, .47, .63] if side == 1 else [.63, .38, .23], (len(wall), 1))])
        names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                 "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        data = np.zeros(len(means), dtype=[(n, "<f4") for n in names])
        for i, n in enumerate(["x", "y", "z"]):
            data[n] = means[:, i]
            data[f"f_dc_{i}"] = (colors[:, i] - .5) / .28209479177387814
            data[f"scale_{i}"] = np.log(.18 if i < 2 else .055)
        data["opacity"] = 4.
        data["rot_0"] = 1.
        PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(ply))
    glb = directory / "vehicle.glb"
    if not glb.exists():
        import trimesh
        body = trimesh.creation.box(extents=(4.2, 1.8, .65))
        body.apply_translation([0, 0, .65])
        body.visual.face_colors = [45, 112, 180, 255]
        cabin = trimesh.creation.box(extents=(2.0, 1.65, .65))
        cabin.apply_translation([-.2, 0, 1.25])
        cabin.visual.face_colors = [45, 62, 80, 255]
        mesh = trimesh.util.concatenate([body, cabin])
        mesh.export(str(glb))

