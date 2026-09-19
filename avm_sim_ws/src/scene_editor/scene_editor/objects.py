"""Viser display objects only. Sensor rendering never reads this module."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import trimesh
from scene_manager.models import Scene, Board
from scene_manager.transforms import pose_matrix, wxyz_from_matrix, T_LINK_OPTICAL
from gs_renderer.gaussian import GaussianScene


def board_mesh(board: Board) -> trimesh.Trimesh:
    vertices, faces, colors = [], [], []
    for r in range(board.rows):
        for c in range(board.columns):
            x, y = c * board.square_size - board.width/2, r * board.square_size - board.height/2
            s = board.square_size
            n = len(vertices)
            vertices.extend([[x,y,0], [x+s,y,0], [x+s,y+s,0], [x,y+s,0]])
            faces.extend([[n,n+1,n+2], [n,n+2,n+3]])
            color = [240,240,240,255] if (r+c)%2 == 0 else [12,12,12,255]
            colors.extend([color, color])
    return trimesh.Trimesh(vertices=vertices, faces=faces + [f[::-1] for f in faces], face_colors=colors + colors, process=False)


class ObjectView:
    def __init__(self, server, on_select):
        self.server, self.on_select = server, on_select
        self.handles = []
        self.frames = {}
        self.signature = None

    def sync(self, scene: Scene) -> None:
        signature = repr((scene.vehicle.asset if scene.vehicle else None,
            scene.vehicle.asset_pose if scene.vehicle else None,
            scene.vehicle.asset_scale if scene.vehicle else None,
            scene.vehicle.dimensions if scene.vehicle else None,
            [(n,c.parent,c.width,c.height,c.K) for n,c in scene.cameras.items()],
            [(n,b.rows,b.columns,b.square_size) for n,b in scene.boards.items()]))
        if self.signature != signature:
            for handle in reversed(self.handles):
                handle.remove()
            self.handles, self.frames = [], {}
            if scene.vehicle:
                frame = self.server.scene.add_frame("/objects/base_link", show_axes=False)
                self.frames["base_link"] = frame
                self.handles.append(frame)
                vehicle = scene.vehicle
                t = pose_matrix(vehicle.asset_pose)
                if vehicle.asset and vehicle.is_gaussian:
                    gs = GaussianScene.load(vehicle.asset, vehicle.asset_scale, t)
                    handle = self.server.scene.add_gaussian_splats(
                        "/objects/base_link/gaussian", centers=gs.means,
                        covariances=gs.covariances, rgbs=gs.colors,
                        opacities=gs.opacities[:, None])
                elif vehicle.asset:
                    mesh = trimesh.load_scene(str(Path(vehicle.asset))).to_mesh()
                    handle = self.server.scene.add_mesh_trimesh(
                        "/objects/base_link/mesh", mesh, scale=vehicle.asset_scale,
                        wxyz=wxyz_from_matrix(t), position=t[:3, 3])
                else:
                    mesh = trimesh.creation.box(extents=vehicle.dimensions)
                    mesh.apply_translation([0,0,vehicle.dimensions[2]/2])
                    handle = self.server.scene.add_mesh_trimesh(
                        "/objects/base_link/mesh", mesh, scale=vehicle.asset_scale,
                        wxyz=wxyz_from_matrix(t), position=t[:3, 3])
                handle.on_click(lambda _: self.on_select("base_link"))
                self.handles.append(handle)
            for name, camera in scene.cameras.items():
                # Use a real parent node so the scene graph follows vehicle motion.
                path = f"/objects/base_link/cameras/{name}" if camera.parent == "base_link" else f"/objects/{name}"
                frame = self.server.scene.add_frame(path, show_axes=False)
                self.frames[name] = frame
                self.handles.append(frame)
                handle = self.server.scene.add_camera_frustum(path + "/optical",
                    fov=1.2, aspect=camera.width/camera.height, scale=.42,
                    color=(50,190,215), wxyz=wxyz_from_matrix(T_LINK_OPTICAL))
                handle.on_click(lambda _, n=name: self.on_select(n))
                self.handles.append(handle)
            for name, board in scene.boards.items():
                handle = self.server.scene.add_mesh_trimesh(f"/objects/{name}", board_mesh(board))
                handle.on_click(lambda _, n=name: self.on_select(n))
                self.frames[name] = handle
                self.handles.append(handle)
            self.signature = signature
        for name, handle in self.frames.items():
            t = pose_matrix(scene.object(name).pose)
            handle.position = t[:3,3]
            handle.wxyz = wxyz_from_matrix(t)
