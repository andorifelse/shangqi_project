"""Dynamic checkerboard overlay in world coordinates, never baked into Gaussian PLY."""
from __future__ import annotations
import numpy as np
from scene_manager.models import Board, Camera, Scene
from scene_manager.camera_models import camera_rays
from scene_manager.transforms import inverse, pose_matrix, transform_points
from .types import RenderResult


def intersect_board(board: Board, origin_world: np.ndarray, rays_world: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    local_from_world = inverse(pose_matrix(board.pose))
    origin = transform_points(local_from_world,origin_world)
    directions = rays_world @ local_from_world[:3,:3].T
    denominator = directions[...,2]
    distance = np.divide(-origin[2],denominator,out=np.full(denominator.shape,np.inf),
                         where=np.abs(denominator)>1e-10)
    # Avoid inf * 0 on parallel rays.
    safe_distance = np.where(np.isfinite(distance),distance,0)
    hit = origin + safe_distance[...,None]*directions
    valid = (distance>.001) & np.isfinite(distance) & (np.abs(hit[...,0])<board.width/2) & (np.abs(hit[...,1])<board.height/2)
    column = np.floor((hit[...,0]+board.width/2)/board.square_size).astype(np.int64)
    row = np.floor((hit[...,1]+board.height/2)/board.square_size).astype(np.int64)
    gray = np.where((row+column)%2==0,240,12).astype(np.uint8)
    rgb = np.repeat(gray[...,None],3,axis=-1)
    return distance,valid,rgb


def overlay_boards(scene: Scene, name: str, result: RenderResult) -> RenderResult:
    camera = scene.cameras[name]
    rays,valid = camera_rays(camera)
    world_optical = scene.camera_optical_pose(name)
    world_rays = rays @ world_optical[:3,:3].T
    for board in scene.boards.values():
        distance,inside,rgb = intersect_board(board,world_optical[:3,3],world_rays)
        visible = valid & inside & (distance < result.depth + .005)
        result.rgb[visible] = rgb[visible]
        result.depth[visible] = distance[visible]
        result.alpha[visible] = 1
    return result
