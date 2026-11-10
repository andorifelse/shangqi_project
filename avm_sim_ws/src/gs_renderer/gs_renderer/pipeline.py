"""Headless sensor pipeline. No UI dependencies or screenshots."""
from __future__ import annotations
import numpy as np
from .gaussian import GaussianScene
from .cpu_renderer import CpuRayRenderer
from .board import overlay_boards
from scene_manager.models import Scene
from scene_manager.transforms import pose_matrix


class SensorPipeline:
    def __init__(self, backend: str = "cpu"):
        self.backend = backend
        self._key = None
        self._environment_key = None
        self._environment = None
        self._vehicle_key = None
        self._vehicle_local = None
        self._board_local = {}
        self._world_parts = {}
        self.renderer = None

    def render(self, scene: Scene):
        environment_key = (scene.gaussian_file, repr(scene.coordinate))
        if environment_key != self._environment_key:
            self._environment = GaussianScene.load(
                scene.gaussian_file, scene.coordinate.scale,
                np.array(scene.coordinate.T_world_from_gs)) if scene.gaussian_file else None
            self._environment_key = environment_key

        parts={}
        if self._environment is not None:parts['environment']=self._environment
        vehicle_key = None
        vehicle_world = None
        if scene.vehicle and scene.vehicle.asset and scene.vehicle.is_gaussian:
            vehicle_key = (scene.vehicle.asset, scene.vehicle.asset_scale,
                           tuple(scene.vehicle.asset_pose))
            if vehicle_key != self._vehicle_key:
                self._vehicle_local = GaussianScene.load(
                    scene.vehicle.asset, scene.vehicle.asset_scale,
                    pose_matrix(scene.vehicle.asset_pose))
                self._vehicle_key = vehicle_key
            cache_key=(vehicle_key,tuple(scene.vehicle.pose))
            cached=self._world_parts.get('vehicle')
            if cached is None or cached[0]!=cache_key:
                cached=(cache_key,self._vehicle_local.transformed(transform=pose_matrix(scene.vehicle.pose)))
                self._world_parts['vehicle']=cached
            vehicle_world=cached[1]
            parts['vehicle']=vehicle_world
        elif self._vehicle_key is not None:
            self._vehicle_key = None
            self._vehicle_local = None

        board_keys, active = [], set()
        for name, board in scene.boards.items():
            if not board.is_gaussian:
                continue
            local_key = (board.asset, board.asset_scale, tuple(board.asset_pose))
            active.add(local_key)
            if local_key not in self._board_local:
                self._board_local[local_key] = GaussianScene.load(
                    board.asset, board.asset_scale, pose_matrix(board.asset_pose))
            board_keys.append((name, local_key, tuple(board.pose)))
            cache_key=(local_key,tuple(board.pose));part_name='board:'+name
            cached=self._world_parts.get(part_name)
            if cached is None or cached[0]!=cache_key:
                cached=(cache_key,self._board_local[local_key].transformed(transform=pose_matrix(board.pose)))
                self._world_parts[part_name]=cached
            parts[part_name]=cached[1]
        self._board_local = {k:v for k,v in self._board_local.items() if k in active}
        key = (environment_key, vehicle_key,
               tuple(scene.vehicle.pose) if vehicle_world is not None else None,
               tuple(board_keys), self.backend)
        self._world_parts={name:entry for name,entry in self._world_parts.items() if name in parts}
        if key != self._key:
            if not parts:
                raise ValueError("Sensor rendering requires an environment or Gaussian object")
            if self.backend == "gsplat":
                from .gsplat_renderer import GsplatRenderer
                if self.renderer is None:self.renderer=GsplatRenderer()
                self.renderer.set_parts(parts)
            else:
                values=list(parts.values())
                gs=GaussianScene.concatenate(*values) if len(values)>1 else values[0]
                self.renderer = CpuRayRenderer(gs)
            self._key = key
        results = self.renderer.render_batch(scene)
        for name,result in results.items():
            overlay_boards(scene,name,result)
        return results
