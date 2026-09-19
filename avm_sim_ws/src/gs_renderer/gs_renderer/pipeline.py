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
        self.renderer = None

    def render(self, scene: Scene):
        environment_key = (scene.gaussian_file, repr(scene.coordinate))
        if environment_key != self._environment_key:
            self._environment = GaussianScene.load(
                scene.gaussian_file, scene.coordinate.scale,
                np.array(scene.coordinate.T_world_from_gs))
            self._environment_key = environment_key

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
            vehicle_world = self._vehicle_local.transformed(
                transform=pose_matrix(scene.vehicle.pose))
        elif self._vehicle_key is not None:
            self._vehicle_key = None
            self._vehicle_local = None

        key = (environment_key, vehicle_key,
               tuple(scene.vehicle.pose) if vehicle_world is not None else None,
               self.backend)
        if key != self._key:
            gs = (GaussianScene.concatenate(self._environment, vehicle_world)
                  if vehicle_world is not None else self._environment)
            if self.backend == "gsplat":
                from .gsplat_renderer import GsplatRenderer
                self.renderer = GsplatRenderer(gs)
            else:
                self.renderer = CpuRayRenderer(gs)
            self._key = key
        results = self.renderer.render_batch(scene)
        for name,result in results.items():
            overlay_boards(scene,name,result)
        return results
