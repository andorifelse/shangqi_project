"""Headless sensor pipeline. No UI dependencies or screenshots."""
from __future__ import annotations
import numpy as np
from .gaussian import GaussianScene
from .cpu_renderer import CpuRayRenderer
from .board import overlay_boards
from scene_manager.models import Scene


class SensorPipeline:
    def __init__(self, backend: str = "cpu"):
        self.backend = backend
        self._key = None
        self.renderer = None

    def render(self, scene: Scene):
        key = (scene.gaussian_file,repr(scene.coordinate),self.backend)
        if key != self._key:
            gs = GaussianScene.load(scene.gaussian_file,scene.coordinate.scale,
                                    np.array(scene.coordinate.T_world_from_gs))
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
