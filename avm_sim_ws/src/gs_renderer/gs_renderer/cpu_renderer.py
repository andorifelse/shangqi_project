"""Small-scene CPU reference renderer using real camera rays.

Each ray evaluates the closest point in the anisotropic Gaussian metric.
Front-to-back alpha compositing is ordered by Gaussian-center distance.
This is a geometry/debug fallback, not a performance substitute for gsplat.
"""
from __future__ import annotations
from functools import lru_cache
import json
import numpy as np
from scipy.spatial import cKDTree
from scene_manager.models import Camera, Scene
from scene_manager.camera_models import camera_rays
from scene_manager.transforms import inverse, transform_points
from .gaussian import GaussianScene
from .types import RenderResult


@lru_cache(maxsize=12)
def ray_index(serialized_camera: str):
    camera = Camera(**json.loads(serialized_camera))
    rays, valid = camera_rays(camera)
    indices = np.flatnonzero(valid.ravel())
    directions = rays.reshape(-1,3)[indices]
    return rays, valid, indices, cKDTree(directions)


class CpuRayRenderer:
    name = "CPU Gaussian ray reference"

    def __init__(self, gaussian: GaussianScene):
        self.gs = gaussian
        self.precision = np.linalg.inv(gaussian.covariances)
        self.radii = 3.5*np.sqrt(np.linalg.eigvalsh(gaussian.covariances)[:,-1])

    def render(self, camera: Camera, world_optical: np.ndarray) -> RenderResult:
        from dataclasses import asdict
        rays,valid,valid_indices,tree = ray_index(json.dumps(asdict(camera),sort_keys=True))
        d = rays.reshape(-1,3)
        t = inverse(world_optical)
        means = transform_points(t,self.gs.means)
        precision = t[:3,:3] @ self.precision @ t[:3,:3].T
        radii = self.radii
        distances = np.linalg.norm(means,axis=1)
        trans = np.ones(len(d),dtype=np.float64)
        rgb = np.zeros((len(d),3))
        depth_sum = np.zeros(len(d))
        for i in np.argsort(distances):
            if distances[i] < 1e-8:
                continue
            angle = np.arcsin(min(1.,radii[i]/distances[i]))
            chord = 2*np.sin(angle*.5)
            if radii[i] >= distances[i]:
                ids = valid_indices
            else:
                ids = valid_indices[np.asarray(tree.query_ball_point(means[i]/distances[i],chord),dtype=int)]
            ids = ids[trans[ids] > .002]
            if not len(ids):
                continue
            dr = d[ids]
            pm = precision[i] @ means[i]
            dpd = np.einsum("ni,ij,nj->n",dr,precision[i],dr)
            dpm = dr @ pm
            distance = dpm/dpd
            mahal = means[i]@pm-dpm*dpm/dpd
            a = np.clip(self.gs.opacities[i]*np.exp(-.5*np.maximum(0,mahal)),0,.995)
            a[(distance < .01) | (mahal > 12.25)] = 0
            weight = trans[ids]*a
            rgb[ids] += weight[:,None]*self.gs.colors[i]
            depth_sum[ids] += weight*distance
            trans[ids] *= 1-a
        alpha = 1-trans
        depth = np.divide(depth_sum,alpha,out=np.full(len(d),np.inf),where=alpha>.01)
        rgb += trans[:,None]*np.array([.055,.07,.095])
        rgb[~valid.ravel()] = 0
        h,w = camera.height,camera.width
        return RenderResult(np.clip(rgb.reshape(h,w,3)*255,0,255).astype(np.uint8),
            depth.reshape(h,w).astype(np.float32),alpha.reshape(h,w).astype(np.float32),valid)

    def render_batch(self, scene: Scene) -> dict[str,RenderResult]:
        return {name:self.render(camera,scene.camera_optical_pose(name)) for name,camera in scene.cameras.items()}
