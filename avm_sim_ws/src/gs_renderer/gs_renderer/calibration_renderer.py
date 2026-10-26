"""Board-only reference images using the production camera rays and board overlay."""
from copy import deepcopy
from functools import lru_cache
import json
from dataclasses import asdict
import cv2
import numpy as np
from scene_manager.models import Camera
from scene_manager.camera_models import camera_rays
from .board import intersect_board


@lru_cache(maxsize=8)
def _rays(serialized_camera):
    return camera_rays(Camera(**json.loads(serialized_camera)))


def render_board_image(scene, name: str, supersample: int = 2, backend: str = "cpu", pipeline=None) -> np.ndarray:
    """Render actual checker texture, never ideal projected corner coordinates.

    Neutral background and a white surrounding board margin make detection
    well-defined. Supersampling integrates pixel area at OpenCV pixel centers.
    """
    if not 1 <= supersample <= 4:
        raise ValueError("supersample must be in [1,4]")
    camera = deepcopy(scene.cameras[name])
    w, h = camera.width, camera.height
    camera.width *= supersample
    camera.height *= supersample
    for i in (0, 4):
        camera.K[i] *= supersample
    for i in (2, 5):
        camera.K[i] = (camera.K[i] + .5)*supersample - .5
    if any(b.is_gaussian for b in scene.boards.values()):
        # Render the actual PLY, never silently replace a reconstructed board
        # with an ideal checker. Keep vehicle pose for parented cameras, but
        # exclude environment/vehicle geometry from this board-only baseline.
        from .pipeline import SensorPipeline
        sample = deepcopy(scene)
        sample.gaussian_file = ""
        if sample.vehicle:
            sample.vehicle.asset = ""
        sample.cameras = {name:camera}
        result = (pipeline or SensorPipeline(backend)).render(sample)[name].rgb
        return cv2.resize(result, (w,h), interpolation=cv2.INTER_AREA) if supersample > 1 else result
    rays, valid = _rays(json.dumps(asdict(camera), sort_keys=True))
    t = scene.camera_optical_pose(name)
    directions = rays @ t[:3, :3].T
    rgb = np.full((*valid.shape, 3), 110, np.uint8)
    depth = np.full(valid.shape, np.inf)
    for board in scene.boards.values():
        # Margin surrounds, rather than changing, the metric checker pattern.
        margin = deepcopy(board)
        margin.square_size *= 1.12
        distance, inside, _ = intersect_board(margin, t[:3, 3], directions)
        mask = valid & inside & (distance < depth + 1e-8)
        rgb[mask] = 240
        depth[mask] = distance[mask]
        distance, inside, colors = intersect_board(board, t[:3, 3], directions)
        mask = valid & inside & (distance < depth + 1e-8)
        rgb[mask], depth[mask] = colors[mask], distance[mask]
    rgb[~valid] = 0
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA) if supersample > 1 else rgb
