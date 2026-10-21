"""AVM inverse mapping using only actual front/rear/left/right sensor images."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from scene_manager.models import Scene
from scene_manager.camera_models import project
from scene_manager.transforms import inverse, pose_matrix, transform_points

REQUIRED_CAMERAS = ("front","rear","left","right")


@dataclass
class AvmResult:
    rgb: np.ndarray
    coverage: np.ndarray


def ground_points(scene: Scene) -> np.ndarray:
    """BEV top is vehicle +X, left is vehicle +Y; intersect world horizontal ground.

    A rolled/pitched vehicle does not tilt the physical ground. The image
    grid is defined by base x/y and its z is solved against world ground_z.
    """
    b = scene.bev
    x = b.x_max-(np.arange(b.height)+.5)*(b.x_max-b.x_min)/b.height
    y = b.y_max-(np.arange(b.width)+.5)*(b.y_max-b.y_min)/b.width
    xx,yy = np.meshgrid(x,y,indexing="ij")
    t = pose_matrix(scene.vehicle.pose) if scene.vehicle else np.eye(4)
    if abs(t[2,2])<1e-4:
        raise ValueError("BEV base XY plane is vertical relative to world ground")
    zz = (scene.coordinate.ground_z-t[2,3]-t[2,0]*xx-t[2,1]*yy)/t[2,2]
    return transform_points(t,np.stack([xx,yy,zz],axis=-1))


def bev_pixel_to_world(scene: Scene, pixels: np.ndarray) -> np.ndarray:
    """Subpixel BEV (u,v) to the same horizontal ground used by ground_points."""
    pixels = np.asarray(pixels, dtype=float)
    b = scene.bev
    x = b.x_max - (pixels[...,1]+.5)*(b.x_max-b.x_min)/b.height
    y = b.y_max - (pixels[...,0]+.5)*(b.y_max-b.y_min)/b.width
    t = pose_matrix(scene.vehicle.pose) if scene.vehicle else np.eye(4)
    if abs(t[2,2]) < 1e-4:
        raise ValueError("BEV base XY plane is vertical relative to world ground")
    z = (scene.coordinate.ground_z-t[2,3]-t[2,0]*x-t[2,1]*y)/t[2,2]
    return transform_points(t,np.stack([x,y,z],axis=-1))


def world_to_bev_pixel(scene: Scene, points: np.ndarray):
    """World ground points to OpenCV-centered BEV pixels; return bounds/ground mask."""
    points = np.asarray(points, dtype=float)
    t = pose_matrix(scene.vehicle.pose) if scene.vehicle else np.eye(4)
    local = transform_points(inverse(t),points)
    b = scene.bev
    u = (b.y_max-local[...,1])*b.width/(b.y_max-b.y_min)-.5
    v = (b.x_max-local[...,0])*b.height/(b.x_max-b.x_min)-.5
    pixels = np.stack([u,v],axis=-1)
    valid = np.isfinite(points).all(axis=-1) & (np.abs(points[...,2]-scene.coordinate.ground_z)<1e-6)
    valid &= (u>=-.5)&(u<b.width-.5)&(v>=-.5)&(v<b.height-.5)
    return pixels, valid


def projection_maps(scene: Scene, name: str) -> tuple[np.ndarray,np.ndarray]:
    camera = scene.cameras[name]
    p = transform_points(inverse(scene.camera_optical_pose(name)),ground_points(scene))
    uv,valid = project(camera,p)
    valid &= (uv[...,0]>=0)&(uv[...,0]<camera.width-1)&(uv[...,1]>=0)&(uv[...,1]<camera.height-1)
    # Favor central rays and nearby source views; normalize only among valid cameras.
    norm = np.linalg.norm(p,axis=-1)
    cos = p[...,2]/np.maximum(norm,1e-8)
    weight = np.maximum(cos,0.)**2/np.maximum(norm,.3)
    weight *= valid
    return uv.astype(np.float32),weight.astype(np.float32)


def projection_debug_maps(scene: Scene) -> dict[str, np.ndarray]:
    """Return per-camera mask/weight maps and normalized overlap diagnostics."""
    uv_weights = {n: projection_maps(scene, n) for n in REQUIRED_CAMERAS}
    raw = {n: item[1] for n, item in uv_weights.items()}
    stack = np.stack([raw[n] for n in REQUIRED_CAMERAS], axis=0)
    total = stack.sum(axis=0)
    masks = stack > 0
    overlap = masks.sum(axis=0).astype(np.uint8)
    normalized = np.divide(stack, total[None,...], out=np.zeros_like(stack),
                           where=total[None,...] > 1e-8)
    dominant = np.argmax(normalized, axis=0).astype(np.uint8)
    dominant[total <= 1e-8] = 255
    result = {"overlap_count": overlap, "dominant_camera": dominant,
              "valid_mask": (total > 1e-8).astype(np.uint8)*255}
    for i, name in enumerate(REQUIRED_CAMERAS):
        result[f"{name}_mask"] = masks[i].astype(np.uint8)*255
        result[f"{name}_weight"] = normalized[i].astype(np.float32)
        result[f"{name}_uv"] = uv_weights[name][0]
    return result


class AvmStitcher:
    def __init__(self):
        self._key = None
        self._maps = {}

    def stitch(self, scene: Scene, images: dict[str,np.ndarray]) -> AvmResult:
        missing = set(REQUIRED_CAMERAS)-images.keys()
        if missing or not set(REQUIRED_CAMERAS).issubset(scene.cameras):
            raise ValueError(f"AVM requires all front/rear/left/right images; missing {sorted(missing)}")
        key = repr((scene.vehicle.pose if scene.vehicle else None,scene.coordinate.ground_z,
                    scene.bev,[(n,scene.cameras[n]) for n in REQUIRED_CAMERAS]))
        if key != self._key:
            self._maps = {n:projection_maps(scene,n) for n in REQUIRED_CAMERAS}
            self._key = key
        h,w = scene.bev.height,scene.bev.width
        total,weights = np.zeros((h,w,3),np.float32),np.zeros((h,w),np.float32)
        for name in REQUIRED_CAMERAS:
            image = images[name]
            camera = scene.cameras[name]
            if image.shape != (camera.height,camera.width,3):
                raise ValueError(f"Image dimensions do not match calibration: {name}")
            uv,weight = self._maps[name]
            sampled = cv2.remap(image,uv[...,0],uv[...,1],cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
            total += sampled*weight[...,None]
            weights += weight
        coverage = weights>1e-6
        rgb = total/np.maximum(weights[...,None],1e-6)
        rgb[~coverage] = [16,20,28]
        # No synthetic top-down image is substituted for missing coverage.
        return AvmResult(np.clip(rgb,0,255).astype(np.uint8),coverage)
