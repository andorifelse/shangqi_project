"""Camera projection and inverse ray generation, shared by all sensor consumers.

Pixel centers use integer coordinates. Fisheye: OpenCV theta polynomial.
Ftheta: same normalized equidistant polynomial, optionally beyond 180 degrees.
Pinhole D: k1,k2,p1,p2,k3. Fisheye/ftheta D: k1,k2,k3,k4.
"""
from __future__ import annotations
import cv2
import numpy as np
from .models import Camera


def project(camera: Camera, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    shape = points.shape[:-1]
    p = points.reshape(-1,3)
    norm = np.linalg.norm(p,axis=1)
    theta = np.arctan2(np.linalg.norm(p[:,:2],axis=1),p[:,2])
    valid = (norm > 1e-9) & (theta <= camera.max_angle)
    if camera.model == "pinhole":
        uv,_ = cv2.projectPoints(p.reshape(-1,1,3), np.zeros(3),np.zeros(3),camera.matrix,np.array(camera.D))
        uv = uv.reshape(-1,2)
        valid &= p[:,2] > 1e-6
    else:
        rho = theta * (1 + sum(d*theta**(2*i+2) for i,d in enumerate(camera.D)))
        radial = np.linalg.norm(p[:,:2],axis=1)
        xy = p[:,:2] * np.divide(rho,radial,out=np.zeros_like(rho),where=radial>1e-12)[:,None]
        uv = xy * [camera.K[0],camera.K[4]] + [camera.K[2],camera.K[5]]
    valid &= np.isfinite(uv).all(axis=1)
    return uv.reshape(*shape,2), valid.reshape(shape)


def unproject(camera: Camera, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(pixels,dtype=np.float64)
    shape = pixels.shape[:-1]
    uv = pixels.reshape(-1,2)
    if camera.model == "pinhole":
        xy = cv2.undistortPoints(uv.reshape(-1,1,2),camera.matrix,np.array(camera.D)).reshape(-1,2)
        rays = np.column_stack([xy,np.ones(len(xy))])
        rays /= np.linalg.norm(rays,axis=1,keepdims=True)
        valid = np.arccos(np.clip(rays[:,2],-1,1)) <= camera.max_angle
    else:
        xy = (uv-[camera.K[2],camera.K[5]])/[camera.K[0],camera.K[4]]
        rho = np.linalg.norm(xy,axis=1)
        # Bisection stays within the validated monotonic calibration domain.
        lo,hi = np.zeros_like(rho),np.full_like(rho,camera.max_angle)
        for _ in range(40):
            theta = (lo+hi)*.5
            value = theta*(1+sum(d*theta**(2*i+2) for i,d in enumerate(camera.D)))
            lo = np.where(value<rho,theta,lo)
            hi = np.where(value>=rho,theta,hi)
        theta = (lo+hi)*.5
        radial_max = camera.max_angle*(1+sum(d*camera.max_angle**(2*i+2) for i,d in enumerate(camera.D)))
        valid = rho <= radial_max+1e-9
        factor = np.divide(np.sin(theta),rho,out=np.ones_like(rho),where=rho>1e-12)
        rays = np.column_stack([xy*factor[:,None],np.cos(theta)])
    # Validate inverse numerically, including strong pinhole distortion.
    reprojection, in_domain = project(camera,rays)
    valid &= in_domain & (np.linalg.norm(reprojection-uv,axis=1)<1e-3)
    return rays.reshape(*shape,3), valid.reshape(shape)


def camera_rays(camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    u,v = np.meshgrid(np.arange(camera.width),np.arange(camera.height))
    return unproject(camera,np.stack([u,v],axis=-1))
