"""Analytic ground scene and debug images for the camera-to-BEV geometry chain."""
from pathlib import Path
import json
import cv2
import numpy as np
from scene_manager.camera_models import camera_rays
from .projection import (REQUIRED_CAMERAS, AvmStitcher, ground_points,
                         projection_debug_maps, world_to_bev_pixel)


def ground_color(points):
    return np.clip(np.stack([(points[...,0]+10)*10, (points[...,1]+10)*10,
                             np.full(points.shape[:-1], 80)], axis=-1),0,255)


def synthetic_ground_images(scene, grid=False):
    images = {}
    for name in REQUIRED_CAMERAS:
        camera = scene.cameras[name]
        rays,valid = camera_rays(camera)
        t = scene.camera_optical_pose(name)
        directions = rays @ t[:3,:3].T
        depth = np.divide(scene.coordinate.ground_z-t[2,3],directions[...,2],
                          out=np.full(valid.shape,np.nan),where=np.abs(directions[...,2])>1e-9)
        points = t[:3,3]+np.nan_to_num(depth)[...,None]*directions
        rgb = ground_color(points)
        if grid:
            lines = (np.abs(points[...,0]-np.round(points[...,0]))<.025) | (np.abs(points[...,1]-np.round(points[...,1]))<.025)
            rgb[lines]=245
        rgb[(depth<=0)|~np.isfinite(depth)|~valid]=0
        images[name]=rgb.astype(np.uint8)
    return images


def export_avm_validation(scene, output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    images=synthetic_ground_images(scene)
    result=AvmStitcher().stitch(scene,images)
    expected=ground_color(ground_points(scene))
    errors=np.abs(result.rgb.astype(float)-expected)[result.coverage]
    grid=AvmStitcher().stitch(scene,synthetic_ground_images(scene,grid=True))
    debug=projection_debug_maps(scene)
    cv2.imwrite(str(output/"bev_grid.png"),cv2.cvtColor(grid.rgb,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output/"bev_coordinate_color.png"),cv2.cvtColor(result.rgb,cv2.COLOR_RGB2BGR))
    for name,array in debug.items():
        if name.endswith("_uv"):
            continue
        png=(np.clip(array*255,0,255).astype(np.uint8) if name.endswith("_weight") else array)
        cv2.imwrite(str(output/f"{name}.png"),png)
        if name.endswith("_weight") or name=="overlap_count":
            value=png if name.endswith("_weight") else (png.astype(float)*255/4).astype(np.uint8)
            cv2.imwrite(str(output/f"{name}_heatmap.png"),cv2.applyColorMap(value,cv2.COLORMAP_TURBO))
    np.savez_compressed(output/"projection_maps.npz",**debug)
    points=ground_points(scene)
    pixels,valid=world_to_bev_pixel(scene,points)
    uu,vv=np.meshgrid(np.arange(scene.bev.width),np.arange(scene.bev.height))
    pixel_error=float(np.max(np.abs(pixels-np.stack([uu,vv],axis=-1))))
    metrics={"ground_bev_roundtrip_max_px":pixel_error,
        "ground_color_median_error":float(np.median(errors)) if errors.size else None,
        "ground_color_p99_error":float(np.quantile(errors,.99)) if errors.size else None,
        "coverage":float(result.coverage.mean()),"all_grid_points_valid":bool(valid.all())}
    (output/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    return metrics
