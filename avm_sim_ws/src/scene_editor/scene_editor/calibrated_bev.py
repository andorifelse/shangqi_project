"""Stitch original capture pixels with estimated lens/poses; never re-render sensors."""
from pathlib import Path
import json
import cv2
import numpy as np
from scene_manager.models import Scene
from scene_manager.persistence import load_scene
from scene_manager.camera_models import camera_rays
from avm_stitcher.projection import REQUIRED_CAMERAS, projection_maps
from .marker_extrinsics import export_calibrated_scene
from .scene_pipeline import put, digest


def generate(directory, result, output=None):
    directory = Path(directory).resolve()
    output = Path(output) if output else directory/'bev'
    if not result.get('calibration_accepted'):
        raise ValueError('内外参全部通过后才能生成鸟瞰图')
    if not set(REQUIRED_CAMERAS).issubset(result.get('extrinsics', {})):
        return {'status': 'SKIPPED', 'reason': '鸟瞰拼接需要 front/rear/left/right 四路标定'}
    if output.exists():
        raise FileExistsError(f'鸟瞰输出已存在：{output}')
    request = json.loads((directory/'request.json').read_text(encoding='utf-8'))
    scene = Scene.from_dict(request['scene_snapshot'])
    images, sources = {}, {}
    for name in REQUIRED_CAMERAS:
        row = result['extrinsics'][name]
        path = (directory/row['image']).resolve()
        if directory not in path.parents or digest(path) != row['image_sha256']:
            raise ValueError(f'原始相机图片路径或哈希不匹配：{name}')
        bgr = cv2.imread(str(path))
        camera = scene.cameras[name]
        if bgr is None or bgr.shape != (camera.height, camera.width, 3):
            raise ValueError(f'原始相机图片尺寸不匹配：{name}')
        images[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        sources[name] = {'image': row['image'], 'sha256': row['image_sha256']}
    output.mkdir(parents=True)
    export_calibrated_scene(scene, result['results'], result['extrinsics'], output)
    estimated = load_scene(output/'calibrated_scene.yaml')
    def stitch(projection_scene, prefix):
        total = np.zeros((projection_scene.bev.height, projection_scene.bev.width, 3), np.float32)
        weights = np.zeros(total.shape[:2], np.float32)
        overlap = np.zeros(weights.shape, np.uint8)
        def save(name, rgb):
            if not cv2.imwrite(str(output/name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise OSError(output/name)
        for name in REQUIRED_CAMERAS:
            uv, weight = projection_maps(projection_scene, name)
            _, source_valid = camera_rays(scene.cameras[name])
            valid = cv2.remap(source_valid.astype(np.float32), uv[..., 0], uv[..., 1],
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) > .999
            weight *= valid
            sampled = cv2.remap(images[name], uv[..., 0], uv[..., 1], cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT)
            total += sampled*weight[..., None]
            weights += weight
            overlap += (weight > 1e-6).astype(np.uint8)
            preview = sampled.copy(); preview[weight <= 1e-6] = [16, 20, 28]
            save(f'{prefix}{name}_ground.png', preview)
            cv2.imwrite(str(output/f'{prefix}{name}_valid.png'), (weight > 1e-6).astype(np.uint8)*255)
        covered = weights > 1e-6
        rgb = np.clip(total/np.maximum(weights[..., None], 1e-6), 0, 255).astype(np.uint8)
        rgb[~covered] = [16, 20, 28]
        save(prefix+'stitched.png', rgb)
        cv2.imwrite(str(output/(prefix+'coverage.png')), covered.astype(np.uint8)*255)
        cv2.imwrite(str(output/(prefix+'overlap_count.png')), overlap)
        return rgb, covered, overlap
    rgb, covered, overlap = stitch(estimated, '')
    truth_rgb, truth_covered, _ = stitch(scene, 'truth_')
    comparison = np.concatenate([truth_rgb, rgb], axis=1)
    save_path = output/'comparison.png'
    if not cv2.imwrite(str(save_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)):
        raise OSError(save_path)
    metrics = {'status': 'GENERATED', 'image': str((output/'stitched.png').relative_to(directory)),
        'truth_image': str((output/'truth_stitched.png').relative_to(directory)),
        'comparison_image': str((output/'comparison.png').relative_to(directory)),
        'comparison_order': ['truth', 'calibrated'],
        'truth_coverage_fraction': float(truth_covered.mean()),
        'coverage_fraction': float(covered.mean()), 'overlap_fraction': float((overlap >= 2).mean()),
        'resolution_px': [estimated.bev.width, estimated.bev.height],
        'sources': sources, 'parameters': 'calibrated_parameters.json',
        'projection': 'estimated K/D and T_world_from_camera_optical; world horizontal ground; weighted blending',
        'notice': '原始固定场景图片，未用估计参数重新渲染。覆盖率不是精度指标；车辆与立体物体受地面假设影响会拉伸，未做深度遮挡或接缝优化。'}
    put(output/'metrics.json', metrics)
    return metrics
