"""ID-resolved external poses from public metric marker geometry and RGB."""
from pathlib import Path
import json
import cv2
import numpy as np
from scene_manager.transforms import pose_matrix, transform_points, inverse
from .fisheye_solver import fit_pose, project
from .scene_pipeline import put, digest


def public_geometry(scene, filename=''):
    """Find the asset package's declared marker corners, applying current edits.

    World-baked PLY package coordinates are asset-local coordinates on import.
    Camera poses, K/D and rendered corner projections are never read here.
    """
    paths = [Path(filename).expanduser().resolve()] if filename else sorted({
        parent/'几何与来源说明.json' for board in scene.boards.values() if board.asset
        for parent in Path(board.asset).resolve().parents if parent != Path('/')
        if (parent/'几何与来源说明.json').is_file()})
    markers = {}
    dictionary = None
    sources = []
    for path in paths:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not data.get('markers'):
            continue
        if dictionary is not None and dictionary != data['marker_dictionary']:
            raise ValueError('外参编码板使用了不同字典')
        dictionary = data['marker_dictionary']
        for row in data['markers']:
            asset = (path.parent/row['file']).resolve()
            matches = [b for b in scene.boards.values() if b.asset and Path(b.asset).resolve() == asset]
            if not matches:
                continue
            if len(matches) != 1 or str(row['id']) in markers:
                raise ValueError('外参编码 ID 或资产重复')
            b = matches[0]
            xyz = np.asarray(row['corners_world_m'], float)
            if xyz.shape != (4, 3) or not np.isfinite(xyz).all():
                raise ValueError('编码板公开角点无效')
            xyz = transform_points(pose_matrix(b.pose)@pose_matrix(b.asset_pose), xyz*b.asset_scale)
            markers[str(row['id'])] = xyz.tolist()
        sources.append({'path': str(path), 'sha256': digest(path)})
    if len(markers) < 3 or dictionary is None or not hasattr(cv2.aruco, dictionary):
        raise ValueError('内外参联合流程需要至少三个已知 ID 的编码板及公开三维角点；请设置编码板几何 JSON')
    return {'coordinate_frame': 'scene_world', 'units': 'm', 'dictionary': dictionary,
        'markers': markers, 'sources': sources}


def detect_markers(rgb, k, d, dictionary):
    """Decode actual RGB; rectify views using the estimated lens if necessary."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), params)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    found = {}
    def collect(image, mapping=None):
        corners, identities, _ = detector.detectMarkers(image)
        for identity, points in zip([] if identities is None else identities.ravel(), corners):
            points = points.reshape(4, 2).astype(np.float64)
            if mapping is not None:
                points = mapping(points)
            # Prefer the least distorted virtual view when raw detection fails.
            found.setdefault(str(int(identity)), points.tolist())
    collect(gray)
    h, w = gray.shape
    size = 900
    focal = size*.5
    virtual_k = np.array([[focal, 0., (size-1)/2], [0., focal, (size-1)/2], [0., 0., 1.]])
    from scipy.spatial.transform import Rotation
    # Overlapping perspective views preserve marker ID/corner orientation.
    for yaw, pitch in ((0, 0), (-45, 0), (45, 0), (0, -45), (0, 45)):
        rotation = Rotation.from_euler('yx', [yaw, pitch], degrees=True).as_matrix()
        mx, my = cv2.fisheye.initUndistortRectifyMap(k, d, rotation, virtual_k,
            (size, size), cv2.CV_32FC1)
        rectified = cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderValue=127)
        def mapping(points):
            rays = np.column_stack([(points[:, 0]-virtual_k[0, 2])/focal,
                (points[:, 1]-virtual_k[1, 2])/focal, np.ones(4)])@rotation
            return project(rays, np.zeros(6), k, d)
        collect(rectified, mapping)
    return found


def fit_extrinsics(document):
    """Known-ID PnP with withheld corners and a fixed independently fitted lens."""
    allowed = {'model', 'K', 'D', 'geometry', 'observations'}
    if set(document) != allowed or document['model'] != 'fisheye':
        raise ValueError('外参求解仅接受已标定内参、公开编码几何和图像观测')
    k = np.asarray(document['K'], float); d = np.asarray(document['D'], float)
    geometry = document['geometry']['markers']; observations = document['observations']
    identities = sorted(set(geometry)&set(observations), key=int)
    if len(identities) < 3:
        return {'status': 'REJECTED', 'extrinsics_accepted': False, 'marker_ids': identities,
            'rejection_reasons': ['至少需要三个可解码且已知几何的编码板']}
    objects = np.concatenate([geometry[i] for i in identities]).astype(np.float64)
    pixels = np.concatenate([observations[i] for i in identities]).astype(np.float64)
    if pixels.shape != (len(objects), 2) or not np.isfinite(pixels).all():
        raise ValueError('外参角点无效')
    keep = np.arange(len(objects)) % 4 != 3
    pose_validation = fit_pose(objects, pixels, k, d, keep)
    held = float(np.sqrt(np.mean(np.sum((project(objects, pose_validation, k, d)[~keep]-pixels[~keep])**2, axis=1))))
    pose = fit_pose(objects, pixels, k, d)
    errors = np.linalg.norm(project(objects, pose, k, d)-pixels, axis=1)
    rms = float(np.sqrt(np.mean(errors**2)))
    r = cv2.Rodrigues(pose[:3])[0]
    transform = np.eye(4); transform[:3, :3] = r; transform[:3, 3] = pose[3:]
    reasons = []
    if rms > 1.: reasons.append('编码角点 RMS 超过 1 px')
    if held > 1.5: reasons.append('留出编码角点 RMS 超过 1.5 px')
    return {'status': 'CHECKED' if not reasons else 'REJECTED', 'extrinsics_accepted': not reasons,
        'coordinate_frame': 'scene_world', 'R_w2c': r.tolist(), 'tvec': pose[3:].tolist(),
        'T_world_from_camera_optical': inverse(transform).tolist(),
        'camera_center_world_m': (-r.T@pose[3:]).tolist(), 'rms_px': rms,
        'heldout_rmse_px': held, 'marker_ids': identities, 'corner_count': len(objects),
        'max_corner_error_px': float(errors.max()), 'rejection_reasons': reasons,
        'protocol': 'decoded marker IDs; fixed estimated K/D; fit 3 corners per marker and validate fourth; final fit all corners'}


def run(scene, request, directory, intrinsics, pipeline, progress):
    """Capture fixed scene after removing only the moving intrinsic board."""
    from copy import deepcopy
    directory = Path(directory)
    geometry = request['marker_geometry']
    for source in geometry['sources']:
        if digest(source['path']) != source['sha256']:
            raise ValueError('编码板几何文件在启动后发生变化')
    output = directory/'extrinsics'; output.mkdir()
    put(output/'public_geometry.json', geometry)
    results = {}
    for name in request['cameras']:
        if not intrinsics[name]['intrinsics_accepted']:
            results[name] = {'status': 'SKIPPED', 'extrinsics_accepted': False,
                'rejection_reasons': ['内参未通过，禁止继续外参']}
            continue
        progress(f'编码板外参采集与求解 {name}')
        sample = deepcopy(scene); del sample.boards[request['intrinsic_board']]
        sample.cameras = {name: sample.cameras[name]}
        c = sample.cameras[name]; w, h = c.width, c.height; factor = request['supersample']
        c.width *= factor; c.height *= factor
        for i in (0, 4): c.K[i] *= factor
        for i in (2, 5): c.K[i] = (c.K[i]+.5)*factor-.5
        rgb = pipeline.render(sample)[name].rgb
        if factor > 1: rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        path = output/f'{name}.png'
        if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)): raise OSError(path)
        lens = intrinsics[name]
        observations = detect_markers(rgb, np.array(lens['K']), np.array(lens['D']), geometry['dictionary'])
        public = {'model': 'fisheye', 'K': lens['K'], 'D': lens['D'],
            'geometry': geometry, 'observations': observations}
        put(directory/'solver_inputs'/f'{name}_extrinsics.json', public)
        result = fit_extrinsics(public)
        result.update(image=str(path.relative_to(directory)), image_sha256=digest(path))
        results[name] = result
        put(output/f'{name}.json', result)
        if result.get('T_world_from_camera_optical'):
            estimated = np.array(result['T_world_from_camera_optical'])
            truth = scene.camera_optical_pose(name)
            relative = estimated[:3, :3].T@truth[:3, :3]
            put(output/f'{name}_evaluation.json', {'usage': 'post-fit evaluation only',
                'translation_error_m': float(np.linalg.norm(estimated[:3, 3]-truth[:3, 3])),
                'rotation_error_deg': float(np.rad2deg(np.arccos(np.clip((np.trace(relative)-1)/2, -1., 1.))))})
    return results


def export_calibrated_scene(scene, intrinsics, extrinsics, directory):
    """Write a separate loadable scene only after every requested stage passes."""
    from copy import deepcopy
    from scene_manager.transforms import matrix_pose, T_LINK_OPTICAL
    from scene_manager.persistence import save_scene
    if not intrinsics or set(intrinsics) != set(extrinsics) or not all(
            intrinsics[n]['intrinsics_accepted'] and extrinsics[n]['extrinsics_accepted'] for n in intrinsics):
        raise ValueError('只允许导出所有请求相机内外参均通过的场景')
    estimated = deepcopy(scene)
    parameters = {}
    for name, lens in intrinsics.items():
        camera = estimated.cameras[name]
        camera.K = np.array(lens['K']).ravel().tolist()
        camera.D = list(lens['D'])
        optical = np.array(extrinsics[name]['T_world_from_camera_optical'])
        link_world = optical@inverse(T_LINK_OPTICAL)
        parent = pose_matrix(estimated.vehicle.pose) if camera.parent == 'base_link' else np.eye(4)
        camera.pose = matrix_pose(inverse(parent)@link_world)
        parameters[name] = {'model': camera.model, 'resolution_px': [camera.width, camera.height],
            'K': lens['K'], 'D': lens['D'], 'parent': camera.parent,
            'camera_link_pose_in_parent': camera.pose,
            'T_world_from_camera_optical': optical.tolist()}
    estimated.validate()
    directory = Path(directory)
    save_scene(estimated, directory/'calibrated_scene.yaml')
    put(directory/'calibrated_parameters.json', {'coordinate_frame': 'scene_world',
        'scope': 'synthetic scene observed rays', 'cameras': parameters})
    return {'scene': 'calibrated_scene.yaml', 'parameters': 'calibrated_parameters.json'}
