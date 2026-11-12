"""Multi-tilt KB4 intrinsics with optional ID-resolved marker extrinsics.

Capture truth never enters either solver.
"""
from copy import deepcopy
from collections import Counter
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scene_manager.board_geometry import board_geometry
from scene_manager.camera_models import world_to_pixel, unproject
from scene_manager.transforms import pose_matrix, matrix_pose, transform_points
from .scene_pipeline import put, digest, region_grids
from .capture_sampling import IntrinsicSampler

ALGORITHM = 'fisheye_multitilt'


def normalize(request, scene):
    out = deepcopy(request)
    names = out.get('cameras') or sorted(scene.cameras)
    if not isinstance(names, list) or not names or len(set(names)) != len(names) or set(names)-scene.cameras.keys():
        raise ValueError('请选择当前场景中存在且不重复的相机')
    if any(scene.cameras[n].model != 'fisheye' for n in names):
        raise ValueError('多倾角鱼眼内参仅支持 fisheye；请单独选择鱼眼相机')
    board_id = out.get('intrinsic_board')
    if board_id not in scene.boards:
        raise ValueError('请选择内参运动小板')
    document = board_geometry(scene.boards[board_id])
    grids = region_grids(document)
    if len(grids) != 1:
        raise ValueError('内参板必须是已知几何的单一完整矩形，不能选择环车条带')
    grid = next(iter(grids.values()))
    if not np.allclose(grid['grid'][:, :, 2], 0., atol=1e-7):
        raise ValueError('请将内参板公开几何对齐到 board 局部 Z=0 平面')
    for key, lo, hi, default in [('views', 12, 100, 24), ('holdout', 3, 30, 4),
            ('seed', 0, 2**32-1, 20261111), ('supersample', 1, 4, 2), ('max_attempts', 1, 50, 8)]:
        value = out.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
            raise ValueError(f'{key} 必须为 {lo}–{hi} 的整数')
        out[key] = value
    for key, lo, hi, default in [('tilt_deg', 20., 55., 40.), ('distance_min', .4, 10., 1.),
            ('distance_max', .4, 10., 2.4)]:
        value = out.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'{key} 必须在 {lo}–{hi} 范围内')
        out[key] = float(value)
    if out['distance_min'] >= out['distance_max']:
        raise ValueError('最大采集距离必须大于最小采集距离')
    if out.get('backend', 'gsplat') not in ('gsplat', 'cpu'):
        raise ValueError('未知渲染后端')
    out.update(algorithm=ALGORITHM, cameras=names, backend=out.get('backend', 'gsplat'),
        intrinsic_geometry=document, pattern_size=[grid['columns'], grid['rows']], square_size_m=grid['pitch'])
    if not isinstance(out.get('run_extrinsics', False), bool):
        raise ValueError('run_extrinsics 必须为布尔值')
    if out.get('run_extrinsics', False):
        from .marker_extrinsics import public_geometry
        out['marker_geometry'] = public_geometry(scene, out.get('marker_geometry_file', ''))
    return out


def detect_target(rgb, pattern, boundary):
    """Exclude pixels outside the projected plate; return image-measured corners."""
    boundary = np.asarray(boundary, np.float32)
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.rint(boundary).astype(np.int32)], 255)
    x, y, width, height = cv2.boundingRect(boundary)
    x0, y0 = max(0, x-24), max(0, y-24)
    x1, y1 = min(w, x+width+24), min(h, y+height+24)
    if x1-x0 < 16 or y1-y0 < 16:
        return None
    gray = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    gray = np.where(mask[y0:y1, x0:x1], gray, 240).astype(np.uint8)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    corners = None
    # SB may report a complete grid with one point on the outer border. Check
    # smooth adjacent grid steps before accepting it, then retry without image
    # normalization (which can amplify the wrong edge) or with the classic path.
    for detector in ('sb_normalized', 'sb', 'classic'):
        if detector == 'classic':
            found, candidate = cv2.findChessboardCorners(gray, tuple(pattern),
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if found:
                candidate = cv2.cornerSubPix(gray, candidate, (5, 5), (-1, -1), (3, 50, .001))
        else:
            used_flags = flags if detector == 'sb_normalized' else flags & ~cv2.CALIB_CB_NORMALIZE_IMAGE
            found, candidate = cv2.findChessboardCornersSB(gray, tuple(pattern), flags=used_flags)
        if found and candidate is not None and regular_grid(candidate, pattern):
            corners = candidate
            break
    if corners is None:
        return None
    points = corners.reshape(-1, 2).astype(np.float64) + [x0, y0]
    if len(points) != pattern[0]*pattern[1] or not np.isfinite(points).all():
        return None
    pixels = np.rint(points).astype(int)
    if not ((pixels >= [0, 0]).all() and (pixels < [w, h]).all()):
        return None
    if not mask[pixels[:, 1], pixels[:, 0]].all():
        return None
    return points


def regular_grid(points, pattern):
    """Reject abrupt grid kinks using measured neighbors, without projected GT."""
    grid = np.asarray(points, float).reshape(pattern[1], pattern[0], 2)
    if not np.isfinite(grid).all():
        return False
    for axis in (0, 1):
        steps = np.diff(grid, axis=axis)
        a = np.take(steps, range(steps.shape[axis]-1), axis=axis)
        b = np.take(steps, range(1, steps.shape[axis]), axis=axis)
        scale = .5*(np.linalg.norm(a, axis=-1)+np.linalg.norm(b, axis=-1))
        if (scale < 2.).any() or (np.linalg.norm(b-a, axis=-1)/scale > .6).any():
            return False
    return True


def propose(scene, name, board, request, rng, depth, observations):
    """Acquisition-only candidate ranking with diverse positions/normals/ranges."""
    checker = IntrinsicSampler(scene, name, board, {**request, 'sampling': 'around_pose',
        'spread_x': 0., 'spread_y': 0., 'yaw_range_deg': 0.}, rng, depth)
    camera = scene.cameras[name]
    optical = scene.camera_optical_pose(name)
    grid = checker.grid.reshape(-1, 3)
    centre = grid.mean(axis=0)
    boundary = np.concatenate([np.linspace(checker.outer[i], checker.outer[(i+1) % 4], 12, endpoint=False)
        for i in range(4)])
    rejected = Counter()
    candidates = []
    for _ in range(80):
        pixel = rng.uniform([.16*camera.width, .16*camera.height], [.84*camera.width, .76*camera.height])
        ray, ok = unproject(camera, pixel)
        if not bool(ok):
            rejected['invalid_ray'] += 1
            continue
        axis_x = np.cross([0., 1., 0.], ray)
        axis_x /= np.linalg.norm(axis_x)
        axis_y = np.cross(-ray, axis_x)
        tilt = np.deg2rad(request['tilt_deg'])
        angles = rng.uniform([-tilt, -tilt, -.7], [tilt, tilt, .7])
        t = np.eye(4)
        t[:3, :3] = np.column_stack([axis_x, axis_y, -ray]) @ Rotation.from_euler('xyz', angles).as_matrix()
        distance = rng.uniform(request['distance_min'], request['distance_max'])
        t[:3, 3] = ray*distance - t[:3, :3]@centre
        world = optical@t
        world_boundary = transform_points(world, boundary)
        if world_boundary[:, 2].min() < scene.coordinate.ground_z+.015:
            rejected['below_ground'] += 1
            continue
        pose = matrix_pose(world)
        candidate, reason = checker.evaluate(pose)
        if candidate is None:
            rejected[reason] += 1
            continue
        edge_uv, valid = world_to_pixel(scene, name, world_boundary)
        if not valid.all() or not (edge_uv >= [5, 5]).all() or not (edge_uv < [camera.width-6, camera.height-6]).all():
            rejected['clipped_boundary'] += 1
            continue
        if candidate['min_square_px'] < 10.:
            rejected['small_squares'] += 1
            continue
        normal = world[:3, 2]
        screen = candidate['centre']/[camera.width, camera.height]
        feature = np.r_[screen*3., normal, distance/(request['distance_max']-request['distance_min'])]
        separation = min((np.linalg.norm(feature-o['feature']) for o in observations), default=1.)
        if separation < .12:
            rejected['duplicate_pose'] += 1
            continue
        candidate.update(boundary=edge_uv, pose=pose, normal=normal, distance=distance, feature=feature)
        candidate['score'] = min(candidate['min_square_px'], 30.)/30. + 2.*min(separation, 2.)
        candidates.append(candidate)
    return (max(candidates, key=lambda c: c['score']) if candidates else None), rejected


def heldout_error(objects, pixels, k, d):
    """Fit only pose on 4/5 held-out corners; score the untouched fifth."""
    pixels = np.asarray(pixels, np.float64)
    keep = np.arange(len(objects)) % 5 != 0
    from .fisheye_solver import fit_pose, project
    pose = fit_pose(objects, pixels, k, d, keep)
    projected = project(objects, pose, k, d)
    return float(np.sqrt(np.mean(np.sum((projected[~keep]-pixels[~keep])**2, axis=1))))


def fit_intrinsics(document):
    """Public observations only; independent pose per image, fixed zero skew."""
    allowed = {'model', 'resolution_px', 'object_points_m', 'observations'}
    if set(document)-allowed:
        raise ValueError('内参求解输入仅允许图像观测、棋盘几何和分辨率')
    if document.get('model') != 'fisheye':
        raise ValueError('只支持 KB4 fisheye')
    objects = np.asarray(document['object_points_m'], np.float64)
    w, h = document['resolution_px']
    if any(isinstance(v, bool) or not isinstance(v, int) or not 16 <= v <= 4096 for v in (w, h)):
        raise ValueError('图片分辨率无效')
    observations = document['observations']
    if objects.ndim != 2 or objects.shape[1] != 3 or len(objects) < 9 or not np.isfinite(objects).all():
        raise ValueError('无效的公开棋盘几何')
    if not np.allclose(objects[:, 2], 0., atol=1e-7):
        raise ValueError('棋盘角点必须在局部 Z=0 平面')
    if any(set(row) != {'image', 'split', 'corners_px'} for row in observations):
        raise ValueError('观测不允许包含相机或板姿态真值')
    if any(row['split'] not in ('train', 'holdout') for row in observations):
        raise ValueError('未知训练/留出划分')
    for row in observations:
        points = np.asarray(row['corners_px'], np.float64)
        if points.shape != (len(objects), 2) or not np.isfinite(points).all():
            raise ValueError('无效的图片角点')
    train = [r for r in observations if r['split'] == 'train']
    holdout = [r for r in observations if r['split'] == 'holdout']
    if len(train) < 12 or len(holdout) < 3:
        raise ValueError('至少需要 12 张训练和 3 张独立留出图片')
    image_points = [np.array(r['corners_px'], np.float64).reshape(1, -1, 2) for r in train]
    obj = [objects.reshape(1, -1, 3) for _ in train]
    from gs_renderer.calibration import cv_fisheye_flag
    flags = sum(cv_fisheye_flag(n) for n in ('CALIB_USE_INTRINSIC_GUESS', 'CALIB_RECOMPUTE_EXTRINSIC', 'CALIB_FIX_SKEW'))
    candidates, failures = [], []
    for fraction in (.25, 1/np.pi, .45):
        k0 = np.array([[w*fraction, 0., (w-1)/2], [0., w*fraction, (h-1)/2], [0., 0., 1.]])
        try:
            rms, k, d, rotations, translations = cv2.fisheye.calibrate(obj, image_points, (w, h), k0,
                np.zeros((4, 1)), flags=flags, criteria=(3, 150, 1e-9))
            from .fisheye_solver import refine_intrinsics
            rms, k, d, rotations, translations = refine_intrinsics(objects,
                [r.reshape(-1, 2) for r in image_points], k, d)
            if not np.isfinite(k).all() or not np.isfinite(d).all() or not np.isfinite(rms):
                raise ValueError('非有限的内参结果')
            if min(k[0, 0], k[1, 1]) <= 0:
                raise ValueError('焦距必须为正')
            normalized = np.concatenate([(r.reshape(-1, 2)-k[:2, 2])/[k[0, 0], k[1, 1]] for r in image_points])
            radius = np.linalg.norm(normalized, axis=1).max()
            theta = np.linspace(0., 1.55, 1000)
            rho = theta*(1+sum(float(c)*theta**(2*i+2) for i, c in enumerate(d.ravel())))
            derivative = 1+sum((2*i+3)*float(c)*theta**(2*i+2) for i, c in enumerate(d.ravel()))
            # Check the entire estimated angular domain used by the observations.
            crossings = np.flatnonzero(rho >= radius)
            if not len(crossings) or derivative[:crossings[0]+1].min() <= .01:
                raise ValueError('观测范围内鱼眼映射不单调或不可逆')
            normals = np.array([cv2.Rodrigues(rv)[0][:, 2] for rv in rotations])
            normal_span = float(np.rad2deg(np.arccos(np.clip(np.abs(normals@normals.T), 0., 1.))).max())
            centres = np.array([tv.ravel()+cv2.Rodrigues(rv)[0]@objects.mean(axis=0) for rv, tv in zip(rotations, translations)])
            distances = np.linalg.norm(centres, axis=1)
            candidates.append({'rms': float(rms), 'K': k, 'D': d, 'normal_span_deg': normal_span,
                'distance_ratio': float(distances.max()/distances.min()), 'start_fraction': fraction,
                'observed_theta_max_rad': float(theta[crossings[0]])})
        except (cv2.error, ValueError) as exc:
            failures.append({'start_fraction': fraction, 'error': str(exc)})
    if not candidates:
        raise RuntimeError('多倾角鱼眼求解失败：'+str(failures))
    best = min(candidates, key=lambda c: c['rms'])
    held = []
    for row in holdout:
        try:
            held.append({'image': row['image'], 'rmse_px': heldout_error(objects, row['corners_px'], best['K'], best['D'])})
        except (cv2.error, ValueError) as exc:
            held.append({'image': row['image'], 'error': str(exc)})
    ambiguity = max(float(np.linalg.norm([c['K'][0, 0]-best['K'][0, 0], c['K'][1, 1]-best['K'][1, 1]])/
        np.linalg.norm([best['K'][0, 0], best['K'][1, 1]])) for c in candidates)
    reasons = []
    if best['rms'] > 1.: reasons.append('训练 RMS 超过 1 px')
    if any('error' in row or row.get('rmse_px', np.inf) > 1.5 for row in held): reasons.append('留出角点评估超过 1.5 px 或失败')
    if best['normal_span_deg'] < 20.: reasons.append('估计板法向变化不足 20°')
    if best['distance_ratio'] < 1.15: reasons.append('估计采集距离变化不足')
    if ambiguity > .03: reasons.append('多初值焦距结果差异超过 3%')
    return {'status': 'CHECKED' if not reasons else 'REJECTED', 'intrinsics_accepted': not reasons,
        'calibration_accepted': False, 'model': 'fisheye', 'K': best['K'].tolist(), 'D': best['D'].ravel().tolist(),
        'rms_px': best['rms'], 'heldout': held, 'heldout_max_frame_rmse_px': max((r.get('rmse_px', 1e30) for r in held), default=None),
        'normal_span_deg': best['normal_span_deg'], 'distance_ratio': best['distance_ratio'],
        'observed_theta_max_rad': best['observed_theta_max_rad'], 'multistart_focal_spread': ambiguity, 'rejection_reasons': reasons,
        'initialization_failures': failures, 'training_complete': len(train), 'heldout_complete': len(holdout),
        'candidate_count': len(candidates), 'initialization': 'image dimensions only, zero D; zero skew fixed',
        'heldout_protocol': 'K/D fixed; pose fit on 4/5 corners; score disjoint 1/5 corners; no heldout intrinsic fitting',
        'scope': 'observed rays only; no whole-FOV accuracy or external pose acceptance'}


def run(scene, request, directory, progress):
    from gs_renderer.pipeline import SensorPipeline
    directory = Path(directory)
    capture = directory/'capture'; capture.mkdir()
    inputs = directory/'solver_inputs'; inputs.mkdir()
    output = directory/'multitilt'; output.mkdir()
    pipeline = SensorPipeline(request['backend'])
    small_id = request['intrinsic_board']; board = scene.boards[small_id]
    corners = sorted(request['intrinsic_geometry']['corners'], key=lambda r: (r['grid_row'], r['grid_col']))
    objects = np.array([r['xyz_m'] for r in corners])
    truth_rows, records, attempts, results = [], [], [], {}
    started = time.perf_counter()
    total = (request['views']+request['holdout'])*len(request['cameras'])
    done = 0
    def snapshot():
        put(capture/'manifest.json', {'records': records, 'complete': done == total, 'camera_motion': False,
            'other_boards_preserved': True, 'intrinsic_board': small_id, 'attempts': attempts})
        put(capture/'render_truth.json', {'usage': 'capture/audit only; never solver input', 'board_poses': truth_rows})
    for name in request['cameras']:
        camera = scene.cameras[name]
        progress(f'准备多倾角 {name} 静态深度', done=done, total=total)
        sample = deepcopy(scene); sample.cameras = {name: sample.cameras[name]}; del sample.boards[small_id]
        background = pipeline.render(sample)[name]
        depth = background.depth
        del background
        seed = int.from_bytes(hashlib.sha256(f"{request['seed']}:{name}".encode()).digest()[:8], 'little')
        rng = np.random.default_rng(seed)
        observations, accepted = [], []
        rejections = Counter()
        for attempt in range((request['views']+request['holdout'])*request['max_attempts']):
            t0 = time.perf_counter()
            candidate, rejected = propose(scene, name, board, request, rng, depth, accepted)
            rejections.update(rejected)
            if candidate is None:
                attempts.append({'camera': name, 'reason': 'no_feasible_pose', 'prefilter_rejections': dict(rejections)})
                snapshot()
                raise RuntimeError(f'{name} 未找到可见的多倾角板姿态，请检查遮挡及距离范围')
            progress(f'多倾角采集 {name} {len(observations)}/{request["views"]+request["holdout"]} · 渲染', done=done, total=total)
            sample = deepcopy(scene); sample.cameras = {name: deepcopy(camera)}
            sample.boards[small_id].pose = candidate['pose']
            factor = request['supersample']; scaled = sample.cameras[name]
            scaled.width *= factor; scaled.height *= factor
            for i in (0, 4): scaled.K[i] *= factor
            for i in (2, 5): scaled.K[i] = (scaled.K[i]+.5)*factor-.5
            render_start = time.perf_counter()
            rgb = pipeline.render(sample)[name].rgb
            if factor > 1: rgb = cv2.resize(rgb, (camera.width, camera.height), interpolation=cv2.INTER_AREA)
            render_s = time.perf_counter()-render_start
            detection_start=time.perf_counter()
            pixels = detect_target(rgb, request['pattern_size'], candidate['boundary'])
            detection_s=time.perf_counter()-detection_start
            row = {'camera': name, 'attempt': attempt+1, 'render_s': render_s, 'elapsed_s': time.perf_counter()-t0,
                'reason': 'accepted' if pixels is not None else 'target_detection_failed', 'detection_s': detection_s}
            attempts.append(row)
            truth_row={'camera': name, 'attempt': attempt+1, 'board_pose': candidate['pose'],
                'normal_world': candidate['normal'].tolist(), 'distance_m': candidate['distance'],
                'roi_boundary_px': candidate['boundary'].tolist(), 'reason': row['reason']}
            truth_rows.append(truth_row)
            if pixels is not None:
                index = len(observations)
                split = 'train' if index < request['views'] else 'holdout'
                relative = Path('capture')/name/f'view_{index:04d}.png'; path = directory/relative
                path.parent.mkdir(exist_ok=True, parents=True)
                if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                    raise OSError(f'图片写入失败：{path}')
                records.append({'camera': name, 'image': str(relative), 'split': split, 'sha256': digest(path)})
                observations.append({'image': str(relative), 'split': split, 'corners_px': pixels.tolist()})
                truth_row['image']=str(relative)
                accepted.append(candidate); done += 1
                progress(f'多倾角采集 {name} {len(observations)}/{request["views"]+request["holdout"]}', done=done, total=total)
            else:
                rejected_dir = capture/'rejected'/name; rejected_dir.mkdir(exist_ok=True, parents=True)
                rejected_path=rejected_dir/f'{attempt:04d}.png'
                if not cv2.imwrite(str(rejected_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                    raise OSError(f'图片写入失败：{rejected_path}')
                truth_row['image']=str(rejected_path.relative_to(directory))
            snapshot()
            if len(observations) == request['views']+request['holdout']: break
        public = {'model': 'fisheye', 'resolution_px': [camera.width, camera.height],
            'object_points_m': objects.tolist(), 'observations': observations}
        put(inputs/f'{name}.json', public)
        if len(observations) < request['views']+request['holdout']:
            raise RuntimeError(f'{name} 仅采到 {len(observations)} 张多倾角图片；拒绝原因已保存')
        progress(f'多倾角独立板姿态求解 {name}', done=done, total=total)
        result = fit_intrinsics(public)
        results[name] = result
        put(output/f'{name}.json', result)
        # Independent evaluation after estimation, never used to select the candidate.
        k = np.array(result['K']); d = np.array(result['D'])
        evaluation = {'usage': 'post-fit evaluation only', 'K_est_minus_truth': (k-camera.matrix).tolist(),
            'D_est_minus_truth': (d-np.array(camera.D)).tolist(),
            'focal_relative_percent': float(100*np.linalg.norm([k[0, 0]-camera.K[0], k[1, 1]-camera.K[4]])/
                np.linalg.norm([camera.K[0], camera.K[4]]))}
        put(output/f'{name}_evaluation.json', evaluation)
    asset_file=directory/'logs/assets.json'
    if asset_file.exists():
        from avm_bringup.experiments import file_sha256
        for info in json.loads(asset_file.read_text()).values():
            if file_sha256(info['path']) != info['sha256']:
                raise RuntimeError('采集期间资产文件发生变化：'+info['path'])
    # Intrinsic target is absent from the downstream copy; original/UI untouched.
    subsequent = deepcopy(scene); del subsequent.boards[small_id]
    put(output/'downstream_scene.json', subsequent.to_dict())
    put(capture/'timing.json', {'elapsed_s': time.perf_counter()-started, 'attempts': len(attempts), 'accepted': done})
    accepted_intrinsics = all(r['intrinsics_accepted'] for r in results.values())
    payload = {'status': 'complete' if accepted_intrinsics else 'rejected',
        'intrinsics_accepted': accepted_intrinsics, 'results': results,
        'scope': 'intrinsics_only', 'input_notice': '多倾角鱼眼内参；每张独立板姿态。未运行外参或 BEV，未修改相机。',
        'summary': [{'kind': 'intrinsics', 'model': 'fisheye', 'camera': n, **r} for n, r in results.items()]}
    if request.get('run_extrinsics', False):
        from .marker_extrinsics import run as run_extrinsics
        external = run_extrinsics(scene, request, directory, results, pipeline, progress)
        accepted_external = all(r['extrinsics_accepted'] for r in external.values())
        payload.update(extrinsics=external, extrinsics_accepted=accepted_external,
            calibration_accepted=accepted_intrinsics and accepted_external,
            status='complete' if accepted_intrinsics and accepted_external else 'rejected',
            scope='intrinsics_and_extrinsics',
            input_notice='多倾角内参 + 已知 ID 编码板外参；通过表示当前合成场景观测范围内数值验收通过。未运行 BEV，未修改 GUI 相机。')
        payload['summary'].extend({'kind': 'extrinsics', 'model': 'fisheye', 'camera': n, **r} for n, r in external.items())
        if payload['calibration_accepted']:
            from .marker_extrinsics import export_calibrated_scene
            payload['exports'] = export_calibrated_scene(scene, results, external, directory)
    return payload
