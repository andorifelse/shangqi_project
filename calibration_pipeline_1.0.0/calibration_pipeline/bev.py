"""Portable ground-plane BEV projection and conditional-candidate preparation.

Image pixels are zero based; camera poses explicitly map world to camera.
Public writing operations default to dry-run and never use camera truth.
"""
from __future__ import annotations

import copy
import math
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


PALETTE = [(232, 146, 42), (66, 187, 239), (80, 190, 79),
           (166, 112, 228), (233, 101, 169), (61, 210, 210)]
METHODS = ("hard", "feather", "graphcut_local")


def first_positive_root(coefficients):
    roots = np.polynomial.polynomial.polyroots(coefficients)
    positive = [float(r.real) for r in roots
                if abs(r.imag) < 1e-8 and r.real > 1e-12]
    return min(positive) if positive else float("inf")


def project_camera(points, K, D, model, theta_max_deg=89.9):
    """Project camera-space points and return an explicit lens-domain mask."""
    p, K, D = np.asarray(points, float), np.asarray(K, float), np.asarray(D, float).reshape(-1)
    if p.ndim < 1 or p.shape[-1] != 3:
        raise ValueError("points must have a final camera-coordinate dimension of three")
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("Need a finite K with positive focal lengths")
    if not np.allclose(K[2], [0, 0, 1]) or abs(K[1, 0]) > 1e-12:
        raise ValueError("Unsupported intrinsic matrix")
    if model not in ("pinhole", "fisheye") or len(D) != (5 if model == "pinhole" else 4):
        raise ValueError("Need Brown5 pinhole or theta4 fisheye distortion")
    if not np.isfinite(D).all():
        raise ValueError("Distortion coefficients must be finite")
    if model == "fisheye" and (not np.isfinite(theta_max_deg) or not 0 < theta_max_deg <= 90):
        raise ValueError("theta_max_deg must be finite and in (0,90]")
    valid = np.isfinite(p).all(axis=-1) & (p[..., 2] > 1e-8)
    depth = np.where(valid, p[..., 2], 1.)
    x, y = np.where(valid, p[..., 0], 0) / depth, np.where(valid, p[..., 1], 0) / depth
    r2 = x*x + y*y
    if model == "pinhole":
        k1, k2, p1, p2, k3 = D
        radial = 1 + k1*r2 + k2*r2*r2 + k3*r2**3
        xd = x*radial + 2*p1*x*y + p2*(r2+2*x*x)
        yd = y*radial + p1*(r2+2*y*y) + 2*p2*x*y
        valid &= r2 < first_positive_root([1, 3*k1, 5*k2, 7*k3])
        derivative = k1 + 2*k2*r2 + 3*k3*r2*r2
        jxx = radial + 2*x*x*derivative + 2*p1*y + 6*p2*x
        jyy = radial + 2*y*y*derivative + 6*p1*y + 2*p2*x
        jxy = 2*x*y*derivative + 2*p1*x + 2*p2*y
        valid &= (jxx > 0) & (jyy > 0) & (jxx*jyy-jxy*jxy > 0)
    else:
        radius = np.sqrt(r2)
        theta = np.arctan(radius)
        td = theta * (1 + sum(d*theta**(2*i+2) for i, d in enumerate(D)))
        factor = np.divide(td, radius, out=np.ones_like(radius), where=radius > 1e-12)
        xd, yd = x*factor, y*factor
        limit = min(np.deg2rad(theta_max_deg),
                    np.sqrt(first_positive_root([1, 3*D[0], 5*D[1], 7*D[2], 9*D[3]])))
        valid &= (theta < limit) & (td >= 0)
    uv = np.stack([K[0, 0]*xd + K[0, 1]*yd + K[0, 2],
                   K[1, 1]*yd + K[1, 2]], axis=-1)
    valid &= np.isfinite(uv).all(axis=-1)
    uv[~valid] = -10000
    return uv, valid


def world_grid(grid):
    keys = ("meters_per_pixel", "x_min_m", "x_max_m", "y_min_m", "y_max_m", "plane_z_m")
    if not isinstance(grid, dict) or any(key not in grid for key in keys):
        raise ValueError("Grid requires metric bounds, plane_z_m and meters_per_pixel")
    if not all(np.isfinite(grid[key]) for key in keys):
        raise ValueError("Grid values must be finite")
    m = float(grid["meters_per_pixel"])
    if not np.isfinite(m) or m <= 0:
        raise ValueError("meters_per_pixel must be positive")
    height = round((grid["x_max_m"]-grid["x_min_m"])/m)
    width = round((grid["y_max_m"]-grid["y_min_m"])/m)
    if min(width, height) < 2 or width*height > 20_000_000:
        raise ValueError("Invalid or excessive BEV dimensions")
    for extent, count in [(grid["x_max_m"]-grid["x_min_m"], height),
                          (grid["y_max_m"]-grid["y_min_m"], width)]:
        if not np.isclose(extent, count*m, atol=1e-8):
            raise ValueError("Grid extents must be multiples of meters_per_pixel")
    xs = grid["x_max_m"]-(np.arange(height)+.5)*m
    ys = grid["y_max_m"]-(np.arange(width)+.5)*m
    yy, xx = np.meshgrid(ys, xs)
    return np.stack([xx, yy, np.full_like(xx, grid["plane_z_m"])], axis=-1)


def rectangle_mask(world, bounds):
    value = np.asarray(bounds, float)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("Rectangle bounds need finite [x_min,x_max,y_min,y_max]")
    a, b, c, d = value
    if a > b or c > d:
        raise ValueError("Rectangle bounds must be ordered")
    return ((world[..., 0] >= a) & (world[..., 0] <= b)
            & (world[..., 1] >= c) & (world[..., 1] <= d))


def sample_camera(world, camera, root):
    root = Path(root)
    image_path = Path(camera["image"])
    if not image_path.is_absolute():
        image_path = root / image_path
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    expected = camera.get("metadata", {}).get("source_image_sha256")
    if expected is not None and image_sha256 != expected:
        raise ValueError(f"Source image hash changed: {image_path}")
    R, t = np.asarray(camera["R_w2c"], float), np.asarray(camera["tvec"], float)
    if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError("Invalid extrinsic matrix")
    if not np.allclose(R@R.T, np.eye(3), atol=1e-5) or np.linalg.det(R) < .999:
        raise ValueError("World-to-camera rotation is not a proper rotation")
    cp = world @ R.T + t
    uv, valid = project_camera(cp, camera["K"], camera["D"], camera["model"],
                               camera.get("theta_max_deg", 89.9))
    h, w = image.shape[:2]
    valid &= (uv[..., 0] >= 0) & (uv[..., 0] <= w-1)
    valid &= (uv[..., 1] >= 0) & (uv[..., 1] <= h-1)
    uv[~valid] = -10000
    mx, my = uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32)
    source_valid = np.full((h, w), 255, np.uint8)
    if camera.get("image_mask", "full") == "fisheye_circle":
        yy, xx = np.mgrid[:h, :w]
        metadata = camera.get("metadata", {})
        cx, cy = metadata.get("source_image_circle_center_xy", [(w-1)/2, (h-1)/2])
        radius = float(metadata.get("source_image_circle_radius_px", min(h, w)/2-1))
        if not np.isfinite([cx, cy, radius]).all() or radius <= 0:
            raise ValueError("Fisheye circle needs a finite centre and positive radius")
        source_valid[((xx-cx)**2+(yy-cy)**2) > radius**2] = 0
    elif camera.get("image_mask", "full") != "full":
        raise ValueError("Unknown image_mask; black pixels are not invalid pixels")
    sampled_mask = cv2.remap(source_valid, mx, my, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
    valid &= sampled_mask >= 254
    sampled = cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    angle = np.arctan2(np.linalg.norm(cp[..., :2], axis=-1), np.maximum(cp[..., 2], 1e-8))
    confidence = np.maximum(np.cos(angle), .02)**2 * valid
    sampled[~valid] = 0
    return sampled, valid, confidence.astype(np.float32), {
        "id": camera["id"], "image": str(image_path.resolve()),
        "image_sha256": image_sha256,
        "resolution_px": [w, h], "metadata": camera.get("metadata", {})}


def select_hard(images, valid, weights):
    weights = np.where(valid, weights, 0)
    covered = np.any(valid, axis=0)
    owners = np.argmax(np.where(valid, weights, -np.inf), axis=0).astype(np.int16)
    owners[~covered] = -1
    output = np.zeros(images.shape[1:], np.uint8)
    for i, image in enumerate(images):
        output[owners == i] = image[owners == i]
    return output, owners


def seam_pixels(owners):
    seam = np.zeros(owners.shape, bool)
    pairs = (owners[:, 1:] != owners[:, :-1]) & (owners[:, 1:] >= 0) & (owners[:, :-1] >= 0)
    seam[:, 1:] |= pairs
    seam[:, :-1] |= pairs
    pairs = (owners[1:] != owners[:-1]) & (owners[1:] >= 0) & (owners[:-1] >= 0)
    seam[1:] |= pairs
    seam[:-1] |= pairs
    return seam


def blend_views(images, valid, weights, band):
    """Hard and feather baselines, plus graph-cut selection and local blending."""
    images, valid, weights = np.asarray(images), np.asarray(valid, bool), np.asarray(weights, float)
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8 or not len(images):
        raise ValueError("Need a nonempty N,H,W,3 uint8 BGR image array")
    if valid.shape != images.shape[:3] or weights.shape != valid.shape:
        raise ValueError("Image, validity and weight shapes differ")
    if not np.isfinite(band) or band < 1:
        raise ValueError("seam_band_px must be positive")
    weights = np.where(valid, weights, 0)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Valid-view weights must be finite and nonnegative")
    hard, baseline_owner = select_hard(images, valid, weights)
    total = weights.sum(axis=0)
    feather = np.divide((images.astype(float)*weights[..., None]).sum(axis=0),
                        total[..., None], out=hard.astype(float), where=total[..., None] > 0)
    active = [i for i in range(len(images)) if valid[i].any()]
    owned = valid.copy()
    if len(active) > 1:
        finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
        masks = [cv2.UMat(valid[i].astype(np.uint8)*255) for i in active]
        result = finder.find([cv2.UMat(images[i].astype(np.float32)) for i in active],
                             [(0, 0)]*len(active), masks)
        for i, mask in zip(active, result):
            owned[i] = (mask.get() if hasattr(mask, "get") else mask) > 0
    graph_hard, graph_owner = select_hard(images, valid, weights*(owned.astype(float)+1e-6))
    local_weights = []
    for i in range(len(images)):
        own = (graph_owner == i).astype(np.uint8)
        if not own.any():
            local_weights.append(np.zeros_like(total))
            continue
        inside = cv2.distanceTransform(own, cv2.DIST_L2, 3)
        outside = cv2.distanceTransform(1-own, cv2.DIST_L2, 3)
        alpha = np.clip((inside-outside+band)/(2*band), 0, 1)
        local_weights.append(alpha*weights[i])
    local_weights = np.stack(local_weights)
    denominator = local_weights.sum(axis=0)
    local = np.divide((images.astype(float)*local_weights[..., None]).sum(axis=0),
                      denominator[..., None], out=graph_hard.astype(float),
                      where=denominator[..., None] > 0)
    return {"hard": hard, "feather": np.rint(feather).clip(0, 255).astype(np.uint8),
            "graphcut_local": np.rint(local).clip(0, 255).astype(np.uint8)}, {
                "hard": baseline_owner, "graphcut_local": graph_owner}


def summary(values):
    a = np.asarray(values, float)
    if not a.size:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {"count": int(a.size), "mean": float(a.mean()),
            "median": float(np.median(a)), "p95": float(np.percentile(a, 95))}


def edge_disagreement(a, b, overlap, mpp):
    """Symmetric nearest-edge distance, a proxy rather than matched feature error."""
    common = cv2.erode(overlap.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ea = (cv2.Canny(a, 60, 150) > 0) & common
    eb = (cv2.Canny(b, 60, 150) > 0) & common
    if min(ea.sum(), eb.sum()) < 10:
        return {"pixels": summary([]), "millimeters": summary([])}
    da = cv2.distanceTransform((~ea).astype(np.uint8), cv2.DIST_L2, 5)
    db = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 5)
    values = np.r_[da[eb], db[ea]]
    return {"pixels": summary(values), "millimeters": summary(values*mpp*1000)}


def compare_views(images, valid, camera_ids, evaluation, mpp):
    pairs = []
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            common = valid[i] & valid[j] & evaluation
            n = int(common.sum())
            if n < 25:
                continue
            delta = images[i].astype(float)-images[j].astype(float)
            rgb = np.abs(delta[common]).mean(axis=1)
            pairs.append({"cameras": [camera_ids[i], camera_ids[j]], "overlap_pixels": n,
                          "rgb_mae_0_255": summary(rgb),
                          "rgb_rmse_0_255": float(np.sqrt(np.mean(delta[common]**2))),
                          "nearest_edge_distance_proxy": edge_disagreement(images[i], images[j], common, mpp)})
    return pairs


def method_scores(image, owner, fixed_owner, overlap, evaluation):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    gx, gy = np.diff(grey, axis=1), np.diff(grey, axis=0)
    ex = evaluation[:, 1:] & evaluation[:, :-1] & overlap[:, 1:] & overlap[:, :-1]
    ey = evaluation[1:] & evaluation[:-1] & overlap[1:] & overlap[:-1]
    sx = ((fixed_owner[:, 1:] != fixed_owner[:, :-1])
          & (fixed_owner[:, 1:] >= 0) & (fixed_owner[:, :-1] >= 0) & ex)
    sy = ((fixed_owner[1:] != fixed_owner[:-1])
          & (fixed_owner[1:] >= 0) & (fixed_owner[:-1] >= 0) & ey)
    grad = np.r_[np.abs(gx[ex]), np.abs(gy[ey])]
    jump = np.r_[np.abs(gx[sx]), np.abs(gy[sy])]
    return {"fixed_boundary_gray_jump_0_255": summary(jump),
            "overlap_gray_gradient_0_255": summary(grad),
            "assignment_boundary_pixels": int((seam_pixels(owner) & evaluation).sum()),
            "caution": "Low jump may reflect blur or a naturally flat patch; inspect edges and coverage."}


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(path)


def decorate(image, excluded, covered):
    result = image.copy()
    result[~covered & ~excluded] = (215, 215, 245)
    result[excluded] = (225, 225, 225)
    return result


def _json_bytes(value):
    def encode(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=encode)+"\n").encode("utf-8")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _camera_ids(cameras):
    if not isinstance(cameras, (list, tuple)) or not cameras:
        raise ValueError("At least one explicit world-to-camera record is required")
    names = []
    for camera in cameras:
        name = camera.get("id") if isinstance(camera, dict) else None
        if (not isinstance(name, str) or not name.strip() or name in (".", "..")
                or "/" in name or "\\" in name or "\x00" in name):
            raise ValueError("Camera IDs must be nonempty safe filename components")
        if name in names:
            raise ValueError("Camera IDs must be unique")
        names.append(name)
    return names


def _validate_pose_and_lens(candidate, solution, model):
    if "R_w2c" not in candidate or "tvec" not in candidate:
        raise ValueError("Need R_w2c/tvec in the declared BEV world; rig poses are not world poses")
    rotation, translation = np.asarray(candidate["R_w2c"], float), np.asarray(candidate["tvec"], float)
    if (rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()):
        raise ValueError("Need finite world-to-camera R (3,3) and t (3,)")
    if (not np.allclose(rotation@rotation.T, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
        raise ValueError("World-to-camera rotation must be proper and orthogonal")
    project_camera(np.empty((0, 3)), solution["K"], solution["D"], model,
                   candidate.get("theta_max_deg", 89.9))


def select_candidate(extrinsic, intrinsic, model):
    """Select a conditional world-pose candidate and its exact lens seed.

    Optimizer success is recorded, not promoted to a uniqueness certificate.
    The first key is the error over all correspondences, never the fit subset.
    """
    solutions = {}
    for row in intrinsic.get("solutions", []):
        seed = row["seed"]
        if seed in solutions:
            raise ValueError("Duplicate intrinsic seed")
        solutions[seed] = row
    eligible = []
    for index, row in enumerate(extrinsic.get("candidates", [])):
        seed = row.get("intrinsic_seed")
        if seed not in solutions:
            raise ValueError(f"Extrinsic candidate {index} has no matching intrinsic seed")
        try:
            all_error = float(row.get("all_correspondence_rmse_px", math.inf))
            subset_error = float(row.get("reprojection_rmse_px", math.inf))
            depth = float(row.get("minimum_depth_m", -math.inf))
            if not (np.isfinite([all_error, subset_error, depth]).all()
                    and min(all_error, subset_error) >= 0 and depth > 0):
                continue
            solution = solutions[seed]
            _validate_pose_and_lens(row, solution, model)
            key = (all_error, -int(row.get("fit_corner_count", 0)), subset_error,
                   str(seed), int(row.get("row_start", 0)), int(row.get("quarter_turns", 0)), index)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        eligible.append((key, index, row, solution))
    if not eligible:
        raise ValueError("No finite positive-depth world-pose candidate with a valid matched lens")
    _, index, row, solution = min(eligible, key=lambda value: value[0])
    return index, row, solution, len(eligible)


def _source_file(root, model, channel, filename):
    root = Path(root).expanduser().resolve()
    possibilities = [root/model/channel/filename, root/channel/filename]
    found = [path for path in possibilities if path.is_file()]
    if not found:
        raise FileNotFoundError(possibilities[0])
    if len(found) > 1:
        raise ValueError(f"Ambiguous model-root and channel-root layouts under {root}")
    return found[0]


def prepare_bev_spec(intrinsics_dir, extrinsics_dir, images_dir, *, model, channels, grid,
                     exclude_rectangles_m=None, evaluation_rectangles_m=None, seam_band_px=12):
    """Read fitted results and PNGs; return a BEV spec without writing anything.

    Each root contains model/channel/file, or channel/file for a model-specific
    root. Image files are big_board_fixed.png. The supplied extrinsic R_w2c/tvec
    must already use the metric, vehicle-aligned world declared by grid. A rig
    result cannot be silently substituted. No camera-truth file is read.
    """
    if model not in ("pinhole", "fisheye"):
        raise ValueError("model must be pinhole or fisheye")
    if isinstance(channels, (str, bytes)):
        raise ValueError("channels must be a sequence of camera IDs")
    channels = list(channels)
    _camera_ids([{"id": channel} for channel in channels])
    world = world_grid(grid)
    if isinstance(seam_band_px, bool) or int(seam_band_px) != seam_band_px or seam_band_px < 1:
        raise ValueError("seam_band_px must be a positive integer")
    excluded = np.zeros(world.shape[:2], bool)
    for bounds in exclude_rectangles_m or []:
        excluded |= rectangle_mask(world, bounds)
    if excluded.all():
        raise ValueError("All BEV pixels are excluded; no evaluation domain remains")
    sources, cameras = {}, []

    def source(path):
        checksum = _digest(path)
        sources[str(path)] = checksum
        return checksum

    for channel in channels:
        ipath = _source_file(intrinsics_dir, model, channel, "calibration.json")
        epath = _source_file(extrinsics_dir, model, channel, "extrinsics.json")
        ihash, ehash = source(ipath), source(epath)
        intrinsic = json.loads(ipath.read_text(encoding="utf-8"))
        extrinsic = json.loads(epath.read_text(encoding="utf-8"))
        relative_image = extrinsic.get("image_relative_to_images")
        if relative_image is not None:
            relative_parts = Path(relative_image).parts
            if relative_parts[:2] != (model, channel) or len(relative_parts) < 3:
                raise ValueError("External observation image has a different model/camera association")
            image_path = _source_file(images_dir, model, channel, str(Path(*relative_parts[2:])))
        else:
            declared = Path(extrinsic.get("image", "big_board_fixed.png")).expanduser()
            try:
                image_path = _source_file(images_dir, model, channel, declared.name)
            except FileNotFoundError:
                if not declared.is_absolute() or not declared.is_file():
                    raise
                image_path = declared.resolve()
        image_hash = source(image_path)
        for result in (intrinsic, extrinsic):
            if result.get("camera_id", channel) != channel:
                raise ValueError("Camera result ID differs from its requested channel")
            if result.get("model", result.get("camera_model", model)) != model:
                raise ValueError("Camera result model differs from its requested lens model")
        index, row, solution, eligible = select_candidate(extrinsic, intrinsic, model)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode source PNG: {image_path}")
        h, w = image.shape[:2]
        for result, key in ((intrinsic, "image_resolution_px"), (extrinsic, "resolution_px")):
            if key in result and list(result[key]) != [w, h]:
                raise ValueError("Fitted lens/image resolution differs from the source PNG")
        metadata = {"kind": "estimated_conditional_candidate", "candidate_index": index,
            "candidate_id": row.get("candidate_id", index), "candidate_count": len(extrinsic["candidates"]),
            "eligible_candidate_count": eligible, "intrinsic_seed": row["intrinsic_seed"],
            "intrinsic_best_seed_not_used_for_pairing": intrinsic.get("best_seed"),
            "source_intrinsic_json": str(ipath), "source_intrinsic_sha256": ihash,
            "source_extrinsic_json": str(epath), "source_extrinsic_sha256": ehash,
            "source_image_sha256": image_hash, "intrinsic_status": intrinsic.get("status", "UNREPORTED"),
            "extrinsic_status": extrinsic.get("status", "UNREPORTED"),
            "optimizer_success": row.get("optimizer_success"),
            "intrinsic_optimizer_success": solution.get("optimizer_success"),
            "global_uniqueness_proved": intrinsic.get("global_uniqueness_proved", False),
            "all_correspondence_rmse_px": row["all_correspondence_rmse_px"],
            "subset_reprojection_rmse_px": row["reprojection_rmse_px"],
            "minimum_depth_m": row["minimum_depth_m"],
            "selection_key": ["all_correspondence_rmse_px", "-fit_corner_count", "subset_rmse_px",
                              "intrinsic_seed_as_string", "row_start", "quarter_turns", "candidate_index"],
            "interpretation": "Conditional diagnostic candidate; low error does not resolve lens or physical corner identities"}
        for key in ("row_start", "quarter_turns", "corner_ids", "observation_indices", "fit_corner_count", "all_correspondence_count"):
            if key in row:
                metadata[key] = copy.deepcopy(row[key])
        camera = {"id": channel, "model": model, "image": str(image_path),
                  "K": np.asarray(solution["K"], float).tolist(),
                  "D": np.asarray(solution["D"], float).reshape(-1).tolist(),
                  "R_w2c": row["R_w2c"], "tvec": row["tvec"],
                  "image_mask": "fisheye_circle" if model == "fisheye" else "full", "metadata": metadata}
        if model == "fisheye":
            camera["theta_max_deg"] = 89.9
            metadata.update(source_image_circle_center_xy=[(w-1)/2, (h-1)/2],
                            source_image_circle_radius_px=min(w, h)/2-1,
                            lens_domain_policy="positive-depth hemisphere, theta<89.9deg and continuous monotone radial domain",
                            source_circle_policy="centred inscribed circle; caller must verify or override for its actual image")
        cameras.append(camera)
    for path, expected in sources.items():
        if _digest(path) != expected:
            raise RuntimeError(f"Input changed during BEV preparation: {path}")
    return {"schema_version": "bev-stitch.v1", "grid": copy.deepcopy(grid), "cameras": cameras,
            "exclude_rectangles_m": copy.deepcopy(exclude_rectangles_m or []),
            "evaluation_rectangles_m": copy.deepcopy(evaluation_rectangles_m or []),
            "seam_band_px": int(seam_band_px), "metadata": {
                "parameter_source": "estimated", "camera_truth_parameters_used": False,
                "status": "DIAGNOSTIC_UNRESOLVED_CALIBRATION",
                "world_frame": "+X vehicle front, +Y vehicle left, +Z up; supplied R_w2c maps this world to camera",
                "translation_unit": "metres; caller must establish the world/target scale",
                "rig_to_world_implicitly_assumed": False,
                "source_inputs": [{"path": path, "sha256": value} for path, value in sorted(sources.items())]}}


def stitch_bev(spec, out_dir, *, execute=False, base_dir=None):
    """Write BEV images/metrics only when execute=True; return a plan or metrics.

    spec is a bev-stitch.v1 dictionary or JSON path. Relative image paths use
    base_dir, otherwise the JSON parent (or cwd for a dictionary). Output paths
    are relative to cwd. This function never estimates or changes calibration.
    """
    if not isinstance(execute, bool):
        raise ValueError("execute must be boolean")
    source_path = None
    if isinstance(spec, (str, Path)):
        source_path = Path(spec).expanduser().resolve()
        original_bytes = source_path.read_bytes()
        data = json.loads(original_bytes)
    elif isinstance(spec, dict):
        data = copy.deepcopy(spec)
        original_bytes = _json_bytes(data)
    else:
        raise TypeError("spec must be a dictionary or JSON path")
    if data.get("schema_version") != "bev-stitch.v1":
        raise ValueError("Unsupported BEV spec schema")
    cameras = data.get("cameras")
    names = _camera_ids(cameras)
    world = world_grid(data["grid"])
    band = data.get("seam_band_px", 12)
    if isinstance(band, bool) or not isinstance(band, (int, np.integer)) or band < 1:
        raise ValueError("seam_band_px must be a positive integer")
    for camera in cameras:
        _validate_pose_and_lens(camera, camera, camera["model"])
    root = Path(base_dir).expanduser().resolve() if base_dir is not None else (
        source_path.parent if source_path is not None else Path.cwd())
    out = Path(out_dir).expanduser().resolve()
    excluded = np.zeros(world.shape[:2], bool)
    for bounds in data.get("exclude_rectangles_m", []):
        excluded |= rectangle_mask(world, bounds)
    evaluation = ~excluded
    if not evaluation.any():
        raise ValueError("All BEV pixels are excluded; no evaluation domain remains")
    regions = data.get("evaluation_rectangles_m", [])
    _camera_ids([{"id": region.get("id")} for region in regions]) if regions else None
    for region in regions:
        rectangle_mask(world, region["bounds"])
    if not execute:
        return {"schema_version": "bev-stitch-plan.v1", "dry_run": True, "execute": False,
                "output": str(out), "shape_px": [world.shape[1], world.shape[0]],
                "camera_count": len(cameras), "camera_ids": names,
                "evaluation_pixels": int(evaluation.sum()), "base_dir": str(root)}
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {out}")
    samples = [sample_camera(world, camera, root) for camera in cameras]
    images, valid, weights = [np.stack([sample[i] for sample in samples]) for i in range(3)]
    valid &= evaluation
    weights *= valid
    methods, owners = blend_views(images, valid, weights, int(band))
    count = valid.sum(axis=0)
    covered, overlap = count > 0, count > 1
    pairs = compare_views(images, valid, names, evaluation, data["grid"]["meters_per_pixel"])
    scores = {name: method_scores(image, owners["hard" if name == "feather" else name],
                                 owners["hard"], overlap, evaluation) for name, image in methods.items()}
    region_records, crop_panels = [], []
    for region in regions:
        rectangle = rectangle_mask(world, region["bounds"])
        mask = rectangle & evaluation
        yy, xx = np.where(rectangle)
        if not len(xx):
            region_records.append({**region, "evaluation_pixels": 0, "coverage_fraction": None,
                                   "overlap_pixels": 0, "status": "OUTSIDE_GRID"})
            continue
        xa, xb, ya, yb = xx.min(), xx.max()+1, yy.min(), yy.max()+1
        patches = [decorate(methods[name], excluded, covered)[ya:yb, xa:xb] for name in METHODS]
        ch, cw = patches[0].shape[:2]
        panel = np.full((ch+30, cw*len(METHODS), 3), 255, np.uint8)
        for i, (name, patch) in enumerate(zip(METHODS, patches)):
            panel[30:, i*cw:(i+1)*cw] = patch
            cv2.putText(panel, name, (i*cw+4, 20), cv2.FONT_HERSHEY_SIMPLEX, .42, (20, 20, 20), 1, cv2.LINE_AA)
        crop_panels.append((region["id"], cv2.resize(panel, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)))
        denominator = int(mask.sum())
        region_records.append({**region, "evaluation_pixels": denominator,
            "coverage_fraction": float(covered[mask].mean()) if denominator else None,
            "overlap_pixels": int((overlap & mask).sum())})
    manifest = {"schema_version": "bev-stitch-result.v1", "output": str(out), "dry_run": False,
        "status": "DIAGNOSTIC_RENDERED" if covered.any() else "NO_COVERAGE",
        "calibration_validity_confirmed": False, "spec": str(source_path) if source_path else None,
        "spec_sha256": hashlib.sha256(original_bytes).hexdigest(), "renderer_code_sha256": _digest(__file__),
        "grid": data["grid"], "shape_px": [world.shape[1], world.shape[0]],
        "orientation": "+X vehicle front up; +Y vehicle left left",
        "image_space": "BGR uint8 display-encoded source PNG, 0..255; not linear radiometry",
        "coverage_fraction": float(covered[evaluation].mean()),
        "overlap_fraction": float(overlap[evaluation].mean()),
        "evaluation_area_m2": float(evaluation.sum()*data["grid"]["meters_per_pixel"]**2),
        "method_metrics": scores, "pair_metrics": pairs, "fixed_regions": region_records,
        "cameras": [sample[3] for sample in samples], "metadata": data.get("metadata", {}),
        "occlusion_handling": "Only explicit footprint exclusions; no per-camera depth/occlusion inference.",
        "proxy_caution": "Nearest-edge distances do not establish feature identity in repeated patterns.",
        "calibration_claim": "Diagnostic candidate reprojection; blending never certifies calibration.",
        "exposure_compensation": False, "native_seam_solver": "OpenCV GraphCut COST_COLOR_GRAD",
        "seam_band_px": int(band), "opencv_version": cv2.__version__}
    manifest_bytes = _json_bytes(manifest)
    out.mkdir(parents=True, exist_ok=False)
    for folder in ("warped", "valid", "crops"):
        (out/folder).mkdir()
    for i, name in enumerate(names):
        write_image(out/"warped"/(name+".png"), images[i])
        write_image(out/"valid"/(name+".png"), valid[i].astype(np.uint8)*255)
    write_image(out/"coverage_count.png", (count*255/len(images)).astype(np.uint8))
    write_image(out/"excluded_mask.png", excluded.astype(np.uint8)*255)
    write_image(out/"overlap_mask.png", overlap.astype(np.uint8)*255)
    for name, image in methods.items():
        owner = owners["hard" if name == "feather" else name]
        write_image(out/(name+".png"), decorate(image, excluded, covered))
        write_image(out/(name+"_raw.png"), image)
        source = np.zeros_like(image)
        for i in range(len(images)):
            source[owner == i] = PALETTE[i % len(PALETTE)]
        write_image(out/(name+"_sources.png"), source)
        overlay = decorate(image, excluded, covered)
        overlay[seam_pixels(owner)] = (255, 0, 255)
        write_image(out/(name+"_seams.png"), overlay)
    h, w = world.shape[:2]
    panel = np.full((h+66, len(METHODS)*w, 3), 255, np.uint8)
    for i, name in enumerate(METHODS):
        panel[66:, i*w:(i+1)*w] = decorate(methods[name], excluded, covered)
        cv2.putText(panel, name, (i*w+14, 27), cv2.FONT_HERSHEY_SIMPLEX, .65, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(panel, "+X front: up | +Y left: left", (i*w+14, 51), cv2.FONT_HERSHEY_SIMPLEX, .42, (50, 50, 50), 1, cv2.LINE_AA)
    write_image(out/"comparison.png", panel)
    for name, panel in crop_panels:
        write_image(out/"crops"/(name+".png"), panel)
    (out/"metrics.json").write_bytes(manifest_bytes)
    (out/"input_spec.json").write_bytes(original_bytes)
    return manifest

