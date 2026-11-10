"""Image-only fixed-camera planar calibration and conditional world poses.

This module deliberately does not import repository or experiment code. Camera
truth is not an accepted solver input. Its default protocol is diagnostic:
fixed camera, coplanar XY/Yaw board motion, free skew and full Brown5/theta4.
Low residuals do not establish unique intrinsics or physical checker identities.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import time
from collections.abc import Mapping

import cv2
import numpy as np
from scipy.optimize import least_squares

MODELS = ("pinhole", "fisheye")
EXPECTED_ERRORS = (cv2.error, ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError)
PROTOCOL = "fixed_camera_ground_board_xy_yaw"

def vij(H,i,j):
    a,b=H[:,i],H[:,j]
    return np.array([a[0]*b[0],a[0]*b[1]+a[1]*b[0],a[1]*b[1],
                     a[2]*b[0]+a[0]*b[2],a[2]*b[1]+a[1]*b[2],a[2]*b[2]])

def homography_spectrum(objects, observations):
    rows=[]
    for uv in observations:
        H,_=cv2.findHomography(objects[:,:2],uv,0)
        for v in (vij(H,0,1),vij(H,0,0)-vij(H,1,1)):
            rows.append(v/np.linalg.norm(v))
    s=np.linalg.svd(rows,compute_uv=False)
    return {'singular_values':s.tolist(),'relative_singular_values':(s/s[0]).tolist(),
            'caution':'Raw distorted/noisy pixels; tiny nonzero singular values are not proof of observability.'}

def lens_from(p, model, size):
    if model not in MODELS:
        raise ValueError(f"Unsupported camera model: {model}")
    w,h=size; fx,fy=np.exp(p[:2]); skew,cx,cy=p[2:5]*[w,w,h]
    return np.array([[fx,skew,cx],[0,fy,cy],[0,0,1.]]),p[5:10 if model=='pinhole' else 9]

def pixels(cp, K, D, model):
    if model not in MODELS:
        raise ValueError(f"Unsupported camera model: {model}")
    x,y=(cp[:,:2]/np.maximum(cp[:,2,None],1e-7)).T
    r2=x*x+y*y
    if model=='pinhole':
        radial=1+D[0]*r2+D[1]*r2*r2+D[4]*r2**3
        xd=x*radial+2*D[2]*x*y+D[3]*(r2+2*x*x)
        yd=y*radial+D[2]*(r2+2*y*y)+2*D[3]*x*y
    else:
        r=np.sqrt(r2); theta=np.arctan(r)
        td=theta*(1+sum(d*theta**(2*i+2) for i,d in enumerate(D)))
        scale=td/np.maximum(r,1e-12); xd=x*scale; yd=y*scale
    return np.stack([K[0,0]*xd+K[0,1]*yd+K[0,2],K[1,1]*yd+K[1,2]],axis=1)

def camera_points(p, nl, objects, n):
    R=cv2.Rodrigues(p[nl:nl+3])[0]; t=p[nl+3:nl+6]
    pose=np.zeros((n,3)); pose[1:]=p[nl+6:].reshape(n-1,3)
    x,y=objects[:,:2].T
    c,s=np.cos(pose[:,0,None]),np.sin(pose[:,0,None])
    xx=c*x-s*y+pose[:,1,None]; yy=s*x+c*y+pose[:,2,None]
    plane=np.stack([xx,yy,np.zeros_like(xx)],axis=-1)
    return plane@R.T+t

def initial_poses(objects, observations, K, D, model):
    poses=[]
    K0=K.copy(); K0[0,1]=0
    for uv in observations:
        q=uv.copy(); q[:,0]-=K[0,1]/K[1,1]*(q[:,1]-K[1,2])
        if model=='fisheye':
            q=cv2.fisheye.undistortPoints(q.reshape(-1,1,2),K0,np.zeros(4),P=K0).reshape(-1,2)
        ok,rv,tv=cv2.solvePnP(objects,q,K0,np.zeros(5),flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:raise RuntimeError('PnP initialization failed')
        poses.append((cv2.Rodrigues(rv)[0],tv.ravel()))
    R0,t0=poses[0]; p=[]
    for Ri,ti in poses[1:]:
        rel=R0.T@Ri; delta=R0.T@(ti-t0)
        p.extend([np.arctan2(rel[1,0],rel[0,0]),delta[0],delta[1]])
    return np.r_[cv2.Rodrigues(R0)[0].ravel(),t0,p]

def fit(objects, observations, model, size, seed):
    w,h=size; nl=10 if model=='pinhole' else 9; n=len(observations)
    f=( [.45,.9,1.6] if model=='pinhole' else [.23,.45,.75] )[seed]*w
    lp=np.r_[np.log(f),np.log(f*(1+[0,.1,-.1][seed])),[0,.06,-.06][seed],
             .5+[0,-.08,.08][seed],.5+[0,.08,-.08][seed],np.zeros(nl-5)]
    K,D=lens_from(lp,model,size)
    p=np.r_[lp,initial_poses(objects,observations,K,D,model)]
    low=np.r_[np.log(w*.03),np.log(h*.03),-2.,-1.,-1.,np.full(nl-5,-10.),np.full(len(p)-nl,-np.inf)]
    high=np.r_[np.log(w*20),np.log(h*20),2.,2.,2.,np.full(nl-5,10.),np.full(len(p)-nl,np.inf)]
    def residual(v):
        K,D=lens_from(v,model,size); cp=camera_points(v,nl,objects,n)
        uv=pixels(cp.reshape(-1,3),K,D,model).reshape(n,-1,2)
        return (uv-observations).ravel()
    start=time.perf_counter()
    result=least_squares(residual,p,bounds=(low,high),x_scale='jac',max_nfev=300,
                         ftol=1e-9,xtol=1e-9,gtol=1e-7)
    K,D=lens_from(result.x,model,size)
    cp=camera_points(result.x,nl,objects,n)
    projected=pixels(cp.reshape(-1,3),K,D,model).reshape(n,-1,2)
    J=result.jac; norm=np.linalg.norm(J,axis=0)
    s=np.linalg.svd(J/np.maximum(norm,1e-12),compute_uv=False)
    return {'seed':seed,'K':K.tolist(),'D':D.tolist(),'all_parameters':result.x.tolist(),
        'initial_lens_parameters':lp.tolist(),'training_rmse_px':float(np.sqrt(np.mean(np.sum((projected-observations)**2,axis=-1)))),
        'optimizer_success':bool(result.success),'optimizer_message':result.message,'nfev':int(result.nfev),
        'seconds':time.perf_counter()-start,'minimum_camera_depth_m':float(cp[:,:,2].min()),
        'column_normalized_jacobian_singular_values':s.tolist(),
        'jacobian_condition_number':float(s[0]/max(s[-1],1e-30)),
        'weak_directions_below_1e_minus_5':int((s/s[0]<1e-5).sum()),
        'numerical_safety_bounds':'fx/fy positive and finite; skew in +/-2 image widths; principal point -1..2 image extents; distortion +/-10. No parameter fixed.',
        'projected_training_corners':projected.tolist()}

def heldout(objects, observation, model, K, D):
    # Fit the pose to four fifths of corners, score the remaining fifth.
    ids=np.arange(len(objects)); keep=ids%5!=0; test=~keep
    K0=K.copy(); K0[0,1]=0
    q=observation.copy(); q[:,0]-=K[0,1]/K[1,1]*(q[:,1]-K[1,2])
    if model=='fisheye':q=cv2.fisheye.undistortPoints(q.reshape(-1,1,2),K0,D,P=K0).reshape(-1,2)
    ok,rv,tv=cv2.solvePnP(objects[keep],q[keep],K0,np.zeros(5) if model=='fisheye' else D)
    if not ok:raise RuntimeError('Held-out pose failed')
    def fun(p):return (pixels(objects@cv2.Rodrigues(p[:3])[0].T+p[3:],K,D,model)[keep]-observation[keep]).ravel()
    opt=least_squares(fun,np.r_[rv.ravel(),tv.ravel()],max_nfev=200)
    pred=pixels(objects@cv2.Rodrigues(opt.x[:3])[0].T+opt.x[3:],K,D,model)
    return {'pose_fit_corner_indices':ids[keep].tolist(),'scored_corner_indices':ids[test].tolist(),
            'scored_rmse_px':float(np.sqrt(np.mean(np.sum((pred[test]-observation[test])**2,axis=1)))),
            'rvec':opt.x[:3].tolist(),'tvec':opt.x[3:].tolist(),'projected':pred.tolist()}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _Sources:
    def __init__(self):
        self.records = {}

    def path(self, path, role):
        path = Path(path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"{role} is not a file: {path}")
        record = {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path), "role": role}
        key = str(path)
        old = self.records.get(key)
        if old and old["sha256"] != record["sha256"]:
            raise RuntimeError(f"Input changed while reading: {path}")
        self.records[key] = record
        return path

    def document(self, value, role):
        if isinstance(value, Mapping):
            document = copy.deepcopy(dict(value))
            encoded = json.dumps(document, sort_keys=True, allow_nan=False).encode()
            self.records[f"memory:{role}"] = {
                "path": None, "role": role, "sha256": hashlib.sha256(encoded).hexdigest(),
                "bytes": len(encoded), "kind": "in-memory JSON",
            }
            return document
        path = self.path(value, role)
        return json.loads(path.read_text(encoding="utf-8"),
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant: {value}")))

    def verify(self):
        for record in self.records.values():
            if record["path"] is not None and _sha(record["path"]) != record["sha256"]:
                raise RuntimeError(f"Input changed during calibration: {record['path']}")


def _new_output(path):
    requested = Path(path).expanduser().absolute()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output root: {requested}")
    return requested.resolve()


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image: {path}")


def _identifier(value, label):
    if (not isinstance(value, str) or not value or value in (".", "..")
            or any(c in value for c in ("/", "\\", "\0"))
            or any(ord(c) < 32 for c in value)):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def _array(value, shape, label):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite array of shape {shape}")
    return array


def _lens(K, D, model):
    if model not in MODELS:
        raise ValueError(f"Unsupported camera model: {model}")
    K = _array(K, (3, 3), "K")
    D = _array(D, (5,) if model == "pinhole" else (4,), "D")
    if (K[0, 0] <= 0 or K[1, 1] <= 0 or abs(K[1, 0]) > 1e-12
            or not np.allclose(K[2], [0, 0, 1])):
        raise ValueError("K requires positive focal lengths and last row [0,0,1]")
    return K, D


def _public_solver_spec(spec):
    forbidden = {"K", "D", "R_w2c", "tvec", "camera_center_world", "board_to_world",
                 "rig_frames", "truth_for_evaluation_only"}
    def visit(value):
        if isinstance(value, dict):
            bad = forbidden & set(value)
            if bad:
                raise ValueError(f"Camera/pose truth is not allowed in solver_spec: {sorted(bad)}")
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(spec)
    protocol = spec.get("acquisition_protocol", PROTOCOL)
    if protocol not in (PROTOCOL, "fixed_camera_plane_xy_yaw"):
        raise ValueError(f"Only the fixed-camera coplanar XY/Yaw protocol is implemented: {protocol}")


def _board(spec):
    geometry = spec.get("geometry", spec)
    declared = geometry.get("pattern_size", spec.get("pattern_size"))
    legacy = None
    if "corners_x" in geometry or "corners_y" in geometry:
        if not {"corners_x", "corners_y"} <= set(geometry):
            raise ValueError("Both corners_x and corners_y are required")
        legacy = [geometry["corners_x"], geometry["corners_y"]]
    if declared is not None and legacy is not None and list(declared) != legacy:
        raise ValueError("pattern_size conflicts with corners_x/corners_y")
    pattern = declared if declared is not None else legacy
    if (not isinstance(pattern, (list, tuple)) or len(pattern) != 2
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 3 for x in pattern)):
        raise ValueError("Declare pattern_size [columns,rows], each at least three")
    pitch = geometry.get("square_size", spec.get("square_size"))
    if isinstance(pitch, bool) or not isinstance(pitch, (int, float)) or not math.isfinite(pitch) or pitch <= 0:
        raise ValueError("Declare a finite positive square_size")
    nx, ny = pattern
    objects = np.array([[(x-(nx-1)/2)*pitch, (y-(ny-1)/2)*pitch, 0.]
                        for y in range(ny) for x in range(nx)])
    return tuple(pattern), float(pitch), objects


def _unknowns(model):
    return ["fx", "fy", "skew", "cx", "cy"] + (
        ["k1", "k2", "p1", "p2", "k3"] if model == "pinhole" else ["k1", "k2", "k3", "k4"])


def _model_specs(spec):
    models = spec.get("models")
    if not isinstance(models, dict) or not models or set(models) - set(MODELS):
        raise ValueError("models must explicitly declare pinhole and/or fisheye")
    for model, settings in models.items():
        if not isinstance(settings, dict):
            raise ValueError("Each model needs a settings object")
        size = settings.get("resolution_px", settings.get("image_size"))
        if (not isinstance(size, (list, tuple)) or len(size) != 2
                or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in size)):
            raise ValueError(f"{model} requires resolution_px [width,height]")
        if "unknown" in settings and settings["unknown"] != _unknowns(model):
            raise ValueError(f"{model} currently fits exactly {_unknowns(model)}")
    return models


def _per_camera(value, model, channel):
    if isinstance(value, Mapping) and model in value:
        value = value[model]
    if isinstance(value, Mapping) and channel in value:
        value = value[channel]
    return value


def _view_plan(spec, images, model, channel):
    directory = images / model / channel
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    if "views" in spec and "image_splits" in spec:
        raise ValueError("Choose views or image_splits, not both")
    records = []
    if "views" in spec:
        value = _per_camera(spec["views"], model, channel)
        if not isinstance(value, list):
            raise ValueError("views must resolve to a list of {image,split} records per camera")
        for row in value:
            if not isinstance(row, dict):
                raise ValueError("Each view must be an object")
            if row.get("model", model) != model or row.get("camera_id", channel) != channel:
                continue
            image = row.get("image")
            if image is None and "name" in row:
                image = row["name"] if Path(row["name"]).suffix else row["name"] + ".png"
            records.append({"image": image, "split": row.get("split")})
    elif "image_splits" in spec:
        value = _per_camera(spec["image_splits"], model, channel)
        if not isinstance(value, Mapping) or set(value) != {"train", "holdout"}:
            raise ValueError("image_splits must resolve to explicit train/holdout filename lists")
        for split in ("train", "holdout"):
            if not isinstance(value[split], list):
                raise ValueError("Split filenames must be lists")
            records.extend({"image": name, "split": split} for name in value[split])
    elif "split_by_suffix" in spec:
        split = spec["split_by_suffix"]
        if not isinstance(split, dict) or set(split) != {"train", "holdout"}:
            raise ValueError("split_by_suffix requires explicit train/holdout integer lists")
        if any(not isinstance(v, list) or not v or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in v)
               for v in split.values()):
            raise ValueError("split_by_suffix values must be nonempty nonnegative integer lists")
        if set(split["train"]) & set(split["holdout"]):
            raise ValueError("Training and holdout suffixes overlap")
        found = {}
        for path in sorted(directory.glob("*.png")):
            match = re.search(r"(\d+)$", path.stem)
            if not match:
                continue
            index = int(match.group(1))
            if index not in split["train"] + split["holdout"]:
                raise ValueError(f"Image suffix has no declared split: {path.name}")
            if index in found:
                raise ValueError(f"Ambiguous duplicate image suffix {index}: {directory}")
            found[index] = path
            records.append({"image": path.name, "split": "train" if index in split["train"] else "holdout"})
        missing = set(split["train"] + split["holdout"]) - set(found)
        if missing:
            raise ValueError(f"Declared image suffixes are missing for {channel}: {sorted(missing)}")
    else:
        raise ValueError("Declare views, image_splits, or explicit split_by_suffix")
    result, seen, names = [], set(), set()
    for row in records:
        if row["split"] not in ("train", "holdout") or not isinstance(row["image"], str):
            raise ValueError("Each view needs an image path and train/holdout split")
        path = Path(row["image"]).expanduser()
        path = (path if path.is_absolute() else directory / path).resolve(strict=True)
        if not path.is_file() or path in seen or path.name in names:
            raise ValueError(f"Missing, duplicate, or ambiguous view: {path}")
        seen.add(path); names.add(path.name)
        result.append({"path": path, "split": row["split"]})
    if not result or {r["split"] for r in result} != {"train", "holdout"}:
        raise ValueError("Every camera/model needs declared training and holdout views")
    return result


def _stage_result(out, results, execute, sources, failures=False):
    result = {
        "status": ("COMPLETED_WITH_FAILURES" if failures else "DIAGNOSTIC_COMPLETED") if execute else "DRY_RUN",
        "output": str(out), "results": results, "execute": execute,
        "calibration_accepted": False, "cross_camera_bundle_adjustment": False,
        "input_count": len(sources.records),
    }
    sources.verify()
    if execute:
        _write(out / "source_inputs.json", {"inputs": list(sources.records.values())})
        _write(out / "stage.json", result)
    return result


def calibrate_intrinsics(images, solver_spec, out_dir, *, channels=None, execute=False):
    """Fit diagnostic fixed-camera planar intrinsics from raw image observations.

    Layout is images/model/camera/image. The spec explicitly declares board
    dimensions, model resolutions and view splits. Dry-run validates references
    and returns planned result paths; it performs no detection or optimization.
    """
    if not isinstance(execute, bool):
        raise TypeError("execute must be boolean")
    out = _new_output(out_dir)
    images = Path(images).expanduser().resolve(strict=True)
    sources = _Sources()
    sources.path(__file__, "calibration_module")
    spec = sources.document(solver_spec, "public_solver_spec")
    _public_solver_spec(spec)
    pattern, pitch, objects = _board(spec)
    models = _model_specs(spec)
    channels = list(channels) if channels is not None else spec.get("camera_ids")
    if not isinstance(channels, list) or not channels or len(set(channels)) != len(channels):
        raise ValueError("Declare unique camera_ids or pass channels explicitly")
    channels = [_identifier(c, "camera_id") for c in channels]
    seeds = spec.get("seeds", [0, 1, 2])
    if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds)
            or any(isinstance(s, bool) or not isinstance(s, int) or s not in (0, 1, 2) for s in seeds)):
        raise ValueError("seeds must be a nonempty selection of the three established initializations [0,1,2]")
    minimum = spec.get("minimum_training_images", 4)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 4:
        raise ValueError("minimum_training_images must be an integer at least four")
    plans, results = {}, {}
    for model in models:
        for channel in channels:
            plan = _view_plan(spec, images, model, channel)
            for row in plan:
                sources.path(row["path"], f"{model}/{channel}/{row['split']}")
            key = f"{model}/{channel}"
            plans[key] = plan
            results[key] = {"status": "PLANNED", "output": str(out / model / channel / "calibration.json")}
    if not execute:
        return _stage_result(out, results, False, sources)
    out.mkdir(parents=True, exist_ok=False)
    failures = False
    for model, settings in models.items():
        size = settings.get("resolution_px", settings.get("image_size"))
        for channel in channels:
            key = f"{model}/{channel}"; directory = out / model / channel
            observations, names, held, records = [], [], [], []
            for view in plans[key]:
                path, split = view["path"], view["split"]
                record = {"image": str(path), "split": split, "detected": False, "corners": []}
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    record["error"] = "Image decoding failed"
                else:
                    record["resolution_px"] = [image.shape[1], image.shape[0]]
                    if record["resolution_px"] != list(size):
                        raise ValueError(f"Image size does not match declared model resolution: {path}")
                    try:
                        ok, corners = cv2.findChessboardCornersSB(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), pattern,
                            flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
                        if ok and corners is not None and corners.size == len(objects)*2 and np.isfinite(corners).all():
                            corners = corners.reshape(-1, 2)
                            record.update(detected=True, corners=corners.tolist())
                            if split == "train":
                                observations.append(corners); names.append(path.name)
                            else:
                                held.append((path.name, corners))
                            overlay = image.copy()
                            cv2.drawChessboardCorners(overlay, pattern, corners.reshape(-1, 1, 2), True)
                            for i, (x, y) in enumerate(corners):
                                cv2.putText(overlay, str(i), (round(x)+3, round(y)-3),
                                            cv2.FONT_HERSHEY_SIMPLEX, .34, (0, 0, 255), 1, cv2.LINE_AA)
                        else:
                            record["error"] = "No finite complete image-observed grid"
                            overlay = image
                    except cv2.error as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        overlay = image
                    _write_image(directory / "detections" / path.name, overlay)
                records.append(record)
            _write(directory / "detections.json", records)
            document = {
                "status": "INSUFFICIENT_DETECTIONS", "global_uniqueness_proved": False,
                "camera_model": model, "camera_id": channel, "image_resolution_px": list(size),
                "pattern_size": list(pattern), "square_size": pitch,
                "training_images": names, "heldout_images": [name for name, _ in held],
                "training_complete": len(observations), "heldout_complete": len(held),
                "expected_training": sum(v["split"] == "train" for v in plans[key]),
                "unknowns": _unknowns(model), "acquisition_protocol": PROTOCOL,
                "fixed_camera_and_one_plane_enforced_in_fit": True,
                "cross_camera_bundle_adjustment": False, "truth_read_by_solver": False,
                "solutions": [], "failed_seeds": [],
                "interpretation": "Fixed-camera parallel-plane metric constraints are degenerate for pinhole; low residual or local stability proves neither global uniqueness nor accuracy.",
            }
            if len(observations) >= minimum:
                array = np.asarray(observations, dtype=float)
                for seed in seeds:
                    try:
                        result = fit(objects, array, model, size, seed)
                        result["heldout"] = []
                        for name, observation in held:
                            try:
                                result["heldout"].append({"image": name, **heldout(objects, observation, model,
                                                               np.asarray(result["K"]), np.asarray(result["D"]))})
                            except EXPECTED_ERRORS as exc:
                                result["heldout"].append({"image": name, "error": f"{type(exc).__name__}: {exc}"})
                        document["solutions"].append(result)
                        _write(directory / f"seed_{seed}.json", result)
                    except EXPECTED_ERRORS as exc:
                        failure = {"seed": seed, "status": "FIT_FAILED", "error": f"{type(exc).__name__}: {exc}"}
                        document["failed_seeds"].append(failure)
                        _write(directory / f"seed_{seed}.json", failure)
                if document["solutions"]:
                    best = min(document["solutions"], key=lambda row: row["training_rmse_px"])
                    close = [r for r in document["solutions"] if r["training_rmse_px"] <= best["training_rmse_px"]+.10]
                    fxs = [r["K"][0][0] for r in close]
                    spread = (max(fxs)-min(fxs))/np.mean(fxs) if len(fxs) > 1 else 0.
                    status = ("DEGENERATE_PARALLEL_PLANE_ACQUISITION" if model == "pinhole" else
                              "MULTISTART_INCOMPLETE" if document["failed_seeds"] else
                              "SINGLE_START_LOCAL_SOLUTION" if len(seeds) == 1 else
                              "MULTISTART_UNSTABLE" if spread > .1 else "MULTISTART_STABLE_LOCAL_SOLUTION")
                    document.update(status=status, best_seed=best["seed"], best_training_rmse_px=best["training_rmse_px"],
                        near_equal_fit_threshold_px=.10, near_equal_fit_seeds=[r["seed"] for r in close],
                        near_equal_fit_fx_relative_spread=float(spread))
                    try:
                        document["homography_constraints"] = homography_spectrum(objects, array)
                    except EXPECTED_ERRORS as exc:
                        document["homography_constraints"] = {"status": "UNAVAILABLE", "error": str(exc)}
                else:
                    document["status"] = "INTRINSIC_FIT_FAILED"
            failures |= (not document["solutions"] or bool(document["failed_seeds"])
                         or len(held) != sum(v["split"] == "holdout" for v in plans[key])
                         or any("error" in h for s in document["solutions"] for h in s["heldout"]))
            target = directory / "calibration.json"; _write(target, document)
            results[key] = {"status": document["status"], "output": str(target),
                            "training_complete": len(observations), "heldout_complete": len(held)}
    return _stage_result(out, results, True, sources, failures)


def validate_pixels(im, uv, nr, nc, model):
    points=uv.reshape(-1,2); accepted=[]; details=[]
    angles=np.linspace(0,2*np.pi,40,endpoint=False)
    for x,y in points:
        good=False;circles=[]
        for radius in [4,6,8,12]:
            xx=np.clip(np.rint(x+radius*np.cos(angles)).astype(int),0,im.shape[1]-1)
            yy=np.clip(np.rint(y+radius*np.sin(angles)).astype(int),0,im.shape[0]-1)
            color=im[yy,xx].astype(int); neutral=color.max(1)-color.min(1)<20
            white=(color.min(1)>135)&neutral; black=color.max(1)<65
            labels=np.where(white,1,np.where(black,0,-1)); labels=labels[labels>=0]
            transitions=int((labels!=np.roll(labels,1)).sum()) if len(labels) else 0
            valid=bool(white.mean()>=.15 and black.mean()>=.15 and len(labels)>=28 and 4<=transitions<=8)
            circles.append({'radius_px':radius,'classified_samples':len(labels),'transitions':transitions,'passed':valid})
            good|=valid
        accepted.append(good); details.append(circles)
    accepted=np.array(accepted); geometry={}
    if model=='pinhole' and accepted.sum()>=6:
        grid=np.array([[x,y] for y in range(nr) for x in range(nc)],float)
        H,inliers=cv2.findHomography(grid[accepted],points[accepted],cv2.RANSAC,2.)
        if H is not None:
            prediction=cv2.perspectiveTransform(grid[:,None],H).reshape(-1,2)
            errors=np.linalg.norm(prediction-points,axis=1)
            geometry={'homography_residuals_px':errors.tolist(),'threshold_px':2.}
            accepted&=errors<2.
    enough=int(accepted.sum())>=6 and (model=='fisheye' or accepted.mean()>=.7)
    return accepted,{'stage':'observed_black_white_cross_and_plane_quality','accepted':bool(enough),
        'accepted_indices':np.flatnonzero(accepted).tolist(),'rejected_indices':np.flatnonzero(~accepted).tolist(),
        'circular_photometric_tests':details,'plane_geometry':geometry,
        'thresholds':'40 samples at radii 4/6/8/12 px; achromatic white min>135/chroma<20; black max<65; >=28 classified, >=15% each color, 4..8 transitions. Pinhole RANSAC residual<2 px.'}


def _world_geometry(document):
    if document.get("units", "m") not in ("m", "metres", "meters"):
        raise ValueError("Public world corner geometry must use metres")
    corners = document.get("corners")
    if not isinstance(corners, list) or not corners:
        raise ValueError("corners_world requires a nonempty corners list")
    regions, ids = {}, set()
    for row in corners:
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
            raise ValueError("World corner IDs must be nonempty and unique")
        ids.add(row["id"])
        region = row.get("region")
        r, c = row.get("grid_row"), row.get("grid_col")
        # Legacy public IDs explicitly encode grid identities; no camera-name
        # or projected-position inference is used.
        encoded = re.fullmatch(r"(.+)/r(-?\d+)/c(-?\d+)", row["id"])
        if encoded:
            region = region if region is not None else encoded.group(1)
            r = r if r is not None else int(encoded.group(2))
            c = c if c is not None else int(encoded.group(3))
        if (not isinstance(region, str) or not region
                or any(isinstance(v, bool) or not isinstance(v, int) for v in (r, c))):
            raise ValueError("Each corner needs explicit region, integer grid_row and grid_col")
        if "xyz_world_m" in row:
            point = row["xyz_world_m"]
        elif "world_xyz_m" in row:
            point = row["world_xyz_m"]
        elif "xyz_m" in row and document.get("coordinate_frame") == "world":
            point = row["xyz_m"]
        else:
            raise ValueError("World corners require xyz_world_m, or explicitly world-qualified xyz_m")
        point = _array(point, (3,), "world corner")
        target = regions.setdefault(region, {"cells": {}})
        if (r, c) in target["cells"]:
            raise ValueError("Duplicate physical region/row/column identity")
        target["cells"][r, c] = {"id": row["id"], "point": point}
    for region, geometry in regions.items():
        cells = geometry["cells"]
        geometry["rows"] = sorted({r for r, _ in cells})
        geometry["columns"] = sorted({c for _, c in cells})
        if len(geometry["rows"]) < 3 or len(geometry["columns"]) < 3:
            raise ValueError(f"Region {region} needs at least three rows and columns")
        points = np.array([r["point"] for r in cells.values()])
        if not np.allclose(points[:, 2], points[0, 2], atol=1e-8, rtol=0):
            raise ValueError("Conditional planar solver currently requires horizontal world regions")
        if np.linalg.matrix_rank(points[:, :2]-points[:, :2].mean(0)) != 2:
            raise ValueError("World region corners must span a two-dimensional plane")
    return regions


def _region_settings(value, geometries):
    settings = {"region": value} if isinstance(value, str) else copy.deepcopy(value)
    if not isinstance(settings, dict) or settings.get("region") not in geometries:
        raise ValueError("Every camera needs an explicit region present in public corners")
    geometry = geometries[settings["region"]]
    nc, nr = len(geometry["columns"]), len(geometry["rows"])
    patterns = settings.get("patterns")
    if patterns is None and "pattern_size" in settings:
        patterns = [settings["pattern_size"]]
    if patterns is None:
        patterns = [[nc, min(nr, 4)]]
        if nr > 4 and nc <= nr and [nc, nc] not in patterns:
            patterns.append([nc, nc])
    if (not isinstance(patterns, list) or not patterns
            or any(not isinstance(p, (list, tuple)) or len(p) != 2
                   or any(isinstance(v, bool) or not isinstance(v, int) or v < 3 for v in p)
                   or p[0] != nc or p[1] > nr for p in patterns)):
        raise ValueError("Detection patterns must fit the public region grid and use its declared columns")
    settings["patterns"] = [tuple(p) for p in patterns]
    settings.setdefault("allow_larger", nr > 4)
    settings.setdefault("rectify", nr > 4)
    if any(not isinstance(settings[k], bool) for k in ("allow_larger", "rectify")):
        raise ValueError("allow_larger and rectify must be boolean")
    return settings


def detect_large(image, geometry, model, intrinsic, *, settings):
    """Return measured raw-image grid candidates and diagnostics without writing."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    if settings["allow_larger"]:
        flags |= cv2.CALIB_CB_LARGER
    attempts, extras = [], {}
    expected_columns, maximum_rows = len(geometry["columns"]), len(geometry["rows"])

    def detect(source, stage):
        for pattern in settings["patterns"]:
            try:
                ok, points, meta = cv2.findChessboardCornersSBWithMeta(source, pattern, flags)
                event = {"stage": stage, "pattern": list(pattern), "detected": bool(ok)}
                attempts.append(event)
                if ok and points is not None and meta is not None:
                    nr, nc = meta.shape
                    event["shape"] = [nr, nc]
                    if nc != expected_columns and nr == expected_columns:
                        points = points.reshape(nr, nc, 2).transpose(1, 0, 2).reshape(-1, 1, 2)
                        nr, nc = nc, nr
                    if (nc == expected_columns and nr <= maximum_rows
                            and points.size == nr*nc*2 and np.isfinite(points).all()):
                        return points, nr, nc
                    event["rejection"] = "Observed grid does not fit the public region"
            except cv2.error as exc:
                attempts.append({"stage": stage, "pattern": list(pattern), "detected": False,
                                 "error": f"{type(exc).__name__}: {exc}"})
        return None

    found = detect(gray, "original_pixels")
    if found:
        points, nr, nc = found
        if model == "pinhole" and nr == nc:
            grid = np.array([[x, y] for y in range(nr) for x in range(nc)], float)
            raw = points.reshape(-1, 2).copy()
            H, inliers = cv2.findHomography(grid, raw, cv2.RANSAC, 2.)
            if H is not None and inliers is not None and int(inliers.sum()) >= 6:
                predictions = cv2.perspectiveTransform(grid[:, None], H).reshape(-1, 2)
                features = cv2.goodFeaturesToTrack(gray, 1000, .0001, 3, blockSize=5, useHarrisDetector=True)
                if features is None:
                    attempts.append({"stage": "pixel_Harris_refinement_after_grid_RANSAC",
                                     "accepted": False, "error": "No measured Harris features"})
                else:
                    refined, distances = [], []
                    for prediction in predictions:
                        candidates = features.reshape(-1, 2)
                        distance = np.linalg.norm(candidates-prediction, axis=1)
                        corner = np.array([[candidates[distance.argmin()]]], np.float32)
                        distances.append(float(distance.min()))
                        cv2.cornerSubPix(gray, corner, (5, 5), (-1, -1), (3, 60, .001))
                        refined.append(corner.ravel().copy())
                    H2, _ = cv2.findHomography(grid, np.asarray(refined))
                    error = None
                    if H2 is not None:
                        projected = cv2.perspectiveTransform(grid[:, None], H2).reshape(-1, 2)
                        error = float(np.sqrt(np.mean(np.sum((projected-refined)**2, axis=1))))
                    accepted = max(distances) < 20 and error is not None and error < 1.
                    attempts.append({
                        "stage": "pixel_Harris_refinement_after_grid_RANSAC", "accepted": accepted,
                        "raw_SB_pixels": raw.tolist(), "RANSAC_inliers": inliers.ravel().tolist(),
                        "nearest_observed_feature_distances_px": distances,
                        "refined_observed_pixels": np.asarray(refined).tolist(), "grid_homography_rmse_px": error,
                        "method": "Homography limits search regions; observations remain image Harris features and cornerSubPix locations.",
                    })
                    if accepted:
                        points = np.asarray(refined, np.float32).reshape(-1, 1, 2)
        return points, nr, nc, attempts, extras
    if not settings["rectify"] or not intrinsic.get("solutions"):
        return None, None, None, attempts, extras
    candidates = [s for s in intrinsic["solutions"] if "all_parameters" in s]
    if not candidates:
        attempts.append({"stage": "image_only_plane_rectification", "detected": False,
                         "error": "No image-estimated reference-plane parameters available"})
        return None, None, None, attempts, extras
    solution = min(candidates, key=lambda row: row["training_rmse_px"])
    K, D = _lens(solution["K"], solution["D"], model)
    nl = 10 if model == "pinhole" else 9
    p = np.asarray(solution["all_parameters"], float)
    if len(p) < nl+6 or not np.isfinite(p).all():
        raise ValueError("Invalid estimated reference-plane parameters")
    R, t = cv2.Rodrigues(p[nl:nl+3])[0], p[nl+3:nl+6]
    size, step = 1600, .005
    xx, yy = np.meshgrid(np.arange(size)*step-4, np.arange(size)*step-4)
    plane = np.stack([xx, yy, np.zeros_like(xx)], axis=-1)
    mapping = pixels((plane @ R.T+t).reshape(-1, 3), K, D, model).reshape(size, size, 2)
    rectified = cv2.remap(image, mapping[:, :, 0].astype(np.float32),
                          mapping[:, :, 1].astype(np.float32), cv2.INTER_LINEAR)
    extras["image_only_plane_rectification.png"] = rectified
    extras["rectification.json"] = {
        "resolution_px": [size, size], "plane_origin_pixels": [800, 800],
        "metres_per_pixel": step, "estimated_R": R.tolist(), "estimated_t": t.tolist(),
        "intrinsic_seed": solution["seed"], "K": K.tolist(), "D": D.tolist(),
        "source": "Image-estimated small-board plane; no rendering camera truth",
    }
    found = detect(cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY), "image_only_plane_rectification")
    if not found:
        return None, None, None, attempts, extras
    points, nr, nc = found
    q = points.reshape(-1, 2)*step-4
    camera = np.c_[q, np.zeros(len(q))] @ R.T+t
    raw = pixels(camera, K, D, model).astype(np.float32).reshape(-1, 1, 2)
    cv2.cornerSubPix(gray, raw, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, .001))
    return raw, nr, nc, attempts, extras


def _conditional_candidates(uv, observed_indices, nr, nc, geometry, intrinsic, model):
    candidates, rejected = [], []
    rows, columns = geometry["rows"], geometry["columns"]
    seen_seeds = set()
    for lens in intrinsic.get("solutions", []):
        seed = lens.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed in seen_seeds:
            raise ValueError("Intrinsic candidate seeds must be explicit unique integers")
        seen_seeds.add(seed)
        _lens(lens["K"], lens["D"], model)
    for offset in range(len(rows)-nr+1):
        window = rows[offset:offset+nr]
        if any(b-a != 1 for a, b in zip(window, window[1:])):
            continue
        if any((r, c) not in geometry["cells"] for r in window for c in columns):
            continue
        ids = np.asarray([[geometry["cells"][r, c]["id"] for c in columns] for r in window])
        points = np.asarray([[geometry["cells"][r, c]["point"] for c in columns] for r in window])
        for turns in ([0, 1, 2, 3] if nr == nc else [0, 2]):
            obj = np.rot90(points, turns, axes=(0, 1)).reshape(-1, 3)[observed_indices].copy()
            mapped = np.rot90(ids, turns).ravel()[observed_indices].tolist()
            for lens in intrinsic.get("solutions", []):
                identity = {"intrinsic_seed": lens["seed"], "row_start": window[0], "quarter_turns": turns}
                K, D = _lens(lens["K"], lens["D"], model)
                K0 = K.copy(); K0[0, 1] = 0
                corrected = uv.copy()
                corrected[:, 0] -= K[0, 1]/K[1, 1]*(corrected[:, 1]-K[1, 2])
                try:
                    if model == "fisheye":
                        corrected = cv2.fisheye.undistortPoints(corrected.reshape(-1, 1, 2), K0, D, P=K0).reshape(-1, 2)
                    else:
                        undistort = getattr(cv2, "undistortPointsIter", cv2.undistortPoints)
                        corrected = undistort(corrected[:, None], K0, D, R=None, P=K0,
                                              criteria=(3, 80, 1e-10)).reshape(-1, 2)
                    _, mask = cv2.findHomography(obj[:, :2], corrected, cv2.RANSAC, 2.)
                    if mask is None or int(mask.sum()) < 6:
                        rejected.append({**identity, "error": "Fewer than six metric-grid RANSAC inliers under this lens"})
                        continue
                    used = np.flatnonzero(mask.ravel()); objfit, uvfit = obj[used], uv[used]
                    good, rv, tv = cv2.solvePnP(objfit, corrected[used], K0, np.zeros(5))
                    if not good:
                        rejected.append({**identity, "error": "PnP initialization failed"})
                        continue
                    def residual(p):
                        return (pixels(objfit @ cv2.Rodrigues(p[:3])[0].T+p[3:], K, D, model)-uvfit).ravel()
                    result = least_squares(residual, np.r_[rv.ravel(), tv.ravel()], max_nfev=120)
                    R, t = cv2.Rodrigues(result.x[:3])[0], result.x[3:]
                    camera = obj @ R.T+t
                    if not np.isfinite(camera).all() or camera[:, 2].min() <= 0:
                        rejected.append({**identity, "error": "Non-finite or nonpositive depth"})
                        continue
                    predicted = pixels(camera, K, D, model)
                    if not np.isfinite(predicted).all():
                        rejected.append({**identity, "error": "Non-finite raw-image projection"})
                        continue
                    candidates.append({
                        **identity, "corner_ids": np.asarray(mapped)[used].tolist(),
                        "all_corner_ids": mapped, "object_points_world_m": objfit.tolist(),
                        "R_w2c": R.tolist(), "tvec": t.tolist(), "camera_center_world_m": (-R.T@t).tolist(),
                        "reprojection_rmse_px": float(np.sqrt(np.mean(np.sum((predicted[used]-uvfit)**2, axis=1)))),
                        "all_correspondence_rmse_px": float(np.sqrt(np.mean(np.sum((predicted-uv)**2, axis=1)))),
                        "observation_indices": used.tolist(), "raw_grid_fit_indices": observed_indices[used].tolist(),
                        "fit_corner_count": len(used), "all_correspondence_count": len(uv),
                        "projected_pixels": predicted[used].tolist(), "projected_all_correspondences": predicted.tolist(),
                        "minimum_depth_m": float(camera[:, 2].min()), "optimizer_success": bool(result.success),
                        "optimizer_message": result.message, "nfev": result.nfev,
                        "interpretation": "Conditional pose; low subset residual does not resolve the lens or physical corner identities.",
                    })
                except EXPECTED_ERRORS as exc:
                    rejected.append({**identity, "error": f"{type(exc).__name__}: {exc}"})
    return candidates, rejected


def calibrate_extrinsics(images, intrinsics, corners_world, out_dir, *, regions=None, models=None, execute=False):
    """Fit independent conditional world-to-camera candidates.

    regions explicitly maps camera IDs to public target regions; no camera name
    convention is inferred. Values may be a region name or a dict containing
    region, patterns, allow_larger, rectify and an optional image filename.
    The default image filename is the existing big_board_fixed.png convention.
    """
    if not isinstance(execute, bool):
        raise TypeError("execute must be boolean")
    out = _new_output(out_dir)
    images, intrinsics = Path(images).expanduser().resolve(strict=True), Path(intrinsics).expanduser().resolve(strict=True)
    sources = _Sources(); sources.path(__file__, "calibration_module")
    public = sources.document(corners_world, "public_board_world_geometry")
    geometries = _world_geometry(public)
    regions = regions if regions is not None else public.get("camera_regions")
    if not isinstance(regions, Mapping) or not regions:
        raise ValueError("Pass an explicit camera-to-region map; camera names do not imply regions")
    settings = {_identifier(channel, "camera_id"): _region_settings(value, geometries)
                for channel, value in regions.items()}
    if models is None:
        models = [model for model in MODELS if (images / model).is_dir()]
    if (not isinstance(models, (list, tuple)) or not models or len(set(models)) != len(models)
            or set(models)-set(MODELS)):
        raise ValueError("models must select pinhole and/or fisheye")
    plans, results = {}, {}
    for model in models:
        for channel, target in settings.items():
            path = Path(target.get("image", "big_board_fixed.png")).expanduser()
            path = path if path.is_absolute() else images / model / channel / path
            path = sources.path(path, f"{model}/{channel}/large_board_image")
            lens_path = sources.path(intrinsics / model / channel / "calibration.json", f"{model}/{channel}/intrinsics")
            lens = sources.document(lens_path, f"{model}/{channel}/intrinsics")
            if lens.get("camera_id", channel) != channel or lens.get("camera_model", model) != model:
                raise ValueError("Intrinsic camera/model identity differs from the requested image")
            key = f"{model}/{channel}"
            plans[key] = (path, lens)
            results[key] = {"status": "PLANNED", "output": str(out / model / channel / "extrinsics.json")}
    if not execute:
        return _stage_result(out, results, False, sources)
    out.mkdir(parents=True, exist_ok=False)
    failures = False
    for model in models:
        for channel, target in settings.items():
            key = f"{model}/{channel}"; path, intrinsic = plans[key]
            directory = out / model / channel
            geometry = geometries[target["region"]]
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            document = {
                "camera_id": channel, "model": model, "image": str(path),
                "image_relative_to_images": str(path.relative_to(images)) if images in path.parents else None,
                "region_annotation": target["region"],
                "region_annotation_source": "Caller-supplied public target-region association; no projected camera truth",
                "cross_camera_bundle_adjustment": False,
                "intrinsic_source": "Caller-supplied intrinsic candidates; inspect source_inputs.json and intrinsic records",
                "directed_corner_identity": "Unmarked repeated grid; retain row offsets and direction alternatives",
                "detected": False, "detection_attempts": [], "candidates": [],
                "status": "LARGE_BOARD_DETECTION_FAILED", "global_uniqueness_proved": False,
                "all_correspondence_scope": "Quality-retained observations before per-lens RANSAC, not the complete physical target",
            }
            if image is None:
                document.update(status="IMAGE_DECODE_FAILED", error="Large-board image decoding failed")
            else:
                document["resolution_px"] = [image.shape[1], image.shape[0]]
                work = image.copy(); rgb = image.astype(int)
                work[rgb.max(2)-rgb.min(2) >= 20] = 127
                _write_image(directory / "achromatic_detection_input.png", work)
                document["detection_attempts"].append({
                    "stage": "achromatic_image_preprocessing",
                    "method": "Set chroma>=20 to RGB127 in a detection copy; original PNG remains unchanged",
                })
                observed_indices, all_grid, uv = None, None, None
                for retry in range(4):
                    try:
                        points, nr, nc, events, extras = detect_large(work if retry < 2 else image, geometry,
                                                                    model, intrinsic, settings=target)
                        document["detection_attempts"].extend(events)
                        for name, value in extras.items():
                            destination = directory / name
                            if not destination.exists():
                                _write(destination, value) if name.endswith(".json") else _write_image(destination, value)
                        if points is None:
                            continue
                        mask, quality = validate_pixels(image, points, nr, nc, model)
                        quality.update(retry=retry, raw_grid_pixels=points.reshape(-1, 2).tolist())
                        document["detection_attempts"].append(quality)
                        if quality["accepted"]:
                            all_grid = points.reshape(-1, 2).copy()
                            observed_indices = np.flatnonzero(mask)
                            uv = all_grid[observed_indices].astype(float)
                            document["detected"] = True
                            break
                    except EXPECTED_ERRORS as exc:
                        document["detection_attempts"].append({"stage": "detection_retry", "retry": retry,
                                                               "error": f"{type(exc).__name__}: {exc}"})
                overlay = image.copy()
                if document["detected"]:
                    for i, (x, y) in enumerate(all_grid):
                        color = (0, 255, 0) if i in observed_indices else (0, 0, 255)
                        cv2.circle(overlay, (round(x), round(y)), 4, color, 1)
                        cv2.putText(overlay, str(i), (round(x)+3, round(y)-3),
                                    cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
                    document.update(detected_grid_shape=[nc, nr], observed_pixels=uv.tolist(),
                                    observed_grid_indices=observed_indices.tolist(), all_grid_pixels=all_grid.tolist(),
                                    status="UNRESOLVED_INTRINSICS_AND_CORNER_IDS")
                    if intrinsic.get("solutions"):
                        candidates, rejected = _conditional_candidates(uv, observed_indices, nr, nc,
                                                                       geometry, intrinsic, model)
                        document["candidates"], document["rejected_candidates"] = candidates, rejected
                        document["candidate_count"] = len(candidates)
                        if candidates:
                            best = min(candidates, key=lambda row: row["reprojection_rmse_px"])
                            document["minimum_conditional_reprojection_rmse_px"] = best["reprojection_rmse_px"]
                            residual_overlay = overlay.copy()
                            for p, q in zip(uv[best["observation_indices"]], np.asarray(best["projected_pixels"])):
                                cv2.circle(residual_overlay, tuple(np.rint(q).astype(int)), 3, (255, 0, 255), 1)
                                cv2.line(residual_overlay, tuple(np.rint(p).astype(int)),
                                         tuple(np.rint(q).astype(int)), (0, 0, 255), 1)
                            _write_image(directory / "conditional_reprojection.png", residual_overlay)
                        else:
                            document["status"] = "NO_VALID_CONDITIONAL_POSES"
                    else:
                        document["status"] = "INTRINSIC_SOLUTIONS_UNAVAILABLE"
                _write_image(directory / "detection.png", overlay)
            filename = directory / "extrinsics.json"; _write(filename, document)
            failures |= not bool(document["candidates"])
            results[key] = {"status": document["status"], "output": str(filename),
                            "detected": document["detected"], "candidate_count": len(document["candidates"])}
    return _stage_result(out, results, True, sources, failures)
