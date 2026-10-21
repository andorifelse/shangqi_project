"""Reproducible synthetic checkerboard images and OpenCV baseline calibration."""
from __future__ import annotations
import csv
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scene_manager.ground_truth import board_corners
from scene_manager.ground_truth import GroundTruth
from scene_manager.persistence import save_scene
from scene_manager.models import Scene
from scene_manager.transforms import inverse, pose_matrix, transform_points
from scene_manager.camera_models import project, unproject
from .calibration_renderer import render_board_image


def cv_fisheye_flag(name):
    """OpenCV 4 uses cv2.fisheye flags; OpenCV 5 moved them to cv2."""
    return getattr(cv2.fisheye, name) if hasattr(cv2.fisheye, name) else getattr(cv2, name)


def sample_board_poses(scene: Scene, camera_name: str, count: int, seed: int,
                       board_id: str | None = None) -> list[list[float]]:
    """Sample varied board views in front of a selected optical camera."""
    if camera_name not in scene.cameras or not scene.boards:
        raise ValueError("Camera and at least one board are required")
    if count < 8:
        raise ValueError("At least 8 board views are required for calibration")
    board_id = board_id or next(iter(scene.boards))
    if board_id not in scene.boards:
        raise KeyError(board_id)
    rng = np.random.default_rng(seed)
    camera = scene.cameras[camera_name]
    cols = int(np.ceil(np.sqrt(count * camera.width / camera.height)))
    rows = int(np.ceil(count / cols))
    if camera.model == "pinhole":
        xlo,xhi,ylo,yhi,dlo,dhi = .27,.73,.29,.71,1.7,2.3
    else:
        xlo,xhi,ylo,yhi,dlo,dhi = .20,.80,.22,.78,1.0,1.35
    xs = np.linspace(max(48., camera.width*xlo), min(camera.width-48., camera.width*xhi), cols)
    ys = np.linspace(max(42., camera.height*ylo), min(camera.height-42., camera.height*yhi), rows)
    targets = np.array([(x,y) for y in ys for x in xs])[:count]
    targets += rng.uniform(-.12,.12,targets.shape) * np.array([
        (xs[-1]-xs[0])/max(cols-1,1), (ys[-1]-ys[0])/max(rows-1,1)])
    rng.shuffle(targets)
    center_rays, ray_valid = unproject(camera, targets)
    if not ray_valid.all():
        raise ValueError("Sampling grid includes pixels outside camera calibration domain")
    T_world_camera = scene.camera_optical_pose(camera_name)
    board = scene.boards[board_id]
    board_local_corners = board_corners(board)
    poses = []
    # Keep a seeded candidate sequence, but only return poses whose complete
    # checkerboard remains inside the lens' valid projection domain. A margin
    # prevents tiny pixel-rounding differences from invalidating a saved view.
    max_attempts = max(count * 100, 1000)
    attempts = 0
    while len(poses) < count and attempts < max_attempts:
        target_index = len(poses) if attempts < count else int(rng.integers(0, len(center_rays)))
        ray = center_rays[target_index]
        attempts += 1
        distance = rng.uniform(dlo, dhi) * max(board.width, board.height) / 1.2
        perturb = Rotation.from_euler("xyz", rng.uniform([-.42,-.38,-.65], [.42,.38,.65])).as_matrix()
        T_camera_board = np.eye(4)
        axis_x = np.cross([0.,1.,0.], ray)
        axis_x /= np.linalg.norm(axis_x)
        axis_y = np.cross(-ray, axis_x)
        T_camera_board[:3,:3] = np.column_stack((axis_x,axis_y,-ray)) @ perturb
        T_camera_board[:3,3] = ray * distance
        T_world_board = T_world_camera @ T_camera_board
        world_corners = transform_points(T_world_board, board_local_corners)
        camera_corners = transform_points(inverse(T_world_camera), world_corners)
        _, corner_valid = project(camera, camera_corners)
        if not corner_valid.all():
            continue
        # Keep every corner a few pixels inside the raster so the pattern and
        # its white border are not clipped even at high distortion/FOV edges.
        uv, _ = project(camera, camera_corners)
        margin_px = 3.
        if (np.any(uv[:,0] < margin_px) or np.any(uv[:,0] > camera.width-1-margin_px)
                or np.any(uv[:,1] < margin_px) or np.any(uv[:,1] > camera.height-1-margin_px)):
            continue
        poses.append([*T_world_board[:3,3], *Rotation.from_matrix(T_world_board[:3,:3]).as_euler("xyz")])
    if len(poses) < count:
        raise RuntimeError(f"Could only sample {len(poses)}/{count} fully visible board poses for {camera_name} "
                           f"after {attempts} attempts; check camera FOV, board size, and camera intrinsics")
    return poses


def detect_corners(image: np.ndarray, board) -> tuple[bool, np.ndarray | None]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    pattern = (board.columns - 1, board.rows - 1)
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
    return bool(found), corners.astype(np.float32) if found else None


def make_calibration_dataset(scene: Scene, output: str | Path, views: int = 24,
                             seed: int = 20261020, supersample: int = 2,
                             camera_names: list[str] | None = None) -> dict:
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    allowed_existing = {"scene.yaml", "ground_truth.yaml", "config.yaml", "logs"}
    unexpected = {p.name for p in output.iterdir()} - allowed_existing
    if unexpected:
        raise FileExistsError(f"Dataset directory already contains generated data: {sorted(unexpected)}")
    scene.validate()
    camera_names = camera_names or sorted(scene.cameras)
    if not camera_names or set(camera_names) - scene.cameras.keys():
        raise ValueError("Requested calibration camera is missing from the scene")
    (output / "cameras").mkdir(exist_ok=True)
    if not scene.boards:
        raise ValueError("Calibration requires a checkerboard in SceneState")
    board_id = next(iter(scene.boards))
    board = scene.boards[board_id]
    if board.rows < 4 or board.columns < 4:
        raise ValueError("Calibration requires at least 4x4 squares")
    if any(scene.cameras[n].model not in ("pinhole","fisheye") for n in camera_names):
        raise ValueError("OpenCV baseline supports pinhole/fisheye only")
    truth = GroundTruth.capture(scene)
    save_scene(scene, output / "scene.yaml")
    truth.save(output / "ground_truth.yaml")
    ground_truth_file_sha256 = hashlib.sha256((output/"ground_truth.yaml").read_bytes()).hexdigest()
    base_corners = board_corners(board).reshape(board.rows-1, board.columns-1, 3)
    metadata = {"dataset_version": 1, "seed": int(seed), "views_per_camera": int(views),
                "scene_sha256": truth.sha256, "ground_truth_file_sha256":ground_truth_file_sha256,
                "opencv_version":cv2.__version__, "numpy_version":np.__version__,
                "renderer":"board_ray_reference", "background":"neutral; no Gaussian occlusion",
                "corner_identity":"GT only resolves checkerboard row/column symmetry; detected pixels are not replaced",
                "supersample": int(supersample), "pattern_inner_corners": [board.columns-1,board.rows-1],
                "board_square_size_m": board.square_size, "cameras": {}, "views": []}
    with (output / "board_poses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["camera","view","board_id","x","y","z","roll","pitch","yaw"])
        writer.writeheader()
        for name in camera_names:
            camera = scene.cameras[name]
            metadata["cameras"][name] = {"model": camera.model, "width": camera.width,
                "height": camera.height, "K": camera.K, "D": camera.D,
                "pattern_inner_corners": [board.columns-1, board.rows-1]}
            camera_seed = int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8],"little")
            poses = sample_board_poses(scene, name, views, camera_seed, board_id)
            for index, pose in enumerate(poses):
                sample = deepcopy(scene)
                sample.boards = {board_id:sample.boards[board_id]}
                sample.boards[board_id].pose = pose
                image = render_board_image(sample, name, supersample)
                relative = Path("cameras") / name / f"view_{index:04d}.png"
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(destination), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                    raise OSError(f"Failed to write {destination}")
                found, detected = detect_corners(image, board)
                if found and detected is not None:
                    T_world_board = pose_matrix(pose)
                    world_points = transform_points(T_world_board, base_corners.reshape(-1,3))
                    ideal, valid = project(camera, transform_points(inverse(scene.camera_optical_pose(name)), world_points))
                    if not valid.all():
                        raise RuntimeError(f"Sampled board {name}/{index} leaves camera domain")
                    ideal = ideal.reshape(board.rows-1, board.columns-1, 2)
                    detected_grid = detected.reshape(board.rows-1, board.columns-1, 2)
                    choices = [(bool(rf),bool(cf),
                                float(np.mean(np.sum((detected_grid[::-1 if rf else 1, ::-1 if cf else 1]-ideal)**2,axis=-1))))
                               for rf in (False,True) for cf in (False,True)]
                    row_flip, col_flip, correspondence_error = min(choices,key=lambda x:x[2])
                    if correspondence_error > 4.:
                        raise RuntimeError(f"Corner ordering/projection mismatch in {name}/{index}: {np.sqrt(correspondence_error):.2f}px RMS")
                else:
                    row_flip = col_flip = False
                writer.writerow({"camera":name,"view":index,"board_id":board_id,
                    **dict(zip(("x","y","z","roll","pitch","yaw"),pose))})
                metadata["views"].append({"camera":name,"view":index,"board_id":board_id,
                    "image":relative.as_posix(),"board_pose":pose,
                    "image_sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),
                    "corner_flip_rows":row_flip,"corner_flip_columns":col_flip,
                    "detected_during_generation":found})
    (output / "metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    manifest={name:hashlib.sha256((output/name).read_bytes()).hexdigest()
              for name in ("scene.yaml","ground_truth.yaml","board_poses.csv","metadata.json")}
    (output/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding="utf-8")
    return metadata


def _rotation_error_degrees(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    return float(np.rad2deg(Rotation.from_matrix(R_est @ R_gt.T).magnitude()))


def calibrate_camera_dataset(dataset: str | Path, camera_name: str, truth,
                             min_views: int = 8) -> dict:
    dataset = Path(dataset).resolve()
    manifest=json.loads((dataset/"dataset_manifest.json").read_text(encoding="utf-8"))
    for file,digest in manifest.items():
        if file not in ("scene.yaml","ground_truth.yaml","board_poses.csv","metadata.json"):
            raise ValueError(f"Unexpected dataset manifest entry: {file}")
        if hashlib.sha256((dataset/file).read_bytes()).hexdigest()!=digest:
            raise ValueError(f"Dataset manifest checksum mismatch: {file}")
    metadata = json.loads((dataset/"metadata.json").read_text(encoding="utf-8"))
    if camera_name not in metadata["cameras"]:
        raise KeyError(f"No dataset views for camera {camera_name}")
    gt_path=dataset/"ground_truth.yaml"
    if not gt_path.is_file() or hashlib.sha256(gt_path.read_bytes()).hexdigest()!=metadata.get("ground_truth_file_sha256"):
        raise ValueError("Dataset Ground Truth file checksum mismatch")
    exported_truth=GroundTruth.load(gt_path)
    if exported_truth.scene.to_dict() != truth.scene.to_dict():
        raise ValueError("Dataset was generated from a different Ground Truth snapshot")
    scene = truth.scene
    camera, board = scene.cameras[camera_name], scene.boards[next(iter(scene.boards))]
    object_grid = board_corners(board).reshape(board.rows-1, board.columns-1, 3)
    object_points, image_points, accepted, rejected = [], [], [], []
    for view in metadata["views"]:
        if view["camera"] != camera_name:
            continue
        path = dataset / view["image"]
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != view["image_sha256"]:
            raise ValueError(f"Dataset image checksum mismatch: {path}")
        if bgr.shape[:2] != (camera.height,camera.width):
            raise ValueError(f"Dataset image resolution disagrees with Ground Truth: {path}")
        found, corners = detect_corners(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),board)
        if not found or corners is None or corners.size != object_grid.shape[0]*object_grid.shape[1]*2:
            rejected.append(view["view"])
            continue
        object_points.append(np.ascontiguousarray(object_grid.reshape(1,-1,3), dtype=np.float32))
        grid = corners.reshape(board.rows-1, board.columns-1, 2)
        grid = grid[::-1 if view.get("corner_flip_rows") else 1,
                    ::-1 if view.get("corner_flip_columns") else 1]
        image_points.append(np.ascontiguousarray(grid.reshape(1,-1,2), dtype=np.float32))
        accepted.append(view)
    if len(object_points) < min_views:
        raise RuntimeError(f"Only {len(object_points)} checkerboards detected for {camera_name}; need {min_views}; rejected={rejected}")
    h,w = camera.height,camera.width
    # Initialize from image geometry only. Do not seed calibration from the
    # Ground Truth intrinsics that are evaluated in the error report.
    focal0 = max(w,h)/(np.pi if camera.model == "fisheye" else 2.)
    K0=np.array([[focal0,0,(w-1)/2],[0,focal0,(h-1)/2],[0,0,1]],np.float64)
    D0=np.zeros((4,1) if camera.model=="fisheye" else (5,1),np.float64)
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT,200,1e-11)
    if camera.model=="fisheye":
        candidates, initialization_failures = [], []
        for fraction in (1/np.pi, .4, .5):
            guess = K0.copy()
            guess[0,0] = guess[1,1] = max(w,h)*fraction
            try:
                fit = cv2.fisheye.calibrate(
                    [p.astype(np.float64) for p in object_points],
                    [p.reshape(1,-1,2).astype(np.float64) for p in image_points],(w,h),guess,D0.copy(),
                    flags=cv_fisheye_flag("CALIB_USE_INTRINSIC_GUESS")|cv_fisheye_flag("CALIB_RECOMPUTE_EXTRINSIC")|cv_fisheye_flag("CALIB_FIX_SKEW"),
                    criteria=criteria)
                if np.isfinite(fit[0]) and np.isfinite(fit[1]).all() and np.isfinite(fit[2]).all() and min(fit[1][0,0],fit[1][1,1])>0:
                    candidates.append(fit)
            except cv2.error as exc:
                initialization_failures.append(str(exc).splitlines()[0])
        if not candidates:
            raise RuntimeError(f"OpenCV fisheye initialization failed for {camera_name}: {initialization_failures}")
        rms,K,D,rvecs,tvecs=min(candidates,key=lambda fit:fit[0])
        project=lambda obj,rv,tv:cv2.fisheye.projectPoints(obj.astype(np.float64),rv,tv,K,D)[0]
    elif camera.model=="pinhole":
        rms,K,D,rvecs,tvecs=cv2.calibrateCamera([p.reshape(-1,1,3) for p in object_points],
                                                 [p.reshape(-1,1,2) for p in image_points],(w,h),K0,D0,
                                                 flags=cv2.CALIB_USE_INTRINSIC_GUESS,criteria=criteria)
        project=lambda obj,rv,tv:cv2.projectPoints(obj,rv,tv,K,D)[0]
    else:
        raise ValueError("OpenCV baseline supports pinhole and fisheye, not ftheta")
    errs=[]; per_view=[]; t_errors=[]; r_errors=[]; estimates=[]
    T_world_camera_gt=scene.camera_optical_pose(camera_name)
    for obj,img,rv,tv,view in zip(object_points,image_points,rvecs,tvecs,accepted):
        pred=project(obj,rv,tv).reshape(-1,2); observed=img.reshape(-1,2)
        e=np.linalg.norm(pred-observed,axis=1); errs.extend(e.tolist())
        Rcb,_=cv2.Rodrigues(rv); Tcb=np.eye(4);Tcb[:3,:3]=Rcb;Tcb[:3,3]=np.asarray(tv).reshape(3)
        Twb=pose_matrix(view["board_pose"]); Twc_est=Twb@inverse(Tcb)
        estimates.append(Twc_est)
        te=float(np.linalg.norm(Twc_est[:3,3]-T_world_camera_gt[:3,3]))
        re=_rotation_error_degrees(Twc_est[:3,:3],T_world_camera_gt[:3,:3])
        t_errors.append(te);r_errors.append(re)
        per_view.append({"view":view["view"],"reprojection_rms_px":float(np.sqrt(np.mean(e**2))),
                         "translation_error_m":te,"rotation_error_deg":re})
    diffK=K-camera.matrix; dg=np.asarray(camera.D,np.float64).reshape(-1);de=np.asarray(D,np.float64).reshape(-1)
    common=min(len(dg),len(de))
    fused = np.eye(4)
    fused[:3,:3] = Rotation.from_matrix(np.array(estimates)[:,:3,:3]).mean().as_matrix()
    fused[:3,3] = np.mean(np.array(estimates)[:,:3,3],axis=0)
    return {"camera":camera_name,"model":camera.model,"views_detected":len(object_points),
        "views_rejected":rejected,"reprojection_rms_px":float(np.sqrt(np.mean(np.square(errs)))),
        "opencv_rms_px":float(rms),"K_estimated":K.tolist(),"D_estimated":de.tolist(),
        "initialization":"image dimensions only; D=0; no GT intrinsic initial guess",
        "T_world_from_camera_optical_estimated":fused.tolist(),
        "extrinsic_assumption":"Known per-view board world poses; not an unobservable free-world calibration",
        "intrinsic_error":{"fx_px":float(diffK[0,0]),"fy_px":float(diffK[1,1]),
            "cx_px":float(diffK[0,2]),"cy_px":float(diffK[1,2]),
            "focal_relative_percent":float(100*np.linalg.norm([diffK[0,0],diffK[1,1]])/np.linalg.norm([camera.K[0],camera.K[4]]))},
        "distortion_error_est_minus_gt":(de[:common]-dg[:common]).tolist(),
        "distortion_error_l2":float(np.linalg.norm(de[:common]-dg[:common])),
        "extrinsics_error":{"translation_mean_m":float(np.mean(t_errors)),
            "translation_rmse_m":float(np.sqrt(np.mean(np.square(t_errors)))),
            "rotation_mean_deg":float(np.mean(r_errors)),"rotation_rmse_deg":float(np.sqrt(np.mean(np.square(r_errors)))),
            "per_view":per_view}}


def save_calibration_report(report: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
