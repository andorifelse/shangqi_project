"""Pixel-domain KB4 pose fitting; inputs contain observations, never camera truth."""
import cv2
import numpy as np
from scipy.optimize import least_squares


def project(objects, pose, k, d):
    return cv2.fisheye.projectPoints(np.asarray(objects, np.float64)[:, None],
        pose[:3], pose[3:], k, d)[0].reshape(-1, 2)


def fit_pose(objects, pixels, k, d, keep=None):
    """Refine planar PnP alternatives in distorted pixels and enforce cheirality.

    Candidate ranking and optimization use only the selected corners, including
    when the remaining corners form an independent validation set.
    """
    objects = np.asarray(objects, np.float64)
    pixels = np.asarray(pixels, np.float64)
    if keep is None:
        keep = np.ones(len(objects), bool)
    xyz, uv = objects[keep], pixels[keep]
    ideal = cv2.fisheye.undistortPoints(uv[:, None], k, d).reshape(-1, 2)
    if not np.isfinite(ideal).all() or np.max(np.abs(ideal)) > 1e5:
        raise ValueError('角点超出可逆鱼眼范围')
    seeds = []
    for flag in (cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_EPNP):
        try:
            result = cv2.solvePnPGeneric(xyz, ideal, np.eye(3), np.zeros(5), flags=flag)
            if result[0]:
                seeds.extend(np.r_[r.ravel(), t.ravel()] for r, t in zip(result[1], result[2]))
        except cv2.error:
            continue
    candidates = []
    for seed in seeds:
        opt = least_squares(lambda p: (project(xyz, p, k, d)-uv).ravel(), seed,
            max_nfev=150, ftol=1e-10, xtol=1e-10, gtol=1e-8)
        depths = (xyz @ cv2.Rodrigues(opt.x[:3])[0].T + opt.x[3:])[:, 2]
        if opt.success and np.isfinite(opt.x).all() and (depths > 1e-6).all():
            candidates.append((float(np.mean(opt.fun**2)), opt.x))
    if not candidates:
        raise ValueError('未找到所有拟合角点均在相机前方的位姿')
    return min(candidates, key=lambda row: row[0])[1]


def refine_intrinsics(objects, pixels, k, d):
    """Joint lens and independent-image poses, with an analytic sparse Jacobian."""
    from scipy.sparse import lil_matrix, csr_matrix
    objects = np.asarray(objects, np.float64)
    pixels = np.asarray(pixels, np.float64)
    poses = np.array([fit_pose(objects, uv, k, d) for uv in pixels])
    initial = np.r_[k[0, 0], k[1, 1], k[0, 2], k[1, 2], d.ravel(), poses.ravel()]
    def unpack(p):
        lens = np.array([[p[0], 0., p[2]], [0., p[1], p[3]], [0., 0., 1.]])
        return lens, p[4:8], p[8:].reshape(-1, 6)
    def residual(p):
        lens, distortion, extrinsics = unpack(p)
        return np.concatenate([(project(objects, pose, lens, distortion)-uv).ravel()
            for pose, uv in zip(extrinsics, pixels)])
    def jacobian(p):
        lens, distortion, extrinsics = unpack(p)
        rows = len(objects)*2
        jac = lil_matrix((len(pixels)*rows, len(p)))
        for i, pose in enumerate(extrinsics):
            _, j = cv2.fisheye.projectPoints(objects[:, None], pose[:3], pose[3:], lens, distortion)
            jac[i*rows:(i+1)*rows, :8] = j[:, :8]
            jac[i*rows:(i+1)*rows, 8+i*6:14+i*6] = j[:, 8:14]
        return csr_matrix(jac)
    lower = np.full(len(initial), -np.inf); lower[:2] = 1.
    opt = least_squares(residual, initial, jac=jacobian, bounds=(lower, np.inf),
        x_scale='jac', loss='soft_l1', f_scale=1., max_nfev=150,
        ftol=1e-9, xtol=1e-9, gtol=1e-7)
    lens, distortion, poses = unpack(opt.x)
    errors = residual(opt.x).reshape(len(pixels), len(objects), 2)
    rms = float(np.sqrt(np.mean(np.sum(errors**2, axis=2))))
    return rms, lens, distortion, poses[:, :3], poses[:, 3:]
