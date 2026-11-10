"""Portable CPU preparation of SH3 Gaussian scenes; Python 3.7 compatible."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


FIELDS = (["x", "y", "z", "nx", "ny", "nz"]
          + ["f_dc_%d" % i for i in range(3)]
          + ["f_rest_%d" % i for i in range(45)]
          + ["opacity"] + ["scale_%d" % i for i in range(3)]
          + ["rot_%d" % i for i in range(4)])
DTYPE = np.dtype([(name, "<f4") for name in FIELDS])


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Not JSON serializable: " + type(value).__name__)


def _jsonable(value):
    return json.loads(json.dumps(value, allow_nan=False, default=_json_default))


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def _read_document(value, base=None):
    base = Path.cwd() if base is None else Path(base)
    if isinstance(value, dict):
        return _jsonable(value), base, None
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else base/path).resolve()
    return json.loads(path.read_text(encoding="utf-8")), path.parent, path


def _name(value):
    if (not isinstance(value, str) or not value or value in (".", "..")
            or "/" in value or "\\" in value):
        raise ValueError("Unsafe object/variant name: %r" % value)
    return value


def _finite(value, shape, label):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("Expected finite %s with shape %s" % (label, shape))
    return result


def euler_xyz_to_R(rx, ry, rz):
    """Blender XYZ Euler convention: column vectors are acted on by Rz Ry Rx."""
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.]])
            @ np.array([[cy, 0, sy], [0, 1., 0], [-sy, 0, cy]])
            @ np.array([[1., 0, 0], [0, cx, -sx], [0, sx, cx]]))


def rotation_quaternion(rotation):
    """Return a unit wxyz quaternion for a proper rotation."""
    r = _finite(rotation, (3, 3), "rotation")
    if not np.allclose(r@r.T, np.eye(3), atol=1e-7, rtol=0) or not np.isclose(np.linalg.det(r), 1, atol=1e-7):
        raise ValueError("Rotation must be proper and orthonormal")
    trace = np.trace(r)
    if trace > 0:
        s = np.sqrt(trace+1)*2
        q = [s/4, (r[2, 1]-r[1, 2])/s, (r[0, 2]-r[2, 0])/s, (r[1, 0]-r[0, 1])/s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i+1) % 3, (i+2) % 3
        s = np.sqrt(1+r[i, i]-r[j, j]-r[k, k])*2
        q = [(r[k, j]-r[j, k])/s, 0., 0., 0.]
        q[i+1], q[j+1], q[k+1] = s/4, (r[j, i]+r[i, j])/s, (r[k, i]+r[i, k])/s
    q = np.asarray(q, dtype=np.float64)
    return q/np.linalg.norm(q)


def _quat_mul(a, b):
    w, x, y, z = np.asarray(a).T
    p, q, r, s = np.asarray(b).T
    return np.stack((w*p-x*q-y*r-z*s, w*q+x*p+y*s-z*r,
                     w*r-x*s+y*p+z*q, w*s+x*r-y*q+z*p), axis=-1)


def _sh_band(degree, directions):
    x, y, z = directions.T
    if degree == 1:
        c = .4886025119029199
        return np.stack((-c*y, c*z, -c*x), axis=1)
    if degree == 2:
        return np.stack((1.0925484305920792*x*y, -1.0925484305920792*y*z,
                         .31539156525252005*(2*z*z-x*x-y*y), -1.0925484305920792*x*z,
                         .5462742152960396*(x*x-y*y)), axis=1)
    return np.stack((-.5900435899266435*y*(3*x*x-y*y), 2.890611442640554*x*y*z,
                     -.4570457994644658*y*(4*z*z-x*x-y*y),
                     .3731763325901154*z*(2*z*z-3*x*x-3*y*y),
                     -.4570457994644658*x*(4*z*z-x*x-y*y),
                     1.445305721320277*z*(x*x-y*y), -.5900435899266435*x*(x*x-3*y*y)), axis=1)


def sh_rotation(degree, rotation):
    """Graphdeco real-SH order; f_world(w) = f_local(R.T @ w)."""
    rng = np.random.default_rng(0)
    directions = rng.standard_normal((8*(2*degree+1), 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return np.linalg.lstsq(_sh_band(degree, directions),
                           _sh_band(degree, directions@rotation), rcond=None)[0]


def canonical_gaussians(vertices):
    """Normalize scalar properties by name; absent normals are compatibility zeros."""
    names = set(vertices.dtype.names or ())
    missing = set(FIELDS)-names
    if missing-{"nx", "ny", "nz"}:
        raise ValueError("Missing SH3 Gaussian fields: " + str(sorted(missing-{"nx", "ny", "nz"})))
    result = np.zeros(len(vertices), dtype=DTYPE)
    limit = np.finfo(np.float32).max
    for name in FIELDS:
        if name not in names:
            continue
        values = vertices[name]
        if values.ndim != 1 or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
            raise ValueError("Need finite scalar Gaussian property: " + name)
        if np.any(np.abs(values.astype(np.float64)) > limit):
            raise ValueError("Gaussian property exceeds float32: " + name)
        result[name] = values
    if not len(result):
        raise ValueError("Gaussian asset must be nonempty")
    q = np.column_stack([result["rot_%d" % i] for i in range(4)]).astype(np.float64)
    if np.any(np.linalg.norm(q, axis=1) < 1e-12):
        raise ValueError("Gaussian quaternion must be nonzero")
    return result


def read_gaussians(path):
    from plyfile import PlyData
    ply = PlyData.read(str(path), mmap="r")
    if "vertex" not in ply:
        raise ValueError("PLY has no Gaussian vertex element")
    return canonical_gaussians(ply["vertex"].data)


def bake_gaussians(vertices, rotation, scale, translation):
    """Apply a positive uniform Sim3 to raw PLY parameters, including all SH3 bands."""
    r = _finite(rotation, (3, 3), "rotation")
    q_r = rotation_quaternion(r)
    t = _finite(translation, (3,), "translation")
    if isinstance(scale, bool) or not np.isscalar(scale) or not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("Scale must be one positive finite scalar")
    scale = float(scale)
    output = canonical_gaussians(vertices)
    positions = np.column_stack([output[name] for name in ("x", "y", "z")]).astype(np.float64)
    positions = (positions*scale)@r.T+t
    normals = np.column_stack([output[name] for name in ("nx", "ny", "nz")]).astype(np.float64)@r.T
    for i, name in enumerate(("x", "y", "z")):
        output[name] = positions[:, i]
    for i, name in enumerate(("nx", "ny", "nz")):
        output[name] = normals[:, i]
    q = np.column_stack([output["rot_%d" % i] for i in range(4)]).astype(np.float64)
    q = _quat_mul(np.broadcast_to(q_r, q.shape), q)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    for i in range(4):
        output["rot_%d" % i] = q[:, i]
    for i in range(3):
        output["scale_%d" % i] += math.log(scale)
    features = np.column_stack([output["f_rest_%d" % i] for i in range(45)]).astype(np.float64).reshape(-1, 3, 15)
    for degree, start, stop in ((1, 0, 3), (2, 3, 8), (3, 8, 15)):
        features[:, :, start:stop] = np.einsum("ij,ncj->nci", sh_rotation(degree, r), features[:, :, start:stop])
    features = features.reshape(-1, 45)
    for i in range(45):
        output["f_rest_%d" % i] = features[:, i]
    for name in FIELDS:
        if not np.isfinite(output[name]).all():
            raise ValueError("Transformed Gaussian exceeds finite float32: " + name)
    return output


def _trs(record):
    angles = _finite(record.get("rotation_euler", [0, 0, 0]), (3,), "Euler angles")
    r = _finite(record["rotation_matrix"], (3, 3), "rotation") if "rotation_matrix" in record else euler_xyz_to_R(*angles)
    rotation_quaternion(r)
    scale = record.get("scale", 1.)
    if isinstance(scale, bool) or not np.isscalar(scale) or not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("Object scale must be positive and uniform")
    return r, float(scale), _finite(record.get("location", [0, 0, 0]), (3,), "location")


def _write_ply(path, vertices):
    from plyfile import PlyData, PlyElement
    with Path(path).open("xb") as stream:
        PlyData([PlyElement.describe(vertices, "vertex")], text=False, byte_order="<").write(stream)


def prepare_scene(spec, out_dir, *, execute=False):
    """Bake a placement into new scene_only.ply, base.ply, fused.ply and objects.

    spec: {placement: dict|path, asset_root: path, base_exclude_objects: [id],
           variants: {name: [object_id]}, ground_cut: {rectangles: [...], z_range: [...]}}.
    Input paths are relative to the spec file, then to asset_root for PLY locators.
    ground_cut removes background centers in a world-XY rectangle union only.
    """
    if type(execute) is not bool:
        raise ValueError("execute must be a bool")
    config, config_base, config_path = _read_document(spec)
    placement, placement_base, placement_path = _read_document(config["placement"], config_base)
    asset_root = Path(config.get("asset_root", str(placement_base))).expanduser()
    asset_root = (asset_root if asset_root.is_absolute() else config_base/asset_root).resolve()
    out = Path(out_dir).expanduser().absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError("Refusing to overwrite scene output: " + str(out))
    out = out.resolve()
    scene_r, scene_s, scene_t = _trs(placement["scene"])
    if not np.allclose(scene_r, np.eye(3), atol=1e-12, rtol=0) or scene_s != 1 or not np.allclose(scene_t, 0, atol=1e-12, rtol=0):
        raise ValueError("Scene placement must remain identity; align a derived source asset first")
    records = {"scene": placement["scene"]}
    objects = placement.get("objects", {})
    for name, record in objects.items():
        _name(name)
        if name == "scene":
            raise ValueError("Object name scene is reserved")
        _trs(record)
        records[name] = record
    paths = {}
    for name, record in records.items():
        path = Path(record["ply"]).expanduser()
        path = (path if path.is_absolute() else asset_root/path).resolve()
        if not path.is_file() or path == out or out in path.parents:
            raise ValueError("Invalid scene input/output relation: " + str(path))
        paths[name] = path
    excluded = config.get("base_exclude_objects", config.get("calibration_board_names", []))
    if not isinstance(excluded, list) or len(set(excluded)) != len(excluded) or set(excluded)-set(objects):
        raise ValueError("base_exclude_objects must name unique existing objects")
    variants = config.get("variants", {})
    for name, names in variants.items():
        _name(name)
        if name in ("base", "fused", "scene_only", "objects", "scene_manifest"):
            raise ValueError("Reserved scene variant name: " + name)
        if not isinstance(names, list) or len(set(names)) != len(names) or set(names)-set(objects):
            raise ValueError("Variant contains unknown or duplicate objects: " + name)
    cut = config.get("ground_cut")
    if cut is not None:
        rectangles = np.asarray(cut["rectangles"], dtype=np.float64)
        z_range = _finite(cut["z_range"], (2,), "ground cut Z range")
        if (rectangles.ndim != 2 or rectangles.shape[1] != 4 or not len(rectangles)
                or not np.isfinite(rectangles).all() or np.any(rectangles[:, 0] >= rectangles[:, 1])
                or np.any(rectangles[:, 2] >= rectangles[:, 3]) or z_range[0] > z_range[1]):
            raise ValueError("Invalid ground-cut rectangles or Z range")
    result = {"schema_version": "calibration-scene.v1", "dry_run": not execute,
              "output": str(out), "fused_ply": str(out/"fused.ply"), "base_ply": str(out/"base.ply"),
              "scene_only_ply": str(out/"scene_only.ply"),
              "object_plys": {name: str(out/"objects"/(name+".ply")) for name in objects},
              "variants": {name: str(out/(name+".ply")) for name in variants},
              "manifest_path": str(out/"scene_manifest.json"), "base_exclude_objects": list(excluded),
              "input_paths": {name: str(path) for name, path in paths.items()},
              "placement": placement, "units_status": config.get("units_status", "unspecified; no metric evidence asserted")}
    result = _jsonable(result)
    if not execute:
        return result
    inputs = {name: {"path": str(path), "sha256": _sha(path)} for name, path in paths.items()}
    for label, path in (("config", config_path), ("placement", placement_path)):
        if path is not None:
            inputs[label] = {"path": str(path), "sha256": _sha(path)}
    scene = read_gaussians(paths["scene"])
    original_scene_count = len(scene)
    if cut is not None:
        xyz = np.column_stack([scene[name] for name in ("x", "y", "z")]).astype(np.float64)
        mask = np.zeros(len(scene), dtype=bool)
        for x0, x1, y0, y1 in rectangles:
            mask |= (xyz[:, 0] >= x0)&(xyz[:, 0] < x1)&(xyz[:, 1] >= y0)&(xyz[:, 1] < y1)
        mask &= (xyz[:, 2] >= z_range[0])&(xyz[:, 2] <= z_range[1])
        scene = scene[~mask]
        if not len(scene):
            raise ValueError("Ground cut removed the entire background")
    baked = {name: bake_gaussians(read_gaussians(paths[name]), *_trs(record)) for name, record in objects.items()}
    out.mkdir(parents=True, exist_ok=False)
    (out/"objects").mkdir()
    _write_ply(out/"scene_only.ply", scene)
    for name, vertices in baked.items():
        _write_ply(out/"objects"/(name+".ply"), vertices)
    groups = {"base": [name for name in objects if name not in excluded], "fused": list(objects)}
    groups.update(variants)
    counts = {"scene_input": original_scene_count, "scene_kept": len(scene),
              "scene_removed": original_scene_count-len(scene), "objects": {name: len(v) for name, v in baked.items()}}
    for name, selected in groups.items():
        vertices = np.concatenate([scene]+[baked[key] for key in selected])
        expected = len(scene)+sum(len(baked[key]) for key in selected)
        if len(vertices) != expected:
            raise RuntimeError("Gaussian fusion count did not conserve points")
        _write_ply(out/(name+".ply"), vertices)
        counts[name] = expected
    for entry in inputs.values():
        if _sha(entry["path"]) != entry["sha256"]:
            raise RuntimeError("Scene input changed during preparation: " + entry["path"])
    result.update(complete=True, counts=counts, inputs=inputs, source_snapshot_sha256=_sha(__file__),
                  gaussian_parameter_contract="SH3 channel-major rest; raw log-scale and opacity logit; quaternion wxyz",
                  normal_contract="Missing normals are compatibility zeros, not measured normals",
                  transform_contract="xyz'=R(s xyz)+t; q'=q_R*q; log_scale+=log(s); SH bands rotated; DC/opacity unchanged",
                  ground_cut=cut, manual_acceptance="pending")
    produced = [out/"scene_only.ply"]+[out/"objects"/(name+".ply") for name in objects]
    produced += [out/(name+".ply") for name in groups]
    result["outputs"] = {str(p.relative_to(out)): {"sha256": _sha(p), "bytes": p.stat().st_size} for p in produced}
    with (out/"scene_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result
