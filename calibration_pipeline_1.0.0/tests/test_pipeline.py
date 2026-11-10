"""Portable numerical regressions and an actual-PNG CPU integration fixture."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings

import cv2
import numpy as np
from calibration_pipeline import Pipeline, run_pipeline
from calibration_pipeline import bev, scene, rendering, calibration

def grid_spec():
    return {"x_min_m": -1., "x_max_m": 1., "y_min_m": -1.5,
            "y_max_m": 1.5, "meters_per_pixel": .5, "plane_z_m": .12}


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[100., 10., 20.], [0., 200., 30.], [0., 0., 1.]])

    def test_pinhole_nonzero_skew_has_independent_numeric_expectation(self):
        uv, valid = bev.project_camera([[1., 2., 4.]], self.K, np.zeros(5), "pinhole")
        self.assertTrue(valid[0])
        np.testing.assert_allclose(uv[0], [50., 130.], atol=1e-12)

    def test_brown_radial_and_tangential_coefficient_order(self):
        # x=.2, y=.1; radial=1.005 and Brown tangential terms give (.2031,.1023).
        K = np.array([[100., 7., 12.], [0., 120., 15.], [0., 0., 1.]])
        uv, valid = bev.project_camera([[.2, .1, 1.]], K,
                                       [.1, 0., .02, .01, 0.], "pinhole")
        self.assertTrue(valid[0])
        np.testing.assert_allclose(uv[0], [33.0261, 27.276], atol=1e-10)

    def test_fisheye_equidistant_skew_and_optical_axis(self):
        uv, valid = bev.project_camera([[0., 1., 1.], [0., 0., 2.]],
                                       self.K, np.zeros(4), "fisheye")
        self.assertTrue(valid.all())
        np.testing.assert_allclose(uv[0], [20.+10.*np.pi/4, 30.+200.*np.pi/4], atol=1e-12)
        np.testing.assert_allclose(uv[1], [20., 30.], atol=1e-12)

    def test_positive_depth_and_nonfinite_points_are_explicitly_masked(self):
        points = np.array([[0., 0., 1.], [1., 2., -1.], [0., 0., 0.],
                           [np.nan, 0., 1.], [0., np.inf, 1.]])
        with np.errstate(invalid="ignore", over="ignore"):
            for model, count in [("pinhole", 5), ("fisheye", 4)]:
                with self.subTest(model=model):
                    uv, valid = bev.project_camera(points, self.K, np.zeros(count), model)
                    np.testing.assert_array_equal(valid, [True, False, False, False, False])
                    self.assertTrue(np.isfinite(uv).all())

    def test_fisheye_cannot_reenter_after_earlier_radial_fold(self):
        # d(theta_d)/d(theta)=(1-5*theta**2)*(1-2*theta**2).
        # At theta=.8 the derivative and theta_d are positive again, but the
        # valid continuous branch already ended at sqrt(.2).
        D = [-7./3, 2., 0., 0.]
        angles = np.array([.3, .8])
        derivative_at_second = (1-5*angles[1]**2)*(1-2*angles[1]**2)
        self.assertGreater(derivative_at_second, 0.)
        points = np.c_[np.tan(angles), np.zeros(2), np.ones(2)]
        _, valid = bev.project_camera(points, self.K, D, "fisheye")
        np.testing.assert_array_equal(valid, [True, False])

    def test_fisheye_declared_angle_limit(self):
        angles = np.deg2rad([49., 51.])
        points = np.c_[np.tan(angles), np.zeros(2), np.ones(2)]
        _, valid = bev.project_camera(points, self.K, np.zeros(4), "fisheye", theta_max_deg=50.)
        np.testing.assert_array_equal(valid, [True, False])

    def test_pinhole_radial_fold_and_tangential_fold_are_invalid(self):
        _, radial_valid = bev.project_camera([[.5, 0., 1.], [.7, 0., 1.]],
                                             self.K, [-1., 0., 0., 0., 0.], "pinhole")
        np.testing.assert_array_equal(radial_valid, [True, False])
        _, tangent_valid = bev.project_camera([[.2, 0., 1.]], self.K,
                                              [0., 0., 0., -1., 0.], "pinhole")
        self.assertFalse(tangent_valid[0])

    def test_zero_distortion_plane_projection_matches_known_homography(self):
        R = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
        t = np.array([.2, -.1, 2.])
        world = np.array([[-.4, -.2, 0.], [.3, -.5, 0.], [.7, .6, 0.]])
        H = self.K @ np.column_stack([R[:, 0], R[:, 1], t])
        homogeneous = np.c_[world[:, :2], np.ones(len(world))] @ H.T
        expected = homogeneous[:, :2]/homogeneous[:, 2:]
        uv, valid = bev.project_camera(world @ R.T+t, self.K, np.zeros(5), "pinhole")
        self.assertTrue(valid.all())
        np.testing.assert_allclose(uv, expected, atol=1e-12)

    def test_invalid_lens_parameters_are_rejected(self):
        for K, D, model in [
            (np.diag([-1., 1., 1.]), np.zeros(5), "pinhole"),
            (self.K, np.zeros(4), "pinhole"),
            (self.K, [0., np.nan, 0., 0.], "fisheye"),
        ]:
            with self.subTest(model=model, K=K, D=D), self.assertRaises(ValueError):
                bev.project_camera([[0., 0., 1.]], K, D, model)


class WorldGridTests(unittest.TestCase):
    def test_pixel_centers_and_vehicle_orientation(self):
        world = bev.world_grid(grid_spec())
        self.assertEqual(world.shape, (4, 6, 3))
        np.testing.assert_allclose(world[0, 0], [.75, 1.25, .12])
        np.testing.assert_allclose(world[-1, -1], [-.75, -1.25, .12])
        np.testing.assert_allclose(world[1, 0]-world[0, 0], [-.5, 0., 0.])
        np.testing.assert_allclose(world[0, 1]-world[0, 0], [0., -.5, 0.])

    def test_invalid_grid_extent_or_spacing_is_rejected(self):
        for change in [{"meters_per_pixel": 0.}, {"x_max_m": 1.1}]:
            spec = {**grid_spec(), **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                bev.world_grid(spec)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        yy, xx = np.mgrid[:7, :9]
        self.image = np.stack([xx*10, yy*20, xx*8+yy*4], axis=-1).astype(np.uint8)
        self.assertTrue(cv2.imwrite(str(self.root/"gradient.png"), self.image))
        self.camera = {"id": "synthetic", "image": "gradient.png", "R_w2c": np.eye(3).tolist(),
                       "tvec": [0., 0., 1.], "K": np.eye(3).tolist(),
                       "D": [0.]*5, "model": "pinhole"}

    @staticmethod
    def image_plane(height, width, offset=0.):
        yy, xx = np.mgrid[:height, :width]
        return np.stack([xx+offset, yy+offset, np.zeros_like(xx)], axis=-1)

    def test_source_pixel_identity_and_inclusive_last_pixel(self):
        sampled, valid, weights, meta = bev.sample_camera(self.image_plane(7, 9), self.camera, self.root)
        np.testing.assert_array_equal(sampled, self.image)
        self.assertTrue(valid.all())
        self.assertTrue((weights > 0).all())
        self.assertEqual(meta["resolution_px"], [9, 7])

    def test_half_pixel_sampling_is_bilinear_without_extra_offset(self):
        sampled, valid, _, _ = bev.sample_camera(self.image_plane(6, 8, .5), self.camera, self.root)
        expected = (self.image[:-1, :-1].astype(float)+self.image[1:, :-1]
                    +self.image[:-1, 1:]+self.image[1:, 1:])/4
        self.assertTrue(valid.all())
        np.testing.assert_allclose(sampled, expected, atol=1e-12)

    def test_black_pixels_remain_valid_samples(self):
        self.assertTrue(cv2.imwrite(str(self.root/"black.png"), np.zeros_like(self.image)))
        camera = {**self.camera, "image": "black.png"}
        sampled, valid, weights, _ = bev.sample_camera(self.image_plane(7, 9), camera, self.root)
        self.assertTrue(valid.all())
        self.assertTrue((weights > 0).all())
        self.assertFalse(sampled.any())

    def test_outside_source_and_behind_camera_are_not_coverage(self):
        world = np.array([[[0., 0., 0.], [-.1, 0., 0.], [9., 0., 0.], [0., 0., -2.]]])
        sampled, valid, weights, _ = bev.sample_camera(world, self.camera, self.root)
        np.testing.assert_array_equal(valid, [[True, False, False, False]])
        self.assertTrue((weights[~valid] == 0).all())
        self.assertFalse(sampled[~valid].any())

    def test_fisheye_circle_is_geometric_mask_not_black_pixel_test(self):
        self.assertTrue(cv2.imwrite(str(self.root/"circle.png"), np.zeros((9, 9, 3), np.uint8)))
        K = [[1., 0., 4.], [0., 1., 4.], [0., 0., 1.]]
        camera = {**self.camera, "image": "circle.png", "K": K, "image_mask": "fisheye_circle"}
        world = np.array([[[0., 0., 0.], [-4., -4., 0.]]])
        sampled, valid, weights, _ = bev.sample_camera(world, camera, self.root)
        np.testing.assert_array_equal(valid, [[True, False]])
        self.assertGreater(weights[0, 0], 0.)
        self.assertFalse(sampled.any())

    def test_source_image_hash_pin_rejects_changed_input(self):
        expected = hashlib.sha256((self.root/"gradient.png").read_bytes()).hexdigest()
        camera = {**self.camera, "metadata": {"source_image_sha256": expected}}
        _, valid, _, metadata = bev.sample_camera(self.image_plane(7, 9), camera, self.root)
        self.assertTrue(valid.all())
        self.assertEqual(metadata["image_sha256"], expected)
        camera["metadata"] = {"source_image_sha256": "0"*64}
        with self.assertRaisesRegex(ValueError, "Source image hash changed"):
            bev.sample_camera(self.image_plane(7, 9), camera, self.root)

    def test_huge_finite_outside_projection_is_masked_before_float32_cast(self):
        camera = {**self.camera, "K": [[1e40, 0., 0.], [0., 1., 0.], [0., 0., 1.]]}
        world = np.array([[[1., 0., 0.]]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            sampled, valid, weights, _ = bev.sample_camera(world, camera, self.root)
        self.assertFalse(valid.any())
        self.assertFalse(sampled.any())
        self.assertFalse(weights.any())


class SeamMetricTests(unittest.TestCase):
    def test_fixed_boundary_score_ignores_along_seam_and_nearby_natural_edge(self):
        image = np.zeros((3, 3, 3), np.uint8)
        image[1, 2] = 255
        fixed_owner = np.tile([0, 1, 1], (3, 1)).astype(np.int16)
        overlap = np.ones((3, 3), bool)
        evaluation = np.ones((3, 3), bool)
        scores = bev.method_scores(image, fixed_owner, fixed_owner, overlap, evaluation)
        jump = scores["fixed_boundary_gray_jump_0_255"]
        self.assertEqual(jump["count"], 3)
        self.assertEqual(jump["mean"], 0.)
        self.assertEqual(jump["median"], 0.)
        self.assertEqual(jump["p95"], 0.)
        self.assertGreater(scores["overlap_gray_gradient_0_255"]["mean"], 0.)


class BlendTests(unittest.TestCase):
    def assert_owner_validity(self, owners, valid):
        covered = valid.any(axis=0)
        for name, owner in owners.items():
            with self.subTest(owner=name):
                self.assertTrue((owner[~covered] == -1).all())
                for i in range(len(valid)):
                    self.assertFalse(((owner == i) & ~valid[i]).any())

    def test_same_color_inputs_do_not_darken_or_lose_coverage(self):
        color = np.array([23, 47, 89], np.uint8)
        images = np.broadcast_to(color, (3, 32, 40, 3)).copy()
        valid = np.ones((3, 32, 40), bool)
        weights = np.broadcast_to(np.array([.1, .3, .8], np.float32)[:, None, None], valid.shape).copy()
        methods, owners = bev.blend_views(images, valid, weights, 4)
        self.assertEqual(set(methods), set(bev.METHODS))
        for name, image in methods.items():
            with self.subTest(method=name):
                np.testing.assert_array_equal(image, np.broadcast_to(color, image.shape))
        self.assert_owner_validity(owners, valid)

    def test_nonoverlapping_masks_preserve_each_color_and_gap(self):
        images = np.zeros((2, 32, 40, 3), np.uint8)
        images[0] = [10, 30, 70]
        images[1] = [90, 60, 20]
        valid = np.zeros((2, 32, 40), bool)
        valid[0, :, :15] = True
        valid[1, :, 25:] = True
        methods, owners = bev.blend_views(images, valid, valid.astype(np.float32), 4)
        for name, image in methods.items():
            with self.subTest(method=name):
                np.testing.assert_array_equal(image[valid[0]], images[0][valid[0]])
                np.testing.assert_array_equal(image[valid[1]], images[1][valid[1]])
                self.assertFalse(image[:, 15:25].any())
        self.assert_owner_validity(owners, valid)

    def test_invalid_nonzero_color_cannot_pollute_normal_masked_weights(self):
        images = np.empty((2, 32, 40, 3), np.uint8)
        images[0] = [17, 33, 65]
        images[1] = [255, 0, 255]
        valid = np.zeros((2, 32, 40), bool)
        valid[0] = True
        methods, owners = bev.blend_views(images, valid, valid.astype(np.float32), 4)
        for name, image in methods.items():
            with self.subTest(method=name):
                np.testing.assert_array_equal(image, images[0])
        self.assert_owner_validity(owners, valid)

    def test_valid_mask_overrides_stale_nonzero_invalid_weights(self):
        images = np.empty((2, 32, 40, 3), np.uint8)
        images[0] = [17, 33, 65]
        images[1] = [255, 0, 255]
        valid = np.zeros((2, 32, 40), bool)
        valid[0] = True
        weights = np.ones((2, 32, 40), np.float32)
        weights[1] *= 100
        methods, owners = bev.blend_views(images, valid, weights, 4)
        for name, image in methods.items():
            with self.subTest(method=name):
                np.testing.assert_array_equal(image, images[0])
        self.assert_owner_validity(owners, valid)

    def test_no_coverage_stays_empty_for_all_methods(self):
        images = np.full((2, 32, 40, 3), 200, np.uint8)
        valid = np.zeros((2, 32, 40), bool)
        methods, owners = bev.blend_views(images, valid, np.zeros(valid.shape, np.float32), 4)
        for name, image in methods.items():
            with self.subTest(method=name):
                self.assertFalse(image.any())
        self.assert_owner_validity(owners, valid)

    def test_nonpositive_blending_band_is_rejected(self):
        with self.assertRaises(ValueError):
            bev.blend_views(np.zeros((1, 2, 2, 3), np.uint8), np.ones((1, 2, 2), bool),
                            np.ones((1, 2, 2), np.float32), 0)



def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def make_cpu_demo(root):
    """Analytic checkerboard cubes; deliberately not a 3DGS render benchmark."""
    root = Path(root)
    side, core, pad = 288, 256, 16
    R = np.diag([1., -1., -1.])
    center = np.array([0., 0., 2.])
    yy, xx = np.mgrid[:side, :side]
    local = np.stack([(xx-(side-1)/2)/(core/2),
                      (yy-(side-1)/2)/(core/2), np.ones_like(xx)], -1)
    camera = {'frame': {'R_w2c': R.tolist(), 'camera_center_world': center.tolist(),
                        'tvec': (-R @ center).tolist()}, 'views': []}
    manifests = {'intrinsics': {'cube_face_padding_px': pad, 'captures': {'camera': {}}},
                 'extrinsics': {'cube_face_padding_px': pad, 'captures': {'camera': {}}}}
    # Small-board square size .15 gives actual detectable corners in a small fixture.
    views = []
    for i in range(8):
        angle = [-.22, .14, -.1, .25, -.3, .04, .18, -.15][i]
        c, s = math.cos(angle), math.sin(angle)
        B = np.array([[c, -s, 0., (i%3-1)*.17], [s, c, 0., (i//3-1)*.12],
                      [0., 0., 1., 0.], [0., 0., 0., 1.]])
        name = 'camera__pose_%02d' % i
        camera['views'].append({'name': name, 'board_to_world': B.tolist(),
                                'split': 'train' if i < 6 else 'holdout'})
        views.append(('intrinsics', name, B, .15, 8, 6))
    views.append(('extrinsics', 'big_board_fixed', np.eye(4), .25, 4, 5))
    for kind, name, B, square, nx, ny in views:
        faces = {}
        for face, axes in rendering.FACE_AXES.items():
            rays = (local @ np.asarray(axes)) @ R
            distance = np.divide(-center[2], rays[..., 2],
                out=np.full((side, side), -1., dtype=float), where=abs(rays[..., 2]) > 1e-8)
            points = center + rays * distance[..., None]
            board = (points - B[:3, 3]) @ B[:3, :3]
            gx, gy = board[..., 0]/square+nx/2, board[..., 1]/square+ny/2
            valid = (distance > 0) & (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
            parity = (np.floor(gx).astype(int) + np.floor(gy).astype(int)) % 2
            image = np.full((side, side, 3), 175, np.uint8)
            image[valid & (parity == 0)] = 15
            image[valid & (parity == 1)] = 245
            image = cv2.GaussianBlur(image, (3, 3), .35)
            path = root/'input'/'cubes'/kind/name/(face+'.png')
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), image):
                raise OSError(path)
            # Relative to capture manifest: relocation must remain valid.
            faces[face] = str(path.relative_to(root/'input'))
        manifests[kind]['captures']['camera'][name] = {'faces': faces, 'frame_id': name,
                                                      'board_id': 'small' if kind == 'intrinsics' else 'large'}
    spec = {'cube_size': core, 'cube_face_padding_px': pad, 'cameras': {'camera': camera},
            'rig_ground_plane': {'normal': [0., 0., 1.], 'offset': 0.},
            'pinhole': {'model': 'pinhole', 'width': 480, 'height': 360,
                       'K': [[340., 0., 239.5], [0., 340., 179.5], [0., 0., 1.]], 'distortion': [0.]*5}}
    spec['fisheye'] = {'width':360,'height':360,'field_of_view_deg':170.,'distortion':[-.035,.008,0.,0.]}
    dump(root/'input/capture.json', spec)
    for kind, manifest in manifests.items():
        dump(root/('input/'+kind+'_manifest.json'), manifest)
    solver = {'geometry': {'corners_x': 7, 'corners_y': 5, 'square_size': .15},
              'camera_ids': ['camera'], 'models': {'pinhole': {'resolution_px': [480, 360],
                  'unknown': ['fx', 'fy', 'skew', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3']}},
              'split_by_suffix': {'train': list(range(6)), 'holdout': [6, 7]},
              'pose_prior': 'fixed camera; plane translation and yaw only'}
    solver['models']['fisheye'] = {'resolution_px':[360,360], 'unknown':['fx','fy','skew','cx','cy','k1','k2','k3','k4']}
    dump(root/'input/solver.json', solver)
    corners = {'corners': [{'id': 'front_small/r%02d/c%02d' % (r, c),
                            'region':'front_small', 'grid_row':r, 'grid_col':c,
                            'xyz_world_m': [(c-2)*.25, (r-2.5)*.25, 0.]}
                           for r in range(1, 5) for c in range(1, 4)]}
    dump(root/'input/corners_world.json', corners)
    config = {'schema_version': 'calibration-pipeline.v1', 'output_dir': 'output',
              'stages': ['compose', 'intrinsics', 'extrinsics', 'bev'],
              'compose': {'models': ['pinhole','fisheye'], 'kinds': ['intrinsics', 'extrinsics'],
                  **{kind: {'capture_spec': 'input/capture.json',
                             'manifest': 'input/'+kind+'_manifest.json'} for kind in manifests}},
              'intrinsics': {'solver_spec': 'input/solver.json'},
              'extrinsics': {'corners_world': 'input/corners_world.json',
                  'regions': {'camera': 'front_small'}, 'models': ['pinhole','fisheye']},
              'bev': {'models': ['pinhole','fisheye'], 'channels': ['camera'],
                  'grid': {'x_min_m': -.5, 'x_max_m': .5, 'y_min_m': -.5,
                      'y_max_m': .5, 'meters_per_pixel': .02, 'plane_z_m': 0.}}}
    dump(root/'demo.json', config)
    return root/'demo.json'


class PipelineContractTests(unittest.TestCase):
    def test_cli_and_api_same_plan_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {'schema_version': 'calibration-pipeline.v1', 'output_dir': 'result',
                      'stages': ['bev'], 'bev': {'spec': 'missing.json'}}
            path = root/'case.json'
            dump(path, config)
            before = sorted(str(p.relative_to(root)) for p in root.rglob('*'))
            expected = Pipeline(path).run()
            result = subprocess.run([sys.executable, '-B', '-m', 'calibration_pipeline', 'run',
                                     '--config', str(path)], cwd=temp, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout), expected)
            self.assertEqual(before, sorted(str(p.relative_to(root)) for p in root.rglob('*')))
            self.assertFalse((root/'result').exists())

    def test_pause_and_bad_stage_fail_before_any_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {'schema_version': 'calibration-pipeline.v1', 'output_dir': 'result',
                      'execution_status': 'paused_by_user', 'bev': {'spec': {}}}
            program = Pipeline(config, base_dir=root)
            with self.assertRaises(ValueError):
                program.run('bev', execute=True)
            with self.assertRaises(ValueError):
                program.run('made_up', execute=True)
            self.assertFalse((root/'result').exists())

    def test_stage_failure_is_recorded_and_propagated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            program = Pipeline({'schema_version': 'calibration-pipeline.v1', 'output_dir': 'result',
                'bev': {'spec': 'does_not_exist.json'}}, base_dir=root)
            with self.assertRaises(FileNotFoundError):
                program.run('bev', execute=True)
            runs = list((root/'result/pipeline_runs').glob('*/run.json'))
            self.assertEqual(len(runs), 1)
            record = json.loads(runs[0].read_text())
            self.assertEqual(record['status'], 'FAILED')
            self.assertEqual(record['stages']['bev']['status'], 'FAILED')

    def test_cpu_import_does_not_import_torch(self):
        code = ('import sys; import calibration_pipeline; '
                'from calibration_pipeline import scene,rendering,calibration,bev; '
                'assert "torch" not in sys.modules; assert "torchvision" not in sys.modules')
        subprocess.run([sys.executable, '-B', '-c', code], check=True, capture_output=True)



    def test_nested_bev_spec_uses_its_own_image_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'nested').mkdir()
            cv2.imwrite(str(root/'nested/gray.png'), np.full((40,40,3),127,np.uint8))
            spec = {'schema_version':'bev-stitch.v1','grid':{'x_min_m':-1.,'x_max_m':1.,
                'y_min_m':-1.,'y_max_m':1.,'meters_per_pixel':.1,'plane_z_m':0.},
                'cameras':[{'id':'camera','image':'gray.png','model':'pinhole','D':[0.]*5,
                    'K':[[10.,0.,20.],[0.,10.,20.],[0.,0.,1.]],
                    'R_w2c':np.diag([1.,-1.,-1.]).tolist(),'tvec':[0.,0.,2.]}]}
            dump(root/'nested/bev.json',spec)
            program=Pipeline({'schema_version':'calibration-pipeline.v1','output_dir':'out',
                              'bev':{'spec':'nested/bev.json'}},base_dir=root)
            result=program.run('bev',execute=True)
            self.assertEqual(result['status'],'EXECUTED')
            self.assertTrue((root/'out/bev/provided/comparison.png').exists())

    def test_partial_algorithm_failure_is_not_reported_as_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            program=Pipeline({'schema_version':'calibration-pipeline.v1','output_dir':'out',
                              'intrinsics':{}},base_dir=root)
            program._intrinsics=lambda:{'status':'COMPLETED_WITH_FAILURES','results':{}}
            with self.assertRaises(RuntimeError):
                program.run('intrinsics',execute=True)
            run=json.loads(next((root/'out/pipeline_runs').glob('*/run.json')).read_text())
            self.assertEqual(run['status'],'FAILED')
            self.assertEqual(run['stages']['intrinsics']['result']['status'],'COMPLETED_WITH_FAILURES')


class EndToEndTests(unittest.TestCase):
    def test_real_png_composition_estimation_and_bev(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = make_cpu_demo(root)
            result = run_pipeline(config, execute=True)
            self.assertEqual(result['status'], 'EXECUTED')
            intrinsic = json.loads((root/'output/intrinsics/pinhole/camera/calibration.json').read_text())
            self.assertEqual(intrinsic['status'], 'DEGENERATE_PARALLEL_PLANE_ACQUISITION')
            self.assertFalse(intrinsic['global_uniqueness_proved'])
            self.assertEqual(len(intrinsic['solutions']), 3)
            self.assertGreater(len(intrinsic['solutions'][0]['heldout']), 0)
            external = json.loads((root/'output/extrinsics/pinhole/camera/extrinsics.json').read_text())
            self.assertGreater(len(external['candidates']), 0)
            metrics = json.loads((root/'output/bev/pinhole/metrics.json').read_text())
            self.assertGreater(metrics['coverage_fraction'], .7)
            self.assertTrue((root/'output/bev/pinhole/comparison.png').is_file())
            fish = json.loads((root/'output/intrinsics/fisheye/camera/calibration.json').read_text())
            self.assertFalse(fish['global_uniqueness_proved'])
            self.assertGreater(len(fish['solutions']), 0)
            fish_metrics = json.loads((root/'output/bev/fisheye/metrics.json').read_text())
            self.assertGreater(fish_metrics['coverage_fraction'], .7)
            with self.assertRaises(FileExistsError):
                run_pipeline(config, stage='compose', execute=True)



class GeometryContractTests(unittest.TestCase):
    @staticmethod
    def gaussian_vertices():
        fields = ['x','y','z']+['f_dc_%d'%i for i in range(3)]
        fields += ['f_rest_%d'%i for i in range(45)]+['opacity']
        fields += ['scale_%d'%i for i in range(3)]+['rot_%d'%i for i in range(4)]
        raw = np.zeros(2, dtype=[(n, '<f4') for n in reversed(fields)])
        raw['x'] = [1., 2.]; raw['rot_0'] = 1.
        raw['scale_0'] = raw['scale_1'] = raw['scale_2'] = -4.
        return raw

    def test_field_names_missing_normal_and_rigid_sh_action(self):
        vertices = scene.canonical_gaussians(self.gaussian_vertices())
        self.assertEqual(len(vertices.dtype.names), 62)
        self.assertTrue((vertices['nx'] == 0).all())
        vertices['f_rest_0'] = 1.
        R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        out = scene.bake_gaussians(vertices, R, 2., [3., 4., 5.])
        xyz = np.column_stack([out[n] for n in ('x','y','z')])
        np.testing.assert_allclose(xyz, [[3., 6., 5.], [3., 8., 5.]], atol=1e-6)
        np.testing.assert_allclose(out['scale_0'], -4+np.log(2), atol=1e-6)
        # Graphdeco's l=1 basis is (-Cy, Cz, -Cx). Rz90 changes c_y=1 to c_x=-1.
        np.testing.assert_allclose(out['f_rest_2'], -1., atol=1e-6)
        np.testing.assert_allclose(out['f_rest_0'], 0., atol=1e-6)
        np.testing.assert_allclose(out['rot_0'], np.sqrt(.5), atol=1e-6)
        np.testing.assert_allclose(out['rot_3'], np.sqrt(.5), atol=1e-6)
        np.testing.assert_array_equal(out['opacity'], vertices['opacity'])

    def test_scene_preparation_conserves_each_variant_count(self):
        from plyfile import PlyData, PlyElement
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ['background','car','large_board']:
                PlyData([PlyElement.describe(self.gaussian_vertices(), 'vertex')]).write(root/(name+'.ply'))
            spec = {'asset_root': str(root), 'placement': {
                'scene': {'ply': 'background.ply'},
                'objects': {'car': {'ply': 'car.ply', 'scale': 1.},
                            'large_board': {'ply': 'large_board.ply', 'scale': 1.}}},
                'base_exclude_objects': ['large_board']}
            scene.prepare_scene(spec, root/'dry')
            self.assertFalse((root/'dry').exists())
            result = scene.prepare_scene(spec, root/'made', execute=True)
            self.assertEqual(result['counts']['base'], 4)
            self.assertEqual(result['counts']['fused'], 6)
            self.assertEqual(len(PlyData.read(result['fused_ply'])['vertex']), 6)
            with self.assertRaises(FileExistsError):
                scene.prepare_scene(spec, root/'made', execute=True)
            spec['placement']['scene']['location'] = [1.,0.,0.]
            with self.assertRaises(ValueError):
                scene.prepare_scene(spec, root/'invalid', execute=True)
            self.assertFalse((root/'invalid').exists())

    def test_capture_dry_run_retains_fixed_center_and_plane_constraints(self):
        from plyfile import PlyData, PlyElement
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root/'one.ply'
            PlyData([PlyElement.describe(scene.canonical_gaussians(self.gaussian_vertices()), 'vertex')]).write(path)
            frame = {'R_w2c': np.diag([1.,-1.,-1.]).tolist(),
                     'camera_center_world': [0.,0.,2.], 'tvec': [0.,0.,2.]}
            spec = {'cube_size': 32, 'cube_face_padding_px': 4,
                    'rig_ground_plane': {'normal':[0.,0.,1.], 'offset':0.},
                    'cameras': {'cam': {'frame': frame, 'views': [{'name':'pose_00',
                        'board_to_world':np.eye(4).tolist(), 'split':'train'}]}}}
            result = rendering.capture(spec, path, root/'dry', board_ply=path)
            self.assertEqual(result['expected_cube_faces'], 12)
            self.assertFalse((root/'dry').exists())
            bad = deepcopy(spec)
            bad['cameras']['cam']['views'][0]['board_to_world'][2][3] = .1
            with self.assertRaises(ValueError):
                rendering.capture(bad, path, root/'bad', board_ply=path)
            bad = deepcopy(spec)
            bad['cameras']['cam']['views'][0]['camera_frame'] = {**frame,'camera_center_world':[1.,0.,2.]}
            with self.assertRaises(ValueError):
                rendering.capture(bad, path, root/'bad', board_ply=path)

    def test_bev_candidate_selects_all_correspondences_and_matching_seed(self):
        intrinsic = {'solutions': [{'seed':i, 'K':[[100.+i,0.,50.],[0.,100.,50.],[0.,0.,1.]],
                                  'D':[0.]*5} for i in (0,1)]}
        base = {'R_w2c': np.eye(3).tolist(), 'tvec':[0.,0.,2.], 'minimum_depth_m':1.,
                'fit_corner_count':8, 'optimizer_success':False}
        external = {'candidates': [{**base,'intrinsic_seed':0,'all_correspondence_rmse_px':3.,'reprojection_rmse_px':.01},
                                   {**base,'intrinsic_seed':1,'all_correspondence_rmse_px':.5,'reprojection_rmse_px':.1}]}
        index, row, lens, count = bev.select_candidate(external, intrinsic, 'pinhole')
        self.assertEqual(index, 1)
        self.assertEqual(lens['seed'], 1)
        self.assertEqual(count, 2)
        self.assertFalse(row['optimizer_success'])


    def test_custom_external_image_is_reused_by_bev_after_relocation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            image=root/'images/pinhole/cam/custom/IMG_001.png'
            image.parent.mkdir(parents=True)
            cv2.imwrite(str(image),np.full((40,40,3),127,np.uint8))
            lens={'seed':0,'K':[[10.,0.,20.],[0.,10.,20.],[0.,0.,1.]],'D':[0.]*5}
            candidate={'intrinsic_seed':0,'R_w2c':np.diag([1.,-1.,-1.]).tolist(),'tvec':[0.,0.,2.],
                       'minimum_depth_m':1.,'all_correspondence_rmse_px':.1,'reprojection_rmse_px':.1,'fit_corner_count':12}
            dump(root/'intrinsics/pinhole/cam/calibration.json',{'solutions':[lens]})
            dump(root/'extrinsics/pinhole/cam/extrinsics.json',{'candidates':[candidate],
                'image':'/old/machine/IMG_001.png','image_relative_to_images':'pinhole/cam/custom/IMG_001.png'})
            spec=bev.prepare_bev_spec(root/'intrinsics',root/'extrinsics',root/'images',model='pinhole',channels=['cam'],
                 grid={'x_min_m':-1.,'x_max_m':1.,'y_min_m':-1.,'y_max_m':1.,'meters_per_pixel':.1,'plane_z_m':0.})
            self.assertEqual(Path(spec['cameras'][0]['image']).resolve(),image.resolve())

    def test_empty_model_selection_and_nonboolean_execute_never_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for section, payload in [('compose', {'models':[]}), ('compose', {'kinds':[]}), ('bev', {'models':[]})]:
                program = Pipeline({'schema_version':'calibration-pipeline.v1', 'output_dir':'out',section:payload},base_dir=root)
                with self.assertRaises(ValueError):
                    program.run(section,execute=True)
            program = Pipeline({'schema_version':'calibration-pipeline.v1','output_dir':'out','bev':{'spec':{}}},base_dir=root)
            with self.assertRaises(ValueError):
                program.run('bev',execute='false')
            self.assertFalse((root/'out').exists())


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--make-demo':
        print(make_cpu_demo(Path(sys.argv[2]).resolve()))
    else:
        unittest.main()
