import cv2
import numpy as np
import pytest
from scene_manager.models import default_scene
from scene_manager.camera_models import project, world_to_pixel, pixels_to_world_plane, unproject
from scene_manager.transforms import (pose_matrix,matrix_pose,inverse,transform_points,
    wxyz_from_matrix,matrix_from_wxyz_position)
from avm_stitcher.projection import (ground_points,world_to_bev_pixel,bev_pixel_to_world,projection_debug_maps)
from avm_stitcher.validation import export_avm_validation
from scene_manager.transforms import R_LINK_OPTICAL
from gs_renderer.gsplat_renderer import gsplat_intrinsics


@pytest.mark.parametrize("model",["pinhole","fisheye"])
def test_world_optical_opencv_pixel_ray_plane_roundtrip(tmp_path,model):
    s=default_scene(tmp_path)
    s.vehicle.pose=[.4,-.7,.3,.15,-.2,.65]
    rng=np.random.default_rng(20261020)
    for n,c in s.cameras.items():
        c.model=model
        c.D=[.03,-.004,.0004,-.0001] if model=="fisheye" else [.03,-.004,.001,-.0007,.0004]
        optical=np.column_stack([rng.uniform(-.3,.3,100),rng.uniform(-.3,.3,100),rng.uniform(1,3,100)])
        t=s.camera_optical_pose(n)
        world=transform_points(t,optical)
        uv,valid=world_to_pixel(s,n,world)
        assert valid.all()
        tcw=inverse(t)
        rv,_=cv2.Rodrigues(np.ascontiguousarray(tcw[:3,:3]))
        tv=np.ascontiguousarray(tcw[:3,3])
        fn=cv2.fisheye.projectPoints if model=="fisheye" else cv2.projectPoints
        expected=fn(world.reshape(-1,1,3),rv,tv,c.matrix,np.asarray(c.D))[0].reshape(-1,2)
        np.testing.assert_allclose(uv,expected,atol=1e-8)
        # An arbitrary plane perpendicular to this camera, not world z=0.
        z=2.
        point=transform_points(t,np.array([0,0,z]))
        normal=t[:3,2]
        plane_world=transform_points(t,np.column_stack([optical[:,:2],np.full(100,z)]))
        pixels,ok=world_to_pixel(s,n,plane_world)
        hit,ok=pixels_to_world_plane(s,n,pixels,normal,-normal@point)
        assert ok.all()
        np.testing.assert_allclose(hit,plane_world,atol=1e-6)
        np.testing.assert_allclose(inverse(t)@t,np.eye(4),atol=1e-12)


def test_quaternion_euler_degree_radian_roundtrip():
    rng=np.random.default_rng(120)
    for pose in rng.uniform([-3,-3,-3,-1,-1,-3],[3,3,3,1,1,3],(100,6)):
        t=pose_matrix(pose)
        np.testing.assert_allclose(pose_matrix(matrix_pose(t)),t,atol=1e-12)
        np.testing.assert_allclose(matrix_from_wxyz_position(wxyz_from_matrix(t),t[:3,3]),t,atol=1e-12)
        pose[3:]=np.deg2rad(np.rad2deg(pose[3:]))
        np.testing.assert_allclose(pose_matrix(pose),t,atol=1e-12)


def test_ros_link_optical_opencv_and_gsplat_pixel_center_conventions(tmp_path):
    c=default_scene(tmp_path).cameras["front"]
    np.testing.assert_array_equal(R_LINK_OPTICAL@np.array([0,0,1]),[1,0,0])
    np.testing.assert_array_equal(R_LINK_OPTICAL@np.array([1,0,0]),[0,-1,0])
    k=gsplat_intrinsics(c)
    np.testing.assert_allclose(k[:2,:2],c.matrix[:2,:2])
    np.testing.assert_allclose(k[:2,2],c.matrix[:2,2]+.5)
    np.testing.assert_allclose(c.matrix[2],[0,0,1])


def test_bev_inverse_rotated_vehicle_weights_and_grid(tmp_path):
    s=default_scene(tmp_path)
    s.vehicle.pose=[1.,-.3,.6,.1,-.12,.4]
    s.coordinate.ground_z=.2
    ground=ground_points(s)
    uv,valid=world_to_bev_pixel(s,ground)
    assert valid.all()
    np.testing.assert_allclose(bev_pixel_to_world(s,uv),ground,atol=1e-12)
    uu,vv=np.meshgrid(np.arange(s.bev.width),np.arange(s.bev.height))
    np.testing.assert_allclose(uv,np.stack([uu,vv],-1),atol=1e-10)
    maps=projection_debug_maps(s)
    summed=sum(maps[f"{n}_weight"] for n in s.cameras)
    covered=maps["valid_mask"]>0
    np.testing.assert_allclose(summed[covered],1,atol=1e-6)
    assert np.all(summed[~covered]==0)
    metrics=export_avm_validation(s,tmp_path/"debug")
    assert metrics["ground_bev_roundtrip_max_px"]<1e-8
    assert metrics["ground_color_p99_error"]<4
