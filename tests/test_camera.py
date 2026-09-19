import numpy as np
import cv2
import pytest
from scene_manager.models import Camera
from scene_manager.camera_models import project, unproject, camera_rays


@pytest.mark.parametrize("model",["pinhole","fisheye","ftheta"])
def test_camera_round_trip(model):
    camera=Camera(model=model,D=[.01,-.001,0,0,0] if model=="pinhole" else [.01,-.001,0,0])
    uv=np.array([[159.5,119.5],[130,100],[190,170],[80,60]])
    rays,valid=unproject(camera,uv)
    pixels,ok=project(camera,rays)
    assert valid.all() and ok.all()
    np.testing.assert_allclose(pixels,uv,atol=1e-3)


def test_fisheye_opencv_agreement():
    c=Camera(D=[.05,-.01,.002,0])
    points=np.array([[.2,.4,1.],[-.8,.1,.6],[0,0,1.]],dtype=float)
    ours,valid=project(c,points)
    expected,_=cv2.fisheye.projectPoints(points.reshape(-1,1,3),np.zeros(3),np.zeros(3),c.matrix,np.array(c.D))
    np.testing.assert_allclose(ours,expected.reshape(-1,2),atol=1e-10)
    assert valid.all()


def test_axis_and_back_rays():
    c=Camera()
    uv,valid=project(c,np.array([[0,0,1],[1,0,1],[0,1,1],[0,0,-1.]]))
    np.testing.assert_allclose(uv[0],[159.5,119.5])
    assert uv[1,0]>uv[0,0] and uv[2,1]>uv[0,1]
    assert not valid[-1]
    c.model,c.max_angle="ftheta",2.
    _,ok=project(c,np.array([[1,0,-.1]]))
    assert ok[0]
