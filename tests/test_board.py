import numpy as np
from scene_manager.models import Board,default_scene
from scene_manager.transforms import pose_matrix
from gs_renderer.board import intersect_board,overlay_boards
from gs_renderer.types import RenderResult


def test_ray_plane_metric_checker_and_parallel():
    b=Board(rows=2,columns=2,square_size=.5,pose=[0]*6)
    distance,valid,color=intersect_board(b,np.array([0,0,1.]),np.array([[0,0,-1.],[1,0,0.],[0,0,1.]]))
    assert distance[0]==1 and valid.tolist()==[True,False,False]
    assert color[0,0]==240
    b.pose=[0,0,2.,0,np.pi/2,0]
    distance,valid,_=intersect_board(b,np.array([1,0,2.]),np.array([[-1.,0,0]]))
    assert valid[0] and abs(distance[0]-1)<1e-12


def test_board_overlay_moves_and_occludes(tmp_path):
    scene=default_scene(tmp_path)
    c=scene.cameras["front"]
    shape=(c.height,c.width)
    def result(depth):
        return RenderResult(np.zeros((*shape,3),np.uint8),np.full(shape,depth,dtype=float),np.zeros(shape),np.ones(shape,bool))
    near=overlay_boards(scene,"front",result(.01))
    far=overlay_boards(scene,"front",result(np.inf))
    assert not near.rgb.any() and far.rgb.any()
    scene.boards["board_001"].pose[1]=1.5
    moved=overlay_boards(scene,"front",result(np.inf))
    assert not np.array_equal(moved.rgb,far.rgb)
