"""Exercise editor geometry/state boundaries using the real installed Viser API."""
from types import SimpleNamespace
import asyncio
import numpy as np
import pytest
import viser
from scene_manager.models import default_scene
from scene_manager.store import SceneStore
from scene_manager.transforms import wxyz_from_matrix,pose_matrix
from scene_editor.editor import Editor
from avm_bringup.assets import generate_assets


def test_gizmo_world_to_camera_relative_and_delete(tmp_path):
    generate_assets(tmp_path)
    scene=default_scene(tmp_path)
    scene.vehicle.pose=[1,2,0,0,0,.5]
    store=SceneStore(scene)
    server=viser.ViserServer(host="127.0.0.1",port=0,verbose=False)
    try:
        editor=Editor(server,store)
        editor._handle("select","front")
        desired=[2.4,.2,1.1,.1,.6,-.15]
        world=scene.world_pose("base_link")@pose_matrix(desired)
        event=SimpleNamespace(client=True,target=SimpleNamespace(
            wxyz=wxyz_from_matrix(world),position=world[:3,3]+editor.gizmo_offset))
        asyncio.run(editor._drag(event))
        editor.tick()
        np.testing.assert_allclose(store.snapshot()[1].cameras["front"].pose,desired,atol=1e-10)
        editor._handle("add","board")
        editor.tick()
        assert editor.tree.value=="board_002"
        editor._handle("delete",None)
        editor.tick()
        assert "board_002" not in store.snapshot()[1].boards
    finally:
        server.stop()
