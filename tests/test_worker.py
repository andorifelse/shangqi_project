import time
from avm_bringup.assets import generate_assets
from scene_manager.models import default_scene
from gs_renderer.worker import SensorWorker


def test_worker_survives_invalid_scene_then_renders_four_cameras(tmp_path):
    generate_assets(tmp_path)
    scene=default_scene(tmp_path)
    for c in scene.cameras.values():
        c.width,c.height=64,48
        c.K=[22.,0,31.5,0,22.,23.5,0,0,1]
    worker=SensorWorker("cpu")
    def wait():
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            response=worker.poll()
            if response:
                return response
            time.sleep(.02)
        raise AssertionError("worker timed out")
    try:
        bad=default_scene(tmp_path)
        bad.gaussian_file=str(tmp_path/"missing.ply")
        worker.submit(1,bad)
        assert wait()[3]
        worker.submit(2,scene)
        revision,snapshot,results,error=wait()
        assert revision==2 and not error
        assert set(results)=={"front","rear","left","right"}
        assert all(r.rgb.shape==(48,64,3) for r in results.values())
    finally:
        worker.close()
    assert not worker.process.is_alive()
