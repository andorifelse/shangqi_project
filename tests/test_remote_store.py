from concurrent.futures import Future
from copy import deepcopy
from types import SimpleNamespace
import json
from scene_editor.remote_store import RemoteStore
from scene_manager.models import default_scene


class Client:
    def __init__(self):
        self.calls = []

    def service_is_ready(self):
        return True

    def call_async(self, request):
        future = Future()
        self.calls.append((json.loads(request.command_json), future))
        return future


def test_pending_pose_survives_old_echo_and_writes_are_coalesced(tmp_path):
    client = Client()
    store = RemoteStore(client, SimpleNamespace, lambda _: None)
    scene = default_scene(tmp_path)
    store.accept_state(scene, 0)
    for x in (1., 2., 3.):
        store.command({"action":"pose", "id":"left", "pose":[x,1,1,0,0,1.57]})
    assert len(client.calls) == 1
    assert store.snapshot()[1].cameras["left"].pose[0] == 3.
    old = deepcopy(scene)
    old.cameras["left"].pose[0] = 1.
    store.accept_state(old, 1)
    client.calls[0][1].set_result(SimpleNamespace(success=True, revision=1))
    assert len(client.calls) == 2
    assert client.calls[1][0]["pose"][0] == 3.
    assert store.snapshot()[1].cameras["left"].pose[0] == 3.
    client.calls[1][1].set_result(SimpleNamespace(success=True, revision=2))
    store.accept_state(old, 1)
    assert store.snapshot()[1].cameras["left"].pose[0] == 3.
    latest = deepcopy(scene)
    latest.cameras["left"].pose = [3.,1,1,0,0,1.57]
    store.accept_state(latest, 2)
    assert not store._poses
    assert store.snapshot()[1].cameras["right"].pose == scene.cameras["right"].pose


def test_rejected_pose_restores_authoritative_scene(tmp_path):
    client = Client()
    store = RemoteStore(client, SimpleNamespace, lambda _: None)
    scene = default_scene(tmp_path)
    store.accept_state(scene, 0)
    store.command({"action":"pose", "id":"left", "pose":[4,1,1,0,0,1.57]})
    client.calls[0][1].set_result(SimpleNamespace(success=False, revision=0,error="rejected"))
    assert store.snapshot()[1].cameras["left"].pose == scene.cameras["left"].pose
