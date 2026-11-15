"""ROS scene mirror with ordered writes and pending-pose feedback."""
from collections import deque
from copy import deepcopy
import json
from scene_manager.models import Scene
from .ui_text import error_text
from scene_manager.transforms import pose_matrix


class RemoteStore:
    def __init__(self, client, request_type, report):
        self.client, self.request_type, self.report = client, request_type, report
        self.scene, self.revision = Scene(), -1
        self._queue = deque()
        self._inflight = False
        self._serial = 0
        self._poses = {}

    def snapshot(self):
        scene = deepcopy(self.scene)
        for name, (_, pose, _) in self._poses.items():
            if name in scene.object_ids():
                scene.object(name).pose = list(pose)
        return self.revision, scene

    def accept_state(self, scene, revision):
        if revision < self.revision:
            return
        self.scene, self.revision = scene, revision
        for name, (_, _, acknowledged) in list(self._poses.items()):
            if name not in scene.object_ids() or (acknowledged is not None and revision >= acknowledged):
                del self._poses[name]

    def command(self, command, expected_revision=-1):
        if not self.client.service_is_ready():
            raise ValueError("场景管理器尚未就绪")
        command = deepcopy(command)
        self._serial += 1
        serial = self._serial
        if command["action"] == "pose":
            pose_matrix(command["pose"])
            self.scene.object(command["id"])
            self._poses[command["id"]] = (serial, list(command["pose"]), None)
            if self._queue and self._queue[-1][0]["action"] == "pose" and self._queue[-1][0]["id"] == command["id"]:
                self._queue.pop()
        self._queue.append((command, expected_revision, serial))
        self._dispatch()
        return self.revision

    def _dispatch(self):
        if self._inflight or not self._queue:
            return
        command, expected, serial = self._queue.popleft()
        request = self.request_type()
        request.command_json = json.dumps(command)
        request.expected_revision = expected
        self._inflight = True
        future = self.client.call_async(request)

        def finished(f):
            try:
                response = f.result()
                if command["action"] == "pose":
                    current = self._poses.get(command["id"])
                    if current and current[0] == serial:
                        if not response.success or self.revision >= response.revision:
                            del self._poses[command["id"]]
                        else:
                            self._poses[command["id"]] = (serial, current[1], response.revision)
                self.report("就绪" if response.success else f"**修改未生效：** {error_text(response.error)}")
            except Exception as exc:
                if command["action"] == "pose" and self._poses.get(command["id"], (None,))[0] == serial:
                    self._poses.pop(command["id"], None)
                self.report(f"**场景通信失败：** {error_text(exc)}")
            finally:
                self._inflight = False
                self._dispatch()

        future.add_done_callback(finished)
