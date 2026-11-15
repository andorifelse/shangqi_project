"""ROS editor: subscribes authoritative state and images; sends asynchronous edits."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import viser
from scene_manager.models import Scene
from scene_manager.ros_images import from_image, stamp_key
from .editor import Editor
from .files import SceneFiles
from .preview import PreviewDisplay, SensorPreviewBuffer
from .remote_store import RemoteStore
from .calibration_ui import CalibrationPanel


def main(args=None) -> None:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,DurabilityPolicy
    from std_msgs.msg import String
    from sensor_msgs.msg import Image
    from avm_interfaces.srv import SceneCommand

    class EditorNode(Node):
        def __init__(self):
            super().__init__("scene_editor")
            self.server=viser.ViserServer(host=self.declare_parameter("host","127.0.0.1").value,
                                          port=self.declare_parameter("port",8080).value)
            self.server.scene.set_up_direction("+z")
            self.server.gui.configure_theme(control_layout="fixed", control_width="large",show_share_button=False)
            self.server.scene.add_grid("/grid",plane="xy",width=14,height=14)
            self.store=RemoteStore(self.create_client(SceneCommand,"/sim/scene/command"),
                                   SceneCommand.Request, self.edit_status)
            self.editor=Editor(self.server,self.store)
            self.files=SceneFiles(self.server,self.store,self.editor.status,Path.cwd()/"scene.yaml")
            self.calibration=CalibrationPanel(self.server,self.store,self.declare_parameter("backend","cpu").value)
            self.preview=PreviewDisplay(self.server)
            self.images,self.image_subs={},{}
            self.sensor_batches = SensorPreviewBuffer()
            self.displayed_revision = -1
            self.render_sub = self.create_subscription(String,"/sim/render/state",self.render_state,10)
            self.scene_sub=self.create_subscription(String,"/sim/scene/state",self.state,
                QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
            self.avm_sub=self.create_subscription(Image,"/sim/avm/image",self.avm,10)
            self.create_timer(.03,self.tick)
            @self.server.on_client_connect
            def connected(client):
                client.camera.position=(10,-12,10)
                client.camera.look_at=(0,0,0)
                client.camera.up_direction=(0,0,1)

        def state(self,msg):
            data=json.loads(msg.data)
            self.store.accept_state(Scene.from_dict(data["scene"]),data["revision"])
            for name in self.store.scene.cameras:
                if name not in self.image_subs:
                    self.image_subs[name]=self.create_subscription(Image,f"/sim/camera/{name}/image_raw",
                        lambda m,n=name:self.image(n,m),10)
            for name in list(self.image_subs):
                if name not in self.store.scene.cameras:
                    self.destroy_subscription(self.image_subs.pop(name))
                    self.images.pop(name,None)

        def edit_status(self, text):
            self.editor.status.content = text

        def image(self,name,msg):
            if msg.header.frame_id != f"camera_{name}_optical":
                self.get_logger().error(f"Rejected image for {name}: frame_id={msg.header.frame_id}")
                return
            self.sensor_batches.image(stamp_key(msg.header.stamp), name, from_image(msg))

        def render_state(self, msg):
            data = json.loads(msg.data)
            self.sensor_batches.metadata(tuple(data["stamp"]), data["revision"], data["scene"]["cameras"])

        def avm(self,msg):
            self.preview.avm.image=from_image(msg)

        def tick(self):
            self.files.tick()
            self.editor.tick()
            self.calibration.tick()
            ready = None if self.store._poses else self.sensor_batches.ready(self.store.revision)
            if ready:
                stamp, self.images = ready
                self.preview.update(self.images)
                self.displayed_revision = self.store.revision
                self.preview.status.content = f"相机图像 · 场景版本 {self.displayed_revision} · 时间戳 {stamp[0]}.{stamp[1]:09d}"
            elif self.displayed_revision != self.store.revision or self.store._poses:
                self.preview.status.content = f"等待场景版本 {self.store.revision} 的图像；当前显示版本 {self.displayed_revision}"

    rclpy.init(args=args)
    node=EditorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.calibration.close()
        node.server.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
