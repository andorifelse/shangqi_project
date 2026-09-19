"""ROS editor: subscribes authoritative state and images; sends asynchronous edits."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import viser
from scene_manager.models import Scene
from scene_manager.ros_images import from_image
from .editor import Editor
from .files import SceneFiles
from .preview import PreviewDisplay


def main(args=None) -> None:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,DurabilityPolicy
    from std_msgs.msg import String
    from sensor_msgs.msg import Image
    from avm_interfaces.srv import SceneCommand

    class RemoteStore:
        def __init__(self,node):
            self.node=node
            self.scene,self.revision=Scene(),-1
            self.client=node.create_client(SceneCommand,"/sim/scene/command")

        def snapshot(self):
            return self.revision,deepcopy(self.scene)

        def command(self,command,expected_revision=-1):
            if not self.client.service_is_ready():
                raise ValueError("Scene manager is not ready")
            request=SceneCommand.Request()
            request.command_json=json.dumps(command)
            request.expected_revision=expected_revision
            future=self.client.call_async(request)
            def finished(f):
                try:
                    response=f.result()
                    self.node.editor.status.content="Ready" if response.success else f"**Edit rejected:** {response.error}"
                except Exception as exc:
                    self.node.editor.status.content=f"**Service failure:** {exc}"
            future.add_done_callback(finished)
            return self.revision

    class EditorNode(Node):
        def __init__(self):
            super().__init__("scene_editor")
            self.server=viser.ViserServer(host=self.declare_parameter("host","127.0.0.1").value,
                                          port=self.declare_parameter("port",8080).value)
            self.server.scene.set_up_direction("+z")
            self.server.gui.configure_theme(control_layout="fixed", control_width="large",show_share_button=False)
            self.server.scene.add_grid("/grid",plane="xy",width=14,height=14)
            self.store=RemoteStore(self)
            self.editor=Editor(self.server,self.store)
            self.files=SceneFiles(self.server,self.store,self.editor.status,Path.cwd()/"scene.yaml")
            self.preview=PreviewDisplay(self.server)
            self.images,self.image_subs={},{}
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
            self.store.scene,self.store.revision=Scene.from_dict(data["scene"]),data["revision"]
            for name in self.store.scene.cameras:
                if name not in self.image_subs:
                    self.image_subs[name]=self.create_subscription(Image,f"/sim/camera/{name}/image_raw",
                        lambda m,n=name:self.image(n,m),10)
            for name in list(self.image_subs):
                if name not in self.store.scene.cameras:
                    self.destroy_subscription(self.image_subs.pop(name))
                    self.images.pop(name,None)

        def image(self,name,msg):
            self.images[name]=from_image(msg)
            self.preview.update(self.images)
            self.preview.status.content="ROS sensor images received"

        def avm(self,msg):
            self.preview.avm.image=from_image(msg)

        def tick(self):
            self.files.tick()
            self.editor.tick()

    rclpy.init(args=args)
    node=EditorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.server.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
