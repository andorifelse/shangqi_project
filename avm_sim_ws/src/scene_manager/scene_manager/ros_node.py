"""Authoritative ROS 2 scene owner, state topic, atomic edit service and dynamic TF."""
from __future__ import annotations
import json
from pathlib import Path
from .persistence import load_scene, save_scene
from .store import SceneStore
from .frames import scene_frames
from .transforms import wxyz_from_matrix


def main(args=None) -> None:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from std_msgs.msg import String
    from geometry_msgs.msg import TransformStamped
    from tf2_ros import TransformBroadcaster
    from avm_interfaces.srv import SceneCommand

    class ManagerNode(Node):
        def __init__(self):
            super().__init__("scene_manager")
            scene_file = self.declare_parameter("scene", "").value
            if not scene_file:
                raise ValueError("scene parameter required; use avm_bringup launch")
            self.store = SceneStore(load_scene(scene_file))
            self.pub = self.create_publisher(String, "/sim/scene/state",
                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            self.tf = TransformBroadcaster(self)
            self.service = self.create_service(SceneCommand, "/sim/scene/command", self.command)
            self.last_revision = -1
            self.create_timer(.05, self.tick)

        def command(self, request, response):
            try:
                response.revision = self.store.command(json.loads(request.command_json), request.expected_revision)
                response.success = True
            except (ValueError, KeyError, TypeError, OSError) as exc:
                response.success, response.error = False, str(exc)
                response.revision = self.store.snapshot()[0]
            return response

        def tick(self):
            revision, scene = self.store.snapshot()
            if revision != self.last_revision:
                self.pub.publish(String(data=json.dumps({"revision":revision, "scene":scene.to_dict()})))
                self.last_revision = revision
            stamp = self.get_clock().now().to_msg()
            messages = []
            for frame in scene_frames(scene):
                msg = TransformStamped()
                msg.header.stamp, msg.header.frame_id, msg.child_frame_id = stamp, frame.parent, frame.child
                msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float, frame.transform[:3,3])
                w,x,y,z = wxyz_from_matrix(frame.transform)
                msg.transform.rotation.w, msg.transform.rotation.x = w,x
                msg.transform.rotation.y, msg.transform.rotation.z = y,z
                messages.append(msg)
            self.tf.sendTransform(messages)  # Editable extrinsics are never static TF.

    rclpy.init(args=args)
    node = ManagerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
