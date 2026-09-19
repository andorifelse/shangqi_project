"""ROS sensor node with consistent snapshot and common timestamp across all cameras."""
from __future__ import annotations
import json
from scene_manager.models import Scene
from scene_manager.ros_images import to_image,camera_info
from .worker import SensorWorker


def main(args=None) -> None:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,DurabilityPolicy
    from std_msgs.msg import String
    from sensor_msgs.msg import Image,CameraInfo

    class RendererNode(Node):
        def __init__(self):
            super().__init__("gs_renderer")
            backend = self.declare_parameter("backend","gsplat").value
            hz = self.declare_parameter("render_hz",2.).value
            self.worker = SensorWorker(backend)
            self.scene,self.revision,self.pubs = None,-1,{}
            self.period,self.last_submit = 1/max(float(hz),.1),0.
            self.sub = self.create_subscription(String,"/sim/scene/state",self.state,
                QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
            self.metadata = self.create_publisher(String,"/sim/render/state",10)
            self.create_timer(.02,self.tick)

        def state(self,msg):
            data = json.loads(msg.data)
            self.scene,self.revision = Scene.from_dict(data["scene"]),data["revision"]

        def tick(self):
            import time
            response = self.worker.poll()
            if response:
                revision,scene,results,error = response
                if error:
                    self.get_logger().error(error)
                else:
                    stamp = self.get_clock().now().to_msg()
                    self.metadata.publish(String(data=json.dumps({
                        "stamp":[stamp.sec,stamp.nanosec],"revision":revision,"scene":scene.to_dict()})))
                    for name,result in results.items():
                        if name not in self.pubs:
                            base=f"/sim/camera/{name}"
                            self.pubs[name]=(self.create_publisher(Image,base+"/image_raw",10),
                                             self.create_publisher(CameraInfo,base+"/camera_info",10))
                        self.pubs[name][0].publish(to_image(result.rgb,stamp,f"camera_{name}_optical"))
                        self.pubs[name][1].publish(camera_info(scene.cameras[name],stamp))
                    for name in list(self.pubs):
                        if name not in scene.cameras:
                            for pub in self.pubs.pop(name):
                                self.destroy_publisher(pub)
            if self.scene is not None and not self.worker.busy and time.monotonic()-self.last_submit>=self.period:
                self.worker.submit(self.revision,self.scene)
                self.last_submit=time.monotonic()

    rclpy.init(args=args)
    node=RendererNode()
    try:
        rclpy.spin(node)
    finally:
        node.worker.close()
        node.destroy_node()
        rclpy.shutdown()
