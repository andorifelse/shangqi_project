"""Acceptance probe for an already launched ROS stack; requires real rclpy.

Run only after sourcing the built workspace and starting avm_sim.launch.py.
Checks standard image payloads, common timestamps, CameraInfo and TF numerics.
"""
from __future__ import annotations
import json
import time
import numpy as np


def main() -> None:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,DurabilityPolicy
    from sensor_msgs.msg import Image,CameraInfo
    from std_msgs.msg import String
    from tf2_ros import Buffer,TransformListener
    from rclpy.time import Time
    from scene_manager.models import Scene
    from scene_manager.transforms import matrix_from_wxyz_position

    rclpy.init()
    node=Node("avm_runtime_acceptance")
    tf=Buffer()
    listener=TransformListener(tf,node)
    frames,infos,avm,states={},{},{},[]
    subscriptions=[]
    def image(name,msg):
        assert msg.encoding=="rgb8" and msg.step==msg.width*3
        assert len(msg.data)==msg.height*msg.step
        stamp=(msg.header.stamp.sec,msg.header.stamp.nanosec)
        frames.setdefault(stamp,set()).add(name)
    for name in ("front","rear","left","right"):
        subscriptions.append(node.create_subscription(Image,f"/sim/camera/{name}/image_raw",
            lambda m,n=name:image(n,m),10))
        subscriptions.append(node.create_subscription(CameraInfo,f"/sim/camera/{name}/camera_info",
            lambda m,n=name:infos.__setitem__(n,m),10))
    subscriptions.append(node.create_subscription(Image,"/sim/avm/image",
        lambda m:avm.__setitem__((m.header.stamp.sec,m.header.stamp.nanosec),m),10))
    subscriptions.append(node.create_subscription(String,"/sim/scene/state",
        lambda m:states.append(Scene.from_dict(json.loads(m.data)["scene"])),
        QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)))
    try:
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.1)
            complete=[k for k,v in frames.items() if len(v)==4 and k in avm]
            if complete and len(infos)==4 and states and tf.can_transform("world","camera_front_optical",Time()):
                scene=states[-1]
                message=tf.lookup_transform("world","camera_front_optical",Time())
                p,q=message.transform.translation,message.transform.rotation
                actual=matrix_from_wxyz_position(np.array([q.w,q.x,q.y,q.z]),np.array([p.x,p.y,p.z]))
                np.testing.assert_allclose(actual,scene.camera_optical_pose("front"),atol=1e-5)
                for name,info in infos.items():
                    assert info.header.frame_id==f"camera_{name}_optical"
                    np.testing.assert_allclose(info.k,scene.cameras[name].K)
                    assert info.distortion_model=="equidistant"
                print(f"PASS: four RGB images, CameraInfo, matching AVM stamp, TF; stamp={complete[-1]}")
                return
        raise RuntimeError("Timed out waiting for complete ROS image / calibration / TF data")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__=="__main__":
    main()
