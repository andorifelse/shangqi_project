"""Exact-stamp synchronization prevents mixing images and edited extrinsics."""
from __future__ import annotations
import json
from collections import OrderedDict
from scene_manager.models import Scene
from scene_manager.ros_images import to_image,from_image,stamp_key
from .projection import AvmStitcher,REQUIRED_CAMERAS


def main(args=None) -> None:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from sensor_msgs.msg import Image
    from builtin_interfaces.msg import Time

    class StitcherNode(Node):
        def __init__(self):
            super().__init__("avm_stitcher")
            self.stitcher,self.batches = AvmStitcher(),OrderedDict()
            self.pub=self.create_publisher(Image,"/sim/avm/image",10)
            self.meta_sub=self.create_subscription(String,"/sim/render/state",self.meta,10)
            self.subs=[self.create_subscription(Image,f"/sim/camera/{name}/image_raw",
                lambda msg,n=name:self.image(n,msg),10) for name in REQUIRED_CAMERAS]

        def batch(self,key):
            value=self.batches.setdefault(key,{"images":{}})
            while len(self.batches)>12:
                self.batches.popitem(last=False)
            return value

        def meta(self,msg):
            data=json.loads(msg.data)
            key=tuple(data["stamp"])
            self.batch(key)["scene"]=Scene.from_dict(data["scene"])
            self.try_publish(key)

        def image(self,name,msg):
            key=stamp_key(msg.header.stamp)
            self.batch(key)["images"][name]=from_image(msg)
            self.try_publish(key)

        def try_publish(self,key):
            batch=self.batches.get(key)
            if batch and "scene" in batch and all(n in batch["images"] for n in REQUIRED_CAMERAS):
                result=self.stitcher.stitch(batch["scene"],batch["images"])
                self.pub.publish(to_image(result.rgb,Time(sec=key[0],nanosec=key[1]),"base_link"))
                del self.batches[key]

    rclpy.init(args=args)
    node=StitcherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
