"""Standard ROS Image/CameraInfo helpers; imported only by ROS adapters."""
import numpy as np


def to_image(rgb: np.ndarray, stamp, frame_id: str):
    from sensor_msgs.msg import Image
    msg = Image()
    msg.header.stamp,msg.header.frame_id = stamp,frame_id
    msg.height,msg.width = rgb.shape[:2]
    msg.encoding,msg.is_bigendian,msg.step = "rgb8",0,msg.width*3
    msg.data = np.ascontiguousarray(rgb,dtype=np.uint8).tobytes()
    return msg


def from_image(msg) -> np.ndarray:
    if msg.encoding not in ("rgb8","bgr8"):
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    rows = np.frombuffer(bytes(msg.data),dtype=np.uint8).reshape(msg.height,msg.step)
    rgb = rows[:,:msg.width*3].reshape(msg.height,msg.width,3).copy()
    return rgb if msg.encoding=="rgb8" else rgb[...,::-1].copy()


def camera_info(camera, stamp):
    from sensor_msgs.msg import CameraInfo
    msg = CameraInfo()
    msg.header.stamp,msg.header.frame_id = stamp,f"camera_{camera.name}_optical"
    msg.width,msg.height = camera.width,camera.height
    msg.distortion_model = {"pinhole":"plumb_bob","fisheye":"equidistant","ftheta":"avm_ftheta"}[camera.model]
    msg.k,msg.d = list(map(float,camera.K)),list(map(float,camera.D))
    msg.r = np.eye(3).ravel().tolist()
    msg.p = np.column_stack([camera.matrix,np.zeros(3)]).ravel().tolist()
    return msg


def stamp_key(stamp) -> tuple[int,int]:
    return stamp.sec,stamp.nanosec
