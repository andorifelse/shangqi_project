"""Capture authoritative ROS SceneState, record rosbag2, and save an experiment manifest."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
for package in (ROOT/"avm_sim_ws"/"src").iterdir():
    if package.is_dir(): sys.path.insert(0,str(package))
from avm_bringup.experiments import begin_experiment_from_snapshot,finish_experiment,append_log
from scene_manager.models import Scene


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path)
    parser.add_argument("--experiment-id",default=datetime.now().strftime("rosbag_%Y%m%d_%H%M%S"))
    parser.add_argument("--duration",type=float,default=30.)
    args=parser.parse_args()
    if args.duration<=0:
        parser.error("--duration must be positive")
    import rclpy
    from rclpy.qos import QoSProfile,DurabilityPolicy
    from std_msgs.msg import String
    rclpy.init()
    node=rclpy.create_node("avm_experiment_recorder")
    states=[]
    sub=node.create_subscription(String,"/sim/scene/state",lambda msg:states.append(json.loads(msg.data)),
        QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    deadline=time.monotonic()+10
    try:
        while not states and time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.1)
        if not states:
            raise RuntimeError("No authoritative /sim/scene/state received; start the simulation in the same ROS domain")
        latest=states[-1]
        scene=Scene.from_dict(latest["scene"])
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
    topics=["/tf","/sim/scene/state","/sim/ground_truth","/sim/render/state","/sim/avm/image"]
    topics += [f"/sim/camera/{name}/{suffix}" for name in scene.cameras for suffix in ("image_raw","camera_info")]
    output=args.output or ROOT/"experiments"/args.experiment_id
    from scene_manager.ground_truth import GroundTruth
    snapshot=GroundTruth.capture(scene,latest["revision"])
    output=begin_experiment_from_snapshot(snapshot,output,{"pipeline":"rosbag2","duration_s":args.duration,
        "topics":topics,"ROS_DOMAIN_ID":os.environ.get("ROS_DOMAIN_ID","0")},ROOT)
    command=["ros2","bag","record","-o",str(output/"logs"/"rosbag2"),*topics]
    append_log(output,f"Recording {command}")
    started=time.monotonic()
    try:
        bag_path=output/"logs"/"rosbag2"
        with (output/"logs"/"rosbag_record.log").open("w",encoding="utf-8") as log:
            command=["ros2","bag","record","-o",str(bag_path),*topics]
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                process.wait(timeout=args.duration)
                if process.returncode:
                    raise RuntimeError(f"rosbag2 exited early with code {process.returncode}; see logs/rosbag_record.log")
            except (subprocess.TimeoutExpired,KeyboardInterrupt):
                os.killpg(process.pid,signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    process.wait(timeout=5)
            if process.returncode not in (0,-signal.SIGINT):
                raise RuntimeError(f"rosbag2 exit code {process.returncode}")
        info=subprocess.run(["ros2","bag","info",str(bag_path)],capture_output=True,text=True,timeout=15,check=True)
        (output/"logs"/"rosbag_info.txt").write_text(info.stdout,encoding="utf-8")
        elapsed=time.monotonic()-started
        (output/"metrics.csv").write_text(f"duration_s,topics\n{elapsed:.6f},{len(topics)}\n",encoding="utf-8")
        finish_experiment(output,{"status":"complete","recorded_seconds":elapsed,"topics":topics,
            "scene_revision":snapshot.revision,"bag":str(bag_path)})
        print(info.stdout)
        print(f"Recorded experiment: {output}")
    except Exception as exc:
        finish_experiment(output,{"status":"failed","error":f"{type(exc).__name__}: {exc}"})
        raise


if __name__=="__main__": main()
