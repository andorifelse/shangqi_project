#!/usr/bin/env bash
# Replay recorded ROS topics from an experiment's rosbag2 folder.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: tools/replay_experiment.sh experiments/<id>" >&2
    exit 2
fi
EXPERIMENT_DIR="$(cd "$1" && pwd)"
BAG_DIR="$EXPERIMENT_DIR/logs/rosbag2"
if [[ ! -f "$BAG_DIR/metadata.yaml" ]]; then
    echo "No rosbag2 recording found at $BAG_DIR" >&2
    exit 2
fi
if ! command -v ros2 >/dev/null 2>&1; then
    echo "ROS 2 environment is not sourced." >&2
    exit 2
fi
exec ros2 bag play "$BAG_DIR"
