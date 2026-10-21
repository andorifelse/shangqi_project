#!/usr/bin/env bash
# Record current ROS simulation topics into an existing experiment's logs directory.
set -euo pipefail
if [[ $# -lt 1 ]]; then
    echo "Usage: tools/record_rosbag.sh experiments/<id> [topic ...]" >&2
    exit 2
fi
EXPERIMENT_DIR="$(cd "$1" && pwd)"
shift
if [[ ! -f "$EXPERIMENT_DIR/config.yaml" ]]; then
    echo "Not an experiment directory: $EXPERIMENT_DIR" >&2
    exit 2
fi
if ! command -v ros2 >/dev/null 2>&1; then
    echo "ROS 2 environment is not sourced. Source Humble and .venv-ros first." >&2
    exit 2
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BAG_DIR="$EXPERIMENT_DIR/logs/rosbag2_$STAMP"
mkdir -p "$EXPERIMENT_DIR/logs"
if [[ $# -eq 0 ]]; then
    set -- /tf /sim/scene/state /sim/ground_truth /sim/render/state \
        /sim/camera/front/image_raw /sim/camera/front/camera_info \
        /sim/camera/rear/image_raw /sim/camera/rear/camera_info \
        /sim/camera/left/image_raw /sim/camera/left/camera_info \
        /sim/camera/right/image_raw /sim/camera/right/camera_info /sim/avm/image
fi
echo "Recording to $BAG_DIR (Ctrl+C stops recording)"
exec ros2 bag record -o "$BAG_DIR" "$@"
