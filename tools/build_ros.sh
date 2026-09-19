#!/usr/bin/env bash
# Run after ROS Jazzy and the project virtualenv have been installed.
set -eo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT/avm_sim_ws"
python -m colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE="$PROJECT_ROOT/.venv/bin/python"
source install/setup.bash
python -c "import rclpy, viser, scene_manager, scene_editor, gs_renderer, avm_stitcher"
