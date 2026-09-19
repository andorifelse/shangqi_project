#!/usr/bin/env bash
# Build with the ROS distribution that matches the host (22.04=Humble, 24.04=Jazzy).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${ROS_DISTRO:-}" && -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    case "${VERSION_ID:-}" in
        22.04) ROS_DISTRO=humble ;;
        24.04) ROS_DISTRO=jazzy ;;
    esac
fi

if [[ -z "${ROS_DISTRO:-}" ]]; then
    echo "Unable to select ROS distribution. Set ROS_DISTRO=humble or ROS_DISTRO=jazzy." >&2
    exit 1
fi

ROS_SETUP="${ROS_SETUP:-/opt/ros/$ROS_DISTRO/setup.bash}"
VENV_DIR="${AVM_VENV:-$PROJECT_ROOT/.venv-ros}"

if [[ ! -r "$ROS_SETUP" ]]; then
    echo "ROS $ROS_DISTRO is not installed: missing $ROS_SETUP" >&2
    echo "Install ROS first, then run tools/setup_ros_env.sh." >&2
    exit 1
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "ROS virtual environment is missing: $VENV_DIR" >&2
    echo "Create it with tools/setup_ros_env.sh." >&2
    exit 1
fi

# ROS/venv-generated setup scripts may read optional variables before defining
# them, which is incompatible with nounset. Keep strict mode for our own code.
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
set -u

if ! python -c "import rclpy" 2>/dev/null; then
    echo "rclpy cannot be imported by $VENV_DIR/bin/python." >&2
    echo "The ROS venv must use /usr/bin/python3 and --system-site-packages." >&2
    exit 1
fi

cd "$PROJECT_ROOT/avm_sim_ws"
python -m colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE="$VENV_DIR/bin/python"
set +u
# shellcheck disable=SC1091
source install/setup.bash
set -u
python -c "import rclpy, viser, scene_manager, scene_editor, gs_renderer, avm_stitcher"
echo "ROS $ROS_DISTRO build and import check passed."
