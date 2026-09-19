#!/usr/bin/env bash
# Create the system-Python virtualenv required by binary ROS packages.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYSTEM_PYTHON="${AVM_SYSTEM_PYTHON:-/usr/bin/python3}"
VENV_DIR="${AVM_VENV:-$PROJECT_ROOT/.venv-ros}"

if [[ -z "${ROS_DISTRO:-}" && -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    case "${VERSION_ID:-}" in
        22.04) ROS_DISTRO=humble ;;
        24.04) ROS_DISTRO=jazzy ;;
    esac
fi

if [[ -z "${ROS_DISTRO:-}" ]]; then
    echo "Unsupported host. Set ROS_DISTRO explicitly (for example, humble)." >&2
    exit 1
fi

ROS_SETUP="${ROS_SETUP:-/opt/ros/$ROS_DISTRO/setup.bash}"
if [[ ! -r "$ROS_SETUP" ]]; then
    echo "ROS $ROS_DISTRO is not installed: missing $ROS_SETUP" >&2
    echo "Ubuntu 22.04 should install ROS 2 Humble before running this script." >&2
    exit 1
fi
if [[ ! -x "$SYSTEM_PYTHON" ]]; then
    echo "System Python is missing: $SYSTEM_PYTHON" >&2
    exit 1
fi

# ROS-generated setup scripts may read optional variables before defining them,
# which is incompatible with nounset. Keep strict mode for our own code.
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
set -u

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$SYSTEM_PYTHON" -m venv --system-site-packages "$VENV_DIR"
fi

SYSTEM_VERSION="$($SYSTEM_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
VENV_VERSION="$($VENV_DIR/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$SYSTEM_VERSION" != "$VENV_VERSION" ]]; then
    echo "$VENV_DIR uses Python $VENV_VERSION, but ROS $ROS_DISTRO uses system Python $SYSTEM_VERSION." >&2
    echo "Move or remove that environment, then rerun this script." >&2
    exit 1
fi

"$VENV_DIR/bin/python" -m pip install \
    --index-url "${AVM_PYPI_INDEX:-https://pypi.org/simple}" \
    -r "$PROJECT_ROOT/requirements.txt" setuptools wheel colcon-common-extensions

if ! "$VENV_DIR/bin/python" -c "import rclpy" 2>/dev/null; then
    echo "rclpy is not visible in $VENV_DIR. Ensure the venv was created with --system-site-packages." >&2
    exit 1
fi

if command -v rosdep >/dev/null 2>&1; then
    rosdep install --from-paths "$PROJECT_ROOT/avm_sim_ws/src" \
        --ignore-src -r -y --rosdistro "$ROS_DISTRO" --skip-keys ament_python
else
    echo "rosdep is missing. Install ros-dev-tools before continuing." >&2
    exit 1
fi

ROS_DISTRO="$ROS_DISTRO" AVM_VENV="$VENV_DIR" "$PROJECT_ROOT/tools/build_ros.sh"
