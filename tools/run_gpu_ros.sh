#!/usr/bin/env bash
# Start the real 3DGS scene with the verified CUDA 11.8 / Humble environment.
set -eo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-humble}"
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "$PROJECT_ROOT/.venv-ros/bin/activate"
source "$PROJECT_ROOT/avm_sim_ws/install/setup.bash"
export CUDA_HOME="${CUDA_HOME:-/home/wzc/cuda-11.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA device unavailable; check nvidia-smi"; print("GPU:", torch.cuda.get_device_name(0))'
cd "$PROJECT_ROOT"
exec ros2 launch avm_bringup avm_sim.launch.py \
    scene:="$PROJECT_ROOT/outputs/vehicle_3dgs_scene.yaml" \
    backend:=gsplat render_hz:=10.0 "$@"
