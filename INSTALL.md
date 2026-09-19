# 安装与运行

主开发环境为 **Ubuntu 22.04 / ROS 2 Humble / 系统 Python 3.10**。构建脚本同时支持 Ubuntu 24.04 / ROS 2 Jazzy；Windows standalone 继续作为非 ROS 开发路径。CUDA/gsplat 是可选能力，不影响 CPU 版本运行。

## 1. Ubuntu 22.04 standalone（已验证）

不安装 ROS 也可运行编辑器、CPU 四路渲染和 AVM：

```bash
cd /path/to/shangqi_project
/usr/bin/python3 -m venv .venv
.venv/bin/python -m pip install --index-url https://pypi.org/simple -r requirements.txt
.venv/bin/python tools/check_environment.py
.venv/bin/python -m pytest -q
.venv/bin/python tools/render_smoke.py
.venv/bin/python tools/run_demo.py
```

打开 [http://127.0.0.1:8080](http://127.0.0.1:8080)，退出用 Ctrl+C。Ubuntu 上明确使用 `/usr/bin/python3`，避免激活 Conda 后 `python3` 指向其他版本；ROS 模式另用 `.venv-ros`，以读取 apt 安装的 `rclpy`。

## 2. Windows standalone 已验证路径

项目内已创建 .venv，依赖没有写进系统 Python。

```powershell
cd D:\shangqi_project
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 tools/check_environment.py
.\.venv\Scripts\python.exe -X utf8 tools/run_demo.py
```

打开 [http://127.0.0.1:8080](http://127.0.0.1:8080)。退出用 Ctrl+C。完整测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -X utf8 tools/render_smoke.py
```

已测主要版本：Python 3.12.4、Viser 1.1.1、NumPy 2.5.3、SciPy 1.18.1、OpenCV 5.0.0.93、trimesh 5.1.0、plyfile 1.1.5、PyYAML 6.0.3。完整记录见 requirements-tested.txt；它只记录本机环境，不强制 Ubuntu 使用未来或平台限定轮子。

本机 NVIDIA GeForce RTX 4060 Laptop GPU 8GB，驱动 566.24。nvidia-smi 的 CUDA 12.7 是驱动支持能力；nvcc 实际 Toolkit 为 11.8。原始环境无 torch、gsplat、ROS、WSL，后续只安装了 gsplat 1.5.3 Python 包用于检查官方源码；未安装 PyTorch/CUDA 编译依赖。当前不能运行 native backend。

原 pip 镜像出现 TLS EOF，改用官方 PyPI 成功。不禁用 TLS 验证。

## 3. Ubuntu 22.04 与 ROS 2 Humble

按 [ROS 2 Humble 官方 Ubuntu 安装说明](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) 配置 ROS apt 源，再安装：

```bash
sudo apt update
sudo apt install ros-humble-ros-base ros-humble-tf2-ros ros-humble-sensor-msgs \
  ros-humble-geometry-msgs ros-humble-launch-ros ros-dev-tools \
  python3-venv python3-pip python3-colcon-common-extensions \
  build-essential cmake ninja-build git
source /opt/ros/humble/setup.bash
```

如本机尚未初始化 rosdep，执行一次：

```bash
sudo rosdep init
rosdep update
```

然后使用自动脚本创建 `.venv-ros`、安装 Python/ROS 依赖并构建：

```bash
cd /path/to/shangqi_project
bash tools/setup_ros_env.sh
```

脚本在 Ubuntu 22.04 自动选择 Humble，并强制使用 `/usr/bin/python3` 和 `--system-site-packages` 创建环境。已有 `.venv-ros` 不会被覆盖；如果其 Python 版本不匹配，脚本会停止并给出错误。`rosdep` 负责 ROS 包，requirements.txt 提供第三方 Python 包。核心图像编码直接使用 sensor_msgs，不依赖 cv_bridge 的 NumPy ABI。

CPU 启动：

```bash
source /opt/ros/humble/setup.bash
source .venv-ros/bin/activate
source avm_sim_ws/install/setup.bash
ros2 launch avm_bringup avm_sim.launch.py
```

默认 backend 已改为 CPU。源码包含测试资产与 scene.yaml，colcon 安装时复制到 share/avm_bringup，示例相对路径保持可用。`tools/build_ros.sh` 自动识别 Humble/Jazzy，并在虚拟环境内通过 `python -m colcon` 构建。

Ubuntu 24.04 使用 Jazzy 时流程相同：安装 `ros-jazzy-*` 软件包后运行 `bash tools/setup_ros_env.sh`，脚本会自动选择 `/opt/ros/jazzy` 和系统 Python 3.12。也可以通过 `ROS_DISTRO`、`ROS_SETUP`、`AVM_VENV` 显式覆盖。

## 4. CUDA 与 gsplat

先确认 `nvidia-smi` 和 `nvcc --version` 都正常。不要只安装 CUDA runtime 就假定可以 JIT 编译 gsplat；PyTorch wheel 的 CUDA 版本应与本地 `nvcc` 主版本一致。

本机已有 CUDA Toolkit 11.8，因此 Ubuntu 22.04 建议先验证以下组合。PyTorch 官方提供 Python 3.10 的 2.7.1 + cu118 wheel；gsplat 固定为 requirements-gpu.txt 中的 1.5.3。**该组合尚未在当前 GPU 设备上实测**：

```bash
source .venv-ros/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --index-url https://pypi.org/simple -r requirements-gpu.txt
# 当前机器的 nvcc 位于此处；如果安装位置不同，相应修改。
export CUDA_HOME=/home/wzc/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python tools/render_smoke.py --backend gsplat
AVM_TEST_GPU=1 python -m pytest -q tests/test_gpu_optional.py
ros2 launch avm_bringup avm_sim.launch.py backend:=gsplat render_hz:=5.0
```

第一次 CUDA 调用可能 JIT 编译。显存不足时先减少相机分辨率或 Gaussian 数量；内参与图像尺寸必须同步缩放，不要只改 width/height。CPU fallback 从来不是“上帝视角截图”。选 gsplat 而缺少依赖时会明确报错，不悄悄换后端；ftheta 是唯一明确标注的 CPU 子路径。

发布版 API 已检查：
- [Viser Gaussian splats](https://viser.studio/main/examples/scene/gaussian_splats/)
- [gsplat v1.5.3 rendering.py](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/rendering.py)
- 本机安装包中 ProjectionUT3DGSFused.cu 使用 mean_c.z 深度；RasterizeToPixelsFromWorld3DGSFwd.cu 使用像素中心 +.5。

## 5. ROS 运行验收命令

在第二个终端同样 source 环境后：

```bash
python tools/verify_ros_runtime.py
ros2 node list
ros2 topic list
ros2 topic hz /sim/camera/front/image_raw
ros2 topic hz /sim/avm/image
ros2 topic echo /sim/camera/front/camera_info --once
ros2 run tf2_ros tf2_echo world camera_front_optical
```

浏览器移动车辆后 world->camera_front_optical 应变，base_link->camera_front_link 应不变。移动相机时后者应变。四路图像的同批 stamp 和 CameraInfo 应一致。

Ubuntu 22.04/Humble 已通过上述 Python 验证脚本：四路 RGB、CameraInfo、匹配的 AVM stamp 和 TF 均正常。CUDA native smoke 尚未验收。

## 6. 替换资产

- 训练 Gaussian PLY 需 x/y/z、f_dc_0..2、opacity、scale_0..2、rot_0..3。opacity 为 logit，scale 为 log 标准差，rot 为 WXYZ。普通 XYZ 点云会被拒绝。
- PLY 可为 ASCII 或 binary，均通过 plyfile 读取。
- scene.gaussian_file、vehicle.asset 相对于 YAML 文件；保存到不同目录会重写相对路径。
- vehicle GLB/GLTF 作者坐标不一定为 Z-up，请在 vehicle.mesh_pose / mesh_scale 配置；业务 pose 始终按统一米制。
- camera、board 不写入 PLY，也不需要重新训练。
