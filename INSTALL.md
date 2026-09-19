# 安装与运行

目标环境为 **Ubuntu 24.04 / ROS 2 Jazzy / Python 3.12 / NVIDIA GPU**。Windows standalone 是本轮已经运行的开发 fallback；不替代 ROS 验收。

## 1. 本机 Windows 已验证路径

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

## 2. Ubuntu 24.04 与 ROS 2 Jazzy

按 [ROS 2 Jazzy 官方 Debian 包安装说明](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html) 配置 ROS apt 源，再安装：

```bash
sudo apt update
sudo apt install ros-jazzy-ros-base ros-jazzy-tf2-ros ros-jazzy-sensor-msgs \
  ros-jazzy-geometry-msgs ros-jazzy-launch-ros ros-dev-tools \
  python3-venv python3-pip python3-colcon-common-extensions \
  build-essential cmake ninja-build git
source /opt/ros/jazzy/setup.bash
```

使用 Ubuntu 自带 Python 3.12 创建可读取 ROS 系统包的虚拟环境。不要用 Conda Python 构建 rclpy：

```bash
cd /path/to/shangqi_project
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --index-url https://pypi.org/simple -r requirements.txt
python -m pip install setuptools wheel colcon-common-extensions
python -c "import rclpy, numpy, cv2, viser; print('Core dependencies OK')"
python -m pytest -q
python tools/render_smoke.py
bash tools/build_ros.sh
```

如本机尚未初始化 rosdep，可按 ROS 官方说明先初始化，再执行：

```bash
rosdep update
rosdep install --from-paths avm_sim_ws/src --ignore-src -r -y --rosdistro jazzy
```

rosdep 负责 ROS 包；Python 第三方包由 requirements.txt 提供。核心图像编码直接用 sensor_msgs，不依赖 cv_bridge 的 NumPy ABI。

CPU 启动：

```bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source avm_sim_ws/install/setup.bash
ros2 launch avm_bringup avm_sim.launch.py backend:=cpu
```

源码已包含测试资产与 scene.yaml，colcon 安装时复制到 share/avm_bringup，示例相对路径保持可用。tools/build_ros.sh 在虚拟环境内用 python -m colcon 保证 console scripts 的解释器一致。

## 3. CUDA 与 gsplat

先使用 NVIDIA 官方方式安装与驱动兼容的 CUDA Toolkit 和 C++ 编译器。不要只安装 CUDA runtime 就假定可以 JIT 编译 gsplat。Ubuntu 24.04 建议使用支持其编译器的 CUDA 12.6 Toolkit。

以下为可选的固定版本组合，来自 [PyTorch 官方历史版本说明](https://docs.pytorch.org/get-started/previous-versions/) 和 [gsplat 1.5.3](https://pypi.org/project/gsplat/)，**本轮尚未在目标主机实测**：

```bash
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install --index-url https://pypi.org/simple -r requirements-gpu.txt
export CUDA_HOME=/usr/local/cuda-12.6
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

## 4. ROS 运行验收命令

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

只有这些命令和 native smoke 在目标环境通过后，才可把 ROS/CUDA 阶段标为已验收。本轮没有执行这些命令的有效 ROS 环境。

## 5. 替换资产

- 训练 Gaussian PLY 需 x/y/z、f_dc_0..2、opacity、scale_0..2、rot_0..3。opacity 为 logit，scale 为 log 标准差，rot 为 WXYZ。普通 XYZ 点云会被拒绝。
- PLY 可为 ASCII 或 binary，均通过 plyfile 读取。
- scene.gaussian_file、vehicle.asset 相对于 YAML 文件；保存到不同目录会重写相对路径。
- vehicle GLB/GLTF 作者坐标不一定为 Z-up，请在 vehicle.mesh_pose / mesh_scale 配置；业务 pose 始终按统一米制。
- camera、board 不写入 PLY，也不需要重新训练。
