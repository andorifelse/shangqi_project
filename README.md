# ROS 2 AVM 标定仿真 MVP

已实现并在本机运行一个可交互原型：3DGS PLY + Viser 编辑器、车辆/棋盘格/4–6 路相机、6DoF 数值与 Gizmo、场景保存加载、真实相机射线生成四路鱼眼图像，以及由这四路图像生成的 BEV/AVM。

**当前验收范围：Windows standalone + CPU 几何参考渲染。** Ubuntu 24.04 / ROS 2 Jazzy 架构、节点、TF 和 launch 已提供；本机没有 ROS 2/WSL，CUDA backend 缺 PyTorch 和编译环境，因此两条目标路径仍须在配置好的主机验收。未宣称达到实时 30 FPS 或真实资产保真度。

## 立即运行

在项目根目录 PowerShell：

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/run_demo.py
```

打开 [本地编辑器](http://127.0.0.1:8080)。首次会生成 example_assets 测试 PLY 和车辆 GLB。安装步骤见 [INSTALL.md](INSTALL.md)。

```powershell
# 只启动编辑器，不启动传感器
.\.venv\Scripts\python.exe -X utf8 tools/run_demo.py --no-render

# 指定保存过的配置
.\.venv\Scripts\python.exe -X utf8 tools/run_demo.py --scene outputs/scene.yaml

# 更换真实 Gaussian PLY
.\.venv\Scripts\python.exe -X utf8 tools/run_demo.py --ply assets/factory.ply

# 全部本地测试、独立四路与 AVM 出图
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -X utf8 tools/render_smoke.py
```

独立输出位于 outputs/smoke/front.png、rear.png、left.png、right.png、avm.png，性能数据在 metrics.json。这些图像来自 SensorPipeline，与浏览器视口无关。

## 操作

1. 左键拖动空白处 Orbit，右键拖动 Pan，滚轮 Zoom。
2. Scene Objects 下拉或点击模型选择物体；Add Vehicle / Add Camera / Add Board 添加；Delete selected 删除。MVP 一个车辆，最多六个相机；删除车辆同时删除挂载于该车的相机。
3. Transform 的 XYZ 为米，Roll/Pitch/Yaw 为度。相机数值相对于 parent。拖动彩色箭头平移、圆弧旋转，数值同步回写。Gizmo 模式可切换 Translate / Rotate。
4. Camera parameters 编辑分辨率、fx/fy/cx/cy、D、有效半角后 Apply。pinhole 需 5 个 D，fisheye/ftheta 需 4 个 D；无效参数会显示错误并保留原状态。
5. Board parameters 编辑方格行列数和边长后 Apply，物理宽高自动计算。标定板是独立动态对象。
6. Scene files 输入服务器本地路径；Save Scene / Load Scene 保存加载。Load Gaussian PLY 导入 3DGS；Set vehicle asset 设置 GLB/GLTF。
7. 展开 Camera Preview 和 AVM Preview。CPU 模式编辑后约数秒刷新一次；3D 视口保持独立交互。Rendering 可暂停传感器计算。

示例 [scene.yaml](avm_sim_ws/src/avm_bringup/config/scene.yaml) 使用相对资产路径，可整体复制。YAML pose 角度是弧度，与 UI 度数不同。真实资产需要设置 coordinate.scale / T_world_from_gs / ground_z，以及车辆 mesh_pose / mesh_scale；不会隐式归一化尺度。

## ROS 2 启动

先按 INSTALL.md 安装 Jazzy 并构建，然后：

```bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source avm_sim_ws/install/setup.bash
ros2 launch avm_bringup avm_sim.launch.py backend:=cpu
# 配好 CUDA 后运行原生 gsplat
ros2 launch avm_bringup avm_sim.launch.py backend:=gsplat render_hz:=5.0
```

可设置 scene:=/absolute/path/scene.yaml、port:=8080、host:=127.0.0.1。默认只监听本机。

标准 topic：

```text
/tf
/sim/camera/{front,rear,left,right}/image_raw
/sim/camera/{front,rear,left,right}/camera_info
/sim/avm/image
```

额外两相机同样生成图像 topic，AVM 当前只使用指定四路。

## 阶段与验收状态

| 阶段 | 实现和验证 |
| --- | --- |
| M1 | 六个 ROS package、Gaussian PLY、米制 Sim(3)、Viser；已实际运行与浏览器旋转缩放 |
| M2 | 车辆/板/相机增删、选择、XYZ/RPY、Gizmo；已验证数值、平移和旋转回写 |
| M3 | YAML 原子保存/加载、父子位姿、TF 描述；本地测试通过，ROS /tf 尚未实机验收 |
| M4 | pinhole CPU 射线渲染已测位姿变化；原生 gsplat adapter 已写，CUDA 未验收 |
| M5 | 四路 fisheye 与 ftheta CPU 射线；OpenCV 投影对照通过；native ftheta 待实现 |
| M6 | checkerboard 射线平面相交、纹理采样、深度遮挡；本地通过 |
| M7 | 四路图像地面投影、重叠融合、AVM；本地出图通过 |
| M8 | worker 隔离、静态数据/投影缓存、文档和测试完成；目标 GPU 优化待进行 |

验证证据和环境边界见 [VALIDATION.md](docs/VALIDATION.md)。目录与坐标协议见 [ARCHITECTURE.md](docs/ARCHITECTURE.md)，需求和视频分析见 [REFERENCE_ANALYSIS.md](docs/REFERENCE_ANALYSIS.md)。

## 已知问题与未完成工作

- ROS 2 Jazzy 的 colcon build、节点通信、launch、真实 /tf 和 CameraInfo 尚未在目标系统运行。这是当前最主要的验收缺口。
- 本机已安装 gsplat Python 包用于核对源码，但未安装 PyTorch，也无 CUDA C++ 编译器链；--backend gsplat 会明确报错。CPU backend 可以正常使用。
- CPU 参考路径针对小测试场景，数万/百万 Gaussian 会很慢；每批四路 320×240 + 512×512 AVM 的实测约 2.7–3.6 秒。它是按射线计算的近似体渲染，未追求与 CUDA rasterizer 像素一致。
- ftheta 当前走 CPU，并使用归一化 theta 多项式；厂商六系数 pixel-space FTheta 参数适配未实现。fisheye 有效域限制小于 180° 完整 FOV，宽角使用 ftheta 参考路径。
- 原生 gsplat 只读取 DC 颜色；PLY 的高阶球谐尚未参与渲染，无法重现视角相关反光。
- 车辆 mesh 在编辑器显示；传感器目前未加入车辆 mesh 的遮挡/材质渲染（棋盘格已经加入）。镜头应放在车体外。GLTF 的外部纹理应和文件一起保存。
- Gaussian expected depth 的体渲染特性导致透明区域、近地棋盘格边缘可能有遮挡偏差；棋盘格贴地过近时 Viser 中也可能被 Gaussian 视觉遮盖。可通过真实物理摆放调整 Z。
- AVM 是平面假设，立体物体会拉伸/重影；无 seam/exposure/multiband，中心盲区保持暗色。测试场景约 95.4% 范围有源图覆盖，非精度指标。
- GUI 的相机 frustum 是方向提示，不表示鱼眼的完整非线性有效边界。对象选择为列表，Viser 内置 scene tree 可查真实层级。
- 目前 overlay、AVM 和 ROS/GUI 编码在 CPU，CUDA 后处理与端到端零拷贝尚未实现。独立 worker 保留替换边界。
- 未做真实相机内外参、真实训练 PLY、真实车模的精度与性能验证；示例均为明确标记的测试数据。
- 用户管理、日志中心、报告、AI 生成、版本管理、自动标定与批量标定按本轮要求不实现。

## 下一阶段

先在 Ubuntu 24.04 + Jazzy 上完成 colcon、TF、Image/CameraInfo 和 exact-stamp 拼接验收，再接入真实 PLY、GLB 与四路标定参数。随后进行 CUDA 原生渲染数值对照、动态物体遮挡精化、GPU overlay/BEV 优化，并测量实际 FPS、延迟与显存。暂不扩展外围业务功能。
