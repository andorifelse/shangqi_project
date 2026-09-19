# 运行验证记录

首次记录：2026-09-17，Windows 11 / Python 3.12.4 / RTX 4060 Laptop 8GB。Linux 复验：2026-09-19，Ubuntu 22.04 / Python 3.13.5 standalone。

## 实际执行

| 检查 | 结果 |
| --- | --- |
| 项目需求.docx | 使用 bundled python-docx 提取全部段落并阅读，无修改原件 |
| demo_example.mp4 | 解码 1535 帧元数据，24 张关键帧覆盖 107.16 秒并检查 |
| python tools/run_demo.py | 多轮启动、修复错误、浏览器连接成功 |
| 原生 Viser Gaussian | PLY splats 可见；Orbit / Zoom 经鼠标验证 |
| 对象管理 | 车辆、棋盘格、相机 frustum 显示；添加板、选择和数值编辑验证 |
| Gizmo 平移 | 鼠标拖动车辆 Z 箭头，Z 数值从 0 变为约 1.23 m |
| Gizmo 旋转 | 鼠标拖动车辆圆弧，Pitch 回写为 24 degree |
| Camera Preview | 浏览器展开后显示 front/rear/left/right；front 包含独立棋盘格 |
| Save / Load GUI | 实际保存后添加 board_002，重载后仅剩保存时 board_001 |
| Worker 错误恢复 | 无效 PLY 返回错误后仍可继续渲染有效四路场景 |
| Python packaging | 五个 setup.py check、wheel build 与 compileall 成功；bringup 安装包包含 launch/YAML/PLY/GLB；不等同于 colcon |
| AVM Preview | 已在浏览器展开查看由四路图像生成的 BEV |
| CPU 四路 + AVM | tools/render_smoke.py 成功写五张 PNG |
| pytest | Ubuntu 22.04 / Python 3.10：22 passed, 1 skipped |
| CUDA smoke | 实际运行 --backend gsplat，明确失败：未安装 torch |
| ROS2 / colcon | Humble 六个 package 构建成功；launch、四路图像、CameraInfo、AVM stamp 与 TF 通过 |

## Ubuntu 22.04 standalone 复验

在 `/home/wzc/shangqi_project` 建立隔离 `.venv` 后实际执行：

- `tools/check_environment.py`：Viser 1.1.1、NumPy 2.5.3、SciPy 1.18.1、OpenCV 5.0.0.93 等核心依赖导入成功。
- Python 3.13 standalone 的 Viser server 生命周期用例退出异常；改用 Humble 对应的系统 Python 3.10 后完整测试为 22 passed、1 skipped，GPU 用例按环境变量设计跳过。
- `tools/render_smoke.py` 成功输出四路图像和 AVM，CPU 批次 1.880 s，覆盖率 0.954025。
- 完整 CPU 编辑器已在 `127.0.0.1:8080` 启动，浏览器确认场景、Camera Preview、AVM Preview 和 Rendering 控件存在。
- `setup_ros_env.sh` 建立 Python 3.10 `.venv-ros`，六个 ROS package 完成 colcon build；launch 默认 CPU。
- `verify_ros_runtime.py` 实际通过：四路 RGB、CameraInfo、AVM 同时间戳和 TF 均正常。
- 修复 Humble setup 与 Bash `nounset` 的兼容性；Ctrl+C 时四个节点均干净退出。

原始环境与包版本可用 tools/check_environment.py 重现，写入 outputs/environment.json。没有为绕过验证而模拟 ROS 节点或伪造 GPU 成功。

## 几何证据

测试覆盖：
- RPY/quaternion/矩阵与 optical 轴方向、逆变换。
- Gaussian Sim(3) 中心和协方差尺度变换。
- 修改车辆后相机世界位姿变化、局部外参不变。
- 增删 4/6 相机、参数验证、revision 冲突和原子回滚。
- YAML 相对资产路径、保存/加载一致、非法输入保留旧状态。
- pinhole/fisheye/ftheta 投影反投影、OpenCV fisheye 精确对照、背向有效域。
- pinhole 相机位姿变化引起传感器图像变化。
- 棋盘格 ray-plane、平行/背向射线、旋转和深度遮挡。
- BEV 朝向、车辆 roll/pitch 下 world 地面仍水平。
- 相机像素变化引起 AVM 变化；已知世界坐标颜色经完整投影链重建误差中位数 <1.1 灰度、99 分位 <4。
- 使用真实 Viser API 验证 Gizmo world pose 回写相机 parent pose，并验证添加/删除状态。

## 性能

默认 synthetic PLY，四路 320x240，AVM 512x512，CPU：

- 独立脚本实测 2.732 s / batch（包含初次载入/射线构建）。
- 浏览器 worker 首批约 3.55 s（含进程启动/数据传递）。
- 默认 BEV 有效源图覆盖 0.954025（95.4%），这是覆盖率，不是标定精度。
- 静止场景 standalone 只渲染一次，编辑产生新 revision 后再渲染；ROS 节点周期输出。
- 编辑器与 worker 解耦，不为等待传感器批次阻塞 Orbit/Gizmo。

未测 1080P、30 FPS、长期稳定性、真实 PLY 质量或真实标定精度。

## 已发现并修复

- pip 沙箱网络与原镜像 TLS EOF：获准联网后切换官方 PyPI，未禁用证书校验。
- Windows 控制台编码：文件统一 UTF-8，入口使用 -X utf8，UI 标签使用 ASCII 避免乱码。
- Viser add_mesh_trimesh 不接收 side 参数：棋盘格明确构造双面三角形。
- 可视 Gizmo 不易点选：固定屏幕尺寸、加粗并置于模型上方，正确设置 depth_test，浏览器平移/旋转回归通过。
- 浮动面板遮挡场景中心：改用固定侧栏。
- 恶意/损坏 schema 引起 AttributeError：加载时显式结构验证并转换为可见的 ValueError，旧状态保留。
- 场景在编辑时外参易与旧图像混用：ROS stitcher 按渲染批次 stamp 使用对应状态快照。

## 下一次必须进行的验收

在 CUDA 可用环境执行 native render_smoke 与 `AVM_TEST_GPU=1 pytest`，并使用真实资产做长期运行、topic hz 和性能验收。Ubuntu 24.04/Jazzy 保留为兼容复验目标。
