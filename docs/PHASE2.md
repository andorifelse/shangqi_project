# 第二阶段：Ground Truth、标定闭环与实验复现

截至 2026-10-20，第二阶段核心链路已在现有 `Scene`/`SceneStore` 上增量实现。原始 scene schema 继续作为 renderer、GUI、ROS 和 AVM 的共同可写状态；`SceneState` 是其正式接口别名。Ground Truth 是该状态在指定 revision 上的不可变快照，不再维护第二份可编辑参数表。

## 已完成能力

| 模块 | 实现 |
| --- | --- |
| Ground Truth | `SceneStore.ground_truth()` 冻结同一 revision；导出版本、SHA-256、坐标约定、相机 link/optical 正逆变换、棋盘格角点和尺寸 |
| Scene 导出 | GUI Scene files 下新增 Export Ground Truth；ROS scene command 支持 `export_ground_truth`；`/sim/ground_truth` 发布同 revision 的 transient-local 快照；相对 PLY/GLB 资产路径随 YAML 保存 |
| 几何 API | 新增 `world_to_pixel`、`pixel_to_world_rays`、`pixels_to_world_plane`；明确 OpenCV integer pixel center、ROS optical、内部 WXYZ 和 UI degree 约定 |
| BEV 几何 | 新增 `bev_pixel_to_world` 与 `world_to_bev_pixel` 互逆映射 |
| 标定板采样 | 固定 seed 的相机画面覆盖采样，按模型选择视野/距离，保存每张板的世界 pose |
| 仿真图像 | 复用生产相机 unprojection、board ray-plane、checker texture，输出 supersampled RGB PNG；不是理想角点投影图 |
| Checker detection | 使用 OpenCV `findChessboardCornersSB` 检查保存 PNG，保存 detected corner 的对称编号决策；Ground Truth 投影只用于行列对称消歧和一致性检查，OpenCV 拟合使用检测出来的像素坐标 |
| Calibration | OpenCV 4/5 fisheye flag 兼容与 pinhole baseline，图像尺寸初始化 K、零初始化 D；报告重投影 RMS、K/D 误差、camera optical 平移/旋转误差和逐视图结果 |
| Dataset | `scene.yaml`、`ground_truth.yaml`、`cameras/<name>/*.png`、`board_poses.csv`、`metadata.json`、SHA-256 manifest；seed、软件版本、模型参数和图片 pose 可重放 |
| AVM debug | 导出 coordinate-color BEV、grid BEV、camera mask、weight heatmap、overlap count / heatmap、dominant camera 和 projection map NPZ |
| Experiment | `experiments/<id>/scene.yaml`、`ground_truth.yaml`、`config.yaml`、`result.json`、`metrics.csv`、`logs/`；记录运行时依赖、Git diff/status、源文件快照、资产 hash 和运行日志 |
| ROS 记录 | `record_experiment.py` 捕获 ROS authoritative scene + revision 后记录 rosbag2；bag info 保存在 logs，可用 replay 工具播放 |

OpenCV per-view 外参比较采用已知合成板位姿：`T_world_camera_est = T_world_board @ inverse(T_camera_board_est)`。这验证由图像恢复相机光学坐标系的能力；如果现实标定板位姿也未知，则单相机标定存在整体 world gauge，需要机器人位姿或固定参考另行约束。

## 验证记录

运行场景：`yaml/scene_v2.yaml`；40 个 seed pose/相机，seed `20261020`，保持此场景原有 world、vehicle、board 和 camera 参数。

| Camera | 角点检出 | Reprojection RMS | Focal 相对误差 | Translation RMSE | Rotation RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| front | 36/40 | 0.0946 px | 0.174% | 4.30 mm | 0.184° |
| left | 38/40 | 0.1075 px | 0.152% | 4.86 mm | 0.186° |
| rear | 39/40 | 0.1029 px | 0.384% | 5.67 mm | 0.184° |
| right | 36/40 | 0.0984 px | 0.125% | 4.47 mm | 0.180° |

结果目录：`experiments/calibration_final_20261020/`。重新读取这批已保存的 PNG 标定，四相机完整 result JSON 与原运行逐值一致。样例参数畸变全为零，因此其 coefficient error 是优化在离散/模糊角点上的估计误差，不是算法误差界。

AVM debug 验证：

- Ground world point ↔ BEV pixel 往返最大像素误差：0。
- Synthetic coordinate-color 图重建：median 0.301、p99 1.809 灰度级。
- scene_v2 对应相机覆盖率：71.28%。

最终版 rosbag2 实测：录制命令运行 8 秒，bag 时间跨度 7.377 秒，sqlite3 大小 6.8 MiB、190 条消息、13 个 topic，包含 `/tf`、scene/render state、Ground Truth、四路 Image/CameraInfo 和 AVM。记录目录：`experiments/rosbag_phase2_gt_final_20261020/logs/rosbag2/`。在另一个 ROS 域执行 `replay_experiment.sh`，实际收到 1 条 Ground Truth、1 条 scene state、4 张 AVM；回放 Ground Truth 与保存 YAML 一致。另通过 ROS service 导出 YAML，确认与 `/sim/ground_truth` 同 revision 的完整状态一致。

最终 Humble 六包重新构建和导入检查通过。默认回归为 41 passed、4 skipped（CUDA opt-in）；启用 `AVM_TEST_GPU=1` 后完整回归为 **45 passed，零跳过**，包括 pinhole/fisheye Gaussian 图像质心与数学投影对照。AVM 数学命令已实跑，几何 debug 输出位于 Git 忽略的 `outputs/avm_validation_phase2_final/`。

本机复验命令：

```bash
.venv-ros/bin/python -m pytest -q
AVM_TEST_GPU=1 CUDA_HOME=/home/wzc/cuda-11.8 TORCH_CUDA_ARCH_LIST=8.9 \
  .venv-ros/bin/python -m pytest -q
bash tools/build_ros.sh
```

CUDA 回归必须能访问本机 NVIDIA 驱动；沙箱若禁止 GPU 或 DDS socket，需要在具备对应权限的终端运行。上述测试未修改 `yaml/scene_v2.yaml`，也未开展真实环境坐标标定或真实车辆配置。

## 运行入口

```bash
cd /home/wzc/shangqi_project
.venv-ros/bin/python tools/run_calibration.py \
  --scene yaml/scene_v2.yaml \
  --experiment-id calibration_my_run \
  --views 40 --seed 20261020

.venv-ros/bin/python tools/calibrate_dataset.py \
  experiments/calibration_my_run \
  --output /tmp/calibration_replay.json

.venv-ros/bin/python tools/generate_dataset.py \
  --scene yaml/scene_v2.yaml \
  --output outputs/dataset_my_run \
  --views 40 --seed 20261020

.venv-ros/bin/python tools/verify_avm.py \
  --scene yaml/scene_v2.yaml \
  --output outputs/avm_debug_scene_v2

# ROS 记录/回放需要先加载环境；录制时另一个终端启动仿真，使用相同 ROS_DOMAIN_ID
source /opt/ros/humble/setup.bash
source .venv-ros/bin/activate
source avm_sim_ws/install/setup.bash
.venv-ros/bin/python tools/record_experiment.py \
  --duration 30 --experiment-id rosbag_my_run
# 回放时停止该域里的仿真，或在独立 ROS_DOMAIN_ID 下启动回放和订阅者
tools/replay_experiment.sh experiments/rosbag_my_run
```

OpenCV baseline 支持 pinhole 和 fisheye。ftheta 与任意厂商 pixel-space FTheta 模型尚未适配 OpenCV 标定。标定图使用简洁 checkerboard ray renderer，以隔离板/相机几何；真实 3DGS 遮挡、反光和图像质量会令检测率、估计误差变差，须另用真实场景做遮挡/噪声实验。
