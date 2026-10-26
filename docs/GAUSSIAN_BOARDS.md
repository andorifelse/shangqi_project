# 高斯重建标定板

高斯 PLY 标定板复用已有 Board 对象，不是合并到环境里的静态点云。支持选择/删除、XYZ 和 Roll/Pitch/Yaw 数值编辑、Translate/Rotate Gizmo、独立缩放、PLY 局部对齐、场景保存加载、Ground Truth、ROS SceneCommand/TF 和相机/AVM 渲染。

## 在编辑器中导入

1. 启动现有编辑器。大场景/真实 PLY 使用 `backend:=gsplat`，不要使用默认 CPU 渲染几十万个 Gaussian。
2. 点击 **Add Board**，或在 Selected 中选一个已有 `board_*` 对象。
3. 展开 **Board parameters**，在 **Board Gaussian PLY path** 填服务器本地 PLY 完整路径，点击 **Apply board asset**。原虚拟棋盘格随之被 PLY 替代，不会同时叠加。
4. 在 **Transform** 中编辑 XYZ/RPY，或拖动 Gizmo。XYZ 单位米，GUI 角度单位度。PLY、TF 和相机图像遵循同一个 `board.pose`。
5. **Board asset scale** 控制重建模型的尺寸，**Asset X/Y/Z/Roll/Pitch/Yaw** 校正模型原始轴向与中心，修改后再次点击 **Apply board asset**。
6. **Use virtual board** 切回原虚拟棋盘格；**Delete selected** 删除整个对象。通过 Scene files 的 **Save Scene** 保存导入路径、缩放、局部对齐和位姿。

PLY 必须有完整 3DGS 的位置、SH DC、opacity、scale、rotation 属性；普通 RGB 点云不能冒充 Gaussian PLY。

## 用命令生成新场景

不覆盖输入场景和已有输出：

```bash
.venv-ros/bin/python tools/add_gaussian_board.py \
  --scene SCENE_YAML/scene_v5.yaml \
  --ply assets/vehicle_3dgs_calibration_board_2026-09-19/calibration_board_checker_strips_hole_k16_opaque2.ply \
  --name board_gaussian --output outputs/scene_v5_gaussian_board.yaml \
  --scale 1 --asset-pose 0 0 0.030752 -1.5707963267948966 0 0

# 单独启动；或将同一 YAML 用于现有 ROS launch 的 scene:= 参数
.venv-ros/bin/python tools/run_demo.py \
  --scene outputs/scene_v5_gaussian_board.yaml --backend gsplat
```

这份现有 PLY 是 **环车棋盘条带，中间有孔**，不是默认 6×8 的单块完整棋盘。原始范围约 X=6.21、Z=7.20，薄面位于 Y≈0.030752；示例仅将它旋到世界 XY 并把薄面居中到局部 Z=0，保留原始尺度，不假装完成真实尺寸标定。可在编辑器中继续调整。

## 位姿与真实角点

```text
PLY 坐标 → 乘 asset_scale → asset_pose → board.pose → 世界坐标
```

`board.pose` 是整个对象的 6DoF；`asset_pose` 只负责模型原始坐标到棋盘格逻辑坐标的对齐。逻辑棋盘格中心为原点，位于 XY 平面，法向 +Z。YAML/CLI 的 RPY 使用弧度，GUI 使用度。保存 YAML 时，PLY 路径相对 YAML 目录；移动 YAML 后需保持相对路径有效。

`rows/columns/square_size` 是物理角点描述，不会重画或缩放 PLY。调整 PLY scale 后须相应检查实际格长。未经对齐和真实尺寸确认，不能把合成规则角点当成重建资产的 Ground Truth。

## 标定数据集

单块完整棋盘在尺寸、行列数、方向正确配置后可沿用生成流程，并加 `--board-backend gsplat`。若选中的第一块标定板是 Gaussian，生成器会真实渲染 PLY，不会用理想棋盘替代；该 baseline 仍不包含环境/车辆遮挡。当前 OpenCV 检测器只支持完整矩形棋盘，不支持此环车条带/中心孔的联合角点身份识别，因此不能承诺现有条带 PLY 自动跑通矩形棋盘标定。原虚拟板的 baseline 不受影响。

## 回归验证

```bash
.venv-ros/bin/python -m pytest -q tests/test_gaussian_board.py
bash tools/build_ros.sh
```

新增测试覆盖：向后兼容、原子状态修改、相对资产路径、Ground Truth 和 TF、6DoF/尺度及协方差合成、缓存/删除、传感器成像变化、不叠加虚拟棋盘，以及真实 Viser API 的导入、拖动和切换。

真实资产出图验证：

```bash
CUDA_HOME=/home/wzc/cuda-11.8 TORCH_CUDA_ARCH_LIST=8.9 \
  .venv-ros/bin/python tools/render_gaussian_board_smoke.py \
    --scene outputs/scene_v5_gaussian_board.yaml \
    --output outputs/gaussian_board_my_smoke --backend gsplat
```

该命令保存四路移动前/后 PNG、AVM、Ground Truth 和移动后的场景，检查静止帧复用几何、移动后至少一路图像发生变化。已使用现有 684,032 个 Gaussian 的条带 PLY 实跑，结果位于 `outputs/gaussian_board_smoke_20261026/`；左相机 27,500 像素、右相机 22,258 像素变化。前后相机在这一具体姿态下不受板运动影响，不要求每路强行出现变化。

2026-10-26 最终验证：含 CUDA 的完整回归 **52 passed**；Humble 六包重新构建/导入通过。独立 ROS 域 87 下启动此真实场景，收到四路 Image 和 AVM，通过 SceneCommand 移动/旋转板，使 revision 从 0 更新为 1；对应 TF、Ground Truth 和同 revision 相机图像均匹配。验证记录保存在同目录的 `ros_validation.json`。测试启动的 ROS 进程已停止，使用时自行启动正常实例。

已有 ROS 环境下可以直接运行（先 source Humble、`.venv-ros` 和工作区环境）：

```bash
ros2 launch avm_bringup avm_sim.launch.py \
  scene:=/home/wzc/shangqi_project/outputs/scene_v5_gaussian_board.yaml \
  backend:=gsplat render_hz:=1.0
```

保留原场景的两块虚拟板，并额外加入 `board_gaussian`；如不需要原虚拟板，可在编辑器中选择后删除再另存。修改 PLY 路径、缩放或局部对齐会重新加载几何；只拖动位置/旋转不会重新读 PLY，但仍需重新合成并上传 Gaussian，不能据此承诺实时帧率。
