# 当前场景 → 1.0.0 一键标定（2026-11-03）

## 操作

重新启动编辑器，展开 **标定 Calibration**，选择 **标定流程 1.0.0**。输入来源默认 **当前 GUI 场景（一键采集）**，不再预填演示 JSON。高级手工 JSON 和 CPU 演示仍是另外两个选项。

1. 选择当前相机（全部或单个）。
2. 分别选择 **内参运动小板** 和 **外参固定大板**，必须是不同对象。小板必须是完整矩形，局部棋盘在 board XY 面，世界姿态允许非零 Roll/Pitch。环车条带只能作为多 region 外参板。
3. 在 **相机 → 大板 region** 明确选择每路对应区域，不从相机名字猜测。
4. 设置训练数、留出数、seed、后端、超采样及板局部 XY/面内旋转范围。默认自动在相机视野对应的初始板平面上采样；也可选择在当前小板附近沿局部 XY 采样。板允许倾斜或竖直；任何模式均固定相机和板平面法线，世界 Z 与欧拉角可能随面内运动变化。
5. 点击 **保存本场景标定设置**，再在 Scene files 中 **Save Scene**。配置、板角色、角点和相机区域映射保存在同一 SceneState/YAML 中。
6. 点击 **开始标定**。后台冻结点击时 revision，生成实际 PNG，运行原版求解器和 BEV。之后修改 GUI 不影响本次实验；ROS 有待确认编辑时拒绝开始。

## 板几何：不需要手写流程 JSON，但必须知道物理定义

展开 **标定板公开几何 / Regions**：选择板，填写 region、内角点行列数、真实格长、坐标系、首角点 XYZ、网格面内角度及来源，确认实物对应后 **保存/替换这个 region**。每个 region 可分别添加，自动生成稳定的 `region/rNN/cNN` 角点 ID。中央孔两侧不能拼成一个矩形 region。

- `board`：米制逻辑板坐标，原点/轴向按项目定义，随 board.pose 移动，不再乘 asset_scale。
- `asset`：PLY 原始坐标/原始单位，系统先乘 asset_scale，再应用 asset_pose，最后应用 board.pose。
- 面板网格生成器生成所选坐标系 XY 网格；原始 XZ 资产建议先以 asset_pose 对齐到 board XY 后填写 board 坐标。任意原始朝向可高级导入显式角点 JSON：`{frame, verified, source, corners:[{id,region,grid_row,grid_col,xyz_m}]}`。
- 虚拟板由明确的 rows/columns/square_size 自动得到真实合成几何；**Gaussian 板没有公开定义时绝不沿用默认 6×8**。Ground Truth 新导出也会把其宽高/虚拟角点标记为空及 `MISSING_OR_UNVERIFIED`，而不是伪造测量真值。

### 你的环车 PLY 实际检查

资产 SHA-256 与包说明一致，684,032 个 Gaussian，双层薄面在 Y≈0.028752/0.032752，面内为 X/Z。按说明中的 K=16 规则采样，可验证 38×44 格布局；缺口列 12–25、行 10–33，即中央 14×24 格，绝不是 6×8 实心板。模型中格距约 **0.163758 原始单位**，不自动声称为米。

详细证据：`outputs/ring_board_inspection_20261103.json`，可复跑 `tools/inspect_board_asset.py`。

要使用你的这份资产完成真实标定，还需要你确认：

1. 实际米制格长，或 PLY 原始单位到米的尺度及来源。
2. 内参小板资产（也可选明确的虚拟矩形板），内角点行列、格长、局部原点/方向。
3. 环车每个有效矩形子区域的首角点坐标、行列数、行列方向和物理 ID 定义；不要跨中央孔。
4. 每个相机对应哪个 region。

这些数据可以直接在 GUI 填写保存。不具备时系统会具体报缺项，原场景资产不会被自动“校正”或猜测。

## 两套系统的适配结论

| 项目 | 实现 |
| --- | --- |
| 原 1.0.0 采集 | Graphdeco 六面采集 → compose；仍保留在高级 JSON 路径 |
| 当前场景采集 | 已验证 SensorPipeline/gsplat 直出 pinhole/fisheye PNG；明确跳过 prepare/capture/compose，不冒充六面图片 |
| 采集协议 | 相机中心/外参完全固定；同一初始板平面内局部 XY 平移/绕法线旋转，支持倾斜或竖直板；分相机独立观测序列，不冒称同步联合 BA |
| 资产合成 | 环境 `T_world_from_gs(scale*p)`；板/车 `T_world_object @ T_object_asset(scale*p)`；保留真实 Gaussian 深度遮挡 |
| 标定板生命周期 | 内参采集保留全部板，仅移动所选内参板；所有相机内参采集完成后，仅从实验采集场景移除内参板。外参及其图片生成的 BEV 保留其他板的位姿、几何和资产；原始 YAML/UI 场景不变 |
| 坐标 | 世界/车辆 X前Y左Z上；optical X右Y下Z前；`T_world_camera` 逆得到 w2c；欧拉角弧度，GUI 度；完整数值变换写入采集记录 |
| PNG | uint8 RGB → OpenCV BGR 编码，原分辨率；超采样仅在渲染端做面积积分，求解器不缩放图片 |
| 光心 | OpenCV 整数像素中心；gsplat 的 +0.5 仅在内部 rasterizer K，不改保存 K |
| 混合尺寸 | 按 model/width/height 分组运行原 1.0.0，再按同模型合并 BEV；不偷改相机参数统一分辨率 |
| 真值隔离 | K/D/相机位姿只在 Ground Truth、capture/render_truth、完整场景快照中；solver_inputs 只有 PNG 路径、分组、公开棋盘尺寸/世界角点与 region 映射 |
| BEV 参考 | 显式采用车辆 XY/Yaw、world ground_z 的水平参考系。真实滚转/俯仰仍用于相机采集；其 BEV frame 与 live AVM 的 base XY 平面相交定义差别有记录，不隐式混用 |

gsplat 保留现有 DC-only/expected depth 特性，不声称和 Graphdeco 含高阶 SH、六面合成像素等价。GLB/GLTF 车辆仍仅显示，不会被伪称能产生 Gaussian 遮挡。有效像素从实际镜头射线生成独立 mask；黑色 RGB 不作为无效像素判断。

原估计代码保持不变；仅对 `calibration_pipeline/bev.py` 增加向后兼容的可选 `source_valid_mask`（带 hash 检查），避免直出任意 K/D 镜头被旧的中心内切圆假设裁错。未提供该字段时原功能不变，原 38 项测试已复跑。

## 实验记录与验收

```text
experiments/<id>/
  scene.yaml / ground_truth.yaml / request.json / config.yaml
  coordinate_contract.json / generated_config.json
  capture/
    render_truth.json / manifest.json / sampling_attempts.json
    board_poses.csv / valid/ / rejected/
  solver_inputs/             # 自动生成，不含相机真值
  pipeline/images/          # 实际采集 PNG
  pipeline/groups/ / intrinsics/ / extrinsics/ / bev/
  ground_truth_comparison.json / scene_pipeline_result.json
  result.json / metrics.csv / progress.json / logs/
```

进度按采集张数和求解阶段显示，支持取消；已采图片/位姿/拒绝原因增量保存。失败不会覆盖旧实验或相机参数。候选依然由原算法的**观测误差**选择，不用 GT 挑“最接近”的候选；GT 仅在求解完成后逐初值/逐候选对比。

针孔固定平面采集的不可辨识性、鱼眼多初值局部稳定性边界、重复图案行偏移/旋转歧义均保留。`complete` 仅为工程执行完成，`calibration_accepted=false`。低重投影误差不代表位姿正确。

## 实测

- 项目最终完整 CPU/CUDA 回归：70 passed；原流程源码测试：38 passed；ROS Humble 六包构建通过。
- 已知几何的 Gaussian 矩形板夹具分别跑通 pinhole/fisheye：gsplat → PNG → 原版 1.0.0 内外参 → BEV → GT 比较。
- 可检查的持久实验：`experiments/current_scene_gsplat_20261103/`。这里训练 6 张、留出 2 张、固定大板 1 张。该已知几何夹具不是你的条带 PLY，不把它的结果当你的资产精度。
- 该例针孔内参 RMS≈0.0434 px、外参子集 RMS≈0.0245 px，却仍有约 4.45 m / 179.84° 的 GT 差异（观测选择的候选）；原状态保持平面退化/角点身份未决，说明不能以低 RMS 掩盖错误或不可辨识性。
- 验证场景：`outputs/current_scene_fixture_20261103/scene.yaml`。可用 `tools/run_demo.py --scene ... --backend gsplat`，选择当前场景输入直接操作。原真实场景 YAML 未修改。
- 真实浏览器当前场景模式点击“开始标定”也完成了 6 张训练、2 张留出、1 张外参的高斯板采集，实验目录 `experiments/calib_gui_20261103_184717_a3cf7e/`。配置自动生成，原始资产 hash 复核未变；显示 K/D、候选、BEV 和明显的 GT 差异，没有改写诊断状态。
- 独立 ROS 域 93 实测通过服务保存板角色和映射，同 revision 1 的 SceneState/Ground Truth 保持一致，板公开角点未丢失；记录 `outputs/current_scene_fixture_20261103/ros_profile_validation.json`。测试实例会在验证结束后关闭。

## 无 JSON 的命令行复现（与 GUI 共用后台）

```bash
CUDA_HOME=/home/wzc/cuda-11.8 TORCH_CUDA_ARCH_LIST=8.9 \
.venv-ros/bin/python tools/run_scene_calibration.py \
  --scene outputs/current_scene_fixture_20261103/scene.yaml \
  --small-board board_small --large-board board_large \
  --camera-region front=board_large --views 6 --holdout 2 \
  --sampling around_pose --spread-x .17 --spread-y .12 --yaw-range-deg 15 \
  --experiment-id current_scene_my_run
```

请使用新的实验名，已有目录不会覆盖。以上参数仅用于已知几何测试场景，不能直接当作你的环车资产角点定义。

## 2026-11-10 采集耗时优化

- 相机射线按镜头参数和分辨率缓存，分块计算降低鱼眼逆投影峰值内存；全高斯场景跳过无用的平面叠加。
- 不变物体的世界坐标高斯缓存到 CPU/GPU，小板移动只更新其 GPU 数据切片；Ftheta 的 CPU 后备数据仅在需要时生成。
- 每相机先渲染不含内参板的静态深度用于保守遮挡预筛选。候选在渲染前检查完整板边界、格子像素间距和倾斜程度；优先围绕成功姿态采样，保留全局探索和观测多样性，避免重复失败邻域。
- 原相机参数、图片分辨率、采样张数、超采样设置、其他板和 1.0.0 求解器保持用户配置；不通过减少输入或替换高斯板来加速。
- `capture/timing.json` 记录静态深度准备、预筛选、渲染和角点检测时间；`sampling_attempts.json` 记录逐次耗时，进度区分渲染和角点检测。

全相机在加载 GPU 场景前先做几何预检；未找到候选时在 `capture/preflight.json` 中说明是哪一路及拒绝原因，避免先采完其他相机。视野搜索覆盖有效画面，并对失败邻域和过于重复的观测进行避让。

### 真实场景计时（2026-11-10）

使用 `calib_gui_20261110_191859_3dfdb4/request.json` 场景快照，经 `CalibrationJobs` 的同一 UI 后台路径验证 front：40 张训练 + 2 张留出，1280×960、supersample=2，保留全部其他板。总耗时约 75.8 秒，作业 complete；43 次渲染尝试成功 42 张。采集阶段约 57.7 秒（含静态深度准备约 9.3 秒），内外参/BEV 也已执行。复现产物：`/home/wzc/tmp_pip/shangqi-speed-ui-front42-20261110-rzkkb6ek/calib_gui_20261110_195022_a646d0/`，其中 `capture/timing.json`、`result.json`、原始 PNG 和求解器输出可检查。另一个直接调用的计时未限制 BLAS 线程，耗时 153 秒，不作为 UI 性能数字。

这是单路 front 验证，不冒称完整四路已通过。该快照的 left 在当前内参板平面中未通过成像尺寸预检，需要调整板初始平面/位置或分路采集；保持原相机和板平面的情况下，不自动伪造可见观测。原有并行平面退化、诊断状态和 calibration_accepted=false 保留。

## 2026-11-10 逐相机自动布板

自动采样（`sampling: visible_plane`）不再要求四路共用同一个世界板平面。若原平面在某路不可见或格子过小，采集适配器会将内参小板布置在该相机前方，并结合地面与静态深度检查选取平面。每路开始后平面固定，仍按 1.0.0 的面内 XY/旋转模型采集；每路单独求解，相机和其他板不动。当前小板附近模式（`around_pose`）继续锁定用户配置的平面。实际逐路板位姿、法线和来源保存在 `capture/preflight.json`、`capture/render_truth.json` 和 `coordinate_contract.json`，不进入求解器输入。上述 left 原平面预检限制现由自动模式的逐路布板处理。

逐相机布板真实 GPU 验证：使用失败任务 `calib_gui_20261110_195542_83eeb2` 的同一快照和四路 40+2 张设置，独立验证目录 `/home/wzc/tmp_pip/shangqi-four-auto-plane-20261110-p_l7hfev/calib_gui_20261110_200130_0cabd0/`。四路均采满并完成内参求解：front 43 次、left 59 次、rear 55 次、right 42 次渲染尝试，各成功 42 张；采集约 273 秒。front/left/right 外参检测并产生候选，rear 检测成功但 `NO_VALID_CONDITIONAL_POSES`，因此完整作业约 351 秒后按失败报告，BEV 不接受无效候选；不能将此次记录描述为完整标定通过。逐路布板的采集问题已解决，rear 外参有效性是后续独立问题。26 项相关 CPU 回归通过，真实 GPU 四路采集证据保留。
