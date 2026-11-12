# 界面标定与 1.0.0 流程集成

> 2026-11-03 更新：1.0.0 已默认支持当前 GUI 场景一键生成输入/采集图片，不再默认 CPU 演示。最新操作、两类板几何、真实资产缺项及坐标/真值隔离约定见 [CURRENT_SCENE_CALIBRATION.md](CURRENT_SCENE_CALIBRATION.md)。下文配置文件模式保留为高级入口和历史验证记录。

## 使用入口

重新启动现有编辑器/ROS launch，展开 **标定 Calibration**。独立版与 ROS 版共用同一个面板：

1. 在 **标定算法** 下拉菜单选择算法。
2. 配置对应参数，实验名称可留空自动生成。
3. 点击 **开始标定**。后台子进程执行，不阻塞编辑器；显示运行阶段，运行时禁止重复启动，支持 **取消标定**。
4. 完成后显示参数、误差/诊断状态。点击 **导出结果 JSON** 可在当前浏览器保存结果。完整产物和日志仍保存在服务器的独立实验目录。

不会自动把估计 K/D/外参写回场景。尤其是 1.0.0 中允许非零 skew、并可能存在退化和候选歧义，不能未经审查直接替换当前相机。

## 选项一：OpenCV 棋盘格基线（当前场景）

选择全部或单个相机、明确选择一块标定板、采样数、seed、超采样倍数，以及 Gaussian 板的渲染后端。点击时冻结同 revision 的 SceneState；后续编辑不会改变正在进行的实验。ROS 有尚未回传确认的编辑时拒绝启动，避免使用混合状态。

复用已有 `make_calibration_dataset`、`findChessboardCornersSB`、`calibrateCamera` / `fisheye.calibrate`。输出 K、D、世界 optical 相机位姿估计、重投影 RMS、内参/畸变误差、平移和旋转误差。GT 只用于生成、角点对称消歧与误差评价，不用作内参优化初值。

虚拟棋盘格可直接使用。Gaussian 资产必须是完整矩形棋盘，并且行列数、格长、PLY 局部坐标/尺度对应；界面要求显式确认。项目现有环车条带/中央孔 PLY 不是 6×8 完整棋盘，不能靠勾选框就使它适用于矩形检测器。

产物：`scene.yaml`、`ground_truth.yaml`、`config.yaml`、`request.json`、`dataset/`、`baseline_result.json`、统一 `result.json`、`metrics.csv`、`progress.json` 和 `logs/`。

## 选项二：标定流程 1.0.0（配置文件）

直接调用 `/home/wzc/shangqi_project/calibration_pipeline_1.0.0/calibration_pipeline` 中交付的 `Pipeline`，不复制算法、不修改原始源码、不更换默认 Python 环境。按 JSON 的 stages 运行 prepare/capture/compose/intrinsics/extrinsics/bev，可在阶段框填写 `all` 或有序列表。缺少依赖/输入会明确报错，失败产物不当成成功。

**输入来自 JSON 指向的数据，不是把当前 GUI 场景自动转换为该流程的采集配置。** 保存的当前场景和 Ground Truth 只是实验上下文，不会被注入该流程的 solver_spec；原流程的 fixed-camera/单平面协议也不会被替换成原有 6DoF 随机采样。

默认 JSON：

```text
/home/wzc/shangqi_project/calibration_pipeline_1.0.0/examples/cpu_demo/demo.json
```

它使用包内已有 CPU 解析棋盘图片，执行 compose → intrinsics → extrinsics → bev，同时验证 pinhole 和 fisheye；用于先确认 GUI/流程集成，不代表当前真实场景标定。要处理自己的数据，将输入框改为自己的完整 JSON。

输入相对路径始终相对**原 JSON 所在目录**解析；不把配置移动后错误地重定向到 experiments。输出目录则统一隔离到新实验的 `pipeline/`，不会覆盖 JSON 内原先指定的输出位置、原始图片、配置或之前的实验。只跑下游阶段时，要在配置里明确填写先前产物的 images/intrinsics_dir/extrinsics_dir。

### 原始计算保持不变

- 内参：SB 图像角点、固定相机平面 XY/Yaw 模型、SciPy least_squares 三初值，保留训练/留出划分、skew、Brown5/theta4、病态/退化证据。这里的三个优化初值与 GUI OpenCV 模式的随机采样 seed 不是同一概念。
- 外参：公开世界角点与明确 camera→region 映射，逐相机条件 PnP，保留行偏移、旋转、内参种子和角点身份候选；不是多相机联合 BA。
- BEV：沿用原流程的候选配对和地平面重投影，输出 hard/feather/GraphCut、接缝指标与 comparison.png。
- `DEGENERATE_PARALLEL_PLANE_ACQUISITION`、`MULTISTART_UNSTABLE`、`UNRESOLVED_INTRINSICS_AND_CORNER_IDS` 等不会被改成“标定成功”。作业 `complete` 仅表示代码执行完毕，`calibration_accepted` 始终为 false。

真实 GPU 六面采集仍要求原 Graphdeco repo、匹配的 rasterizer/解释器和真实 capture/solver/public corners 配置；本次仅实际执行 CPU demo，未声称把现有 gsplat 后端冒充原 Graphdeco capture，也未声称完成真实条带板的角点身份/标定。

## 输出与复现

```text
experiments/<id>/
  scene.yaml / ground_truth.yaml / config.yaml / request.json
  effective_pipeline.json / pipeline_plan.json    # 1.0.0 模式
  result.json / metrics.csv / progress.json
  logs/
    calibration.log / run.log / runtime.json / assets.json
    source.zip / imported_pipeline_source.zip
  pipeline/
    images/ intrinsics/ extrinsics/ bev/ pipeline_runs/
```

界面显示 K/D、各阶段原始状态和 RMS、条件外参候选数量及一个按全部对应点 RMSE 排序的预览、BEV 覆盖率和首张对比图。全部候选与逐图诊断在原流程 JSON 中；预览不消除物理角点歧义，覆盖率不是精度指标。

## 实测记录（2026-10-26）

真实浏览器已点击两种算法运行：

- 1.0.0 CPU demo：`experiments/calib_gui_20261026_222637_90489c/`。内参训练 RMS pinhole≈0.34739 px、fisheye≈0.24235 px；仍分别标记平面退化、多初值不稳定。两种 BEV 对比图已生成，覆盖率 100% 仅反映这个单相机演示夹具。
- 当前示例场景四路 OpenCV：`experiments/calib_gui_20261026_223034_99bae2/`。完整参数/误差在对应 result.json。
- 交付 wheel 在独立临时目录离线安装，从 `/tmp` 运行原版 **38 项测试全部通过**；不向现有环境安装/升级依赖。
- 新增集成测试覆盖实际子进程标定、原版 1.0.0 全链路、场景快照/不自动回写、非法输入、失败、取消、拒覆盖、Viser 菜单/开始按钮与结果导出。
- 最终含 CUDA 的项目完整回归 **57 passed**；ROS Humble 六包构建和导入通过。
- ROS 域 91 的临时实例也通过浏览器实际点击“开始标定”，使用 revision 0 跑完四路，RMS 分别为 front 0.10500、left 0.10747、rear 0.09889、right 0.09647 px。结果目录 `experiments/calib_gui_20261026_223743_9e94af/`；点击“导出结果 JSON”后已验证浏览器出现约 42.4 KB 下载链接。
- 七个原始 runtime Python 源文件逐一核对 `SOURCE_MANIFEST.json`，SHA-256 全部一致。

代码入口：`scene_editor/calibration_ui.py`（界面）、`scene_editor/calibration_jobs.py`（任务与适配）、`tools/run_gui_calibration.py`（隔离 worker）。

## 多倾角鱼眼内参

新增独立“多倾角采集：鱼眼内参”入口，使用逐图片板姿态模型并限定小板检测区域。操作、留出评估、输出和验证边界见 [MULTITILT_INTRINSICS.md](MULTITILT_INTRINSICS.md)。原 1.0.0 入口保留；新入口不生成外参/BEV，不自动应用参数。
