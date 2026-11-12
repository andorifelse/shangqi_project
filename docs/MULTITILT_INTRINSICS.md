# 多倾角采集：鱼眼内参与编码板外参

本入口 `fisheye_multitilt` 与原“标定流程 1.0.0”独立。未接入 rc9 固定法向路线。选择当前场景中的鱼眼相机及完整内参小板，由 gsplat/CPU 当前场景渲染真实 RGB，再通过 OpenCV SB 检测和 KB4 独立板姿态拟合估计内参。

## UI 操作

重启现有 UI/ROS 编辑器，展开“标定 Calibration”，选择“多倾角采集：鱼眼内参”。选择相机（或全部）、内参运动小板、后端及超采样。建议先使用 24 张训练、4 张留出、最大倾角 40°、距离 1.0–2.4 m、gsplat、两倍超采样。最少 12 张训练、3 张留出；大板和区域映射不属于此入口。点击“保存本场景标定设置”再 Save Scene 可保存算法及设置。

点击“开始标定”冻结场景并在独立子进程运行。界面显示采集进度，支持取消和结果导出。结果不会自动修改当前相机参数。勾选“内参通过后继续编码板外参”（GUI 默认开启）可连续完成两阶段；取消勾选时仅输出内参。`CHECKED` 是对应阶段的数值检查状态，完整结果的 `calibration_accepted` 仅在所有相机的内参、外参均通过时为 true；其范围限当前合成场景观测，不代表真实硬件验收。此入口不执行 BEV。

编码板几何可留空，程序从当前资产路径的父目录查找 `几何与来源说明.json`，按文件路径精确匹配场景中的编码板。也可指定 JSON，格式为 `marker_dictionary`（例如 `DICT_4X4_1000`）及 `markers` 列表；每项包含 `id`、相对 JSON 的 `file` 和按 ArUco 解码顺序排列的四个 `corners_world_m`。该场景包在导入时将世界坐标烘焙进 PLY，因此这些点是导入资产的局部坐标；程序再应用当前板和资产的位姿、缩放。缺少至少三个已知 ID 时启动前报错；不会猜测角点或使用相机真值编号。

CLI 复现同一快照与设置并重新采集（自动创建新实验，原日志保留）：

```bash
.venv-ros/bin/python tools/run_fisheye_calibration.py \
  --from-experiment experiments/calib_gui_20261112_113642_45cb54
```

也支持 `--scene /absolute/scene.yaml --small-board board_001`，以及 `--intrinsics-only`、`--views`、`--holdout`、`--supersample`、`--marker-geometry` 等参数。旧的直接请求未指定 `run_extrinsics` 时仍按仅内参执行。

本机 ROS 安装为 symlink/develop 链接，已核实 `source /opt/ros/humble/setup.bash` 与 `source avm_sim_ws/install/setup.bash` 后能从 build 链接导入新增模块及 UI 入口，无需更换环境。其他机器如使用复制安装，需按原方式重新构建/安装 scene_editor。运行中的 UI 保留旧模块，需要重启；本次未擅自终止现有服务。

## 采集和求解边界

- 相机固定。小板逐张改变倾角、距离、画面位置及面内旋转，筛选完整板边界、格子大小、遮挡和重复姿态，保留静态高斯与射线缓存。
- 小板 ROI 来自采集侧几何，ROI 外的真实背景像素只在检测副本中屏蔽；保存的 PNG 是完整原图，原场景所有其他板、环境及车辆保留。像素检测结果恢复到全图坐标；不使用 GT 投影角点替代或重新排序观测。
- 求解器仅接受公开板局部角点、分辨率、图片名、训练/留出划分、实际检测像素。多倾角图片不送入固定平面 1.0.0 模型。KB4 拟合每张图的独立姿态，skew 固定零，其他镜头参数由多初值优化估计；初值仅由图像尺寸及零畸变生成。
- 留出帧不参加 K/D 拟合。评估固定 K/D，仅用留出帧 4/5 角点估计该帧姿态，在未参与位姿拟合的 1/5 角点上评分。这是留出角点误差，不是无需位姿估计的全图预测。
- 检查训练 RMS、留出角点误差、估计板法向和距离差异、多初值焦距差异、观测范围内映射可逆性；不以 GT 选择参数。不支持未观测视角的全视场精度保证。
- 所有内参采集结束后导出 downstream_scene.json：仅移除本次运动小板，其他板不变；原 YAML、GUI 场景和相机参数均不改。启用外参后继续渲染该固定场景。
- 检测阶段检查网格相邻步长变化，拒绝将外边缘当成内角点的折线网格，并尝试其他检测模式。内参用多候选 PnP 重置逐图位姿，再做像素域稀疏联合优化；最终仍以普通 RMS 和原 1 px/1.5 px 阈值检查。留出帧的候选排名仅使用参与拟合的角点。
- 外参使用实际 RGB 解码的 ID、公开三维点和已通过的内参。鱼眼外围可用估计内参生成重叠透视图辅助解码，再变换回原图像素。多候选 PnP 检查正深度；每个编码板留出一个角点验证，固定 K/D 后重投影 RMS ≤1 px、留出 RMS ≤1.5 px 且至少三个编码板才通过。全部已知 ID 的观测均参与评分，不按 GT 筛选。

## 输出

`experiments/<id>/` 保留原有 scene.yaml / ground_truth.yaml / request.json / config.yaml 和运行身份、源码、资产哈希。额外包含：

- `capture/<camera>/view_*.png`：实际全分辨率 PNG。
- `capture/rejected/`：检测失败的图片。
- `capture/manifest.json`：采集划分、图片哈希、尝试和耗时。
- `capture/render_truth.json`：所有尝试的板姿态、距离、法线和 ROI，仅审计使用。
- `capture/timing.json`：本次采集/求解总耗时与计数。
- `solver_inputs/<camera>.json`：无相机/板姿态真值的公开观测。
- `multitilt/<camera>.json`：K/D、留出误差、候选和拒绝原因。
- `multitilt/<camera>_evaluation.json`：求解结束后独立 GT 对比。
- `multitilt/downstream_scene.json`：移除内参板的后续场景。
- `extrinsics/<camera>.png`：固定场景实际图片。
- `solver_inputs/<camera>_extrinsics.json`：估计 K/D、公开编码几何与实测编码角点；无相机真值。
- `extrinsics/<camera>.json`：R_w2c、tvec、T_world_from_camera_optical、训练与留出 RMS。坐标为 scene_world，光学轴 X右/Y下/Z前。
- `extrinsics/<camera>_evaluation.json`：求解后独立评估的平移、旋转真值误差。
- `calibrated_scene.yaml`、`calibrated_parameters.json`：全部请求相机内外参均通过后导出，可手工加载 YAML 应用估计值；不会覆盖原场景。对于已有通过实验，可运行 `tools/run_fisheye_calibration.py --export-experiment <目录>` 导出至该实验的 `exports/`，保留原结果与源码快照。

## 2026-11-12 失败修复与内外参验证

原实验 `calib_gui_20261112_113642_45cb54` 的左路训练 RMS 2.0443 px，留出最大 25.4908 px；错误包含逐图姿态局部解和 SB 将外边缘当成内角点。该实验设置只有内参，因此没有运行外参。修复增加网格检测重试、逐图多候选姿态与联合优化，并接入公开已知 ID 编码板外参，验收阈值保持不变。

使用同一原始场景快照、40 张训练 + 3 张留出、seed 20261020、两倍超采样、gsplat 完整重采集：`experiments/calib_repaired_20261112_144714`。172 张内参图全部采集完成，四路内外参均通过，`calibration_accepted=true`。真值仅在估计后评估，未参与初始化、筛选或验收。

| 相机 | 内参 RMS px | 内参留出最大 px | 外参 RMS px | 外参留出 px | 事后位置误差 mm | 事后旋转误差 ° |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| front | 0.3391 | 0.1351 | 0.3745 | 0.5979 | 2.85 | 0.0816 |
| left | 0.2894 | 0.1343 | 0.7197 | 1.1691 | 4.03 | 0.1167 |
| rear | 0.2740 | 0.3094 | 0.5346 | 0.5832 | 0.89 | 0.0764 |
| right | 0.2897 | 0.2733 | 0.5211 | 0.7212 | 11.68 | 0.1064 |

内参/外参/GUI 任务针对性测试通过。扩大原场景流程测试时，复现下文已记录的旧 BEV 掩码测试失败；不在本次路径内。

## 本机验证

使用原失败实验 `calib_gui_20261110_201730_db9034/request.json` 的场景快照，四路各 24 张训练、4 张留出，1280×960、两倍超采样；包含原高斯小板、地面大板及 24 个编码板。独立 UI 后台实验目录见 `experiments/multitilt_four_final_20261111/`。数值检查范围限于本场景观测射线，不代表外参、BEV、真实硬件或任意鱼眼标定已通过。

最终版四路 UI 后台验证约 **109.7 秒**，全部 CHECKED，112 张图片均保留。

| 相机 | 训练 RMS (px) | 留出角点最大 RMS (px) | 焦距相对真值误差（事后） |
| --- | ---: | ---: | ---: |
| front | 0.1319 | 0.1040 | 0.2855% |
| left | 0.2414 | 0.2625 | 0.4367% |
| rear | 0.5899 | 0.1816 | 0.0348% |
| right | 0.3118 | 0.3558 | 0.2591% |

针对性回归及旧入口回归通过；另单独复现既有 `test_direct_mask_preserves_black_pixels_and_rejects_modified_mask` 失败（旧 BEV 有效像素掩码），未修改该模块。该失败不属于本新入口的内参路径。

## 标定后的鸟瞰拼接

四路 front/rear/left/right 内外参均通过后，当前流程自动生成 `bev/stitched.png` 并在结果面板预览。单相机标定仍可执行，BEV 将明确标记 SKIPPED。拼接读取 `extrinsics/<camera>.png` 原始固定场景图片并校验哈希，投影只使用估计的 K/D 与外参；不会用估计参数重新渲染传感器图片。采集有效域由原采集模型提供，黑色像素本身不判为无效。

输出包括 `bev/metrics.json`（来源图片哈希、覆盖率、重叠率）、各相机地面投影 `<camera>_ground.png`、有效掩码、`coverage.png` 与 `overlap_count.png`。采用地平面逆映射与加权融合；覆盖率不是精度指标，车辆/立体物体会拉伸，未做深度遮挡或接缝优化。原失败日志和 GUI 相机参数保持不变。

已有通过实验可单独补生成，原 `result.json` 不覆盖：

```bash
.venv-ros/bin/python tools/run_fisheye_calibration.py \
  --bev-experiment experiments/calib_repaired_20261112_144714
```

该实验已实际生成 512×512 鸟瞰图，覆盖率 97.4716%，重叠覆盖率 55.8987%。重启编辑器后新任务将自动生成并预览；旧任务的补生成结果位于 `bev/metrics.json`，不会改写原任务结果。

标定结果现同时显示两个独立预览：“真值参数鸟瞰图（同一组采集图片）”与“标定参数鸟瞰图”。现有实时鸟瞰图继续使用当前场景参数，不被标定结果替换。实验输出 `truth_stitched.png`（原场景真值）、`stitched.png`（估计参数）以及 `comparison.png`（左真值、右标定）；两路使用相同 RGB、画幅和融合方式，仅投影内外参不同。真值不进入标定求解。
