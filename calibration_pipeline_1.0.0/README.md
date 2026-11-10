# 标定与 BEV 主干程序 1.0.0

这是可独立构建、安装和调用的流程程序：**场景融合 → 固定相机采集 → 镜头图像合成 → 内参估计 → 世界外参候选 → BEV 重投影与接缝评价**。主干源代码集中在 7 个 Python 文件；命令行与函数调用使用相同实现。既有项目内仍可由 `src/pipeline/run.py calibration-pipeline` 进入。

## 安装与运行

CPU 环境要求 Python 3.10+，依赖 NumPy、SciPy、OpenCV 和 plyfile。在解压目录安装：

```bash
python -m pip install .
calibration-pipeline --version
calibration-pipeline plan --config examples/cpu_demo/demo.json
calibration-pipeline run --config examples/cpu_demo/demo.json --execute
```

交付包的 `dist/` 同时提供 wheel，可以用 `python -m pip install dist/calibration_scene_pipeline-1.0.0-py3-none-any.whl` 安装。依赖需要预先安装或能从软件源获取；wheel 不含 CUDA 二进制、外部渲染器和大型模型。

省略 `--execute` 时只返回计划。顶层配置的相对路径以该 JSON 所在目录为基准；外部 BEV spec 内的 image 以 spec 目录为基准，capture manifest 内的 face 路径以 manifest 目录为基准；把配置作为 dict 传入时显式指定 `base_dir`。运行不会更改工作目录。输出阶段目录若已存在会报错，重新运行请更换 `output_dir`，或只选择尚未执行的下游阶段。失败日志留在 `output_dir/pipeline_runs/<run_id>/run.json`，部分产物不会被自动认作成功。

Python 调用：

```python
from calibration_pipeline import Pipeline, run_pipeline

pipeline = Pipeline("examples/cpu_demo/demo.json")
print(pipeline.plan())
result = pipeline.run(execute=True)
# 复用已经生成的内外参，只执行下游：
# result = pipeline.run("bev", execute=True)
# 等价便捷接口：run_pipeline(config, stage="all", execute=True, base_dir=...)
```

单阶段命令：

```bash
calibration-pipeline run --config my_config.json --stage prepare --execute
calibration-pipeline run --config my_config.json --stage capture,compose --execute
calibration-pipeline run --config my_config.json --stage intrinsics,extrinsics,bev --execute
```

## 模块与接口

| 文件 | 职责 | 主要函数 |
|---|---|---|
| `scene.py` | 读取 Gaussian PLY、TRS/SH 变换、可选地面裁剪、融合 | `prepare_scene(spec, out_dir, execute=False)` |
| `rendering.py` | 原生 Graphdeco 六面采集、针孔/鱼眼合成 | `capture(...)`、`compose(...)` |
| `calibration.py` | 真实 PNG 角点、平面内参拟合、条件 PnP | `calibrate_intrinsics(...)`、`calibrate_extrinsics(...)` |
| `bev.py` | 候选与镜头配对、地平面重投影、三种融合和接缝指标 | `prepare_bev_spec(...)`、`stitch_bev(...)` |
| `pipeline.py` | 配置解析、阶段衔接、运行记录、CLI | `Pipeline`、`run_pipeline`、`main` |
| `__init__.py` / `__main__.py` | 导入与模块命令薄入口 | `python -m calibration_pipeline` |

公共写入函数的 `execute` 都默认为 False；纯数学函数返回数组。算法模块之间直接传递配置、路径和结果，既有实验脚本不在运行依赖中。`tests/test_pipeline.py` 是单独一个集中测试文件，包含数值回归和实际 PNG 的 CPU 贯通夹具。

## 接入实际资产

`examples/full_pipeline.json` 是全流程配置模板，先补充其 `input/` 数据和外部 renderer 路径，再运行。它不绑定具体车辆、实验日期或服务器。配置 `stages` 指定整条流程；从实拍图像接入时可仅保留 intrinsics/extrinsics/bev，分别指定 images、intrinsics_dir、extrinsics_dir。已有世界坐标相机参数可通过 `bev.spec` 直接执行 BEV。

场景输入：

- `scene.placement` 使用 `{scene: {...}, objects: {id: {...}}}`。每项含 `ply`；对象变换使用 `location`、`rotation_euler`（XYZ 欧拉角，弧度）、正的均匀 `scale`。scene 保持 identity，物理尺度由使用者先核实。
- `scene.asset_root` 为 PLY 根目录；`scene.base_exclude_objects` 指定动态小板采集底图中应去掉的大板对象 ID。`scene/base.ply` 用于内参采集，`scene/fused.ply` 包含大板用于外参采集。小板用独立局部坐标 PLY，经 capture_spec 中每帧 SE3 放到世界。
- 支持标准 Graphdeco SH3 原始属性：xyz、f_dc_0..2、f_rest_0..44、opacity、scale_0..2、rot_0..3。字段按名字识别，缺失 nx/ny/nz 时填零只为格式兼容，不冒称测量法线。旋转同步处理 SH、四元数与法线，均匀尺度在 log 域更新。

采集规格包含 `cube_size`、`cube_face_padding_px`、`cameras`。每相机 `frame` 给出 R_w2c、camera_center_world、可选 tvec；动态 `views` 含 name、board_to_world、split。相机不动，板只按配置移动；有 rig_ground_plane 时强制平贴。镜头定义在同文件的 pinhole/fisheye 中，供后续 compose 使用。pinhole 含 width、height、K、distortion(Brown5)；fisheye 含 width、height、field_of_view_deg、distortion(theta4)。

内参规格：`geometry` 含 corners_x、corners_y、square_size（米）；`camera_ids` 明确相机；`models` 指明模型及 resolution_px。训练/留出集合由 views、image_splits 或 split_by_suffix 显式声明；原始 PNG 不缩放，实际分辨率与规格必须一致。目录为 images/<model>/<camera>/图片。内参拟合始终使用像素、已知格长和分组，不读取 renderer 的 K/D 或相机位姿。

外参图像默认使用 `big_board_fixed.png`；可在 `regions[camera].image` 指定其他文件名或路径，自动 BEV 会沿用这张实际观测图，并记录来源。

外参输入：`corners_world` 含公开实物大板角点 `{id, region, grid_row, grid_col, xyz_world_m}`（旧 `region/r01/c01` 格式 ID 也可解析网格身份），`regions` 明确相机到板区域的映射。它定义与车辆 BEV 一致的世界坐标，单位为米；不是隐藏的相机渲染真值。外参输出 `R_w2c`、`tvec` 将这个世界映射到相机。重复图案候选保留行偏移与旋转歧义；BEV 按全部对应角点 RMSE 选择有限、正深度候选，并绑定同一个内参种子，记录选择依据。

输出目录固定：scene、captures/{intrinsics,extrinsics}、images/{intrinsics,extrinsics}/{model}、intrinsics/{model}/{camera}、extrinsics/{model}/{camera}、bev/{model}。每次编排另保存独立 run.json，含配置、源码 SHA256、阶段状态、错误信息与运行环境；算法产物包含实际输入和图像校验和。

## GPU 后端与旧环境

真实 3DGS capture 需要外部 Graphdeco gaussian-splatting 仓库、与其匹配的 PyTorch/torchvision/CUDA 和 diff-gaussian-rasterization。`capture.repo` 指定仓库，包不改共享仓库或环境；运行 CPU 阶段不需要这些依赖。渲染采用原生逐面相机 Z 排序，分面接缝需另行检查。移动小板要求高阶 SH 恰为零，否则显式报错；静态物体允许完整 SH3。

GPU 环境与 CPU 环境不同时，在 capture 中配置 `render_python` 为 GPU Python 的路径；编排使用子进程调用同一 rendering.py 的函数，不生成第二套 worker 源码。渲染/场景模块保持 Python 3.7 语法兼容，但 wheel 的完整 CPU 安装要求 3.10+。旧 GPU 解释器及 CUDA 组合是否可执行，须在目标机器独立核实；本轮不重启已暂停的室外实验。

## 科学与验证边界

当前内参协议是固定相机、标定板仅在同一平面平移和转动。针孔未知内参存在不可辨识性，输出明确标记 DEGENERATE_PARALLEL_PLANE_ACQUISITION；多初值局部稳定不证明鱼眼全局唯一。程序运行成功不等于物理标定通过。

这条主干使用逐相机条件 PnP，尚未合并独立的共享物理角点 ID 多相机联合 BA。相机相对 rig 坐标不能默认当作车辆世界坐标。BEV 使用地平面假设，仅排除配置的车辆 footprint，没有逐相机深度遮挡推断；硬切、羽化和局部 GraphCut 对比不自动修正错误外参。RMSE 单位 px；灰度跳变单位是显示编码 0–255；空接缝指标为 null，不能当成零误差。

CPU 示例的图像由解析棋盘场景生成，只验证软件链路与退化状态保留，不能代表实拍、真实 3DGS 渲染或画质验收。真实 GPU 采集、完整实际资产重跑和人工画质评鉴尚未在本轮执行。验证记录见交付包 VALIDATION.md。

## 测试与构建

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m pip wheel . --no-deps --wheel-dir dist
```

测试从已安装包导入；可在仓库外执行该测试文件验证分发独立性。不要把原项目目录放入 PYTHONPATH 来掩盖缺失依赖。
