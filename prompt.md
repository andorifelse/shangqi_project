你现在是这个项目的主开发工程师，请直接负责项目的架构设计、代码实现、运行验证、调试和文档编写。

我已经上传：

1. 项目需求文档
2. 一个目标交互效果的参考视频demo\_example

请先完整阅读需求文档，并分析参考视频的交互方式。

注意：不要一次性实现需求文档中的所有功能。

当前只实现第一阶段 MVP，暂时不要实现日志管理、用户管理、报告生成、AI 场景生成、场景版本管理、自动标定等外围功能。

# 一、第一阶段目标

实现一个基于 ROS 2 的环视相机标定仿真原型系统。

需要具备：

1. 3DGS 场景导入
2. 可自由旋转、平移、缩放的 3D 上帝视角
3. 场景中插入车辆
4. 场景中插入标定板
5. 场景中插入 4 路或 6 路环视相机
6. 车辆、标定板、相机的删除
7. 车辆、标定板、相机的 6DoF 位姿编辑
8. XYZ 数值输入
9. Roll / Pitch / Yaw 数值输入
10. 鼠标 Gizmo 拖动平移和旋转
11. 车辆移动时，安装在车辆上的相机跟随移动
12. 相机可以单独调整相对于车辆的外参
13. 配置相机内参和畸变参数
14. 从每一个虚拟环视相机生成仿真图像
15. 输出 front / rear / left / right 四路鱼眼图像
16. 根据四路相机图像生成最终 AVM / Bird's Eye View 环视图像
17. 场景配置保存和重新加载

最终目标是做出一个类似参考视频的可交互原型，而不是只写架构设计文档。

# 二、技术路线

优先采用：

- Ubuntu 24.04
- ROS 2 Jazzy
- Python 3
- rclpy
- PyTorch
- CUDA
- gsplat
- Viser
- OpenCV
- NumPy
- YAML / JSON

除非实际实现过程中发现存在明确技术障碍，否则不要自行更换这一技术路线。

UI 和传感器渲染必须解耦。

推荐结构：

Viser
负责：

- 3DGS 上帝视角
- mesh 显示
- camera frustum
- object selection
- transform gizmo
- 属性编辑 UI

ROS 2
负责：

- Scene State
- TF
- Object management
- Camera state
- 模块通信

gsplat
负责：

- 3D Gaussian Splatting 渲染
- 虚拟相机图像生成

OpenCV / PyTorch
负责：

- fisheye projection
- remap
- BEV projection
- AVM stitching

不要直接使用“上帝视角截图”作为 AVM 输出。

AVM 必须来源于四个真实虚拟环视相机生成的图像。

# 三、软件架构

首先建立 ROS 2 workspace：

avm\_sim\_ws/
└── src/
├── avm\_interfaces/
├── scene\_manager/
├── scene\_editor/
├── gs\_renderer/
├── avm\_stitcher/
└── avm\_bringup/

建议职责如下。

## scene\_manager

负责：

- Scene State
- 场景加载
- 场景保存
- object registry
- vehicle
- calibration board
- camera
- object create/delete
- object pose update

## scene\_editor

使用 Viser。

负责：

- 浏览器 GUI
- 3DGS 场景显示
- vehicle mesh
- calibration board
- camera frustum
- object tree
- object selection
- XYZ / RPY 编辑
- transform gizmo
- add/delete object

## gs\_renderer

负责：

- Gaussian scene 加载
- virtual camera
- camera pose
- camera intrinsic
- distortion
- batch camera rendering
- calibration board rendering
- ROS Image 输出

## avm\_stitcher

负责：

front
rear
left
right

四路图像到：

AVM / BEV

的转换。

## avm\_bringup

负责：

launch 文件
配置文件
示例场景
启动整个系统。

# 四、坐标系统

必须从第一天开始统一坐标系统。

world：

Z-up

车辆 base\_link：

X = forward
Y = left
Z = up

相机：

camera\_xxx\_link

同时建立：

camera\_xxx\_optical

optical frame：

X = image right
Y = image down
Z = camera forward

TF 大致结构：

world
└── base\_link
├── camera\_front\_link
│   └── camera\_front\_optical
├── camera\_rear\_link
│   └── camera\_rear\_optical
├── camera\_left\_link
│   └── camera\_left\_optical
└── camera\_right\_link
└── camera\_right\_optical

标定板直接挂：

world
├── board\_001
├── board\_002
└── ...

相机安装位姿是可编辑参数，因此编辑期间不要错误地把可变相机外参当成永远不变的 static TF。

# 五、3DGS 场景

第一阶段至少支持：

PLY Gaussian Splat 场景加载。

必须考虑 3DGS 世界坐标与真实米制坐标之间的关系。

场景配置里预留：

scene:
gaussian\_file:

coordinate:
scale:
T\_world\_from\_gs:
ground\_z:

内部设计允许：

p\_world = scale \* R \* p\_gs + t

也就是允许未来进行 Sim(3) 对齐。

当前版本至少保证系统内部所有车辆、标定板和相机尺寸使用 meter。

# 六、车辆

车辆第一版可以使用：

GLB / GLTF mesh

实现：

- add vehicle
- delete vehicle
- move vehicle
- rotate vehicle
- XYZ
- Roll/Pitch/Yaw
- transform gizmo

车辆作为 camera 的父节点。

车辆移动：

camera world pose 必须同步改变。

相机自己的 extrinsic：

保持相对于 base\_link 不变。

# 七、标定板

实现 checkerboard calibration board。

参数至少包括：

- ID
- rows
- columns
- square\_size
- width
- height
- position
- roll
- pitch
- yaw

标定板尺寸必须使用真实物理尺寸，单位 meter。

支持：

- 添加
- 删除
- 拖动
- 旋转
- 数值输入

不能把标定板永久训练进 3DGS。

标定板必须是独立动态对象。

# 八、Camera

第一版默认建立：

front
rear
left
right

4 个 camera。

设计上支持未来扩展到 6 个。

Camera 参数包括：

name

parent

pose:
x
y
z
roll
pitch
yaw

image:
width
height

intrinsic:
fx
fy
cx
cy

distortion:
model
coefficients

camera\_model 至少为未来保留：

pinhole
fisheye
ftheta

GUI 中 camera 必须显示成 camera frustum。

选择 camera 后可以拖动 gizmo 修改相机外参。

# 九、ROS Topic

至少设计：

/tf

/sim/camera/front/image\_raw
/sim/camera/rear/image\_raw
/sim/camera/left/image\_raw
/sim/camera/right/image\_raw

/sim/camera/front/camera\_info
/sim/camera/rear/camera\_info
/sim/camera/left/camera\_info
/sim/camera/right/camera\_info

/sim/avm/image

尽量使用标准：

sensor\_msgs/Image
sensor\_msgs/CameraInfo
geometry\_msgs
tf2

不要设计没有必要的自定义消息。

# 十、虚拟相机渲染

这是项目核心。

不要简单做：

普通 perspective 图像
→ OpenCV warp
→ 假鱼眼

优先研究并实现：

3DGS
→ 根据真实 camera rays
→ fisheye / ftheta rendering

如果 gsplat 当前接口能够直接支持目标 camera model，则优先使用 gsplat 原生实现。

如果必须自行实现，则把 camera ray generation 独立成模块。

四个相机尽可能 batch rendering。

接口设计允许一次输入：

4 个 camera poses
4 组 K
4 组 distortion

一次得到：

front
rear
left
right

图像。

第一阶段先保证几何正确。

性能优化放在几何正确以后。

# 十一、标定板进入虚拟相机画面

标定板不属于 Gaussian scene。

设计动态物体 overlay pipeline。

第一版 checkerboard 可以使用：

camera ray
→ ray-plane intersection
→ board local coordinate
→ checker texture sampling

结合 Gaussian renderer 的 depth 做遮挡判断。

要求：

修改 board pose 后，相机画面中的 checkerboard 必须实时跟随变化。

# 十二、AVM

AVM 不允许直接使用上帝视角画面。

必须：

front fisheye
rear fisheye
left fisheye
right fisheye
↓
ground projection
↓
BEV
↓
overlap blending
↓
AVM

第一版定义：

BEV 范围例如：

X = -5m \~ +5m
Y = -5m \~ +5m

配置化，不要硬编码。

输出例如：

1024 × 1024。

实现：

BEV pixel
→ base\_link ground point
→ camera coordinate
→ fisheye projection
→ source image sampling

重叠区域第一版允许使用简单：

angle weighted blend

或：

distance weighted blend。

先保证投影正确。

后续再考虑：

seam finding
exposure compensation
multi-band blending。

# 十三、GUI 目标

请参考我上传的视频。

希望 GUI 大致包括：

左侧或者中间：

3D 上帝视角。

能够：

Orbit
Pan
Zoom

能够显示：

3DGS
Vehicle
Calibration Board
Camera Frustum

右侧：

Scene Objects

例如：

Scene
├── Vehicle
├── Camera
│   ├── Front
│   ├── Rear
│   ├── Left
│   └── Right
└── Calibration Boards
├── Board001
└── Board002

选中对象后显示：

Transform

Position:
X
Y
Z

Rotation:
Roll
Pitch
Yaw

同时存在：

Transform Gizmo。

GUI 还需要：

Add Vehicle
Add Camera
Add Board
Delete

Camera Preview：

Front
Rear
Left
Right

AVM Preview。

第一版 UI 不追求漂亮，优先实现功能完整和稳定。

# 十四、场景配置文件

设计一个清晰的：

scene.yaml

例如：

scene:
gaussian: assets/factory.ply

coordinate:
scale: 1.0
ground\_z: 0.0
T\_world\_from\_gs: []

vehicle:
asset: assets/car.glb
pose: []

cameras:
front:
parent: base\_link
pose: []
model: fisheye
width: 1920
height: 1080
K: []
D: []

boards:

- id: board\_001
  type: checkerboard
  size: []
  pose: []

实现：

Save Scene

Load Scene。

# 十五、开发顺序

严格按照以下顺序推进。

Milestone 1：

ROS 2 workspace
\+
基础 package
\+
3DGS PLY 加载
\+
Viser 上帝视角

验收：

浏览器可以打开场景并自由 Orbit / Pan / Zoom。

Milestone 2：

Vehicle
Board
Camera

对象管理和 Transform Gizmo。

验收：

可以添加、选择、删除和修改位姿。

Milestone 3：

TF tree
\+
scene.yaml

验收：

车辆移动时 camera 跟随；
scene 保存后重新打开保持一致。

Milestone 4：

gsplat virtual camera。

先 pinhole。

验收：

改变 camera pose 后 camera image 正确改变。

Milestone 5：

fisheye / ftheta camera。

验收：

4 个 virtual cameras 可以稳定输出图像。

Milestone 6：

动态 checkerboard。

验收：

标定板移动和旋转会正确出现在相机图像中。

Milestone 7：

AVM stitcher。

验收：

四路相机图像能够生成 Bird's Eye View 环视图。

Milestone 8：

性能优化和 README。

# 十六、开发工作方式

非常重要：

不要只给我建议、架构图或者伪代码。

你需要真正创建项目文件、编写代码并运行。

每完成一个阶段都要：

1. 运行代码
2. 查看错误
3. 修复错误
4. 再运行
5. 编写最小测试
6. 确认上一阶段正常后继续

不要因为第一次运行失败就停止。

如果遇到依赖/API 版本问题：

优先查看官方文档和源码确认当前 API，然后修改实现。

不要凭记忆猜测 API。

如果某一个高级功能短时间不能完成：

不要阻塞整个项目。

实现 fallback，并在 README 中记录：

- 当前实现
- 限制
- 后续替换方式

# 十七、代码质量要求

代码必须：

- 模块化
- 有类型提示
- 有清晰注释
- 避免巨型 Python 文件
- 避免全局状态
- pose / coordinate conversion 集中管理
- camera model 集中管理
- renderer 与 GUI 解耦

特别建立：

transforms.py

统一处理：

quaternion
rotation matrix
Euler
ROS coordinates
graphics coordinates
camera optical coordinates

不要在不同文件中重复手写坐标转换。

# 十八、测试

至少增加：

transform test

camera projection test

scene save/load test

board ray intersection test

AVM projection test

重点测试坐标转换。

# 十九、性能原则

第一目标：

正确。

第二目标：

可交互。

第三目标：

性能。

不要为了追求第一版本 30 FPS 而牺牲坐标和几何正确性。

但架构上避免：

CPU ? GPU 不必要的数据拷贝。

图像处理尽可能保留在 GPU。

# 二十、当前不做

当前明确不要花时间实现：

- 用户管理
- 权限管理
- 日志中心
- AI 场景生成
- 自动生成报告
- PDF/Word 报告
- 云端场景库
- 场景版本回滚
- 自动标定算法
- AI 标定优化
- 批量标定

为这些功能留下接口即可。

# 二十一、你现在立即开始执行的任务

不要先给我写一篇很长的方案。

第一步：

1. 阅读需求文档
2. 分析参考视频
3. 检查当前开发环境
4. 检查 GPU / CUDA / Python / ROS 2
5. 创建 ROS 2 workspace
6. 建立上述 package
7. 创建 dependency/setup 文档
8. 实现第一个可以运行的 3DGS + Viser demo

然后实际运行它。

如果缺少真实 3DGS PLY 或车辆 GLB：

不要停止开发。

建立明确的：

assets/
example\_assets/

接口和 fallback。

可以先使用最小测试资产验证整个 pipeline，并保证以后替换真实资产不需要修改业务架构。

完成 Milestone 1 后继续 Milestone 2。

除非遇到必须由我提供的信息，例如缺失的真实相机标定参数或私有资产，否则请自主做合理工程决策，不要每一步都停下来向我确认。

最终我要得到的是：

一个真正可以运行和继续开发的工程，而不是项目规划书。

最终同时输出：

1. 完整源码
2. README.md
3. INSTALL.md
4. 项目目录说明
5. 运行命令
6. ROS 2 launch 命令
7. 示例 scene.yaml
8. 已完成功能列表
9. 未完成功能列表
10. 已知问题
11. 下一阶段建议

现在开始实际实现。