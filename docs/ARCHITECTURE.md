# 工程结构与几何契约

```text
avm_sim_ws/src/
  avm_interfaces/   SceneCommand.srv，唯一自定义事务服务
  scene_manager/   models、store、persistence、transforms、camera_models、frames
  scene_editor/    Viser 对象、属性、Gizmo、文件操作、预览；无传感器截图
  gs_renderer/     PLY、CPU 射线、gsplat adapter、棋盘格、worker、ROS 节点
  avm_stitcher/    BEV 逆映射、权重融合、ROS 同步
  avm_bringup/     launch、config/scene.yaml、example_assets
assets/             外部真实资产入口
tests/              几何、模型、持久化、投影与交互边界测试
tools/              启动、环境检查、独立渲染与参考视频检查
```

## 状态与通信

SceneStore 是唯一可写状态，使用原子验证、深拷贝快照与 revision。所有 UI 事件通过队列进入主循环；拖动事件按 Viser 异步回调顺序入队。ROS 模式只有 scene_manager 修改状态，编辑器通过服务请求修改并订阅返回状态。

ROS 图像使用 sensor_msgs/Image（rgb8）与 CameraInfo。标准消息无法表达“添加物体并返回错误和 revision”的事务，因此 avm_interfaces 只定义一个 SceneCommand 服务。请求载荷为受验证的 JSON 命令，避免大量自定义传感器消息。scene/state 和 render/state 暂为 std_msgs/String JSON，schema_version=1。只在内部仿真网络使用，未实现远程认证。

| Topic / Service | 用途 |
| --- | --- |
| /sim/scene/state | transient-local 可靠状态快照 |
| /sim/scene/command | 增删、位姿、属性、保存/加载服务 |
| /tf | 20 Hz 动态 TF；含可变相机外参 |
| /sim/camera/{name}/image_raw | 独立虚拟相机 RGB 图像 |
| /sim/camera/{name}/camera_info | 同帧内参与畸变 |
| /sim/render/state | 此批次 stamp、revision 和完整标定快照 |
| /sim/avm/image | 四路相机投影融合图 |

四路图像共享时间戳。stitcher 按完全相同的 stamp 收齐图像与该次渲染的 scene，而不是读取最新外参去拼接旧图像。缓冲最多 12 批；不完整批次丢弃。ROS TF 自身有历史缓存，删除物体不会清除其他节点旧的 tf2 缓存。

standalone 使用同一 SceneStore 与核心模块，传感器驻留独立进程，最多一批任务在运行，不堆积编辑中间态。ROS 模式是四个独立节点；Viser 上帝相机完全不参与 SensorPipeline。

## 坐标

全部长度 meter、YAML 角度 radian、UI 角度 degree。pose 固定为 [x,y,z,roll,pitch,yaw]；旋转为外禀 XYZ，即 Rz(yaw) Ry(pitch) Rx(roll)。四元数内部 WXYZ，ROS 消息字段 XYZW。

world、base_link：X 前、Y 左、Z 上。T_parent_child 采用列向量语义。相机安装 frame 与 base 同轴定义；optical 的 X 右、Y 下、Z 前。固定 R_link_optical 为：

```text
 0  0  1
-1  0  0
 0 -1  0
```

T_world_optical = T_world_base × T_base_camera_link × T_link_optical。车辆移动不修改 T_base_camera_link。Gizmo 在 world 显示，拖动相机时逆变换回 parent。手柄显示在对象上方避免被模型遮挡，该显示偏移在回写前扣除，旋转仍绕对象自己的原点。

Sim(3)：p_world = scale × R × p_gs + t；协方差同步转换 scale² R Sigma R^T。T_world_from_gs 只允许刚体变换，scale 单独为正数。ground_z 是 world 水平地面，不跟随车辆滚转。

环境与车辆可分别来自独立 3DGS PLY。环境使用 `coordinate.scale / T_world_from_gs` 映射到 world；车辆先用 `asset_scale / asset_pose` 映射到 `base_link`，再用动态 `vehicle.pose` 映射到 world。合成时位置、Gaussian 四元数和协方差同步变换，然后作为一个 Gaussian 集合交给 CPU 或 gsplat 后端，因此车辆参与相机遮挡。原始车辆 PLY 会被缓存，拖动车辆不重复读取文件；当前后端仍需重建合成 renderer/GPU buffer。

Checkerboard 在自身 XY 平面，中心为 pose 原点，法线局部 +Z，可双面显示。rows/columns 指方格数，不是内部角点数；width=columns×square_size，height=rows×square_size。尺寸由这些独立参数唯一决定。

## 相机、深度与 AVM

pinhole D=[k1,k2,p1,p2,k3]；fisheye D=[k1,k2,k3,k4] 为 OpenCV theta 多项式。ftheta 当前也采用归一化 theta 多项式，允许 max_angle>pi/2，不能直接导入厂商任意 pixel-space 多项式。max_angle 为光轴到有效射线的半角，不是完整 FOV；真实镜头应使用标定有效域。

像素中心位于整数坐标（OpenCV）。已核对安装的 gsplat 1.5.3 CUDA 源码按 j+.5、i+.5 采样，adapter 给 K 的主点加 .5，其他模块保持 OpenCV 定义。

所有 overlay 输入深度均为沿单位相机射线的距离 meter。CPU 为 alpha 加权的射线高斯响应深度；gsplat 1.5.3 RGB+ED 是 Gaussian 中心 optical Z 的期望值，通过除以 ray.z 转换。二者都是体渲染近似深度，透明/大 Gaussian 的遮挡边界存在偏差。

BEV 行向上为 base +X，列向左为 base +Y。每像素 base x/y 由配置范围与像素中心确定，求 base z 使变换到 world 后 z=ground_z，故车辆 pitch/roll 不会错误倾斜地面。再变换到相机 optical 坐标、使用同一 camera_model 投影，通过 OpenCV remap 从传感器图像采样。重叠权重为中心角余弦平方除以距离。无有效源像素处保持暗色。对高于地面的物体，地面假设会造成拉伸/重影。
