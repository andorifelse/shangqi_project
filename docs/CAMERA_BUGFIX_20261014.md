# 相机拖动与预览问题排查（2026-10-14）

排查输入：用户提供的 `yaml/scene_v2.yaml`。保留原文件，不修改环境坐标、车辆尺度或相机参数。

## 坐标与渲染检查

此配置车辆 yaw 为 0，左相机位于 base_link 的 +Y 侧，yaw=+π/2；右相机位于 −Y 侧，yaw=−π/2，前后相机分别朝 +X/−X。`board_001` 位于世界 +Y 一侧。

按光学坐标计算，标定板中心在左相机前方，其他三路的 optical Z 为负。射线命中像素数为 left=1697、rear/right/front=0。真实 GPU 出图比较有板/无板，板引起的图像变化像素为 left=1624，其他三路为 0（部分射线受 Gaussian 深度遮挡）。

采用四种不同颜色 Gaussian 放到每路相机前方，验证 rear/left/right/front 的批量 GPU 渲染与逐相机渲染结果逐像素一致。没有发现 GPU 相机索引交换或 optical 坐标反转。

运行中的原系统只有一套 ROS 节点，每路图像只有一个发布者。其内存场景存在额外 `board_002`，该板在 −Y 侧，处于右相机有效视野；不能将这块板与文件中的左侧 `board_001` 混为同一块。

## 已定位并修复的编辑器问题

1. 所有选中对象的 Gizmo 使用同一个 `/gizmo` 路径。Viser/WebSocket 可延迟提交拖动事件，旧控件被删除后相同名称的新控件可能收到旧消息。同时 `_drag` 使用当前 `self.selected` 决定对象，旧左相机位姿可能写入新选中的右相机。这样右相机会被放到左侧且朝向左侧，出现标定板“串路”的现象。
2. ROS 服务回包和权威状态 topic 有延迟。原代码会在拖动过程中把旧状态写回 PivotControls，干扰持续拖动参考，造成跳动/回弹。原服务修改也未按单个在途请求顺序串行化。
3. ROS 预览只逐路刷新最新图像，未核对渲染 revision，也没有显示待刷新状态。编辑后仍在显示旧图像时容易误判新位姿没有生效；相机标签顺序还随 YAML 字典顺序变化。

修复措施：

- 每次选择使用单独 Gizmo wire name；只接受当前控件的回调，拒绝被替换控件的迟到事件。
- 识别拖动 start/update/end；拖动中不使用权威状态覆写手柄。
- `RemoteStore` 串行提交服务请求，合并同一相机等待中的连续位姿，保留未确认的本地位姿，直到对应 revision 的权威状态返回。
- 预览按 stamp 和 revision 组装完整相机批次，拒绝旧 revision；未完成当前位姿渲染时显示等待状态。
- 标签固定为 front/rear/left/right 顺序并附 topic，校验收到的 Image.frame_id。

单次稳定选择的水平反向拖动没有在独立测试场景中复现；同方向拖动在旧 standalone/ROS 上均向鼠标方向移动。本次修复的是已明确确认的迟到事件串对象和异步回写跳动，未通过翻转坐标轴掩盖问题。

## 验证结果

- 新增迟到 Gizmo 事件、拖动状态回写、RemoteStore 排序合并/拒绝恢复、预览跨批次同步回归。
- GPU 回归：调整 YAML 相机顺序为 rear/left/right/front；左侧标定板只进入 left；单独移动 left 后其余三路图像逐像素不变。
- 完整回归：31 passed（80.54 s），含两项 CUDA 用例。
- 独立 ROS_DOMAIN_ID=84 中启动原始真实 `scene_v2.yaml`，移动 left 的 X 0.3 m：left 像素均值变化约 9.3725，rear/right/front 均为 0；复原后四路逐像素恢复。该探针只修改隔离实例的内存状态。
- 修复后的浏览器测试副本中，左相机绿色箭头向屏幕左方拖动，手柄同向移动，X 从 0 变为约 −1.81；切换 right 后其 X/Y/yaw 保持原值（约 0/−1.72/−90°），没有接收左相机的位姿。预览状态从等待 revision 7 转为显示 revision 7。

## 使用更新

Python 包采用 colcon symlink-install，改动已进入源目录；已运行进程仍持有旧 Python 代码，需要停止并重启。启动时明确选择用户的 YAML：

```bash
cd /home/wzc/shangqi_project
bash tools/run_gpu_ros.sh \
  scene:=/home/wzc/shangqi_project/yaml/scene_v2.yaml \
  host:=0.0.0.0
```

相机预览会标出当前 revision 和 stamp；编辑后先显示等待，随后显示新批次。在 Mac 上测试时刷新浏览器，确保连接重启后的系统。
