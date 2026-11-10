# 验证记录

版本1.0.0，分支codex/calibration-pipeline-package；实际源码以SOURCE_MANIFEST.json中SHA256为准。没有安装到用户默认Python或更换共享环境。

- 独立wheel在新建Python3.12.13环境中离线安装；从/tmp运行集中测试，38项全部通过。
- 环境：NumPy2.5.4，SciPy1.18.1，OpenCV5.0.0（distribution5.0.0.93），plyfile1.1.5。
- 真实PNG CPU贯通夹具同时走针孔与鱼眼：六面合成→SB角点→3初值平面内参→条件大板PnP→BEV。保留退化、角点ID未决与非唯一性状态。
- 数值回归包括skew、Brown5/theta4、镜头折叠有效域、双线性采样、黑像素覆盖、融合与接缝、全部对应点候选选择、PLY字段重排与SH/TRS变换和点数守恒。
- 安全与接口检查包括默认无写、拒覆盖、暂停状态、非布尔execute、空阶段选择、嵌套spec路径、自定义外参图像搬移和失败记录；CLI/API计划相同。
- GPU worker导入链的Python3.7语法检查通过，只代表语法兼容；GPU、CUDA后端、实际六相机资产渲染与实拍均未执行。
- 原项目plan、doctor通过；默认系统Python缺PIL的可选警告与新包无关。

命令行示例已搬离原仓库并从/tmp执行：针孔与鱼眼各6张训练、2张留出，3组内参候选、6组外参候选，BEV覆盖率均为1.0；针孔仍退化，鱼眼仍标记多初值不稳定。压缩包CRC、源码/轮子版本和CLI/API计划一致性已核对，详见validation/independent_verification.json。

人工使用评鉴待用户确认：安装后按README命令运行examples/cpu_demo/demo.json，打开output/bev/pinhole/comparison.png及fisheye/comparison.png，检查流程可调用、输出可追溯、退化状态没有被改写为标定成功。这个单相机软件夹具不用于评价多相机接缝画质。
