"""Chinese display strings. Serialized algorithm/status/model IDs stay unchanged."""
MODEL_LABELS = {'fisheye':'鱼眼', 'pinhole':'针孔', 'ftheta':'等距多项式'}
MODEL_VALUES = {v:k for k,v in MODEL_LABELS.items()}
BACKEND_LABELS = {'cpu':'处理器参考渲染', 'gsplat':'显卡高斯渲染'}
BACKEND_VALUES = {v:k for k,v in BACKEND_LABELS.items()}
CAMERA_LABELS = {'front':'前相机','rear':'后相机','left':'左相机','right':'右相机',
                 'extra_left':'左侧扩展相机','extra_right':'右侧扩展相机','all':'全部相机'}
STATUS_LABELS = {
    'CHECKED':'数值检查通过', 'REJECTED':'未通过', 'SKIPPED':'未执行',
    'ESTIMATED':'已估计', 'GENERATED':'已生成', 'FAILED':'失败',
    'RUNNING':'运行中', 'EXECUTED':'执行完成', 'DIAGNOSTIC_COMPLETED':'诊断完成',
    'DEGENERATE_PARALLEL_PLANE_ACQUISITION':'固定平面退化（参数不可唯一确定）',
    'MULTISTART_UNSTABLE':'多初值结果不稳定',
    'MULTISTART_STABLE_LOCAL_SOLUTION':'多初值局部一致（不保证唯一）',
    'MULTISTART_INCOMPLETE':'部分初值求解失败', 'SINGLE_START_LOCAL_SOLUTION':'单初值局部解',
    'UNRESOLVED_INTRINSICS_AND_CORNER_IDS':'内参与角点身份仍有歧义',
    'NO_VALID_CONDITIONAL_POSES':'无有效条件外参', 'DIAGNOSTIC_RENDERED':'已生成诊断图像',
    'NO_COVERAGE':'无有效覆盖', 'INSUFFICIENT_DETECTIONS':'有效检测数量不足',
    'IMAGE_DECODE_FAILED':'图像读取失败', 'PLANNED':'待执行', 'DRY_RUN':'仅检查配置',
    'NOT_RUN':'未运行', 'PARTIAL':'部分完成', 'ACCEPTED':'已通过',
    'INTRINSIC_FIT_FAILED':'内参拟合失败', 'LARGE_BOARD_DETECTION_FAILED':'大板检测失败',
    'INTRINSIC_SOLUTIONS_UNAVAILABLE':'无可用内参', 'COMPLETED_WITH_FAILURES':'部分阶段失败',
    'complete':'完成', 'failed':'失败', 'cancelled':'已取消', 'rejected':'未通过',
    'intrinsics':'内参', 'extrinsics':'外参', 'baseline':'基线', 'bev':'鸟瞰',
    'prepare':'场景准备', 'capture':'图像采集', 'compose':'镜头图像合成',
    'opencv_baseline':'棋盘格基线', 'pipeline_1_0_0':'标定流程 1.0.0',
    'fisheye_multitilt':'多倾角鱼眼标定',
}


def camera_name(name):
    return CAMERA_LABELS.get(name,name)


def object_label(name):
    if name=='base_link':return '车辆（base_link）'
    if name in CAMERA_LABELS:return f'{camera_name(name)}（{name}）'
    if name.startswith('board_'):return f'标定板（{name}）'
    return name


def object_id(label):
    if label.endswith('）') and '（' in label:return label.rsplit('（',1)[1][:-1]
    return label


def display_text(value):
    value=str(value)
    return STATUS_LABELS.get(value, MODEL_LABELS.get(value, value))


def error_text(error):
    """Translate known core errors at the display boundary, retaining diagnostics."""
    message=str(error)
    translations={
        'Scene manager is not ready':'场景管理器尚未就绪',
        'Scene changed; refresh before editing':'场景已变化，请刷新后重试',
        'MVP allows one vehicle':'当前仅支持一辆车',
        'Object already exists':'对象已存在',
        'Camera parent must exist':'相机的参考坐标系不存在',
        'Pose must contain six finite XYZ/RPY values':'位姿需要六个有限的位置和角度数值',
        'Board asset_scale must be positive':'标定板缩放倍数必须大于零',
        'Board asset must be a Gaussian .ply file':'标定板模型必须是高斯 PLY 文件',
        'No such file or directory':'文件或目录不存在',
        'Permission denied':'没有访问权限',
        'Asset does not exist:':'模型文件不存在：',
        'Object asset does not exist:':'对象模型文件不存在：',
        'Object does not exist:':'对象不存在：',
        'Not a Gaussian PLY; missing properties:':'不是有效的高斯 PLY，缺少属性：',
        'AVM requires all front/rear/left/right images; missing':'环视需要前后左右四路图像，缺少',
        'Property cannot be edited:':'此属性不可编辑：',
        'Invalid scene schema:':'场景配置格式无效：',
        'scene.yaml must contain a mapping':'场景配置必须为键值结构',
        'Invalid YAML:':'配置文件格式无效：',
        'Vehicle dimensions and scale must be positive':'车辆尺寸和缩放必须大于零',
        'Invalid board dimensions':'标定板尺寸无效',
        'Image dimensions must be integers in [16,4096]':'图像尺寸必须为16到4096之间的整数',
        'K requires positive fx/fy, zero skew, homogeneous last row':'内参须满足正焦距、零倾斜及有效齐次末行',
        'Camera max_angle outside supported domain':'相机最大光线半角超出支持范围',
        'fisheye requires 4 distortion coefficients':'鱼眼模型需要4个畸变系数',
        'pinhole requires 5 distortion coefficients':'针孔模型需要5个畸变系数',
        'ftheta requires 4 distortion coefficients':'等距多项式模型需要4个畸变系数',
        'Fisheye polynomial must be monotonic within max_angle':'鱼眼畸变多项式在有效视角内必须单调',
        'could not convert string to float:':'无法将输入转换为数值：',
    }
    for old,new in translations.items():message=message.replace(old,new)
    return message
