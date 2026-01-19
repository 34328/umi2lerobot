import dataclasses


@dataclasses.dataclass(frozen=True)
class RobotConfig:
    motors: list[str]
    cameras: list[str]
    camera_to_image_key: dict[str, str]
    umi_state_data_name: list[str]
    umi_action_data_name: list[str]
    # 触觉传感器配置: key是zarr中的数据键名, value是lerobot中的字段名
    tactile_sensors: dict[str, str] = dataclasses.field(default_factory=dict)
    # 触觉数据形状配置: key是传感器名, value是shape (不包含batch维度)
    tactile_shapes: dict[str, tuple[int, ...]] = dataclasses.field(default_factory=dict)
    # 音频传感器配置: key是zarr中的数据键名, value是lerobot中的字段名
    audio_sensors: dict[str, str] = dataclasses.field(default_factory=dict)
    # 音频数据形状配置: key是传感器名, value是shape (不包含batch维度)
    audio_shapes: dict[str, tuple[int, ...]] = dataclasses.field(default_factory=dict)
    # Demo pose 配置: key是zarr中的数据键名(不含robot0_前缀), value是lerobot中的字段名
    demo_pose_sensors: dict[str, str] = dataclasses.field(default_factory=dict)
    # Demo pose 形状配置: key是传感器名, value是shape
    demo_pose_shapes: dict[str, tuple[int, ...]] = dataclasses.field(default_factory=dict)
    # 是否为双臂配置
    is_bimanual: bool = False
    # 双臂机器人前缀列表
    robot_prefixes: tuple[str, ...] = ("robot0",)
    
    # ============================================================================
    # MCAP 专用配置 (用于 GenRobot 等 MCAP 数据源)
    # ============================================================================
    # MCAP 相机 topic 映射: key是lerobot字段名, value是MCAP topic路径
    mcap_camera_topics: dict[str, str] = dataclasses.field(default_factory=dict)
    # MCAP 数值数据 topic 映射: key是lerobot字段名, value是(topic路径, 插值类型, 数据形状)
    # 插值类型: "linear" (线性插值), "pose" (位姿SLERP插值), "nearest" (最近邻)
    mcap_numeric_topics: dict[str, tuple[str, str, tuple[int, ...]]] = dataclasses.field(default_factory=dict)
    # MCAP 参考 topic (用作时间对齐基准，通常是主相机)
    mcap_ref_topic: str = ""



MV_UMI_CONFIG = RobotConfig(
    motors=[
        "x",
        "y",
        "z",
        "rx",
        "ry",
        "rz",
        "gripper",
    ],

    cameras=[
        "camera0_rgb",
        # "camera1_rgb"
    ],

    camera_to_image_key={
        "camera0_rgb": "camera0_rgb",
        # "camera1_rgb": "camera1_rgb",
    },  
    umi_state_data_name=["eef_pos", "eef_rot_axis_angle", "gripper_width"],
    umi_action_data_name=["eef_pos", "eef_rot_axis_angle", "gripper_width"],
)


# ManiWAV 配置 - 包含音频传感器
MANIWAV_CONFIG = RobotConfig(
    motors=[
        "x",
        "y",
        "z",
        "rx",
        "ry",
        "rz",
        "gripper",
    ],

    cameras=[
        "camera0_rgb",
    ],

    camera_to_image_key={
        "camera0_rgb": "camera0_rgb",
    },  
    umi_state_data_name=["eef_pos", "eef_rot_axis_angle", "gripper_width"],
    umi_action_data_name=["eef_pos", "eef_rot_axis_angle", "gripper_width"],
    # 音频传感器配置
    audio_sensors={
        "mic_0": "mic_0",  # zarr中的key -> lerobot中的字段名
        "mic_1": "mic_1",
    },
    audio_shapes={
        "mic_0": (800,),  # 音频数据形状
        "mic_1": (800,),
    },
)


UMI_CONFIG = RobotConfig(
    motors=[
        "x",
        "y",
        "z",
        "rx",
        "ry",
        "rz",
        "gripper",
    ],

    cameras=[
        "camera0_rgb",
        "camera1_rgb"
    ],

    camera_to_image_key={
        "camera0_rgb": "camera0_rgb",
        "camera1_rgb": "camera1_rgb",
    },  
    umi_state_data_name=["eef_pos", "eef_rot_axis_angle", "gripper_width"],
    umi_action_data_name=["eef_pos", "eef_rot_axis_angle", "gripper_width"],
)



TOUCH_IN_THE_WILD_CONFIG = RobotConfig(
    motors=[
        "x",
        "y",
        "z",
        "rx",
        "ry",
        "rz",
        "gripper",
    ],

    cameras=[
        "camera0_rgb",
    ],

    camera_to_image_key={
        "camera0_rgb": "camera0_rgb"
    },  
    umi_state_data_name=["eef_pos", "eef_rot_axis_angle", "gripper_width"],
    umi_action_data_name=["eef_pos", "eef_rot_axis_angle", "gripper_width"],
    # 触觉传感器配置
    tactile_sensors={
        "camera0_tactile": "camera0_tactile",  # zarr中的key -> lerobot中的字段名
    },
    tactile_shapes={
        "camera0_tactile": (12, 64),  # 触觉数据形状
    },
)



# 双臂UMI配置 - 包含 robot0 和 robot1
BIMANUAL_UMI_CONFIG = RobotConfig(
    motors=[
        # robot0 (左臂或第一个臂)
        "robot0_x",
        "robot0_y",
        "robot0_z",
        "robot0_rx",
        "robot0_ry",
        "robot0_rz",
        "robot0_gripper",
        # robot1 (右臂或第二个臂)
        "robot1_x",
        "robot1_y",
        "robot1_z",
        "robot1_rx",
        "robot1_ry",
        "robot1_rz",
        "robot1_gripper",
    ],

    cameras=[
        "camera0_rgb",
        "camera1_rgb",
    ],

    camera_to_image_key={
        "camera0_rgb": "camera0_rgb",
        "camera1_rgb": "camera1_rgb",
    },
    # 双臂数据名称 - 使用元组表示 (robot_prefix, data_fields)
    umi_state_data_name=["eef_pos", "eef_rot_axis_angle", "gripper_width"],
    umi_action_data_name=["eef_pos", "eef_rot_axis_angle", "gripper_width"],
    # 标记为双臂配置
    is_bimanual=True,
    robot_prefixes=("robot0", "robot1"),
)


# ViTaMIn 配置 - 包含触觉传感器和demo pose
VITAMIN_CONFIG = RobotConfig(
    motors=[
        "x",
        "y",
        "z",
        "rx",
        "ry",
        "rz",
        "gripper",
    ],

    cameras=[
        "camera0_rgb",
        "left_tactile",
        "right_tactile",
    ],

    camera_to_image_key={
        "camera0_rgb": "camera0_rgb",
        "left_tactile": "camera0_left_tactile",   # lerobot中的字段名 -> zarr中的key
        "right_tactile": "camera0_right_tactile",
    },  
    umi_state_data_name=["eef_pos", "eef_rot_axis_angle", "gripper_width"],
    umi_action_data_name=["eef_pos", "eef_rot_axis_angle", "gripper_width"],

    # Demo pose 配置
    demo_pose_sensors={
        "demo_start_pose": "demo_start_pose",  # zarr中的key (不含robot0_前缀) -> lerobot中的字段名
        "demo_end_pose": "demo_end_pose",
    },
    demo_pose_shapes={
        "demo_start_pose": (6,),
        "demo_end_pose": (6,),
    },
)


# ManiForce 配置 - 包含力传感器数据和双相机
# 数据结构:
#   action: (N, 8) - 8维动作
#   state: (N, 7) - 7维状态
#   pose_wrt_start: (N, 7) - 相对起始位姿
#   handeye_cam_1: (N, 800, 1280, 3) - 高分辨率相机
#   handeye_cam_2: (N, 480, 640, 3) - 低分辨率相机
#   img_timestamps: 图像时间戳
MANIFORCE_CONFIG = RobotConfig(
    motors=[
        "x",
        "y", 
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
        "gripper",
    ],

    cameras=[
        "handeye_cam_1",
        "handeye_cam_2",
    ],

    camera_to_image_key={
        "handeye_cam_1": "handeye_cam_1",
        "handeye_cam_2": "handeye_cam_2",
    },
    # ManiForce 直接使用 state 和 action 字段，不需要组合
    umi_state_data_name=[],  # 特殊处理，直接读取 state 字段
    umi_action_data_name=[],  # 特殊处理，直接读取 action 字段

    # pose_wrt_start 作为额外的观测状态
    demo_pose_sensors={
        "pose_wrt_start": "pose_wrt_start",
    },
    demo_pose_shapes={
        "pose_wrt_start": (7,),  # x, y, z, qx, qy, qz, qw
    },
)


# LEGATO 配置 - 双手腕灰度相机 + 多种观测数据
# 数据结构:
#   actions: (N, 7) - 7维动作 (x, y, z, rx, ry, rz, gripper)
#   obs/left_gray: (N, 128, 128, 1) - 左手腕灰度图
#   obs/right_gray: (N, 128, 128, 1) - 右手腕灰度图
#   obs/delta_eulers: (N, 6) - 欧拉角增量
#   obs/delta_positions: (N, 6) - 位置增量
#   obs/delta_quaternions: (N, 8) - 四元数增量
#   obs/quaternions: (N, 8) - 四元数
#   obs/position_diffs: (N, 6) - 位置差异
#   dones: (N,) - 完成标志
#   rewards: (N,) - 奖励
LEGATO_SIM_CONFIG = RobotConfig(
    motors=[
        "x",
        "y",
        "z",
        "rx",
        "ry",
        "rz",
        "gripper",
    ],

    cameras=[
        "left_gray",
        "right_gray",
    ],

    camera_to_image_key={
        "left_gray": "left_gray",
        "right_gray": "right_gray",
    },
    umi_state_data_name=[],  # 特殊处理，直接读取 actions 作为 state
    umi_action_data_name=[],  # 特殊处理，直接读取 actions 字段
    
    # LEGATO 观测数据配置
    demo_pose_sensors={
        "delta_eulers": "delta_eulers",
        "delta_positions": "delta_positions",
        "delta_quaternions": "delta_quaternions",
        "quaternions": "quaternions",
        "position_diffs": "position_diffs",
    },
    demo_pose_shapes={
        "delta_eulers": (6,),
        "delta_positions": (6,),
        "delta_quaternions": (8,),
        "quaternions": (8,),
        "position_diffs": (6,),
    },
)


# LEGATO 配置 - 真实数据 (双手腕灰度相机 + 多种观测数据)
# 数据结构:
#   actions: (N, 7) - 7维动作 (x, y, z, rx, ry, rz, gripper)
#   obs/left_gray: (N, 128, 128, 1) - 左手腕灰度图
#   obs/right_gray: (N, 128, 128, 1) - 右手腕灰度图
#   obs/delta_eulers: (N, 6) - 欧拉角增量
#   obs/delta_positions: (N, 6) - 位置增量
#   obs/delta_quaternions: (N, 8) - 四元数增量
#   obs/graspings: (N, 2) - 抓取状态
#   dones: (N,) - 完成标志
#   rewards: (N,) - 奖励
LEGATO_REAL_CONFIG = RobotConfig(
    motors=[
        "x",
        "y",
        "z",
        "rx",
        "ry",
        "rz",
        "gripper",
    ],

    cameras=[
        "left_gray",
        "right_gray",
    ],

    camera_to_image_key={
        "left_gray": "left_gray",
        "right_gray": "right_gray",
    },
    umi_state_data_name=[],  # 特殊处理，直接读取 actions 作为 state
    umi_action_data_name=[],  # 特殊处理，直接读取 actions 字段
    
    # LEGATO 真实数据观测配置
    demo_pose_sensors={
        "delta_eulers": "delta_eulers",
        "delta_positions": "delta_positions",
        "delta_quaternions": "delta_quaternions",
    },
    demo_pose_shapes={
        "delta_eulers": (6,),
        "delta_positions": (6,),
        "delta_quaternions": (8,),
    },
)


# DexUMI 配置 - 灵巧手操作数据 (XHand 12 DOF)
# 数据结构:
#   camera_0/rgb: (N, 400, 640, 3) - RGB相机图像
#   fsr: (N, 3) - 力敏电阻传感器
#   hand_action: (N, 12) - 手部动作 (XHand 12个电机关节)
#   pose: (N, 6) - 末端位姿 (x, y, z, rx, ry, rz)
#   proprioception: (N, 12) - 本体感知 (XHand 12个电机关节)
DEXUMI_CONFIG = RobotConfig(
    motors=[
        # inspire1
        # "kHandPinky",
        # "kHandRing",
        # "kHandMiddle",
        # "kHandIndex",
        # "kHandThumbBend",
        # "kHandThumbRotation"

        # Xhand
        # --- 小指 (Pinky) 2 DoF ---
        "kHandPinkyTip",       # 指尖弯曲
        "kHandPinkyBend",      # 指根弯曲

        # --- 无名指 (Ring) 2 DoF ---
        "kHandRingTip",
        "kHandRingBend",

        # --- 中指 (Middle) 2 DoF ---
        "kHandMiddleTip",
        "kHandMiddleBend",

        # --- 食指 (Index) 3 DoF ---
        "kHandIndexTip",       # 指尖弯曲
        "kHandIndexBend",      # 指根弯曲
        "kHandIndexRotation",  # 侧摆/旋转 (这是 XHand 食指特有的)

        # --- 拇指 (Thumb) 3 DoF ---
        "kHandThumbTip",       # 指尖弯曲
        "kHandThumbBend",      # 指根弯曲
        "kHandThumbRotation"   # 拇指旋转 (对掌)
    ],

    cameras=[
        "camera_0",
    ],

    camera_to_image_key={
        "camera_0": "camera_0",
    },
    umi_state_data_name=[],  # 特殊处理，直接读取 proprioception 字段
    umi_action_data_name=[],  # 特殊处理，直接读取 hand_action 字段
    
    # DexUMI 额外观测数据配置
    demo_pose_sensors={
        "fsr": "fsr",
        "pose": "pose",
    },
    demo_pose_shapes={
        "fsr": (3,),
        "pose": (6,),  # x, y, z, rx, ry, rz
    },
)


# FastUMI 配置 - 单臂任务数据结构
# 数据结构 (每个 episode 一个 HDF5 文件):
#   action: (N, 7) - 7维动作 (x, y, z, rx, ry, rz, gripper)
#   observations/images/front: (N, 1080, 1920, 3) - 前置相机图像
#   observations/qpos: (N, 7) - 7维关节位置 (x, y, z, rx, ry, rz, gripper)
FASTUMI_CONFIG = RobotConfig(
    motors=[
        "x",
        "y",
        "z",
        "rx",
        "ry",
        "rz",
        "gripper",
    ],

    cameras=[
        "front",
    ],

    camera_to_image_key={
        "front": "front",
    },
    umi_state_data_name=[],  # 特殊处理，直接读取 observations/qpos 字段
    umi_action_data_name=[],  # 特殊处理，直接读取 action 字段
    
    # FastUMI 数据结构简单，不需要额外的 demo_pose
    demo_pose_sensors={},
    demo_pose_shapes={},
    is_bimanual=False,
    robot_prefixes=(),
)


# DexWild 配置 - 单手任务数据结构
# 相机: head_cam, right_pinky_cam, right_thumb_cam, zed_obs
# 数值: right_leapv2, right_manus, head_right_tracker, right_tracker_world, zed_right_tracker, zed_pose, zed_ts
DEXWILD_CONFIG = RobotConfig(
    motors=[],

    cameras=[
        "head_cam",
        "right_pinky_cam",
        "right_thumb_cam",
        "zed_obs",
    ],

    camera_to_image_key={
        "head_cam": "head_cam",
        "right_pinky_cam": "right_pinky_cam",
        "right_thumb_cam": "right_thumb_cam",
        "zed_obs": "zed_obs",
    },
    umi_state_data_name=[],
    umi_action_data_name=[],
    
    # 数值字段
    demo_pose_sensors={
        "right_leapv2": "right_leapv2",
        "right_manus": "right_manus",
        "head_right_tracker": "head_right_tracker",
        "right_tracker_world": "right_tracker_world",
        "zed_right_tracker": "zed_right_tracker",
        "zed_pose": "zed_pose",
        "zed_ts": "zed_ts",
    },
    demo_pose_shapes={
        "right_leapv2": (17,),   # 18D - 时间戳列
        "right_manus": (70,),   # 71D - 时间戳列
        "head_right_tracker": (7,),  # 8D - 时间戳列
        "right_tracker_world": (7,),
        "zed_right_tracker": (7,),
        "zed_pose": (7,),
        "zed_ts": (1,),
    },
    is_bimanual=False,
    robot_prefixes=("right",),
)


# GenRobot MCAP 配置 - 10Kh-RealOmin-OpenData 双臂遥操作数据
# MCAP 数据结构:
#   robot0/sensor/camera0/compressed: (N, 1300, 1600, 3) - Robot0 相机图像 @30Hz
#   robot0/vio/eef_pose: (N, 7) - Robot0 末端位姿 [x,y,z,qx,qy,qz,qw] @30Hz
#   robot0/sensor/magnetic_encoder: (N, 1) - Robot0 夹爪开合 @50Hz
#   robot0/sensor/imu: (N, 6) - Robot0 IMU [acc_xyz, gyro_xyz] @200Hz
#   robot1/sensor/camera0/compressed: (N, 1300, 1600, 3) - Robot1 相机图像 @30Hz
#   robot1/vio/eef_pose: (N, 7) - Robot1 末端位姿 @30Hz
#   robot1/sensor/magnetic_encoder: (N, 1) - Robot1 夹爪开合 @50Hz
#   robot1/sensor/imu: (N, 6) - Robot1 IMU @200Hz
GENROBOT_MCAP_CONFIG = RobotConfig(
    motors=[],  # 不使用 motors 字段

    cameras=[
        "robot0_camera0",
        "robot1_camera0",
    ],

    camera_to_image_key={
        "robot0_camera0": "robot0_camera0",
        "robot1_camera0": "robot1_camera0",
    },
    umi_state_data_name=[],
    umi_action_data_name=[],
    
    # 使用 MCAP 原始字段名
    demo_pose_sensors={
        "robot0_eef_pose": "robot0_eef_pose",
        "robot0_gripper": "robot0_gripper",
        "robot0_imu": "robot0_imu",
        "robot1_eef_pose": "robot1_eef_pose",
        "robot1_gripper": "robot1_gripper",
        "robot1_imu": "robot1_imu",
    },
    demo_pose_shapes={
        "robot0_eef_pose": (7,),  # x, y, z, qx, qy, qz, qw
        "robot0_gripper": (1,),
        "robot0_imu": (6,),  # acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        "robot1_eef_pose": (7,),
        "robot1_gripper": (1,),
        "robot1_imu": (6,),
    },
    is_bimanual=True,
    robot_prefixes=("robot0", "robot1"),
    
    # ============================================================================
    # MCAP 专用配置
    # ============================================================================
    # 参考 topic (用作时间对齐基准)
    mcap_ref_topic="/robot0/sensor/camera0/compressed",
    
    # 相机 topic 映射: lerobot字段名 -> MCAP topic
    mcap_camera_topics={
        "robot0_camera0": "/robot0/sensor/camera0/compressed",
        "robot1_camera0": "/robot1/sensor/camera0/compressed",
    },
    
    # 数值数据 topic 映射: lerobot字段名 -> (MCAP topic, 插值类型, 数据形状)
    # 插值类型: "linear", "pose", "nearest"
    mcap_numeric_topics={
        "robot0_eef_pose": ("/robot0/vio/eef_pose", "pose", (7,)),
        "robot0_gripper": ("/robot0/sensor/magnetic_encoder", "linear", (1,)),
        "robot0_imu": ("/robot0/sensor/imu", "linear", (6,)),
        "robot1_eef_pose": ("/robot1/vio/eef_pose", "pose", (7,)),
        "robot1_gripper": ("/robot1/sensor/magnetic_encoder", "linear", (1,)),
        "robot1_imu": ("/robot1/sensor/imu", "linear", (6,)),
    },
)



# ExUMI 配置 - 包含触觉图像和 demo pose
# 数据结构:
#   camera0_rgb: (N, 224, 224, 3) - RGB 相机图像
#   tactile_combined: (N, 460, 680, 3) - 触觉图像
#   robot0_eef_pos: (N, 3) - 末端位置 (x, y, z)
#   robot0_eef_rot_axis_angle: (N, 3) - 末端旋转 (rx, ry, rz)
#   robot0_gripper_width: (N, 1) - 夹爪宽度
#   robot0_demo_start_pose: (N, 6) - demo 起始位姿
#   robot0_demo_end_pose: (N, 6) - demo 结束位姿
EXUMI_CONFIG = RobotConfig(
    motors=[
        "x",
        "y",
        "z",
        "rx",
        "ry",
        "rz",
        "gripper",
    ],

    cameras=[
        "camera0_rgb",
        # "tactile_combined",  # 触觉图像作为相机处理
    ],

    camera_to_image_key={
        "camera0_rgb": "camera0_rgb",
        # "tactile_combined": "tactile_combined",
    },
    umi_state_data_name=["eef_pos", "eef_rot_axis_angle", "gripper_width"],
    umi_action_data_name=["eef_pos", "eef_rot_axis_angle", "gripper_width"],
    
    # Demo pose 配置
    demo_pose_sensors={
        "demo_start_pose": "demo_start_pose",
        "demo_end_pose": "demo_end_pose",
    },
    demo_pose_shapes={
        "demo_start_pose": (6,),
        "demo_end_pose": (6,),
    },
)


ROBOT_CONFIGS = {
    # "Unitree_G1_Inspire": G1_INSPIRE_CONFIG,
    "Norm_EE": MV_UMI_CONFIG,
    "Touch_In_The_Wild": TOUCH_IN_THE_WILD_CONFIG,
    "UMI": UMI_CONFIG,
    "Bimanual_UMI": BIMANUAL_UMI_CONFIG,
    "ManiWAV": MANIWAV_CONFIG,
    "ViTaMIn": VITAMIN_CONFIG,
    "ManiForce": MANIFORCE_CONFIG,
    "LEGATO_SIM": LEGATO_SIM_CONFIG,
    "LEGATO_REAL": LEGATO_REAL_CONFIG,
    "DexUMI": DEXUMI_CONFIG,
    "FastUMI": FASTUMI_CONFIG,
    "DexWild": DEXWILD_CONFIG,
    "GenRobot_MCAP": GENROBOT_MCAP_CONFIG,
    "ExUMI": EXUMI_CONFIG,
}

