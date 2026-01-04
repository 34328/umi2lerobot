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
}
