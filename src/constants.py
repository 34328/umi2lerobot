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


ROBOT_CONFIGS = {
    # "Unitree_G1_Inspire": G1_INSPIRE_CONFIG,
    "Norm_EE": MV_UMI_CONFIG,
    "Touch_In_The_Wild": TOUCH_IN_THE_WILD_CONFIG,
    "UMI": UMI_CONFIG,
    "Bimanual_UMI": BIMANUAL_UMI_CONFIG,
}
