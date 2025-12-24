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



ROBOT_CONFIGS = {
    # "Unitree_G1_Inspire": G1_INSPIRE_CONFIG,
    "Norm_EE": MV_UMI_CONFIG,
    "Touch_In_The_Wild": TOUCH_IN_THE_WILD_CONFIG,
    "UMI": UMI_CONFIG,
}
