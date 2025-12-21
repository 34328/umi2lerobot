import dataclasses


@dataclasses.dataclass(frozen=True)
class RobotConfig:
    motors: list[str]
    cameras: list[str]
    camera_to_image_key: dict[str, str]
    umi_state_data_name: list[str]
    umi_action_data_name: list[str]



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


ROBOT_CONFIGS = {
    # "Unitree_G1_Inspire": G1_INSPIRE_CONFIG,
    "Norm_EE": UMI_CONFIG,
}
