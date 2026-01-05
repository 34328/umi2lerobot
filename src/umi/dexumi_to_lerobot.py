import os
import sys
import re
import tqdm
import tyro
import dataclasses
import numpy as np
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lerobot" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "utils"))

import zarr
# from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from constants import ROBOT_CONFIGS

# 注册 imagecodecs_numcodecs 编解码器
from imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()



@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


class DexUmiDataset:
    """DexUMI 数据集加载器 - 处理按 episode 分离的 zarr 目录结构"""
    
    def __init__(self, data_dir: Path, robot_type: str, task_text: str | None = None) -> None:
        assert data_dir is not None, "Data directory cannot be None"
        assert robot_type is not None, "Robot type cannot be None"
        
        self.data_dir = Path(data_dir)
        self.robot_type = robot_type
        self.task_text = task_text
        
        # Initialize paths and cache
        self._init_episodes()
        
    def _init_episodes(self) -> None:
        """扫描目录，按自然数顺序读取 episode_* 目录"""
        self.episode_paths = []
        
        # 查找所有 episode_* 目录
        pattern = re.compile(r'^episode_(\d+)$')
        episode_dirs = []
        
        for item in os.listdir(self.data_dir):
            item_path = self.data_dir / item
            if item_path.is_dir():
                match = pattern.match(item)
                if match:
                    episode_idx = int(match.group(1))
                    episode_dirs.append((episode_idx, item_path))
        
        # 按自然数顺序排序
        episode_dirs.sort(key=lambda x: x[0])
        self.episode_paths = [path for _, path in episode_dirs]
        
        print(f"\n{'='*60}")
        print(f"Found {len(self.episode_paths)} episodes in {self.data_dir}")
        print(f"{'='*60}\n")
    
    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return len(self.episode_paths)
    
    def get_item(self, index: int) -> dict:
        """Get episode data by index."""
        episode_path = self.episode_paths[index]
        root = zarr.open(str(episode_path), mode='r')
        
        # 读取各字段数据
        # camera_0/rgb: (N, 400, 640, 3) uint8
        camera_rgb = root['camera_0']['rgb'][:]
        
        # fsr: (N, 2) float64 -> float32
        fsr = root['fsr'][:].astype(np.float32)
        
        # hand_action: (N, 6) float32 - 作为 action
        hand_action = root['hand_action'][:]
        
        # pose: (N, 6) float64 -> float32
        pose = root['pose'][:].astype(np.float32)
        
        # proprioception: (N, 6) float32 - 作为 state
        proprioception = root['proprioception'][:]
        
        episode_length = len(hand_action)
        
        # 获取任务描述
        if self.task_text:
            task = self.task_text
        else:
            task = f"DexUMI manipulation task"
        
        # 获取图像形状
        cam_height, cam_width = camera_rgb.shape[1:3]
        
        data_cfg = {
            "camera_names": ["camera_0"],
            "cam_height": cam_height,
            "cam_width": cam_width,
            "state_dim": proprioception.shape[1],  # 6
            "action_dim": hand_action.shape[1],    # 6
        }
        
        return {
            "episode_index": index,
            "episode_length": episode_length,
            "state": proprioception,        # observation.state: proprioception (6D)
            "action": hand_action,          # action: hand_action (6D)
            "fsr": fsr,                     # observation.fsr (2D)
            "pose": pose,                   # observation.pose (6D)
            "cameras": {"camera_0": camera_rgb},  # observation.images.camera_0
            "task": task,
            "data_cfg": data_cfg,
        }


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: Path | None = None,
    fps: int = 60,
    image_shape: tuple[int, int, int] = (400, 640, 3),
) -> LeRobotDataset:
    """创建 DexUMI 数据集结构"""
    
    # 从 ROBOT_CONFIGS 获取电机名称
    motors = ROBOT_CONFIGS[robot_type].motors
    
    # DexUMI 字段定义
    features = {
        # proprioception 作为 state (Inspire Hand 6个关节)
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        # hand_action 作为 action (Inspire Hand 6个关节)
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        # fsr 力敏电阻 (2D)
        "observation.fsr": {
            "dtype": "float32",
            "shape": (2,),
            "names": [
                ["fsr_0", "fsr_1"],
            ],
        },
        # pose 末端位姿 (笛卡尔空间)
        "observation.pose": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                ["x", "y", "z", "rx", "ry", "rz"],
            ],
        },
        # camera_0 相机图像
        "observation.images.camera_0": {
            "dtype": mode,
            "shape": image_shape,
            "names": [
                "height",
                "width",
                "channel",
            ],
        },
    }

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
        root=root,
    )


def populate_dataset(
    dataset: LeRobotDataset,
    raw_dir: Path,
    robot_type: str,
    task_text: str | None = None,
) -> LeRobotDataset:
    """填充 DexUMI 数据到 LeRobot 数据集"""
    dexumi_dataset = DexUmiDataset(raw_dir, robot_type, task_text)
    
    for i in tqdm.tqdm(range(len(dexumi_dataset)), desc="Converting episodes"):
        episode = dexumi_dataset.get_item(i)

        state = episode["state"]
        action = episode["action"]
        fsr = episode["fsr"]
        pose = episode["pose"]
        cameras = episode["cameras"]
        task = episode["task"]
        episode_length = episode["episode_length"]

        for j in range(episode_length):
            frame = {
                "observation.state": state[j],
                "action": action[j],
                "observation.fsr": fsr[j],
                "observation.pose": pose[j],
            }

            for camera_name, img_array in cameras.items():
                frame[f"observation.images.{camera_name}"] = img_array[j]

            dataset.add_frame(frame, task=task)

        dataset.save_episode()

    return dataset


def dexumi_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,
    project: str,
    text: str | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    fps: int = 60,
    image_shape: tuple[int, int, int] = (400, 640, 3),
):
    """
    Convert DexUMI dataset to LeRobot format.
    
    Args:
        raw_dir: Path to the date folder containing episode_* directories
        repo_id: Dataset name (subtask name)
        robot_type: Robot configuration type
        project: Project name for organizing datasets
        text: Human instruction or task description for the dataset
        push_to_hub: Whether to push to Hugging Face Hub
        mode: "video" or "image" format
        dataset_config: Dataset configuration
        fps: Frames per second for the dataset
        image_shape: Image shape as (height, width, channels)
    
    The dataset will be saved to: {lerbot_home}/{project}/{repo_id}
    """
    # 构建保存路径: {HF_LEROBOT_HOME}/{project}/{subtask}
    lerbot_home = Path("/mnt/raid0/UMI2Lerobot/lerobot")
    dataset_root = lerbot_home / project / repo_id
    
    print(f"\n{'='*60}")
    print(f"Dataset will be saved to: {dataset_root}")
    print(f"{'='*60}\n")
    
    if dataset_root.exists():
        raise FileExistsError(f"Dataset already exists at {dataset_root}. Please remove it manually if you want to recreate it.")

    dataset = create_empty_dataset(
        repo_id,
        robot_type=robot_type,
        mode=mode,
        dataset_config=dataset_config,
        root=dataset_root,
        fps=fps,
        image_shape=image_shape,
    )
    dataset = populate_dataset(
        dataset,
        raw_dir,
        robot_type=robot_type,
        task_text=text,
    )

    if push_to_hub:
        dataset.push_to_hub(upload_large_folder=True)


@dataclasses.dataclass
class ArgsConfig:
    """配置参数类 - 用于命令行参数或直接配置"""
    
    # 数据路径相关
    raw_dir: Path = Path("/mnt/raid0/UMI2Lerobot/rawData/DexUMI")
    """原始 DexUMI 数据根目录"""
    
    task_name: str = "inspire_egg_carton"
    """任务名称 (子文件夹名)"""
    
    date_folder: str = "eggbox_1_29_dataset"
    """日期数据文件夹名 (如 eggbox_1_29_dataset)"""
    
    project: str = "DexUMI"
    """项目名称 - 用于组织数据集"""
    
    subtask: str = "eggbox_1_29"
    """子任务名称 - 用作数据集名称"""
    
    # 机器人类型配置
    robot_type: str = "DexUMI"
    """机器人类型 - 必须在 constants.py 的 ROBOT_CONFIGS 中定义"""

    # 文本表述 
    text: str = "Pick up the egg carton with dexterous hand"
    """人类指令"""
    
    # 数据集配置
    fps: int = 60
    """数据集帧率 (frames per second)"""
    
    image_shape: tuple[int, int, int] = (400, 640, 3)
    """图像形状 (高度, 宽度, 通道数)"""
    
    mode: Literal["video", "image"] = "video"
    """存储模式: 'video' 或 'image'"""
    
    # 上传配置
    push_to_hub: bool = False
    """是否上传到 Hugging Face Hub"""
    
    # 高级配置
    use_videos: bool = True
    """是否使用视频格式存储"""
    
    tolerance_s: float = 0.0001
    """时间戳容差 (秒)"""
    
    image_writer_processes: int = 10
    """图像写入进程数"""
    
    image_writer_threads: int = 5
    """图像写入线程数"""
    
    video_backend: str | None = None
    """视频后端 (None 表示使用默认)"""


if __name__ == "__main__":
    # 使用 tyro 解析命令行参数，如果没有命令行参数则使用默认值
    config = tyro.cli(ArgsConfig)
    
    # 构建完整的数据路径: raw_dir / task_name / date_folder
    full_data_path = config.raw_dir / config.task_name / config.date_folder
    
    print(f"\n{'='*60}")
    print(f"DexUMI to LeRobot Converter")
    print(f"{'='*60}")
    print(f"Raw data path: {full_data_path}")
    print(f"Task: {config.task_name}")
    print(f"Date folder: {config.date_folder}")
    print(f"Output: {config.project}/{config.subtask}")
    print(f"{'='*60}\n")
    
    # 检查路径是否存在
    if not full_data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {full_data_path}")
    
    # 创建 DatasetConfig
    dataset_config = DatasetConfig(
        use_videos=config.use_videos,
        tolerance_s=config.tolerance_s,
        image_writer_processes=config.image_writer_processes,
        image_writer_threads=config.image_writer_threads,
        video_backend=config.video_backend,
    )
    
    # 执行转换
    dexumi_to_lerobot(
        raw_dir=full_data_path,
        repo_id=config.subtask,
        robot_type=config.robot_type,
        project=config.project,
        text=config.text,
        push_to_hub=config.push_to_hub,
        mode=config.mode,
        dataset_config=dataset_config,
        fps=config.fps,
        image_shape=config.image_shape,
    )