import os
import sys
import tqdm
import tyro
import glob
import dataclasses
import shutil
import numpy as np
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "lerobot" / "src"))

import zarr
from lerobot.constants import HF_LEROBOT_HOME
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


class UmiDataset:
    def __init__(self, data_dirs: Path, robot_type: str, task_text: str | None = None) -> None:

        assert data_dirs is not None, "Data directory cannot be None"
        assert robot_type is not None, "Robot type cannot be None"
        self.data_dirs = data_dirs
        self.robot_type = robot_type
        self.task_text = task_text

        # Initialize paths and cache
        self._init_paths()
        self._init_cache()
        self.umi_state_data_name = ROBOT_CONFIGS[robot_type].umi_state_data_name
        self.umi_action_data_name = ROBOT_CONFIGS[robot_type].umi_action_data_name
        self.camera_to_image_key = ROBOT_CONFIGS[robot_type].camera_to_image_key
        self.audio_sensors = ROBOT_CONFIGS[robot_type].audio_sensors
        self.audio_shapes = ROBOT_CONFIGS[robot_type].audio_shapes

    def _init_paths(self) -> None:
        """Initialize zarr file paths."""
        self.zarr_paths = []
        
        # Convert Path to string for string operations
        data_path_str = str(self.data_dirs)
        
        # Check if data_dirs is a file or directory
        if os.path.isfile(data_path_str):
            # If it's a file, check if it's a zarr file
            if data_path_str.endswith('.zarr') or data_path_str.endswith('.zarr.zip'):
                self.zarr_paths.append(data_path_str)
            else:
                raise ValueError(f"File {data_path_str} is not a valid zarr file (.zarr or .zarr.zip)")
        elif os.path.isdir(data_path_str):
            # If it's a directory, find all .zarr or .zarr.zip files
            for file_path in glob.glob(os.path.join(data_path_str, "*.zarr")):
                self.zarr_paths.append(file_path)
            for file_path in glob.glob(os.path.join(data_path_str, "*.zarr.zip")):
                self.zarr_paths.append(file_path)
        else:
            raise ValueError(f"Path {data_path_str} is neither a file nor a directory")
        
        if not self.zarr_paths:
            raise ValueError(f"No zarr files found in {data_path_str}")
        
        self.zarr_paths = sorted(self.zarr_paths)
        print(f"Found {len(self.zarr_paths)} zarr file(s): {[os.path.basename(p) for p in self.zarr_paths]}")

    def _init_cache(self) -> None:
        """Load zarr data and parse episodes."""
        self.episodes = []
        self.zarr_roots = {}  # Keep zarr files open for access
        
        for zarr_path in tqdm.tqdm(self.zarr_paths, desc="Loading Zarr files"):
            root = zarr.open(zarr_path, mode='r')
            # root= root["markers_placement_data.zarr"]
            self.zarr_roots[zarr_path] = root  # Store the opened zarr root
            
            # Get episode boundaries
            episode_ends = root['meta']['episode_ends'][:]
            episode_starts = np.concatenate([[0], episode_ends[:-1]])
            
            print(f"\n  {os.path.basename(zarr_path)}:")
            print(f"    - Total frames: {episode_ends[-1]}")
            print(f"    - Episodes: {len(episode_ends)}")
            print(f"    - Episode lengths: min={np.min(np.diff(np.concatenate([[0], episode_ends])))}, "
                  f"max={np.max(np.diff(np.concatenate([[0], episode_ends])))}, "
                  f"mean={np.mean(np.diff(np.concatenate([[0], episode_ends]))):.1f}")
            
            # Store each episode's metadata
            for ep_idx, (start_idx, end_idx) in enumerate(zip(episode_starts, episode_ends)):
                self.episodes.append({
                    'zarr_path': zarr_path,
                    'start_idx': int(start_idx),
                    'end_idx': int(end_idx),
                    'length': int(end_idx - start_idx),
                    'zarr_episode_idx': ep_idx  # Episode index within this zarr file
                })
        
        print(f"\n==> Total: {len(self.episodes)} episodes from {len(self.zarr_paths)} zarr file(s)")

    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return len(self.episodes)

    def _extract_umi_data(self, root, start_idx: int, end_idx: int, data_names: list[str]) -> np.ndarray:
        """
        Extract and concatenate UMI data for specified data names.
        
        Args:
            root: Zarr root
            start_idx: Episode start index
            end_idx: Episode end index
            data_names: List of data keys (e.g., ['eef_pos', 'eef_rot_axis_angle', 'gripper_width'])
        
        Returns:
            Concatenated numpy array of shape (episode_length, total_dim)
        """
        data_arrays = []
        
        for data_name in data_names:
            key = f"robot0_{data_name}"
            data = root['data'][key][start_idx:end_idx]
            # Ensure 2D shape
            if len(data.shape) == 1:
                data = data[:, np.newaxis]
            data_arrays.append(data)
        
        return np.concatenate(data_arrays, axis=1).astype(np.float32)

    def _parse_images(self, root, start_idx: int, end_idx: int) -> dict[str, np.ndarray]:
        """Load and stack images for all cameras in the episode."""
        images = {}
        
        for camera_key, image_key in self.camera_to_image_key.items():
            # Map from camera name to zarr key
            zarr_key = image_key  # e.g., 'camera0_rgb', 'camera1_rgb'
            
            if zarr_key in root['data']:
                # Load all frames for this episode
                frames = root['data'][zarr_key][start_idx:end_idx]
                images[image_key] = frames
        
        return images

    def _parse_audio(self, root, start_idx: int, end_idx: int) -> dict[str, np.ndarray]:
        """Load audio data for all audio sensors in the episode."""
        audio_data = {}
        
        for zarr_key, audio_key in self.audio_sensors.items():
            if zarr_key in root['data']:
                # Load all frames for this episode
                data = root['data'][zarr_key][start_idx:end_idx]
                audio_data[audio_key] = data.astype(np.float32)
        
        return audio_data

    def get_item(self, index: int) -> dict:
        """Get a training sample from the dataset."""
        episode_info = self.episodes[index]
        zarr_path = episode_info['zarr_path']
        root = self.zarr_roots[zarr_path]  # Use the cached zarr root
        start_idx = episode_info['start_idx']
        end_idx = episode_info['end_idx']
        
        # Extract action data (current timestep t)
        action = self._extract_umi_data(root, start_idx, end_idx, self.umi_action_data_name)
        
        # Extract state data (previous timestep t-1)
        # State at time t is the action at time t-1
        # For the first frame, we use the same action as state (no previous action available)
        state = np.zeros_like(action)
        state[0] = action[0]  # First state = first action
        state[1:] = action[:-1]  # State[t] = Action[t-1] for t > 0
        
        episode_length = end_idx - start_idx
        state_dim = state.shape[1]
        action_dim = action.shape[1]
        
        # Use provided task description if available; otherwise fall back to file-derived text
        if self.task_text:
            task = self.task_text
        else:
            task_name = os.path.basename(zarr_path).replace('.zarr.zip', '').replace('.zarr', '').replace('_', ' ')
            task = f"{task_name} manipulation task"
        
        # Load camera images
        cameras = self._parse_images(root, start_idx, end_idx)
        
        # Load audio data
        audio = self._parse_audio(root, start_idx, end_idx)
        
        # Extract camera configuration
        if cameras:
            cam_height, cam_width = next(iter(cameras.values())).shape[1:3]
        else:
            cam_height, cam_width = 224, 224
        
        data_cfg = {
            "camera_names": list(cameras.keys()),
            "cam_height": cam_height,
            "cam_width": cam_width,
            "state_dim": state_dim,
            "action_dim": action_dim,
        }
        
        return {
            "episode_index": index,
            "episode_length": episode_length,
            "state": state,
            "action": action,
            "cameras": cameras,
            "audio": audio,
            "task": task,
            "data_cfg": data_cfg,
        }


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: Path | None = None,
    fps: int = 60,
    image_shape: tuple[int, int, int] = (224, 224, 3),
) -> LeRobotDataset:
    motors = ROBOT_CONFIGS[robot_type].motors
    cameras = ROBOT_CONFIGS[robot_type].cameras
    audio_sensors = ROBOT_CONFIGS[robot_type].audio_sensors
    audio_shapes = ROBOT_CONFIGS[robot_type].audio_shapes

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": image_shape,
            "names": [
                "height",
                "width",
                "channel",
            ],
        }

    # 添加音频传感器特征
    for audio_key, audio_name in audio_sensors.items():
        audio_shape = audio_shapes.get(audio_key, (800,))
        features[f"observation.audio.{audio_name}"] = {
            "dtype": "float32",
            "shape": audio_shape,
            "names": [
                "samples",
            ],
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
    umi_dataset = UmiDataset(raw_dir, robot_type, task_text)
    for i in tqdm.tqdm(range(len(umi_dataset))):
        episode = umi_dataset.get_item(i)

        state = episode["state"]
        action = episode["action"]
        cameras = episode["cameras"]
        audio = episode["audio"]
        task = episode["task"]
        episode_length = episode["episode_length"]

        num_frames = episode_length
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            for camera, img_array in cameras.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            # 添加音频数据
            for audio_key, audio_array in audio.items():
                frame[f"observation.audio.{audio_key}"] = audio_array[i]

            dataset.add_frame(frame, task=task)

        dataset.save_episode()

    return dataset


def umi_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,
    project: str,
    text: str | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    fps: int = 60,
    image_shape: tuple[int, int, int] = (224, 224, 3),
):
    """
    Convert UMI dataset to LeRobot format.
    
    Args:
        raw_dir: Path to the raw zarr file or directory
        repo_id: Dataset name (subtask name)
        robot_type: Robot configuration type
        project: Project name for organizing datasets
    text: Human instruction or task description for the dataset
        push_to_hub: Whether to push to Hugging Face Hub
        mode: "video" or "image" format
        dataset_config: Dataset configuration
        fps: Frames per second for the dataset
        image_shape: Image shape as (height, width, channels), e.g., (224, 224, 3)
    
    The dataset will be saved to: {HF_LEROBOT_HOME}/{project}/{repo_id}
    Example: ~/.cache/huggingface/lerobot/mv-umi/bottles_rack
    """
    # 构建保存路径: {HF_LEROBOT_HOME}/{project}/{subtask}
    dataset_root = HF_LEROBOT_HOME / project / repo_id
    
    print(f"\n{'='*60}")
    print(f"Dataset will be saved to: {dataset_root}")
    print(f"{'='*60}\n")
    
    if dataset_root.exists():
        raise FileExistsError(f"Dataset already exists at {dataset_root}. Please remove it manually if you want to recreate it.")

    dataset = create_empty_dataset(
        repo_id,
        robot_type=robot_type,
        mode=mode,
        has_effort=False,
        has_velocity=False,
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
    raw_dir: Path = Path("/home/unitree/桌面/umi2lerobot/rawData/ManiWAV/Whiteboard Shape Wipe/replay_buffer.zarr.zip")
    """原始 zarr 文件路径或目录"""
    
    project: str = "ManiWAV"
    """项目名称 - 用于组织数据集"""
    
    subtask: str = "Whiteboard Shape Wipe"
    """子任务名称 - 用作数据集名称"""
    
    # EEF 6Dof + Vision 信息配置
    robot_type: str = "ManiWAV"
    """机器人类型 - 必须在 constants.py 的 ROBOT_CONFIGS 中定义"""

    # 文本表述 
    text: str = "Wipe the words on the whiteboard clean"
    """人类指令"""
    
    # 数据集配置
    fps: int = 60
    """数据集帧率 (frames per second)"""
    
    image_shape: tuple[int, int, int] = (224, 224, 3)
    """图像形状 (高度, 宽度, 通道数)，例如 (224, 224, 3)"""
    
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
    
    # 创建 DatasetConfig
    dataset_config = DatasetConfig(
        use_videos=config.use_videos,
        tolerance_s=config.tolerance_s,
        image_writer_processes=config.image_writer_processes,
        image_writer_threads=config.image_writer_threads,
        video_backend=config.video_backend,
    )
    
    # 执行转换
    umi_to_lerobot(
        raw_dir=config.raw_dir,
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