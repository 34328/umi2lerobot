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

import zarr
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from constants import ROBOT_CONFIGS

# 将父目录添加到 Python 搜索路径，以便导入 imagecodecs_numcodecs
sys.path.insert(0, str(Path(__file__).parent.parent))
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
    def __init__(self, data_dirs: Path, robot_type: str) -> None:

        assert data_dirs is not None, "Data directory cannot be None"
        assert robot_type is not None, "Robot type cannot be None"
        self.data_dirs = data_dirs
        self.robot_type = robot_type

        # Initialize paths and cache
        self._init_paths()
        self._init_cache()
        self.umi_state_data_name = ROBOT_CONFIGS[robot_type].umi_state_data_name
        self.umi_action_data_name = ROBOT_CONFIGS[robot_type].umi_action_data_name
        self.camera_to_image_key = ROBOT_CONFIGS[robot_type].camera_to_image_key

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
        
        # Load task description (customize based on zarr file name)
        task_name = os.path.basename(zarr_path).replace('.zarr.zip', '').replace('.zarr', '').replace('_', ' ')
        task = f"UMI {task_name} manipulation task"
        
        # Load camera images
        cameras = self._parse_images(root, start_idx, end_idx)
        
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
) -> LeRobotDataset:
    motors = ROBOT_CONFIGS[robot_type].motors
    cameras = ROBOT_CONFIGS[robot_type].cameras

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
            "shape": (224, 224, 3),
            "names": [
                "height",
                "width",
                "channel",
            ],
        }

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=60,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def populate_dataset(
    dataset: LeRobotDataset,
    raw_dir: Path,
    robot_type: str,
) -> LeRobotDataset:
    umi_dataset = UmiDataset(raw_dir, robot_type)
    for i in tqdm.tqdm(range(len(umi_dataset))):
        episode = umi_dataset.get_item(i)

        state = episode["state"]
        action = episode["action"]
        cameras = episode["cameras"]
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

            dataset.add_frame(frame, task=task)

        dataset.save_episode()

    return dataset


def umi_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,  
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    dataset = create_empty_dataset(
        repo_id,
        robot_type=robot_type,
        mode=mode,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        raw_dir,
        robot_type=robot_type,
    )

    if push_to_hub:
        dataset.push_to_hub(upload_large_folder=True)



if __name__ == "__main__":
    # 手动配置参数，方便调试
    # 直接指定 zarr 文件路径，而不是目录
    raw_dir = Path("/home/alex/unitree/UMI_2_LeRobot/mv-umi-dataset/bottles_rack_data.zarr.zip")
    repo_id = "bottles_rack"
    robot_type = "Norm_EE"
    push_to_hub = False
    mode = "video"  # or "image"
    
    umi_to_lerobot(
        raw_dir=raw_dir,
        repo_id=repo_id,
        robot_type=robot_type,
        push_to_hub=push_to_hub,
        mode=mode,
        dataset_config=DEFAULT_DATASET_CONFIG,
    )
    
    # 如果需要使用命令行参数，取消下面这行的注释
    # tyro.cli(json_to_lerobot)