import gc
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
    image_writer_processes: int = 2
    image_writer_threads: int = 2
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


class ManiForceDataset:
    """ManiForce 数据集加载器，适配 ManiForce zarr 格式。
    
    数据结构:
        /data
            action: (N, 8) float32
            state: (N, 7) float32
            pose_wrt_start: (N, 7) float32
            handeye_cam_1: (N, 800, 1280, 3) uint8
            handeye_cam_2: (N, 480, 640, 3) uint8
            img_timestamps: (N,) float64
        /meta
            episode_ends: (num_episodes,) int64
    """
    
    def __init__(self, data_dirs: Path, robot_type: str, task_text: str | None = None) -> None:

        assert data_dirs is not None, "Data directory cannot be None"
        assert robot_type is not None, "Robot type cannot be None"
        self.data_dirs = data_dirs
        self.robot_type = robot_type
        self.task_text = task_text

        # Initialize paths and cache
        self._init_paths()
        self._init_cache()
        self.camera_to_image_key = ROBOT_CONFIGS[robot_type].camera_to_image_key
        self.demo_pose_sensors = ROBOT_CONFIGS[robot_type].demo_pose_sensors
        self.demo_pose_shapes = ROBOT_CONFIGS[robot_type].demo_pose_shapes

    def _init_paths(self) -> None:
        """Initialize zarr file paths."""
        self.zarr_paths = []
        
        # Convert Path to string for string operations
        data_path_str = str(self.data_dirs)
        
        # .zarr 可以是目录，.zarr.zip 是文件
        if data_path_str.endswith('.zarr.zip') and os.path.isfile(data_path_str):
            # 单个 zarr.zip 文件
            self.zarr_paths.append(data_path_str)
        elif data_path_str.endswith('.zarr') and os.path.isdir(data_path_str):
            # 单个 zarr 目录
            self.zarr_paths.append(data_path_str)
        elif os.path.isdir(data_path_str):
            # 目录中查找 .zarr 目录和 .zarr.zip 文件
            for item in os.listdir(data_path_str):
                item_path = os.path.join(data_path_str, item)
                if item.endswith('.zarr') and os.path.isdir(item_path):
                    self.zarr_paths.append(item_path)
                elif item.endswith('.zarr.zip') and os.path.isfile(item_path):
                    self.zarr_paths.append(item_path)
        else:
            raise ValueError(f"Path {data_path_str} is not a valid zarr path")
        
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

    def _parse_images(self, root, start_idx: int, end_idx: int) -> dict[str, np.ndarray]:
        """Load and stack images for all cameras in the episode."""
        images = {}
        
        for camera_key, zarr_key in self.camera_to_image_key.items():
            # camera_key: lerobot中的字段名
            # zarr_key: zarr中的数据键名
            
            if zarr_key in root['data']:
                # Load all frames for this episode
                frames = root['data'][zarr_key][start_idx:end_idx]
                images[camera_key] = frames
        
        return images

    def _parse_demo_poses(self, root, start_idx: int, end_idx: int) -> dict[str, np.ndarray]:
        """Load demo poses for all demo pose sensors in the episode.
        
        ManiForce 格式: 直接使用 pose_wrt_start 字段，不带 robot0_ 前缀
        """
        demo_poses = {}
        
        for pose_name, pose_key in self.demo_pose_sensors.items():
            # ManiForce 直接使用字段名，不带 robot0_ 前缀
            if pose_name in root['data']:
                data = root['data'][pose_name][start_idx:end_idx]
                demo_poses[pose_key] = data.astype(np.float32)
        
        return demo_poses

    def get_item(self, index: int) -> dict:
        """Get a training sample from the dataset."""
        episode_info = self.episodes[index]
        zarr_path = episode_info['zarr_path']
        root = self.zarr_roots[zarr_path]
        start_idx = episode_info['start_idx']
        end_idx = episode_info['end_idx']
        
        # ManiForce 直接读取 action 和 state 字段，保持原样
        action = root['data']['action'][start_idx:end_idx].astype(np.float32)
        state = root['data']['state'][start_idx:end_idx].astype(np.float32)
        
        episode_length = end_idx - start_idx
        state_dim = state.shape[1]
        action_dim = action.shape[1]
        
        # Use provided task description if available
        if self.task_text:
            task = self.task_text
        else:
            task_name = os.path.basename(zarr_path).replace('.zarr.zip', '').replace('.zarr', '').replace('_', ' ')
            task = f"{task_name} manipulation task"
        
        # Load camera images
        cameras = self._parse_images(root, start_idx, end_idx)
        
        # Load demo poses (pose_wrt_start)
        demo_poses = self._parse_demo_poses(root, start_idx, end_idx)
        
        # Extract camera configuration
        if cameras:
            first_cam = next(iter(cameras.values()))
            cam_height, cam_width = first_cam.shape[1:3]
        else:
            cam_height, cam_width = 480, 640
        
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
            "demo_poses": demo_poses,
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
    camera_shapes: dict[str, tuple[int, int, int]] | None = None,
    state_dim: int = 7,
    action_dim: int = 8,
) -> LeRobotDataset:
    """创建空的 LeRobot 数据集。
    
    Args:
        repo_id: 数据集 ID
        robot_type: 机器人类型
        mode: 存储模式 "video" 或 "image"
        has_velocity: 是否包含速度
        has_effort: 是否包含力矩
        dataset_config: 数据集配置
        root: 保存路径
        fps: 帧率
        camera_shapes: 各相机的图像形状，格式 {camera_name: (height, width, channels)}
                      如果为 None，使用默认值
        state_dim: state 维度 (默认 7)
        action_dim: action 维度 (默认 8)
    """
    cameras = ROBOT_CONFIGS[robot_type].cameras
    demo_pose_sensors = ROBOT_CONFIGS[robot_type].demo_pose_sensors
    demo_pose_shapes = ROBOT_CONFIGS[robot_type].demo_pose_shapes
    
    # 默认相机形状
    default_camera_shapes = {
        "handeye_cam_1": (800, 1280, 3),
        "handeye_cam_2": (480, 640, 3),
    }
    if camera_shapes is None:
        camera_shapes = default_camera_shapes

    # state: 7维 (x, y, z, qx, qy, qz, qw)
    state_names = ["x", "y", "z", "qx", "qy", "qz", "qw"][:state_dim]
    # action: 8维 (x, y, z, qx, qy, qz, qw, gripper)
    action_names = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"][:action_dim]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [
                state_names,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [
                action_names,
            ],
        },
    }


    for cam in cameras:
        shape = camera_shapes.get(cam, (480, 640, 3))
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": shape,
            "names": [
                "height",
                "width",
                "channel",
            ],
        }

    # 添加 pose_wrt_start 特征 (作为状态输入)
    for pose_name, pose_key in demo_pose_sensors.items():
        shape = demo_pose_shapes.get(pose_name, (7,))
        features[f"observation.state.{pose_key}"] = {
            "dtype": "float32",
            "shape": shape,
            "names": [
                ["x", "y", "z", "qx", "qy", "qz", "qw"][:shape[0]],
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
    maniforce_dataset = ManiForceDataset(raw_dir, robot_type, task_text)
    for ep_idx in tqdm.tqdm(range(len(maniforce_dataset))):
        episode = maniforce_dataset.get_item(ep_idx)

        state = episode["state"]
        action = episode["action"]
        cameras = episode["cameras"]
        demo_poses = episode["demo_poses"]
        task = episode["task"]
        episode_length = episode["episode_length"]

        for frame_idx in range(episode_length):
            frame = {
                "observation.state": state[frame_idx],
                "action": action[frame_idx],
            }

            for camera, img_array in cameras.items():
                frame[f"observation.images.{camera}"] = img_array[frame_idx]

            # 添加 pose_wrt_start
            for pose_key, pose_array in demo_poses.items():
                frame[f"observation.state.{pose_key}"] = pose_array[frame_idx]

            dataset.add_frame(frame, task=task)

        dataset.save_episode()
        
        # 释放内存
        del episode, state, action, cameras, demo_poses
        gc.collect()

    return dataset


def maniforce_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,
    project: str,
    text: str | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    fps: int = 60,
    camera_shapes: dict[str, tuple[int, int, int]] | None = None,
):
    """
    Convert ManiForce dataset to LeRobot format.
    
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
        camera_shapes: Dict of camera names to shapes, e.g., 
                       {"handeye_cam_1": (800, 1280, 3), "handeye_cam_2": (480, 640, 3)}
    
    The dataset will be saved to: {HF_LEROBOT_HOME}/{project}/{repo_id}
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
        camera_shapes=camera_shapes,
        state_dim=7,   # state: (N, 7)
        action_dim=8,  # action: (N, 8)
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
    """ManiForce 数据集转换配置参数"""
    
    # 数据路径相关
    raw_dir: Path = Path("/home/unitree/桌面/umi2lerobot/rawData/ManipForce/Open_lid.zarr")
    """原始 zarr 文件路径或目录"""
    
    project: str = "ManiForce"
    """项目名称 - 用于组织数据集"""
    
    subtask: str = "Open_lid"
    """子任务名称 - 用作数据集名称"""
    
    # 机器人配置
    robot_type: str = "ManiForce"
    """机器人类型 - 必须在 constants.py 的 ROBOT_CONFIGS 中定义"""

    # 文本表述 
    text: str = "Open the lid"
    """人类指令"""
    
    # 数据集配置
    fps: int = 30
    """数据集帧率 (frames per second)"""
    
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
    
    image_writer_processes: int = 2
    """图像写入进程数（减少可降低内存占用）"""
    
    image_writer_threads: int = 2
    """图像写入线程数（减少可降低内存占用）"""
    
    video_backend: str | None = None
    """视频后端 (None 表示使用默认)"""


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    
    dataset_config = DatasetConfig(
        use_videos=config.use_videos,
        tolerance_s=config.tolerance_s,
        image_writer_processes=config.image_writer_processes,
        image_writer_threads=config.image_writer_threads,
        video_backend=config.video_backend,
    )
    
    # ManiForce 相机形状配置
    camera_shapes = {
        "handeye_cam_1": (800, 1280, 3),
        "handeye_cam_2": (480, 640, 3),
    }
    
    maniforce_to_lerobot(
        raw_dir=config.raw_dir,
        repo_id=config.subtask,
        robot_type=config.robot_type,
        project=config.project,
        text=config.text,
        push_to_hub=config.push_to_hub,
        mode=config.mode,
        dataset_config=dataset_config,
        fps=config.fps,
        camera_shapes=camera_shapes,
    )