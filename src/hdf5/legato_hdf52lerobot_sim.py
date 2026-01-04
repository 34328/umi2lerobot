"""
LEGATO HDF5 to LeRobot 格式转换脚本

用于将 LEGATO 数据集的 HDF5 格式转换为 LeRobot 格式。
支持双手腕灰度图像、多种观测数据和 train/valid mask 信息。

HDF5 数据结构:
├── data/
│   ├── demo_1/
│   │   ├── actions (N, 7) float32
│   │   ├── dones (N,) uint8
│   │   ├── rewards (N,) float32
│   │   └── obs/
│   │       ├── left_gray (N, 128, 128, 1) uint8
│   │       ├── right_gray (N, 128, 128, 1) uint8
│   │       ├── delta_eulers (N, 6) float32
│   │       ├── delta_positions (N, 6) float32
│   │       ├── delta_quaternions (N, 8) float32
│   │       ├── quaternions (N, 8) float32
│   │       └── position_diffs (N, 6) float32
│   └── demo_N/
└── mask/
    ├── train (M,) |S8
    └── valid (K,) |S8
"""

import os
import sys
import tqdm
import tyro
import json
import dataclasses
import numpy as np
from pathlib import Path
from typing import Literal

import h5py

# 修正路径: 从 src/hdf5/ 到项目根目录，再到 lerobot/src
_project_root = Path(__file__).parent.parent.parent  # umi2lerobot/
sys.path.insert(0, str(_project_root))  # umi2lerobot/ (项目根目录，包含 utils/)
sys.path.insert(0, str(_project_root / "src"))  # umi2lerobot/src
sys.path.insert(0, str(_project_root / "lerobot" / "src"))  # umi2lerobot/lerobot/src

from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from utils.constants import ROBOT_CONFIGS


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


class LegatoHDF5Dataset:
    """LEGATO HDF5 数据集解析器"""
    
    def __init__(self, hdf5_path: Path, robot_type: str, task_text: str | None = None) -> None:
        assert hdf5_path is not None, "HDF5 path cannot be None"
        assert robot_type is not None, "Robot type cannot be None"
        
        self.hdf5_path = hdf5_path
        self.robot_type = robot_type
        self.task_text = task_text
        
        # 加载配置
        self.config = ROBOT_CONFIGS[robot_type]
        self.camera_to_image_key = self.config.camera_to_image_key
        self.demo_pose_sensors = self.config.demo_pose_sensors
        
        # 打开 HDF5 文件
        self.hdf5_file = h5py.File(hdf5_path, 'r')
        
        # 初始化 episodes
        self._init_episodes()
        
        # 解析 mask 信息
        self._parse_mask()
    
    def _init_episodes(self) -> None:
        """初始化 episode 列表，按自然数顺序排列"""
        data_group = self.hdf5_file['data']
        
        # 获取所有 demo 名称，按自然数顺序排序
        demo_names = [k for k in data_group.keys() if k.startswith('demo_')]
        self.demo_names = sorted(demo_names, key=lambda x: int(x.split('_')[1]))
        
        # 统计信息
        lengths = []
        for demo_name in self.demo_names:
            demo = data_group[demo_name]
            length = demo['actions'].shape[0]
            lengths.append(length)
        
        print(f"\n{'='*60}")
        print(f"HDF5 File: {self.hdf5_path.name}")
        print(f"  - Total demos: {len(self.demo_names)}")
        print(f"  - Episode lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
        print(f"{'='*60}\n")
    
    def _parse_mask(self) -> None:
        """解析 train/valid mask 信息"""
        self.train_demos = set()
        self.valid_demos = set()
        
        if 'mask' in self.hdf5_file:
            mask_group = self.hdf5_file['mask']
            
            if 'train' in mask_group:
                train_mask = mask_group['train'][:]
                self.train_demos = set(
                    x.decode() if isinstance(x, bytes) else x 
                    for x in train_mask
                )
            
            if 'valid' in mask_group:
                valid_mask = mask_group['valid'][:]
                self.valid_demos = set(
                    x.decode() if isinstance(x, bytes) else x 
                    for x in valid_mask
                )
        
        print(f"Mask info: {len(self.train_demos)} train, {len(self.valid_demos)} valid demos")
    
    def get_mask_info(self) -> dict:
        """获取 mask 信息，用于保存到元数据"""
        return {
            "train": sorted(list(self.train_demos)),
            "valid": sorted(list(self.valid_demos)),
        }
    
    def __len__(self) -> int:
        """返回 episode 数量"""
        return len(self.demo_names)
    
    def get_item(self, index: int) -> dict:
        """获取指定索引的 episode 数据"""
        demo_name = self.demo_names[index]
        demo = self.hdf5_file['data'][demo_name]
        
        # 读取 actions
        actions = demo['actions'][:].astype(np.float32)
        episode_length = actions.shape[0]
        
        # 构建 state: state[t] = action[t-1], state[0] = action[0]
        state = np.zeros_like(actions)
        state[0] = actions[0]
        state[1:] = actions[:-1]
        
        # 读取 dones 和 rewards
        dones = demo['dones'][:].astype(np.uint8)
        rewards = demo['rewards'][:].astype(np.float32)
        
        # 读取观测数据
        obs = demo['obs']
        
        # 读取图像
        cameras = {}
        for camera_key in self.camera_to_image_key.keys():
            if camera_key in obs:
                # 灰度图: (N, 128, 128, 1) -> 转换为 3 通道 (N, 128, 128, 3)
                gray_img = obs[camera_key][:]
                # 将单通道灰度图复制到 3 通道
                cameras[camera_key] = np.repeat(gray_img, 3, axis=-1)
        
        # 读取 pose 相关观测数据
        pose_data = {}
        for hdf5_key, lerobot_key in self.demo_pose_sensors.items():
            if hdf5_key in obs:
                pose_data[lerobot_key] = obs[hdf5_key][:].astype(np.float32)
        
        # 任务描述
        if self.task_text:
            task = self.task_text
        else:
            task_name = self.hdf5_path.stem.replace('_', ' ')
            task = f"{task_name} manipulation task"
        
        # 提取相机配置 (转换后为 3 通道)
        if cameras:
            first_cam = next(iter(cameras.values()))
            cam_height, cam_width, cam_channels = first_cam.shape[1:4]
        else:
            cam_height, cam_width, cam_channels = 128, 128, 3
        
        # 判断是否为 train/valid
        is_train = demo_name in self.train_demos
        is_valid = demo_name in self.valid_demos
        
        return {
            "episode_index": index,
            "demo_name": demo_name,
            "episode_length": episode_length,
            "state": state,
            "action": actions,
            "dones": dones,
            "rewards": rewards,
            "cameras": cameras,
            "pose_data": pose_data,
            "task": task,
            "is_train": is_train,
            "is_valid": is_valid,
            "data_cfg": {
                "camera_names": list(cameras.keys()),
                "cam_height": cam_height,
                "cam_width": cam_width,
                "cam_channels": cam_channels,
                "state_dim": state.shape[1],
                "action_dim": actions.shape[1],
            },
        }
    
    def close(self):
        """关闭 HDF5 文件"""
        if self.hdf5_file:
            self.hdf5_file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: Path | None = None,
    fps: int = 10,
    image_shape: tuple[int, int, int] = (128, 128, 3),  # LeRobot 要求 3 通道
) -> LeRobotDataset:
    """创建空的 LeRobot 数据集"""
    
    config = ROBOT_CONFIGS[robot_type]
    motors = config.motors
    cameras = config.cameras
    demo_pose_sensors = config.demo_pose_sensors
    demo_pose_shapes = config.demo_pose_shapes
    
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
    }
    
    # 添加相机图像特征（灰度图）
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }
    
    # 添加 pose 观测数据特征
    for sensor_name, shape in demo_pose_shapes.items():
        features[f"observation.{sensor_name}"] = {
            "dtype": "float32",
            "shape": shape,
            "names": None,
        }
    
    # 添加 dones 和 rewards 特征
    features["observation.dones"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }
    features["observation.rewards"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
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
    hdf5_path: Path,
    robot_type: str,
    task_text: str | None = None,
) -> tuple[LeRobotDataset, dict]:
    """填充 LeRobot 数据集"""
    
    with LegatoHDF5Dataset(hdf5_path, robot_type, task_text) as legato_dataset:
        mask_info = legato_dataset.get_mask_info()
        
        for j in tqdm.tqdm(range(len(legato_dataset)), desc="Converting episodes"):
            episode = legato_dataset.get_item(j)
            
            state = episode["state"]
            action = episode["action"]
            cameras = episode["cameras"]
            pose_data = episode["pose_data"]
            dones = episode["dones"]
            rewards = episode["rewards"]
            task = episode["task"]
            episode_length = episode["episode_length"]
            
            for i in range(episode_length):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "observation.dones": np.array([dones[i]], dtype=np.int64),
                    "observation.rewards": np.array([rewards[i]], dtype=np.float32),
                }
                
                # 添加图像
                for camera, img_array in cameras.items():
                    frame[f"observation.images.{camera}"] = img_array[i]
                
                # 添加 pose 观测数据
                for pose_name, pose_array in pose_data.items():
                    frame[f"observation.{pose_name}"] = pose_array[i]
                
                dataset.add_frame(frame, task=task)
            
            dataset.save_episode()
    
    return dataset, mask_info


def save_mask_info(dataset_root: Path, mask_info: dict) -> None:
    """保存 mask 信息到数据集目录"""
    mask_file = dataset_root / "mask.json"
    with open(mask_file, 'w') as f:
        json.dump(mask_info, f, indent=2)
    print(f"Mask info saved to: {mask_file}")


def legato_hdf5_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,
    project: str,
    text: str | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    fps: int = 10,
    image_shape: tuple[int, int, int] = (128, 128, 3),  # LeRobot 要求 3 通道
):
    """
    将 LEGATO HDF5 数据集转换为 LeRobot 格式。
    
    Args:
        raw_dir: HDF5 文件路径
        repo_id: 数据集名称（子任务名称）
        robot_type: 机器人配置类型
        project: 项目名称
        text: 任务描述
        push_to_hub: 是否上传到 Hugging Face Hub
        mode: "video" 或 "image" 格式
        dataset_config: 数据集配置
        fps: 帧率
        image_shape: 图像形状 (高度, 宽度, 通道数)
    """
    # 构建保存路径
    dataset_root = HF_LEROBOT_HOME / project / repo_id
    
    print(f"\n{'='*60}")
    print(f"Converting LEGATO HDF5 to LeRobot format")
    print(f"Input: {raw_dir}")
    print(f"Output: {dataset_root}")
    print(f"{'='*60}\n")
    
    if dataset_root.exists():
        raise FileExistsError(
            f"Dataset already exists at {dataset_root}. "
            "Please remove it manually if you want to recreate it."
        )
    
    # 创建空数据集
    dataset = create_empty_dataset(
        repo_id,
        robot_type=robot_type,
        mode=mode,
        dataset_config=dataset_config,
        root=dataset_root,
        fps=fps,
        image_shape=image_shape,
    )
    
    # 填充数据集
    dataset, mask_info = populate_dataset(
        dataset,
        raw_dir,
        robot_type=robot_type,
        task_text=text,
    )
    
    # 保存 mask 信息
    save_mask_info(dataset_root, mask_info)
    
    print(f"\n{'='*60}")
    print(f"Conversion completed!")
    print(f"Dataset saved to: {dataset_root}")
    print(f"{'='*60}\n")
    
    if push_to_hub:
        dataset.push_to_hub(upload_large_folder=True)


@dataclasses.dataclass
class ArgsConfig:
    """配置参数类"""
    
    # 数据路径相关
    raw_dir: Path = Path("/home/unitree/桌面/umi2lerobot/rawData/LEGATO/ladle_reorganization.hdf5")
    """原始 HDF5 文件路径"""
    
    project: str = "LEGATO"
    """项目名称"""
    
    subtask: str = "ladle_reorganization"
    """子任务名称"""
    
    robot_type: str = "LEGATO_SIM"
    """机器人类型"""
    
    text: str = "Put the ladle on the plate"
    """任务描述"""
    
    fps: int = 30
    """帧率"""
    
    image_shape: tuple[int, int, int] = (128, 128, 3)
    """图像形状 (高度, 宽度, 通道数) - LeRobot 要求 3 通道，灰度图会自动复制到 RGB"""
    
    mode: Literal["video", "image"] = "video"
    """存储模式: 'video' 或 'image'"""
    
    push_to_hub: bool = False
    """是否上传到 Hugging Face Hub"""
    
    # 高级配置
    use_videos: bool = True
    """是否使用视频格式存储"""
    
    tolerance_s: float = 0.0001
    """时间戳容差"""
    
    image_writer_processes: int = 10
    """图像写入进程数"""
    
    image_writer_threads: int = 5
    """图像写入线程数"""
    
    video_backend: str | None = None
    """视频后端"""


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    
    dataset_config = DatasetConfig(
        use_videos=config.use_videos,
        tolerance_s=config.tolerance_s,
        image_writer_processes=config.image_writer_processes,
        image_writer_threads=config.image_writer_threads,
        video_backend=config.video_backend,
    )
    
    legato_hdf5_to_lerobot(
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
