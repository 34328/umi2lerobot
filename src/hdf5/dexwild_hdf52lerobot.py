
import sys
import tqdm
import tyro
import dataclasses
import numpy as np
from pathlib import Path
from typing import Literal

import h5py

# 路径设置
_project_root = Path(__file__).parent.parent.parent  # umi2lerobot/
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root / "lerobot" / "src"))

from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from utils.constants import ROBOT_CONFIGS
from utils.data_processing import align_all_data


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


class DexWildHDF5Dataset:
    """DexWild HDF5 数据集解析器"""
    
    def __init__(self, hdf5_path: Path, robot_type: str, task_text: str | None = None) -> None:
        self.hdf5_path = hdf5_path
        self.robot_type = robot_type
        self.task_text = task_text
        
        # 加载配置
        self.config = ROBOT_CONFIGS[robot_type]
        
        # 打开 HDF5 文件
        self.hdf5_file = h5py.File(hdf5_path, 'r')
        
        # 初始化 episodes
        self._init_episodes()
    
    def _init_episodes(self) -> None:
        """初始化 episode 列表"""
        # 获取所有 episode 名称，按自然数顺序排序
        ep_names = [k for k in self.hdf5_file.keys() if k.startswith('ep_')]
        self.ep_names = sorted(ep_names, key=lambda x: int(x.split('_')[1]))
        
        print(f"\n{'='*60}")
        print(f"DexWild HDF5 File: {self.hdf5_path.name}")
        print(f"  - Total episodes: {len(self.ep_names)}")
        print(f"{'='*60}\n")
    
    def __len__(self) -> int:
        return len(self.ep_names)
    
    def get_item(self, index: int) -> dict:
        """获取指定索引的 episode 数据（已对齐）"""
        ep_name = self.ep_names[index]
        ep = self.hdf5_file[ep_name]
        
        # 使用 align_all_data 对齐所有数据
        aligned = align_all_data(ep)
        
        # 获取帧数
        first_key = next(iter(aligned.keys()))
        episode_length = aligned[first_key].shape[0]
        
        # 获取图像尺寸 (从任意存在的相机获取)
        cam_height, cam_width, cam_channels = 240, 320, 3
        for cam_name in self.config.cameras:
            if cam_name in aligned:
                cam_height, cam_width, cam_channels = aligned[cam_name].shape[1:4]
                break
        
        # 提取相机图像 - 缺失的相机用黑图填充
        cameras = {}
        for cam_name in self.config.cameras:
            if cam_name in aligned:
                cameras[cam_name] = aligned[cam_name]
            else:
                # 用黑图填充缺失的相机
                black_images = np.zeros((episode_length, cam_height, cam_width, cam_channels), dtype=np.uint8)
                cameras[cam_name] = black_images
                print(f"  ⚠️ Camera '{cam_name}' missing, filled with black images")
        
        # 提取所有数值数据 - 缺失的字段用0填充
        pose_data = {}
        for hdf5_key, lerobot_key in self.config.demo_pose_sensors.items():
            found = False
            # 精确匹配: aligned key 的最后一部分必须完全等于 hdf5_key
            for aligned_key in aligned:
                # aligned_key 格式: "group/field" 或 "field"
                field_name = aligned_key.split('/')[-1]
                if field_name == hdf5_key:
                    pose_data[lerobot_key] = aligned[aligned_key].astype(np.float32)
                    found = True
                    break
            
            # 缺失的字段用0填充
            if not found:
                shape = self.config.demo_pose_shapes.get(hdf5_key, (7,))
                pose_data[lerobot_key] = np.zeros((episode_length, *shape), dtype=np.float32)
                print(f"  ⚠️ Pose '{hdf5_key}' missing, filled with zeros")
        
        # 任务描述
        if self.task_text:
            task = self.task_text
        else:
            task = "DexWild manipulation task"
        
        return {
            "episode_index": index,
            "ep_name": ep_name,
            "episode_length": episode_length,
            "cameras": cameras,
            "pose_data": pose_data,
            "task": task,
            "data_cfg": {
                "camera_names": list(cameras.keys()),
                "cam_height": cam_height,
                "cam_width": cam_width,
                "cam_channels": cam_channels,
            },
        }
    
    def close(self):
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
    fps: int = 30,
    image_shape: tuple[int, int, int] = (240, 320, 3),
) -> LeRobotDataset:
    """创建空的 LeRobot 数据集"""
    
    config = ROBOT_CONFIGS[robot_type]
    cameras = config.cameras
    demo_pose_shapes = config.demo_pose_shapes
    
    features = {}
    
    # 添加相机图像特征 (保持 observation.images.xxx 格式)
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }
    
    # 添加数值数据特征 (直接命名，不使用 observation. 前缀)
    for sensor_name, shape in demo_pose_shapes.items():
        features[sensor_name] = {
            "dtype": "float32",
            "shape": shape,
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


import gc

def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_path: Path,
    robot_type: str,
    task_text: str | None = None,
) -> LeRobotDataset:
    """填充 LeRobot 数据集"""
    
    skipped_episodes = {}  # {ep_name: reason}
    
    with DexWildHDF5Dataset(hdf5_path, robot_type, task_text) as dexwild_dataset:
        for j in tqdm.tqdm(range(len(dexwild_dataset)), desc="Converting episodes"):
            try:
                episode = dexwild_dataset.get_item(j)
                
                cameras = episode["cameras"]
                pose_data = episode["pose_data"]
                task = episode["task"]
                episode_length = episode["episode_length"]
                
                for i in range(episode_length):
                    frame = {}
                    
                    # 添加图像 (保持 observation.images.xxx 格式)
                    for camera, img_array in cameras.items():
                        frame[f"observation.images.{camera}"] = img_array[i]
                    
                    # 添加数值数据 (直接命名)
                    for pose_name, pose_array in pose_data.items():
                        frame[pose_name] = pose_array[i]
                    
                    dataset.add_frame(frame, task=task)
                
                dataset.save_episode()
                
                del episode, cameras, pose_data
                gc.collect()
                
            except (OSError, Exception) as e:
                ep_name = dexwild_dataset.ep_names[j]
                reason = str(e)[:100]  # 截取前100个字符
                print(f"\n⚠️ Skipping corrupted episode {ep_name} (index {j}): {reason}")
                skipped_episodes[ep_name] = reason
                gc.collect()
                continue
    
    if skipped_episodes:
        print(f"\n⚠️ Total skipped episodes: {len(skipped_episodes)}")
        for ep_name, reason in skipped_episodes.items():
            print(f"   - {ep_name}: {reason}")
    
    return dataset


def dexwild_hdf5_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,
    project: str,
    text: str | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    fps: int = 30,
    image_shape: tuple[int, int, int] = (240, 320, 3),
):
    """将 DexWild HDF5 数据集转换为 LeRobot 格式"""
    
    dataset_root = HF_LEROBOT_HOME / project / repo_id
    
    print(f"\n{'='*60}")
    print(f"Converting DexWild HDF5 to LeRobot format")
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
    dataset = populate_dataset(
        dataset,
        raw_dir,
        robot_type=robot_type,
        task_text=text,
    )
    
    print(f"\n{'='*60}")
    print(f"Conversion completed!")
    print(f"Dataset saved to: {dataset_root}")
    print(f"{'='*60}\n")
    
    if push_to_hub:
        dataset.push_to_hub(upload_large_folder=True)


@dataclasses.dataclass
class ArgsConfig:
    """配置参数"""
    
    raw_dir: Path = Path("/home/unitree/桌面/umi2lerobot/rawData/DexWild/spray_data/human/human_spray_data.hdf5")
    """原始 HDF5 文件路径"""
    
    project: str = "DexWild"
    """项目名称"""
    
    subtask: str = "human_spray_data"
    """子任务名称"""
    
    robot_type: str = "DexWild"
    """机器人类型"""
    
    text: str = "Use the spray bottle to spray the cloth on the table"
    """任务描述"""
    
    fps: int = 30
    """帧率"""
    
    image_shape: tuple[int, int, int] = (240, 320, 3)
    """图像形状"""
    
    mode: Literal["video", "image"] = "video"
    """存储模式"""
    
    push_to_hub: bool = False
    """是否上传到 Hugging Face Hub"""
    
    # 高级配置
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    
    dataset_config = DatasetConfig(
        use_videos=config.use_videos,
        tolerance_s=config.tolerance_s,
        image_writer_processes=config.image_writer_processes,
        image_writer_threads=config.image_writer_threads,
        video_backend=config.video_backend,
    )
    
    dexwild_hdf5_to_lerobot(
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
