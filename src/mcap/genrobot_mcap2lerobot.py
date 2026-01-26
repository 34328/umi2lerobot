"""
GenRobot MCAP to LeRobot 转换脚本

将 10Kh-RealOmin-OpenData 的 MCAP 数据转换为 LeRobot 格式
支持双机器人 (robot0/robot1) 数据
"""

import sys
import gc
import cv2
import tqdm
import tyro
import dataclasses
import numpy as np
from pathlib import Path
from typing import Literal

# 路径设置
_project_root = Path(__file__).parent.parent.parent  # umi2lerobot/
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "das-datakit"))
sys.path.insert(0, str(_project_root / "lerobot" / "src"))

# 导入模块 - tools 是项目的 (避免与 das-datakit/utils 冲突)
from tools.constants import ROBOT_CONFIGS
from tools.data_processing import align_mcap_data
from utils.mcaploader import McapLoader  # 这个是 das-datakit 的 utils

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


class GenRobotMCAPDataset:
    """GenRobot MCAP 数据集解析器"""
    
    def __init__(self, mcap_dir: Path, robot_type: str, task_text: str | None = None) -> None:
        self.mcap_dir = mcap_dir
        self.robot_type = robot_type
        self.task_text = task_text or "XXXXXX"
        
        # 加载配置
        self.config = ROBOT_CONFIGS[robot_type]
        
        # 初始化 episodes (每个 .mcap 文件是一个 episode)
        self._init_episodes()
    
    def _init_episodes(self) -> None:
        """初始化 episode 列表"""
        # 获取所有 .mcap 文件
        mcap_files = sorted(self.mcap_dir.glob("*.mcap"))
        self.mcap_files = mcap_files
        
        print(f"\n{'='*60}")
        print(f"GenRobot MCAP Directory: {self.mcap_dir}")
        print(f"  - Total episodes: {len(self.mcap_files)}")
        print(f"{'='*60}\n")
    
    def __len__(self) -> int:
        return len(self.mcap_files)
    
    def get_item(self, index: int) -> dict:
        """获取指定索引的 episode 数据（已对齐）"""
        mcap_file = self.mcap_files[index]
        
        # 加载 MCAP 文件
        bag = McapLoader(str(mcap_file))
        
        # 使用配置进行数据对齐
        aligned = align_mcap_data(bag, config=self.config)
        
        episode_length = aligned["episode_length"]
        
        # 从配置中获取相机列表
        camera_names = list(self.config.mcap_camera_topics.keys())
        
        # 获取图像尺寸 (从第一个相机的第一帧获取)
        first_cam = camera_names[0]
        first_img = aligned[first_cam][0]
        cam_height, cam_width, cam_channels = first_img.shape

        
        # 提取相机图像 (根据配置)
        cameras = {}
        for cam_name in camera_names:
            cameras[cam_name] = aligned[cam_name]
        
        # 提取数值数据 (根据配置)
        pose_data = {}
        for field_name in self.config.mcap_numeric_topics.keys():
            pose_data[field_name] = aligned[field_name]

        
        # 关闭 bag 并清理内存
        bag._bag_data.clear()  # 清理解码后的图像数据！关键！
        bag.close()
        
        return {
            "episode_index": index,
            "mcap_file": mcap_file.name,
            "episode_length": episode_length,
            "cameras": cameras,
            "pose_data": pose_data,
            "task": self.task_text,
            "data_cfg": {
                "camera_names": list(cameras.keys()),
                "cam_height": cam_height,
                "cam_width": cam_width,
                "cam_channels": cam_channels,
            },
        }


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: Path | None = None,
    fps: int = 30,
    image_shape: tuple[int, int, int] = (480, 640, 3),
) -> LeRobotDataset:
    """创建空的 LeRobot 数据集"""
    
    config = ROBOT_CONFIGS[robot_type]
    cameras = config.cameras
    demo_pose_shapes = config.demo_pose_shapes
    
    features = {}
    
    # 添加相机图像特征
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }
    
    # 添加数值数据特征 (使用 MCAP 字段名)
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


def populate_dataset(
    dataset: LeRobotDataset,
    mcap_dir: Path,
    robot_type: str,
    task_text: str | None = None,
    start_episode: int = 0,
    max_episodes: int | None = None,
    target_size: tuple[int, int] | None = (480, 640),
) -> LeRobotDataset:
    """填充 LeRobot 数据集"""
    
    skipped_episodes = {}
    
    mcap_dataset = GenRobotMCAPDataset(mcap_dir, robot_type, task_text)
    total_episodes = len(mcap_dataset)
    
    if max_episodes is not None:
        total_episodes = min(total_episodes, start_episode + max_episodes)
    
    if start_episode > 0:
        print(f"\n🔄 断点续传: 跳过前 {start_episode} 个 episode，从第 {start_episode} 个开始")
    
    for j in tqdm.tqdm(range(start_episode, total_episodes), desc="Converting episodes", initial=start_episode, total=total_episodes):
        try:
            episode = mcap_dataset.get_item(j)
            
            cameras = episode["cameras"]
            pose_data = episode["pose_data"]
            task = episode["task"]
            episode_length = episode["episode_length"]
            
            for i in range(episode_length):
                frame = {}
                
                # 添加图像 (带 resize 和 BGR→RGB 转换)
                for camera, img_array in cameras.items():
                    img = img_array[i]
                    # Resize 图像到目标尺寸 (H, W)
                    if target_size is not None:
                        h, w = target_size
                        if img.shape[:2] != (h, w):
                            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                    # BGR → RGB (das-datakit 解码输出是 BGR24)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frame[f"observation.images.{camera}"] = img
                    # 立即清除原始图像引用，让 GC 可以回收
                    img_array[i] = None
                
                # 添加数值数据
                for pose_name, pose_array in pose_data.items():
                    frame[pose_name] = pose_array[i]
                
                dataset.add_frame(frame, task=task)
                del frame  # 显式删除 frame
            
            dataset.save_episode()
            
            # 重置 hf_dataset 释放内存 (LeRobot 的 concatenate_datasets 会导致内存累积)
            dataset.hf_dataset = dataset.create_hf_dataset()
            
            del episode, cameras, pose_data
            gc.collect()
            
        except Exception as e:
            mcap_file = mcap_dataset.mcap_files[j].name
            reason = str(e)[:100]
            print(f"\n⚠️ Skipping episode {mcap_file} (index {j}): {reason}")
            skipped_episodes[mcap_file] = reason
            gc.collect()
            continue
    
    if skipped_episodes:
        print(f"\n⚠️ Total skipped episodes: {len(skipped_episodes)}")
        for ep_name, reason in skipped_episodes.items():
            print(f"   - {ep_name}: {reason}")
    
    return dataset


def genrobot_mcap_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,
    project: str,
    text: str | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    fps: int = 30,
    image_shape: tuple[int, int, int] = (480, 640, 3),
    resume: bool = False,
    max_episodes: int | None = None,
):
    """将 GenRobot MCAP 数据转换为 LeRobot 格式"""
    HF_LEROBOT_HOME = Path("/mnt/raid0/UMI2Lerobot/test")
    dataset_root = HF_LEROBOT_HOME / project / repo_id
    
    print(f"\n{'='*60}")
    print(f"Converting GenRobot MCAP to LeRobot format")
    print(f"Input: {raw_dir}")
    print(f"Output: {dataset_root}")
    print(f"{'='*60}\n")
    
    start_episode = 0
    
    if dataset_root.exists():
        if resume:
            try:
                existing_dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
                start_episode = existing_dataset.meta.total_episodes
                print(f"\n🔄 检测到已存在的数据集，已转换 {start_episode} 个 episodes")
                
                # 清理可能不完整的视频文件
                videos_dir = dataset_root / "videos"
                if videos_dir.exists():
                    orphan_files = []
                    for video_file in videos_dir.glob("*.mp4"):
                        try:
                            ep_num = int(video_file.stem.split("_episode_")[-1])
                            if ep_num >= start_episode:
                                orphan_files.append(video_file)
                        except (ValueError, IndexError):
                            pass
                    
                    if orphan_files:
                        print(f"🧹 清理 {len(orphan_files)} 个未完成的视频文件...")
                        for f in orphan_files:
                            f.unlink()
                
                dataset = existing_dataset
            except Exception as e:
                print(f"⚠️ 无法加载已存在的数据集: {e}")
                print("将重新开始转换...")
                import shutil
                shutil.rmtree(dataset_root)
                resume = False
        else:
            raise FileExistsError(
                f"Dataset already exists at {dataset_root}. "
                "Use --resume to continue from where it left off, or remove it manually."
            )
    
    if not resume or start_episode == 0:
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
        start_episode=start_episode,
        max_episodes=max_episodes,
        target_size=(image_shape[0], image_shape[1]),
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
    
    raw_dir: Path = Path("/mnt/raid0/UMI2Lerobot/rawData/10Kh-RealOmin-OpenData/Cooking_and_Kitchen_Clean/clean_container/00001")
    """原始 MCAP 目录路径 (包含 .mcap 文件)"""
    
    project: str = "GenRobot"
    """项目名称"""
    
    subtask: str = "clean_container"
    """子任务名称"""
    
    robot_type: str = "GenRobot_MCAP"
    """机器人类型"""
    
    text: str = "Pick up the cloth and clean the container."
    """任务描述"""
    
    fps: int = 30
    """帧率"""
    
    image_shape: tuple[int, int, int] = (480, 640, 3)
    """图像形状 (H, W, C) - 默认 480x640，会自动 resize"""
    
    mode: Literal["video", "image"] = "video"
    """存储模式"""
    
    push_to_hub: bool = False
    """是否上传到 Hugging Face Hub"""
    
    resume: bool = True
    """断点续传"""
    
    max_episodes: int | None = 5
    """最大转换 episode 数量 (用于测试)"""
    
    # 高级配置
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 16
    image_writer_threads: int = 4
    video_backend: str | None = "pyav"


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    
    dataset_config = DatasetConfig(
        use_videos=config.use_videos,
        tolerance_s=config.tolerance_s,
        image_writer_processes=config.image_writer_processes,
        image_writer_threads=config.image_writer_threads,
        video_backend=config.video_backend,
    )
    
    genrobot_mcap_to_lerobot(
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
        resume=config.resume,
        max_episodes=config.max_episodes,
    )
