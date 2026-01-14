"""
FastUMI HDF5 to LeRobot 转换脚本

FastUMI 数据集特点:
1. 每个 episode 是一个独立的 HDF5 文件 (episode_1.hdf5, episode_2.hdf5, ...)
2. 数据结构:
   - action: (N, 7) - 7维动作
   - observations/images/front: (N, H, W, 3) - 前置相机图像
   - observations/qpos: (N, 7) - 7维关节位置
3. 支持合并同一任务的多个版本 (如 clean_table_v0, v1, v2, ...)
"""

import sys
import re
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

from tools.constants import ROBOT_CONFIGS


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def find_task_versions(raw_dir: Path, subtask: str) -> list[Path]:
    """
    查找指定任务的所有版本目录
    
    例如: subtask="clean_table" 会匹配:
    - clean_table_v0
    - clean_table_v1
    - clean_table_v2
    ...
    
    返回按版本号排序的目录列表
    """
    # 匹配 subtask_v{数字} 格式
    pattern = re.compile(rf"^{re.escape(subtask)}_v(\d+)$")
    
    version_dirs = []
    for item in raw_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                version_num = int(match.group(1))
                version_dirs.append((version_num, item))
    
    # 按版本号排序
    version_dirs.sort(key=lambda x: x[0])
    
    if not version_dirs:
        raise FileNotFoundError(
            f"未找到任务 '{subtask}' 的任何版本目录 (格式: {subtask}_v0, {subtask}_v1, ...)"
        )
    
    result = [d for _, d in version_dirs]
    print(f"\n找到 {len(result)} 个版本目录:")
    for d in result:
        print(f"  - {d.name}")
    
    return result


def get_episode_files(task_dir: Path) -> list[Path]:
    """获取任务目录下所有 episode 文件，按自然顺序排序"""
    episode_files = sorted(
        task_dir.glob("episode_*.hdf5"),
        key=lambda x: int(x.stem.split("_")[1])
    )
    return episode_files


def load_episode_data(episode_path: Path) -> dict:
    """
    加载单个 episode 的数据
    
    返回:
        {
            "action": ndarray (N, 7),
            "observations/qpos": ndarray (N, 7),
            "observations/images/front": ndarray (N, H, W, 3),
        }
    """
    data = {}
    
    with h5py.File(episode_path, 'r') as f:
        # 读取 action
        if 'action' in f:
            data['action'] = f['action'][:]
        else:
            raise KeyError(f"Missing 'action' field in {episode_path}")
        
        # 读取 observations/qpos
        if 'observations' in f and 'qpos' in f['observations']:
            data['observations/qpos'] = f['observations/qpos'][:]
        else:
            raise KeyError(f"Missing 'observations/qpos' field in {episode_path}")
        
        # 读取 observations/images/front
        if 'observations' in f and 'images' in f['observations'] and 'front' in f['observations/images']:
            data['observations/images/front'] = f['observations/images/front'][:]
        else:
            raise KeyError(f"Missing 'observations/images/front' field in {episode_path}")
    
    return data


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    root: Path | None = None,
    fps: int = 30,
    image_shape: tuple[int, int, int] = (1080, 1920, 3),
) -> LeRobotDataset:
    """创建空的 LeRobot 数据集"""
    
    config = ROBOT_CONFIGS[robot_type]
    motors = config.motors
    
    features = {}
    
    # 添加相机图像特征
    for cam in config.cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }
    
    # 定义 motor names (7维: x, y, z, rx, ry, rz, gripper)
    motor_names = motors  # ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    
    # 添加 observation.state (qpos)
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (len(motors),),
        "names": motor_names,
    }
    
    # 添加 action
    features["action"] = {
        "dtype": "float32",
        "shape": (len(motors),),
        "names": motor_names,
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
    episode_files: list[Path],
    task_text: str,
    start_episode: int = 0,
) -> LeRobotDataset:
    """填充 LeRobot 数据集"""
    
    skipped_episodes = {}  # {episode_path: reason}
    total_episodes = len(episode_files)
    
    if start_episode > 0:
        print(f"\n🔄 断点续传: 跳过前 {start_episode} 个 episode，从第 {start_episode} 个开始")
    
    for j in tqdm.tqdm(
        range(start_episode, total_episodes),
        desc="Converting episodes",
        initial=start_episode,
        total=total_episodes
    ):
        episode_path = episode_files[j]
        
        try:
            # 加载 episode 数据
            data = load_episode_data(episode_path)
            
            action = data['action']
            qpos = data['observations/qpos']
            images = data['observations/images/front']
            
            episode_length = action.shape[0]
            
            for i in range(episode_length):
                frame = {
                    "observation.images.front": images[i],
                    "observation.state": qpos[i].astype(np.float32),
                    "action": action[i].astype(np.float32),
                }
                dataset.add_frame(frame, task=task_text)
            
            dataset.save_episode()
            
            del data, action, qpos, images
            gc.collect()
            
        except (OSError, KeyError, Exception) as e:
            reason = str(e)[:100]
            print(f"\n⚠️ Skipping corrupted episode {episode_path.name}: {reason}")
            skipped_episodes[str(episode_path)] = reason
            gc.collect()
            continue
    
    if skipped_episodes:
        print(f"\n⚠️ Total skipped episodes: {len(skipped_episodes)}")
        for ep_path, reason in skipped_episodes.items():
            print(f"   - {ep_path}: {reason}")
    
    return dataset


def fastumi_hdf5_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,
    project: str,
    subtask: str,
    output_dir: Path | None = None,
    text: str | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    fps: int = 30,
    image_shape: tuple[int, int, int] = (1080, 1920, 3),
    resume: bool = False,
):
    """将 FastUMI HDF5 数据集转换为 LeRobot 格式"""
    
    # 输出目录：优先使用指定的 output_dir，否则使用默认的 HF_LEROBOT_HOME
    if output_dir is not None:
        dataset_root = output_dir / project / repo_id
    else:
        dataset_root = HF_LEROBOT_HOME / project / repo_id
    
    print(f"\n{'='*60}")
    print(f"Converting FastUMI HDF5 to LeRobot format")
    print(f"Input: {raw_dir}")
    print(f"Task: {subtask}")
    print(f"Output: {dataset_root}")
    print(f"{'='*60}\n")
    
    # 查找所有版本目录
    version_dirs = find_task_versions(raw_dir, subtask)
    
    # 收集所有 episode 文件
    all_episode_files = []
    for version_dir in version_dirs:
        episode_files = get_episode_files(version_dir)
        print(f"  {version_dir.name}: {len(episode_files)} episodes")
        all_episode_files.extend(episode_files)
    
    print(f"\n📊 总计: {len(all_episode_files)} episodes")
    
    if not all_episode_files:
        raise FileNotFoundError(f"未找到任何 episode 文件")
    
    # 获取图像尺寸 (从第一个 episode 获取)
    with h5py.File(all_episode_files[0], 'r') as f:
        sample_image = f['observations/images/front'][0]
        image_shape = sample_image.shape
        print(f"📷 图像尺寸: {image_shape}")
    
    # 任务描述
    task_text = text if text else f"FastUMI {subtask} task"
    
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
    
    # 填充数据集
    dataset = populate_dataset(
        dataset,
        all_episode_files,
        task_text=task_text,
        start_episode=start_episode,
    )
    
    print(f"\n{'='*60}")
    print(f"Conversion completed!")
    print(f"Dataset saved to: {dataset_root}")
    print(f"Total episodes: {dataset.meta.total_episodes}")
    print(f"{'='*60}\n")
    
    if push_to_hub:
        dataset.push_to_hub(upload_large_folder=True)


@dataclasses.dataclass
class ArgsConfig:
    """配置参数"""
    
    raw_dir: Path = Path("/mnt/raid0/UMI2Lerobot/rawData/FastUMI_extracted")
    """原始数据目录 (包含 task_v0, task_v1 等子目录)"""
    
    output_dir: Path = Path("/mnt/raid0/UMI2Lerobot/lerobot")
    """输出目录 (LeRobot 数据集保存位置)"""
    
    project: str = "FastUMI"
    """项目名称"""
    
    subtask: str = "open_drawer"
    """子任务名称 (会自动匹配 subtask_v0, subtask_v1, ...)"""
    
    robot_type: str = "FastUMI"
    """机器人类型"""
    
    text: str = "Open the drawer"
    """任务描述 (可选，默认自动生成)"""
    
    fps: int = 30
    """帧率"""
    
    mode: Literal["video", "image"] = "video"
    """存储模式"""
    
    push_to_hub: bool = False
    """是否上传到 Hugging Face Hub"""
    
    resume: bool = False
    """断点续传：如果数据集已存在，从上次停止的位置继续"""
    
    # 高级配置
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 24
    image_writer_threads: int = 6
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
    
    fastumi_hdf5_to_lerobot(
        raw_dir=config.raw_dir,
        repo_id=config.subtask,
        robot_type=config.robot_type,
        project=config.project,
        subtask=config.subtask,
        output_dir=config.output_dir,
        text=config.text,
        push_to_hub=config.push_to_hub,
        mode=config.mode,
        dataset_config=dataset_config,
        fps=config.fps,
        resume=config.resume,
    )
