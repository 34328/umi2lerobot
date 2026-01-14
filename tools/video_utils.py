"""
视频生成工具 - 解决 SSH 远程连接时 Notebook 无法播放视频的问题

使用方法:

1. 从原始 MCAP 数据提取视频:
```python
from video_utils import mcap_to_video
from pathlib import Path

mcap_file = Path('/mnt/raid0/UMI2Lerobot/rawData/10Kh-RealOmin-OpenData/Cooking_and_Kitchen_Clean/clean_bowl/00001/xxx.mcap')
video = mcap_to_video(mcap_file, robot_name='robot0')
```

2. 从转换后的 HDF5 数据提取视频:
```python
from video_utils import episode_to_video
from pathlib import Path

data_root = Path('/mnt/raid0/UMI2Lerobot/rawData/FastUMI_extracted')
task_name = 'cover_beef_v0'
episode_idx = 1
video = episode_to_video(data_root, task_name, episode_idx)
```

视频会保存到 parsing&visualization 目录下，可以通过 scp 命令下载到本地查看。
"""

from pathlib import Path
import h5py
import cv2
import os
import sys
from tqdm import tqdm

# 添加 das-datakit 路径
_current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_current_dir / "das-datakit"))
from utils.mcaploader import McapLoader

# 视频保存目录 (与 notebook 同目录)
VIDEO_SAVE_DIR = Path(__file__).parent


def mcap_to_video(
    mcap_file: Path,
    robot_name: str = "robot0",
    fps: int = 30,
    show_info: bool = True,
    max_width: int = 960,
    save_dir: Path = None,
) -> Path:
    """从原始 MCAP 文件提取 robot 相机视频
    
    Args:
        mcap_file: MCAP 文件路径
        robot_name: 机器人名称 (robot0 或 robot1)
        fps: 视频帧率
        show_info: 是否在视频上显示帧信息
        max_width: 视频最大宽度
        save_dir: 保存目录，默认为当前脚本目录
    
    Returns:
        视频文件路径
    """
    if save_dir is None:
        save_dir = VIDEO_SAVE_DIR
    
    mcap_file = Path(mcap_file)
    if not mcap_file.exists():
        print(f'❌ MCAP 文件不存在: {mcap_file}')
        return None
    
    print(f'📂 MCAP 文件: {mcap_file.name}')
    print(f'🤖 机器人: {robot_name}')
    
    # 加载 MCAP
    bag = McapLoader(str(mcap_file))
    camera_topic = f"/{robot_name}/sensor/camera0/compressed"
    
    print(f'🎥 相机 topic: {camera_topic}')
    
    # 加载相机数据
    bag.load_topics([camera_topic], auto_sync=False)
    camera_data = bag.get_topic_data(camera_topic)
    
    if not camera_data or len(camera_data) == 0:
        print(f'❌ 找不到相机数据: {camera_topic}')
        print(f'可用 topics: {bag.all_topic_names}')
        bag.close()
        return None
    
    num_frames = len(camera_data)
    print(f'📊 总帧数: {num_frames}')
    
    # 获取第一帧来确定分辨率
    first_frame = camera_data[0]["decode_data"]
    if first_frame is None:
        print(f'❌ 无法解码第一帧图像')
        bag.close()
        return None
    
    height, width = first_frame.shape[:2]
    print(f'📊 原始分辨率: {width}x{height}')
    
    scale = max_width / width
    new_width, new_height = max_width, int(height * scale)
    
    # 保存视频
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_filename = f'{mcap_file.stem}_{robot_name}_camera.mp4'
    video_path = save_dir / video_filename
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (new_width, new_height))
    
    for i in tqdm(range(num_frames), desc='生成视频'):
        frame = camera_data[i]["decode_data"]
        if frame is None:
            continue
        
        frame = cv2.resize(frame, (new_width, new_height))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if show_info:
            font, scale_f, color = cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0)
            cv2.putText(frame_bgr, f'Frame: {i+1}/{num_frames}', (10, 25), font, scale_f, color, 1)
            cv2.putText(frame_bgr, f'Robot: {robot_name}', (10, 50), font, scale_f, color, 1)
            cv2.putText(frame_bgr, f'MCAP: {mcap_file.name}', (10, 75), font, scale_f, color, 1)
        
        video_writer.write(frame_bgr)
    
    video_writer.release()
    bag.close()
    
    print(f'✅ 视频时长: {num_frames/fps:.2f}s')
    print(f'💾 视频已保存到: {video_path}')
    print(f'\n📥 下载命令 (在本地终端执行):')
    print(f'   scp unitree@<服务器IP>:{video_path} ./')
    
    return video_path


def episode_to_video(
    data_root: Path,
    task_name: str,
    episode_idx: int = 1,
    fps: int = 30,
    show_info: bool = True,
    max_width: int = 960,
    save_dir: Path = None,
) -> Path:
    """将指定 episode 转换为视频并保存到本地文件
    
    Args:
        data_root: 数据根目录
        task_name: 任务名称
        episode_idx: episode 索引
        fps: 视频帧率
        show_info: 是否在视频上显示帧信息
        max_width: 视频最大宽度
        save_dir: 保存目录，默认为 notebook 所在目录
    
    Returns:
        视频文件路径
    
    注意: 通过 SSH 远程连接时，notebook 中无法直接播放视频。
    视频会保存到指定目录，可以通过 scp 或其他方式下载到本地查看。
    """
    if save_dir is None:
        save_dir = VIDEO_SAVE_DIR
    
    task_dir = data_root / task_name
    episode_file = task_dir / f'episode_{episode_idx}.hdf5'
    
    if not episode_file.exists():
        print(f'❌ Episode 文件不存在: {episode_file}')
        return None
    
    print(f'📂 Task: {task_name}')
    print(f'🎬 Episode: {episode_idx}')
    
    with h5py.File(episode_file, 'r') as f:
        images = f['observations/images/front'][:]
        actions = f['action'][:]
    
    num_frames, height, width, _ = images.shape
    print(f'📊 帧数: {num_frames}, 分辨率: {width}x{height}')
    
    scale = max_width / width
    new_width, new_height = max_width, int(height * scale)
    
    # 保存到固定目录，使用有意义的文件名
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_filename = f'{task_name}_episode_{episode_idx}.mp4'
    video_path = save_dir / video_filename
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (new_width, new_height))
    
    for i in tqdm(range(num_frames), desc='生成视频'):
        frame = cv2.resize(images[i], (new_width, new_height))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if show_info:
            font, scale_f, color = cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0)
            cv2.putText(frame_bgr, f'Frame: {i+1}/{num_frames}', (10, 25), font, scale_f, color, 1)
            cv2.putText(frame_bgr, f'Task: {task_name}', (10, 50), font, scale_f, color, 1)
            a = actions[i]
            cv2.putText(frame_bgr, f'XYZ: [{a[0]:.2f}, {a[1]:.2f}, {a[2]:.2f}]', (10, 75), font, scale_f, color, 1)
            cv2.putText(frame_bgr, f'Rot: [{a[3]:.2f}, {a[4]:.2f}, {a[5]:.2f}] G:{a[6]:.2f}', (10, 100), font, scale_f, color, 1)
        
        video_writer.write(frame_bgr)
    
    video_writer.release()
    
    print(f'✅ 视频时长: {num_frames/fps:.2f}s')
    print(f'💾 视频已保存到: {video_path}')
    print(f'\n📥 下载命令 (在本地终端执行):')
    print(f'   scp unitree@<服务器IP>:{video_path} ./')
    
    return video_path


def display_video_in_notebook(video_path: Path, width: int = 960):
    """尝试在 notebook 中显示视频 (SSH 远程可能无法播放)"""
    try:
        from IPython.display import Video, display
        display(Video(str(video_path), embed=True, width=width))
    except Exception as e:
        print(f"⚠️ 无法在 notebook 中播放视频: {e}")
        print(f"请下载视频文件到本地查看: {video_path}")


if __name__ == "__main__":
    # 测试 - 从 MCAP 原始数据提取视频
    print("=" * 60)
    print("测试: 从 MCAP 提取视频")
    print("=" * 60)
    
    mcap_dir = Path('/mnt/raid0/UMI2Lerobot/rawData/10Kh-RealOmin-OpenData/Cooking_and_Kitchen_Clean/clean_bowl/00001')
    # 找第一个 .mcap 文件（排序后）
    mcap_files = sorted(mcap_dir.glob("*.mcap"))
    if mcap_files:
        test_mcap = mcap_files[0]
        print(f"\n使用测试文件: {test_mcap.name}")
        print(f"共找到 {len(mcap_files)} 个 MCAP 文件")
        video_path = mcap_to_video(test_mcap, robot_name='robot0')
        print(f"\n视频路径: {video_path}")
    else:
        print(f"未找到 MCAP 文件: {mcap_dir}")
    
    # 测试 - 从 HDF5 提取视频
    # data_root = Path('/mnt/raid0/UMI2Lerobot/rawData/FastUMI_extracted')
    # task_name = 'unplug_charger_v0'
    # episode_idx = 1
    # video_path = episode_to_video(data_root, task_name, episode_idx)
    # print(f"\n视频路径: {video_path}")
