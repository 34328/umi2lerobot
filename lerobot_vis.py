import sys
import os

# 方法2: 直接指向包含lerobot包的目录
project_root = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(project_root, 'lerobot', 'src'))

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(repo_id="bottles_rack", root="")

# 打印数据集基本信息
print("=== 数据集基本信息 ===")
print(f"数据集: {dataset}")
print(f"总帧数: {dataset.num_frames}")
print(f"总剧集数: {dataset.num_episodes}")
print(f"FPS: {dataset.fps}")

# 打印第一个样本的层级结构和shape
print("\n=== 第一个样本结构 ===")
first_item = dataset[0]

def print_structure(data, indent=0):
    prefix = "  " * indent
    if isinstance(data, dict):
        print(f"{prefix}{{")
        for key, value in data.items():
            if hasattr(value, 'shape'):
                print(f"{prefix}  {key}: {type(value).__name__} with shape {value.shape}")
            elif isinstance(value, (list, tuple)):
                print(f"{prefix}  {key}: {type(value).__name__} with length {len(value)}")
            elif isinstance(value, dict):
                print(f"{prefix}  {key}:")
                print_structure(value, indent + 2)
            else:
                print(f"{prefix}  {key}: {type(value).__name__}")
        print(f"{prefix}}}")
    else:
        if hasattr(data, 'shape'):
            print(f"{prefix}{type(data).__name__} with shape {data.shape}")
        else:
            print(f"{prefix}{type(data).__name__}")

print_structure(first_item)
print("\n=== 标量值 ===")
print("时间戳(timestamp):", first_item['timestamp'].item())        # 使用 .item() 提取标量值
print("帧索引(frame_index):", first_item['frame_index'].item())
print("剧集索引(episode_index):", first_item['episode_index'].item())
print("全局索引(index):", first_item['index'].item())
print("任务索引(task_index):", first_item['task_index'].item())
print("任务描述(task):", first_item['task'])  





# # 打印特征信息
# print("\n=== 特征信息 ===")
# for key, feature in dataset.features.items():
#     print(f"{key}: {feature}")

# # 打印数据集统计信息（如果有）
# if hasattr(dataset, 'stats') and dataset.stats:
#     print("\n=== 统计信息 ===")
#     for key, stat in dataset.stats.items():
#         if isinstance(stat, dict) and 'mean' in stat and 'std' in stat:
#             print(f"{key}: mean={stat['mean']}, std={stat['std']}")
#         else:
#             print(f"{key}: {stat}")


