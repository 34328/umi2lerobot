import sys
import os
import torch

# 设置 PyTorch 打印精度，显示更多小数位
torch.set_printoptions(precision=10, sci_mode=False)

# 方法2: 直接指向包含lerobot包的目录
project_root = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(project_root, 'lerobot', 'src'))

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 数据已经位于 ~/.cache/huggingface/lerobot/mv-umi/bottles_rack
subtask = "human_clothes_data"
dataset_root = os.path.expanduser(f"~/.cache/huggingface/lerobot/DexWild/{subtask}")
dataset = LeRobotDataset(repo_id=subtask, root=dataset_root)

# 打印数据集基本信息
print("=== 数据集基本信息 ===")
print(f"数据集: {dataset}")
print(f"总帧数: {dataset.num_frames}")
print(f"总剧集数: {dataset.num_episodes}")
print(f"FPS: {dataset.fps}")

# 打印第一个样本的层级结构和shape
print("\n=== 第一个样本结构 ===")
idx = 6

zero_item = dataset[idx-1]
first_item = dataset[idx]
second_item = dataset[idx+1]
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

# 打印指定 item 的所有非图像观测数据
def print_item_obs(name, item):
    print(f"\n--- {name} ---")
    
    # 打印 action
    if 'hand_action' in item:
        print(f"Action: {item['hand_action']}")
    
    # 打印所有非图像的 observation 字段
    for key, value in item.items():
        if key.startswith('observation.') and 'images' not in key:
            # 跳过 state (因为通常和 action 类似，如果已经在上面打印过 action)
            # 但这里用户也想看 state，所以都打印
            if hasattr(value, 'shape') and len(value.shape) > 0:
                 print(f"{key}: {value}")
            elif hasattr(value, 'item'):
                 print(f"{key}: {value.item()}")
            else:
                 print(f"{key}: {value}")

# print_item_obs("Item (idx-1)", zero_item)
# print_item_obs("Item (idx)", first_item)
# print_item_obs("Item (idx+1)", second_item)




# # 打印触觉数据
# print("\n=== 触觉数据 ===")
# tactile_data = zero_item['observation.tactile.camera0_tactile']
# print(tactile_data[0])

# print(f"Action values: {action.tolist()}")  # 转换为Python列表以便更清晰地显示数值

print("\n=== 标量值 ===")
print("时间戳(timestamp):", first_item['timestamp'].item())        # 使用 .item() 提取标量值
print("帧索引(frame_index):", first_item['frame_index'].item())
print("剧集索引(episode_index):", first_item['episode_index'].item())
print("全局索引(index):", first_item['index'].item())
print("任务索引(task_index):", first_item['task_index'].item())
print("任务描述(task):", first_item['task'])