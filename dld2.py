"""
下载超大数据集的文件索引中的指定文件夹内容
"""

import os
import concurrent.futures
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

# --- 配置区 ---
# 1. 强制镜像和长超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" # 这里暂时关掉 rust 加速，改用多线程稳定下载
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

REPO_ID = "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
# 你想下载的特定文件夹路径
TARGET_FOLDER = "sim_behavior_r1_pro.task-0000_turning_on_radio"
LOCAL_DIR = "./rawData"
MAX_WORKERS = 8  # 并发下载数

def download_file(file_path):
    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=file_path,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )
        return True
    except Exception as e:
        print(f"\n❌ 下载失败: {file_path} - {e}")
        return False

def main():
    api = HfApi()
    
    print(f"📡 正在连接 API，仅获取文件夹 '{TARGET_FOLDER}' 的清单...")
    # 使用 list_repo_tree 递归获取特定文件夹下的文件
    try:
        # 获取所有文件，然后过滤出目标文件夹下的文件
        all_files = api.list_repo_tree(
            repo_id=REPO_ID,
            repo_type="dataset",
            recursive=True,
            path_in_repo=TARGET_FOLDER
        )
        # 只保留文件（过滤掉文件夹），并获取文件路径
        # RepoFile 有 path 属性，RepoFolder 没有 path 属性或可以通过类型判断
        files = []
        for item in all_files:
            # 检查是否为 RepoFile 类型
            if hasattr(item, '__class__') and item.__class__.__name__ == 'RepoFile':
                files.append(item.path)
        print(f"✅ 成功获取清单！该文件夹下共有 {len(files)} 个文件。")
    except Exception as e:
        print(f"❌ 获取文件列表失败。可能原因：网络问题或该文件夹路径不存在。\n错误信息: {e}")
        return

    print(f"🚀 开始并发下载 (线程数: {MAX_WORKERS})...")
    
    # 使用线程池并发下载
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 使用 tqdm 显示进度条
        results = list(tqdm(executor.map(download_file, files), total=len(files), unit="file"))

    print("\n🎉 所有任务处理完毕！")

if __name__ == "__main__":
    main()
    