# # dld.py
import os
import sys
from huggingface_hub import snapshot_download
# Retrieve Hugging Face token from environment variable (set HF_TOKEN)
# HF_TOKEN = os.getenv("HF_TOKEN")
# if not HF_TOKEN:
#     raise RuntimeError("Hugging Face token not found. Please set the HF_TOKEN environment variable with your access token.")

repo_id = "boardd/dexwild-dataset"
local_dir = "/home/unitree/桌面/umi2lerobot/rawData/DexWild/"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    max_workers=8,
    # token=HF_TOKEN,
    cache_dir=None,
    allow_patterns=["clothes_data/robot/**"]
)
print("🎉 下载完成！")
