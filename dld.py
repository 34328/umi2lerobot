# # dld.py
import os
import sys
from huggingface_hub import snapshot_download

repo_id = "IPEC-COMMUNITY/FastUMI-Data"
local_dir = "./rawData/FastUMI"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    max_workers=8,
    # token=False,
    cache_dir=None,
    # allow_patterns=["in_the_wild_data/**"]
)
print("🎉 下载完成！")
