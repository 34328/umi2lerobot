# # dld.py
import os
import sys

# === 彻底禁用所有可能的代理（在导入 huggingface_hub 之前！）===
proxy_keys = [
    "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
    "ALL_PROXY", "all_proxy"
]
for key in proxy_keys:
    os.environ[key] = ""  # 设为空字符串比 pop 更安全（避免 KeyError）

# 强制使用国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("✅ 已禁用所有代理，强制使用 hf-mirror.com")

# 现在才导入 huggingface_hub（避免它提前读取代理）
from huggingface_hub import snapshot_download

repo_id = "Fanqi-Lin/Processed-Task-Dataset"
local_dir = "./rawData/Data_Scaling_Laws"

try:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        max_workers=8,
        token=False,
        cache_dir=None,
        # allow_patterns=["in_the_wild_data/**"]
    )
    print("🎉 下载完成！")
except Exception as e:
    print(f"💥 下载失败: {e}", file=sys.stderr)
    sys.exit(1)