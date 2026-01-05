import os
import sys
import zarr
from utils.imagecodecs_numcodecs import register_codecs, JpegXl
# 临时重定向 stderr 来抑制注册信息
old_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

try:
    register_codecs()
finally:
    sys.stderr.close()
    sys.stderr = old_stderr
path = "/mnt/raid0/UMI2Lerobot/rawData/DexUMI/inspire_egg_carton/eggbox_1_29_dataset/episode_0"

# mode='r' 表示只读，避免误操作
root = zarr.open(path, mode='r')
print(root.tree())