from modelscope.hub.snapshot_download import snapshot_download
import os

# 简易模型下载脚本，需确保网络连接可用

# 创建模型目录
model_dir = "models/oceangpt"
os.makedirs(model_dir, exist_ok=True)

# 下载模型
model_id = "zjunlp/OceanGPT-o-7B"
print(f"开始下载模型: {model_id}")
model_path = snapshot_download(model_id, cache_dir=model_dir)
print(f"模型下载完成，保存在: {model_path}") 