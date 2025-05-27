# 模型文件

本目录用于存放蓝海智询项目所需的大型模型文件。由于GitHub的文件大小限制，这些模型文件不直接包含在仓库中。

## 下载模型

您可以通过以下方式获取所需的模型文件：

### 方法一：使用自动下载脚本（推荐）

```bash
python download_model.py
```

### 方法二：手动下载

从HuggingFace或ModelScope下载以下模型：

1. **OceanGPT主模型**：
   ```bash
   git clone https://huggingface.co/zjunlp/OceanGPT-o-7B-v0.1 ./models/OceanGPT-o-7B-v0.1
   ```

2. **文本嵌入模型**：
   ```bash
   git clone https://huggingface.co/BAAI/bge-large-zh-v1.5 ./models/bge-large-zh-v1.5
   ```

3. **重排序模型**：
   ```bash
   git clone https://huggingface.co/BAAI/bge-reranker-large ./models/bge-reranker-large
   ```

## 目录结构

下载完成后，模型目录结构应如下所示：

```
models/
├── OceanGPT-o-7B-v0.1/      # 主模型
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   └── tokenizer.model
├── bge-large-zh-v1.5/       # 嵌入模型
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer.json
└── bge-reranker-large/      # 重排序模型
    ├── config.json
    ├── model.safetensors
    └── tokenizer.json
```

## 模型配置

模型配置在`config.json`文件中定义。您可以根据需要调整以下参数：

- `model_provider`: 模型提供者
- `model_name`: 模型名称
- `embed_model`: 嵌入模型
- `reranker`: 重排序模型

## 故障排除

如果您在下载或加载模型时遇到问题，请尝试以下操作：

1. 确保您有足够的磁盘空间（至少需要10GB）
2. 检查网络连接是否稳定
3. 验证您是否有安装Git LFS
4. 参考错误日志中的具体信息 