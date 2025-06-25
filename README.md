# 蓝海智询

基于大模型 RAG 检索增强与知识图谱技术的水生动物疾病问答平台。本仓库包含后端服务、示例数据和相关脚本，旨在为水产养殖从业者提供疾病诊断、治疗建议以及海洋环境分析能力。

## 主要特性

- **RAG 检索增强**：结合 FAISS 向量检索与 BM25 关键词检索，提高问答准确度。
- **知识图谱推理**：使用 Neo4j 构建疾病–症状–治疗关系网络，支持基于图谱的诊断。
- **OceanGPT 模型**：针对海洋场景微调的大语言模型，支持 LoRA 与量化加载。
- **多模态诊断**：可接收文本描述与环境参数，生成个性化治疗方案。
- **完整 API**：基于 FastAPI 提供问答、诊断、训练等接口，同时附带 React 前端示例。

## 环境准备

1. **克隆项目**
   ```bash
   git clone https://github.com/GodCzy/Blueocean-rag.git
   cd Blueocean-rag
   ```
2. **安装依赖**（推荐在虚拟环境中执行）
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows 下使用 venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **模型与数据**
   - 运行 `scripts/setup_project.py` 可自动创建目录并尝试下载所需模型与数据。
   - 如需手动下载，可执行 `python download_model.py`，或按 `docs/DEPLOYMENT_GUIDE.md` 的说明自行配置。
4. **启动服务**
   ```bash
   python src/main.py
   ```
   默认服务地址为 `http://localhost:8000`，可通过 `http://localhost:8000/docs` 查看接口文档。

## 目录结构概览

```
├── datasets/           # 样例数据集
├── data/               # 向量库、知识图谱等运行数据
├── models/             # OceanGPT 等模型文件
├── src/                # 后端源代码
│   ├── api/            # 业务 API 模块
│   ├── routers/        # FastAPI 路由
│   ├── core/           # 核心功能实现
│   └── ...
├── scripts/            # 辅助脚本
└── docs/               # 额外文档
```

## 运行测试

项目提供了一些基础单元测试，可通过下列命令运行：

```bash
python -m pytest -v
```

如测试因缺少依赖或网络限制而失败，可根据 `requirements.txt` 安装额外包，或参考文档手动配置。

## 贡献

欢迎通过 Pull Request 提交改进，或在 Issues 中反馈问题。提交代码前请确保通过相应单元测试。

## 许可证

项目代码遵循 [MIT](LICENSE) 协议发布。
