# 蓝海智询

基于大模型 RAG 检索增强与知识图谱技术的水生动物疾病问答平台。本仓库包含 FastAPI 后端、OceanGPT 模型管理脚本、示例前端以及初始化脚本，旨在为水产养殖从业者提供疾病诊断、治疗建议与海洋环境分析能力。

## 主要特性

- **RAG 检索增强**：结合 FAISS 向量检索与 BM25 关键词检索，提高问答准确度。
- **知识图谱推理**：使用 Neo4j 构建疾病–症状–治疗关系网络，支持基于图谱的诊断。
- **OceanGPT 模型**：针对海洋场景微调的大语言模型，支持 LoRA 与量化加载。
- **多模态诊断**：可接收文本描述与环境参数，生成个性化治疗方案。
- **完整 API**：基于 FastAPI 提供问答、诊断、训练等接口，同时附带 Vue/Vite 前端示例。

## 环境要求

- Python 3.8+
- 推荐具备 CUDA GPU（否则自动使用 CPU）
- Node.js 16+（仅在运行前端示例时需要）

## 快速开始

1. **克隆项目**

   ```bash
   git clone https://github.com/GodCzy/Blueocean-rag.git
   cd Blueocean-rag
   ```

2. **安装依赖**（建议使用虚拟环境）

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows 使用 venv\Scripts\activate
   pip install -r requirements.txt
   ```

   可执行 `python quick_start.py` 进行环境自检并自动创建常用目录。

3. **下载模型与数据**

   使用模型管理脚本列出和下载推荐模型：

   ```bash
   python manage_oceangpt.py list
   python manage_oceangpt.py download OceanGPT-o-7B-v0.1
   python manage_oceangpt.py switch OceanGPT-o-7B-v0.1
   ```

   模型和数据路径可在 `config.json` 中调整。若需一键初始化目录、模型及示例数据，可运行 `python scripts/setup_project.py`。

4. **启动后端服务**

   ```bash
   python run.py --host 0.0.0.0 --port 8000
   ```

   服务启动后访问 `http://localhost:8000/docs` 查看 API 文档。

5. **启动前端（可选）**

   ```bash
   cd web
   npm install
   npm run dev
   ```

   默认前端地址为 `http://localhost:5173`。

## 目录结构概览

```
├── data/               # 运行产生的数据（向量库、知识图谱等）
├── datasets/           # 示例数据集
├── models/             # 预训练模型存放目录
├── src/                # 后端源代码
│   ├── api/            # 业务 API 模块
│   ├── routers/        # FastAPI 路由
│   ├── core/           # 核心功能实现
│   └── ...
├── scripts/            # 辅助脚本（模型下载、项目初始化等）
├── web/                # Vue/Vite 前端示例
├── run.py              # 后端启动脚本
├── quick_start.py      # 环境检查与快速启动向导
└── manage_oceangpt.py  # OceanGPT 模型管理脚本
```

## 运行测试

```bash
python -m pytest -v
```

## 贡献

欢迎通过 Pull Request 提交改进，或在 Issues 中反馈问题。提交代码前请确保通过相应单元测试。

## 许可证

项目代码遵循 [MIT](LICENSE) 协议发布。
