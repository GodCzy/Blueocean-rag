# 蓝海智询项目说明书

## 1. 项目简介

蓝海智询（Blueocean RAG）是面向水产养殖场景的智能问答与疾病诊断平台，结合检索增强生成（RAG）、知识图谱推理与 OceanGPT 模型，为使用者提供以下能力：

- **RAG 检索**：整合稠密向量检索与稀疏检索策略，提高回答的准确性与可靠性。
- **知识图谱推理**：利用 Neo4j 存储疾病、症状、治疗关系，提供可解释的诊断依据。
- **OceanGPT 模型**：支持加载本地或远程的 OceanGPT 及 LoRA 权重，提供海洋领域优化的回答。
- **多模态诊疗**：结合环境参数和文本描述输出诊疗建议，并支持后续模型调优。
- **可扩展接口**：FastAPI 后端公开统一的 RESTful API，前端示例基于 Vue/Vite。

## 2. 系统架构与目录结构

项目采用“路由层（Routers）- 服务层（Services）- 核心能力（Core）”的分层结构：

- `src/services/`：封装核心业务逻辑，包括索引管理、问答流程、诊断分析与模型调度，供路由与脚本直接调用。
- `src/routers/`：定义 FastAPI 路由，将 HTTP 请求转换为服务层调用。
- `src/api/`：兼容旧接口或组合式 API 的薄包装，内部统一委派给服务层。
- `src/core/`：底层检索、图谱、模型管理、提示词模板等核心能力实现。
- `src/utils/`：通用工具，如日志、配置读取、路径工具等。
- `scripts/`：部署、数据处理、模型管理、采集脚本等命令行工具。
- `datasets/`：用于存放用户自备的原始文档或结构化数据，仓库默认不包含示例数据。
- `data/`：运行时生成的向量库、知识图谱导出、上传缓存等目录，默认通过 `.gitignore` 排除。
- `web/`：Vue/Vite 前端示例项目，可独立构建与部署。

更详尽的目录说明可参考仓库根目录的 `README.md`。

## 3. 环境要求

- **后端运行环境**：Python 3.8 及以上版本。
- **依赖库**：详见 `requirements.txt`，推荐在虚拟环境中安装；可选 GPU（CUDA 11+）用于模型加速。
- **向量检索**：默认使用 FAISS，安装 `faiss-cpu` 或 `faiss-gpu` 版本。
- **知识图谱**：需要 Neo4j 5.x 服务，推荐使用 Docker Compose 启动。
- **前端构建**：Node.js 16+ 与 npm 或 pnpm。
- **其他服务**：根据需要配置第三方模型 API（如 OpenAI、SiliconFlow、DashScope 等）。

## 4. 安装与配置步骤

1. **克隆仓库并创建虚拟环境**

   ```bash
   git clone https://github.com/GodCzy/Blueocean-rag.git
   cd Blueocean-rag
   python -m venv venv
   source venv/bin/activate  # Windows 使用 venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **配置环境变量**

   ```bash
   cp .env.template .env
   ```

   编辑项目根目录下的 `.env`，填写如下常用变量：

   | 变量 | 说明 |
   | --- | --- |
   | `DEBUG` | 是否开启调试模式，默认 `false` |
   | `API_KEY` | 保护管理接口的访问密钥（可选） |
   | `MODEL_DIR` | 本地模型缓存路径，默认为 `./models` |
   | `LLM_MODEL_NAME` | 自定义默认使用的大模型名称 |
   | `EMBEDDING_MODEL_NAME` | 检索使用的嵌入模型 |
   | `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | Neo4j 连接配置 |
   | 第三方 API Key | 如 `OPENAI_API_KEY`、`SILICONFLOW_API_KEY` 等 |

3. **准备模型与数据**

   - 使用 `scripts/model/manage_oceangpt.py` 下载或切换 OceanGPT 模型；
   - 将自备文档、知识图谱导出等资源放入 `datasets/` 对应子目录；
   - 若需要生成向量索引，可运行 `scripts/data_processing/` 下的预处理脚本。

4. **初始化辅助目录（可选）**

   执行 `python scripts/deployment/quick_start.py` 或 `python scripts/deployment/setup_project.py`，自动创建 `data/` 子目录并检查依赖。

## 5. 运行方式

1. **启动后端服务**

   ```bash
   python scripts/deployment/run.py --host 0.0.0.0 --port 8000
   ```

   - 服务启动后可访问 `http://localhost:8000/docs` 查看 OpenAPI 文档；
   - 健康检查与日志接口位于 `/health`、`/log`。

2. **启动前端（可选）**

   ```bash
   cd web
   npm install
   npm run dev
   ```

   默认前端运行于 `http://localhost:5173`，可通过 `.env` 或配置文件调整 API 地址。

3. **运行测试与诊断**

   - 使用 `pytest` 运行后端单元测试：`python -m pytest -v`；
   - `scripts/deployment/run.py` 内置依赖检查，可用于部署前验证环境；
   - 若在当前环境无法运行测试，请在部署文档中提醒使用者在本地执行。

## 6. 常见问题与排障

- **模型无法加载**：检查 `.env` 中的 `MODEL_DIR`、`LLM_MODEL_NAME` 是否正确，确认模型文件已下载到本地。
- **向量索引为空**：确认已执行数据预处理脚本并在配置中指定正确的数据集路径。
- **Neo4j 连接失败**：确保数据库服务已启动、网络端口开放，`NEO4J_*` 环境变量与数据库配置一致。
- **第三方 API 调用受限**：确认对应 API Key 已在 `.env` 中设置，且账户额度充足。
- **日志排查**：后端日志默认输出到 `log/` 目录，可通过 `/log` 接口查看最新日志片段。

## 7. 贡献指南

- 在提交 Pull Request 前请确保代码通过 `pytest` 与必要的静态检查；
- 统一使用 `src/utils/logger.py` 提供的 `get_logger` 创建日志，避免重复配置；
- 新增服务请放置在 `src/services/` 并编写相应单元测试或集成测试。

## 8. 许可证

项目基于 [MIT License](../LICENSE) 开源，使用时请保留原始许可证文件与版权声明。
