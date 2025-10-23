# 环境初始化与配置指引

本文档帮助首次部署蓝海智询（Blueocean RAG）的同学完成基础环境搭建、`.env` 环境变量配置与可选外部服务准备工作。

## 1. 基础环境要求

| 组件 | 版本建议 | 说明 |
| --- | --- | --- |
| Python | 3.8 及以上 | 后端运行环境，建议使用虚拟环境隔离依赖。 【F:README.md†L27-L65】|
| Node.js | 16 及以上 | 仅在使用 `web/` 前端示例时需要。 【F:README.md†L27-L87】|
| pip/venv | 最新 | 建议 `python -m venv venv && source venv/bin/activate` 创建虚拟环境。 【F:README.md†L39-L65】|
| CUDA 驱动 | 11.x+（可选） | 若需 GPU 加速模型推理，请提前安装匹配的 CUDA 与显卡驱动。 【F:README.md†L27-L65】|
| Docker & Docker Compose | 最新稳定版（可选） | 便于一键启动 Neo4j、Milvus 等外部服务。 |

> 📌 建议在克隆项目后执行 `pip install -r requirements.txt` 安装后端依赖。 【F:README.md†L39-L67】

## 2. `.env` 文件准备

1. 复制模板：
   ```bash
   cp .env.template .env
   ```
   如仓库未包含模板，可手动新建 `.env` 并按照下表填写。 【F:README.md†L39-L67】
2. `.env` 位于项目根目录，默认不会提交至版本库，请妥善保管。
3. 推荐在团队协作中使用密钥管理工具（如 1Password、Vault）共享敏感配置。

## 3. 核心环境变量说明

| 变量 | 作用 | 默认值/参考 | 备注 |
| --- | --- | --- | --- |
| `MODEL_DIR` | OceanGPT 及嵌入模型存放根目录。 | `./models` | 生产环境建议指向挂载盘或共享存储。 【F:src/config/settings.py†L32-L75】|
| `API_KEY` | 后端路由访问密钥。留空时接口默认开放。 | 空 | 生产环境务必设置，前端请求需携带。 【F:src/config/settings.py†L36-L56】|
| `DEBUG` | 是否开启调试模式。 | `false` | 开发环境可设为 `true`。 【F:docs/PROJECT_MANUAL.md†L43-L88】|
| `LLM_MODEL_NAME` | 默认加载的大语言模型。 | `internlm/internlm2-chat-7b` | 可与 `config.json` 中 `model_name` 协同调整。 【F:src/config/settings.py†L42-L75】【F:config.json†L1-L40】|
| `EMBEDDING_MODEL_NAME` | 嵌入模型名称。 | `BAAI/bge-large-zh-v1.5` | 需与向量库维度匹配。 【F:src/config/settings.py†L42-L75】【F:config.json†L1-L22】|
| `NEO4J_URI` | Neo4j Bolt 连接地址。 | `bolt://localhost:7687` | Docker Compose 开发默认 `bolt://graph:7687`。 【F:src/core/graphbase.py†L32-L44】【F:docker/docker-compose.dev.yml†L15-L62】|
| `NEO4J_USERNAME` | Neo4j 用户名。 | `neo4j` | 与数据库创建的用户保持一致。 【F:src/core/graphbase.py†L32-L44】|
| `NEO4J_PASSWORD` | Neo4j 密码。 | `0123456789`（本地示例） | 建议在生产环境修改。 【F:src/core/graphbase.py†L32-L44】【F:docker/docker-compose.dev.yml†L15-L62】|
| `MILVUS_URI` 或 `MILVUS_HOST`/`MILVUS_PORT` | 连接 Milvus/AnyScale 等向量数据库。 | 空 | 仅在启用外部向量库时必填。 |
| `TAVILY_API_KEY` | Web 搜索 API 密钥。 | 空 | 未设置时 `src/utils/web_search.py` 会报错。 |
| `OPENAI_API_KEY`、`DASHSCOPE_API_KEY` 等 | 第三方模型或服务密钥。 | 空 | 按需填写，避免提交到仓库。 |

> ℹ️ 更多变量可参考 `docs/PROJECT_MANUAL.md` 与 `scripts/deployment/setup_project.py`。其中 `settings.py` 会在未设置时为若干路径填充默认值。 【F:docs/PROJECT_MANUAL.md†L43-L104】【F:scripts/deployment/setup_project.py†L260-L410】

## 4. 外部服务与可选组件

### 4.1 Neo4j 知识图谱

- **用途**：存储疾病、症状、治疗等实体关系，支撑图谱检索。启用需在 `config.json` 中保持 `enable_knowledge_graph: true`。 【F:config.json†L1-L60】
- **本地快速启动**：
  ```bash
  docker compose -f docker/docker-compose.dev.yml up -d graph
  ```
  该服务默认暴露 `bolt://localhost:7687`，账号密码为 `neo4j/0123456789`，并启用了 APOC 插件。 【F:docker/docker-compose.dev.yml†L15-L64】
- **配置校验**：确保 `.env` 中 `NEO4J_URI`、`NEO4J_USERNAME`、`NEO4J_PASSWORD` 与运行实例一致。

### 4.2 向量数据库

- **默认选项**：项目默认使用本地 FAISS 索引（无需额外服务）。相关参数见 `config.json` 的 `vector_store` 配置。 【F:config.json†L60-L112】
- **Milvus/其他托管服务**：若需分布式向量库，可部署 Milvus，并在 `.env` 中填充 `MILVUS_URI`（或主机/端口、Access Key 等）。Docker 部署通常包含 `minio`、`etcd` 等依赖，请参考官方文档。

### 4.3 Web 搜索

- 需要外部 API 密钥（如 Tavily）。未配置 `TAVILY_API_KEY` 时请在 `config.json` 中将 `enable_web_search` 设置为 `false` 以避免运行时报错。 【F:config.json†L1-L24】

### 4.4 前端开发环境（可选）

- 进入 `web/` 目录后执行 `npm install && npm run dev` 即可启动前端，默认连接后端 `http://localhost:8000`。请确保 `.env` 中的 `API_KEY` 与前端环境变量保持一致。 【F:README.md†L65-L107】

## 5. 数据与模型准备

- 推荐使用 `scripts/model/manage_oceangpt.py` 列出、下载并切换 OceanGPT 模型，具体流程见 `README.md`。 【F:README.md†L65-L107】
- `MODEL_DIR` 指向的目录需有足够存储空间；若使用外部挂载盘，请在 `.env` 中改写路径，并确保容器或运行用户拥有读写权限。 【F:src/config/settings.py†L32-L75】
- 自备数据集放入 `datasets/` 对应子目录，并通过 `scripts/data_processing/` 下脚本构建向量索引或知识图谱。

## 6. 安全与最佳实践

- `.env`、`models/`、`data/` 目录包含敏感信息或大文件，默认已在 `.gitignore` 中排除，请勿手动提交。 【F:README.md†L39-L87】
- 生产环境需设置强密码、限制网络访问，并考虑通过反向代理为 API 添加额外认证。
- 将 API 密钥存储在专业的密钥管理服务中，定期轮换。

---

完成上述配置后，可运行 `python scripts/deployment/run.py --host 0.0.0.0 --port 8000` 启动后端，并访问 `http://localhost:8000/docs` 进行验证。祝部署顺利！ 【F:README.md†L65-L107】
