# Blueocean-rag 管理指南 v1.0 (2025-10-23)

## 目录
- [项目结构与关键文件](#项目结构与关键文件)
- [模块关系图](#模块关系图)
- [快速上手](#快速上手)
- [新手常见问题 FAQ](#新手常见问题-faq)
- [贡献指南](#贡献指南)

## 项目结构与关键文件
以下条目按目录划分，为每个文件夹及关键文件提供一句话简介，帮助快速理解各组件职责。

### 根目录
- `README.md`：中文概览文档，介绍系统特性、快速开始与目录结构。【F:README.md†L1-L95】
- `config.json`：集中管理模型、检索、知识图谱与数据源的默认配置，可由脚本动态修改。【F:config.json†L1-L162】
- `requirements.txt`：列出后端运行所需的 Python 依赖，涵盖 FastAPI、LLM、图数据库与可选组件。【F:requirements.txt†L1-L74】
- `download_model.py`：最小化的 ModelScope 下载脚本，便于快速拉取 OceanGPT 模型；可按需忽略。【F:download_model.py†L1-L15】
- `monitor_download.py`：监控模型下载进度的工具脚本，配合模型管理脚本使用（可选）。【F:monitor_download.py†L1-L75】
- `LICENSE`：MIT 开源协议文本。

### `data/`
- `README.md`：说明运行时生成的向量库、知识图谱、上传缓存等存放规则，提醒勿提交到仓库。【F:data/README.md†L1-L12】

### `datasets/`
- `README.md`：描述自备数据集的放置方式与 `.env` 配置提示。【F:datasets/README.md†L1-L9】
- `fish_docs/`：默认的原始鱼病文档占位目录，初始为空等待用户导入。
- `marine_diseases/`：结构化疾病知识占位目录，可按需自定义格式。

### `models/`
- `README.md`：指导如何下载 OceanGPT 及相关嵌入/重排模型，并提供推荐目录结构。【F:models/README.md†L1-L45】
- 其他子目录：用于存放真实模型文件，仓库中默认留空。

### `docs/`
- `PROJECT_MANUAL.md` 等：详述架构、API 与授权策略的官方文档（供深入阅读）。
- `api.md`、`auth.md`、`index.md`：Jekyll 站点页面源码，可选用于静态站点发布。
- `Gemfile`、`_config.yml`：Jekyll 文档站配置文件，非运行必需。

### `images/`
- `*.png`、`demo.gif`：宣传与文档插图素材，可用于产品演示或 README 展示。

### `scripts/`
- `data_collection/crawl_ocean_api.py`：示例爬虫脚本，抓取海洋相关数据源（可选）。
- `data_processing/parse_fish_data.py`：解析鱼病原始资料为结构化文本；`tag_and_index.py`：打标签并构建索引用于知识库更新。【F:scripts/data_processing/parse_fish_data.py†L1-L24】【F:scripts/data_processing/tag_and_index.py†L1-L20】
- `deployment/run.py`：CLI 启动器，检查依赖后以 Uvicorn 方式拉起 FastAPI 服务。【F:scripts/deployment/run.py†L1-L83】
- `deployment/quick_start.py`：交互式环境自检与目录初始化工具，帮助新手排障。【F:scripts/deployment/quick_start.py†L1-L65】
- `deployment/run.sh`、`run_docker.sh`：Shell 封装启动脚本（可选）。
- `deployment/setup_project.py`：初始化工程结构的脚本（适用于批量部署）。
- `model/manage_oceangpt.py`：提供模型列举、下载、切换等命令行管理能力。【F:scripts/model/manage_oceangpt.py†L1-L72】
- `model/download_oceangpt.py`、`download_models.sh`：批量下载模型的辅助脚本（按需执行）。
- `model/vllm/main.py`、`run.sh`：使用 vLLM 部署 OceanGPT 的示例入口（可选）。

### `docker/`
- `api.Dockerfile`：构建后端服务镜像的基础 Dockerfile。
- `web.Dockerfile`：前端构建与开发镜像定义。
- `docker-compose.dev.yml`：开发态编排文件，集成 API、前端、Neo4j、Milvus、MinIO、Etcd 等组件。【F:docker/docker-compose.dev.yml†L1-L95】
- `nginx/`：反向代理配置模板，生产部署时可挂载。
- `test/test_neo4j.py`：容器内 Neo4j 连接测试脚本（可选排障）。

### `tests/`
- `test_auth.py`、`test_concurrency.py` 等：覆盖认证、并发与数据处理逻辑的 PyTest 用例。
- `data/`：包含红楼梦文本样本的测试数据，便于模拟索引构建流程（示例用途）。

### `web/`
- `package.json`：前端依赖清单，基于 Vue 3 + Vite。
- `vite.config.js`：Vite 构建配置，含 API 代理设置。
- `components/KnowledgeGraph.vue`、`SearchInterface.vue`：独立可嵌入组件，方便演示或外部集成。
- `src/main.js`：Vue 应用入口，挂载全局样式与路由。
- `src/App.vue`：根组件，负责布局容器。
- `src/layouts/AppLayout.vue`、`BlankLayout.vue`：定义含导航栏和空白容器的两种页面骨架。
- `src/views/ChatView.vue`、`DataBaseView.vue` 等：页面级视图，分别承载对话、知识库管理、图谱、工具与设置界面。
- `src/components/ChatComponent.vue`、`DebugComponent.vue`、`RefsComponent.vue`：前端核心组件，提供聊天窗口、调试信息与参考资料展示。
- `src/components/tools/ConvertToTxtComponent.vue`、`TextChunkingComponent.vue`：可选的文本预处理工具面板。
- `src/router/index.js`：声明路由表与布局嵌套。
- `src/stores/config.js`、`database.js`：Pinia 状态管理，缓存配置项与知识库列表。
- `src/utils/modelIcon.js`：根据模型提供方返回对应图标资源。
- `public/`：静态资源（图标、占位图），在构建时直接复制。

### `src/`
- `main.py`：FastAPI 应用入口，注册中间件、路由与生命周期钩子。【F:src/main.py†L1-L80】
- `entry.py`：命令行演示脚本，演示如何离线加载文档并与 OceanGPT 交互（示例用途）。【F:src/entry.py†L1-L80】
- `auth.py`：实现简单的 API-Key 认证依赖。【F:src/auth.py†L1-L11】
- `config/settings.py`：基于 `pydantic-settings` 的全局配置管理，自动创建数据与日志目录。【F:src/config/settings.py†L1-L76】
- `api/rag_api.py`：保留的兼容层，提示迁移到 `src.services`（弃用）。【F:src/api/rag_api.py†L1-L19】
- `api/config_api.py`：提供配置读写、模型信息查询等 REST 接口，支持自定义模型注册。【F:src/api/config_api.py†L1-L90】
- `routers/`：集中定义 FastAPI 路由模块；如 `rag_router.py` 负责问答接口，`chat_router.py` 管理对话流，`data_router.py` 处理文件上传索引，`diagnosis.py`、`ocean_env.py` 分别处理疾病诊断与环境数据，`admin.py`、`tool_router.py`、`training.py`、`stats.py` 则覆盖管理与工具端点。
- `services/rag.py`：RAG 服务层，负责加载/重建 FAISS 索引并调度 OceanGPT 生成回答。【F:src/services/rag.py†L1-L78】
- `core/`：底层能力模块集合：
  - `oceangpt_manager.py`：OceanGPT 模型加载、切换与推理的统一管理器。
  - `knowledgebase.py`：Milvus/向量库管理、文档入库与查询逻辑，支持重排与阈值控制。【F:src/core/knowledgebase.py†L1-L75】
  - `knowledge_graph.py`、`graphbase.py`：Neo4j 图谱交互与实体写入、向量索引构建。【F:src/core/graphbase.py†L1-L80】
  - `retriever.py`：对知识库、图谱、Web 搜索进行融合检索与查询重写。【F:src/core/retriever.py†L1-L87】
  - `indexing.py`：文档切分与 PDF/TXT 读取工具，封装 LlamaIndex 节点构建流程。【F:src/core/indexing.py†L1-L60】
  - `fine_tuning.py`、`history.py`、`water_analysis.py`：负责增量训练、会话记录与水质分析等扩展能力（按需启用）。
- `models/`：模型适配层：
  - `embedding.py`：封装 HuggingFace 等嵌入模型的加载。
  - `chat_model.py`：统一 OpenAI 兼容类 API、厂商 SDK 与自定义模型调用逻辑。【F:src/models/chat_model.py†L1-L60】
  - `rerank_model.py`、`index_model.py`：重排模型与索引模型适配器。
- `plugins/_ocr.py`：基于 RapidOCR 的 PDF/OCR 插件，需要额外模型文件；`oneke.py`：示例插件扩展。
- `rag/`：实验性 RAG 抽象层，提供 `rag_engine.py`、`pipeline.py`、`embedder.py`、`vector_store.py` 等接口，用于自定义检索流水线。【F:src/rag/rag_engine.py†L1-L60】
- `utils/logger.py`：统一日志工具，支持文件轮转与 Rich 控制台输出。【F:src/utils/logger.py†L1-L52】
- `utils/prompts.py`：预置提示词模板，涵盖知识库问答、实体抽取与关键词提取。【F:src/utils/prompts.py†L1-L27】
- `utils/web_search.py`：Tavily Web 搜索封装，默认关闭（需设置 API Key）。【F:src/utils/web_search.py†L1-L33】
- `static/`：自定义 Swagger/Redoc 静态资源与模型图标配置。

## 模块关系图
用户请求在系统中的核心流向如下：

`前端 Vue 客户端 → FastAPI 路由 (src/routers/*) → 服务层 (src/services/rag.RAGService 等) → 核心模块 (src/core/knowledgebase & graphbase & retriever) → 模型适配层 (src/models/* & src/core/oceangpt_manager) → 外部依赖 (FAISS/Milvus、Neo4j、OceanGPT 模型、Tavily Web 搜索)`

## 快速上手

### 1. 克隆仓库与环境准备
```bash
git clone https://github.com/GodCzy/Blueocean-rag.git
cd Blueocean-rag
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
> 如需一键检查依赖与目录，可运行 `python scripts/deployment/quick_start.py` 执行自检与初始化。【F:scripts/deployment/quick_start.py†L1-L65】

### 2. 配置 `.env`
1. 复制模板：`cp .env.template .env`（若仓库未包含模板，可直接新建）。
2. 核心配置项（参考 `Settings` 与 `config.json`）【F:src/config/settings.py†L1-L76】【F:config.json†L1-L162】：
   - `MODEL_DIR`：模型根目录，默认为 `<项目>/models`，生产环境建议指向挂载盘。
   - `API_KEY`：后端 API-Key（若留空则路由默认开放，建议生产环境务必设置）。
   - `NEO4J_URI`/`NEO4J_USERNAME`/`NEO4J_PASSWORD`：启用知识图谱时的连接信息；本地 Docker 默认为 `bolt://localhost:7687`、`neo4j/0123456789`。
   - `MILVUS_URI` 或相关凭据：使用 Milvus 向量库时需要；如未启用可留空。
   - `TAVILY_API_KEY`：启用 Web 搜索时必填，否则 `src/utils/web_search.py` 会抛错。
   - 其它第三方模型密钥（`OPENAI_API_KEY`、`DASHSCOPE_API_KEY` 等）按需填写。
3. 安全提示：`.env` 与模型目录包含敏感信息，请勿提交到版本库；团队协作时建议借助密钥管理工具。

### 3. 准备模型与数据
- 运行 `python scripts/model/manage_oceangpt.py list` 查看可选模型，并使用 `download`/`switch` 子命令同步 `config.json`。【F:scripts/model/manage_oceangpt.py†L1-L72】
- 模型文件应放入 `models/`，数据文档放入 `datasets/`（可用 `scripts/data_processing/tag_and_index.py` 构建索引）。【F:scripts/data_processing/tag_and_index.py†L1-L20】
- 若暂时没有模型，可先运行 `python scripts/deployment/run.py --no-reload` 体验空索引模式，OceanGPT 将在首次调用时加载失败并回落。

### 4. 启动后端
```bash
python scripts/deployment/run.py --host 0.0.0.0 --port 8000
```
成功后访问 `http://localhost:8000/docs` 查看 Swagger UI；若希望使用自定义静态页，可访问 `http://localhost:8000/redoc`。

### 5. 启动前端（可选）
```bash
cd web
npm install
npm run dev
```
默认开发地址为 `http://localhost:5173`，可在 `.env` 或 `VITE_API_URL` 中调整 API 代理。

### 6. 可选组件与跳过方式
- **Neo4j**：在 `.env` 中关闭 `enable_knowledge_graph` 或注释 Docker Compose 的 `graph` 服务即可跳过。【F:config.json†L1-L54】
- **Milvus/向量库**：若不想使用 Milvus，可保持 `enable_knowledge_base=true` 让系统回退至内置 FAISS，或将其设为 `false` 完全禁用知识库功能。【F:config.json†L1-L54】
- **Docker 全家桶**：运行 `docker compose -f docker/docker-compose.dev.yml up` 可一次启动所有依赖；如只需后端，可注释 `milvus`/`graph` 服务并在 `.env` 中禁用对应能力。【F:docker/docker-compose.dev.yml†L1-L95】
- **Web 搜索**：默认关闭，如需启用需设置 `TAVILY_API_KEY`，并在 `config.json` 中打开 `enable_web_search`。

## 新手常见问题 FAQ

**Q1：启动时报 Neo4j 连接失败怎么办？**  
A：确认 `.env` 中的 `NEO4J_URI` 与账号密码正确，并检查 Docker Compose 中 `graph` 服务是否启动；若无需知识图谱，可在 `config.json` 中将 `enable_knowledge_graph` 设为 `false` 以跳过。【F:src/core/graphbase.py†L23-L43】【F:config.json†L1-L54】

**Q2：`models/` 目录为空，服务能启动吗？**  
A：可以启动，但 OceanGPT 首次调用会失败并回退到检索模式；建议使用 `scripts/model/manage_oceangpt.py download <model>` 或 `download_model.py` 拉取推荐模型。【F:src/services/rag.py†L1-L78】【F:download_model.py†L1-L15】

**Q3：Swagger 页显示 401，如何通过认证？**  
A：`src/auth.py` 默认使用 `X-API-Key` 头校验 `.env` 中的 `API_KEY`，访问 Swagger 时需在右上角 Authorize 输入同样的 Key；若不填 `.env`，认证会被自动放行。【F:src/auth.py†L1-L11】

**Q4：知识库检索不到文档？**  
A：确认 `datasets/` 下存在文本，并使用 `scripts/data_processing/tag_and_index.py` 或 `scripts/model/manage_oceangpt.py rebuild` 构建索引；若仍为空，检查 `config.json` 中的向量库路径及 `enable_knowledge_base` 开关。【F:src/core/indexing.py†L1-L60】【F:src/services/rag.py†L1-L78】

**Q5：Tavily API Key 缺失导致 Web 搜索报错？**  
A：`src/utils/web_search.py` 在初始化时会检查 `TAVILY_API_KEY`，若不打算使用网络搜索，可保持 `enable_web_search=false` 并忽略此错误。【F:src/utils/web_search.py†L1-L33】【F:config.json†L1-L54】

**Q6：如何新增数据集或多份知识源？**  
A：将新文档放入 `datasets/` 子目录，运行 `scripts/data_processing/parse_fish_data.py` 做清洗，再执行 `tag_and_index.py` 或后端的 `/rag/rebuild` 接口重建索引；必要时可在 `config.json` 的 `data_sources` 中登记路径。【F:scripts/data_processing/parse_fish_data.py†L1-L20】【F:config.json†L163-L220】

**Q7：Docker Compose 启动报 `MODEL_DIR` 卷未定义？**  
A：按 `docker/docker-compose.dev.yml` 注释提示，在 `.env` 或 shell 中导出 `MODEL_DIR=/path/to/models`，或临时注释该 volume 映射以使用容器内默认路径。【F:docker/docker-compose.dev.yml†L1-L22】

## 贡献指南

1. **提出 Issue**：使用 GitHub Issue 模板描述问题或改进建议，附带复现步骤、日志与环境信息。
2. **分支命名**：推荐格式 `feature/<topic>`、`fix/<bug-id>`、`docs/<scope>`、`chore/<task>` 以便识别工作类型。
3. **提交规范**：保持原子化提交，Commit Message 建议遵循 `type(scope): summary`（如 `feat(rag): support milvus fallback`）。
4. **代码风格**：Python 遵循 PEP8（120 字符软限制），前端遵循 ESLint/Prettier 默认规则；新增模块需接入 `src/utils/logger.get_logger` 做统一日志处理。【F:src/utils/logger.py†L1-L52】
5. **测试要求**：为新增功能补充 `tests/` 下的 PyTest 用例，前端使用 `npm run test`（如引入新逻辑）；提交前运行 `python -m pytest -v` 并确保通过。
6. **Pull Request**：在 PR 描述中列出变更摘要、测试结果及影响范围，如涉及配置变动请附升级指引；维护者会依据 `docs/PROJECT_MANUAL.md` 的流程进行 Review。
7. **文档更新**：若调整接口或部署流程，请同步更新 `README.md`、`docs/PROJECT_MANUAL.md` 或本指南中的相关段落，确保用户获取到最新指引。

> 感谢对 Blueocean-rag 的支持！欢迎在 Issue 与社区中分享行业经验与使用反馈。
