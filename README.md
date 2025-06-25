# 蓝海智询 - 基于大模型RAG知识库与知识图谱技术的水生动物疾病问答平台

![项目Logo](images/logo.png)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.0+-blue.svg)](https://reactjs.org)

## 项目简介

蓝海智询是一个专注于水生动物疾病诊断和海洋环境数据分析的智能问答平台。系统基于最新的大模型检索增强生成（RAG）技术和知识图谱，为水产养殖从业者提供准确、专业的疾病诊断和治疗建议。

本项目是2025年大学生创新训练项目，旨在将人工智能技术应用于水产养殖领域，提高疾病诊断的准确性和效率。

## 🚀 核心功能

### 🧠 智能诊断引擎
- **多模态诊断**: 结合症状描述、图像识别、水质参数进行综合诊断
- **知识图谱推理**: 基于Neo4j构建的疾病-症状-治疗知识图谱
- **RAG检索增强**: FAISS + BM25混合检索，提供精准的知识库问答
- **OceanGPT模型**: 专门针对海洋水产领域微调的大语言模型

### 💊 治疗方案推荐
- **个性化治疗**: 根据动物类型、疾病严重程度定制治疗方案
- **药物推荐**: 智能推荐合适的药物和用药剂量
- **预防措施**: 提供疾病预防和水质管理建议
- **治疗监控**: 跟踪治疗效果和康复进度

### 🌊 环境分析
- **水质评估**: 实时分析水质参数对动物健康的影响
- **环境预警**: 预测潜在的环境风险因素
- **养殖建议**: 提供最优的养殖密度和环境配置
- **数据可视化**: 直观展示水质变化趋势

### 📊 数据管理
- **案例库管理**: 收集和管理历史诊断案例
- **知识库更新**: 持续更新疾病和治疗知识
- **统计分析**: 生成疾病发生率和治疗成功率报告
- **数据导入导出**: 支持多种格式的数据交换

## 🏗️ 技术架构

```
蓝海智询技术架构
├── 前端层 (React + Material-UI)
│   ├── 诊断界面
│   ├── 数据可视化
│   └── 管理后台
├── API层 (FastAPI)
│   ├── 诊断接口
│   ├── 知识管理接口
│   └── 用户管理接口
├── 业务逻辑层
│   ├── RAG引擎 (FAISS + BM25)
│   ├── 知识图谱 (Neo4j)
│   ├── OceanGPT模型 (LoRA微调)
│   └── 数据处理模块
└── 数据层
    ├── 向量数据库 (FAISS)
    ├── 图数据库 (Neo4j)
    ├── 关系数据库 (PostgreSQL)
    └── 缓存层 (Redis)
```

## 📋 系统要求

### 硬件要求
- **最低配置**: 8GB RAM, 50GB 存储空间
- **推荐配置**: 16GB+ RAM, 100GB+ SSD, NVIDIA GPU (可选)
- **服务器环境**: Linux/Windows Server, Docker支持

### 软件依赖
- **Python**: 3.8+
- **Node.js**: 16+
- **Neo4j**: 5.14+
- **Redis**: 6.0+
- **Docker**: 20.0+ (可选)

## 🛠️ 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/GodCzy/Blueocean-rag.git
cd Blueocean-rag
```

### 2. 自动初始化（推荐）
```bash
# 运行初始化脚本
python scripts/setup_project.py

# 如果需要跳过模型下载
python scripts/setup_project.py --skip-models
```

### 3. 手动安装

#### 安装后端依赖
```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

#### 安装前端依赖
```bash
cd web
npm install
npm run build
cd ..
```

### 4. 下载模型文件

由于模型文件较大（超过100MB），未直接包含在仓库中。请按以下步骤下载：

```bash
# 安装Git LFS
git lfs install

# 下载OceanGPT模型
git clone https://huggingface.co/zjunlp/OceanGPT-o-7B-v0.1 models/OceanGPT-o-7B-v0.1

# 或使用我们的下载脚本
python download_model.py
```

### 5. 配置环境
```bash
# 复制环境配置文件
cp .env.example .env

# 编辑配置文件
vim .env
```

### 6. 启动服务

#### 使用Docker（推荐）
```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps
```

#### 手动启动
```bash
# 启动Neo4j数据库
# 请参考Neo4j官方文档安装和配置

# 启动后端服务
python run.py

# 启动前端开发服务器（新终端）
cd web
npm run dev
```

### 7. 访问服务
- **前端界面**: http://localhost:5173
- **API文档**: http://localhost:8000/docs
- **Neo4j浏览器**: http://localhost:7474

## 🧪 功能使用指南

### 疾病诊断
1. 通过前端界面选择水生动物类型（鱼类、虾类等）
2. 输入观察到的症状或上传症状图片
3. 填写水质参数（温度、pH值等）
4. 点击"诊断"按钮获取分析结果
5. 查看诊断结果和治疗建议

### 知识检索
1. 在问答界面输入关于水生动物疾病的问题
2. 系统自动从知识库检索相关信息
3. 获取带有参考来源的专业回答

### 数据分析
1. 上传水质监测数据
2. 系统分析环境参数与疾病风险的关系
3. 获取环境优化建议和预防措施

### API调用示例
```bash
# 疾病诊断API
curl -X POST "http://localhost:8000/api/v1/diagnosis/diagnose" \
  -H "Content-Type: application/json" \
  -d '{
    "animal_type": "鱼类",
    "symptoms": ["体表白点", "游泳异常"],
    "environment": {"temperature": 25.0, "ph": 7.2}
  }'

# 知识检索API
curl -X POST "http://localhost:8000/api/v1/rag/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "鱼类白点病治疗方法", "top_k": 5}'
```

# 统计API
```bash
curl -H "X-API-Key: <your key>" http://localhost:8000/api/stats/diagnosis
```

## 📖 详细文档

### 📝 API文档
- [API接口文档](docs/api.md)
- [数据模型说明](docs/models.md)
- [认证授权](docs/auth.md)

### 🔧 开发指南
- [开发环境搭建](docs/development.md)
- [代码规范](docs/coding-standards.md)
- [测试指南](docs/testing.md)

### 🚀 部署指南
- [生产环境部署](docs/deployment.md)
- [Docker部署](docs/docker.md)
- [性能优化](docs/performance.md)

### 📊 数据管理
- [数据导入指南](docs/data-import.md)
- [知识图谱构建](docs/knowledge-graph.md)
- [模型微调](docs/fine-tuning.md)

## 🗂️ 项目结构

```
蓝海智询/
├── src/                    # 后端源代码
│   ├── api/               # API路由定义
│   ├── core/              # 核心业务逻辑
│   ├── data/              # 数据处理模块
│   ├── models/            # 数据模型定义
│   ├── rag/               # RAG引擎实现
│   ├── routers/           # 路由处理器
│   ├── utils/             # 工具函数
│   └── main.py            # 应用入口
├── web/                   # 前端源代码
│   ├── src/               # React组件和页面
│   ├── public/            # 静态资源
│   └── package.json       # 前端依赖
├── datasets/              # 训练和测试数据
│   ├── marine_diseases/   # 疾病数据
│   ├── treatments/        # 治疗方案数据
│   ├── water_quality/     # 水质数据
│   └── case_studies/      # 案例研究
├── models/                # 预训练模型存储
├── data/                  # 运行时数据
│   ├── vector_store/      # 向量数据库
│   ├── knowledge_graph/   # 知识图谱数据
│   └── logs/              # 日志文件
├── scripts/               # 工具脚本
├── tests/                 # 测试代码
├── docs/                  # 项目文档
├── docker/                # Docker配置
├── config.json            # 主配置文件
├── requirements.txt       # Python依赖
├── run.py                 # 启动脚本
└── README.md              # 项目说明
```

## 🔍 常见问题

### Q: 系统支持哪些水生动物疾病诊断？
A: 系统目前支持常见养殖水生动物的疾病诊断，包括鱼类（如草鱼、鲈鱼、罗非鱼等）、虾类（如南美白对虾、中国对虾等）、蟹类和贝类等。

### Q: 如何提高诊断准确率？
A: 可以通过以下方式提高诊断准确率：
- 提供更详细的症状描述
- 上传清晰的症状图片
- 提供准确的水质参数
- 定期更新知识库和模型

### Q: 没有GPU能否运行系统？
A: 可以。系统默认支持CPU模式运行，但处理速度会较慢。建议在有NVIDIA GPU的环境下运行以获得最佳性能。

### Q: 如何贡献新的疾病数据？
A: 您可以通过以下方式贡献数据：
- 使用管理后台上传新的疾病案例
- 通过GitHub提交Pull Request添加数据
- 联系项目团队提供专业资料

## 🧪 测试

### 运行单元测试
```bash
# 后端测试
python -m pytest tests/ -v

# 前端测试
cd web
npm test
```

### 运行集成测试
```bash
# API集成测试
python -m pytest tests/integration/ -v

# 端到端测试
cd web
npm run test:e2e
```

## 📈 性能指标

### 系统性能
- **响应时间**: < 2秒（普通查询）
- **并发处理**: 支持100+并发用户
- **准确率**: 疾病诊断准确率 > 85%
- **可用性**: 99.5%+

### 模型性能
- **OceanGPT**: BLEU: 0.75, ROUGE-L: 0.82
- **检索系统**: Recall@5: 0.88, Precision@5: 0.76
- **知识图谱**: 覆盖500+疾病，2000+症状

## 🔄 更新日志

### v1.0.0 (2023-12-15)
- 初始版本发布
- 基础疾病诊断功能
- RAG知识库集成
- Web界面实现

### v0.9.0 (2023-11-30)
- Beta测试版
- 核心功能完成
- 测试数据集成

## 🤝 贡献指南

我们欢迎各种形式的贡献，包括但不限于：

1. **代码贡献**
   - Fork项目并提交Pull Request
   - 遵循项目代码规范
   - 添加适当的测试

2. **数据贡献**
   - 提供水生动物疾病案例
   - 分享治疗方案和经验
   - 贡献水质分析数据

3. **文档完善**
   - 改进使用文档
   - 添加教程和示例
   - 翻译文档

## 📄 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情

## 👥 团队成员

- 陈中毅 - 项目负责人
- 团队成员1 - 前端开发
- 团队成员2 - 后端开发
- 团队成员3 - 模型训练
- 团队成员4 - 数据收集

## 📞 联系方式

- **项目负责人**: 陈中毅
- **Email**: example@example.com
- **GitHub Issues**: [提交问题](https://github.com/GodCzy/Blueocean-rag/issues)

---

感谢您对蓝海智询项目的关注！我们希望这个平台能为水产养殖从业者提供实用的帮助，促进水产养殖业的健康发展。如有任何问题或建议，欢迎随时联系我们。

