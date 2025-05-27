# 🌊 蓝海智询 - 手动操作步骤清单

## 📋 必需操作步骤

### 1. 环境准备
```bash
# 1.1 检查Python版本 (需要 3.8+)
python --version

# 1.2 创建虚拟环境
python -m venv venv

# 1.3 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 1.4 安装依赖
pip install -r requirements.txt
```

### 2. 数据库配置
```bash
# 2.1 安装并启动Neo4j
# 下载: https://neo4j.com/download/
# 启动Neo4j服务，确保端口7687可用

# 2.2 (可选) 安装并启动Redis
# 用于缓存，提升性能
```

### 3. 模型准备
```bash
# 3.1 创建模型目录
mkdir -p models

# 3.2 下载OceanGPT模型 (约7GB)
git lfs install
git clone https://huggingface.co/zjunlp/OceanGPT-o-7B-v0.1 models/OceanGPT-o-7B-v0.1
```

### 4. 快速启动
```bash
# 4.1 运行环境检查
python quick_start.py

# 4.2 启动系统
python src/main.py

# 或使用启动脚本:
# Windows: 双击 start.bat
# Linux/Mac: ./start.sh
```

### 5. 验证系统
访问以下URL确认系统正常运行：
- 🏥 API文档: http://localhost:8000/docs
- ❤️ 健康检查: http://localhost:8000/health
- ℹ️ 系统信息: http://localhost:8000/info

## 🔧 可选操作步骤

### 6. 初始化数据
```bash
# 6.1 初始化知识图谱
curl -X POST "http://localhost:8000/api/v1/admin/init-knowledge-graph"

# 6.2 导入样本疾病数据
curl -X POST "http://localhost:8000/api/v1/data/import" \
  -F "file=@datasets/marine_diseases/sample_diseases.json"
```

### 7. 测试核心功能
```bash
# 7.1 测试疾病诊断
curl -X POST "http://localhost:8000/api/v1/diagnosis/diagnose" \
  -H "Content-Type: application/json" \
  -d '{
    "animal_type": "鱼类",
    "symptoms": ["体表白点", "游泳异常"],
    "environment": {"temperature": 25.0, "ph": 7.2}
  }'

# 7.2 测试知识检索
curl -X POST "http://localhost:8000/api/v1/rag/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "鱼类白点病治疗方法", "top_k": 5}'
```

### 8. 启动模型训练 (可选)
```bash
# 8.1 生成训练数据
curl -X POST "http://localhost:8000/api/v1/training/data/generate?count=1000"

# 8.2 开始微调训练
curl -X POST "http://localhost:8000/api/v1/training/start" \
  -H "Content-Type: application/json" \
  -d '{
    "use_generated_data": true,
    "generated_data_count": 1000,
    "num_train_epochs": 3
  }'
```

### 9. 前端部署 (可选)
```bash
# 9.1 安装Node.js依赖
cd web
npm install

# 9.2 启动开发服务器
npm run dev

# 9.3 生产构建
npm run build
```

## ⚡ 快速命令参考

| 操作 | Windows | Linux/Mac |
|------|---------|-----------|
| 启动系统 | `start.bat` | `./start.sh` |
| 环境检查 | `python quick_start.py` | `python quick_start.py` |
| 手动启动 | `python src/main.py` | `python src/main.py` |
| 激活环境 | `venv\Scripts\activate` | `source venv/bin/activate` |

## 🚨 常见问题解决

### 1. 模型加载失败
- **原因**: GPU内存不足或模型文件缺失
- **解决**: 
  - 检查GPU内存: `nvidia-smi`
  - 使用CPU模式: 在config.json中设置 `"device": "cpu"`
  - 重新下载模型文件

### 2. 数据库连接失败
- **原因**: Neo4j服务未启动
- **解决**: 
  - 启动Neo4j: `neo4j console`
  - 检查端口7687是否被占用
  - 修改数据库配置

### 3. 依赖包安装失败
- **原因**: 网络问题或Python版本不兼容
- **解决**: 
  - 使用国内镜像: `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`
  - 升级pip: `pip install --upgrade pip`
  - 检查Python版本是否>=3.8

### 4. 前端无法访问
- **原因**: CORS配置或端口问题
- **解决**: 
  - 检查后端服务是否启动
  - 确认端口8000未被占用
  - 检查防火墙设置

## 📊 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| Python | 3.8+ | 3.9+ |
| 内存 | 8GB | 16GB+ |
| 硬盘 | 20GB | 50GB+ |
| GPU | 无 | NVIDIA RTX 3060+ |
| 操作系统 | Windows 10/Linux/Mac | Ubuntu 20.04+ |

## 📞 技术支持

如遇问题，请按以下顺序排查：

1. **查看日志**: `logs/app.log`
2. **运行诊断**: `python quick_start.py`
3. **检查服务**: 访问 http://localhost:8000/health
4. **参考文档**: `DEPLOYMENT_GUIDE.md`
5. **联系团队**: support@blueocean.ai

---

**蓝海智询团队** 📧 support@blueocean.ai 🌐 http://blueocean.ai 