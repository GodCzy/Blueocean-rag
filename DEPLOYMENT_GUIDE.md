# 蓝海智询系统部署指南

> 如果只需要查看简化的手动步骤，请参阅 [MANUAL_STEPS.md](MANUAL_STEPS.md)。

## 🚀 完整部署流程

### 1. 环境准备

#### 1.1 系统要求
- Python 3.8+
- Node.js 16+
- GPU (推荐，支持CUDA)
- 内存 16GB+
- 硬盘空间 50GB+

#### 1.2 安装基础环境
```bash
# 检查Python版本
python --version

# 检查Node.js版本
node --version

# 检查CUDA (可选)
nvidia-smi
```

### 2. 项目初始化

#### 2.1 克隆或创建项目目录
```bash
mkdir Blueocean-rag
cd Blueocean-rag
```

#### 2.2 创建Python虚拟环境
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

#### 2.3 安装Python依赖
```bash
pip install -r requirements.txt
```

### 3. 数据库配置

#### 3.1 安装Neo4j数据库
```bash
# 下载Neo4j Community Edition
# 访问: https://neo4j.com/download/

# 启动Neo4j
# Windows: 双击 neo4j.bat
# Linux: ./bin/neo4j start
```

#### 3.2 配置Redis (可选)
```bash
# 安装Redis
# Windows: 下载Redis for Windows
# Linux: sudo apt-get install redis-server
# Mac: brew install redis

# 启动Redis
redis-server
```

### 4. 模型准备

#### 4.1 下载OceanGPT模型
```bash
# 创建模型目录
mkdir -p models/OceanGPT-o-7B-v0.1

# 下载模型文件 (需要Git LFS)
git lfs install
git clone https://huggingface.co/zjunlp/OceanGPT-o-7B-v0.1 models/OceanGPT-o-7B-v0.1
```

#### 4.2 配置模型路径
编辑 `config.json` 文件：
```json
{
  "models": {
    "oceangpt": {
      "model_path": "./models/OceanGPT-o-7B-v0.1",
      "device": "auto"
    }
  }
}
```

### 5. 前端构建

#### 5.1 安装前端依赖
```bash
cd web
npm install
```

#### 5.2 构建前端
```bash
# 开发模式
npm run dev

# 生产构建
npm run build
```

### 6. 启动系统

#### 6.1 启动后端服务
```bash
# 回到项目根目录
cd ..

# 启动FastAPI服务
python src/main.py
```

#### 6.2 验证服务启动
访问以下URL验证系统启动：
- API文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health
- 系统信息: http://localhost:8000/info

### 7. 数据初始化

#### 7.1 初始化知识图谱
```bash
# 通过API初始化
curl -X POST "http://localhost:8000/api/v1/admin/init-knowledge-graph"
```

#### 7.2 导入疾病数据
```bash
# 上传样本数据
curl -X POST "http://localhost:8000/api/v1/data/import" \
  -F "file=@datasets/marine_diseases/sample_diseases.json"
```

### 8. 功能测试

#### 8.1 测试疾病诊断API
```bash
curl -X POST "http://localhost:8000/api/v1/diagnosis/diagnose" \
  -H "Content-Type: application/json" \
  -d '{
    "animal_type": "鱼类",
    "symptoms": ["体表白点", "游泳异常"],
    "environment": {
      "temperature": 25.0,
      "ph": 7.2
    }
  }'
```

#### 8.2 测试RAG检索
```bash
curl -X POST "http://localhost:8000/api/v1/rag/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "鱼类白点病治疗方法",
    "top_k": 5
  }'
```

### 9. 微调训练 (可选)

#### 9.1 准备训练数据
```bash
# 生成训练数据
curl -X POST "http://localhost:8000/api/v1/training/data/generate?count=1000"
```

#### 9.2 开始训练
```bash
curl -X POST "http://localhost:8000/api/v1/training/start" \
  -H "Content-Type: application/json" \
  -d '{
    "use_generated_data": true,
    "generated_data_count": 1000,
    "num_train_epochs": 3
  }'
```

### 10. 生产部署

#### 10.1 使用Docker部署
```bash
# 构建Docker镜像
docker build -t blueocean-rag .

# 运行容器
docker run -p 8000:8000 blueocean-rag
```

#### 10.2 使用Nginx反向代理
创建 `/etc/nginx/sites-available/blueocean` 配置：
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🔧 故障排除

### 常见问题

#### 1. 模型加载失败
- 检查GPU内存是否足够
- 尝试使用CPU模式：设置 `device: "cpu"`
- 检查模型文件是否完整

#### 2. 数据库连接失败
- 确认Neo4j服务已启动
- 检查连接配置和端口
- 验证用户名密码

#### 3. 前端无法访问后端
- 检查CORS配置
- 确认端口号正确
- 检查防火墙设置

#### 4. 内存不足
- 减小批次大小
- 启用梯度检查点
- 使用模型量化

### 性能优化

#### 1. GPU优化
```python
# 在config.json中配置
{
  "models": {
    "oceangpt": {
      "load_in_4bit": true,
      "use_flash_attention": true
    }
  }
}
```

#### 2. 并发优化
```python
# 增加worker数量
uvicorn.run("src.main:app", workers=4)
```

## 📋 系统检查清单

### 部署前检查
- [ ] Python环境已配置
- [ ] 依赖包已安装
- [ ] 数据库已启动
- [ ] 模型文件已下载
- [ ] 配置文件已修改

### 启动后检查
- [ ] API服务正常响应
- [ ] 数据库连接成功
- [ ] 模型加载完成
- [ ] 前端页面可访问
- [ ] 核心功能可用

### 功能验证
- [ ] 疾病诊断接口
- [ ] 知识检索接口
- [ ] 聊天对话接口
- [ ] 数据管理接口
- [ ] 训练管理接口

## 📞 技术支持

如遇问题，请：
1. 查看应用日志：`logs/app.log`
2. 检查系统状态：访问 `/health` 接口
3. 查看详细错误信息
4. 联系技术团队

---
**蓝海智询团队**  
📧 support@blueocean.ai  
🌐 http://blueocean.ai 