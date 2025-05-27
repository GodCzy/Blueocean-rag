#!/usr/bin/env python3
"""
蓝海智询项目初始化脚本
用于自动设置环境、下载模型、初始化数据库等
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
import argparse
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProjectSetup:
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.models_dir = self.project_root / "models"
        self.data_dir = self.project_root / "data"
        self.datasets_dir = self.project_root / "datasets"
        
        # 加载配置
        config_path = self.project_root / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
    def create_directories(self):
        """创建必要的目录结构"""
        logger.info("创建项目目录结构...")
        
        directories = [
            self.models_dir,
            self.data_dir / "vector_store",
            self.data_dir / "knowledge_graph",
            self.data_dir / "logs",
            self.datasets_dir / "marine_diseases",
            self.datasets_dir / "treatments",
            self.datasets_dir / "water_quality",
            self.datasets_dir / "case_studies",
            self.datasets_dir / "fine_tuning",
            self.datasets_dir / "ocean_instruct",
            "logs",
            "saves/models",
            "saves/checkpoints"
        ]
        
        for directory in directories:
            path = self.project_root / directory if isinstance(directory, str) else directory
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"创建目录: {path}")
    
    def install_dependencies(self):
        """安装Python依赖"""
        logger.info("安装Python依赖...")
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True, cwd=self.project_root)
            logger.info("Python依赖安装完成")
        except subprocess.CalledProcessError as e:
            logger.error(f"安装Python依赖失败: {e}")
            raise
    
    def install_frontend_dependencies(self):
        """安装前端依赖"""
        logger.info("安装前端依赖...")
        
        web_dir = self.project_root / "web"
        if not web_dir.exists():
            logger.warning("web目录不存在，跳过前端依赖安装")
            return
        
        try:
            subprocess.run(["npm", "install"], check=True, cwd=web_dir)
            logger.info("前端依赖安装完成")
        except subprocess.CalledProcessError as e:
            logger.error(f"安装前端依赖失败: {e}")
        except FileNotFoundError:
            logger.warning("未找到npm，跳过前端依赖安装")
    
    def download_models(self, source="modelscope"):
        """下载预训练模型"""
        logger.info("下载预训练模型...")
        
        # 基础嵌入模型
        base_models = [
            {
                "name": "bge-large-zh-v1.5",
                "modelscope_id": "AI-ModelScope/bge-large-zh-v1.5",
                "huggingface_id": "BAAI/bge-large-zh-v1.5",
                "local_path": self.models_dir / "bge-large-zh-v1.5"
            },
            {
                "name": "bge-reranker-large",
                "modelscope_id": "AI-ModelScope/bge-reranker-large",
                "huggingface_id": "BAAI/bge-reranker-large", 
                "local_path": self.models_dir / "bge-reranker-large"
            }
        ]
        
        # OceanGPT官方模型
        oceangpt_models = self.config.get("oceangpt_models", {})
        for model_key, model_info in oceangpt_models.items():
            if model_info.get("recommended", False):  # 只下载推荐的模型
                base_models.append({
                    "name": model_info["name"],
                    "modelscope_id": model_info["modelscope_id"],
                    "huggingface_id": model_info["huggingface_id"],
                    "local_path": self.models_dir / model_info["name"],
                    "is_oceangpt": True
                })
        
        for model in base_models:
            if model["local_path"].exists():
                logger.info(f"模型已存在: {model['name']}")
                continue
            
            logger.info(f"下载模型: {model['name']}")
            success = False
            
            # 尝试从指定源下载
            if source == "modelscope":
                success = self._download_from_modelscope(model["modelscope_id"], model["local_path"])
            else:
                success = self._download_from_huggingface(model["huggingface_id"], model["local_path"])
            
            # 如果失败，尝试备用源
            if not success:
                logger.warning(f"从{source}下载失败，尝试备用源...")
                if source == "modelscope":
                    success = self._download_from_huggingface(model["huggingface_id"], model["local_path"])
                else:
                    success = self._download_from_modelscope(model["modelscope_id"], model["local_path"])
            
            if success:
                logger.info(f"模型下载完成: {model['name']}")
            else:
                logger.error(f"模型下载失败: {model['name']}")
    
    def _download_from_modelscope(self, model_id: str, local_path: Path) -> bool:
        """从ModelScope下载模型"""
        try:
            from modelscope import snapshot_download
            
            logger.info(f"从ModelScope下载: {model_id}")
            snapshot_download(
                model_id=model_id,
                cache_dir=str(local_path.parent),
                local_dir=str(local_path)
            )
            return True
            
        except ImportError:
            logger.error("ModelScope未安装，请先安装: pip install modelscope")
            return False
        except Exception as e:
            logger.error(f"ModelScope下载失败: {e}")
            return False
    
    def _download_from_huggingface(self, model_id: str, local_path: Path) -> bool:
        """从HuggingFace下载模型"""
        try:
            from huggingface_hub import snapshot_download
            
            logger.info(f"从HuggingFace下载: {model_id}")
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_path),
                local_dir_use_symlinks=False
            )
            return True
            
        except ImportError:
            logger.error("HuggingFace Hub未安装，请先安装: pip install huggingface_hub")
            return False
        except Exception as e:
            logger.error(f"HuggingFace下载失败: {e}")
            return False
    
    def download_ocean_instruct_datasets(self, source="modelscope"):
        """下载OceanInstruct指令数据集"""
        logger.info("下载OceanInstruct指令数据集...")
        
        datasets_info = self.config.get("ocean_instruct_datasets", {})
        
        for dataset_key, dataset_info in datasets_info.items():
            dataset_path = self.datasets_dir / "ocean_instruct" / dataset_info["name"]
            
            if dataset_path.exists():
                logger.info(f"数据集已存在: {dataset_info['name']}")
                continue
            
            logger.info(f"下载数据集: {dataset_info['name']} ({dataset_info['description']})")
            success = False
            
            # 尝试从指定源下载
            if source == "modelscope":
                success = self._download_from_modelscope(dataset_info["modelscope_id"], dataset_path)
            else:
                success = self._download_from_huggingface(dataset_info["huggingface_id"], dataset_path)
            
            # 如果失败，尝试备用源
            if not success:
                logger.warning(f"从{source}下载失败，尝试备用源...")
                if source == "modelscope":
                    success = self._download_from_huggingface(dataset_info["huggingface_id"], dataset_path)
                else:
                    success = self._download_from_modelscope(dataset_info["modelscope_id"], dataset_path)
            
            if success:
                logger.info(f"数据集下载完成: {dataset_info['name']}")
            else:
                logger.error(f"数据集下载失败: {dataset_info['name']}")
    
    def setup_database(self):
        """设置数据库"""
        logger.info("设置数据库...")
        
        # 创建Neo4j数据文件夹
        neo4j_data = self.data_dir / "neo4j"
        neo4j_data.mkdir(exist_ok=True)
        
        # 创建FAISS索引文件夹
        faiss_data = self.data_dir / "vector_store"
        faiss_data.mkdir(exist_ok=True)
        
        logger.info("数据库目录创建完成")
    
    def load_sample_data(self):
        """加载示例数据"""
        logger.info("加载示例数据...")
        
        sample_diseases_file = self.datasets_dir / "marine_diseases" / "sample_diseases.json"
        
        if sample_diseases_file.exists():
            logger.info("示例疾病数据已存在")
        else:
            logger.warning("示例疾病数据不存在，请检查数据文件")
    
    def create_environment_file(self):
        """创建环境配置文件"""
        logger.info("创建环境配置文件...")
        
        # 获取推荐的OceanGPT模型路径
        recommended_model = None
        oceangpt_models = self.config.get("oceangpt_models", {})
        for model_key, model_info in oceangpt_models.items():
            if model_info.get("recommended", False):
                recommended_model = model_info["name"]
                break
        
        env_content = f"""# 蓝海智询环境配置文件
# 数据库配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# OceanGPT模型配置
MODEL_PATH=./models
OCEANGPT_MODEL_NAME={recommended_model or "OceanGPT-o-7B-v0.1"}
OCEANGPT_MODEL_PATH=./models/{recommended_model or "OceanGPT-o-7B-v0.1"}
BGE_MODEL_PATH=./models/bge-large-zh-v1.5
RERANKER_MODEL_PATH=./models/bge-reranker-large

# 模型下载配置
MODEL_SOURCE=modelscope
ENABLE_AUTO_DOWNLOAD=true
SUPPORTS_MULTIMODAL=true

# 向量数据库配置
FAISS_INDEX_PATH=./data/vector_store/faiss_index
FAISS_METADATA_PATH=./data/vector_store/metadata

# OceanInstruct数据集配置
OCEAN_INSTRUCT_PATH=./datasets/ocean_instruct
USE_OCEAN_INSTRUCT=true

# 应用配置
LOG_LEVEL=INFO
MAX_CONCURRENT_REQUESTS=10
CACHE_TTL=3600
GPU_MEMORY_FRACTION=0.8

# API配置
API_HOST=0.0.0.0
API_PORT=8000

# 前端配置
FRONTEND_URL=http://localhost:5173

# 多模态配置
ENABLE_VISION=true
MAX_IMAGE_SIZE=10MB
SUPPORTED_IMAGE_FORMATS=jpg,jpeg,png,bmp,gif
"""
        
        env_file = self.project_root / ".env"
        if not env_file.exists():
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(env_content)
            logger.info("环境配置文件创建完成: .env")
        else:
            logger.info("环境配置文件已存在")
    
    def create_docker_files(self):
        """创建Docker配置文件"""
        logger.info("创建Docker配置文件...")
        
        # Dockerfile
        dockerfile_content = """FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建必要目录
RUN mkdir -p models data datasets logs

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "run.py"]
"""
        
        dockerfile_path = self.project_root / "Dockerfile"
        if not dockerfile_path.exists():
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            logger.info("Dockerfile创建完成")
        
        # docker-compose.yml
        compose_content = """version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./datasets:/app/datasets
      - ./logs:/app/logs
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USERNAME=neo4j
      - NEO4J_PASSWORD=password
      - MODEL_SOURCE=modelscope
      - ENABLE_AUTO_DOWNLOAD=true
    depends_on:
      - neo4j
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  neo4j:
    image: neo4j:5.14
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
    volumes:
      - ./data/neo4j:/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - ./data/redis:/data

  frontend:
    build: ./web
    ports:
      - "5173:5173"
    environment:
      - VITE_API_BASE_URL=http://localhost:8000
    depends_on:
      - app
"""
        
        compose_path = self.project_root / "docker-compose.yml"
        if not compose_path.exists():
            with open(compose_path, 'w') as f:
                f.write(compose_content)
            logger.info("docker-compose.yml创建完成")
    
    def create_model_management_script(self):
        """创建模型管理脚本"""
        logger.info("创建模型管理脚本...")
        
        script_content = '''#!/usr/bin/env python3
"""
OceanGPT模型管理脚本
用于下载、切换和管理不同版本的OceanGPT模型
"""

import argparse
import json
import sys
from pathlib import Path

def list_available_models():
    """列出可用的模型"""
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    models = config.get("oceangpt_models", {})
    print("可用的OceanGPT模型:")
    for key, model_info in models.items():
        status = "推荐" if model_info.get("recommended") else "可选"
        multimodal = "多模态" if model_info.get("supports_multimodal") else "文本"
        print(f"  {model_info['name']} - {model_info['description']} ({multimodal}) [{status}]")

def download_model(model_name, source="modelscope"):
    """下载指定模型"""
    from scripts.setup_project import ProjectSetup
    
    setup = ProjectSetup()
    # 临时修改配置只下载指定模型
    original_config = setup.config.copy()
    
    # 找到指定模型
    target_model = None
    for key, model_info in setup.config.get("oceangpt_models", {}).items():
        if model_info["name"] == model_name:
            target_model = model_info
            break
    
    if not target_model:
        print(f"未找到模型: {model_name}")
        return False
    
    # 下载模型
    print(f"下载模型: {model_name}")
    model_path = setup.models_dir / model_name
    
    if source == "modelscope":
        success = setup._download_from_modelscope(target_model["modelscope_id"], model_path)
    else:
        success = setup._download_from_huggingface(target_model["huggingface_id"], model_path)
    
    return success

def switch_model(model_name):
    """切换默认模型"""
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 检查模型是否存在
    models = config.get("oceangpt_models", {})
    target_model = None
    for key, model_info in models.items():
        if model_info["name"] == model_name:
            target_model = model_info
            break
    
    if not target_model:
        print(f"未找到模型: {model_name}")
        return False
    
    # 更新配置
    config["model_name"] = model_name
    
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"已切换到模型: {model_name}")
    return True

def main():
    parser = argparse.ArgumentParser(description="OceanGPT模型管理工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 列出模型
    subparsers.add_parser("list", help="列出可用模型")
    
    # 下载模型
    download_parser = subparsers.add_parser("download", help="下载模型")
    download_parser.add_argument("model_name", help="模型名称")
    download_parser.add_argument("--source", choices=["modelscope", "huggingface"], 
                               default="modelscope", help="下载源")
    
    # 切换模型
    switch_parser = subparsers.add_parser("switch", help="切换默认模型")
    switch_parser.add_argument("model_name", help="模型名称")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_available_models()
    elif args.command == "download":
        success = download_model(args.model_name, args.source)
        sys.exit(0 if success else 1)
    elif args.command == "switch":
        success = switch_model(args.model_name)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
'''
        
        script_path = self.project_root / "manage_models.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        script_path.chmod(0o755)
        logger.info("模型管理脚本创建完成: manage_models.py")
    
    def run_setup(self, skip_models=False, skip_frontend=False, skip_datasets=False, model_source="modelscope"):
        """运行完整的项目设置"""
        logger.info("开始项目初始化...")
        
        try:
            self.create_directories()
            self.create_environment_file()
            self.install_dependencies()
            
            if not skip_frontend:
                self.install_frontend_dependencies()
            
            if not skip_models:
                self.download_models(source=model_source)
            
            if not skip_datasets:
                self.download_ocean_instruct_datasets(source=model_source)
            
            self.setup_database()
            self.load_sample_data()
            self.create_docker_files()
            self.create_model_management_script()
            
            logger.info("项目初始化完成！")
            logger.info("下一步:")
            logger.info("1. 配置 .env 文件中的数据库连接")
            logger.info("2. 启动Neo4j数据库")
            logger.info("3. 运行 python run.py 启动应用")
            logger.info("4. 使用 python manage_models.py list 查看可用模型")
            logger.info("5. 使用 python manage_models.py download <模型名> 下载其他模型")
            
        except Exception as e:
            logger.error(f"项目初始化失败: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="蓝海智询项目初始化脚本")
    parser.add_argument("--skip-models", action="store_true", help="跳过模型下载")
    parser.add_argument("--skip-frontend", action="store_true", help="跳过前端依赖安装")
    parser.add_argument("--skip-datasets", action="store_true", help="跳过数据集下载")
    parser.add_argument("--model-source", choices=["modelscope", "huggingface"], 
                       default="modelscope", help="模型下载源")
    parser.add_argument("--project-root", help="项目根目录路径")
    
    args = parser.parse_args()
    
    setup = ProjectSetup(args.project_root)
    setup.run_setup(
        skip_models=args.skip_models,
        skip_frontend=args.skip_frontend,
        skip_datasets=args.skip_datasets,
        model_source=args.model_source
    )

if __name__ == "__main__":
    main() 