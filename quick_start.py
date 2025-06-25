#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速启动脚本 - 蓝海智询系统

自动化环境检查、配置和启动流程

 使用前请确保已阅读 DEPLOYMENT_GUIDE.md。

作者: 蓝海智询团队
"""

import os
import sys
import json
import subprocess
import platform
import time
from pathlib import Path
from typing import Dict, List, Tuple

class QuickStartManager:
    """快速启动管理器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.errors = []
        self.warnings = []
        
    def print_banner(self):
        """显示启动横幅"""
        banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                        🌊 蓝海智询                          ║
    ║            基于RAG知识库与知识图谱的水生动物疾病问答平台      ║
    ║                                                            ║
    ║                     快速启动向导                            ║
    ╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def check_python_version(self) -> bool:
        """检查Python版本"""
        print("🐍 检查Python版本...")
        
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            self.errors.append(f"Python版本不支持: {version.major}.{version.minor}.{version.micro}，需要Python 3.8+")
            return False
    
    def check_dependencies(self) -> bool:
        """检查依赖包"""
        print("📦 检查依赖包...")
        
        required_packages = [
            'fastapi', 'uvicorn', 'torch', 'transformers', 
            'faiss', 'neo4j', 'pandas', 'numpy'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                if package == 'faiss':
                    # faiss-cpu包的模块名是faiss
                    __import__('faiss')
                else:
                    __import__(package.replace('-', '_'))
                print(f"   ✅ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"   ❌ {package}")
        
        if missing_packages:
            self.errors.append(f"缺少依赖包: {', '.join(missing_packages)}")
            return False
        
        return True
    
    def check_gpu(self) -> bool:
        """检查GPU环境"""
        print("🎮 检查GPU环境...")
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    print(f"   ✅ GPU {i}: {gpu_name} ({memory:.1f}GB)")
                return True
            else:
                self.warnings.append("未检测到CUDA GPU，将使用CPU模式")
                print("   ⚠️ 未检测到GPU，将使用CPU模式")
                return False
        except ImportError:
            self.warnings.append("PyTorch未安装，无法检查GPU")
            return False
    
    def create_directories(self):
        """创建必要目录"""
        print("📁 创建项目目录...")
        
        directories = [
            'data/vector_store',
            'data/knowledge_graph',
            'logs',
            'saves/fine_tuned_models',
            'saves/checkpoints',
            'uploads',
            'static'
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {directory}")
    
    def create_default_config(self):
        """创建默认配置文件"""
        print("⚙️ 创建配置文件...")
        
        config_file = self.project_root / "config.json"
        
        if config_file.exists():
            print("   ✅ 配置文件已存在")
            return
        
        default_config = {
            "app": {
                "name": "蓝海智询",
                "version": "1.0.0",
                "debug": True,
                "host": "0.0.0.0",
                "port": 8000
            },
            "models": {
                "oceangpt": {
                    "model_name": "OceanGPT-o-7B-v0.1",
                    "model_path": "./models/OceanGPT-o-7B-v0.1",
                    "device": "auto",
                    "load_in_4bit": True,
                    "max_length": 2048
                }
            },
            "rag": {
                "vector_store_path": "./data/vector_store",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "top_k": 5
            },
            "knowledge_graph": {
                "neo4j_uri": "bolt://localhost:7687",
                "neo4j_user": "neo4j",
                "neo4j_password": "password"
            },
            "training": {
                "output_dir": "./saves/fine_tuned_models",
                "batch_size": 1,
                "learning_rate": 1e-4,
                "num_epochs": 3
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        
        print("   ✅ 已创建默认配置文件")
    
    def check_model_files(self) -> bool:
        """检查模型文件"""
        print("🤖 检查模型文件...")
        
        model_path = self.project_root / "models" / "OceanGPT-o-7B-v0.1"
        
        if not model_path.exists():
            self.warnings.append("OceanGPT模型文件不存在，需要手动下载")
            print("   ⚠️ 模型文件不存在")
            print("   💡 请运行: git clone https://huggingface.co/zjunlp/OceanGPT-o-7B-v0.1 models/OceanGPT-o-7B-v0.1")
            return False
        
        # 检查关键文件
        required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
        missing_files = []
        
        for file_name in required_files:
            if not (model_path / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            self.warnings.append(f"模型文件不完整，缺少: {', '.join(missing_files)}")
            print(f"   ⚠️ 模型文件不完整")
            return False
        
        print("   ✅ 模型文件完整")
        return True
    
    def check_databases(self) -> bool:
        """检查数据库连接"""
        print("🗄️ 检查数据库连接...")
        
        # 检查Neo4j
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
            with driver.session() as session:
                session.run("RETURN 1")
            driver.close()
            print("   ✅ Neo4j连接正常")
            neo4j_ok = True
        except Exception as e:
            self.warnings.append(f"Neo4j连接失败: {e}")
            print("   ⚠️ Neo4j连接失败")
            neo4j_ok = False
        
        # 检查Redis（可选）
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            print("   ✅ Redis连接正常")
            redis_ok = True
        except Exception as e:
            self.warnings.append(f"Redis连接失败（可选）: {e}")
            print("   ⚠️ Redis连接失败（可选）")
            redis_ok = False
        
        return neo4j_ok  # Neo4j是必需的
    
    def install_frontend_dependencies(self) -> bool:
        """安装前端依赖"""
        print("🌐 检查前端环境...")
        
        web_dir = self.project_root / "web"
        
        if not web_dir.exists():
            self.warnings.append("前端目录不存在")
            print("   ⚠️ 前端目录不存在")
            return False
        
        # 检查Node.js
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"   ✅ Node.js {result.stdout.strip()}")
            else:
                self.errors.append("Node.js未安装")
                return False
        except FileNotFoundError:
            self.errors.append("Node.js未安装")
            return False
        
        # 检查package.json
        package_json = web_dir / "package.json"
        if not package_json.exists():
            self.warnings.append("前端package.json不存在")
            print("   ⚠️ package.json不存在")
            return False
        
        # 检查node_modules
        node_modules = web_dir / "node_modules"
        if not node_modules.exists():
            print("   🔄 安装前端依赖...")
            try:
                subprocess.run(['npm', 'install'], cwd=web_dir, check=True)
                print("   ✅ 前端依赖安装完成")
                return True
            except subprocess.CalledProcessError:
                self.errors.append("前端依赖安装失败")
                return False
        else:
            print("   ✅ 前端依赖已安装")
            return True
    
    def run_quick_test(self) -> bool:
        """运行快速测试"""
        print("🧪 运行快速测试...")
        
        try:
            # 测试导入核心模块
            sys.path.append(str(self.project_root))
            
            from src.config.settings import get_settings
            settings = get_settings()
            print("   ✅ 配置模块加载成功")
            
            from src.utils.logger import get_logger
            logger = get_logger("test")
            print("   ✅ 日志模块加载成功")
            
            # 测试数据加载
            sample_data_file = self.project_root / "datasets" / "marine_diseases" / "sample_diseases.json"
            if sample_data_file.exists():
                with open(sample_data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"   ✅ 样本数据加载成功 ({len(data)} 条记录)")
            else:
                self.warnings.append("样本数据文件不存在")
            
            return True
            
        except Exception as e:
            self.errors.append(f"模块测试失败: {e}")
            return False
    
    def print_summary(self):
        """打印检查摘要"""
        print("\n" + "="*60)
        print("📋 环境检查摘要")
        print("="*60)
        
        if not self.errors:
            print("🎉 所有必需组件检查通过！")
        else:
            print("❌ 发现以下错误:")
            for error in self.errors:
                print(f"   • {error}")
        
        if self.warnings:
            print("\n⚠️ 警告:")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        print("\n" + "="*60)
    
    def print_next_steps(self):
        """打印后续步骤"""
        print("🚀 后续步骤:")
        print()
        
        if self.errors:
            print("1. 解决上述错误后重新运行此脚本")
            print("2. 参考 DEPLOYMENT_GUIDE.md 获取详细安装说明")
        else:
            print("1. 启动系统:")
            print("   python src/main.py")
            print()
            print("2. 访问系统:")
            print("   • API文档: http://localhost:8000/docs")
            print("   • 健康检查: http://localhost:8000/health")
            print()
            print("3. 测试核心功能:")
            print("   • 疾病诊断: POST /api/v1/diagnosis/diagnose")
            print("   • 知识检索: POST /api/v1/rag/search")
            print()
            print("4. 可选: 启动模型微调训练")
            print("   访问: http://localhost:8000/docs#/训练服务")
    
    def run(self):
        """运行快速启动检查"""
        self.print_banner()
        
        print("🔍 开始环境检查...\n")
        
        # 执行所有检查
        checks = [
            ("Python版本", self.check_python_version),
            ("依赖包", self.check_dependencies),
            ("GPU环境", self.check_gpu),
            ("项目目录", lambda: (self.create_directories(), True)[1]),
            ("配置文件", lambda: (self.create_default_config(), True)[1]),
            ("模型文件", self.check_model_files),
            ("数据库", self.check_databases),
            ("前端环境", self.install_frontend_dependencies),
            ("快速测试", self.run_quick_test)
        ]
        
        for name, check_func in checks:
            try:
                success = check_func()
                time.sleep(0.5)  # 让输出更清晰
            except Exception as e:
                self.errors.append(f"{name}检查失败: {e}")
            print()
        
        self.print_summary()
        self.print_next_steps()

def main():
    """主函数"""
    try:
        manager = QuickStartManager()
        manager.run()
    except KeyboardInterrupt:
        print("\n\n❌ 用户中断")
    except Exception as e:
        print(f"\n\n💥 未预期的错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 