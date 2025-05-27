#!/usr/bin/env python3
"""
OceanGPT模型下载脚本
支持ModelScope和HuggingFace，具有断点续传功能
"""

import os
import json
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/download.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """加载配置文件"""
    with open('config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def download_from_modelscope(model_id: str, local_dir: str) -> bool:
    """从ModelScope下载模型"""
    try:
        from modelscope import snapshot_download
        logger.info(f"开始从ModelScope下载模型: {model_id}")
        logger.info(f"下载到目录: {local_dir}")
        
        # 创建目录
        os.makedirs(local_dir, exist_ok=True)
        
        # 下载模型
        snapshot_download(
            model_id=model_id,
            cache_dir=local_dir,
            revision='master'
        )
        
        logger.info(f"✅ ModelScope下载完成: {model_id}")
        return True
        
    except ImportError:
        logger.error("❌ ModelScope SDK未安装，请运行: pip install modelscope")
        return False
    except Exception as e:
        logger.error(f"❌ ModelScope下载失败: {e}")
        return False

def download_from_huggingface(model_id: str, local_dir: str) -> bool:
    """从HuggingFace下载模型"""
    try:
        from huggingface_hub import snapshot_download
        logger.info(f"开始从HuggingFace下载模型: {model_id}")
        logger.info(f"下载到目录: {local_dir}")
        
        # 创建目录
        os.makedirs(local_dir, exist_ok=True)
        
        # 下载模型
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin", "*.model", "*.jpg", "*.png"]
        )
        
        logger.info(f"✅ HuggingFace下载完成: {model_id}")
        return True
        
    except ImportError:
        logger.error("❌ HuggingFace Hub未安装，请运行: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"❌ HuggingFace下载失败: {e}")
        return False

def check_model_integrity(model_path: str) -> bool:
    """检查模型文件完整性"""
    model_path = Path(model_path)
    
    # 检查必要文件
    required_files = [
        'config.json',
        'tokenizer.json',
        'tokenizer_config.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    # 检查模型权重文件
    safetensors_files = list(model_path.glob('*.safetensors'))
    if not safetensors_files:
        missing_files.append('*.safetensors (模型权重文件)')
    
    if missing_files:
        logger.warning(f"⚠️ 缺少文件: {', '.join(missing_files)}")
        return False
    
    logger.info("✅ 模型文件完整性检查通过")
    return True

def download_oceangpt_model(model_name: str = None) -> bool:
    """下载OceanGPT模型"""
    
    # 加载配置
    config = load_config()
    
    # 确定要下载的模型
    if model_name is None:
        model_name = config.get('model_name', 'OceanGPT-o-7B-v0.1')
    
    if model_name not in config['oceangpt_models']:
        logger.error(f"❌ 未找到模型配置: {model_name}")
        return False
    
    model_config = config['oceangpt_models'][model_name]
    local_path = config['model_local_paths'][model_name]
    
    logger.info(f"🚀 开始下载OceanGPT模型: {model_name}")
    logger.info(f"📝 模型描述: {model_config['description']}")
    logger.info(f"📂 本地路径: {local_path}")
    
    # 检查现有文件
    if os.path.exists(local_path):
        logger.info("📁 发现现有模型目录，检查完整性...")
        if check_model_integrity(local_path):
            logger.info("✅ 模型已完整下载")
            return True
        else:
            logger.info("🔄 模型不完整，继续下载...")
    
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    
    # 尝试下载
    preferred_source = config['model_download']['preferred_source']
    fallback_source = config['model_download']['fallback_source']
    
    success = False
    
    if preferred_source == 'modelscope':
        logger.info("🌐 使用ModelScope作为首选下载源")
        success = download_from_modelscope(
            model_config['modelscope_id'], 
            local_path
        )
        
        if not success and fallback_source == 'huggingface':
            logger.info("🔄 ModelScope失败，尝试HuggingFace...")
            success = download_from_huggingface(
                model_config['huggingface_id'], 
                local_path
            )
    
    elif preferred_source == 'huggingface':
        logger.info("🌐 使用HuggingFace作为首选下载源")
        success = download_from_huggingface(
            model_config['huggingface_id'], 
            local_path
        )
        
        if not success and fallback_source == 'modelscope':
            logger.info("🔄 HuggingFace失败，尝试ModelScope...")
            success = download_from_modelscope(
                model_config['modelscope_id'], 
                local_path
            )
    
    if success:
        # 验证下载完整性
        if check_model_integrity(local_path):
            logger.info(f"🎉 {model_name} 下载完成且验证通过!")
            return True
        else:
            logger.error(f"❌ {model_name} 下载完成但验证失败")
            return False
    else:
        logger.error(f"❌ {model_name} 下载失败")
        return False

def download_all_models():
    """下载所有推荐的OceanGPT模型"""
    config = load_config()
    
    for model_name, model_config in config['oceangpt_models'].items():
        if model_config.get('recommended', False):
            logger.info(f"\n{'='*50}")
            logger.info(f"下载推荐模型: {model_name}")
            logger.info(f"{'='*50}")
            
            success = download_oceangpt_model(model_name)
            if not success:
                logger.error(f"模型 {model_name} 下载失败")
            
            time.sleep(2)  # 短暂休息

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='下载OceanGPT模型')
    parser.add_argument('--model', '-m', type=str, help='指定要下载的模型名称')
    parser.add_argument('--all', '-a', action='store_true', help='下载所有推荐模型')
    
    args = parser.parse_args()
    
    if args.all:
        download_all_models()
    elif args.model:
        download_oceangpt_model(args.model)
    else:
        # 默认下载配置中指定的当前模型
        download_oceangpt_model() 