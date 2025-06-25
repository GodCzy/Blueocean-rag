#!/usr/bin/env python3
"""
OceanGPT模型管理脚本
用于下载、切换和管理不同版本的OceanGPT模型

 更多使用说明见 DEPLOYMENT_GUIDE.md。
"""

import argparse
import json
import sys
import os
from pathlib import Path
import subprocess

def load_config():
    """加载配置文件"""
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(config):
    """保存配置文件"""
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def list_available_models():
    """列出可用的模型"""
    config = load_config()
    models = config.get("oceangpt_models", {})
    
    print("🌊 可用的OceanGPT模型:")
    print("-" * 80)
    
    for key, model_info in models.items():
        status = "✅ 推荐" if model_info.get("recommended") else "📋 可选"
        features = []
        
        if model_info.get("supports_multimodal"):
            features.append("🖼️ 多模态")
        if model_info.get("supports_code"):
            features.append("💻 代码生成")
        if not features:
            features.append("📝 文本")
        
        model_path = Path("models") / model_info["name"]
        installed = "✅ 已安装" if model_path.exists() else "❌ 未安装"
        
        print(f"📦 {model_info['name']}")
        print(f"   📋 描述: {model_info['description']}")
        print(f"   🏗️  基础模型: {model_info['base_model']}")
        print(f"   🔧 功能: {' | '.join(features)}")
        print(f"   📍 状态: {status} | {installed}")
        print(f"   🔗 ModelScope: {model_info['modelscope_id']}")
        print(f"   🔗 HuggingFace: {model_info['huggingface_id']}")
        print()

def list_datasets():
    """列出可用的数据集"""
    config = load_config()
    datasets = config.get("ocean_instruct_datasets", {})
    
    print("📚 可用的OceanInstruct数据集:")
    print("-" * 60)
    
    for key, dataset_info in datasets.items():
        dataset_path = Path("datasets/ocean_instruct") / dataset_info["name"]
        installed = "✅ 已下载" if dataset_path.exists() else "❌ 未下载"
        
        print(f"📊 {dataset_info['name']}")
        print(f"   📋 描述: {dataset_info['description']}")
        print(f"   📏 大小: {dataset_info['size']}")
        print(f"   🏷️  类型: {dataset_info['type']}")
        print(f"   📍 状态: {installed}")
        print(f"   🔗 ModelScope: {dataset_info['modelscope_id']}")
        print(f"   🔗 HuggingFace: {dataset_info['huggingface_id']}")
        print()

def download_from_modelscope(model_id: str, local_path: Path) -> bool:
    """从ModelScope下载模型"""
    try:
        print(f"📥 从ModelScope下载: {model_id}")
        
        # 检查是否安装了modelscope
        try:
            import modelscope
        except ImportError:
            print("❌ ModelScope未安装，正在安装...")
            subprocess.run([sys.executable, "-m", "pip", "install", "modelscope"], check=True)
            import modelscope
        
        from modelscope import snapshot_download
        
        snapshot_download(
            model_id=model_id,
            cache_dir=str(local_path.parent),
            local_dir=str(local_path)
        )
        
        print(f"✅ 下载完成: {local_path}")
        return True
        
    except Exception as e:
        print(f"❌ ModelScope下载失败: {e}")
        return False

def download_from_huggingface(model_id: str, local_path: Path) -> bool:
    """从HuggingFace下载模型"""
    try:
        print(f"📥 从HuggingFace下载: {model_id}")
        
        # 检查是否安装了huggingface_hub
        try:
            import huggingface_hub
        except ImportError:
            print("❌ HuggingFace Hub未安装，正在安装...")
            subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
            import huggingface_hub
        
        from huggingface_hub import snapshot_download
        
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_path),
            local_dir_use_symlinks=False
        )
        
        print(f"✅ 下载完成: {local_path}")
        return True
        
    except Exception as e:
        print(f"❌ HuggingFace下载失败: {e}")
        return False

def download_model(model_name: str, source: str = "modelscope") -> bool:
    """下载指定模型"""
    config = load_config()
    models = config.get("oceangpt_models", {})
    
    # 查找指定模型
    target_model = None
    for key, model_info in models.items():
        if model_info["name"] == model_name or key == model_name:
            target_model = model_info
            break
    
    if not target_model:
        print(f"❌ 未找到模型: {model_name}")
        print("💡 使用 'python manage_oceangpt.py list' 查看可用模型")
        return False
    
    model_path = Path("models") / target_model["name"]
    
    if model_path.exists():
        print(f"✅ 模型已存在: {target_model['name']}")
        return True
    
    print(f"🚀 开始下载模型: {target_model['name']}")
    print(f"📋 描述: {target_model['description']}")
    print(f"📍 下载源: {source}")
    
    # 创建模型目录
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 尝试下载
    success = False
    if source == "modelscope":
        success = download_from_modelscope(target_model["modelscope_id"], model_path)
    else:
        success = download_from_huggingface(target_model["huggingface_id"], model_path)
    
    # 如果失败，尝试备用源
    if not success:
        fallback_source = "huggingface" if source == "modelscope" else "modelscope"
        print(f"⚠️  主要源下载失败，尝试备用源: {fallback_source}")
        
        if fallback_source == "modelscope":
            success = download_from_modelscope(target_model["modelscope_id"], model_path)
        else:
            success = download_from_huggingface(target_model["huggingface_id"], model_path)
    
    return success

def download_dataset(dataset_name: str, source: str = "modelscope") -> bool:
    """下载指定数据集"""
    config = load_config()
    datasets = config.get("ocean_instruct_datasets", {})
    
    # 查找指定数据集
    target_dataset = None
    for key, dataset_info in datasets.items():
        if dataset_info["name"] == dataset_name or key == dataset_name:
            target_dataset = dataset_info
            break
    
    if not target_dataset:
        print(f"❌ 未找到数据集: {dataset_name}")
        print("💡 使用 'python manage_oceangpt.py list-datasets' 查看可用数据集")
        return False
    
    dataset_path = Path("datasets/ocean_instruct") / target_dataset["name"]
    
    if dataset_path.exists():
        print(f"✅ 数据集已存在: {target_dataset['name']}")
        return True
    
    print(f"🚀 开始下载数据集: {target_dataset['name']}")
    print(f"📋 描述: {target_dataset['description']}")
    print(f"📍 下载源: {source}")
    
    # 创建数据集目录
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 尝试下载
    success = False
    if source == "modelscope":
        success = download_from_modelscope(target_dataset["modelscope_id"], dataset_path)
    else:
        success = download_from_huggingface(target_dataset["huggingface_id"], dataset_path)
    
    # 如果失败，尝试备用源
    if not success:
        fallback_source = "huggingface" if source == "modelscope" else "modelscope"
        print(f"⚠️  主要源下载失败，尝试备用源: {fallback_source}")
        
        if fallback_source == "modelscope":
            success = download_from_modelscope(target_dataset["modelscope_id"], dataset_path)
        else:
            success = download_from_huggingface(target_dataset["huggingface_id"], dataset_path)
    
    return success

def switch_model(model_name: str) -> bool:
    """切换默认模型"""
    config = load_config()
    models = config.get("oceangpt_models", {})
    
    # 检查模型是否存在
    target_model = None
    for key, model_info in models.items():
        if model_info["name"] == model_name or key == model_name:
            target_model = model_info
            break
    
    if not target_model:
        print(f"❌ 未找到模型: {model_name}")
        return False
    
    # 检查模型是否已下载
    model_path = Path("models") / target_model["name"]
    if not model_path.exists():
        print(f"❌ 模型未下载: {target_model['name']}")
        print(f"💡 使用 'python manage_oceangpt.py download {model_name}' 下载模型")
        return False
    
    # 更新配置
    config["model_name"] = target_model["name"]
    
    # 更新模型路径
    if "model_local_paths" in config:
        config["model_local_paths"][target_model["name"]] = f"./models/{target_model['name']}"
    
    save_config(config)
    
    print(f"✅ 已切换到模型: {target_model['name']}")
    print(f"📋 描述: {target_model['description']}")
    
    features = []
    if target_model.get("supports_multimodal"):
        features.append("🖼️ 多模态支持")
    if target_model.get("supports_code"):
        features.append("💻 代码生成")
    
    if features:
        print(f"🔧 功能: {' | '.join(features)}")
    
    return True

def status():
    """显示当前状态"""
    config = load_config()
    current_model = config.get("model_name", "未设置")
    
    print("🌊 蓝海智询 - OceanGPT状态")
    print("=" * 50)
    print(f"📦 当前模型: {current_model}")
    
    if current_model != "未设置":
        models = config.get("oceangpt_models", {})
        current_model_info = None
        
        for key, model_info in models.items():
            if model_info["name"] == current_model:
                current_model_info = model_info
                break
        
        if current_model_info:
            print(f"📋 描述: {current_model_info['description']}")
            print(f"🏗️  基础模型: {current_model_info['base_model']}")
            
            features = []
            if current_model_info.get("supports_multimodal"):
                features.append("🖼️ 多模态")
            if current_model_info.get("supports_code"):
                features.append("💻 代码生成")
            if features:
                print(f"🔧 功能: {' | '.join(features)}")
            
            model_path = Path("models") / current_model_info["name"]
            status = "✅ 已安装" if model_path.exists() else "❌ 未安装"
            print(f"📍 状态: {status}")
    
    print()
    print("📊 已安装的模型:")
    models_dir = Path("models")
    if models_dir.exists():
        installed_models = [d.name for d in models_dir.iterdir() if d.is_dir()]
        if installed_models:
            for model in installed_models:
                print(f"  ✅ {model}")
        else:
            print("  ❌ 暂无已安装的模型")
    else:
        print("  ❌ 模型目录不存在")

def main():
    parser = argparse.ArgumentParser(
        description="🌊 蓝海智询 - OceanGPT模型管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python manage_oceangpt.py list                          # 列出可用模型
  python manage_oceangpt.py download OceanGPT-o-7B-v0.1  # 下载模型
  python manage_oceangpt.py switch OceanGPT-o-7B-v0.1    # 切换模型
  python manage_oceangpt.py status                        # 查看状态
  python manage_oceangpt.py list-datasets                 # 列出数据集
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 列出模型
    subparsers.add_parser("list", help="列出可用模型")
    
    # 列出数据集
    subparsers.add_parser("list-datasets", help="列出可用数据集")
    
    # 下载模型
    download_parser = subparsers.add_parser("download", help="下载模型")
    download_parser.add_argument("model_name", help="模型名称")
    download_parser.add_argument("--source", choices=["modelscope", "huggingface"], 
                               default="modelscope", help="下载源 (默认: modelscope)")
    
    # 下载数据集
    dataset_parser = subparsers.add_parser("download-dataset", help="下载数据集")
    dataset_parser.add_argument("dataset_name", help="数据集名称")
    dataset_parser.add_argument("--source", choices=["modelscope", "huggingface"], 
                               default="modelscope", help="下载源 (默认: modelscope)")
    
    # 切换模型
    switch_parser = subparsers.add_parser("switch", help="切换默认模型")
    switch_parser.add_argument("model_name", help="模型名称")
    
    # 状态
    subparsers.add_parser("status", help="显示当前状态")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_available_models()
    elif args.command == "list-datasets":
        list_datasets()
    elif args.command == "download":
        success = download_model(args.model_name, args.source)
        sys.exit(0 if success else 1)
    elif args.command == "download-dataset":
        success = download_dataset(args.dataset_name, args.source)
        sys.exit(0 if success else 1)
    elif args.command == "switch":
        success = switch_model(args.model_name)
        sys.exit(0 if success else 1)
    elif args.command == "status":
        status()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 