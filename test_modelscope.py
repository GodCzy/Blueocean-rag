#!/usr/bin/env python3
"""
测试ModelScope连接和模型可用性
"""

import sys
import os

def test_modelscope():
    try:
        import modelscope
        print(f"✅ ModelScope版本: {modelscope.__version__}")
        
        # 测试简单的模型查询
        from modelscope.hub.api import HubApi
        api = HubApi()
        
        # 尝试搜索zjunlp的模型
        print("\n🔍 搜索zjunlp的OceanGPT模型...")
        try:
            models = api.list_models(namespace="zjunlp")
            ocean_models = [m for m in models if 'ocean' in m.get('model_name', '').lower()]
            print(f"找到 {len(ocean_models)} 个相关模型:")
            for model in ocean_models[:5]:  # 只显示前5个
                print(f"  - {model.get('model_name', 'N/A')}")
        except Exception as e:
            print(f"❌ 搜索模型失败: {e}")
        
        return True
    except ImportError:
        print("❌ ModelScope未安装")
        return False
    except Exception as e:
        print(f"❌ ModelScope错误: {e}")
        return False

def test_huggingface():
    try:
        import huggingface_hub
        print(f"✅ HuggingFace Hub版本: {huggingface_hub.__version__}")
        
        # 测试连接
        from huggingface_hub import HfApi
        api = HfApi()
        
        print("\n🔍 测试HuggingFace连接...")
        try:
            # 测试一个公开的模型
            info = api.model_info("microsoft/DialoGPT-medium")
            print(f"✅ HuggingFace连接正常")
        except Exception as e:
            print(f"❌ HuggingFace连接失败: {e}")
        
        return True
    except ImportError:
        print("❌ HuggingFace Hub未安装")
        return False
    except Exception as e:
        print(f"❌ HuggingFace错误: {e}")
        return False

def test_direct_download():
    """测试直接下载一个小模型"""
    print("\n🚀 测试直接下载...")
    
    try:
        from modelscope import snapshot_download
        
        # 使用一个确实存在的小模型进行测试
        test_model = "damo/nlp_structbert_backbone_base_std"
        test_dir = "./test_download"
        
        print(f"📂 测试下载到: {test_dir}")
        
        # 只下载配置文件，不下载大文件
        snapshot_download(
            model_id=test_model,
            cache_dir=test_dir,
            allow_patterns=["*.json"]
        )
        
        print("✅ 测试下载成功")
        
        # 清理测试文件
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print("🧹 清理测试文件完成")
        
        return True
    except Exception as e:
        print(f"❌ 测试下载失败: {e}")
        return False

if __name__ == "__main__":
    print("🔬 开始环境测试...")
    print("=" * 50)
    
    ms_ok = test_modelscope()
    print("\n" + "=" * 50)
    
    hf_ok = test_huggingface()
    print("\n" + "=" * 50)
    
    if ms_ok:
        test_direct_download()
    
    print("\n" + "=" * 50)
    print("📊 测试结果:")
    print(f"  ModelScope: {'✅' if ms_ok else '❌'}")
    print(f"  HuggingFace: {'✅' if hf_ok else '❌'}") 