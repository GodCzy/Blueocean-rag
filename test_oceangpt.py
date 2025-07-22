#!/usr/bin/env python3
"""
OceanGPT 集成测试脚本
测试 OceanGPT 模型在蓝海智询系统中的使用
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.oceangpt_manager import OceanGPTManager, ModelConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def test_oceangpt():
    """测试 OceanGPT 模型"""
    print("🌊 OceanGPT 集成测试")
    print("=" * 50)
    
    # 配置 OceanGPT 模型 - 使用正确的模型名称
    print("⚙️ 配置 OceanGPT 模型...")
    model_config = ModelConfig(
        model_name="OceanGPT-o-7B",  # 简化的模型名称，会自动匹配
        model_path="./models/OceanGPT-o-7B",
        load_in_4bit=True,
        device="auto",
        supports_multimodal=True,
        model_source="huggingface"  # 或 "modelscope"
    )
    
    # 创建 OceanGPT 管理器
    print("🤖 初始化 OceanGPT 管理器...")
    oceangpt_manager = OceanGPTManager(model_config)
    
    # 检查模型文件是否存在
    model_path = Path(model_config.model_path)
    if not model_path.exists():
        print(f"📥 模型文件不存在: {model_path}")
        print("🔄 正在尝试下载模型...")
        
        try:
            download_success = await oceangpt_manager.download_model_from_official()
            if not download_success:
                print("❌ 模型下载失败")
                print("💡 请手动下载模型:")
                print("   HuggingFace:")
                print(f"   git clone https://huggingface.co/zjunlp/OceanGPT-o-7B {model_path}")
                print("   或者")
                print(f"   huggingface-cli download --resume-download zjunlp/OceanGPT-o-7B --local-dir {model_path}")
                print()
                print("   ModelScope:")
                print(f"   git clone https://www.modelscope.cn/zjunlp/OceanGPT-o-7B.git {model_path}")
                return False
        except Exception as e:
            print(f"❌ 下载过程中出错: {e}")
            return False
    
    # 加载模型
    print("📦 正在加载 OceanGPT 模型...")
    try:
        model_loaded = await oceangpt_manager.load_model()
        if not model_loaded:
            print("❌ 模型加载失败")
            return False
        
        print("✅ OceanGPT 模型加载成功！")
        
    except Exception as e:
        print(f"❌ 模型加载出错: {e}")
        print("💡 可能的解决方案:")
        print("   1. 检查模型文件完整性")
        print("   2. 确保有足够的内存 (建议 8GB+)")
        print("   3. 如果使用 CPU，请确保有足够的内存")
        print("   4. 检查是否已安装所需依赖: qwen-vl-utils")
        return False
    
    # 测试文本生成
    print("\n🧪 测试文本生成功能...")
    test_prompts = [
        "海洋中鱼类白点病的症状和治疗方法是什么？",
        "海水鱼养殖中如何预防细菌性疾病？",
        "什么是海洋鱼类烂鳃病？如何诊断和治疗？"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🔍 测试 {i}: {prompt}")
        try:
            response = await oceangpt_manager.generate_response(prompt)
            print(f"🤖 OceanGPT回答: {response[:200]}..." if len(response) > 200 else f"🤖 OceanGPT回答: {response}")
        except Exception as e:
            print(f"❌ 生成回答失败: {e}")
    
    # 获取模型信息
    model_info = oceangpt_manager.get_model_info()
    print(f"\n📊 模型信息:")
    print(f"   模型名称: {model_info.get('model_name', 'Unknown')}")
    print(f"   设备: {model_info.get('device', 'Unknown')}")
    print(f"   支持多模态: {model_info.get('supports_multimodal', False)}")
    
    print("\n✅ OceanGPT 集成测试完成！")
    return True

async def test_rag_with_oceangpt():
    """测试 RAG 与 OceanGPT 的集成"""
    print("\n🔗 测试 RAG + OceanGPT 集成")
    print("=" * 50)
    
    from src.api.rag_api import RAGService
    
    # 创建 OceanGPT 配置
    oceangpt_config = ModelConfig(
        model_name="zjunlp/OceanGPT-o-7B",
        model_path="./models/OceanGPT-o-7B",
        load_in_4bit=True,
        device="auto"
    )
    
    # 创建 RAG 服务
    print("🔧 初始化 RAG 服务...")
    rag_service = RAGService(
        data_dir="src/data",
        index_path="./temp_index",
        oceangpt_config=oceangpt_config
    )
    
    # 初始化 OceanGPT
    print("🤖 初始化 OceanGPT...")
    oceangpt_ready = await rag_service.initialize_oceangpt()
    
    if oceangpt_ready:
        print("✅ OceanGPT 初始化成功")
        
        # 测试问答
        test_query = "海洋鱼类烂鳃病的症状有哪些？"
        print(f"\n🧪 测试查询: {test_query}")
        
        result = await rag_service.ask(test_query)
        print(f"🤖 回答: {result['answer']}")
        print(f"📚 使用了 {len(result['source_documents'])} 个文档")
        print(f"⏱️ 耗时: {result['elapsed_time']:.2f}秒")
        print(f"🔗 OceanGPT状态: {'可用' if result['has_oceangpt'] else '不可用'}")
    else:
        print("⚠️ OceanGPT 初始化失败，但 RAG 服务仍可使用检索功能")

def main():
    """主函数"""
    print("🚀 开始 OceanGPT 集成测试")
    print("\n💡 提示：本测试将使用正确的 OceanGPT 模型")
    print("   - 模型: zjunlp/OceanGPT-o-7B")
    print("   - 基于: Qwen2.5-VL-7B-Instruct") 
    print("   - 专业: 海洋科学和水生动物疾病诊断")
    print("   - 多模态: 支持图像和文本输入")
    
    try:
        # 测试基本的 OceanGPT 功能
        success = asyncio.run(test_oceangpt())
        
        if success:
            # 测试 RAG 集成
            asyncio.run(test_rag_with_oceangpt())
        else:
            print("⚠️ 基础模型测试失败，跳过 RAG 集成测试")
            
    except KeyboardInterrupt:
        print("\n🛑 用户中断测试")
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 