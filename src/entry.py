import asyncio
import os
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.core.oceangpt_manager import OceanGPTManager, ModelConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def main():
    """蓝海智询系统主入口"""
    print("🌊 蓝海智询 - 基于OceanGPT的水生动物疾病问答系统")
    print("="*60)
    
    # 配置本地 embedding 模型
    print("📦 配置Embedding模型...")
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 初始化OceanGPT管理器
    print("🤖 初始化OceanGPT模型...")
    model_config = ModelConfig(
        model_name="OceanGPT-o-7B",
        model_path="./models/OceanGPT-o-7B",
        load_in_4bit=True,
        device="auto",
        supports_multimodal=True
    )
    
    oceangpt_manager = OceanGPTManager(model_config)
    
    # 创建数据目录（如果不存在）
    data_dir = "src/data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 检查数据目录是否有文件
    if not os.listdir(data_dir):
        print("📁 数据目录为空，创建示例文档...")
        # 创建示例文档
        sample_content = """水生动物疾病知识库示例文档

常见鱼类疾病诊断指南：

1. 烂鳃病（细菌性鳃病）
   症状：鳃部发白、腐烂、分泌粘液增多
   病因：弧菌、爱德华氏菌感染
   治疗：使用抗生素如氟苯尼考，改善水质
   预防：保持良好水质，定期消毒

2. 肠炎病
   症状：腹部肿胀、食欲不振、排便异常
   病因：饲料变质、水质恶化
   治疗：停食1-2天，使用肠道调理剂
   预防：投喂新鲜饲料，控制投喂量

3. 水霉病（真菌病）
   症状：体表出现白色絮状物
   病因：水霉菌感染，多发生在受伤鱼体
   治疗：使用抗真菌药物，提高水温
   预防：避免鱼体受伤，保持水质清洁

4. 白点病（小瓜虫病）
   症状：体表、鳍条出现白色小点
   病因：小瓜虫寄生
   治疗：升温至28-30℃，加盐治疗
   预防：新鱼检疫隔离，保持水温稳定"""
        
        with open(os.path.join(data_dir, "marine_disease_guide.txt"), "w", encoding="utf-8") as f:
            f.write(sample_content)
        print("✅ 已创建示例知识库文档")

    # 读取本地知识文件
    print("📚 加载知识库文档...")
    documents = SimpleDirectoryReader(input_dir=data_dir, recursive=True).load_data()

    if not documents:
        print("❌ 没有找到文档，请在 src/data 目录中添加知识文档")
        return

    print(f"✅ 已加载 {len(documents)} 个文档")

    # 构建向量索引
    print("🔍 构建向量索引...")
    index = VectorStoreIndex.from_documents(documents)
    
    # 尝试加载OceanGPT模型
    print("🤖 加载OceanGPT模型...")
    model_loaded = await oceangpt_manager.load_model()
    
    if model_loaded:
        print("✅ OceanGPT模型加载成功！")
        query_engine = index.as_query_engine()
        use_oceangpt = True
    else:
        print("⚠️ OceanGPT模型加载失败，使用纯检索模式")
        print("💡 提示：请确保已下载OceanGPT模型文件或检查网络连接")
        query_engine = None
        use_oceangpt = False

    print("\n🚀 蓝海智询系统已启动！")
    print("💡 使用说明：")
    print("   - 输入鱼病相关问题，系统将基于知识库和OceanGPT模型回答")
    print("   - 输入 'q' 或 'quit' 退出系统")
    print("   - 支持中英文问答")
    print("="*60)

    # 处理用户查询
    while True:
        try:
            query = input("\n🐟 请输入鱼病问题：").strip()
            
            if query.lower() in ['q', 'quit', '退出']:
                print("👋 感谢使用蓝海智询系统！")
                break
            
            if not query:
                print("⚠️ 请输入有效问题")
                continue
                
            print("\n🔍 正在分析...")
            
            if use_oceangpt and query_engine:
                # 使用OceanGPT完整模式
                response = query_engine.query(query)
                print(f"\n🤖 OceanGPT回答：")
                print(f"{response.response}")
                
                # 显示相关文档片段
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    print(f"\n📖 参考文档：")
                    for i, node in enumerate(response.source_nodes[:2], 1):
                        print(f"{i}. {node.text[:100]}...")
            else:
                # 纯检索模式
                retriever = index.as_retriever(similarity_top_k=3)
                nodes = retriever.retrieve(query)
                
                print(f"\n📖 找到相关文档：")
                for i, node in enumerate(nodes, 1):
                    print(f"\n{i}. 相关度: {node.score:.3f}")
                    print(f"内容: {node.text}")
                    
                # 使用OceanGPT直接回答（如果可用）
                if use_oceangpt:
                    try:
                        # 构建上下文
                        context = "\n".join([node.text for node in nodes[:2]])
                        prompt = f"""基于以下水生动物疾病知识，回答用户问题：

知识库内容：
{context}

用户问题：{query}

请提供专业、准确的回答："""
                        
                        oceangpt_response = await oceangpt_manager.generate_response(prompt)
                        print(f"\n🤖 OceanGPT专业分析：")
                        print(f"{oceangpt_response}")
                    except Exception as e:
                        logger.error(f"OceanGPT回答生成失败: {e}")
            
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，系统退出")
            break
        except Exception as e:
            logger.error(f"处理查询时出错: {e}")
            print(f"❌ 处理查询时出错: {e}")

def run():
    """同步入口函数"""
    asyncio.run(main())

if __name__ == "__main__":
    run()
