"""
RAG API for aquatic animals disease diagnosis and QA
实现基于FAISS的向量检索和OceanGPT问答逻辑
"""
import os
from typing import List, Dict, Any, Optional
import time

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.llms import LLM

from src.utils.logger import setup_logger
from src.core.oceangpt_manager import OceanGPTManager, ModelConfig

# 配置日志
logger = setup_logger("rag_api")

class RAGService:
    """基于FAISS的检索增强生成问答服务，集成OceanGPT"""
    
    def __init__(
        self, 
        data_dir: str = "datasets/fish_docs", 
        index_path: str = "datasets/processed/faiss_index",
        oceangpt_config: ModelConfig = None,
        top_k: int = 5,
    ):
        """
        初始化RAG服务
        
        Args:
            data_dir: 知识库文件目录
            index_path: FAISS索引存储路径
            oceangpt_config: OceanGPT模型配置
            top_k: 检索返回的文档数量
        """
        self.data_dir = data_dir
        self.index_path = index_path
        self.top_k = top_k
        self.index = None
        self.oceangpt_manager = None
        
        # 初始化OceanGPT
        if oceangpt_config:
            self.oceangpt_manager = OceanGPTManager(oceangpt_config)
        
        # 加载索引
        self._load_or_create_index()
    
    async def initialize_oceangpt(self) -> bool:
        """异步初始化OceanGPT模型"""
        if self.oceangpt_manager:
            try:
                success = await self.oceangpt_manager.load_model()
                if success:
                    logger.info("OceanGPT模型加载成功")
                    return True
                else:
                    logger.warning("OceanGPT模型加载失败")
                    return False
            except Exception as e:
                logger.error(f"OceanGPT初始化失败: {e}")
                return False
        return False
    
    def _load_or_create_index(self):
        """加载或创建向量索引"""
        try:
            # 如果索引文件存在，则加载
            if os.path.exists(self.index_path):
                logger.info(f"Loading existing index from {self.index_path}")
                vector_store = FaissVectorStore.from_persist_dir(self.index_path)
                self.index = VectorStoreIndex.from_vector_store(vector_store)
            else:
                # 否则创建新索引
                logger.info(f"Creating new index from {self.data_dir}")
                # 创建必要的目录
                os.makedirs(self.data_dir, exist_ok=True)
                os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
                
                # 检查是否有文档
                if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir):
                    logger.warning(f"No documents found in {self.data_dir}. Creating empty index.")
                    # 创建一个空索引
                    vector_store = FaissVectorStore(dim=1536)
                    self.index = VectorStoreIndex.from_documents(
                        [],
                        vector_store=vector_store,
                    )
                    vector_store.persist(persist_dir=self.index_path)
                    return
                
                # 加载文档
                documents = SimpleDirectoryReader(
                    input_dir=self.data_dir, 
                    recursive=True
                ).load_data()
                
                # 创建向量存储
                vector_store = FaissVectorStore(dim=1536)  # 根据具体的嵌入模型维度设置
                
                # 创建索引
                if self.oceangpt_manager:
                    Settings.llm = self.oceangpt_manager.get_llm()
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    vector_store=vector_store,
                )
                
                # 持久化索引
                vector_store.persist(persist_dir=self.index_path)
                
            logger.info("Index loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load or create index: {str(e)}")
            raise
    
    def rebuild_index(self):
        """重建索引"""
        import shutil
        try:
            # 删除现有索引
            if os.path.exists(self.index_path):
                shutil.rmtree(self.index_path)
            
            # 重新构建索引
            self._load_or_create_index()
            return {"status": "success", "message": "Index rebuilt successfully"}
        except Exception as e:
            logger.error(f"Failed to rebuild index: {str(e)}")
            return {"status": "error", "message": f"Failed to rebuild index: {str(e)}"}
    
    async def ask(self, query: str, response_mode: str = "compact") -> Dict[str, Any]:
        """
        处理用户问题，执行向量检索和OceanGPT回答生成
        
        Args:
            query: 用户问题
            response_mode: 响应模式，可选值为compact、refine、simple等
        
        Returns:
            包含回答和相关文本的字典
        """
        start_time = time.time()
        try:
            # 检查索引是否已加载
            if self.index is None:
                raise ValueError("索引未加载，请先初始化索引")
                
            # 创建检索器
            retriever = self.index.as_retriever(similarity_top_k=self.top_k)
            
            # 执行检索
            retrieved_nodes = retriever.retrieve(query)
            
            # 获取检索结果文本
            source_texts = []
            context = ""
            if retrieved_nodes:
                source_texts = [node.node.text for node in retrieved_nodes]
                context = "\n\n".join(source_texts[:3])  # 使用前3个最相关的文档作为上下文
            
            # 生成回答
            answer = ""
            if self.oceangpt_manager and self.oceangpt_manager.model:
                # 使用OceanGPT生成回答
                try:
                    prompt = f"""你是一个专业的水生动物疾病诊断专家。基于以下知识库内容，回答用户的问题。

知识库内容：
{context}

用户问题：{query}

请提供专业、准确、详细的回答："""
                    
                    answer = await self.oceangpt_manager.generate_response(prompt)
                except Exception as e:
                    logger.error(f"OceanGPT生成回答失败: {e}")
                    answer = f"OceanGPT生成回答时出现错误，但找到了相关文档。请查看检索结果。"
            else:
                # 如果OceanGPT不可用，返回检索结果
                if source_texts:
                    answer = f"找到相关信息：\n\n{context}"
                else:
                    answer = "抱歉，没有找到相关信息。"
            
            elapsed_time = time.time() - start_time
            
            # 构建结果
            result = {
                "answer": answer,
                "source_documents": source_texts,
                "elapsed_time": elapsed_time,
                "has_oceangpt": self.oceangpt_manager and self.oceangpt_manager.model is not None,
            }
            
            return result
        except Exception as e:
            logger.error(f"Error in ask: {str(e)}")
            return {
                "answer": f"处理您的问题时发生错误: {str(e)}",
                "source_documents": [],
                "elapsed_time": time.time() - start_time,
                "has_oceangpt": False,
            }
    
    def diagnose(self, symptoms: List[str], species: Optional[str] = None) -> Dict[str, Any]:
        """
        根据症状进行疾病诊断
        
        Args:
            symptoms: 症状列表
            species: 水生动物种类（可选）
        
        Returns:
            诊断结果字典
        """
        start_time = time.time()
        
        try:
            # 构建诊断查询
            query = f"水生动物疾病诊断: "
            
            if species:
                query += f"针对{species}，"
            
            query += f"出现以下症状: {', '.join(symptoms)}。这可能是什么疾病？给出详细诊断和处理建议。"
            
            # 使用RAG执行诊断
            result = self.ask(query, response_mode="tree_summarize")
            
            # 添加诊断特定字段
            result["symptoms"] = symptoms
            result["species"] = species
            result["diagnosis_time"] = time.time() - start_time
            
            return result
        except Exception as e:
            logger.error(f"Error in diagnose: {str(e)}")
            return {
                "answer": f"处理诊断请求时发生错误: {str(e)}",
                "source_documents": [],
                "symptoms": symptoms,
                "species": species,
                "elapsed_time": time.time() - start_time,
            }


# 单例实例，用于全局访问
rag_service = None

def get_rag_service(
    data_dir: str = "datasets/fish_docs",
    index_path: str = "datasets/processed/faiss_index",
    oceangpt_config: ModelConfig = None,
    top_k: int = 5
) -> RAGService:
    """
    获取RAG服务的单例实例
    
    Args:
        data_dir: 知识库文件目录
        index_path: FAISS索引存储路径
        oceangpt_config: OceanGPT模型配置
        top_k: 检索返回的文档数量
    
    Returns:
        RAG服务实例
    """
    global rag_service
    
    if rag_service is None:
        rag_service = RAGService(
            data_dir=data_dir,
            index_path=index_path,
            oceangpt_config=oceangpt_config,
            top_k=top_k
        )
    
    return rag_service
