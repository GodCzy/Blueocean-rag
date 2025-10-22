"""
嵌入器模块 - 负责文本向量化
"""
from typing import List, Dict, Any, Optional, Union
import numpy as np
import os
import time

# 尝试导入各种嵌入模型库
try:
    from langchain.embeddings import HuggingFaceEmbeddings
    HAVE_HUGGINGFACE = True
except ImportError:
    HAVE_HUGGINGFACE = False

try:
    from FlagEmbedding import FlagModel
    HAVE_FLAGEMBEDDING = True
except ImportError:
    HAVE_FLAGEMBEDDING = False

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
    HAVE_OPENAI = True
except ImportError:
    HAVE_OPENAI = False

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """文本嵌入器，支持多种嵌入模型"""
    
    def __init__(
        self, 
        model_type: str = "bge",
        model_name: str = "BAAI/bge-large-zh-v1.5",
        openai_api_key: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        初始化嵌入器
        
        Args:
            model_type: 嵌入模型类型，支持 "bge" (FlagEmbedding), "huggingface", "openai"
            model_name: 模型名称或路径
            openai_api_key: OpenAI API密钥，仅当model_type为"openai"时需要
            device: 设备，"cpu"或"cuda"
        """
        self.model_type = model_type
        self.model_name = model_name
        self.device = device
        self.model = None
        
        # 根据指定的模型类型加载模型
        self._load_model(openai_api_key)
    
    def _load_model(self, openai_api_key: Optional[str] = None):
        """加载嵌入模型"""
        try:
            if self.model_type == "bge":
                # 使用FlagEmbedding的BGE模型
                if not HAVE_FLAGEMBEDDING:
                    raise ImportError("FlagEmbedding库未安装，请使用 pip install FlagEmbedding 安装")
                
                self.model = FlagModel(self.model_name, 
                                      query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                      device=self.device)
                logger.info(f"Loaded BGE model: {self.model_name}")
            
            elif self.model_type == "huggingface":
                # 使用HuggingFace模型
                if not HAVE_HUGGINGFACE:
                    raise ImportError("langchain.embeddings库未安装，请使用 pip install langchain-community 安装")
                
                self.model = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={"device": self.device}
                )
                logger.info(f"Loaded HuggingFace model: {self.model_name}")
            
            elif self.model_type == "openai":
                # 使用OpenAI模型
                if not HAVE_OPENAI:
                    raise ImportError("llama_index.embeddings库未安装，请使用 pip install llama-index 安装")
                
                if not openai_api_key:
                    openai_api_key = os.environ.get("OPENAI_API_KEY")
                    if not openai_api_key:
                        raise ValueError("OpenAI API密钥未提供，请设置openai_api_key参数或OPENAI_API_KEY环境变量")
                
                self.model = OpenAIEmbedding(api_key=openai_api_key)
                logger.info("Loaded OpenAI embedding model")
            
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        except Exception as e:
            logger.error(f"加载嵌入模型时出错: {str(e)}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        对查询文本进行向量化
        
        Args:
            query: 查询文本
        
        Returns:
            查询文本的嵌入向量
        """
        try:
            if not query or not query.strip():
                raise ValueError("查询文本不能为空")
            
            # 根据模型类型调用相应的嵌入方法
            if self.model_type == "bge":
                # BGE模型使用encode_queries方法
                vector = self.model.encode_queries([query])[0]
            
            elif self.model_type == "huggingface":
                # HuggingFace模型使用embed_query方法
                vector = self.model.embed_query(query)
            
            elif self.model_type == "openai":
                # OpenAI模型使用get_query_embedding方法
                vector = self.model.get_query_embedding(query)
            
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
            
            # 转换为NumPy数组
            return np.array(vector)
        
        except Exception as e:
            logger.error(f"对查询文本进行向量化时出错: {str(e)}")
            raise
    
    def embed_documents(self, documents: List[str]) -> List[np.ndarray]:
        """
        对文档列表进行向量化
        
        Args:
            documents: 文档文本列表
        
        Returns:
            文档嵌入向量列表
        """
        try:
            if not documents:
                raise ValueError("文档列表不能为空")
            
            # 过滤空文档
            valid_documents = [doc for doc in documents if doc and doc.strip()]
            if not valid_documents:
                raise ValueError("所有文档都是空的")
            
            # 根据模型类型调用相应的嵌入方法
            if self.model_type == "bge":
                # BGE模型使用encode_corpus方法
                vectors = self.model.encode_corpus(valid_documents)
            
            elif self.model_type == "huggingface":
                # HuggingFace模型使用embed_documents方法
                vectors = self.model.embed_documents(valid_documents)
            
            elif self.model_type == "openai":
                # OpenAI模型使用get_text_embedding方法
                vectors = []
                for doc in valid_documents:
                    vectors.append(self.model.get_text_embedding(doc))
            
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
            
            # 转换为NumPy数组列表
            return [np.array(vector) for vector in vectors]
        
        except Exception as e:
            logger.error(f"对文档列表进行向量化时出错: {str(e)}")
            raise
    
    def embed_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        批量计算embeddings
        
        Args:
            texts: 文本列表
            
        Returns:
            embeddings列表
        """
        try:
            if self.model_type == "bge":
                # BGE模型批量处理
                vectors = self.model.encode_corpus(texts)
                return [vector.tolist() for vector in vectors]
            
            elif self.model_type == "huggingface":
                # HuggingFace模型批量处理
                vectors = self.model.embed_documents(texts)
                return [vector if isinstance(vector, list) else vector.tolist() for vector in vectors]
            
            elif self.model_type == "openai":
                # OpenAI模型逐个处理
                vectors = []
                for text in texts:
                    vector = self.model.get_text_embedding(text)
                    vectors.append(vector)
                return vectors
            
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
                
        except ImportError:
            raise ImportError("所需的embedding库未安装，请检查依赖")
        except Exception as e:
            logger.error(f"批量embedding计算失败: {str(e)}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "device": self.device
        }
