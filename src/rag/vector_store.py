"""
向量存储模块 - 基于FAISS的向量数据库实现
"""
import os
from typing import List, Optional, Dict, Any, Union
import numpy as np
import faiss
import pickle
import json
import time

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FAISSVectorStore:
    """
    基于FAISS的向量存储实现
    
    提供文档嵌入向量的存储、检索和管理功能
    """
    
    def __init__(
        self, 
        index_path: str = "src/data/faiss_index", 
        dimension: int = 1536,
        index_type: str = "L2"
    ):
        """
        初始化FAISS向量存储
        
        Args:
            index_path: 索引文件存储路径
            dimension: 向量维度，与使用的嵌入模型相关
            index_type: 索引类型，L2表示欧氏距离, IP表示内积，COSINE表示余弦相似度
        """
        self.index_path = index_path
        self.dimension = dimension
        self.index_type = index_type
        
        # 索引文件路径
        self.index_file = os.path.join(index_path, "faiss.index")
        self.metadata_file = os.path.join(index_path, "metadata.json")
        self.docstore_file = os.path.join(index_path, "docstore.pkl")
        
        # 初始化索引
        self.index = None
        self.docstore = {}  # 文档ID到文档内容的映射
        self.metadata = {
            "dimension": dimension,
            "index_type": index_type,
            "count": 0,
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        # 加载或创建索引
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """加载或创建FAISS索引"""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                # 加载现有索引
                self.index = faiss.read_index(self.index_file)
                
                # 加载元数据
                with open(self.metadata_file, "r") as f:
                    self.metadata = json.load(f)
                
                # 加载文档存储
                if os.path.exists(self.docstore_file):
                    with open(self.docstore_file, "rb") as f:
                        self.docstore = pickle.load(f)
                
                logger.info(f"Loaded FAISS index with {self.metadata.get('count', 0)} vectors")
            else:
                # 创建目录
                os.makedirs(self.index_path, exist_ok=True)
                
                # 创建新索引
                if self.index_type == "L2":
                    self.index = faiss.IndexFlatL2(self.dimension)
                elif self.index_type == "IP":
                    self.index = faiss.IndexFlatIP(self.dimension)
                elif self.index_type == "COSINE":
                    # 对于余弦相似度，我们使用IP索引，但需要在添加向量时进行归一化
                    self.index = faiss.IndexFlatIP(self.dimension)
                else:
                    raise ValueError(f"Unsupported index type: {self.index_type}")
                
                # 初始化元数据
                self.metadata = {
                    "dimension": self.dimension,
                    "index_type": self.index_type,
                    "count": 0,
                    "created_at": time.time(),
                    "updated_at": time.time()
                }
                
                # 初始化文档存储
                self.docstore = {}
                
                # 保存索引和元数据
                self._save_index()
                
                logger.info(f"Created new FAISS index with dimension {self.dimension}")
        except Exception as e:
            logger.error(f"Error loading or creating FAISS index: {str(e)}")
            raise
    
    def _save_index(self):
        """保存索引、元数据和文档存储"""
        try:
            # 更新元数据
            self.metadata["updated_at"] = time.time()
            
            # 保存索引
            faiss.write_index(self.index, self.index_file)
            
            # 保存元数据
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
            
            # 保存文档存储
            with open(self.docstore_file, "wb") as f:
                pickle.dump(self.docstore, f)
            
            logger.info(f"Saved FAISS index with {self.metadata.get('count', 0)} vectors")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
            raise
    
    def add_vectors(self, vectors: List[np.ndarray], documents: List[Dict[str, Any]]) -> List[int]:
        """
        添加向量和对应的文档
        
        Args:
            vectors: 文档嵌入向量列表
            documents: 文档内容列表，每个文档是一个字典
        
        Returns:
            添加的文档ID列表
        """
        if not vectors or len(vectors) != len(documents):
            raise ValueError("Vectors and documents must have the same length")
        
        try:
            # 将向量转换为NumPy数组
            vectors_np = np.array(vectors).astype("float32")
            
            # 如果是余弦相似度索引，则对向量进行归一化
            if self.index_type == "COSINE":
                faiss.normalize_L2(vectors_np)
            
            # 添加向量到索引
            start_id = self.metadata.get("count", 0)
            self.index.add(vectors_np)
            
            # 更新文档存储
            doc_ids = list(range(start_id, start_id + len(vectors)))
            for i, doc_id in enumerate(doc_ids):
                self.docstore[doc_id] = documents[i]
            
            # 更新元数据
            self.metadata["count"] = start_id + len(vectors)
            
            # 保存索引
            self._save_index()
            
            return doc_ids
        except Exception as e:
            logger.error(f"Error adding vectors to FAISS index: {str(e)}")
            raise
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索最相似的向量
        
        Args:
            query_vector: 查询向量
            k: 返回的结果数量
        
        Returns:
            包含相似度分数和文档内容的结果列表
        """
        try:
            # 确保查询向量是二维数组
            query_vector = query_vector.reshape(1, -1).astype("float32")
            
            # 如果是余弦相似度索引，则对查询向量进行归一化
            if self.index_type == "COSINE":
                faiss.normalize_L2(query_vector)
            
            # 执行搜索
            distances, indices = self.index.search(query_vector, k)
            
            # 构建结果
            results = []
            for i in range(len(indices[0])):
                doc_id = indices[0][i]
                distance = distances[0][i]
                
                # 只返回有效的文档ID
                if doc_id != -1 and doc_id in self.docstore:
                    # 计算相似度分数
                    if self.index_type in ["IP", "COSINE"]:
                        # 对于内积或余弦相似度，分数越高越好
                        score = float(distance)
                    else:
                        # 对于L2距离，分数越低越好，我们转换为[0,1]范围
                        score = 1.0 / (1.0 + float(distance))
                    
                    # 添加结果
                    result = {
                        "doc_id": int(doc_id),
                        "score": score,
                        "document": self.docstore[doc_id]
                    }
                    results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error searching FAISS index: {str(e)}")
            raise
    
    def delete(self, doc_ids: List[int]) -> bool:
        """
        删除文档
        
        注意：FAISS不支持直接删除向量，此方法仅删除文档存储中的内容
        要完全删除向量，需要重建索引
        
        Args:
            doc_ids: 要删除的文档ID列表
        
        Returns:
            操作是否成功
        """
        try:
            for doc_id in doc_ids:
                if doc_id in self.docstore:
                    del self.docstore[doc_id]
            
            # 保存文档存储
            with open(self.docstore_file, "wb") as f:
                pickle.dump(self.docstore, f)
            
            logger.info(f"Deleted {len(doc_ids)} documents from docstore")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    def clear(self) -> bool:
        """
        清空向量存储
        
        Returns:
            操作是否成功
        """
        try:
            # 创建新索引
            if self.index_type == "L2":
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "IP":
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "COSINE":
                self.index = faiss.IndexFlatIP(self.dimension)
            
            # 重置元数据
            self.metadata["count"] = 0
            self.metadata["updated_at"] = time.time()
            
            # 清空文档存储
            self.docstore = {}
            
            # 保存索引
            self._save_index()
            
            logger.info("Cleared FAISS index and docstore")
            return True
        except Exception as e:
            logger.error(f"Error clearing FAISS index: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取向量存储统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "dimension": self.metadata.get("dimension"),
            "index_type": self.metadata.get("index_type"),
            "vector_count": self.metadata.get("count"),
            "document_count": len(self.docstore),
            "created_at": self.metadata.get("created_at"),
            "updated_at": self.metadata.get("updated_at")
        }
