"""
RAG Pipeline - 集成嵌入器和向量存储，提供完整的RAG流程
"""
from typing import List, Dict, Any, Optional, Union, Tuple
import os
import time
import json
from pathlib import Path
import numpy as np

from src.rag.embedder import Embedder
from src.rag.vector_store import FAISSVectorStore
from src.utils.logging_config import logger


class RAGPipeline:
    """
    检索增强生成管道
    
    整合文档处理、嵌入计算和向量检索的完整RAG流程
    """
    
    def __init__(
        self,
        index_path: str = "src/data/faiss_index",
        embedder_model_type: str = "bge",
        embedder_model_name: str = "BAAI/bge-large-zh-v1.5",
        vector_dimension: int = 1536,
        similarity_top_k: int = 5,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        device: str = "cpu"
    ):
        """
        初始化RAG管道
        
        Args:
            index_path: 索引文件存储路径
            embedder_model_type: 嵌入模型类型
            embedder_model_name: 嵌入模型名称
            vector_dimension: 向量维度
            similarity_top_k: 检索返回的文档数量
            chunk_size: 文档分块大小
            chunk_overlap: 文档分块重叠大小
            device: 设备，"cpu"或"cuda"
        """
        self.index_path = index_path
        self.similarity_top_k = similarity_top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 初始化嵌入器
        try:
            self.embedder = Embedder(
                model_type=embedder_model_type,
                model_name=embedder_model_name,
                device=device
            )
            
            # 获取嵌入向量维度
            if embedder_model_type == "bge" and "bge-large" in embedder_model_name:
                vector_dimension = 1024
            elif embedder_model_type == "bge" and "bge-base" in embedder_model_name:
                vector_dimension = 768
            
            # 初始化向量存储
            self.vector_store = FAISSVectorStore(
                index_path=index_path,
                dimension=vector_dimension
            )
            
            logger.info(f"Initialized RAG pipeline with {embedder_model_type} embedder and FAISS vector store")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            raise
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        处理文档并存入向量库
        
        Args:
            documents: 文档列表，每个文档是一个包含text和metadata的字典
        
        Returns:
            处理是否成功
        """
        try:
            if not documents:
                logger.warning("Empty document list, nothing to process")
                return False
            
            start_time = time.time()
            logger.info(f"Processing {len(documents)} documents")
            
            # 分块文档
            chunks = self._chunk_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            
            # 提取文本内容
            texts = [chunk["text"] for chunk in chunks]
            
            # 生成嵌入向量
            vectors = self.embedder.embed_documents(texts)
            logger.info(f"Generated {len(vectors)} embedding vectors")
            
            # 添加到向量存储
            self.vector_store.add_vectors(vectors, chunks)
            logger.info(f"Added vectors to vector store")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Document processing completed in {elapsed_time:.2f} seconds")
            
            return True
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return False
    
    def _chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将文档分块
        
        Args:
            documents: 文档列表
        
        Returns:
            分块后的文档列表
        """
        chunks = []
        
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            if not text or len(text) < self.chunk_size / 2:
                # 如果文档很短，不进行分块
                chunk = {
                    "text": text,
                    "metadata": metadata,
                    "chunk_id": 0,
                    "source_doc_id": metadata.get("doc_id", "unknown")
                }
                chunks.append(chunk)
                continue
            
            # 分块处理
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk_text = text[i:i + self.chunk_size]
                if not chunk_text:
                    continue
                
                # 如果是最后一个块，但太短，则合并到前一个块
                if i > 0 and len(chunk_text) < self.chunk_size / 2:
                    continue
                
                chunk_id = i // (self.chunk_size - self.chunk_overlap)
                chunk = {
                    "text": chunk_text,
                    "metadata": metadata.copy(),
                    "chunk_id": chunk_id,
                    "source_doc_id": metadata.get("doc_id", "unknown")
                }
                chunks.append(chunk)
        
        return chunks
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        检索与查询相关的文档
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量，如果为None则使用默认值
        
        Returns:
            检索结果列表
        """
        try:
            if not query or not query.strip():
                raise ValueError("查询文本不能为空")
            
            # 使用实际的top_k值
            actual_top_k = top_k if top_k is not None else self.similarity_top_k
            
            # 生成查询向量
            query_vector = self.embedder.embed_query(query)
            
            # 执行检索
            search_results = self.vector_store.search(query_vector, k=actual_top_k)
            
            # 构建结果
            results = []
            for result in search_results:
                doc = result["document"]
                results.append({
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {}),
                    "score": result["score"],
                    "doc_id": result["doc_id"]
                })
            
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise
    
    def process_directory(self, directory_path: str, recursive: bool = True) -> bool:
        """
        处理目录中的所有文本文件
        
        Args:
            directory_path: 目录路径
            recursive: 是否递归处理子目录
        
        Returns:
            处理是否成功
        """
        try:
            directory = Path(directory_path)
            if not directory.is_dir():
                raise ValueError(f"{directory_path}不是有效的目录")
            
            documents = []
            
            # 获取所有文件
            files = list(directory.glob("**/*" if recursive else "*"))
            text_extensions = [".txt", ".md", ".json", ".csv", ".html", ".htm", ".xml", ".rst", ".py", ".js", ".doc", ".docx", ".pdf"]
            
            # 过滤文本文件
            text_files = [f for f in files if f.is_file() and f.suffix.lower() in text_extensions]
            
            if not text_files:
                logger.warning(f"No text files found in {directory_path}")
                return False
            
            logger.info(f"Found {len(text_files)} text files in {directory_path}")
            
            # 处理每个文件
            for file_path in text_files:
                try:
                    # 读取文件内容
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    # 创建文档对象
                    document = {
                        "text": content,
                        "metadata": {
                            "doc_id": str(file_path),
                            "file_name": file_path.name,
                            "file_path": str(file_path),
                            "file_type": file_path.suffix,
                            "file_size": file_path.stat().st_size,
                            "created_at": file_path.stat().st_ctime,
                            "updated_at": file_path.stat().st_mtime
                        }
                    }
                    
                    documents.append(document)
                except Exception as e:
                    logger.warning(f"Error processing file {file_path}: {str(e)}")
            
            # 处理文档
            return self.process_documents(documents)
        
        except Exception as e:
            logger.error(f"Error processing directory: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取RAG管道统计信息
        
        Returns:
            统计信息字典
        """
        embedder_info = self.embedder.get_model_info()
        vector_store_stats = self.vector_store.get_stats()
        
        return {
            "embedder": embedder_info,
            "vector_store": vector_store_stats,
            "similarity_top_k": self.similarity_top_k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }
