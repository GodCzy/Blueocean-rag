#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rag_service.py - RAG服务接口

该模块实现了基于检索增强生成的问答服务，包括：
1. 文档检索功能
2. 知识图谱生成功能
3. 大模型调用功能

作者: 成员C (后端工程师)
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import Depends, HTTPException, APIRouter, Query, status

# 向量检索相关
import faiss
from sentence_transformers import SentenceTransformer

# 大模型调用相关
from vllm import LLM, SamplingParams

# 项目内部导入
from src.models.index_model import SearchResult, DocumentDetail, GraphData
from src.config.settings import Settings, get_settings
from src.utils.logger import setup_logger

# 配置日志
logger = setup_logger("rag_service")

class RAGService:
    """RAG服务类，封装检索增强生成功能"""
    
    def __init__(self, settings: Settings = Depends(get_settings)):
        """初始化RAG服务
        
        Args:
            settings: 应用配置
        """
        self.settings = settings
        self.model_loaded = False
        self.index_loaded = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 向量索引相关
        self.index_path = Path(settings.processed_dir) / "faiss_index"
        self.doc_mapping_path = Path(settings.processed_dir) / "doc_mapping.json"
        self.chunk_mapping_path = Path(settings.processed_dir) / "chunk_mapping.json"
        
        # 知识图谱相关
        self.graph_data_path = Path(settings.processed_dir) / "knowledge_graph.json"
        
        # 大模型相关
        self.llm_model_name = settings.llm_model_name
        self.embedding_model_name = settings.embedding_model_name
        
        # 尝试加载资源
        self._load_resources()
    
    def _load_resources(self):
        """加载所需资源"""
        try:
            # 加载向量索引
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"加载向量索引成功: {self.index_path}")
                
                # 加载文档映射
                with open(self.doc_mapping_path, 'r', encoding='utf-8') as f:
                    self.doc_mapping = json.load(f)
                
                with open(self.chunk_mapping_path, 'r', encoding='utf-8') as f:
                    self.chunk_mapping = json.load(f)
                
                logger.info(f"加载文档映射成功")
                self.index_loaded = True
            else:
                logger.warning(f"向量索引不存在: {self.index_path}")
                
            # 加载知识图谱数据
            if self.graph_data_path.exists():
                with open(self.graph_data_path, 'r', encoding='utf-8') as f:
                    self.graph_data = json.load(f)
                logger.info(f"加载知识图谱数据成功: {self.graph_data_path}")
            else:
                logger.warning(f"知识图谱数据不存在: {self.graph_data_path}")
                self.graph_data = {"nodes": [], "edges": []}
            
            # 加载embedding模型
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"加载embedding模型成功: {self.embedding_model_name}")
            except Exception as e:
                logger.error(f"加载embedding模型失败: {e}")
                raise
            
            # 加载LLM模型
            try:
                # 使用vLLM进行高效推理
                self.llm = LLM(
                    model=self.llm_model_name,
                    tensor_parallel_size=self.settings.tensor_parallel_size,
                    trust_remote_code=True,
                    dtype="float16"
                )
                logger.info(f"加载LLM模型成功: {self.llm_model_name}")
                self.model_loaded = True
            except Exception as e:
                logger.error(f"加载LLM模型失败: {e}")
                self.model_loaded = False
                
        except Exception as e:
            logger.error(f"加载资源失败: {e}")
    
    async def search(self, 
               query: str, 
               filters: Optional[List[str]] = None, 
               top_k: int = 5) -> List[SearchResult]:
        """搜索相关文档
        
        Args:
            query: 用户查询
            filters: 可选的标签过滤条件
            top_k: 返回的最大结果数量
            
        Returns:
            匹配的文档列表
        """
        if not self.index_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="索引未加载，服务不可用"
            )
        
        # 向量化查询
        query_embedding = await asyncio.to_thread(
            self.embedding_model.encode, query, normalize_embeddings=True
        )
        
        # 执行向量检索
        D, I = await asyncio.to_thread(
            self.index.search, 
            np.array([query_embedding]).astype('float32'), 
            min(top_k * 2, len(self.chunk_mapping))  # 先检索更多结果，然后过滤
        )
        
        # 处理搜索结果
        results = []
        seen_doc_ids = set()
        
        for i, (score, idx) in enumerate(zip(D[0], I[0])):
            if idx < 0 or idx >= len(self.chunk_mapping):
                continue
                
            chunk_id = str(idx)
            if chunk_id not in self.chunk_mapping:
                continue
                
            chunk_info = self.chunk_mapping[chunk_id]
            doc_id = chunk_info.get("doc_id")
            
            # 防止同一文档出现多次
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            
            # 获取文档详情
            if doc_id in self.doc_mapping:
                doc_info = self.doc_mapping[doc_id]
                
                # 标签过滤
                if filters and len(filters) > 0:
                    doc_tags = set(doc_info.get("tags", []))
                    if not any(f in doc_tags for f in filters):
                        continue
                
                # 计算相关度分数 (转换为0-1范围)
                relevance = float(max(0.0, min(1.0, (1.0 + score) / 2.0)))
                
                # 提取高亮片段
                highlight = chunk_info.get("text", "")
                if len(highlight) > 200:
                    highlight = highlight[:200] + "..."
                
                results.append(SearchResult(
                    id=doc_id,
                    title=doc_info.get("title", "未知标题"),
                    highlight=highlight,
                    relevance=relevance,
                    tags=doc_info.get("tags", [])
                ))
            
            # 达到所需数量后结束
            if len(results) >= top_k:
                break
        
        # 按相关度排序
        results.sort(key=lambda x: x.relevance, reverse=True)
        return results
    
    async def get_document(self, doc_id: str) -> DocumentDetail:
        """获取文档详细信息
        
        Args:
            doc_id: 文档ID
            
        Returns:
            文档详细信息
        """
        if not self.index_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="索引未加载，服务不可用"
            )
            
        if doc_id not in self.doc_mapping:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"文档不存在: {doc_id}"
            )
        
        doc_info = self.doc_mapping[doc_id]
        
        # 收集所有相关的文本块
        chunks = []
        for chunk_id, chunk_info in self.chunk_mapping.items():
            if chunk_info.get("doc_id") == doc_id:
                chunks.append(chunk_info)
        
        # 按序号排序
        chunks.sort(key=lambda x: x.get("chunk_index", 0))
        
        # 合并文本
        full_text = "\n".join([c.get("text", "") for c in chunks])
        
        # 构建文档详情
        return DocumentDetail(
            id=doc_id,
            title=doc_info.get("title", "未知标题"),
            content=full_text,
            metadata=doc_info.get("metadata", {}),
            tags=doc_info.get("tags", []),
            entities=doc_info.get("entities", []),
            relations=doc_info.get("relations", [])
        )
    
    async def get_graph(self, doc_id: Optional[str] = None) -> GraphData:
        """获取知识图谱数据
        
        Args:
            doc_id: 可选的文档ID，如果提供则返回该文档相关的子图
            
        Returns:
            知识图谱数据
        """
        if not self.graph_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="知识图谱数据不存在"
            )
        
        # 如果没有指定文档ID，返回完整图谱
        if not doc_id:
            return GraphData(
                nodes=self.graph_data.get("nodes", []),
                edges=self.graph_data.get("edges", [])
            )
        
        # 获取文档实体
        if doc_id not in self.doc_mapping:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"文档不存在: {doc_id}"
            )
        
        doc_info = self.doc_mapping[doc_id]
        doc_entities = doc_info.get("entities", [])
        
        # 提取实体ID
        entity_ids = [e.get("id") for e in doc_entities if "id" in e]
        
        # 筛选节点
        nodes = []
        relevant_nodes = set(entity_ids)
        
        for node in self.graph_data.get("nodes", []):
            if node.get("id") in relevant_nodes:
                nodes.append(node)
        
        # 筛选边：两端节点都在相关节点集合中
        edges = []
        for edge in self.graph_data.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            if source in relevant_nodes and target in relevant_nodes:
                edges.append(edge)
        
        return GraphData(nodes=nodes, edges=edges)
    
    async def answer_query(self, 
                     query: str, 
                     context: Optional[List[Dict[str, Any]]] = None) -> str:
        """使用LLM回答问题
        
        Args:
            query: 用户问题
            context: 可选的上下文信息
            
        Returns:
            模型生成的回答
        """
        if not self.model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM模型未加载，服务不可用"
            )
        
        # 如果没有上下文，则进行检索
        if not context:
            search_results = await self.search(query, top_k=3)
            context = []
            
            for result in search_results:
                doc = await self.get_document(result.id)
                context.append({
                    "title": doc.title,
                    "content": doc.content,
                    "relevance": result.relevance
                })
        
        # 构建提示
        system_prompt = f"""你是一个水生动物疾病专家助手，请根据提供的上下文信息回答用户关于水生动物疾病的问题。
如果上下文中没有相关信息，请直接说明无法回答，不要编造信息。
回答时应注重专业性和准确性，并给出相应的治疗建议和预防措施。

上下文信息：
"""

        for i, ctx in enumerate(context):
            system_prompt += f"\n文档{i+1}: {ctx['title']}\n{ctx['content']}\n"
        
        user_prompt = f"问题：{query}"
        
        # 构建提示模板
        prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_prompt} [/INST]"""
        
        # 调用模型生成回答
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
        )
        
        # 使用vLLM进行高效推理
        outputs = await asyncio.to_thread(
            self.llm.generate,
            [prompt],
            sampling_params
        )
        
        # 提取生成的文本
        generated_text = outputs[0].outputs[0].text
        
        return generated_text

# 创建路由器
router = APIRouter()

@router.post("/search", response_model=List[SearchResult])
async def search(
    query: str,
    filters: Optional[List[str]] = Query(None),
    limit: int = Query(10, ge=1, le=50),
    rag_service: RAGService = Depends(),
):
    """搜索API端点"""
    return await rag_service.search(query, filters, limit)

@router.get("/document/{doc_id}", response_model=DocumentDetail)
async def get_document(
    doc_id: str,
    rag_service: RAGService = Depends(),
):
    """获取文档详情"""
    return await rag_service.get_document(doc_id)

@router.get("/graph/{doc_id}", response_model=GraphData)
async def get_document_graph(
    doc_id: str,
    rag_service: RAGService = Depends(),
):
    """获取文档关联的知识图谱"""
    return await rag_service.get_graph(doc_id)

@router.get("/graph", response_model=GraphData)
async def get_full_graph(
    rag_service: RAGService = Depends(),
):
    """获取完整知识图谱"""
    return await rag_service.get_graph()

@router.post("/answer")
async def answer_query(
    query: str,
    context: Optional[List[Dict[str, Any]]] = None,
    rag_service: RAGService = Depends(),
):
    """问答API端点"""
    answer = await rag_service.answer_query(query, context)
    return {"answer": answer} 