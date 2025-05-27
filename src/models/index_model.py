#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
index_model.py - 索引和文档模型

该模块定义了用于RAG系统的各种数据模型和模式。

作者: 成员C (后端工程师)
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class SearchResult(BaseModel):
    """搜索结果模型"""
    
    id: str = Field(..., description="文档唯一标识符")
    title: str = Field(..., description="文档标题")
    highlight: str = Field(..., description="高亮的文本片段")
    relevance: float = Field(..., ge=0.0, le=1.0, description="相关度得分，范围0-1")
    tags: List[str] = Field(default_factory=list, description="文档标签")

class Entity(BaseModel):
    """实体模型"""
    
    id: str = Field(..., description="实体唯一标识符")
    text: str = Field(..., description="实体文本")
    category: str = Field(..., description="实体类别")
    start: Optional[int] = Field(None, description="实体在原文中的起始位置")
    end: Optional[int] = Field(None, description="实体在原文中的结束位置")
    properties: Dict[str, Any] = Field(default_factory=dict, description="实体属性")

class Relation(BaseModel):
    """关系模型"""
    
    relation_type: str = Field(..., description="关系类型")
    source: str = Field(..., description="源实体ID")
    target: str = Field(..., description="目标实体ID")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="关系置信度")
    properties: Dict[str, Any] = Field(default_factory=dict, description="关系属性")

class TextChunk(BaseModel):
    """文本块模型"""
    
    id: str = Field(..., description="块ID")
    doc_id: str = Field(..., description="所属文档ID")
    text: str = Field(..., description="块文本内容")
    chunk_index: int = Field(..., description="在文档中的索引位置")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="块中的实体")

class DocumentDetail(BaseModel):
    """文档详情模型"""
    
    id: str = Field(..., description="文档唯一标识符")
    title: str = Field(..., description="文档标题")
    content: str = Field(..., description="文档完整内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")
    tags: List[str] = Field(default_factory=list, description="文档标签")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="文档中的实体")
    relations: List[Dict[str, Any]] = Field(default_factory=list, description="文档中的关系")

class GraphNode(BaseModel):
    """图谱节点模型"""
    
    id: str = Field(..., description="节点唯一标识符")
    name: str = Field(..., description="节点名称")
    type: str = Field(..., description="节点类型")
    properties: Dict[str, Any] = Field(default_factory=dict, description="节点属性")

class GraphEdge(BaseModel):
    """图谱边模型"""
    
    source: str = Field(..., description="源节点ID")
    target: str = Field(..., description="目标节点ID")
    label: str = Field(..., description="边标签")
    value: float = Field(1.0, description="边权重")
    properties: Dict[str, Any] = Field(default_factory=dict, description="边属性")

class GraphData(BaseModel):
    """知识图谱数据模型"""
    
    nodes: List[Dict[str, Any]] = Field(default_factory=list, description="图谱节点列表")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="图谱边列表")

class AnswerResponse(BaseModel):
    """问答响应模型"""
    
    answer: str = Field(..., description="回答内容")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="信息来源")
    graph_data: Optional[GraphData] = Field(None, description="相关知识图谱")

class QueryRequest(BaseModel):
    """查询请求模型"""
    
    query: str = Field(..., description="用户问题")
    filters: Optional[List[str]] = Field(None, description="过滤标签")
    limit: int = Field(10, ge=1, le=50, description="返回结果数量限制")

class BatchProcessRequest(BaseModel):
    """批量处理请求模型"""
    
    input_dir: str = Field(..., description="输入目录路径")
    output_dir: str = Field(..., description="输出目录路径")
    processor_type: str = Field(..., description="处理器类型")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="处理参数")

class ProcessResult(BaseModel):
    """处理结果模型"""
    
    success: bool = Field(..., description="处理是否成功")
    message: str = Field(..., description="结果消息")
    processed_count: int = Field(0, description="处理的文件数量")
    details: Dict[str, Any] = Field(default_factory=dict, description="详细信息")

class DiagnosisRequest(BaseModel):
    """诊断请求模型"""
    
    symptoms: List[str] = Field(..., description="症状列表")
    species: Optional[str] = Field(None, description="水生动物种类")
    include_references: bool = Field(False, description="是否包含参考资料")

class DiagnosisResponse(BaseModel):
    """诊断响应模型"""
    
    diagnosis: str = Field(..., description="诊断结果")
    possible_diseases: List[str] = Field(default_factory=list, description="可能的疾病")
    confidence: float = Field(..., ge=0.0, le=1.0, description="诊断置信度")
    treatment_suggestions: List[str] = Field(default_factory=list, description="治疗建议")
    prevention_measures: List[str] = Field(default_factory=list, description="预防措施")
    references: Optional[List[Dict[str, Any]]] = Field(None, description="参考资料")
    symptoms: List[str] = Field(default_factory=list, description="输入的症状列表")
    species: Optional[str] = Field(None, description="水生动物种类") 