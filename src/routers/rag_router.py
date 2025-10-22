"""
RAG Router - 为RAG检索增强问答和疾病诊断提供API路由
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Depends

# 目前仅含少量示例接口
from pydantic import BaseModel, Field

from src.services.rag import get_rag_service, RAGService
from src.utils.logger import get_logger

logger = get_logger(__name__)


# 请求和响应模型
class AskRequest(BaseModel):
    """问答请求模型"""
    query: str = Field(..., description="用户问题")
    response_mode: str = Field(default="compact", description="响应模式，可选值：compact/refine/simple/tree_summarize")


class AskResponse(BaseModel):
    """问答响应模型"""
    answer: str = Field(..., description="回答内容")
    source_documents: List[str] = Field(default_factory=list, description="检索到的原文档内容")
    elapsed_time: float = Field(..., description="处理时间(秒)")


class DiagnoseRequest(BaseModel):
    """诊断请求模型"""
    symptoms: List[str] = Field(..., description="症状列表")
    species: Optional[str] = Field(default=None, description="水生动物种类，如：鲤鱼、草鱼等")


class DiagnoseResponse(BaseModel):
    """诊断响应模型"""
    answer: str = Field(..., description="诊断结果")
    source_documents: List[str] = Field(default_factory=list, description="检索到的相关疾病文档")
    symptoms: List[str] = Field(..., description="输入的症状列表")
    species: Optional[str] = Field(default=None, description="水生动物种类")
    elapsed_time: float = Field(..., description="处理时间(秒)")
    diagnosis_time: Optional[float] = Field(default=None, description="诊断时间(秒)")


# 创建路由器
rag = APIRouter(
    prefix="/rag",
    tags=["RAG检索增强问答"],
    responses={404: {"description": "Not found"}},
)


# 依赖函数，获取RAG服务实例
def get_rag():
    return get_rag_service()


@rag.post("/ask", response_model=AskResponse, summary="通用知识问答")
async def ask(
    request: AskRequest,
    rag_service: RAGService = Depends(get_rag)
):
    """
    基于RAG检索增强的通用水生动物知识问答API
    
    - **query**: 用户问题
    - **response_mode**: 响应模式 (compact/refine/simple/tree_summarize)
    
    返回回答内容和检索到的相关文档
    """
    try:
        logger.info(f"Processing question: {request.query}")
        result = await rag_service.ask(request.query, request.response_mode)
        return result
    except Exception as e:
        logger.error(f"Error in ask endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}")


@rag.post("/disease/diagnose", response_model=DiagnoseResponse, summary="水生动物疾病诊断")
async def diagnose(
    request: DiagnoseRequest,
    rag_service: RAGService = Depends(get_rag)
):
    """
    基于症状的水生动物疾病诊断API
    
    - **symptoms**: 症状列表，例如["鳃部发白", "不吃食", "游动异常"]
    - **species**: 水生动物种类，例如"鲤鱼"（可选）
    
    返回可能的疾病诊断结果和处理建议
    """
    try:
        logger.info(f"Processing diagnosis for symptoms: {request.symptoms}, species: {request.species}")
        result = await rag_service.diagnose(request.symptoms, request.species)
        return result
    except Exception as e:
        logger.error(f"Error in diagnose endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"诊断处理失败: {str(e)}")


@rag.post("/rebuild", summary="重建知识库索引")
async def rebuild_index(
    rag_service: RAGService = Depends(get_rag)
):
    """
    重建向量检索索引
    
    此操作可能比较耗时，取决于知识库的大小
    """
    try:
        logger.info("Rebuilding vector index")
        result = rag_service.rebuild_index()
        return result
    except Exception as e:
        logger.error(f"Error rebuilding index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"索引重建失败: {str(e)}")
