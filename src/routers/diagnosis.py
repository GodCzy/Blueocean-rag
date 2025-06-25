#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnosis.py - 疾病诊断路由模块

该模块提供了与水生动物疾病诊断相关的API路由接口。

作者: 团队成员
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body, Depends
from pydantic import BaseModel, Field
import asyncio

from src.api.rag_service import RAGService
from src.models.index_model import DiagnosisRequest, DiagnosisResponse
from src.utils.logger import setup_logger
from ..core.knowledge_graph import DiseaseKnowledgeGraph
from ..core.oceangpt_manager import MarineDiseaseGPT
from ..rag.rag_engine import RAGEngine
from ..utils.logger import get_logger

# 配置日志
logger = setup_logger("diagnosis_router")

# 创建路由器
router = APIRouter(prefix="/api/diagnosis", tags=["疾病诊断"])

# 请求模型
class SymptomInput(BaseModel):
    animal_type: str
    symptoms: List[str]
    environment_info: Optional[Dict[str, Any]] = None
    severity: Optional[str] = "中等"

class WaterParameters(BaseModel):
    temperature: Optional[float] = Field(None, ge=-10, le=40)
    ph: Optional[float] = Field(None, ge=0, le=14)
    dissolved_oxygen: Optional[float] = Field(None, ge=0)


class DiagnosisRequest(BaseModel):
    animal_type: str
    symptoms: List[str]
    environment_info: Optional[Dict[str, Any]] = None
    water_parameters: Optional[WaterParameters] = None
    use_knowledge_graph: bool = True
    use_rag: bool = True

class TreatmentRequest(BaseModel):
    disease_name: str
    animal_type: str
    severity: str = "中等"
    patient_info: Optional[Dict[str, Any]] = None

# 响应模型
class DiagnosisResult(BaseModel):
    disease_candidates: List[Dict[str, Any]]
    confidence_scores: List[float]
    treatment_recommendations: List[Dict[str, Any]]
    knowledge_graph_results: Optional[List[Dict[str, Any]]] = None
    rag_response: Optional[str] = None
    environment_analysis: Optional[str] = None

class TreatmentPlan(BaseModel):
    primary_treatment: Dict[str, Any]
    alternative_treatments: List[Dict[str, Any]]
    prevention_measures: List[str]
    monitoring_indicators: List[str]
    expected_timeline: str

# 依赖注入
async def get_knowledge_graph():
    # 这里应该从应用状态中获取已初始化的知识图谱实例
    kg = DiseaseKnowledgeGraph()
    await kg.initialize()
    return kg

async def get_oceangpt():
    # 这里应该从应用状态中获取已初始化的OceanGPT实例
    gpt = MarineDiseaseGPT()
    # 假设模型已经加载
    return gpt

async def get_rag_engine():
    # 这里应该从应用状态中获取已初始化的RAG引擎实例
    rag = RAGEngine()
    await rag.initialize()
    return rag

@router.post("/diagnose", response_model=DiagnosisResult)
async def diagnose_disease(
    request: DiagnosisRequest,
    kg: DiseaseKnowledgeGraph = Depends(get_knowledge_graph),
    gpt: MarineDiseaseGPT = Depends(get_oceangpt),
    rag: RAGEngine = Depends(get_rag_engine)
):
    """
    综合诊断API：结合知识图谱、RAG和大模型进行疾病诊断
    """
    try:
        results = {}
        
        # 1. 知识图谱诊断
        if request.use_knowledge_graph:
            kg_results = await kg.diagnose_by_symptoms(request.symptoms)
            results["knowledge_graph_results"] = kg_results
            logger.info(f"知识图谱返回 {len(kg_results)} 个候选疾病")
        
        # 2. RAG检索
        rag_response = None
        if request.use_rag:
            # 构建查询
            query = f"动物类型：{request.animal_type}，症状：{', '.join(request.symptoms)}"
            if request.environment_info:
                env_desc = ', '.join([f"{k}:{v}" for k, v in request.environment_info.items()])
                query += f"，环境信息：{env_desc}"
            
            rag_result = await rag.query(query)
            rag_response = rag_result.get("answer", "")
            results["rag_response"] = rag_response
        
        # 3. 大模型诊断
        gpt_diagnosis = await gpt.diagnose_disease(
            animal_type=request.animal_type,
            symptoms=request.symptoms,
            environment_info=request.environment_info
        )
        
        # 4. 水质环境分析
        environment_analysis = None
        if request.water_parameters:
            from ..core.water_analysis import analyze_water_quality
            params = request.water_parameters
            environment_analysis = analyze_water_quality(
                params.temperature, params.ph, params.dissolved_oxygen
            )
            results["environment_analysis"] = environment_analysis
        
        # 5. 整合结果
        disease_candidates = []
        confidence_scores = []
        treatment_recommendations = []
        
        # 从知识图谱结果提取候选疾病
        if "knowledge_graph_results" in results:
            for result in results["knowledge_graph_results"][:5]:  # 取前5个
                disease_candidates.append({
                    "disease_id": result.get("disease_id"),
                    "disease_name": result.get("disease_name"),
                    "description": result.get("description"),
                    "matched_symptoms": result.get("matched_symptoms"),
                    "total_symptoms": result.get("total_symptoms"),
                    "source": "knowledge_graph"
                })
                confidence_scores.append(result.get("confidence", 0.0))
        
        # 添加大模型分析结果
        disease_candidates.append({
            "disease_name": "AI综合分析",
            "description": gpt_diagnosis,
            "source": "oceangpt"
        })
        confidence_scores.append(0.8)  # 给大模型一个默认置信度
        
        # 生成治疗建议
        if disease_candidates:
            top_disease = disease_candidates[0]
            treatment_rec = await gpt.recommend_treatment(
                disease_name=top_disease.get("disease_name", ""),
                severity=request.severity if hasattr(request, 'severity') else "中等"
            )
            treatment_recommendations.append({
                "treatment": treatment_rec,
                "for_disease": top_disease.get("disease_name")
            })
        
        from .stats import increment_diagnosis
        increment_diagnosis()
        return DiagnosisResult(
            disease_candidates=disease_candidates,
            confidence_scores=confidence_scores,
            treatment_recommendations=treatment_recommendations,
            knowledge_graph_results=results.get("knowledge_graph_results"),
            rag_response=rag_response,
            environment_analysis=environment_analysis
        )
    
    except Exception as e:
        logger.error(f"诊断过程发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"诊断失败: {str(e)}")

@router.post("/quick-diagnose")
async def quick_diagnose(
    request: SymptomInput,
    gpt: MarineDiseaseGPT = Depends(get_oceangpt)
):
    """
    快速诊断API：仅使用大模型进行诊断
    """
    try:
        diagnosis = await gpt.diagnose_disease(
            animal_type=request.animal_type,
            symptoms=request.symptoms,
            environment_info=request.environment_info
        )
        
        return {
            "diagnosis": diagnosis,
            "animal_type": request.animal_type,
            "symptoms": request.symptoms,
            "timestamp": "2024-01-01T00:00:00Z"  # 实际应该用当前时间
        }
    
    except Exception as e:
        logger.error(f"快速诊断失败: {e}")
        raise HTTPException(status_code=500, detail=f"快速诊断失败: {str(e)}")

@router.post("/treatment-plan", response_model=TreatmentPlan)
async def generate_treatment_plan(
    request: TreatmentRequest,
    gpt: MarineDiseaseGPT = Depends(get_oceangpt),
    kg: DiseaseKnowledgeGraph = Depends(get_knowledge_graph)
):
    """
    生成治疗方案API
    """
    try:
        # 从知识图谱获取标准治疗方案
        entities = await kg.search_entities_by_name(request.disease_name, "Disease")
        
        standard_treatments = []
        if entities:
            # 获取治疗信息
            disease_entity = entities[0]
            graph_result = await kg.query_entity_neighbors(disease_entity.id)
            
            for relation in graph_result.relations:
                if relation.type == "TREATED_BY":
                    treatment_entity = next(
                        (e for e in graph_result.entities if e.id == relation.target), 
                        None
                    )
                    if treatment_entity:
                        standard_treatments.append({
                            "name": treatment_entity.name,
                            "description": treatment_entity.properties.get("description", ""),
                            "effectiveness": relation.properties.get("effectiveness", "unknown")
                        })
        
        # 使用大模型生成个性化治疗方案
        gpt_treatment = await gpt.recommend_treatment(
            disease_name=request.disease_name,
            severity=request.severity
        )
        
        # 构建综合治疗方案
        primary_treatment = standard_treatments[0] if standard_treatments else {
            "name": "AI推荐治疗",
            "description": gpt_treatment,
            "effectiveness": "待评估"
        }
        
        alternative_treatments = standard_treatments[1:] if len(standard_treatments) > 1 else []
        
        prevention_measures = [
            "定期监测水质参数",
            "控制养殖密度",
            "加强营养管理",
            "建立完善的防疫体系"
        ]
        
        monitoring_indicators = [
            "动物摄食情况",
            "行为活动状态", 
            "水质变化",
            "死亡率统计"
        ]
        
        return TreatmentPlan(
            primary_treatment=primary_treatment,
            alternative_treatments=alternative_treatments,
            prevention_measures=prevention_measures,
            monitoring_indicators=monitoring_indicators,
            expected_timeline="7-14天见效，完全康复需2-4周"
        )
    
    except Exception as e:
        logger.error(f"生成治疗方案失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成治疗方案失败: {str(e)}")

@router.get("/symptoms")
async def get_common_symptoms():
    """
    获取常见症状列表
    """
    common_symptoms = [
        {"name": "体表白斑", "category": "外观异常", "severity": "中等"},
        {"name": "游泳异常", "category": "行为异常", "severity": "高"},
        {"name": "摄食减少", "category": "行为异常", "severity": "中等"},
        {"name": "鳃丝腐烂", "category": "器官病变", "severity": "高"},
        {"name": "呼吸困难", "category": "生理异常", "severity": "高"},
        {"name": "腹部肿胀", "category": "外观异常", "severity": "中等"},
        {"name": "肛门红肿", "category": "器官病变", "severity": "中等"},
        {"name": "鳞片脱落", "category": "外观异常", "severity": "中等"},
        {"name": "体色变化", "category": "外观异常", "severity": "低"},
        {"name": "活动减少", "category": "行为异常", "severity": "中等"}
    ]
    
    return {"symptoms": common_symptoms}

@router.get("/animal-types")
async def get_animal_types():
    """
    获取支持的动物类型
    """
    animal_types = [
        {"name": "对虾", "category": "甲壳类", "common_diseases": ["白斑病", "肝胰腺坏死症"]},
        {"name": "螃蟹", "category": "甲壳类", "common_diseases": ["白斑病", "颤抖病"]},
        {"name": "草鱼", "category": "淡水鱼", "common_diseases": ["烂鳃病", "肠炎病", "赤皮病"]},
        {"name": "鲢鱼", "category": "淡水鱼", "common_diseases": ["烂鳃病", "细菌性败血症"]},
        {"name": "鲤鱼", "category": "淡水鱼", "common_diseases": ["肠炎病", "竖鳞病"]},
        {"name": "小龙虾", "category": "甲壳类", "common_diseases": ["白斑病", "黑腮病"]}
    ]
    
    return {"animal_types": animal_types} 