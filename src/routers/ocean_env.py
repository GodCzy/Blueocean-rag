#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ocean_env.py - 海洋环境数据路由模块

该模块提供了与海洋环境数据相关的API路由接口。

作者: 团队成员

# 本模块仅为示例，可在此基础上扩展业务逻辑。
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from src.config.settings import Settings, get_settings
from src.utils.logger import setup_logger

# 配置日志
logger = setup_logger("ocean_env_router")

# 创建路由器
router = APIRouter()

# 数据模型定义
class EnvironmentData(BaseModel):
    """环境数据模型"""
    timestamp: str = Field(..., description="时间戳")
    value: float = Field(..., description="数值")
    unit: str = Field(..., description="单位")

class RegionData(BaseModel):
    """区域数据模型"""
    region: str = Field(..., description="区域名称")
    lat: float = Field(..., description="纬度")
    lon: float = Field(..., description="经度")
    water_type: str = Field(..., description="水体类型(marine/brackish/freshwater)")
    data: Dict[str, List[EnvironmentData]] = Field(..., description="环境数据")

class StatData(BaseModel):
    """统计数据模型"""
    min: float = Field(..., description="最小值")
    max: float = Field(..., description="最大值")
    avg: float = Field(..., description="平均值")
    unit: str = Field(..., description="单位")

# 辅助函数
def load_ocean_data(settings: Settings = Depends(get_settings)) -> Dict:
    """
    加载海洋数据文件
    
    Args:
        settings: 应用配置
        
    Returns:
        海洋环境数据字典
    """
    try:
        data_file = os.path.join(settings.ocean_data_dir, "ocean_data.json")
        if not os.path.exists(data_file):
            # 如果数据文件不存在，返回空数据结构
            logger.warning(f"海洋环境数据文件不存在: {data_file}")
            return {
                "metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "regions": []
                },
                "data": {}
            }
            
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        return data
    except Exception as e:
        logger.error(f"加载海洋数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"加载海洋数据失败: {e}")

# API路由
@router.get("/regions", summary="获取可用区域列表")
async def get_regions(settings: Settings = Depends(get_settings)):
    """
    获取所有可用的监测区域列表
    
    Returns:
        区域列表，包含名称、位置和水体类型
    """
    try:
        data = load_ocean_data(settings)
        regions = data.get("metadata", {}).get("regions", [])
        return regions
    except Exception as e:
        logger.error(f"获取区域列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取区域列表失败: {e}")

@router.get("/data/{region}", response_model=RegionData, summary="获取区域环境数据")
async def get_region_data(
    region: str,
    parameter: Optional[str] = Query(None, description="参数名称，例如water_temperature"),
    days: int = Query(7, ge=1, le=90, description="查询天数"),
    settings: Settings = Depends(get_settings)
):
    """
    获取指定区域的环境数据
    
    Args:
        region: 区域名称
        parameter: 可选的参数名称，如果不提供则返回所有参数
        days: 查询天数，默认为7天
        
    Returns:
        区域环境数据
    """
    try:
        data = load_ocean_data(settings)
        region_data = data.get("data", {}).get(region)
        
        if not region_data:
            raise HTTPException(status_code=404, detail=f"区域不存在: {region}")
        
        # 获取区域元数据
        region_meta = None
        for r in data.get("metadata", {}).get("regions", []):
            if r.get("name") == region:
                region_meta = r
                break
                
        if not region_meta:
            raise HTTPException(status_code=404, detail=f"区域元数据不存在: {region}")
        
        # 过滤参数
        if parameter:
            if parameter not in region_data:
                raise HTTPException(status_code=404, detail=f"参数不存在: {parameter}")
            filtered_data = {parameter: region_data[parameter]}
        else:
            filtered_data = region_data
        
        # 过滤日期 - 只返回指定天数的数据
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        for param, values in filtered_data.items():
            filtered_data[param] = [
                item for item in values
                if item.get("timestamp", "") >= cutoff_date
            ]
        
        # 构建响应
        return RegionData(
            region=region,
            lat=region_meta.get("lat", 0),
            lon=region_meta.get("lon", 0),
            water_type=region_meta.get("water_type", "unknown"),
            data=filtered_data
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取区域数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取区域数据失败: {e}")

@router.get("/stats/{region}/{parameter}", response_model=StatData, summary="获取参数统计数据")
async def get_parameter_stats(
    region: str,
    parameter: str,
    days: int = Query(30, ge=1, le=365, description="统计天数"),
    settings: Settings = Depends(get_settings)
):
    """
    获取指定区域和参数的统计数据
    
    Args:
        region: 区域名称
        parameter: 参数名称
        days: 统计天数，默认为30天
        
    Returns:
        统计数据，包括最小值、最大值、平均值
    """
    try:
        data = load_ocean_data(settings)
        region_data = data.get("data", {}).get(region)
        
        if not region_data:
            raise HTTPException(status_code=404, detail=f"区域不存在: {region}")
            
        param_data = region_data.get(parameter)
        
        if not param_data:
            raise HTTPException(status_code=404, detail=f"参数不存在: {parameter}")
        
        # 过滤日期
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        filtered_data = [
            item for item in param_data
            if item.get("timestamp", "") >= cutoff_date
        ]
        
        if not filtered_data:
            raise HTTPException(status_code=404, detail=f"指定时间范围内没有数据")
        
        # 计算统计值
        values = [item.get("value", 0) for item in filtered_data]
        unit = filtered_data[0].get("unit", "")
        
        return StatData(
            min=min(values),
            max=max(values),
            avg=sum(values) / len(values),
            unit=unit
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取参数统计数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取参数统计数据失败: {e}")

@router.get("/parameters", summary="获取可用参数列表")
async def get_parameters():
    """获取所有可用的环境参数列表"""
    parameters = [
        {
            "id": "water_temperature",
            "name": "水温",
            "unit": "°C",
            "description": "水体温度",
            "normal_range": {"min": 15, "max": 30}
        },
        {
            "id": "dissolved_oxygen",
            "name": "溶解氧",
            "unit": "mg/L",
            "description": "水中溶解的氧气量",
            "normal_range": {"min": 5, "max": 10}
        },
        {
            "id": "ph",
            "name": "pH值",
            "unit": "",
            "description": "水体酸碱度",
            "normal_range": {"min": 6.5, "max": 8.5}
        },
        {
            "id": "salinity",
            "name": "盐度",
            "unit": "‰",
            "description": "水中溶解盐类的含量",
            "normal_range": {"min": 0, "max": 35}
        },
        {
            "id": "ammonia",
            "name": "氨氮",
            "unit": "mg/L",
            "description": "水中氨和铵离子的总量",
            "normal_range": {"min": 0, "max": 0.5}
        },
        {
            "id": "nitrate",
            "name": "硝酸盐",
            "unit": "mg/L",
            "description": "水中硝酸盐含量",
            "normal_range": {"min": 0, "max": 100}
        },
        {
            "id": "turbidity",
            "name": "浊度",
            "unit": "NTU",
            "description": "水体浑浊程度",
            "normal_range": {"min": 0, "max": 30}
        }
    ]
    
    return parameters

@router.get("/optimal-conditions/{species}", summary="获取最适生长条件")
async def get_optimal_conditions(species: str):
    """
    获取指定水生动物的最适生长条件
    
    Args:
        species: 水生动物种类，如"grass_carp"(草鱼)
        
    Returns:
        最适生长条件
    """
    # 水生动物最适条件数据库
    optimal_conditions = {
        "grass_carp": {  # 草鱼
            "water_temperature": {"min": 20, "max": 30, "optimal": 25, "unit": "°C"},
            "dissolved_oxygen": {"min": 4, "max": 10, "optimal": 7, "unit": "mg/L"},
            "ph": {"min": 6.5, "max": 8.5, "optimal": 7.5, "unit": ""},
            "ammonia": {"max": 0.5, "unit": "mg/L"},
        },
        "common_carp": {  # 鲤鱼
            "water_temperature": {"min": 18, "max": 28, "optimal": 23, "unit": "°C"},
            "dissolved_oxygen": {"min": 5, "max": 10, "optimal": 7, "unit": "mg/L"},
            "ph": {"min": 6.8, "max": 8.0, "optimal": 7.5, "unit": ""},
            "ammonia": {"max": 0.4, "unit": "mg/L"},
        },
        "white_shrimp": {  # 南美白对虾
            "water_temperature": {"min": 24, "max": 32, "optimal": 28, "unit": "°C"},
            "dissolved_oxygen": {"min": 5, "max": 10, "optimal": 7, "unit": "mg/L"},
            "ph": {"min": 7.0, "max": 8.5, "optimal": 8.0, "unit": ""},
            "salinity": {"min": 10, "max": 35, "optimal": 20, "unit": "‰"},
            "ammonia": {"max": 0.3, "unit": "mg/L"},
        }
    }
    
    # 转换常见名称到代码
    species_map = {
        "草鱼": "grass_carp",
        "鲤鱼": "common_carp",
        "南美白对虾": "white_shrimp"
    }
    
    species_code = species_map.get(species, species)
    
    if species_code not in optimal_conditions:
        raise HTTPException(status_code=404, detail=f"未找到物种信息: {species}")
        
    return {
        "species": species,
        "species_code": species_code,
        "conditions": optimal_conditions[species_code]
    }

@router.get("/alerts/{region}", summary="获取区域预警信息")
async def get_region_alerts(
    region: str,
    days: int = Query(7, ge=1, le=30, description="查询天数"),
    settings: Settings = Depends(get_settings)
):
    """
    获取指定区域的环境预警信息
    
    Args:
        region: 区域名称
        days: 查询天数，默认为7天
        
    Returns:
        环境预警信息列表
    """
    try:
        data = load_ocean_data(settings)
        region_data = data.get("data", {}).get(region)
        
        if not region_data:
            raise HTTPException(status_code=404, detail=f"区域不存在: {region}")
        
        # 定义安全阈值
        thresholds = {
            "water_temperature": {"min": 15, "max": 30},
            "dissolved_oxygen": {"min": 5, "max": None},
            "ph": {"min": 6.5, "max": 8.5},
            "ammonia": {"min": None, "max": 0.5},
            "nitrate": {"min": None, "max": 100},
            "turbidity": {"min": None, "max": 30},
        }
        
        # 生成预警信息
        alerts = []
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        for param, values in region_data.items():
            if param not in thresholds:
                continue
                
            param_threshold = thresholds[param]
            filtered_data = [
                item for item in values
                if item.get("timestamp", "") >= cutoff_date
            ]
            
            for item in filtered_data:
                value = item.get("value")
                timestamp = item.get("timestamp")
                
                # 检查是否超出阈值
                if param_threshold["min"] is not None and value < param_threshold["min"]:
                    alerts.append({
                        "parameter": param,
                        "parameter_name": get_parameter_name(param),
                        "value": value,
                        "unit": item.get("unit", ""),
                        "timestamp": timestamp,
                        "type": "低于正常值",
                        "threshold": param_threshold["min"],
                        "level": "warning"
                    })
                elif param_threshold["max"] is not None and value > param_threshold["max"]:
                    alerts.append({
                        "parameter": param,
                        "parameter_name": get_parameter_name(param),
                        "value": value,
                        "unit": item.get("unit", ""),
                        "timestamp": timestamp,
                        "type": "高于正常值",
                        "threshold": param_threshold["max"],
                        "level": "danger"
                    })
        
        # 按时间排序，最新的在前
        alerts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return alerts
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取区域预警信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取区域预警信息失败: {e}")

def get_parameter_name(param_id: str) -> str:
    """获取参数的中文名称"""
    param_names = {
        "water_temperature": "水温",
        "dissolved_oxygen": "溶解氧",
        "ph": "pH值",
        "salinity": "盐度",
        "ammonia": "氨氮",
        "nitrate": "硝酸盐",
        "turbidity": "浊度"
    }
    return param_names.get(param_id, param_id) 