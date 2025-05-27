#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
training.py - 训练管理API

提供OceanGPT模型微调训练的管理接口

作者: 蓝海智询团队
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, UploadFile, File
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path

from ..config.settings import get_settings
from ..utils.logger import get_logger
from ..core.fine_tuning import (
    TrainingManager, 
    FineTuningConfig, 
    DiagnosisExample,
    MarineDiagnosisDataGenerator
)

logger = get_logger(__name__)
router = APIRouter()

# 全局训练管理器
training_manager = None
current_training_task = None

# 请求模型
class TrainingConfigRequest(BaseModel):
    """训练配置请求"""
    base_model_name: str = Field(default="OceanGPT-o-7B-v0.1", description="基础模型名称")
    output_dir: str = Field(default="./saves/fine_tuned_models", description="输出目录")
    
    # LoRA配置
    lora_r: int = Field(default=64, description="LoRA rank")
    lora_alpha: int = Field(default=16, description="LoRA alpha")
    lora_dropout: float = Field(default=0.1, description="LoRA dropout")
    
    # 训练参数
    num_train_epochs: int = Field(default=3, description="训练轮数")
    per_device_train_batch_size: int = Field(default=1, description="训练批次大小")
    gradient_accumulation_steps: int = Field(default=8, description="梯度累积步数")
    learning_rate: float = Field(default=1e-4, description="学习率")
    max_seq_length: int = Field(default=2048, description="最大序列长度")
    
    # 数据配置
    use_generated_data: bool = Field(default=True, description="是否使用生成数据")
    generated_data_count: int = Field(default=1000, description="生成数据数量")

class DataUploadRequest(BaseModel):
    """数据上传请求"""
    data_format: str = Field(description="数据格式: json, csv, excel")
    description: str = Field(default="", description="数据描述")

class TrainingStatusResponse(BaseModel):
    """训练状态响应"""
    is_training: bool
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    training_loss: float = 0.0
    eval_loss: float = 0.0
    estimated_time_remaining: str = ""
    message: str = ""

# 初始化训练管理器
def get_training_manager():
    global training_manager
    if training_manager is None:
        training_manager = TrainingManager()
    return training_manager

@router.get("/status", response_model=TrainingStatusResponse)
async def get_training_status():
    """获取训练状态"""
    global current_training_task
    
    try:
        is_training = current_training_task is not None and not current_training_task.done()
        
        # TODO: 实现更详细的训练状态监控
        status = TrainingStatusResponse(
            is_training=is_training,
            message="训练空闲" if not is_training else "训练进行中"
        )
        
        return status
        
    except Exception as e:
        logger.error(f"获取训练状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取训练状态失败: {e}")

@router.post("/start")
async def start_training(
    config_request: TrainingConfigRequest,
    background_tasks: BackgroundTasks
):
    """开始训练"""
    global current_training_task, training_manager
    
    try:
        # 检查是否已有训练任务在运行
        if current_training_task is not None and not current_training_task.done():
            raise HTTPException(status_code=409, detail="已有训练任务在运行中")
        
        # 创建训练配置
        config = FineTuningConfig(
            base_model_name=config_request.base_model_name,
            output_dir=config_request.output_dir,
            lora_r=config_request.lora_r,
            lora_alpha=config_request.lora_alpha,
            lora_dropout=config_request.lora_dropout,
            num_train_epochs=config_request.num_train_epochs,
            per_device_train_batch_size=config_request.per_device_train_batch_size,
            gradient_accumulation_steps=config_request.gradient_accumulation_steps,
            learning_rate=config_request.learning_rate,
            max_seq_length=config_request.max_seq_length
        )
        
        # 更新训练管理器配置
        training_manager = TrainingManager()
        training_manager.trainer.config = config
        
        # 在后台启动训练任务
        current_training_task = background_tasks.add_task(
            run_training_task,
            config_request.use_generated_data,
            config_request.generated_data_count
        )
        
        logger.info("训练任务已启动")
        
        return {
            "status": "success",
            "message": "训练任务已启动",
            "config": config_request.dict()
        }
        
    except Exception as e:
        logger.error(f"启动训练失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动训练失败: {e}")

async def run_training_task(use_generated_data: bool, generated_count: int):
    """运行训练任务（后台任务）"""
    try:
        logger.info("开始执行训练任务")
        
        manager = get_training_manager()
        success = await manager.quick_start_training(
            use_generated_data=use_generated_data,
            generated_count=generated_count
        )
        
        if success:
            logger.info("训练任务完成")
        else:
            logger.error("训练任务失败")
            
    except Exception as e:
        logger.error(f"训练任务执行出错: {e}")

@router.post("/stop")
async def stop_training():
    """停止训练"""
    global current_training_task
    
    try:
        if current_training_task is None or current_training_task.done():
            raise HTTPException(status_code=404, detail="没有正在运行的训练任务")
        
        # TODO: 实现优雅的训练停止逻辑
        current_training_task.cancel()
        current_training_task = None
        
        logger.info("训练任务已停止")
        
        return {
            "status": "success",
            "message": "训练任务已停止"
        }
        
    except Exception as e:
        logger.error(f"停止训练失败: {e}")
        raise HTTPException(status_code=500, detail=f"停止训练失败: {e}")

@router.get("/models")
async def list_trained_models():
    """列出已训练的模型"""
    try:
        settings = get_settings()
        models_dir = Path("./saves/fine_tuned_models")
        
        if not models_dir.exists():
            return {"models": []}
        
        models = []
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                # 查找训练日志
                log_file = model_dir / "training_log.json"
                model_info = {
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "created_time": datetime.fromtimestamp(model_dir.stat().st_ctime).isoformat(),
                    "size": sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                }
                
                if log_file.exists():
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            log_data = json.load(f)
                            model_info.update({
                                "config": log_data.get("config", {}),
                                "training_loss": log_data.get("train_result", {}).get("training_loss"),
                                "training_time": log_data.get("train_result", {}).get("train_runtime")
                            })
                    except Exception as e:
                        logger.warning(f"读取训练日志失败 {log_file}: {e}")
                
                models.append(model_info)
        
        # 按创建时间排序
        models.sort(key=lambda x: x["created_time"], reverse=True)
        
        return {"models": models}
        
    except Exception as e:
        logger.error(f"列出训练模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"列出训练模型失败: {e}")

@router.delete("/models/{model_name}")
async def delete_trained_model(model_name: str):
    """删除训练模型"""
    try:
        models_dir = Path("./saves/fine_tuned_models")
        model_path = models_dir / model_name
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="模型不存在")
        
        # 删除模型目录
        import shutil
        shutil.rmtree(model_path)
        
        logger.info(f"已删除训练模型: {model_name}")
        
        return {
            "status": "success",
            "message": f"模型 {model_name} 已删除"
        }
        
    except Exception as e:
        logger.error(f"删除训练模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除训练模型失败: {e}")

@router.post("/data/generate")
async def generate_training_data(count: int = 1000):
    """生成训练数据"""
    try:
        generator = MarineDiagnosisDataGenerator()
        examples = generator.generate_training_examples(count)
        
        # 保存生成的数据
        settings = get_settings()
        data_dir = Path(settings.data_dir) / "marine_diseases"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = data_dir / f"generated_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        generator.save_examples_to_file(examples, str(output_file))
        
        return {
            "status": "success",
            "message": f"已生成 {count} 条训练数据",
            "file_path": str(output_file),
            "sample_data": [
                {
                    "animal_type": ex.animal_type,
                    "symptoms": ex.symptoms[:3],  # 只显示前3个症状
                    "diagnosis": ex.diagnosis
                } for ex in examples[:5]  # 只显示前5个示例
            ]
        }
        
    except Exception as e:
        logger.error(f"生成训练数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成训练数据失败: {e}")

@router.post("/data/upload")
async def upload_training_data(
    file: UploadFile = File(...),
    description: str = ""
):
    """上传训练数据"""
    try:
        settings = get_settings()
        upload_dir = Path(settings.data_dir) / "marine_diseases" / "uploaded"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存上传的文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"{timestamp}_{file.filename}"
        file_path = upload_dir / file_name
        
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # 验证文件格式
        file_size = len(content)
        file_extension = file_path.suffix.lower()
        
        supported_formats = ['.json', '.jsonl', '.csv', '.xlsx', '.xls']
        if file_extension not in supported_formats:
            os.remove(file_path)
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件格式: {file_extension}。支持的格式: {supported_formats}"
            )
        
        # 尝试加载和验证数据
        try:
            manager = get_training_manager()
            examples = manager.trainer.load_marine_disease_data(str(upload_dir))
            data_count = len(examples)
        except Exception as e:
            logger.warning(f"数据验证失败: {e}")
            data_count = "未知"
        
        logger.info(f"上传训练数据文件: {file_name}")
        
        return {
            "status": "success",
            "message": "文件上传成功",
            "file_info": {
                "name": file_name,
                "size": file_size,
                "format": file_extension,
                "data_count": data_count,
                "description": description,
                "upload_time": timestamp
            }
        }
        
    except Exception as e:
        logger.error(f"上传训练数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"上传训练数据失败: {e}")

@router.get("/data/list")
async def list_training_data():
    """列出训练数据文件"""
    try:
        settings = get_settings()
        data_dir = Path(settings.data_dir) / "marine_diseases"
        
        if not data_dir.exists():
            return {"files": []}
        
        files = []
        for file_path in data_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.json', '.jsonl', '.csv', '.xlsx', '.xls']:
                file_info = {
                    "name": file_path.name,
                    "path": str(file_path.relative_to(data_dir)),
                    "size": file_path.stat().st_size,
                    "format": file_path.suffix.lower(),
                    "modified_time": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                }
                files.append(file_info)
        
        # 按修改时间排序
        files.sort(key=lambda x: x["modified_time"], reverse=True)
        
        return {"files": files}
        
    except Exception as e:
        logger.error(f"列出训练数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"列出训练数据失败: {e}")

@router.get("/config/default")
async def get_default_config():
    """获取默认训练配置"""
    try:
        default_config = FineTuningConfig()
        
        return {
            "config": {
                "base_model_name": default_config.base_model_name,
                "lora_r": default_config.lora_r,
                "lora_alpha": default_config.lora_alpha,
                "lora_dropout": default_config.lora_dropout,
                "num_train_epochs": default_config.num_train_epochs,
                "per_device_train_batch_size": default_config.per_device_train_batch_size,
                "gradient_accumulation_steps": default_config.gradient_accumulation_steps,
                "learning_rate": default_config.learning_rate,
                "max_seq_length": default_config.max_seq_length
            },
            "description": {
                "base_model_name": "基础模型名称",
                "lora_r": "LoRA rank，控制适配器参数量",
                "lora_alpha": "LoRA缩放参数",
                "lora_dropout": "LoRA dropout比率",
                "num_train_epochs": "训练轮数",
                "per_device_train_batch_size": "每设备批次大小",
                "gradient_accumulation_steps": "梯度累积步数",
                "learning_rate": "学习率",
                "max_seq_length": "最大序列长度"
            }
        }
        
    except Exception as e:
        logger.error(f"获取默认配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取默认配置失败: {e}")

@router.get("/logs/{model_name}")
async def get_training_logs(model_name: str):
    """获取训练日志"""
    try:
        models_dir = Path("./saves/fine_tuned_models")
        model_path = models_dir / model_name
        log_file = model_path / "training_log.json"
        
        if not log_file.exists():
            raise HTTPException(status_code=404, detail="训练日志不存在")
        
        with open(log_file, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        
        return log_data
        
    except Exception as e:
        logger.error(f"获取训练日志失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取训练日志失败: {e}")

@router.get("/hardware/check")
async def check_hardware():
    """检查硬件状态"""
    try:
        import torch
        import psutil
        
        # GPU信息
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info.append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                    "memory_allocated": torch.cuda.memory_allocated(i),
                    "memory_reserved": torch.cuda.memory_reserved(i)
                })
        
        # CPU和内存信息
        memory = psutil.virtual_memory()
        
        hardware_status = {
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_info": gpu_info,
            "cpu_count": psutil.cpu_count(),
            "memory_total": memory.total,
            "memory_available": memory.available,
            "memory_percent": memory.percent,
            "recommended_batch_size": 1 if torch.cuda.is_available() else 1,
            "supports_fp16": torch.cuda.is_available()
        }
        
        return hardware_status
        
    except Exception as e:
        logger.error(f"检查硬件状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"检查硬件状态失败: {e}") 