#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
admin.py - 管理接口路由模块

该模块提供了系统管理相关的API路由接口。

作者: 团队成员

# 更多管理功能可在此扩展
"""

import os
import json
import psutil
import platform
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File, Form
from pydantic import BaseModel, Field

from src.config.settings import Settings, get_settings
from src.utils.logger import setup_logger

# 配置日志
logger = setup_logger("admin_router")

# 创建路由器
router = APIRouter()

# 数据模型
class SystemInfo(BaseModel):
    """系统信息模型"""
    hostname: str = Field(..., description="主机名")
    platform: str = Field(..., description="操作系统平台")
    cpu_usage: float = Field(..., description="CPU使用率(%)")
    memory_used: float = Field(..., description="内存使用量(GB)")
    memory_total: float = Field(..., description="总内存(GB)")
    memory_percent: float = Field(..., description="内存使用率(%)")
    disk_used: float = Field(..., description="磁盘使用量(GB)")
    disk_total: float = Field(..., description="总磁盘空间(GB)")
    disk_percent: float = Field(..., description="磁盘使用率(%)")
    uptime: int = Field(..., description="系统运行时间(秒)")

class AppStatus(BaseModel):
    """应用状态模型"""
    status: str = Field(..., description="运行状态")
    version: str = Field(..., description="应用版本")
    start_time: str = Field(..., description="启动时间")
    documents_count: int = Field(..., description="文档数量")
    entities_count: int = Field(..., description="实体数量")

class LogEntry(BaseModel):
    """日志条目模型"""
    timestamp: str = Field(..., description="时间戳")
    level: str = Field(..., description="日志级别")
    message: str = Field(..., description="日志消息")
    logger: str = Field(..., description="日志器名称")

# API路由
@router.get("/system", response_model=SystemInfo)
async def get_system_info():
    """获取系统信息"""
    try:
        # 获取CPU使用率
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # 获取内存使用情况
        memory = psutil.virtual_memory()
        memory_used = memory.used / (1024 ** 3)  # GB
        memory_total = memory.total / (1024 ** 3)  # GB
        memory_percent = memory.percent
        
        # 获取磁盘使用情况
        disk = psutil.disk_usage('/')
        disk_used = disk.used / (1024 ** 3)  # GB
        disk_total = disk.total / (1024 ** 3)  # GB
        disk_percent = disk.percent
        
        # 获取系统启动时间
        boot_time = psutil.boot_time()
        uptime = int(datetime.now().timestamp() - boot_time)
        
        return SystemInfo(
            hostname=platform.node(),
            platform=f"{platform.system()} {platform.release()}",
            cpu_usage=cpu_usage,
            memory_used=round(memory_used, 2),
            memory_total=round(memory_total, 2),
            memory_percent=memory_percent,
            disk_used=round(disk_used, 2),
            disk_total=round(disk_total, 2),
            disk_percent=disk_percent,
            uptime=uptime
        )
    except Exception as e:
        logger.error(f"获取系统信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统信息失败: {e}")

@router.get("/app/status", response_model=AppStatus)
async def get_app_status(settings: Settings = Depends(get_settings)):
    """获取应用状态"""
    try:
        # 获取文档数量
        doc_count = 0
        entities_count = 0
        
        # 读取文档映射文件计算文档数
        doc_mapping_path = os.path.join(settings.processed_dir, "doc_mapping.json")
        if os.path.exists(doc_mapping_path):
            try:
                with open(doc_mapping_path, 'r', encoding='utf-8') as f:
                    doc_mapping = json.load(f)
                    doc_count = len(doc_mapping)
                    
                    # 计算实体数量
                    for doc_id, doc_info in doc_mapping.items():
                        entities_count += len(doc_info.get("entities", []))
            except Exception as e:
                logger.warning(f"读取文档映射失败: {e}")
        
        # 应用启动时间（简化处理，使用进程创建时间）
        process = psutil.Process(os.getpid())
        start_time = datetime.fromtimestamp(process.create_time()).isoformat()
        
        return AppStatus(
            status="running",
            version="1.0.0",
            start_time=start_time,
            documents_count=doc_count,
            entities_count=entities_count
        )
    except Exception as e:
        logger.error(f"获取应用状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取应用状态失败: {e}")

@router.get("/logs", response_model=List[LogEntry])
async def get_logs(
    level: Optional[str] = Query(None, description="日志级别过滤"),
    limit: int = Query(100, ge=1, le=1000, description="返回的日志条数"),
    settings: Settings = Depends(get_settings)
):
    """获取应用日志"""
    try:
        log_entries = []
        
        # 获取最新的日志文件
        log_dir = settings.log_dir
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        
        if not log_files:
            return []
            
        # 按文件修改时间排序，获取最新的日志文件
        log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
        latest_log = os.path.join(log_dir, log_files[0])
        
        # 读取日志文件
        with open(latest_log, 'r', encoding='utf-8') as f:
            # 从文件末尾开始读取
            lines = f.readlines()[-limit:]
            
            for line in lines:
                parts = line.split(' - ', 3)
                if len(parts) >= 4:
                    timestamp, logger_name, log_level, message = parts
                    
                    # 如果指定了级别过滤，则只返回对应级别的日志
                    if level and log_level.strip().lower() != level.lower():
                        continue
                        
                    log_entries.append(LogEntry(
                        timestamp=timestamp.strip(),
                        level=log_level.strip(),
                        message=message.strip(),
                        logger=logger_name.strip()
                    ))
        
        return log_entries
    except Exception as e:
        logger.error(f"获取日志失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取日志失败: {e}")

@router.post("/data/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_type: str = Form(...),
    tags: str = Form(""),
    settings: Settings = Depends(get_settings)
):
    """上传文档到系统"""
    try:
        # 确保上传目录存在
        upload_dir = os.path.join(settings.fish_docs_dir, document_type)
        os.makedirs(upload_dir, exist_ok=True)
        
        # 保存文件
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # 解析标签
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        logger.info(f"文档上传成功: {file.filename}, 类型: {document_type}, 标签: {tag_list}")
        
        # 返回上传结果
        return {
            "success": True,
            "filename": file.filename,
            "path": file_path,
            "type": document_type,
            "tags": tag_list,
            "size_bytes": os.path.getsize(file_path)
        }
    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档上传失败: {e}")

@router.post("/rebuild-index")
async def rebuild_index(settings: Settings = Depends(get_settings)):
    """重建索引"""
    try:
        # 这里实际应该调用索引重建逻辑
        # 由于是示例，我们只记录日志并返回成功
        logger.info("管理员触发索引重建")
        
        return {
            "success": True,
            "message": "索引重建任务已启动，请稍后检查状态",
            "task_id": "rebuild-task-001"
        }
    except Exception as e:
        logger.error(f"重建索引失败: {e}")
        raise HTTPException(status_code=500, detail=f"重建索引失败: {e}")

@router.get("/tasks")
async def get_task_status():
    """获取后台任务状态"""
    # 示例任务列表
    tasks = [
        {
            "id": "rebuild-task-001",
            "name": "索引重建",
            "status": "完成",
            "progress": 100,
            "start_time": "2023-05-15T08:30:00",
            "end_time": "2023-05-15T08:35:00",
            "details": "处理了152个文档，生成了1240个文本块"
        },
        {
            "id": "import-task-002",
            "name": "数据导入",
            "status": "进行中",
            "progress": 65,
            "start_time": "2023-05-15T09:15:00",
            "end_time": None,
            "details": "已处理65个文档，还有35个待处理"
        }
    ]
    
    return tasks 