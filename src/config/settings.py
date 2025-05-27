#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
settings.py - 应用配置

该模块定义了应用运行所需的各项配置和设置。

作者: 成员C (后端工程师)
"""

from functools import lru_cache
from typing import Dict, List, Any, Optional
from pathlib import Path
import os

from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """应用配置类"""
    
    # 基础配置
    app_name: str = "蓝海智询"
    app_description: str = "基于大模型RAG知识库与知识图谱技术的水生动物疾病问答平台"
    debug: bool = Field(default=False, env="DEBUG")
    
    # 基础目录设置
    base_dir: str = Field(default=str(Path(__file__).parent.parent.parent.absolute()))
    
    # 其他目录路径
    data_dir: str = Field(default="")
    fish_docs_dir: str = Field(default="")
    processed_dir: str = Field(default="")
    ocean_data_dir: str = Field(default="")
    tags_file: str = Field(default="")
    log_dir: str = Field(default="")
    
    # API 相关配置
    allowed_origins: List[str] = Field(default=["*"])
    api_prefix: str = "/api"
    admin_prefix: str = "/admin"
    
    # 模型配置
    llm_model_name: str = Field(default="internlm/internlm2-chat-7b")
    embedding_model_name: str = Field(default="BAAI/bge-large-zh-v1.5")
    tensor_parallel_size: int = Field(default=1)
    
    # 向量索引配置
    vector_dim: int = Field(default=1024)
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=128)
    
    # 知识图谱配置
    graph_update_interval: int = Field(default=3600)
    
    # 缓存配置
    cache_ttl: int = Field(default=3600)
    
    # 日志配置
    log_level: str = Field(default="INFO")
    
    class Config:
        """配置类设置"""
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def __init__(self, **data):
        super().__init__(**data)
        # 初始化时设置依赖于 base_dir 的其他路径
        if not self.data_dir:
            self.data_dir = os.path.join(self.base_dir, "datasets")
        if not self.fish_docs_dir:
            self.fish_docs_dir = os.path.join(self.data_dir, "fish_docs")
        if not self.processed_dir:
            self.processed_dir = os.path.join(self.data_dir, "processed")
        if not self.ocean_data_dir:
            self.ocean_data_dir = os.path.join(self.data_dir, "ocean_data")
        if not self.tags_file:
            self.tags_file = os.path.join(self.data_dir, "tags.json")
        if not self.log_dir:
            self.log_dir = os.path.join(self.base_dir, "logs")
    
    def create_directories(self):
        """创建必要的目录"""
        directories = [
            self.data_dir,
            self.fish_docs_dir,
            self.processed_dir,
            self.ocean_data_dir,
            self.log_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

@lru_cache()
def get_settings() -> Settings:
    """获取应用配置实例（带缓存）
    
    Returns:
        Settings: 应用配置实例
    """
    settings = Settings()
    settings.create_directories()
    return settings 