"""
config_api.py - 配置接口

提供配置信息的获取和设置接口。

作者: 团队成员
"""

import os
import json
import time
import shutil
import subprocess
import asyncio
import sys # 导入 sys 模块
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, Body, BackgroundTasks
from pydantic import BaseModel, validator, root_validator

from src.config.settings import Settings, get_settings
from src.utils.logger import get_logger

# 配置日志
logger = get_logger(__name__)

# 创建路由器
router = APIRouter(prefix="/config", tags=["配置接口"])

# 配置项说明
CONFIG_ITEMS = {
    "embed_model": {
        "des": "向量嵌入模型",
        "choices": ["BAAI/bge-large-zh-v1.5", "BAAI/bge-small-zh", "text2vec-base-chinese"]
    },
    "reranker": {
        "des": "重排序模型",
        "choices": ["BAAI/bge-reranker-large", "BAAI/bge-reranker-base", "none"]
    },
    "enable_knowledge_base": {
        "des": "启用知识库",
        "default": True
    },
    "enable_knowledge_graph": {
        "des": "启用知识图谱",
        "default": False
    },
    "enable_web_search": {
        "des": "启用网络搜索",
        "default": False
    },
    "enable_reranker": {
        "des": "启用重排序",
        "default": True
    },
    "use_rewrite_query": {
        "des": "使用查询重写",
        "choices": ["auto", "always", "never"],
        "default": "auto"
    }
}

# 模型名称和URL
MODEL_NAMES = {
    "openai": {
        "name": "OpenAI API",
        "url": "https://openai.com/",
        "env": ["OPENAI_API_KEY"],
        "models": ["gpt-4", "gpt-4-turbo", "gpt-4-vision", "gpt-3.5-turbo"]
    },
    "qianfan": {
        "name": "千帆大模型API",
        "url": "https://cloud.baidu.com/product/wenxinworkshop",
        "env": ["QIANFAN_AK", "QIANFAN_SK"],
        "models": ["ERNIE-Bot-4", "ERNIE-Bot", "ERNIE-Bot-turbo"]
    },
    "dashscope": {
        "name": "灵积API",
        "url": "https://dashscope.aliyun.com/",
        "env": ["DASHSCOPE_API_KEY"],
        "models": ["qwen-turbo", "qwen-plus", "qwen-max"]
    },
    "zhipu": {
        "name": "智谱AI",
        "url": "https://open.bigmodel.cn/",
        "env": ["ZHIPU_API_KEY"],
        "models": ["glm-4", "glm-3-turbo"]
    }
}

# 默认配置
DEFAULT_CONFIG = {
    "_config_items": CONFIG_ITEMS,
    "model_names": MODEL_NAMES,
    "model_provider_status": {
        "openai": False,
        "qianfan": False,
        "dashscope": False,
        "zhipu": False,
        "custom": True
    },
    "model_provider": "custom",
    "model_name": "internlm/internlm2-chat-7b",
    "embed_model": "BAAI/bge-large-zh-v1.5",
    "reranker": "BAAI/bge-reranker-large",
    "enable_knowledge_base": True,
    "enable_knowledge_graph": False,
    "enable_web_search": False,
    "enable_reranker": True,
    "use_rewrite_query": "auto",
    "model_local_paths": {
        "BAAI/bge-large-zh-v1.5": "",
        "internlm/internlm2-chat-7b": ""
    },
    "custom_models": [
        {
            "custom_id": "internlm2-chat-7b",
            "name": "书生浦语 InternLM2-Chat-7B",
            "api_base": "http://localhost:8000",
            "api_key": ""
        }
    ]
}

# 自定义模型验证
class CustomModel(BaseModel):
    custom_id: str
    name: str
    api_base: str
    api_key: Optional[str] = ""
    
    @validator('name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("模型名称不能为空")
        return v
        
    @validator('api_base')
    def validate_api_base(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("API地址不能为空")
        return v

# 配置更新请求验证
class ConfigUpdateRequest(BaseModel):
    key: str
    value: Any
    
    @validator('key')
    def validate_key(cls, v):
        allowed_keys = list(DEFAULT_CONFIG.keys()) + ["custom_models"]
        if v not in allowed_keys and not v.startswith('_'):
            raise ValueError(f"无效的配置键: {v}")
        return v
    
    @root_validator
    def validate_config(cls, values):
        key = values.get('key')
        value = values.get('value')
        
        # 配置项特定验证
        if key == 'enable_knowledge_graph' and value is True:
            if not DEFAULT_CONFIG.get('enable_knowledge_base', False):
                raise ValueError("启用知识图谱前必须先启用知识库")
        
        elif key == 'custom_models' and isinstance(value, list):
            # 验证自定义模型列表
            try:
                for model in value:
                    CustomModel(**model)
                if len(value) == 0:
                    raise ValueError("自定义模型列表不能为空")
            except Exception as e:
                raise ValueError(f"自定义模型格式无效: {str(e)}")
        
        return values

# 读取配置文件
def load_config_file(file_path: str) -> Dict[str, Any]:
    """从文件加载配置，失败则返回空字典"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"读取配置文件失败: {e}")
        # 如果配置文件损坏，尝试备份并创建新配置
        try:
            if os.path.exists(file_path):
                backup_path = f"{file_path}.bak.{int(time.time())}"
                os.rename(file_path, backup_path)
                logger.warning(f"已创建配置文件备份: {backup_path}")
        except Exception as be:
            logger.error(f"创建配置备份失败: {be}")
    return {}

# 保存配置到文件
def save_config_file(config: Dict[str, Any], file_path: str) -> bool:
    """保存配置到文件，成功返回True，失败返回False"""
    try:
        # 移除一些不需要保存的计算属性
        save_config = {k: v for k, v in config.items() 
                      if not k.startswith("_") 
                      and k != "model_names" 
                      and k != "model_provider_status"}
        
        # 确保目标目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 创建临时文件，确保写入成功后再替换
        temp_file = f"{file_path}.tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(save_config, f, ensure_ascii=False, indent=2)
        
        # 成功写入后替换原文件
        if os.path.exists(temp_file):
            shutil.move(temp_file, file_path)
            return True
    except Exception as e:
        logger.error(f"保存配置文件失败: {e}")
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
    return False

# 重新启动服务
async def restart_service_task():
    """重启服务的异步任务"""
    logger.info("开始执行重启服务任务")
    
    try:
        # 等待一段时间确保当前请求能够完成
        await asyncio.sleep(1)
        
        logger.info("尝试通过 sys.exit() 触发服务重启")
        # 通过退出当前进程来尝试触发Uvicorn的热重载
        # Uvicorn 在 reload 模式下通常会监控主进程，并在其退出后重启
        sys.exit(0) # 使用状态码0表示正常退出
        
    except SystemExit:
        logger.info("sys.exit() 调用成功，等待Uvicorn重载机制响应")
        # SystemExit 是预期的，不需要额外处理
        pass
    except Exception as e:
        logger.error(f"尝试重启服务时发生错误: {e}")
        # 即使这里出错，也可能 sys.exit() 已经被调用
        # 或者Uvicorn的重载机制因其他原因失败
        # 通常不应该在这里重新 raise，避免干扰退出流程

# API路由
@router.get("")
async def get_config(settings: Settings = Depends(get_settings)):
    """获取配置信息"""
    try:
        # 返回完整配置
        config = DEFAULT_CONFIG.copy()
        
        # 更新模型提供者状态
        config["model_provider_status"]["openai"] = os.environ.get("OPENAI_API_KEY") is not None
        config["model_provider_status"]["qianfan"] = (
            os.environ.get("QIANFAN_AK") is not None and 
            os.environ.get("QIANFAN_SK") is not None
        )
        config["model_provider_status"]["dashscope"] = os.environ.get("DASHSCOPE_API_KEY") is not None
        config["model_provider_status"]["zhipu"] = os.environ.get("ZHIPU_API_KEY") is not None
        
        # 从配置文件加载自定义设置
        config_file = os.path.join(settings.base_dir, "config.json")
        user_config = load_config_file(config_file)
        
        # 更新默认配置
        if user_config:
            config.update(user_config)
        
        # 确保必要的配置项存在
        for key, value in DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = value
        
        # 确保custom_models不为空
        if not config.get("custom_models"):
            config["custom_models"] = DEFAULT_CONFIG["custom_models"]
        
        return config
    except Exception as e:
        logger.error(f"获取配置信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取配置信息失败: {e}")

@router.post("")
async def update_config(request: ConfigUpdateRequest, settings: Settings = Depends(get_settings)):
    """更新配置信息"""
    try:
        # 获取当前配置
        config = await get_config(settings)
        
        # 更新配置
        config[request.key] = request.value
        
        # 保存到配置文件
        config_file = os.path.join(settings.base_dir, "config.json")
        
        if not save_config_file(config, config_file):
            logger.error("保存配置文件失败")
            raise HTTPException(status_code=500, detail="保存配置文件失败")
        
        logger.info(f"配置更新成功: {request.key}")
        return config
    except ValueError as e:
        # 配置验证错误
        logger.error(f"配置验证失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"更新配置信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新配置信息失败: {e}")

@router.post("/restart")
async def restart_service(background_tasks: BackgroundTasks):
    """重启服务"""
    try:
        logger.info("重启服务请求已接收")
        
        # 在后台执行重启操作
        background_tasks.add_task(restart_service_task)
        
        return {"status": "success", "message": "服务重启中"}
    except Exception as e:
        logger.error(f"重启服务失败: {e}")
        raise HTTPException(status_code=500, detail=f"重启服务失败: {e}") 