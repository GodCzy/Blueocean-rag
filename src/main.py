#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py - 应用入口

蓝海智询 - 基于大模型RAG知识库与知识图谱技术的水生动物疾病问答平台

作者: 团队成员

 运行前请确认 `config.json` 配置正确。
"""

import uvicorn
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import time
import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 项目内部导入
from src.config.settings import Settings, get_settings
from src.utils.logger import get_logger
from src.core.oceangpt_manager import OceanGPTManager, ModelConfig
from src.routers.rag_router import router as rag_router
from src.routers.diagnosis import router as diagnosis_router
from src.routers.ocean_env import router as ocean_env_router
from src.routers.admin import router as admin_router
from src.api.config_api import router as config_router
from src.routers.chat_router import router as chat_router
from src.routers.data_router import data as data_router
from src.routers.tool_router import tool as tool_router
from src.routers.training import router as training_router
from src.routers.stats import router as stats_router
from src.auth import api_key_auth

# 配置日志
logger = get_logger(__name__)

# 全局变量存储管理器实例
oceangpt_manager = None
app_start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("🌊 启动蓝海智询系统...")
    
    global oceangpt_manager
    try:
        # 初始化OceanGPT管理器
        settings = get_settings()
        model_config = ModelConfig(
            model_name="OceanGPT-o-7B-v0.1",
            device="auto",
            load_in_4bit=True
        )
        oceangpt_manager = OceanGPTManager(model_config)
        
        # 异步加载模型
        logger.info("🤖 正在加载OceanGPT模型...")
        success = await oceangpt_manager.load_model()
        if success:
            logger.info("✅ OceanGPT模型加载成功")
        else:
            logger.warning("⚠️ OceanGPT模型加载失败，将在首次使用时重试")
        
        # 创建必要目录
        settings.create_directories()
        logger.info("📁 系统目录初始化完成")
        
    except Exception as e:
        logger.error(f"❌ 系统启动失败: {e}")
        
    yield
    
    # 关闭时执行
    logger.info("🌊 关闭蓝海智询系统...")

def create_app(settings: Settings = Depends(get_settings)) -> FastAPI:
    """创建FastAPI应用实例
    
    Args:
        settings: 应用配置
        
    Returns:
        FastAPI应用实例
    """
    # 创建FastAPI应用
    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version="1.0.0",
        docs_url=None,  # 自定义文档URL
        redoc_url=None,  # 自定义文档URL
        lifespan=lifespan
    )
    
    # 添加中间件
    
    # GZIP压缩中间件
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # 可信主机中间件（生产环境推荐）
    if not settings.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # 生产环境应该限制具体域名
        )
    
    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 请求处理时间中间件
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # 全局异常处理
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "message": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url)
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"未处理的异常: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": "服务器内部错误",
                "status_code": 500,
                "path": str(request.url)
            }
        )
    
    # 健康检查
    @app.get("/health", tags=["系统"])
    async def health_check():
        """健康检查接口"""
        global oceangpt_manager, app_start_time
        
        uptime = time.time() - app_start_time
        
        return {
            "status": "healthy",
            "uptime_seconds": uptime,
            "oceangpt_loaded": oceangpt_manager and oceangpt_manager.model is not None,
            "version": "1.0.0",
            "timestamp": time.time()
        }
    
    # 系统信息
    @app.get("/info", tags=["系统"])
    async def system_info():
        """系统信息接口"""
        global oceangpt_manager
        
        model_info = {}
        if oceangpt_manager:
            model_info = oceangpt_manager.get_model_info()
        
        return {
            "app_name": settings.app_name,
            "version": "1.0.0",
            "description": settings.app_description,
            "model_info": model_info,
            "debug_mode": settings.debug
        }
    
    # 获取OceanGPT管理器的依赖函数
    def get_oceangpt_manager():
        global oceangpt_manager
        if not oceangpt_manager:
            raise HTTPException(status_code=503, detail="OceanGPT模型尚未加载")
        return oceangpt_manager
    
    # 将依赖函数添加到app状态
    app.state.get_oceangpt_manager = get_oceangpt_manager
    
    # 注册路由
    app.include_router(
        rag_router,
        prefix=settings.api_prefix,
        tags=["知识检索服务"]
    )
    
    app.include_router(
        diagnosis_router,
        prefix=settings.api_prefix,
        tags=["疾病诊断服务"]
    )
    
    app.include_router(
        ocean_env_router,
        prefix=f"{settings.api_prefix}/ocean",
        tags=["海洋环境服务"]
    )
    
    app.include_router(
        admin_router,
        prefix=settings.admin_prefix,
        tags=["管理接口"]
    )
    
    app.include_router(
        config_router,
        prefix=f"{settings.api_prefix}",
        tags=["配置接口"]
    )
    
    app.include_router(
        chat_router,
        prefix=settings.api_prefix,
        tags=["聊天服务"]
    )

    app.include_router(
        data_router,
        prefix=settings.api_prefix,
        tags=["数据服务"]
    )

    app.include_router(
        tool_router,
        prefix=settings.api_prefix,
        tags=["工具服务"]
    )
    
    app.include_router(
        training_router,
        prefix=f"{settings.api_prefix}/training",
        tags=["训练服务"]
    )

    app.include_router(
        stats_router,
        prefix=f"{settings.api_prefix}",
        tags=["统计"]
    )
    
    # 尝试创建静态目录
    os.makedirs("src/static", exist_ok=True)
    os.makedirs("src/static/swagger", exist_ok=True)
    os.makedirs("src/static/redoc", exist_ok=True)
    
    # 挂载静态文件
    app.mount("/static", StaticFiles(directory="src/static"), name="static")
    
    # 自定义API文档
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - API文档",
            swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        )
    
    @app.get("/redoc", include_in_schema=False)
    async def custom_redoc_html():
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - API文档",
            redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2.1.0/bundles/redoc.standalone.js",
        )
    
    # 自定义OpenAPI信息
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        
        # 自定义OpenAPI信息
        openapi_schema["info"]["x-logo"] = {
            "url": "/static/logo.png"
        }
        
        openapi_schema["info"]["contact"] = {
            "name": "蓝海智询团队",
            "email": "support@blueocean.ai"
        }
        
        openapi_schema["info"]["license"] = {
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    return app

# 创建应用实例
app = create_app(get_settings())

if __name__ == "__main__":
    # 获取配置
    settings = get_settings()
    
    # 启动服务
    logger.info(f"🚀 启动 {settings.app_name} 服务")
    logger.info(f"📚 API文档: http://0.0.0.0:8000/docs")
    logger.info(f"🔍 Redoc文档: http://0.0.0.0:8000/redoc")
    logger.info(f"❤️ 健康检查: http://0.0.0.0:8000/health")
    
    uvicorn.run(
        "src.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=settings.debug,
        workers=1,
        access_log=True
    )

