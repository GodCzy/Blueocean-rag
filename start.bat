@echo off
chcp 65001 >nul
title 蓝海智询 - 启动脚本
rem 启动前请按照 DEPLOYMENT_GUIDE.md 完成环境配置

echo.
echo ===================================================
echo                 🌊 蓝海智询
echo     基于RAG知识库与知识图谱的水生动物疾病问答平台
echo ===================================================
echo.

echo [INFO] 正在启动蓝海智询系统...
echo.

:: 检查Python是否存在
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python未安装或未添加到PATH！
    echo [HELP] 请安装Python 3.8+ 并添加到系统PATH
    pause
    exit /b 1
)

:: 检查虚拟环境
if not exist "venv\" (
    echo [INFO] 创建Python虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] 创建虚拟环境失败！
        pause
        exit /b 1
    )
)

:: 激活虚拟环境
echo [INFO] 激活虚拟环境...
call venv\Scripts\activate.bat

:: 检查requirements.txt
if exist "requirements.txt" (
    echo [INFO] 安装Python依赖包...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [WARNING] 依赖包安装可能不完整
    )
) else (
    echo [WARNING] requirements.txt 文件不存在
)

:: 运行快速检查
if exist "quick_start.py" (
    echo [INFO] 运行环境检查...
    python quick_start.py
    echo.
    echo [INFO] 环境检查完成，按任意键继续启动服务...
    pause >nul
    echo.
)

:: 启动服务
echo [INFO] 启动蓝海智询服务...
echo [INFO] API文档: http://localhost:8000/docs
echo [INFO] 健康检查: http://localhost:8000/health
echo [INFO] 按 Ctrl+C 停止服务
echo.

python src/main.py

:: 如果服务意外退出
echo.
echo [INFO] 服务已停止
pause 