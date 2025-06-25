#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run.py - 蓝海智询启动脚本

该脚本用于启动蓝海智询平台的后端服务。

作者：团队成员

 使用方法详见项目 `README.md`。
"""

import os
import sys
import subprocess
import argparse
import webbrowser
import time
import uvicorn
from pathlib import Path

def check_dependencies():
    """检查项目依赖是否已安装"""
    try:
        import fastapi
        import uvicorn
        return True
    except ImportError:
        print("未检测到必要的依赖，正在安装...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return False

def setup_environment():
    """设置环境变量"""
    os.environ["DEBUG"] = "True"
    # 在这里设置其他必要的环境变量

def run_api_server(host="127.0.0.1", port=8000, reload=True):
    """运行API服务器"""
    print(f"启动API服务器：http://{host}:{port}")
    print("API文档：http://{host}:{port}/docs")
    print("退出服务器请按 Ctrl+C")
    
    # 使用uvicorn启动服务
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=reload
    )

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="蓝海智询平台启动脚本")
    parser.add_argument("--host", default="127.0.0.1", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--no-reload", action="store_true", help="禁用热重载")
    parser.add_argument("--open-browser", action="store_true", help="启动后自动打开浏览器")
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 50)
    print("蓝海智询 - 水生动物疾病问答平台")
    print("=" * 50)
    
    # 检查依赖
    check_dependencies()
    
    # 设置环境变量
    setup_environment()
    
    # 如果需要自动打开浏览器
    if args.open_browser:
        # 延迟2秒，等待服务器启动
        def open_browser():
            time.sleep(2)
            webbrowser.open(f"http://{args.host}:{args.port}/docs")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
    
    # 运行API服务器
    run_api_server(
        host=args.host, 
        port=args.port, 
        reload=not args.no_reload
    )

if __name__ == "__main__":
    main() 