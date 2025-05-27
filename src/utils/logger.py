#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
logger.py - 日志工具

配置和管理应用日志。

作者: 成员C (后端工程师)
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path
from datetime import datetime

# 尝试导入 rich 库，提供彩色日志输出
try:
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.theme import Theme
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# 默认日志格式
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 创建日志目录
LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs"
)
os.makedirs(LOG_DIR, exist_ok=True)

# 全局日志字典，避免重复创建
LOGGERS = {}

def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: str = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console: bool = True,
    rich_console: bool = True
) -> logging.Logger:
    """配置并返回日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径
        max_bytes: 单个日志文件最大大小
        backup_count: 保留的日志文件数量
        console: 是否输出到控制台
        rich_console: 是否使用丰富控制台输出
        
    Returns:
        配置好的日志记录器
    """
    # 如果已存在，直接返回
    if name in LOGGERS:
        return LOGGERS[name]
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    
    # 设置日志级别
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    logger.setLevel(level_map.get(level.upper(), logging.INFO))
    
    # 防止日志重复输出
    logger.propagate = False
    
    # 已有处理器，无需再次添加
    if logger.handlers:
        return logger
    
    # 自动生成日志文件路径
    if not log_file:
        today = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(LOG_DIR, f"{name}_{today}.log")
    
    # 添加文件处理器
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 添加控制台处理器
    if console:
        if RICH_AVAILABLE and rich_console:
            # 使用 rich 库提供彩色日志
            custom_theme = Theme({
                "info": "dim cyan",
                "warning": "yellow",
                "error": "bold red",
                "critical": "bold white on red"
            })
            rich_console = Console(theme=custom_theme)
            console_handler = RichHandler(
                rich_tracebacks=True,
                console=rich_console,
                show_time=False,
                omit_repeated_times=False
            )
            console_formatter = logging.Formatter("%(message)s")
        else:
            # 标准控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # 缓存日志记录器
    LOGGERS[name] = logger
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """获取现有的日志记录器，如果不存在则创建
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器
    """
    if name in LOGGERS:
        return LOGGERS[name]
    
    return setup_logger(name) 