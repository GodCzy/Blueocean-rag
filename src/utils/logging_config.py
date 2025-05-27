#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
logging_config.py - 日志配置

配置应用日志。

作者: 成员C (后端工程师)
"""

import os
from datetime import datetime
from src.utils.logger import setup_logger, get_logger

# 创建日志目录
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 获取当前日志记录器
logger = get_logger("logging_config")

# 为项目配置根日志记录器
root_logger = setup_logger("blueocean_rag", "INFO")

logger.info("日志系统初始化完成")
