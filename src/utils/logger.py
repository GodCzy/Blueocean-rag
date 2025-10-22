#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一的日志工具模块"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.theme import Theme

    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - rich 为可选依赖
    RICH_AVAILABLE = False

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOGGERS: dict[str, logging.Logger] = {}


def get_log_file_path(name: str, date: Optional[datetime] = None) -> Path:
    """返回指定 logger 在给定日期对应的日志文件路径。"""

    date = date or datetime.now()
    return LOG_DIR / f"{name}_{date.strftime('%Y%m%d')}.log"


def get_latest_log_file(name: Optional[str] = None) -> Optional[Path]:
    """返回最近更新的日志文件，可选按名称过滤。"""

    pattern = f"{name}_*.log" if name else "*.log"
    candidates = sorted(
        LOG_DIR.glob(pattern),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path | str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console: bool = True,
    rich_console: bool = True,
) -> logging.Logger:
    """配置并返回日志记录器。"""

    if name in LOGGERS:
        return LOGGERS[name]

    logger = logging.getLogger(name)

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    logger.setLevel(level_map.get(level.upper(), logging.INFO))
    logger.propagate = False

    if logger.handlers:
        return logger

    log_file = Path(log_file) if log_file else get_log_file_path(name)

    file_handler = RotatingFileHandler(
        str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if console:
        if RICH_AVAILABLE and rich_console:
            custom_theme = Theme(
                {
                    "info": "dim cyan",
                    "warning": "yellow",
                    "error": "bold red",
                    "critical": "bold white on red",
                }
            )
            rich_console = Console(theme=custom_theme)
            console_handler = RichHandler(
                rich_tracebacks=True,
                console=rich_console,
                show_time=False,
                omit_repeated_times=False,
            )
            console_formatter = logging.Formatter("%(message)s")
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATE_FORMAT)

        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    LOGGERS[name] = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """获取或创建指定名称的日志记录器。"""

    if name in LOGGERS:
        return LOGGERS[name]
    return setup_logger(name)


__all__ = [
    "get_logger",
    "setup_logger",
    "get_log_file_path",
    "get_latest_log_file",
    "LOG_DIR",
]
