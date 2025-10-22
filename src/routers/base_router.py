from collections import deque
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body

from src import (
    get_config as load_config,
    get_graph_base,
    get_knowledge_base,
    get_retriever,
)
from src.utils.logger import get_latest_log_file, get_logger

base = APIRouter()

logger = get_logger(__name__)
APPLICATION_LOGGER_NAME = "blueocean_rag"
get_logger(APPLICATION_LOGGER_NAME)

config = load_config()
retriever = get_retriever()
knowledge_base = get_knowledge_base()
graph_base = get_graph_base()


@base.get("/")
async def route_index():
    return {"message": "You Got It!"}


@base.get("/config")
def read_config():
    return config.get_safe_config()


@base.post("/config")
async def update_config(key: str = Body(...), value = Body(...)):
    if key == "custom_models":
        value = config.compare_custom_models(value)

    config[key] = value
    config.save()
    return config.get_safe_config()


@base.post("/restart")
async def restart():
    knowledge_base.restart()
    graph_base.restart()
    retriever.restart()
    return {"message": "Restarted!"}


def _read_log_tail(path: Path, max_lines: int = 1000) -> str:
    with path.open("r", encoding="utf-8") as handle:
        lines = deque(handle, maxlen=max_lines)
    return "".join(lines)


@base.get("/log")
def get_log():
    log_file: Optional[Path] = get_latest_log_file(APPLICATION_LOGGER_NAME) or get_latest_log_file()
    if not log_file or not log_file.exists():
        return {"log": "", "message": "log file not found", "log_file": None}

    return {
        "log": _read_log_tail(log_file),
        "message": "success",
        "log_file": str(log_file),
    }
