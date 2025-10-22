import os
import random
import time

from .logger import get_logger

logger = get_logger(__name__)


def is_text_pdf(pdf_path: str) -> bool:
    import fitz

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    if total_pages == 0:
        return False

    text_pages = 0
    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        text = page.get_text()
        if text.strip():
            text_pages += 1

    text_ratio = text_pages / total_pages
    return text_ratio > 0.5


def hashstr(input_string: str, length: int = 8, with_salt: bool = False) -> str:
    import hashlib

    if with_salt:
        input_string += str(time.time() + random.random())

    digest = hashlib.md5(str(input_string).encode()).hexdigest()
    return digest[:length]


def get_docker_safe_url(base_url: str) -> str:
    if os.getenv("RUNNING_IN_DOCKER") == "true":
        safe_url = base_url.replace("http://localhost", "http://host.docker.internal")
        safe_url = safe_url.replace("http://127.0.0.1", "http://host.docker.internal")
        logger.info("Running in docker, using %s as base url", safe_url)
        return safe_url
    return base_url


__all__ = [
    "get_logger",
    "logger",
    "hashstr",
    "is_text_pdf",
    "get_docker_safe_url",
]
