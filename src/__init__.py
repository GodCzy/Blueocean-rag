"""Package bootstrap utilities.

Minor edits for code cleanup.

This module used to eagerly create global objects when imported which could
lead to slow start-up and unnecessary resource usage.  The initialization is
now done lazily via accessor functions.  Modules should use the provided
``get_*`` helpers instead of importing the objects directly.
"""

from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv("src/.env")

_executor = None
_config = None
_knowledge_base = None
_graph_base = None
_retriever = None


def get_executor() -> ThreadPoolExecutor:
    """Return a global thread pool executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor()
    return _executor


def get_config():
    """Return the global configuration instance."""
    global _config
    if _config is None:
        from src.config import Config

        _config = Config()
    return _config


def get_knowledge_base():
    """Return the global knowledge base instance."""
    global _knowledge_base
    if _knowledge_base is None:
        from src.core import KnowledgeBase

        _knowledge_base = KnowledgeBase()
    return _knowledge_base


def get_graph_base():
    """Return the global graph database instance."""
    global _graph_base
    if _graph_base is None:
        from src.core import GraphDatabase

        _graph_base = GraphDatabase()
    return _graph_base


def get_retriever():
    """Return the global Retriever instance."""
    global _retriever
    if _retriever is None:
        from src.core.retriever import Retriever

        _retriever = Retriever()
    return _retriever


def __getattr__(name):
    if name == "executor":
        return get_executor()
    if name == "config":
        return get_config()
    if name == "knowledge_base":
        return get_knowledge_base()
    if name == "graph_base":
        return get_graph_base()
    if name == "retriever":
        return get_retriever()
    raise AttributeError(name)
