from .history import *

__all__ = ["HistoryManager", "KnowledgeBase", "GraphDatabase"]


def __getattr__(name):  # pragma: no cover - lazy imports
    if name == "KnowledgeBase":
        from .knowledgebase import KnowledgeBase
        return KnowledgeBase
    if name == "GraphDatabase":
        from .graphbase import GraphDatabase
        return GraphDatabase
    raise AttributeError(name)
