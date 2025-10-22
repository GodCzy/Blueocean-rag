"""Deprecated module kept for backwards compatibility.

The service implementations have moved to :mod:`src.services.rag`. New code
should import from ``src.services`` instead of ``src.api`` to avoid mixing
transport and business logic layers.
"""

from __future__ import annotations

from warnings import warn

from src.services.rag import RAGService, get_rag_service

warn(
    "src.api.rag_api is deprecated; import from src.services.rag instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["RAGService", "get_rag_service"]
