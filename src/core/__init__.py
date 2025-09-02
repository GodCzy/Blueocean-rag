"""Lightweight core package initializer.

This module intentionally avoids importing subpackages that have heavy
optional dependencies (for example the Milvus client used by the
knowledge base).  Importing them here would make *any* module that relies
on :mod:`src.core` require those extras to be installed which breaks test
execution in minimal environments.

Only the history utilities are loaded eagerly; other components should be
imported directly from their respective submodules, e.g. ``from
src.core.knowledgebase import KnowledgeBase``.
"""

from .history import *  # noqa: F401,F403

__all__ = []

