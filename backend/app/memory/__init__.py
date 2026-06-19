"""Memory engine — public API.

Import the three entry points you actually need:

    from backend.app.memory import build_user_context, extract_memories
    from backend.app.memory.store import WeaviateMemoryStore
"""

from backend.app.memory.embeddings import embed, text_hash
from backend.app.memory.extractor import extract_memories
from backend.app.memory.retrieval import build_user_context

__all__ = ["build_user_context", "embed", "extract_memories", "text_hash"]
