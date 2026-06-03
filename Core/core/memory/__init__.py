"""
Memory sub-package.

Owns the Memory_Brain, vault I/O, RAG indexing and retrieval, and the
local vector index.  All notes are stored locally on-device in an
Obsidian-style Markdown vault.

Design reference: Memory, RAG & Learning; Vault + RAG design;
                  Settings & Privacy; Security Considerations.
Requirements: 7, 9.2–9.6, 9.8.
"""

from .chunker import Chunker
from .indexer import Indexer
from .memory_brain import (
    ExportResult,
    ForgetResult,
    LocalStorageGuard,
    MemoryBrain,
    PrivacyManager,
    PrivacyState,
)
from .models import Chunk, Note, NoteSource
from .serializer import NoteSerializationError, NoteSerializer
from .vault import StoreResult, Vault

__all__ = [
    "Note",
    "NoteSource",
    "Chunk",
    "Chunker",
    "Indexer",
    "MemoryBrain",
    "NoteSerializer",
    "NoteSerializationError",
    "StoreResult",
    "Vault",
    "ForgetResult",
    "ExportResult",
    "LocalStorageGuard",
    "PrivacyManager",
    "PrivacyState",
]
