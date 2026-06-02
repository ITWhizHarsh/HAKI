"""
Memory sub-package.

Owns the Memory_Brain, vault I/O, RAG indexing and retrieval, and the
local vector index.  All notes are stored locally on-device in an
Obsidian-style Markdown vault.

Design reference: Memory, RAG & Learning; Vault + RAG design.
Requirements: 7, 9.2–9.6, 9.8.
"""

from .memory_brain import Note, Chunk, MemoryBrain

__all__ = ["Note", "Chunk", "MemoryBrain"]
