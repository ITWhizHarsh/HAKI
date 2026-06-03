"""
Data models for the Memory_Brain subsystem.

Defines the Note and Chunk dataclasses and the NoteSource enum that
represent the in-memory state of a note and a RAG chunk respectively.

Design: Data Models (Note), Memory, RAG & Learning.
Requirements: 7.8, 8.2, 8.3, 8.4.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class NoteSource(str, Enum):
    """Provenance of a stored note (Req 8.4)."""

    USER_STATED = "user_stated"
    LEARNED = "learned"
    OBSERVED = "observed"


@dataclass
class Note:
    """
    Single unit of persistent memory.

    Maps directly to the Markdown-with-YAML-front-matter file format
    described in the design document (Req 7.8).

    Fields
    ------
    id              Stable unique identifier, e.g. ``"2024-06-01T12-03-22-a1b2"``.
    created         UTC timestamp of first creation.
    updated         UTC timestamp of last modification.
    source          How the note was created (user_stated | learned | observed).
    tags            Free-form label strings (e.g. ``["exam", "networks"]``).
    topics          Normalised retrieval terms (e.g. ``["computer-networks"]``).
    superseded_by   ID of the newer note that replaces this one, or ``None``.
                    A non-None value excludes this note from retrieval (Req 8.2, 8.3).
    private         When ``True`` the note must not leave the device (Req 9.2).
    learned_session Session identifier set when ``source == LEARNED`` (Req 8.4).
    body            The Markdown prose content of the note.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: NoteSource = NoteSource.USER_STATED
    tags: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    superseded_by: Optional[str] = None
    private: bool = False
    learned_session: Optional[str] = None
    body: str = ""

    # ------------------------------------------------------------------
    # Convenience serialization helpers (delegate to NoteSerializer)
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Serialize to Obsidian-compatible Markdown with YAML front matter (Req 7.8)."""
        # Import here to avoid circular imports (serializer imports models)
        from .serializer import NoteSerializer
        return NoteSerializer().serialize(self)

    @classmethod
    def from_markdown(cls, text: str) -> "Note":
        """Parse an Obsidian-compatible Markdown note back into a Note (Req 7.8)."""
        from .serializer import NoteSerializer
        return NoteSerializer().deserialize(text)


@dataclass
class Chunk:
    """
    RAG unit — a text slice of a Note with its embedding vector.

    Parameters
    ----------
    note_id:      ID of the parent Note.
    chunk_index:  Zero-based position of this chunk within the note.
    text:         The raw text of this chunk.
    embedding:    Dense vector produced by the embeddings Model_Provider.
                  ``None`` if the chunk has not been embedded yet.
    """

    note_id: str
    chunk_index: int
    text: str
    embedding: Optional[list[float]] = None
