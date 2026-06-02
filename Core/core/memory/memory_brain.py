"""
Memory_Brain — persistent knowledge store and RAG retrieval.

Notes are stored as Obsidian-compatible Markdown files with YAML front
matter.  A rebuildable local vector index provides hybrid retrieval
(vector similarity + term/topic filter, ≤2 s).

Design: Memory, RAG & Learning; Data Models (Note).
Requirements: 7.1–7.8, 9.2–9.6, 9.8.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class NoteSource(str, Enum):
    USER_STATED = "user_stated"
    LEARNED = "learned"
    OBSERVED = "observed"


@dataclass
class Note:
    """
    Single unit of persistent memory.

    Maps directly to the Markdown-with-YAML-front-matter file format
    described in the design document.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: NoteSource = NoteSource.USER_STATED
    tags: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    superseded_by: str | None = None  # id of newer note (Req 8.2, 8.3)
    private: bool = False
    learned_session: str | None = None  # set when source=LEARNED (Req 8.4)
    body: str = ""

    # ------------------------------------------------------------------
    # Serialization helpers — full implementation in Task 13.1
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Serialize to Obsidian-compatible Markdown with YAML front matter."""
        lines = [
            "---",
            f"id: {self.id}",
            f"created: {self.created.isoformat()}",
            f"updated: {self.updated.isoformat()}",
            f"source: {self.source.value}",
            f"tags: [{', '.join(self.tags)}]",
            f"topics: [{', '.join(self.topics)}]",
            f"superseded_by: {self.superseded_by or 'null'}",
            f"private: {str(self.private).lower()}",
            f"learned_session: {self.learned_session or 'null'}",
            "---",
            "",
            self.body,
        ]
        return "\n".join(lines)

    @classmethod
    def from_markdown(cls, text: str) -> "Note":
        """Parse an Obsidian-compatible Markdown note. Stub — full parser in Task 13.1."""
        raise NotImplementedError("Full Note parser implemented in Task 13.1.")


@dataclass
class Chunk:
    """RAG unit: a slice of a Note with its embedding vector."""

    note_id: str
    chunk_index: int
    text: str
    embedding: list[float] = field(default_factory=list)


class MemoryBrain:
    """
    Persistent knowledge store and RAG retrieval engine.

    This is a stub implementation.  The full implementation adds vault
    I/O, the YAML-front-matter parser/serializer, the local vector index,
    hybrid retrieval, and atomic delete / export operations.
    """

    def __init__(self, vault_path: Path | None = None) -> None:
        self._vault_path: Path = vault_path or Path.home() / ".haki" / "vault"
        # In-memory store for the stub; replaced by durable vault in Task 13.2
        self._notes: dict[str, Note] = {}

    # ------------------------------------------------------------------
    # Vault lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        """
        Ensure the vault directory and empty index exist (Req 7.4).

        Called at service startup regardless of whether any notes exist.
        """
        self._vault_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core CRUD — stubs; durability + atomicity added in Task 13.2
    # ------------------------------------------------------------------

    def remember(self, body: str, tags: list[str] | None = None, source: NoteSource = NoteSource.USER_STATED) -> Note:
        """
        Store a new note (Req 7.1, 7.2).

        Confirms the store only after a successful durable write.
        Stub: writes to in-memory dict; durable I/O added in Task 13.2.
        """
        note = Note(body=body, tags=tags or [], source=source)
        self._notes[note.id] = note
        return note

    def retrieve(self, query: str, k: int = 5) -> list[Note]:
        """
        Retrieve notes matching *query* via hybrid term/topic + vector search (Req 7.3, 7.7).

        Returns at most *k* notes, excluding superseded notes.
        Stub: performs simple substring match; real RAG added in Task 14.
        """
        query_lower = query.lower()
        results = [
            note
            for note in self._notes.values()
            if note.superseded_by is None
            and (
                query_lower in note.body.lower()
                or any(query_lower in t.lower() for t in note.tags + note.topics)
            )
        ]
        return results[:k]

    def forget(self, note_id: str) -> bool:
        """
        Delete a specific note and confirm removal (Req 7.6, 9.4).

        Returns True if the note was found and removed, False otherwise.
        Stub: in-memory only; atomicity + confirmation added in Task 15.1.
        """
        return self._notes.pop(note_id, None) is not None

    def forget_all(self) -> None:
        """Delete all stored notes (Req 9.5, 9.6). Stub."""
        self._notes.clear()

    def export(self, destination: Path) -> None:
        """
        Export all notes to a single user-accessible file (Req 9.3, 9.8).

        Stub: writes concatenated Markdown; atomicity added in Task 15.1.
        """
        content = "\n\n---\n\n".join(n.to_markdown() for n in self._notes.values())
        destination.write_text(content, encoding="utf-8")

    def all_notes(self) -> list[Note]:
        """Return all notes (including superseded ones). Useful for tests."""
        return list(self._notes.values())
