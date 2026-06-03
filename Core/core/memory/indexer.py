"""
Indexer — embeds Note chunks and persists them in a local SQLite vector index.

The index is stored in a ``vector_index.db`` file inside the vault directory,
as a sidecar to the Markdown note files.  It is fully rebuildable from the
vault at any time via :meth:`Indexer.rebuild`.

Storage format
--------------
SQLite database with a single ``chunks`` table:

    CREATE TABLE chunks (
        note_id     TEXT    NOT NULL,
        chunk_index INTEGER NOT NULL,
        text        TEXT    NOT NULL,
        embedding   BLOB    NOT NULL,   -- IEEE 754 little-endian float32 array
        PRIMARY KEY (note_id, chunk_index)
    )

Embeddings are stored as raw binary blobs using Python's ``struct``
module (``<{n}f`` — little-endian 32-bit floats).  This avoids any
external dependency beyond the stdlib.

Embedding model
---------------
Chunks are embedded via the :class:`~core.model_provider.ModelProvider`
abstraction using the ``"embeddings"`` capability (``Capability.EMBEDDINGS``).
The provider's ``invoke`` method is called with the chunk text and must
return either:
  - a ``list[float]`` vector, or
  - a ``dict`` whose ``"embedding"`` key contains a ``list[float]``.

This contract covers both the real sentence-embedding backends and the
existing :class:`~core.model_provider.StubModelProvider`.

Rebuild safety
--------------
If the SQLite database is missing or corrupted (detected by a failed
``PRAGMA integrity_check``), :meth:`Indexer.init` creates a fresh empty
database.  :meth:`Indexer.rebuild` re-indexes all notes from scratch
inside a single transaction so the index is either fully replaced or
left unchanged.

Design: Vault + RAG design (indexing).
Requirements: 7.3, 7.4.
"""

from __future__ import annotations

import math
import sqlite3
import struct
from pathlib import Path
from typing import TYPE_CHECKING

from .chunker import Chunker
from .models import Chunk, Note

if TYPE_CHECKING:
    from core.model_provider.model_provider import ModelProvider

# Name of the SQLite database file written alongside the vault notes.
_INDEX_FILENAME = "vector_index.db"

# Struct format string prefix for packing a float32 array.
# '<' = little-endian, 'f' = IEEE 754 32-bit float
_FLOAT_FMT_PREFIX = "<"
_FLOAT_FMT_ITEM = "f"


def _pack_vector(vector: list[float]) -> bytes:
    """Pack a list of floats into a little-endian float32 byte string."""
    fmt = f"{_FLOAT_FMT_PREFIX}{len(vector)}{_FLOAT_FMT_ITEM}"
    return struct.pack(fmt, *vector)


def _unpack_vector(blob: bytes) -> list[float]:
    """Unpack a little-endian float32 byte string into a list of floats."""
    n = len(blob) // 4   # each float32 is 4 bytes
    fmt = f"{_FLOAT_FMT_PREFIX}{n}{_FLOAT_FMT_ITEM}"
    return list(struct.unpack(fmt, blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute the cosine similarity between two vectors.

    Returns a value in [-1.0, 1.0].  Returns 0.0 if either vector is
    all-zeros (to avoid division by zero).
    """
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _extract_embedding(raw: object) -> list[float]:
    """
    Extract a float vector from a ModelProvider invoke result.

    Accepts:
    - ``list[float]``  (direct vector)
    - ``dict`` with an ``"embedding"`` key containing ``list[float]``

    Raises ``TypeError`` for unrecognised shapes.
    """
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, dict):
        if "embedding" in raw:
            return [float(x) for x in raw["embedding"]]
    raise TypeError(
        f"Embeddings provider returned an unrecognised type: {type(raw).__name__}. "
        "Expected list[float] or dict with 'embedding' key."
    )


class Indexer:
    """
    Chunk, embed, and persist Note content in a local SQLite vector index.

    Parameters
    ----------
    vault_path:
        Directory path of the vault (same directory used by :class:`Vault`).
        The SQLite database is stored at ``<vault_path>/vector_index.db``.
    embeddings_provider:
        A :class:`~core.model_provider.ModelProvider` whose ``capability``
        is ``Capability.EMBEDDINGS``.  Its ``invoke(text)`` must return
        ``list[float]`` or ``dict{"embedding": list[float]}``.
    chunker:
        Optional custom :class:`Chunker`.  Defaults to a :class:`Chunker`
        with the standard 200-token / 20-token-overlap settings.
    """

    def __init__(
        self,
        vault_path: Path,
        embeddings_provider: "ModelProvider",
        chunker: Chunker | None = None,
    ) -> None:
        self._vault_path = vault_path
        self._provider = embeddings_provider
        self._chunker = chunker or Chunker()
        self._db_path = vault_path / _INDEX_FILENAME

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        """
        Ensure the index database exists and has the expected schema.

        If the database file is missing it is created with an empty
        ``chunks`` table.  If the file exists but fails an integrity
        check it is replaced with a fresh empty database.

        Safe to call repeatedly (idempotent).
        """
        self._vault_path.mkdir(parents=True, exist_ok=True)

        if self._db_path.exists():
            # Verify integrity; replace on corruption.
            try:
                with sqlite3.connect(str(self._db_path)) as conn:
                    result = conn.execute("PRAGMA integrity_check").fetchone()
                    if result[0] != "ok":
                        raise sqlite3.DatabaseError("integrity check failed")
            except (sqlite3.DatabaseError, sqlite3.OperationalError):
                # Corrupt — remove and recreate
                self._db_path.unlink(missing_ok=True)

        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    note_id     TEXT    NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text        TEXT    NOT NULL,
                    embedding   BLOB    NOT NULL,
                    PRIMARY KEY (note_id, chunk_index)
                )
                """
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_note(self, note: Note) -> None:
        """
        Chunk, embed, and store all chunks for *note*.

        Existing chunks for the same ``note_id`` are replaced atomically.
        If the embeddings provider raises an exception, the exception
        propagates and the existing index entry (if any) is left unchanged.

        Parameters
        ----------
        note:
            The Note whose body should be indexed.
        """
        chunks = self._chunker.chunk(note.id, note.body)
        embedded: list[tuple[str, int, str, bytes]] = []
        for chunk in chunks:
            raw = self._provider.invoke(chunk.text)
            vector = _extract_embedding(raw)
            blob = _pack_vector(vector)
            embedded.append((note.id, chunk.chunk_index, chunk.text, blob))

        with sqlite3.connect(str(self._db_path)) as conn:
            # Remove all existing chunks for this note first (re-indexing)
            conn.execute(
                "DELETE FROM chunks WHERE note_id = ?", (note.id,)
            )
            conn.executemany(
                "INSERT INTO chunks (note_id, chunk_index, text, embedding) "
                "VALUES (?, ?, ?, ?)",
                embedded,
            )
            conn.commit()

    def remove_note(self, note_id: str) -> None:
        """
        Remove all indexed chunks for the note identified by *note_id*.

        No-op if the note has no indexed chunks.
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("DELETE FROM chunks WHERE note_id = ?", (note_id,))
            conn.commit()

    def rebuild(self, notes: list[Note]) -> None:
        """
        Rebuild the entire vector index from scratch.

        All existing rows are deleted and every note in *notes* is re-
        chunked, re-embedded, and re-inserted inside a single transaction.
        If any embedding call raises, the transaction is rolled back and
        the original index state is preserved.

        Parameters
        ----------
        notes:
            All notes currently in the vault (typically from
            :meth:`~core.memory.vault.Vault.list_all`).
        """
        # Build all rows first (before touching the DB) so that a provider
        # error leaves the existing index intact.
        rows: list[tuple[str, int, str, bytes]] = []
        for note in notes:
            chunks = self._chunker.chunk(note.id, note.body)
            for chunk in chunks:
                raw = self._provider.invoke(chunk.text)
                vector = _extract_embedding(raw)
                blob = _pack_vector(vector)
                rows.append((note.id, chunk.chunk_index, chunk.text, blob))

        # Write atomically inside one transaction.
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("DELETE FROM chunks")
            conn.executemany(
                "INSERT INTO chunks (note_id, chunk_index, text, embedding) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        exclude_note_ids: set[str] | None = None,
    ) -> list[Chunk]:
        """
        Return the top-*k* chunks most similar to *query*.

        The query is embedded with the same provider, then cosine
        similarity is computed against all stored chunks.  Results are
        sorted by descending similarity.

        Parameters
        ----------
        query:
            Plain-text query string.
        k:
            Maximum number of Chunk objects to return.
        exclude_note_ids:
            Note IDs to exclude from results (e.g. superseded notes).

        Returns
        -------
        list[Chunk]
            At most *k* Chunk objects with their ``embedding`` field
            populated.
        """
        raw_q = self._provider.invoke(query)
        query_vec = _extract_embedding(raw_q)

        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT note_id, chunk_index, text, embedding FROM chunks"
            ).fetchall()

        exclude = exclude_note_ids or set()
        scored: list[tuple[float, Chunk]] = []
        for note_id, chunk_index, text, blob in rows:
            if note_id in exclude:
                continue
            vec = _unpack_vector(blob)
            sim = _cosine_similarity(query_vec, vec)
            chunk = Chunk(
                note_id=note_id,
                chunk_index=chunk_index,
                text=text,
                embedding=vec,
            )
            scored.append((sim, chunk))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [chunk for _, chunk in scored[:k]]

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of indexed chunks."""
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            return row[0] if row else 0

    def get_chunks_for_note(self, note_id: str) -> list[Chunk]:
        """Return all stored chunks for a given note ID, ordered by chunk_index."""
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT note_id, chunk_index, text, embedding "
                "FROM chunks WHERE note_id = ? ORDER BY chunk_index",
                (note_id,),
            ).fetchall()
        return [
            Chunk(
                note_id=row[0],
                chunk_index=row[1],
                text=row[2],
                embedding=_unpack_vector(row[3]),
            )
            for row in rows
        ]
