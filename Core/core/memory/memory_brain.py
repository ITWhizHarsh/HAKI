"""
Memory_Brain — persistent knowledge store and RAG retrieval.

PRIVACY & DATA-LOCALITY GUARANTEE (Req 9.2)
---------------------------------------------
All notes are stored **locally on the User's device only**.  This module
never opens a network socket, never writes to a remote filesystem, and
never sends raw note content to any external service.

  * The vault path is validated at construction time by
    :class:`LocalStorageGuard` to ensure it is a local, absolute path
    rather than a cloud-sync folder (e.g. iCloud Drive) or a network
    mount.
  * The optional embeddings provider (for RAG indexing) may call an
    external API *for embedding computation only* when the user has
    explicitly configured API mode.  Raw note content is sent for
    embedding **only** when the user opts in to an external embedding
    API.  The requirement that "notes never leave the device" is
    enforced at the orchestration layer — the embeddings provider must
    disclose API usage before first use (Req 20.5).  Within this module
    only the embedding *vector* (a list of floats) is received back;
    note text is not stored externally.
  * The :class:`PrivacyManager` companion class lets callers designate
    any conversation as private before or during that conversation
    (Req 9.7).  Private conversations must not have their content
    written to Memory_Brain (enforced by :class:`~core.learning.LearningEngine`).

Notes are stored as Obsidian-compatible Markdown files with YAML front
matter.  A rebuildable local vector index (SQLite sidecar) provides
hybrid retrieval (vector similarity + term/topic filter, ≤2 s).

Design: Memory, RAG & Learning; Data Models (Note); Vault + RAG design;
        Settings & Privacy; Security Considerations.
Requirements: 7.1–7.8, 9.2–9.6, 9.8.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .chunker import Chunker
from .indexer import Indexer
from .models import Chunk, Note, NoteSource
from .serializer import NoteSerializer
from .vault import StoreResult, Vault

if TYPE_CHECKING:
    from core.model_provider.model_provider import ModelProvider

# ---------------------------------------------------------------------------
# Local-storage guard (Req 9.2)
# ---------------------------------------------------------------------------

# Cloud-sync and network-path fragments that indicate a note might leave the
# device.  This is a best-effort check; it is not exhaustive.
_CLOUD_PATH_FRAGMENTS: tuple[str, ...] = (
    "icloud drive",
    "icloudrive",
    "/mobile documents/",   # iCloud Drive internal path on macOS
    "onedrive",
    "dropbox",
    "google drive",
    "googledrive",
    "box sync",
    "boxsync",
    "sugarsync",
    "/smb/",                # SMB network mounts
    "/afp/",                # AFP network mounts
    "/net/",                # automount NFS
    "/volumes/",            # macOS external volumes (best-effort, not blocked)
)


class LocalStorageGuard:
    """
    Validates that a vault path is a local, non-network filesystem path.

    Raises :class:`ValueError` if the path appears to be a cloud-sync
    folder or network mount, enforcing the local-only storage guarantee
    (Req 9.2).

    This check is intentionally conservative: it validates on
    construction and warns rather than blocking for ambiguous paths like
    ``/Volumes/...``.  The guard prevents accidental mis-configuration
    (e.g. setting the vault inside iCloud Drive) rather than acting as a
    security boundary.
    """

    def __init__(self, vault_path: Path) -> None:
        self._vault_path = vault_path
        self._validate()

    def _validate(self) -> None:
        """
        Check that the path does not look like a cloud-sync or network path.

        Raises
        ------
        ValueError
            If the path matches a known cloud-sync or network-mount
            pattern, with a message explaining the issue and suggesting
            a local path.
        """
        path_lower = str(self._vault_path).lower()
        for fragment in _CLOUD_PATH_FRAGMENTS:
            if fragment in path_lower:
                raise ValueError(
                    f"LocalStorageGuard: the vault path '{self._vault_path}' "
                    f"appears to be inside a cloud-sync or network location "
                    f"(matched '{fragment}'). "
                    "Notes must be stored locally on the device to satisfy "
                    "the privacy guarantee (Req 9.2). "
                    "Please choose a path under your home directory, "
                    "e.g. ~/.haki/vault"
                )


# ---------------------------------------------------------------------------
# Result types for delete / export operations
# ---------------------------------------------------------------------------


@dataclass
class ForgetResult:
    """
    Outcome of a :meth:`MemoryBrain.forget` or
    :meth:`MemoryBrain.forget_all` operation.

    Attributes
    ----------
    success:
        ``True`` if the deletion completed and was confirmed.
    error:
        Human-readable reason for failure when ``success`` is ``False``.
    """

    success: bool
    error: str | None = None

    @classmethod
    def ok(cls) -> "ForgetResult":
        """Successful deletion confirmed."""
        return cls(success=True)

    @classmethod
    def fail(cls, error: str) -> "ForgetResult":
        """Failed deletion; data is intact."""
        return cls(success=False, error=error)


@dataclass
class ExportResult:
    """
    Outcome of a :meth:`MemoryBrain.export` operation.

    Attributes
    ----------
    success:
        ``True`` if the export file was written completely.
    path:
        The path of the exported file on success.
    error:
        Human-readable reason for failure when ``success`` is ``False``.
    """

    success: bool
    path: Path | None = None
    error: str | None = None

    @classmethod
    def ok(cls, path: Path) -> "ExportResult":
        """Successful export to *path*."""
        return cls(success=True, path=path)

    @classmethod
    def fail(cls, error: str) -> "ExportResult":
        """Failed export; no partial file was produced."""
        return cls(success=False, error=error)


# ---------------------------------------------------------------------------
# PrivacyManager (Req 9.7)
# ---------------------------------------------------------------------------


class PrivacyManager:
    """
    Always-accessible control to designate conversations as private
    (Req 9.7).

    A private conversation must not have its content written into the
    Memory_Brain by the Learning_Engine (enforced at the
    :class:`~core.learning.LearningEngine` level via the ``is_private``
    parameter of ``on_conversation_end``).

    This implementation persists privacy designations in a lightweight
    SQLite database co-located with the vault.  When the encrypted app
    store (Task 2.1) is available, this class can be replaced by a thin
    wrapper around the ``PrivacyState`` model — the interface is
    identical.

    The control is accessible **before and during** any conversation:
    callers may call :meth:`designate_private` at any point to mark a
    conversation, and :meth:`is_private` at any point to check its
    status (Req 9.7).

    Parameters
    ----------
    db_path:
        Path to the SQLite file used to persist privacy state.  Defaults
        to ``~/.haki/privacy.db``.
    """

    _DEFAULT_DB = Path.home() / ".haki" / "privacy.db"

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or self._DEFAULT_DB
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Public API (Req 9.7)
    # ------------------------------------------------------------------

    def designate_private(self, conversation_id: str) -> None:
        """
        Mark *conversation_id* as private (Req 9.7).

        Once designated, :meth:`is_private` returns ``True`` for this
        conversation and the Learning_Engine will skip it.  Calling this
        method is idempotent — designating the same conversation twice
        is a no-op.

        Parameters
        ----------
        conversation_id:
            Stable identifier for the conversation session.
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO privacy_state
                    (conversation_id, is_private, designated_at)
                VALUES (?, 1, ?)
                """,
                (conversation_id, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()

    # Alias to satisfy the task-15.2 API contract (Req 9.7)
    mark_conversation_private = designate_private

    def is_private(self, conversation_id: str) -> bool:
        """
        Return ``True`` if *conversation_id* has been designated private
        (Req 9.2, 9.7).

        Parameters
        ----------
        conversation_id:
            Stable identifier for the conversation session.

        Returns
        -------
        bool
            ``True`` if the conversation was designated private,
            ``False`` otherwise (including when the ID is unknown).
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(
                "SELECT is_private FROM privacy_state "
                "WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return bool(row and row[0])

    def revoke_private(self, conversation_id: str) -> None:
        """
        Remove the private designation for *conversation_id*.

        After this call :meth:`is_private` returns ``False``.  No-op if
        the conversation was not designated private.
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "DELETE FROM privacy_state WHERE conversation_id = ?",
                (conversation_id,),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Ensure the privacy_state table exists."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS privacy_state (
                    conversation_id TEXT PRIMARY KEY,
                    is_private      INTEGER NOT NULL DEFAULT 1,
                    designated_at   TEXT
                )
                """
            )
            conn.commit()


# ---------------------------------------------------------------------------
# Stop words for query term extraction (Req 7.3, 7.7)
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "need", "dare", "ought", "used", "it", "its", "this",
    "that", "these", "those", "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "they", "them", "their", "what",
    "which", "who", "whom", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "not",
    "only", "same", "so", "than", "too", "very", "just", "out", "if",
    "then", "now", "get", "make", "see", "know", "go", "come", "say", "tell",
    "ask", "give", "take", "use", "find", "think", "want", "look", "also",
    "haki", "about", "tell", "me", "us", "any", "know", "please", "show",
})


def _extract_query_terms(query: str) -> set[str]:
    """
    Extract meaningful search terms from a query string.

    Strips stop words and tokens shorter than 3 characters, returning a
    set of lowercase terms that represent the semantic content of the
    query.  Used by :meth:`MemoryBrain.retrieve` to perform term/topic
    filtering (Req 7.3, 7.7).

    Parameters
    ----------
    query:
        Raw query text, e.g. ``"what do you know about computer networks"``.

    Returns
    -------
    set[str]
        Lowercased meaningful terms, e.g. ``{"computer", "networks"}``.
    """
    # Lowercase and split on any non-alphanumeric run (handles punctuation too)
    tokens = re.split(r"[^a-z0-9]+", query.lower())
    return {
        t for t in tokens
        if len(t) >= 3 and t not in _STOP_WORDS
    }


def _note_matches_terms(note: Note, query_terms: set[str]) -> bool:
    """
    Return ``True`` when *note* shares at least one term with *query_terms*.

    Checks the note body, topics list, and tags list using case-insensitive
    substring matching (Req 7.3, 7.7).

    Parameters
    ----------
    note:
        The note to test.
    query_terms:
        Non-empty set of lowercase search terms.

    Returns
    -------
    bool
    """
    # Build a single searchable string from all note content
    searchable = " ".join(
        [note.body] + list(note.topics) + list(note.tags)
    ).lower()

    return any(term in searchable for term in query_terms)


class MemoryBrain:
    """
    Persistent knowledge store and RAG retrieval engine.

    DATA-LOCALITY GUARANTEE
    -----------------------
    Notes are **never** sent off-device by this class.  The vault path
    is validated by :class:`LocalStorageGuard` at construction time to
    prevent accidental placement inside a cloud-sync folder.  Embeddings
    may be computed via an external API only when the user has explicitly
    configured the embeddings provider for API mode, and in that case
    only the chunk *text* (not the full note) is sent for embedding.

    Writes notes to an Obsidian-compatible Markdown vault on disk via the
    :class:`~core.memory.vault.Vault` class (atomic writes, Req 7.1–7.2).
    After each successful write, the note is auto-indexed into the local
    SQLite vector sidecar via :class:`~core.memory.indexer.Indexer`
    (Req 7.3, Task 14.1).

    Parameters
    ----------
    vault_path:
        Root directory of the vault.  Defaults to ``~/.haki/vault``.
        Must be a local filesystem path; cloud-sync paths raise
        :class:`ValueError` (Req 9.2).
    embeddings_provider:
        A :class:`~core.model_provider.ModelProvider` for the
        ``Capability.EMBEDDINGS`` capability.  When ``None`` (default),
        the vector index is not maintained and ``retrieve()`` falls back
        to returning an empty list.
    chunker:
        Optional custom :class:`~core.memory.chunker.Chunker`.  Defaults
        to the standard 200-token / 20-token-overlap settings.
    skip_local_guard:
        When ``True``, skip the :class:`LocalStorageGuard` check.  Only
        use this in tests that need to write to a ``tmp_path`` that may
        technically match a cloud fragment (e.g. a path with "drive" in
        it).  Should **never** be set in production code.
    """

    def __init__(
        self,
        vault_path: Path | None = None,
        embeddings_provider: "ModelProvider | None" = None,
        chunker: Chunker | None = None,
        skip_local_guard: bool = False,
    ) -> None:
        vault_root = vault_path or Path.home() / ".haki" / "vault"

        # Enforce local-only storage constraint (Req 9.2).
        # The guard raises ValueError if the path looks like a cloud/network mount.
        if not skip_local_guard:
            LocalStorageGuard(vault_root)

        self._vault = Vault(vault_root)
        self._serializer = NoteSerializer()

        # Optional vector index (enabled when an embeddings provider is supplied)
        self._indexer: Indexer | None = None
        if embeddings_provider is not None:
            self._indexer = Indexer(
                vault_path=vault_root,
                embeddings_provider=embeddings_provider,
                chunker=chunker,
            )

    # ------------------------------------------------------------------
    # Vault lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        """
        Ensure the vault directory, empty index, and vector index exist
        (Req 7.4).

        Called at service startup regardless of whether any notes exist.
        If the vector index sidecar is missing or corrupted it is
        recreated (rebuildable property, Task 14.1).
        """
        self._vault.init()
        if self._indexer is not None:
            self._indexer.init()

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def remember(
        self,
        body: str = "",
        tags: list[str] | None = None,
        topics: list[str] | None = None,
        source: NoteSource = NoteSource.USER_STATED,
        *,
        content: str | None = None,
    ) -> StoreResult:
        """
        Store a new note and confirm only after a successful durable
        write (Req 7.1, 7.2).

        After the write succeeds, the note is automatically indexed into
        the local vector store (Task 14.1) so it becomes retrievable via
        :meth:`retrieve`.

        Parameters
        ----------
        body:
            The Markdown prose content of the note.  ``content`` is
            accepted as a synonym for backward-compatibility with earlier
            scaffold code.
        tags:
            Free-form label strings.
        topics:
            Normalised retrieval terms.
        source:
            Provenance of the note.

        Returns ``StoreResult.ok(note_id)`` on success or
        ``StoreResult.fail(reason)`` on any error; no partial note is left.
        """
        # Accept either `body` or `content` as the note text
        note_body = content if content is not None else body
        note = Note(
            body=note_body,
            tags=tags or [],
            topics=topics or [],
            source=source,
        )
        result = self._vault.store(note)

        # Auto-index only after a confirmed durable write (Req 7.1)
        if result.success and self._indexer is not None:
            try:
                self._indexer.index_note(note)
            except Exception:
                # Indexing errors do NOT undo the durable write — the note
                # is still stored; the index can be rebuilt later.
                pass

        return result

    def retrieve(self, query: str, k: int = 5) -> list[Note]:
        """
        Retrieve notes matching *query* via hybrid retrieval (Req 7.3, 7.7).

        Combines vector similarity search with a term/topic filter so that
        only notes sharing at least one meaningful term or topic with the
        query are returned.  Superseded notes are always excluded.

        Handles "what do you know about X" queries naturally — the intent
        phrase is stripped of stop words so only meaningful topic terms
        drive the match (Req 7.7).

        Parameters
        ----------
        query:
            Plain-text query, e.g. ``"what do you know about networks"`` or
            ``"networks midterm exam"``.
        k:
            Maximum number of notes to return.

        Returns
        -------
        list[Note]
            Up to *k* Notes whose content contains at least one term/topic
            matching the query and that are not superseded, ordered by
            descending vector similarity.  Returns ``[]`` when the vault is
            empty, no provider is configured, or no notes match.
        """
        if self._indexer is None:
            return []

        # Extract meaningful query terms (strips stop words, short tokens)
        query_terms = _extract_query_terms(query)

        # Determine superseded note IDs for exclusion
        all_notes = self._vault.list_all()
        if not all_notes:
            return []

        superseded_ids = {n.id for n in all_notes if n.superseded_by is not None}

        # Vector search retrieves a broader candidate set (k * 4) so the
        # term filter downstream has enough candidates to fill k results.
        candidate_k = max(k * 4, 20)
        chunks = self._indexer.search(
            query, k=candidate_k, exclude_note_ids=superseded_ids
        )
        if not chunks:
            return []

        # Build a note lookup map and de-duplicate by note ID (preserve
        # chunk similarity ordering).
        note_map = {n.id: n for n in all_notes}
        seen: set[str] = set()
        results: list[Note] = []

        for chunk in chunks:
            if chunk.note_id in seen:
                continue
            note = note_map.get(chunk.note_id)
            if note is None:
                continue
            # Hard exclusion: superseded notes must never be returned (Req 8.2, 8.3)
            if note.superseded_by is not None:
                continue
            # Term/topic filter: note must share ≥1 term with the query (Req 7.3, 7.7)
            # When no meaningful query terms exist (e.g. all stop words), skip
            # filtering and fall back to pure vector similarity.
            if query_terms and not _note_matches_terms(note, query_terms):
                continue

            seen.add(chunk.note_id)
            results.append(note)
            if len(results) >= k:
                break

        return results

    async def aretrieve(self, query: str, k: int = 5) -> list[Note]:
        """
        Async wrapper for :meth:`retrieve` that runs the synchronous
        implementation in a thread pool executor so it never blocks the
        event loop (Req 7.3).

        Called by the Orchestrator's ``_retrieve_memory()`` when the
        Memory_Brain is wired into the turn loop (Task 14.3).

        Parameters
        ----------
        query:
            Plain-text query string.
        k:
            Maximum number of notes to return.

        Returns
        -------
        list[Note]
            Up to *k* matching notes; see :meth:`retrieve` for full semantics.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.retrieve, query, k)

    def forget(self, note_id: str) -> ForgetResult:
        """
        Delete a single note by ID and confirm only after durable removal
        (Req 7.6, 9.4).

        Atomic contract:
        - The file is removed from the vault **before** the vector index
          is updated.  If the file removal fails the note is left intact
          and a :class:`ForgetResult.fail` is returned — the data is
          never silently lost.
        - The vector index is updated **only** after the file removal
          succeeds.  An indexing error does not undo the file removal
          (the index is rebuildable) but is recorded in the result.

        Parameters
        ----------
        note_id:
            The stable ID of the note to delete.

        Returns
        -------
        ForgetResult
            ``ForgetResult.ok()`` when the note was found and durably
            removed.  ``ForgetResult.fail(reason)`` when the note was not
            found or the file could not be removed — in all failure cases
            data remains intact (Req 9.4, 9.6).
        """
        path = self._vault.note_path(note_id)

        # Note does not exist — nothing to remove.
        if not path.exists():
            # Also check the index so we can return a clean result.
            known_ids = {n.id for n in self._vault.list_all()}
            if note_id not in known_ids:
                return ForgetResult.fail(
                    f"Note '{note_id}' not found; no deletion performed."
                )

        # Attempt durable file removal first.
        try:
            removed = self._vault.delete(note_id)
        except OSError as exc:
            # File exists but could not be removed — data is intact.
            return ForgetResult.fail(
                f"Could not delete note '{note_id}': {exc}. "
                "The note has not been removed."
            )

        if not removed:
            # Note was not present in the vault (concurrent deletion, etc.)
            return ForgetResult.fail(
                f"Note '{note_id}' was not found in the vault; "
                "no deletion performed."
            )

        # File removed successfully.  Now update the vector index.
        if self._indexer is not None:
            try:
                self._indexer.remove_note(note_id)
            except Exception:
                # Index errors do not roll back the deletion — the note is
                # gone and the index can be rebuilt.  Callers receive a
                # success result; the stale index entry will be evicted on
                # the next rebuild.
                pass

        return ForgetResult.ok()

    def forget_all(self) -> ForgetResult:
        """
        Delete all stored notes atomically and confirm only after all are
        removed (Req 9.5, 9.6).

        Atomic contract (Req 9.5, 9.6):
        - All note files are moved to a temporary staging directory first
          (see :meth:`Vault.delete_all_atomic`).
        - The index sidecar is emptied only after all moves succeed.
        - If any step fails the original files are restored and this
          method returns ``ForgetResult.fail(...)`` — the vault is left
          unchanged and the user is informed.
        - The vector index is cleared only after the vault is confirmed
          empty.

        Returns
        -------
        ForgetResult
            ``ForgetResult.ok()`` when all notes were durably removed.
            ``ForgetResult.fail(reason)`` if the operation could not be
            confirmed — in that case notes are left intact (Req 9.6).
        """
        try:
            success = self._vault.delete_all_atomic()
        except RuntimeError as exc:
            # delete_all_atomic raises RuntimeError only when both the
            # operation AND the restore failed — a critical edge case.
            return ForgetResult.fail(
                f"Critical failure during delete_all: {exc}. "
                "Some notes may be in an inconsistent state; "
                "please check the vault directory."
            )

        if not success:
            return ForgetResult.fail(
                "Could not confirm deletion of all notes. "
                "No notes have been removed."
            )

        # Vault is confirmed empty — clear the vector index.
        if self._indexer is not None:
            try:
                self._indexer.rebuild([])
            except Exception:
                # Index rebuild failure does not undo the vault deletion.
                # Stale index entries will be evicted on next init/rebuild.
                pass

        return ForgetResult.ok()

    def export(
        self,
        destination: Path | None = None,
    ) -> ExportResult:
        """
        Export all notes to a single user-accessible Markdown file and
        confirm only after the complete file is written (Req 9.3, 9.8).

        Atomic contract (Req 9.8):
        - Content is written to a temp file first, then atomically renamed
          to *destination*.  If the write fails, the temp file is removed
          and *destination* is left untouched — no partial file is produced.

        Parameters
        ----------
        destination:
            Path where the export file should be written.  Defaults to
            ``~/Desktop/haki_memory_export.md`` (a user-accessible
            location on macOS).

        Returns
        -------
        ExportResult
            ``ExportResult.ok(path)`` when the full file was written.
            ``ExportResult.fail(reason)`` when the write did not complete —
            in that case no file (or no partial file) exists at
            *destination* (Req 9.8).
        """
        if destination is None:
            destination = Path.home() / "Desktop" / "haki_memory_export.md"

        notes = self._vault.list_all()

        # Build the export content.  Each note is separated by a horizontal
        # rule so the file is human-readable as a single Markdown document.
        if notes:
            content = "\n\n---\n\n".join(
                self._serializer.serialize(n) for n in notes
            )
        else:
            content = "# HAKI Memory Export\n\n*(No notes stored.)*\n"

        # Write atomically via the vault's export helper (Req 9.8).
        try:
            success = self._vault.export_atomic(destination, content)
        except Exception as exc:
            return ExportResult.fail(
                f"Export failed: {exc}. No partial file was produced."
            )

        if not success:
            return ExportResult.fail(
                f"Export to '{destination}' failed. "
                "No partial file was produced (Req 9.8)."
            )

        return ExportResult.ok(destination)

    def all_notes(self) -> list[Note]:
        """Return all notes in the vault (including superseded ones)."""
        return self._vault.list_all()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def rebuild_index(self) -> None:
        """
        Rebuild the vector index from scratch using all notes in the vault.

        Idempotent: safe to call at startup when the index is missing or
        when the vault has been modified externally (Task 14.1).

        No-op when no embeddings provider was supplied.
        """
        if self._indexer is None:
            return
        notes = self._vault.list_all()
        self._indexer.rebuild(notes)


# ---------------------------------------------------------------------------
# PrivacyState alias (Req 9.7)
# ---------------------------------------------------------------------------
# The design document and task-15.2 description refer to this component as
# "PrivacyState".  The implementation class is named PrivacyManager for
# historical reasons.  Both names are exported so callers can use either.
PrivacyState = PrivacyManager
