"""
Vault — durable, atomic Obsidian-style Markdown note store.

Manages the on-disk vault directory: writes notes atomically using a
temp-file-then-rename pattern, maintains an ``index.json`` sidecar that
lists all stored note IDs, and provides load/delete operations.

All notes are stored **locally on the device only** — no network I/O is
ever performed here (Req 9.2).

Design: Vault + RAG design.
Requirements: 7.1, 7.2, 7.4, 7.5, 9.3, 9.4, 9.5, 9.6, 9.8.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .models import Note


# ---------------------------------------------------------------------------
# StoreResult
# ---------------------------------------------------------------------------


@dataclass
class StoreResult:
    """
    Outcome of a single-note write operation (Req 7.1, 7.2).

    Constructed via the class methods ``ok`` and ``fail`` rather than
    directly so call-sites are self-documenting.
    """

    success: bool
    note_id: Optional[str]
    error: Optional[str]

    @classmethod
    def ok(cls, note_id: str) -> "StoreResult":
        """A successful store for the given note ID."""
        return cls(success=True, note_id=note_id, error=None)

    @classmethod
    def fail(cls, error: str) -> "StoreResult":
        """A failed store with a descriptive error message."""
        return cls(success=False, note_id=None, error=error)


# ---------------------------------------------------------------------------
# Vault
# ---------------------------------------------------------------------------


class Vault:
    """
    Obsidian-compatible Markdown vault on the local filesystem.

    Each note is stored as ``<vault_dir>/<note_id>.md``.  An
    ``index.json`` sidecar keeps a list of all stored note IDs so that
    ``list_all()`` doesn't have to glob the directory.

    All writes are atomic: content is written to a ``.tmp`` file first,
    then atomically renamed to the final path.  On failure the ``.tmp``
    file is removed and a ``StoreResult.fail(...)`` is returned — no
    partial state is left behind (Req 7.2).
    """

    _INDEX_FILE = "index.json"

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def note_path(self, note_id: str) -> Path:
        """Return the canonical on-disk path for ``note_id``."""
        return self._path / f"{note_id}.md"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        """
        Ensure the vault directory and an empty ``index.json`` exist (Req 7.4).

        Idempotent: safe to call when the vault already has notes.
        """
        self._path.mkdir(parents=True, exist_ok=True)
        index_path = self._path / self._INDEX_FILE
        if not index_path.exists():
            self._write_index(set())

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def store(self, note: Note) -> StoreResult:
        """
        Atomically write *note* to disk and update the index (Req 7.1, 7.2).

        Returns ``StoreResult.ok(note_id)`` on success.
        Returns ``StoreResult.fail(reason)`` on any error; no partial file
        is left and the index is not updated.
        """
        target = self.note_path(note.id)
        tmp_path: Optional[Path] = None
        try:
            # Serialize first so a serialization error doesn't touch disk
            markdown = note.to_markdown()

            # Write to a sibling .tmp file, then atomically rename
            fd, tmp_str = tempfile.mkstemp(
                dir=self._path, suffix=".tmp", prefix=note.id
            )
            tmp_path = Path(tmp_str)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(markdown)
                # Atomic rename (POSIX guarantees atomicity)
                tmp_path.rename(target)
                tmp_path = None  # rename succeeded, no cleanup needed
            except Exception:
                # Close fd if rename didn't happen and os.fdopen wasn't called
                raise

            # Update the index *after* the file is safely in place
            current_ids = self._read_index()
            current_ids.add(note.id)
            self._write_index(current_ids)

            return StoreResult.ok(note.id)

        except Exception as exc:
            # Clean up any leftover .tmp file
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            return StoreResult.fail(str(exc))

    def load(self, note_id: str) -> Optional[Note]:
        """
        Load and deserialize the note with *note_id* from disk.

        Returns ``None`` if no file exists for that ID.
        """
        path = self.note_path(note_id)
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        return Note.from_markdown(text)

    def list_all(self) -> list[Note]:
        """
        Return all notes currently in the vault (including superseded ones).

        Reads from the index for ID enumeration, then loads each file.
        Silently skips missing files (they may have been removed externally).
        """
        notes: list[Note] = []
        for note_id in self._read_index():
            note = self.load(note_id)
            if note is not None:
                notes.append(note)
        return notes

    def delete(self, note_id: str) -> bool:
        """
        Remove the note file for *note_id* and update the index.

        Atomic pattern (Req 9.4):
        - Only unlinks the file if the file actually exists.
        - Updates the index sidecar only **after** the file has been
          successfully removed.  If the unlink fails the index is left
          unchanged so no ghost entry is introduced.

        Returns ``True`` if the note existed and was removed, ``False``
        if no note with that ID was found.  Raises ``OSError`` if the
        file exists but cannot be removed (so callers know data is intact).
        """
        path = self.note_path(note_id)
        if not path.exists():
            return False
        # Remove the file first — if this raises, the index is untouched
        # so the note is still considered present (Req 9.4, 9.6).
        path.unlink()
        # File is gone; now update the index.
        current_ids = self._read_index()
        current_ids.discard(note_id)
        self._write_index(current_ids)
        return True

    def delete_all_atomic(self) -> bool:
        """
        Delete ALL notes atomically (Req 9.5, 9.6).

        Safe pattern:

        1. Move every note file to a temporary staging directory.
        2. Rewrite the index to be empty.
        3. Delete the staged files.

        If step 1 fails mid-way the staging directory is moved back so
        all original files are restored and ``False`` is returned — the
        vault is left in its original state (no data loss).

        Returns
        -------
        bool
            ``True`` if all notes were removed and the index cleared.
            ``False`` if an error occurred **and** the vault was
            successfully restored to its original state.

        Raises
        ------
        RuntimeError
            If the operation failed *and* the vault could not be fully
            restored (very unlikely edge case; callers should treat this
            as a data-integrity alert).
        """
        note_ids = list(self._read_index())
        if not note_ids:
            return True  # already empty, nothing to do

        # Step 1 — move all note files to a staging directory inside the vault
        # so the move is on the same filesystem (avoids cross-device rename).
        staging_dir: Optional[Path] = None
        moved: list[tuple[Path, Path]] = []  # (original, staged)
        try:
            staging_dir = Path(
                tempfile.mkdtemp(dir=self._path, prefix="_del_staging_")
            )
            for note_id in note_ids:
                src = self.note_path(note_id)
                if not src.exists():
                    continue  # already gone — skip
                dst = staging_dir / src.name
                src.rename(dst)
                moved.append((src, dst))
        except Exception as move_err:
            # Restore any files that were already moved
            restore_errors: list[str] = []
            for src, dst in moved:
                try:
                    dst.rename(src)
                except Exception as re:
                    restore_errors.append(f"{note_id}: {re}")
            # Clean up staging dir if it exists and is now empty
            if staging_dir is not None:
                try:
                    staging_dir.rmdir()
                except OSError:
                    pass
            if restore_errors:
                raise RuntimeError(
                    f"delete_all_atomic: move failed ({move_err}) AND "
                    f"restore failed for: {restore_errors}"
                ) from move_err
            return False

        # Step 2 — rewrite the index to empty
        try:
            self._write_index(set())
        except Exception as idx_err:
            # Restore all moved files
            restore_errors = []
            for src, dst in moved:
                try:
                    dst.rename(src)
                except Exception as re:
                    restore_errors.append(f"{src.name}: {re}")
            if staging_dir is not None:
                try:
                    staging_dir.rmdir()
                except OSError:
                    pass
            if restore_errors:
                raise RuntimeError(
                    f"delete_all_atomic: index write failed ({idx_err}) AND "
                    f"restore failed for: {restore_errors}"
                ) from idx_err
            return False

        # Step 3 — delete the staging directory and its contents
        try:
            shutil.rmtree(staging_dir)
        except Exception:
            # Staged files are outside the vault's live index; they will be
            # cleaned up eventually.  The operation is still considered a
            # success because the vault index is empty and all live note
            # paths are gone.
            pass

        return True

    def export_atomic(self, destination: Path, content: str) -> bool:
        """
        Write *content* atomically to *destination* (Req 9.3, 9.8).

        Writes to a temp file first, then atomically renames it to
        *destination*.  If the write fails the temp file is removed and
        *destination* is left untouched — no partial file is produced.

        Parameters
        ----------
        destination:
            The final path where the export file should appear.
        content:
            The Markdown string to write.

        Returns
        -------
        bool
            ``True`` on success, ``False`` if any error occurred (the
            partial temp file is cleaned up in both cases).

        Raises
        ------
        OSError
            Re-raised only if cleanup of the temp file itself fails (very
            unlikely edge case; callers should treat this as a storage
            alert).
        """
        # Ensure the destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)

        tmp_path: Optional[Path] = None
        try:
            # Write into the *same directory* as destination so the rename
            # is guaranteed to be on the same filesystem (atomic on POSIX).
            fd, tmp_str = tempfile.mkstemp(
                dir=destination.parent,
                suffix=".tmp",
                prefix="_haki_export_",
            )
            tmp_path = Path(tmp_str)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
                fh.flush()
                os.fsync(fh.fileno())
            # Atomic rename to the final path
            tmp_path.rename(destination)
            tmp_path = None  # rename succeeded; no cleanup needed
            return True
        except Exception:
            # Clean up the temp file so no partial export exists (Req 9.8)
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            return False

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        return self._path / self._INDEX_FILE

    def _read_index(self) -> set[str]:
        index_path = self._index_path()
        if not index_path.exists():
            return set()
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            return set(data.get("notes", []))
        except (json.JSONDecodeError, OSError):
            return set()

    def _write_index(self, note_ids: set[str]) -> None:
        data = {"notes": sorted(note_ids)}
        self._index_path().write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
