"""
ProcessTracker — SQLite-backed tracker for conversation log processing state.

Part of the HAKI Brain Memory Processing Pipeline. Tracks which daily
conversation logs (`YYYY-MM-DD.md`) have been fully processed by the
Conversation_Scheduler so they are never re-processed on a later run.

See: .kiro/specs/haki-brain-memory-processing-pipeline/design.md
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS processed_logs (
    filename     TEXT PRIMARY KEY,   -- e.g. "2025-01-15.md"
    processed_at TEXT NOT NULL       -- ISO8601 UTC timestamp
);
"""


class ProcessTracker:
    """
    SQLite-backed tracker recording which conversation logs have been
    processed.

    Uses a single connection with WAL mode enabled (`PRAGMA
    journal_mode=WAL`) so concurrent readers (e.g. a status check while a
    write is in flight) never block on the writer.

    Parameters
    ----------
    db_path : Path
        Filesystem path to the SQLite database file. The parent directory
        is created if it does not already exist. The `processed_logs`
        table is created on first use if it does not already exist.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_DDL)
        self._conn.commit()
        logger.info("[ProcessTracker] Ready (db=%s)", db_path)

    def mark_processed(self, filename: str) -> None:
        """
        Record that *filename* has been fully processed.

        Idempotent — marking the same filename processed more than once is
        a no-op on subsequent calls (the `filename` primary key enforces
        uniqueness and `INSERT OR IGNORE` silently skips duplicates instead
        of raising an `IntegrityError`).
        """
        self._conn.execute(
            "INSERT OR IGNORE INTO processed_logs (filename, processed_at) "
            "VALUES (?, ?)",
            (filename, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()
        logger.debug("[ProcessTracker] Marked processed: %s", filename)

    def is_processed(self, filename: str) -> bool:
        """Return True if *filename* has been marked as processed."""
        cur = self._conn.execute(
            "SELECT 1 FROM processed_logs WHERE filename = ?", (filename,)
        )
        return cur.fetchone() is not None

    def get_unprocessed_conversations(
        self, cutoff_date: date, conversations_dir: Path
    ) -> list[str]:
        """
        Return `YYYY-MM-DD.md` filenames in *conversations_dir* that fall on
        or before *cutoff_date* and have not yet been marked as processed,
        sorted chronologically oldest-first.

        Filenames that do not match the `YYYY-MM-DD.md` pattern (or whose
        stem is not a valid ISO 8601 date) are ignored. The caller is
        responsible for passing a *cutoff_date* that excludes today's
        in-progress log (e.g. yesterday's date), per Requirement 6.4.
        """
        candidates = sorted(conversations_dir.glob("????-??-??.md"))
        result: list[str] = []
        for log_file in candidates:
            try:
                log_date = date.fromisoformat(log_file.stem)
            except ValueError:
                continue
            if log_date <= cutoff_date and not self.is_processed(log_file.name):
                result.append(log_file.name)
        return result  # already chronological because glob results were sorted

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
