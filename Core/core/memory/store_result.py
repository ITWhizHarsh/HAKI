"""
StoreResult — outcome of a vault write operation.

Returned by Vault.store() and MemoryBrain.remember() so callers can
distinguish a confirmed durable write from a failure without relying on
exceptions at the call site.

Design: Vault + RAG design.
Requirements: 7.1 (confirm only after durable write), 7.2 (no partial note on failure).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class StoreResult:
    """
    Outcome of a single note-store operation.

    Attributes:
        success:  True iff the note was durably written to the vault.
        note_id:  The id of the written note when *success* is True; None on failure.
        error:    A human-readable description of the failure when *success* is
                  False; None on success.
    """

    success: bool
    note_id: Optional[str] = None
    error: Optional[str] = None

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def ok(cls, note_id: str) -> "StoreResult":
        """Return a successful result for the given *note_id*."""
        return cls(success=True, note_id=note_id)

    @classmethod
    def fail(cls, error: str) -> "StoreResult":
        """Return a failure result with *error* describing what went wrong."""
        return cls(success=False, error=error)
