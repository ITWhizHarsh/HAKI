"""
Learning_Engine — autonomous knowledge extraction from conversations.

Triggered on explicit conversation-end or 300 s idle.  Extracts durable
facts/preferences via the LLM, writes them atomically to Memory_Brain,
and handles conflict-supersede logic.

Design: Autonomous Learning loop.
Requirements: 8.1–8.7, 9.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


@dataclass
class LearnedItem:
    """A single durable fact or preference extracted from a conversation."""

    id: str
    note_id: str  # ID of the note written into Memory_Brain
    text: str
    learned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    incomplete: bool = False  # True when write failed or nothing extractable (Req 8.6, 8.7)


@dataclass
class LearningReport:
    """
    Summary of what the Learning_Engine did at the end of a conversation.

    Provided to the user when they query recently-learned items (Req 8.4).
    """

    conversation_id: str
    items: list[LearnedItem] = field(default_factory=list)
    skipped_private: bool = False  # True when conversation was private (Req 9.1)


class LearningEngine:
    """
    Extracts durable knowledge from concluded conversations.

    This is a stub implementation.  The full implementation (Task 16) adds
    LLM-based extraction, conflict detection, supersede logic, and atomic
    write handling.
    """

    # Idle timeout in seconds before a conversation is considered ended (Req 8.1)
    IDLE_TIMEOUT_SECONDS: int = 300

    def __init__(self, memory_brain: Any | None = None) -> None:
        self._memory_brain = memory_brain
        self._reports: list[LearningReport] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_conversation_end(self, transcript: str, conversation_id: str, is_private: bool = False) -> LearningReport:
        """
        Process a concluded conversation and extract durable items.

        Parameters
        ----------
        transcript:
            Full text of the conversation.
        conversation_id:
            Stable identifier for this conversation session.
        is_private:
            True when the user has designated the conversation as private;
            no items are written in this case (Req 9.1).

        Returns
        -------
        LearningReport
            Summary of items learned (or skipped for privacy).
        """
        report = LearningReport(conversation_id=conversation_id)

        if is_private:
            report.skipped_private = True
            self._reports.append(report)
            return report

        # Stub: real LLM extraction added in Task 16.1
        extracted: list[str] = self._extract_items_stub(transcript)

        if not extracted:
            # Nothing extractable — mark conversation incomplete (Req 8.6)
            report.items.append(
                LearnedItem(
                    id=f"{conversation_id}:empty",
                    note_id="",
                    text="",
                    incomplete=True,
                )
            )
            self._reports.append(report)
            return report

        for i, item_text in enumerate(extracted):
            item = self._write_item(conversation_id=conversation_id, index=i, text=item_text)
            report.items.append(item)

        self._reports.append(report)
        return report

    def recently_learned(self, days: int = 7) -> list[LearnedItem]:
        """
        Return items learned within the last *days* days (Req 8.4).

        *days* must be in [1, 90]; defaults to 7.
        """
        days = max(1, min(90, days))
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return [
            item
            for report in self._reports
            for item in report.items
            if not item.incomplete and item.learned_at >= cutoff
        ]

    def mark_incorrect(self, item_id: str) -> bool:
        """
        Remove an item marked incorrect by the user and confirm (Req 8.5).

        Returns True if the item was found and removed, False otherwise.
        Stub: real removal from Memory_Brain added in Task 16.3.
        """
        for report in self._reports:
            for item in report.items:
                if item.id == item_id:
                    if self._memory_brain is not None:
                        self._memory_brain.forget(item.note_id)
                    report.items.remove(item)
                    return True
        return False

    # ------------------------------------------------------------------
    # Private helpers — stubs
    # ------------------------------------------------------------------

    def _extract_items_stub(self, transcript: str) -> list[str]:
        """
        Stub: return an empty list (no extraction without an LLM).
        Full LLM-based extraction added in Task 16.1.
        """
        return []

    def _write_item(self, conversation_id: str, index: int, text: str) -> LearnedItem:
        """
        Write one extracted item to Memory_Brain atomically (Req 8.7).
        Stub: real write + conflict-supersede logic added in Task 16.2.
        """
        note_id = f"{conversation_id}:{index}"
        return LearnedItem(
            id=f"{conversation_id}:item:{index}",
            note_id=note_id,
            text=text,
        )
