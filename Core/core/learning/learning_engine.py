"""
Learning_Engine — autonomous knowledge extraction from conversations.

Triggered on explicit conversation-end or 300 s idle.  Extracts durable
facts/preferences via the LLM, writes them atomically to Memory_Brain,
and handles conflict-supersede logic.

Design: Autonomous Learning loop.
Requirements: 8.1–8.7, 9.1.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.memory.memory_brain import MemoryBrain
    from core.memory.models import Note
    from core.model_provider.model_provider import ModelProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class LearnedItem:
    """
    A single durable fact or preference extracted from a conversation and
    successfully written into Memory_Brain.

    Req 8.4 — returned by :meth:`LearningEngine.recently_learned`.
    """

    note_id: str
    fact_or_preference: str
    topics: list[str]
    learned_session: str
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FailedItem:
    """
    A durable item that could not be written into Memory_Brain (Req 8.7).

    Stored in :attr:`LearningReport.failed_items` so the caller knows
    which items were not persisted and why.
    """

    fact_or_preference: str
    topics: list[str]
    reason: str


@dataclass
class LearningReport:
    """
    Summary of what the Learning_Engine did at the end of a conversation.

    Attributes
    ----------
    conversation_id:
        Stable identifier for the conversation session.
    is_private:
        Reflects the ``is_private`` flag passed to
        :meth:`LearningEngine.on_conversation_end`.
    skipped:
        ``True`` when the conversation was private and processing was
        skipped entirely (Req 9.1).
    incomplete:
        ``True`` when no durable items could be extracted from the
        conversation, or when every item failed to write (Req 8.6, 8.7).
    incomplete_reason:
        Human-readable reason for incompleteness, or ``None``.
    learned_items:
        Successfully written :class:`LearnedItem` objects.
    failed_items:
        :class:`FailedItem` objects for items that could not be written
        (Req 8.7).
    """

    conversation_id: str
    is_private: bool = False
    skipped: bool = False
    incomplete: bool = False
    incomplete_reason: str | None = None
    learned_items: list[LearnedItem] = field(default_factory=list)
    failed_items: list[FailedItem] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Extraction result (internal)
# ---------------------------------------------------------------------------


@dataclass
class _ExtractedItem:
    """Internal representation of an item returned by the LLM extraction prompt."""

    fact_or_preference: str
    topics: list[str]


# ---------------------------------------------------------------------------
# IdleWatcher — 300 s idle timeout (Req 8.1)
# ---------------------------------------------------------------------------


class IdleWatcher:
    """
    Asyncio-based idle timer that calls *callback* after *timeout_seconds*
    of inactivity (Req 8.1).

    Usage::

        async def end_handler(transcript, is_private):
            await engine.on_conversation_end(transcript, is_private)

        watcher = IdleWatcher(
            timeout_seconds=300,
            callback=end_handler,
            transcript_getter=lambda: current_transcript,
            is_private_getter=lambda: current_privacy_flag,
        )
        watcher.reset()   # restart timer on each new user message
        watcher.cancel()  # cancel when the user explicitly ends the conversation
    """

    def __init__(
        self,
        timeout_seconds: int,
        callback: Any,           # async callable(transcript, is_private)
        transcript_getter: Any,  # callable() -> str
        is_private_getter: Any,  # callable() -> bool
    ) -> None:
        self._timeout = timeout_seconds
        self._callback = callback
        self._transcript_getter = transcript_getter
        self._is_private_getter = is_private_getter
        self._task: asyncio.Task | None = None

    def reset(self) -> None:
        """
        Restart the idle countdown.

        Called after each user exchange so the timer fires only after
        *timeout_seconds* of silence following the *last* exchange (Req 8.1).
        """
        self.cancel()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return
        self._task = loop.create_task(self._countdown())

    def cancel(self) -> None:
        """Cancel the pending countdown without invoking the callback."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    async def _countdown(self) -> None:
        try:
            await asyncio.sleep(self._timeout)
        except asyncio.CancelledError:
            return
        # Timer fired — invoke the callback
        transcript = self._transcript_getter()
        is_private = self._is_private_getter()
        try:
            await self._callback(transcript, is_private)
        except Exception as exc:
            logger.error("IdleWatcher callback raised: %s", exc)


# ---------------------------------------------------------------------------
# LearningEngine
# ---------------------------------------------------------------------------


# Prompt template for durable-item extraction (Req 8.1)
_EXTRACTION_PROMPT_TEMPLATE = """\
You are a knowledge extraction assistant. Given a conversation transcript, \
extract durable facts and preferences that the user states as applicable \
beyond the current conversation.

Return ONLY a valid JSON array (no extra text, no markdown fences) where each \
element is an object with:
  - "fact_or_preference": a concise statement of the fact or preference
  - "topics": a list of 1-4 lowercase topic strings for retrieval

If no durable items exist, return an empty JSON array: []

Transcript:
{transcript}
"""

# Maximum characters of transcript to send to the LLM
_MAX_TRANSCRIPT_LEN = 8_000

# Session ID format: YYYY-MM-DDTHH-MM
_SESSION_ID_FMT = "%Y-%m-%dT%H-%M"


def _make_session_id(dt: datetime | None = None) -> str:
    """Return a timestamp-based session identifier, e.g. ``"2024-06-01T12-00"``."""
    now = dt or datetime.now(timezone.utc)
    return now.strftime(_SESSION_ID_FMT)


def _topics_overlap(a: list[str], b: list[str]) -> bool:
    """Return True if the two topic lists share at least one common term."""
    set_a = {t.lower().strip() for t in a}
    set_b = {t.lower().strip() for t in b}
    return bool(set_a & set_b)


def _extract_json_from_llm_response(response: str) -> list[dict]:
    """
    Extract a JSON array from a raw LLM response string.

    The LLM may wrap the JSON in markdown fences or include preamble text.
    This function tries several strategies to extract valid JSON.

    Returns an empty list if no valid JSON array can be found.
    """
    text = response.strip()

    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()

    # Try to parse directly
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array with a regex
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return []


class LearningEngine:
    """
    Extracts durable knowledge from concluded conversations and writes it
    into Memory_Brain.

    Parameters
    ----------
    memory_brain:
        The :class:`~core.memory.MemoryBrain` instance to read from and
        write to.
    llm_provider:
        A :class:`~core.model_provider.ModelProvider` for
        ``Capability.LLM`` that will be called to extract durable items
        from transcripts.  When ``None`` the engine always records
        conversations as ``incomplete`` with reason
        ``"no_llm_provider"``.
    idle_timeout_seconds:
        Seconds of inactivity before a conversation is considered ended
        (Req 8.1). Defaults to 300.
    """

    # Idle timeout in seconds before a conversation is considered ended (Req 8.1)
    IDLE_TIMEOUT_SECONDS: int = 300

    def __init__(
        self,
        memory_brain: "MemoryBrain | None" = None,
        llm_provider: "ModelProvider | None" = None,
        idle_timeout_seconds: int = IDLE_TIMEOUT_SECONDS,
    ) -> None:
        self._memory_brain = memory_brain
        self._llm_provider = llm_provider
        self._idle_timeout = idle_timeout_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_conversation_end(
        self,
        transcript: str,
        conversation_id: str,
        is_private: bool = False,
    ) -> LearningReport:
        """
        Process a concluded conversation and extract durable items.

        Subtask 16.1 — privacy gate + extraction + no-extraction path.
        Subtask 16.2 — conflict supersede + per-item atomicity.
        Subtask 16.3 — session tagging.

        Parameters
        ----------
        transcript:
            Full text of the conversation.
        conversation_id:
            Stable identifier for this conversation session.
        is_private:
            ``True`` when the user has designated the conversation as
            private; nothing is written (Req 9.1).

        Returns
        -------
        LearningReport
            Summary of items learned (or skipped/incomplete).
        """
        report = LearningReport(
            conversation_id=conversation_id,
            is_private=is_private,
        )

        # ----------------------------------------------------------------
        # Privacy gate (Req 9.1, Subtask 16.1)
        # ----------------------------------------------------------------
        if is_private:
            report.skipped = True
            return report

        # ----------------------------------------------------------------
        # Durable-item extraction via LLM (Req 8.1, Subtask 16.1)
        # ----------------------------------------------------------------
        extracted = self._extract_items(transcript)

        # No-extraction path (Req 8.6, Subtask 16.1)
        if not extracted:
            report.incomplete = True
            report.incomplete_reason = "no_extractable_items"
            return report

        # ----------------------------------------------------------------
        # Write each item with conflict-supersede + atomicity (Req 8.2, 8.3, 8.7)
        # ----------------------------------------------------------------
        session_id = _make_session_id()

        for item in extracted:
            learned = self._write_item_atomic(
                item=item,
                session_id=session_id,
                report=report,
            )
            if learned is not None:
                report.learned_items.append(learned)

        # If every item failed, mark report incomplete
        if not report.learned_items and report.failed_items:
            report.incomplete = True
            report.incomplete_reason = "all_items_failed_to_write"

        return report

    def recently_learned(self, days: int = 7) -> list[LearnedItem]:
        """
        Return items learned within the last *days* days (Req 8.4).

        Subtask 16.3.

        Parameters
        ----------
        days:
            Window size in days.  Must be in ``[1, 90]`` (inclusive).
            Values outside this range raise :class:`ValueError` (Req 8.4).

        Returns
        -------
        list[LearnedItem]
            Notes tagged with ``source=learned`` created within the
            specified window, ordered from newest to oldest.

        Raises
        ------
        ValueError
            If *days* is outside ``[1, 90]``.
        """
        if not (1 <= days <= 90):
            raise ValueError(
                f"recently_learned: 'days' must be in [1, 90]; got {days}."
            )

        if self._memory_brain is None:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        from core.memory.models import NoteSource

        results: list[LearnedItem] = []
        for note in self._memory_brain.all_notes():
            # Only include notes written by the learning engine
            if note.source != NoteSource.LEARNED:
                continue
            # Exclude superseded notes
            if note.superseded_by is not None:
                continue
            # Apply the time window filter
            note_created = note.created
            if note_created.tzinfo is None:
                note_created = note_created.replace(tzinfo=timezone.utc)
            if note_created < cutoff:
                continue

            results.append(
                LearnedItem(
                    note_id=note.id,
                    fact_or_preference=note.body,
                    topics=list(note.topics),
                    learned_session=note.learned_session or "",
                    created=note_created,
                )
            )

        # Sort newest first
        results.sort(key=lambda x: x.created, reverse=True)
        return results

    def mark_incorrect(self, item_id: str) -> bool:
        """
        Remove a learned item that the user marked as incorrect (Req 8.5).

        Subtask 16.3.

        Parameters
        ----------
        item_id:
            The ``note_id`` of the learned item (as returned by
            :meth:`recently_learned` or :attr:`LearningReport.learned_items`).

        Returns
        -------
        bool
            ``True`` if the note was found and durably removed.
            ``False`` if the note does not exist or the deletion failed.
        """
        if self._memory_brain is None:
            return False

        result = self._memory_brain.forget(item_id)
        return result.success

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_items(self, transcript: str) -> list[_ExtractedItem]:
        """
        Call the LLM to extract durable items from *transcript*.

        Returns an empty list when no LLM provider is configured or the
        LLM returns no items / an unparseable response.
        """
        if self._llm_provider is None:
            return []

        # Truncate very long transcripts to keep within context limits
        truncated = transcript[:_MAX_TRANSCRIPT_LEN]
        prompt = _EXTRACTION_PROMPT_TEMPLATE.format(transcript=truncated)

        try:
            response = self._llm_provider.invoke(prompt)
        except Exception as exc:
            logger.warning("LLM extraction call failed: %s", exc)
            return []

        # The provider may return a dict (StubModelProvider) or a string
        if isinstance(response, dict):
            raw_text = response.get("input", "") or ""
            # Some stubs echo the input; we can't parse facts from that.
            # Treat as no extractable items.
            return []
        elif isinstance(response, str):
            raw_text = response
        else:
            return []

        items_data = _extract_json_from_llm_response(raw_text)

        extracted: list[_ExtractedItem] = []
        for item_data in items_data:
            if not isinstance(item_data, dict):
                continue
            fact = item_data.get("fact_or_preference", "").strip()
            topics_raw = item_data.get("topics", [])
            if not isinstance(topics_raw, list):
                topics_raw = []
            topics = [str(t).lower().strip() for t in topics_raw if t]
            if fact:
                extracted.append(_ExtractedItem(fact_or_preference=fact, topics=topics))

        return extracted

    def _find_conflicting_note(
        self, item: _ExtractedItem
    ) -> "Note | None":
        """
        Search Memory_Brain for an existing note whose topics overlap with
        *item* and that covers the same fact/preference key (Req 8.2).

        Uses topic overlap as the primary signal.  The first matching
        active (non-superseded) note is returned; ``None`` if no conflict
        is found.
        """
        if self._memory_brain is None:
            return None

        from core.memory.models import NoteSource

        for note in self._memory_brain.all_notes():
            # Ignore superseded notes — they are already resolved
            if note.superseded_by is not None:
                continue
            # Only conflict with learned notes (not user-stated context)
            if note.source != NoteSource.LEARNED:
                continue
            # Topics must overlap for this to be a conflict
            if _topics_overlap(note.topics, item.topics):
                return note

        return None

    def _supersede_note(self, old_note_id: str, new_note_id: str) -> None:
        """
        Mark *old_note_id* as superseded by *new_note_id* (Req 8.2, 8.3).

        Reads the existing note, updates its ``superseded_by`` field, and
        re-writes it via Memory_Brain.  Errors are logged but do not
        propagate — a failure here means the old note is not superseded
        yet, but the new note was already written successfully.
        """
        if self._memory_brain is None:
            return

        from core.memory.models import Note, NoteSource

        # Load the note from the vault
        try:
            all_notes = self._memory_brain.all_notes()
        except Exception as exc:
            logger.warning(
                "Could not list notes to supersede '%s': %s", old_note_id, exc
            )
            return

        old_note: Note | None = next(
            (n for n in all_notes if n.id == old_note_id), None
        )
        if old_note is None:
            logger.warning("Note to supersede not found: %s", old_note_id)
            return

        # Update superseded_by and updated timestamp
        old_note.superseded_by = new_note_id
        old_note.updated = datetime.now(timezone.utc)

        # Re-write the note via the vault directly so we update in place.
        # Memory_Brain.remember() always creates a NEW note; we need to
        # overwrite the existing file.  We go through the vault's store()
        # which does an atomic write (same note id = same file path).
        try:
            self._memory_brain._vault.store(old_note)
        except Exception as exc:
            logger.warning(
                "Failed to mark note '%s' as superseded: %s", old_note_id, exc
            )

    def _write_item_atomic(
        self,
        item: _ExtractedItem,
        session_id: str,
        report: LearningReport,
    ) -> "LearnedItem | None":
        """
        Write one extracted item to Memory_Brain atomically (Req 8.7).

        1. Detect any conflicting prior note (Req 8.2).
        2. Write the new note via ``Memory_Brain.remember()``.
        3. Only after a confirmed write, mark the conflicting prior note
           as superseded (Req 8.2, 8.3).
        4. On write failure, record a :class:`FailedItem` in *report* and
           return ``None`` (Req 8.7).

        Parameters
        ----------
        item:
            The extracted item to write.
        session_id:
            The current learning session identifier (e.g. ``"2024-06-01T12-00"``).
        report:
            The report being built; failed items are appended here.

        Returns
        -------
        LearnedItem | None
            The successfully written item, or ``None`` on failure.
        """
        if self._memory_brain is None:
            report.failed_items.append(
                FailedItem(
                    fact_or_preference=item.fact_or_preference,
                    topics=item.topics,
                    reason="no_memory_brain_configured",
                )
            )
            return None

        from core.memory.models import NoteSource

        # 1. Find any conflicting prior note before writing (Req 8.2)
        conflicting_note = self._find_conflicting_note(item)

        # 2. Attempt to write the new note (Req 8.7)
        try:
            result = self._memory_brain.remember(
                body=item.fact_or_preference,
                topics=item.topics,
                source=NoteSource.LEARNED,
                tags=["learned"],
            )
        except Exception as exc:
            # Unexpected exception — no partial note should remain
            report.failed_items.append(
                FailedItem(
                    fact_or_preference=item.fact_or_preference,
                    topics=item.topics,
                    reason=f"write_exception: {exc}",
                )
            )
            return None

        if not result.success:
            # Memory_Brain returned a failure — data is intact (Req 8.7)
            report.failed_items.append(
                FailedItem(
                    fact_or_preference=item.fact_or_preference,
                    topics=item.topics,
                    reason=result.error or "unknown_write_failure",
                )
            )
            return None

        new_note_id = result.note_id  # type: ignore[assignment]

        # Tag the note with learned_session (Req 8.4, Subtask 16.3).
        # We do this by loading and rewriting the newly created note.
        self._tag_learned_session(new_note_id, session_id)

        # 3. Only after successful write, supersede the conflicting note (Req 8.2, 8.3)
        if conflicting_note is not None:
            self._supersede_note(
                old_note_id=conflicting_note.id,
                new_note_id=new_note_id,
            )

        return LearnedItem(
            note_id=new_note_id,
            fact_or_preference=item.fact_or_preference,
            topics=item.topics,
            learned_session=session_id,
        )

    def _tag_learned_session(self, note_id: str, session_id: str) -> None:
        """
        Set the ``learned_session`` field on the note after it has been
        written (Req 8.4).

        Loads the note, updates the field, and re-writes it atomically via
        the vault.  Errors are logged but do not affect the already-confirmed
        write.
        """
        if self._memory_brain is None:
            return

        try:
            all_notes = self._memory_brain.all_notes()
            note = next((n for n in all_notes if n.id == note_id), None)
            if note is None:
                return
            note.learned_session = session_id
            note.updated = datetime.now(timezone.utc)
            self._memory_brain._vault.store(note)
        except Exception as exc:
            logger.warning(
                "Could not tag learned_session on note '%s': %s", note_id, exc
            )
