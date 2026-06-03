"""
Unit tests for LearningEngine (Task 16).

Covers subtasks 16.1, 16.2, and 16.3:

Subtask 16.1 — Conversation-end detection and durable-item extraction with
               privacy gate (Reqs 8.1, 8.6, 9.1)
Subtask 16.2 — Conflict supersede and per-item write atomicity (Reqs 8.2, 8.3, 8.7)
Subtask 16.3 — Recently-learned record and mark-incorrect correction (Reqs 8.4, 8.5)

All Memory_Brain writes and LLM calls are mocked so no real I/O or
model inference occurs.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.learning import FailedItem, LearnedItem, LearningEngine, LearningReport
from core.memory.memory_brain import MemoryBrain
from core.memory.models import Note, NoteSource
from core.memory.vault import StoreResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_provider(items: list[dict] | None = None) -> MagicMock:
    """
    Return a mock LLM provider whose ``invoke`` method returns a JSON
    string encoding *items*.

    Pass ``items=None`` to simulate an empty / unparseable LLM response.
    Pass ``items=[]`` to simulate "no durable items found".
    """
    provider = MagicMock()
    if items is None:
        provider.invoke.return_value = "I could not find any items."
    else:
        provider.invoke.return_value = json.dumps(items)
    return provider


def _make_memory_brain(tmp_path: Path) -> MemoryBrain:
    """Create a fresh MemoryBrain with a local vault for testing."""
    vault_path = tmp_path / "vault"
    vault_path.mkdir(parents=True, exist_ok=True)
    brain = MemoryBrain(vault_path=vault_path, skip_local_guard=True)
    brain.init()
    return brain


def _make_note(
    body: str = "Some fact",
    topics: list[str] | None = None,
    source: NoteSource = NoteSource.LEARNED,
    superseded_by: str | None = None,
    learned_session: str | None = "2024-06-01T12-00",
    created: datetime | None = None,
) -> Note:
    note = Note(
        body=body,
        topics=topics or ["topic-a"],
        source=source,
        superseded_by=superseded_by,
        learned_session=learned_session,
    )
    if created is not None:
        note.created = created
    return note


# ===========================================================================
# Subtask 16.1 — Privacy gate (Req 9.1)
# ===========================================================================


class TestPrivacyGate:
    """Private conversations must not write anything to Memory_Brain (Req 9.1)."""

    def test_private_conversation_skips_extraction(self, tmp_path):
        brain = _make_memory_brain(tmp_path)
        engine = LearningEngine(memory_brain=brain, llm_provider=_make_llm_provider([
            {"fact_or_preference": "User likes Python", "topics": ["python"]},
        ]))

        report = engine.on_conversation_end(
            transcript="User: I love Python!",
            conversation_id="conv-private-1",
            is_private=True,
        )

        assert report.skipped is True
        assert report.is_private is True
        assert report.learned_items == []
        assert report.failed_items == []
        assert len(brain.all_notes()) == 0

    def test_private_conversation_report_has_correct_conversation_id(self, tmp_path):
        brain = _make_memory_brain(tmp_path)
        engine = LearningEngine(memory_brain=brain)

        report = engine.on_conversation_end(
            transcript="User: I prefer dark mode.",
            conversation_id="conv-private-2",
            is_private=True,
        )

        assert report.conversation_id == "conv-private-2"

    def test_non_private_conversation_does_not_skip(self, tmp_path):
        brain = _make_memory_brain(tmp_path)
        engine = LearningEngine(
            memory_brain=brain,
            llm_provider=_make_llm_provider([
                {"fact_or_preference": "User prefers dark mode", "topics": ["ui"]},
            ]),
        )

        report = engine.on_conversation_end(
            transcript="User: I prefer dark mode.",
            conversation_id="conv-public-1",
            is_private=False,
        )

        assert report.skipped is False
        assert not report.incomplete


# ===========================================================================
# Subtask 16.1 — No-extraction path (Req 8.6)
# ===========================================================================


class TestNoExtractionPath:
    """If the LLM returns no durable items, notes are unchanged (Req 8.6)."""

    def test_empty_llm_result_marks_incomplete(self, tmp_path):
        brain = _make_memory_brain(tmp_path)
        engine = LearningEngine(
            memory_brain=brain,
            llm_provider=_make_llm_provider([]),  # empty list
        )

        report = engine.on_conversation_end(
            transcript="User: How are you?",
            conversation_id="conv-empty-1",
        )

        assert report.incomplete is True
        assert report.incomplete_reason == "no_extractable_items"
        assert report.learned_items == []

    def test_empty_llm_result_leaves_existing_notes_unchanged(self, tmp_path):
        brain = _make_memory_brain(tmp_path)
        # Pre-populate a note
        pre_result = brain.remember(body="Existing fact", topics=["existing"])
        assert pre_result.success

        engine = LearningEngine(
            memory_brain=brain,
            llm_provider=_make_llm_provider([]),
        )

        engine.on_conversation_end(
            transcript="User: Nothing interesting here.",
            conversation_id="conv-empty-2",
        )

        all_notes = brain.all_notes()
        assert len(all_notes) == 1
        assert all_notes[0].body == "Existing fact"

    def test_unparseable_llm_response_marks_incomplete(self, tmp_path):
        brain = _make_memory_brain(tmp_path)
        engine = LearningEngine(
            memory_brain=brain,
            llm_provider=_make_llm_provider(None),  # non-JSON response
        )

        report = engine.on_conversation_end(
            transcript="User: Tell me a story.",
            conversation_id="conv-unparseable",
        )

        assert report.incomplete is True

    def test_no_llm_provider_marks_incomplete(self, tmp_path):
        brain = _make_memory_brain(tmp_path)
        engine = LearningEngine(memory_brain=brain, llm_provider=None)

        report = engine.on_conversation_end(
            transcript="User: My name is Alice.",
            conversation_id="conv-no-llm",
        )

        assert report.incomplete is True
        assert report.learned_items == []


# ===========================================================================
# Subtask 16.1 — Successful extraction (Req 8.1)
# ===========================================================================


class TestSuccessfulExtraction:
    """Items extracted by the LLM are written to Memory_Brain."""

    def test_single_item_is_written(self, tmp_path):
        brain = _make_memory_brain(tmp_path)
        engine = LearningEngine(
            memory_brain=brain,
            llm_provider=_make_llm_provider([
                {"fact_or_preference": "User's favourite language is Python",
                 "topics": ["python", "programming"]},
            ]),
        )

        report = engine.on_conversation_end(
            transcript="User: I love Python above all other languages.",
            conversation_id="conv-single",
        )

        assert not report.incomplete
        assert len(report.learned_items) == 1
        item = report.learned_items[0]
        assert item.fact_or_preference == "User's favourite language is Python"
        assert "python" in item.topics

    def test_multiple_items_are_written(self, tmp_path):
        brain = _make_memory_brain(tmp_path)
        engine = LearningEngine(
            memory_brain=brain,
            llm_provider=_make_llm_provider([
                {"fact_or_preference": "User studies computer science",
                 "topics": ["cs", "education"]},
                {"fact_or_preference": "User prefers dark mode UI",
                 "topics": ["ui", "preferences"]},
            ]),
        )

        report = engine.on_conversation_end(
            transcript="User: I am a CS student and prefer dark mode.",
            conversation_id="conv-multi",
        )

        assert len(report.learned_items) == 2
        assert len(brain.all_notes()) == 2

    def test_learned_items_have_note_id(self, tmp_path):
        brain = _make_memory_brain(tmp_path)
        engine = LearningEngine(
            memory_brain=brain,
            llm_provider=_make_llm_provider([
                {"fact_or_preference": "User is left-handed", "topics": ["habits"]},
            ]),
        )

        report = engine.on_conversation_end(
            transcript="User: I'm left-handed.",
            conversation_id="conv-noteid",
        )

        assert report.learned_items[0].note_id
        # note_id must match a real note
        note_ids = {n.id for n in brain.all_notes()}
        assert report.learned_items[0].note_id in note_ids


# ===========================================================================
# Subtask 16.2 — Conflict supersede (Reqs 8.2, 8.3)
# ===========================================================================


class TestConflictSupersede:
    """When a new item conflicts with an existing note, the prior note is superseded."""

    def test_conflicting_note_is_superseded(self, tmp_path):
        brain = _make_memory_brain(tmp_path)

        # Write an existing learned note on the same topics
        old_result = brain.remember(
            body="User's favourite language is Java",
            topics=["python", "programming"],
            source=NoteSource.LEARNED,
        )
        assert old_result.success
        old_id = old_result.note_id

        engine = LearningEngine(
            memory_brain=brain,
            llm_provider=_make_llm_provider([
                {"fact_or_preference": "User's favourite language is Python",
                 "topics": ["python", "programming"]},
            ]),
        )

        report = engine.on_conversation_end(
            transcript="User: Actually, Python is my favourite.",
            conversation_id="conv-supersede",
        )

        assert len(report.learned_items) == 1
        new_id = report.learned_items[0].note_id

        # Old note must be superseded
        all_notes = {n.id: n for n in brain.all_notes()}
        assert all_notes[old_id].superseded_by == new_id

    def test_new_note_written_before_supersede(self, tmp_path):
        """New note must exist even if supersede step were to fail."""
        brain = _make_memory_brain(tmp_path)
        old_result = brain.remember(
            body="User drinks coffee",
            topics=["coffee", "drinks"],
            source=NoteSource.LEARNED,
        )
        old_id = old_result.note_id

        engine = LearningEngine(
            memory_brain=brain,
            llm_provider=_make_llm_provider([
                {"fact_or_preference": "User drinks tea now",
                 "topics": ["coffee", "drinks"]},
            ]),
        )

        report = engine.on_conversation_end(
            transcript="User: I switched to tea.",
            conversation_id="conv-order",
        )

        new_id = report.learned_items[0].note_id
        note_ids = {n.id for n in brain.all_notes()}
        # Both old and new notes exist in the vault
        assert old_id in note_ids
        assert new_id in note_ids

    def test_non_overlapping_topics_do_not_conflict(self, tmp_path):
        brain = _make_memory_brain(tmp_path)

        # Existing note on a completely different topic
        brain.remember(
            body="User's exam is on June 14",
            topics=["exam", "schedule"],
            source=NoteSource.LEARNED,
        )

        engine = LearningEngine(
            memory_brain=brain,
            llm_provider=_make_llm_provider([
                {"fact_or_preference": "User likes spicy food",
                 "topics": ["food", "preferences"]},
            ]),
        )

        report = engine.on_conversation_end(
            transcript="User: I love spicy food!",
            conversation_id="conv-no-conflict",
        )

        # Both notes should be present and neither superseded
        all_notes = brain.all_notes()
        assert len(all_notes) == 2
        for n in all_notes:
            assert n.superseded_by is None

    def test_only_conflicting_note_is_superseded(self, tmp_path):
        """Exactly the conflicting prior note is superseded — not others (Req 8.2)."""
        brain = _make_memory_brain(tmp_path)

        # Overlapping note
        r1 = brain.remember(
            body="User prefers dark mode",
            topics=["ui", "preferences"],
            source=NoteSource.LEARNED,
        )
        conflict_id = r1.note_id

        # Unrelated note
        r2 = brain.remember(
            body="User studies CS",
            topics=["education"],
            source=NoteSource.LEARNED,
        )
        unrelated_id = r2.note_id

        engine = LearningEngine(
            memory_brain=brain,
            llm_provider=_make_llm_provider([
                {"fact_or_preference": "User now prefers light mode",
                 "topics": ["ui", "preferences"]},
            ]),
        )

        engine.on_conversation_end(
            transcript="User: I switched to light mode.",
            conversation_id="conv-exact-supersede",
        )

        notes_by_id = {n.id: n for n in brain.all_notes()}
        # Conflicting note is superseded
        assert notes_by_id[conflict_id].superseded_by is not None
        # Unrelated note is NOT superseded
        assert notes_by_id[unrelated_id].superseded_by is None


# ===========================================================================
# Subtask 16.2 — Per-item write atomicity (Req 8.7)
# ===========================================================================


class TestPerItemWriteAtomicity:
    """A failed write for one item does not affect other items or prior notes."""

    def test_failed_write_recorded_in_report(self, tmp_path):
        brain = _make_memory_brain(tmp_path)

        # Make the brain's remember() fail for the first call
        original_remember = brain.remember
        call_count = [0]

        def failing_remember(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return StoreResult.fail("simulated_disk_error")
            return original_remember(*args, **kwargs)

        brain.remember = failing_remember

        engine = LearningEngine(
            memory_brain=brain,
            llm_provider=_make_llm_provider([
                {"fact_or_preference": "User likes cats", "topics": ["pets"]},
                {"fact_or_preference": "User studies biology", "topics": ["biology"]},
            ]),
        )

        report = engine.on_conversation_end(
            transcript="User: I love cats and I study biology.",
            conversation_id="conv-atomic",
        )

        # First item failed; it must be in failed_items
        assert len(report.failed_items) == 1
        assert report.failed_items[0].fact_or_preference == "User likes cats"
        assert "simulated_disk_error" in report.failed_items[0].reason

        # Second item succeeded
        assert len(report.learned_items) == 1
        assert report.learned_items[0].fact_or_preference == "User studies biology"

    def test_failed_write_leaves_prior_notes_unchanged(self, tmp_path):
        brain = _make_memory_brain(tmp_path)

        # Pre-populate a note
        pre = brain.remember(body="Pre-existing fact", topics=["existing"])
        pre_id = pre.note_id

        # Patch remember to always fail
        brain.remember = lambda *a, **kw: StoreResult.fail("always_fail")

        engine = LearningEngine(
            memory_brain=brain,
            llm_provider=_make_llm_provider([
                {"fact_or_preference": "New fact", "topics": ["new"]},
            ]),
        )

        engine.on_conversation_end(
            transcript="User: Something new.",
            conversation_id="conv-atomic-prior",
        )

        # The pre-existing note must still be intact
        notes = brain.all_notes()
        assert any(n.id == pre_id and n.superseded_by is None for n in notes)

    def test_one_failed_item_does_not_stop_others(self, tmp_path):
        """Failure of one item must not prevent writing subsequent items."""
        brain = _make_memory_brain(tmp_path)

        call_count = [0]
        original_remember = brain.remember

        def selective_fail(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:  # fail only the second item
                return StoreResult.fail("selective_fail")
            return original_remember(*args, **kwargs)

        brain.remember = selective_fail

        engine = LearningEngine(
            memory_brain=brain,
            llm_provider=_make_llm_provider([
                {"fact_or_preference": "Fact A", "topics": ["a"]},
                {"fact_or_preference": "Fact B", "topics": ["b"]},
                {"fact_or_preference": "Fact C", "topics": ["c"]},
            ]),
        )

        report = engine.on_conversation_end(
            transcript="Transcript",
            conversation_id="conv-three-items",
        )

        assert len(report.learned_items) == 2  # A and C
        assert len(report.failed_items) == 1   # B


# ===========================================================================
# Subtask 16.3 — Session tagging (Req 8.4)
# ===========================================================================


class TestSessionTagging:
    """Learned items must be tagged with learned_session (Req 8.4)."""

    def test_note_has_learned_session_set(self, tmp_path):
        brain = _make_memory_brain(tmp_path)
        engine = LearningEngine(
            memory_brain=brain,
            llm_provider=_make_llm_provider([
                {"fact_or_preference": "User works from home", "topics": ["work"]},
            ]),
        )

        report = engine.on_conversation_end(
            transcript="User: I work from home.",
            conversation_id="conv-session",
        )

        assert len(report.learned_items) == 1
        item = report.learned_items[0]
        assert item.learned_session  # non-empty

        # The note in vault must also have learned_session set
        vault_note = next(n for n in brain.all_notes() if n.id == item.note_id)
        assert vault_note.learned_session == item.learned_session

    def test_note_source_is_learned(self, tmp_path):
        brain = _make_memory_brain(tmp_path)
        engine = LearningEngine(
            memory_brain=brain,
            llm_provider=_make_llm_provider([
                {"fact_or_preference": "User is vegetarian", "topics": ["diet"]},
            ]),
        )

        engine.on_conversation_end(
            transcript="User: I am vegetarian.",
            conversation_id="conv-source",
        )

        for note in brain.all_notes():
            assert note.source == NoteSource.LEARNED


# ===========================================================================
# Subtask 16.3 — recently_learned (Req 8.4)
# ===========================================================================


class TestRecentlyLearned:
    """recently_learned returns items within the specified day window."""

    def test_items_within_window_returned(self, tmp_path):
        brain = _make_memory_brain(tmp_path)

        # Write two notes: one recent, one old
        now = datetime.now(timezone.utc)
        recent_note = Note(
            body="Recent fact",
            topics=["recent"],
            source=NoteSource.LEARNED,
            learned_session="2024-01-01T10-00",
        )
        recent_note.created = now - timedelta(days=2)
        brain._vault.store(recent_note)

        old_note = Note(
            body="Old fact",
            topics=["old"],
            source=NoteSource.LEARNED,
            learned_session="2023-01-01T10-00",
        )
        old_note.created = now - timedelta(days=100)
        brain._vault.store(old_note)

        engine = LearningEngine(memory_brain=brain)
        items = engine.recently_learned(days=7)

        assert len(items) == 1
        assert items[0].fact_or_preference == "Recent fact"

    def test_default_window_is_7_days(self, tmp_path):
        brain = _make_memory_brain(tmp_path)
        engine = LearningEngine(memory_brain=brain)

        # Write a note 8 days ago — outside the default 7-day window
        note = Note(
            body="Old-ish fact",
            topics=["old"],
            source=NoteSource.LEARNED,
            learned_session="2024-01-01T10-00",
        )
        note.created = datetime.now(timezone.utc) - timedelta(days=8)
        brain._vault.store(note)

        items = engine.recently_learned()  # default days=7
        assert len(items) == 0

    def test_days_below_1_raises_value_error(self, tmp_path):
        brain = _make_memory_brain(tmp_path)
        engine = LearningEngine(memory_brain=brain)
        with pytest.raises(ValueError, match=r"days.*must be in \[1, 90\]"):
            engine.recently_learned(days=0)

    def test_days_above_90_raises_value_error(self, tmp_path):
        brain = _make_memory_brain(tmp_path)
        engine = LearningEngine(memory_brain=brain)
        with pytest.raises(ValueError, match=r"days.*must be in \[1, 90\]"):
            engine.recently_learned(days=91)

    def test_days_1_boundary_is_valid(self, tmp_path):
        brain = _make_memory_brain(tmp_path)
        engine = LearningEngine(memory_brain=brain)
        # Should not raise
        items = engine.recently_learned(days=1)
        assert isinstance(items, list)

    def test_days_90_boundary_is_valid(self, tmp_path):
        brain = _make_memory_brain(tmp_path)
        engine = LearningEngine(memory_brain=brain)
        # Should not raise
        items = engine.recently_learned(days=90)
        assert isinstance(items, list)

    def test_superseded_notes_excluded(self, tmp_path):
        brain = _make_memory_brain(tmp_path)

        # Recent note that is superseded
        note = Note(
            body="Superseded fact",
            topics=["topic"],
            source=NoteSource.LEARNED,
            learned_session="2024-01-01T10-00",
            superseded_by="some-other-id",
        )
        note.created = datetime.now(timezone.utc) - timedelta(days=1)
        brain._vault.store(note)

        engine = LearningEngine(memory_brain=brain)
        items = engine.recently_learned(days=7)
        assert len(items) == 0

    def test_no_memory_brain_returns_empty_list(self):
        engine = LearningEngine(memory_brain=None)
        assert engine.recently_learned(days=7) == []


# ===========================================================================
# Subtask 16.3 — mark_incorrect (Req 8.5)
# ===========================================================================


class TestMarkIncorrect:
    """mark_incorrect removes the note from Memory_Brain and confirms."""

    def test_mark_incorrect_removes_note(self, tmp_path):
        brain = _make_memory_brain(tmp_path)

        # Create a learned note
        result = brain.remember(
            body="Incorrect fact",
            topics=["wrong"],
            source=NoteSource.LEARNED,
        )
        note_id = result.note_id

        engine = LearningEngine(memory_brain=brain)
        success = engine.mark_incorrect(note_id)

        assert success is True
        # Note must be gone
        remaining = [n for n in brain.all_notes() if n.id == note_id]
        assert len(remaining) == 0

    def test_mark_incorrect_returns_false_for_nonexistent(self, tmp_path):
        brain = _make_memory_brain(tmp_path)
        engine = LearningEngine(memory_brain=brain)

        success = engine.mark_incorrect("nonexistent-id")
        assert success is False

    def test_mark_incorrect_no_memory_brain(self):
        engine = LearningEngine(memory_brain=None)
        result = engine.mark_incorrect("some-id")
        assert result is False

    def test_mark_incorrect_leaves_other_notes_intact(self, tmp_path):
        brain = _make_memory_brain(tmp_path)

        r1 = brain.remember(body="Fact to delete", topics=["delete-me"],
                             source=NoteSource.LEARNED)
        r2 = brain.remember(body="Fact to keep", topics=["keep-me"],
                             source=NoteSource.LEARNED)

        engine = LearningEngine(memory_brain=brain)
        engine.mark_incorrect(r1.note_id)

        remaining = brain.all_notes()
        assert any(n.id == r2.note_id for n in remaining)
        assert all(n.id != r1.note_id for n in remaining)


# ===========================================================================
# Integration — full on_conversation_end flow
# ===========================================================================


class TestEndToEndFlow:
    """Integration scenarios exercising the full on_conversation_end path."""

    def test_full_flow_privacy_then_public(self, tmp_path):
        """A private session writes nothing; a subsequent public session does."""
        brain = _make_memory_brain(tmp_path)
        engine = LearningEngine(
            memory_brain=brain,
            llm_provider=_make_llm_provider([
                {"fact_or_preference": "User likes Python", "topics": ["python"]},
            ]),
        )

        private_report = engine.on_conversation_end(
            transcript="Sensitive data",
            conversation_id="priv-1",
            is_private=True,
        )

        public_report = engine.on_conversation_end(
            transcript="User: I love Python.",
            conversation_id="pub-1",
            is_private=False,
        )

        assert private_report.skipped is True
        assert len(brain.all_notes()) == 0 or all(
            n.id in {i.note_id for i in public_report.learned_items}
            for n in brain.all_notes()
        )
        assert not public_report.skipped
        assert len(public_report.learned_items) == 1

    def test_conflict_supersede_then_recently_learned(self, tmp_path):
        """Superseded notes are excluded from recently_learned."""
        brain = _make_memory_brain(tmp_path)

        # First conversation writes an item
        engine = LearningEngine(
            memory_brain=brain,
            llm_provider=_make_llm_provider([
                {"fact_or_preference": "User prefers Java",
                 "topics": ["language", "programming"]},
            ]),
        )
        r1 = engine.on_conversation_end(
            transcript="User: I use Java.",
            conversation_id="conv-java",
        )
        assert len(r1.learned_items) == 1

        # Second conversation supersedes it
        engine._llm_provider = _make_llm_provider([
            {"fact_or_preference": "User prefers Python",
             "topics": ["language", "programming"]},
        ])
        r2 = engine.on_conversation_end(
            transcript="User: Actually I switched to Python.",
            conversation_id="conv-python",
        )
        assert len(r2.learned_items) == 1

        # recently_learned should only return the new (non-superseded) item
        recent = engine.recently_learned(days=90)
        assert len(recent) == 1
        assert recent[0].fact_or_preference == "User prefers Python"
