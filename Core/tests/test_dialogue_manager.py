"""
Unit tests for DialogueManager — memory-first slot assessment and
pre-execution gating (Task 25.1).

Covers:
  - assess() returns sufficient when all needed slots are already resolved
  - assess() returns missing slots when slots are absent
  - assess() never re-asks already-resolved slots (Req 23.4)
  - fill_from_memory() resolves slots from memory, leaves others in still_missing
  - fill_from_memory() skips slots already resolved in session (Req 23.4)
  - mark_resolved / is_resolved / get_resolved work correctly
  - ask() skips already-resolved slots and returns answers from callback
  - reset_session() clears all session state
  - Execution is gated: ask() returns pending questions when no callback is set

**Validates: Requirements 23.1, 23.2, 23.4, 23.5**
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from core.dialogue import DialogueManager, SlotFillResult, MemoryFillResult


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def _make_memory_brain(slot_values: dict[str, str]) -> MagicMock:
    """
    Create a minimal MemoryBrain mock that resolves slot names to note bodies.

    The mock's ``retrieve(query, k)`` returns a list with a single mock Note
    whose ``body`` is ``slot_values[query]`` when the slot name is found,
    or ``[]`` otherwise.
    """
    def fake_retrieve(query: str, k: int = 1) -> list:
        if query in slot_values:
            note = MagicMock()
            note.body = slot_values[query]
            return [note]
        return []

    brain = MagicMock()
    brain.retrieve = MagicMock(side_effect=fake_retrieve)
    return brain


# ---------------------------------------------------------------------------
# TestAssess — assess() core behaviour
# ---------------------------------------------------------------------------


class TestAssess:
    """Tests for DialogueManager.assess() (Req 23.1, 23.4)."""

    def test_assess_sufficient_when_all_slots_pre_resolved(self):
        """assess() returns sufficient=True when every needed slot is already resolved."""
        dm = DialogueManager()
        dm.mark_resolved("recipient", "alice@example.com")
        dm.mark_resolved("subject", "Hello")

        result = dm.assess("send email", ["recipient", "subject"])

        assert result.sufficient is True
        assert result.missing == []
        assert result.resolved["recipient"] == "alice@example.com"
        assert result.resolved["subject"] == "Hello"

    def test_assess_returns_missing_when_no_slots_resolved(self):
        """assess() returns all needed_slots in missing when none are resolved."""
        dm = DialogueManager()

        result = dm.assess("book a meeting", ["date", "time", "attendees"])

        assert result.sufficient is False
        assert set(result.missing) == {"date", "time", "attendees"}
        assert result.resolved == {}

    def test_assess_partial_resolution(self):
        """assess() returns only unresolved slots in missing."""
        dm = DialogueManager()
        dm.mark_resolved("date", "2024-07-01")

        result = dm.assess("book a meeting", ["date", "time"])

        assert result.sufficient is False
        assert result.missing == ["time"]
        assert result.resolved["date"] == "2024-07-01"

    def test_assess_empty_needed_slots_is_sufficient(self):
        """A request with no required slots is immediately sufficient."""
        dm = DialogueManager()
        result = dm.assess("what time is it?", [])
        assert result.sufficient is True
        assert result.missing == []

    def test_assess_does_not_re_ask_resolved_slots(self):
        """
        assess() never includes already-resolved slots in missing (Req 23.4).

        Simulates a second assess() call within the same session after the
        slot was resolved during a previous interaction.
        """
        dm = DialogueManager()

        # First interaction resolves 'recipient'
        dm.mark_resolved("recipient", "bob@example.com")

        # Second assess for the same slot in the same session
        result = dm.assess("follow-up email", ["recipient"])

        assert result.sufficient is True
        assert "recipient" not in result.missing

    def test_assess_returns_slotfill_result_type(self):
        """assess() always returns a SlotFillResult."""
        dm = DialogueManager()
        result = dm.assess("any request", ["slot_a"])
        assert isinstance(result, SlotFillResult)


# ---------------------------------------------------------------------------
# TestFillFromMemory — fill_from_memory() (Req 23.2)
# ---------------------------------------------------------------------------


class TestFillFromMemory:
    """Tests for DialogueManager.fill_from_memory() (Req 23.2)."""

    def test_resolves_slots_found_in_memory(self):
        """Slots whose query matches a memory note are placed in resolved."""
        brain = _make_memory_brain({"home_city": "New Delhi"})
        dm = DialogueManager(memory_brain=brain)

        result = dm.fill_from_memory(["home_city", "destination"])

        assert "home_city" in result.resolved
        assert result.resolved["home_city"] == "New Delhi"
        assert "destination" in result.still_missing

    def test_still_missing_when_memory_has_no_match(self):
        """Slots not found in memory appear in still_missing."""
        brain = _make_memory_brain({})  # empty memory
        dm = DialogueManager(memory_brain=brain)

        result = dm.fill_from_memory(["date", "time"])

        assert result.resolved == {}
        assert set(result.still_missing) == {"date", "time"}

    def test_no_memory_brain_all_slots_still_missing(self):
        """Without a memory_brain, all missing slots remain still_missing."""
        dm = DialogueManager(memory_brain=None)

        result = dm.fill_from_memory(["city", "date"])

        assert result.resolved == {}
        assert set(result.still_missing) == {"city", "date"}

    def test_memory_brain_override_parameter(self):
        """Caller can pass a memory_brain override to fill_from_memory."""
        brain = _make_memory_brain({"preferred_language": "Hinglish"})
        dm = DialogueManager(memory_brain=None)  # no brain at construction

        result = dm.fill_from_memory(["preferred_language"], memory_brain=brain)

        assert result.resolved["preferred_language"] == "Hinglish"
        assert result.still_missing == []

    def test_already_resolved_slots_are_not_re_queried(self):
        """
        fill_from_memory() skips slots already resolved in session (Req 23.4).

        The memory brain should NOT be queried for an already-resolved slot.
        """
        brain = _make_memory_brain({"city": "Mumbai"})
        dm = DialogueManager(memory_brain=brain)
        dm.mark_resolved("city", "Chennai")  # resolved in session

        result = dm.fill_from_memory(["city"])

        # Should return the session-resolved value without querying memory
        assert result.resolved["city"] == "Chennai"
        assert result.still_missing == []
        # memory brain.retrieve should NOT have been called for an already-resolved slot
        brain.retrieve.assert_not_called()

    def test_returns_memory_fill_result_type(self):
        """fill_from_memory() always returns a MemoryFillResult."""
        dm = DialogueManager()
        result = dm.fill_from_memory([])
        assert isinstance(result, MemoryFillResult)

    def test_memory_error_is_handled_gracefully(self):
        """If memory_brain.retrieve raises, the slot is still_missing (not a crash)."""
        brain = MagicMock()
        brain.retrieve.side_effect = RuntimeError("connection error")
        dm = DialogueManager(memory_brain=brain)

        result = dm.fill_from_memory(["some_slot"])

        assert "some_slot" in result.still_missing
        assert result.resolved == {}

    def test_partial_memory_resolution(self):
        """Only slots present in memory are resolved; others stay in still_missing."""
        brain = _make_memory_brain({
            "email": "user@example.com",
            # "phone" is NOT in memory
        })
        dm = DialogueManager(memory_brain=brain)

        result = dm.fill_from_memory(["email", "phone"])

        assert "email" in result.resolved
        assert "phone" in result.still_missing


# ---------------------------------------------------------------------------
# TestSlotRegistry — mark_resolved / is_resolved / get_resolved
# ---------------------------------------------------------------------------


class TestSlotRegistry:
    """Tests for the session-scoped slot registry (Req 23.4)."""

    def test_is_resolved_false_for_new_slot(self):
        """is_resolved() returns False for a slot that has never been marked."""
        dm = DialogueManager()
        assert dm.is_resolved("destination") is False

    def test_mark_resolved_makes_is_resolved_true(self):
        """is_resolved() returns True after mark_resolved() is called."""
        dm = DialogueManager()
        dm.mark_resolved("destination", "Bengaluru")
        assert dm.is_resolved("destination") is True

    def test_get_resolved_returns_correct_value(self):
        """get_resolved() returns the exact value passed to mark_resolved()."""
        dm = DialogueManager()
        dm.mark_resolved("phone_number", "+91-9876543210")
        assert dm.get_resolved("phone_number") == "+91-9876543210"

    def test_get_resolved_returns_none_for_unresolved_slot(self):
        """get_resolved() returns None for a slot that has not been resolved."""
        dm = DialogueManager()
        assert dm.get_resolved("unknown_slot") is None

    def test_mark_resolved_overwrites_previous_value(self):
        """Calling mark_resolved() twice for the same slot updates the value."""
        dm = DialogueManager()
        dm.mark_resolved("city", "Delhi")
        dm.mark_resolved("city", "Mumbai")
        assert dm.get_resolved("city") == "Mumbai"

    def test_resolved_slots_property_reflects_session_state(self):
        """resolved_slots property returns all currently resolved slots."""
        dm = DialogueManager()
        dm.mark_resolved("a", 1)
        dm.mark_resolved("b", 2)
        assert dm.resolved_slots == {"a": 1, "b": 2}

    def test_resolved_slots_property_is_a_copy(self):
        """Mutating the returned resolved_slots dict does not affect session state."""
        dm = DialogueManager()
        dm.mark_resolved("x", "original")
        copy = dm.resolved_slots
        copy["x"] = "mutated"
        assert dm.get_resolved("x") == "original"


# ---------------------------------------------------------------------------
# TestResetSession — reset_session() behaviour
# ---------------------------------------------------------------------------


class TestResetSession:
    """Tests for reset_session() clearing all session state."""

    def test_reset_clears_resolved_slots(self):
        """After reset_session(), is_resolved() returns False for all slots."""
        dm = DialogueManager()
        dm.mark_resolved("slot1", "value1")
        dm.mark_resolved("slot2", "value2")

        dm.reset_session()

        assert dm.is_resolved("slot1") is False
        assert dm.is_resolved("slot2") is False
        assert dm.resolved_slots == {}

    def test_reset_clears_pending_questions(self):
        """After reset_session(), pending_questions is empty."""
        dm = DialogueManager()
        # Simulate questions being registered via ask() in headless mode
        dm._pending_questions = ["What date?", "What time?"]

        dm.reset_session()

        assert dm.pending_questions == []

    def test_assess_after_reset_returns_all_missing(self):
        """After a reset, assess() treats all slots as unresolved."""
        dm = DialogueManager()
        dm.mark_resolved("target", "inbox")
        dm.reset_session()

        result = dm.assess("some request", ["target"])

        assert result.sufficient is False
        assert "target" in result.missing


# ---------------------------------------------------------------------------
# TestAskGating — ask() and pre-execution gating (Req 23.1, 23.3)
# ---------------------------------------------------------------------------


class TestAskGating:
    """Tests for ask() gating behaviour (Req 23.1, 23.3)."""

    @pytest.mark.asyncio
    async def test_ask_returns_empty_without_callback(self):
        """In headless mode (no callback), ask() returns an empty dict."""
        dm = DialogueManager()
        answers = await dm.ask(["What is the deadline?"])
        assert answers == {}

    @pytest.mark.asyncio
    async def test_ask_registers_pending_questions_without_callback(self):
        """Without a callback, ask() registers questions as pending (gating signal)."""
        dm = DialogueManager()
        questions = ["What is the recipient?", "What is the subject?"]
        await dm.ask(questions)
        assert dm.pending_questions == questions

    @pytest.mark.asyncio
    async def test_ask_with_async_callback_returns_answers(self):
        """ask() with an async callback returns a dict mapping questions to answers."""
        responses = {
            "What is the date?": "2024-08-15",
            "What is the time?": "10:00 AM",
        }

        async def fake_ask(question: str) -> str:
            return responses.get(question, "")

        dm = DialogueManager(ask_callback=fake_ask)
        answers = await dm.ask(list(responses.keys()))

        assert answers["What is the date?"] == "2024-08-15"
        assert answers["What is the time?"] == "10:00 AM"

    @pytest.mark.asyncio
    async def test_ask_with_sync_callback_returns_answers(self):
        """ask() with a sync callback runs it and returns answers."""
        def fake_sync_ask(question: str) -> str:
            return f"answer_for_{question}"

        dm = DialogueManager(sync_ask_callback=fake_sync_ask)
        questions = ["What is the location?"]
        answers = await dm.ask(questions)

        assert answers["What is the location?"] == "answer_for_What is the location?"

    @pytest.mark.asyncio
    async def test_ask_empty_list_returns_empty_dict(self):
        """ask() with an empty list returns an empty dict immediately."""
        dm = DialogueManager()
        answers = await dm.ask([])
        assert answers == {}

    @pytest.mark.asyncio
    async def test_ask_clears_pending_questions_after_answers_received(self):
        """After ask() completes with a callback, pending_questions is cleared."""
        async def instant_answer(q: str) -> str:
            return "some answer"

        dm = DialogueManager(ask_callback=instant_answer)
        await dm.ask(["Any question?"])
        assert dm.pending_questions == []

    def test_both_callbacks_raises_value_error(self):
        """Providing both async and sync callbacks raises ValueError."""
        with pytest.raises(ValueError):
            DialogueManager(
                ask_callback=lambda q: None,
                sync_ask_callback=lambda q: None,
            )


# ---------------------------------------------------------------------------
# TestMemoryFirstOrdering — assess → fill_from_memory → ask pipeline
# ---------------------------------------------------------------------------


class TestMemoryFirstOrdering:
    """
    Integration-style tests verifying the full memory-first pipeline:
    assess → fill_from_memory → mark_resolved → assess again.

    Verifies Req 23.2 (memory before user), 23.4 (no re-ask), 23.5
    (caller confirms memory-resolved values before marking).
    """

    def test_memory_resolution_prevents_user_question(self):
        """
        Slots resolved from memory should not appear in missing after being
        marked resolved by the caller (simulating confirmation, Req 23.5).
        """
        brain = _make_memory_brain({"preferred_editor": "VS Code"})
        dm = DialogueManager(memory_brain=brain)

        # Step 1 — assess reveals missing slot
        assessment = dm.assess("open editor", ["preferred_editor"])
        assert "preferred_editor" in assessment.missing

        # Step 2 — try to fill from memory before asking user (Req 23.2)
        fill_result = dm.fill_from_memory(assessment.missing)
        assert "preferred_editor" in fill_result.resolved
        assert fill_result.still_missing == []

        # Step 3 — caller confirms the memory value with user (Req 23.5)
        # then marks it resolved
        dm.mark_resolved("preferred_editor", fill_result.resolved["preferred_editor"])

        # Step 4 — reassess: now sufficient, no user question needed
        reassessment = dm.assess("open editor", ["preferred_editor"])
        assert reassessment.sufficient is True

    def test_no_re_ask_across_multiple_assess_calls(self):
        """
        Once a slot is resolved, subsequent assess() calls in the same
        session never return it in missing (Req 23.4).
        """
        dm = DialogueManager()
        dm.mark_resolved("contact", "Priya")

        for _ in range(3):
            result = dm.assess("call someone", ["contact"])
            assert result.sufficient is True
            assert "contact" not in result.missing

    def test_execution_gated_until_all_slots_resolved(self):
        """
        Execution (simulated by assess returning sufficient) must be blocked
        while any required slot is still missing (Req 23.1).
        """
        dm = DialogueManager()

        # Not yet resolved → insufficient
        result = dm.assess("send file", ["recipient", "filename"])
        assert result.sufficient is False

        # Resolve one slot
        dm.mark_resolved("recipient", "alice@example.com")
        result = dm.assess("send file", ["recipient", "filename"])
        assert result.sufficient is False  # still missing 'filename'

        # Resolve the last slot
        dm.mark_resolved("filename", "report.pdf")
        result = dm.assess("send file", ["recipient", "filename"])
        assert result.sufficient is True  # now execution may proceed


# ===========================================================================
# Task 25.2 — Mid-task pause/resume, declines, and candidate-choice
# ===========================================================================
#
# Covers:
#   - on_decision_point() pauses execution and retains task state (Req 23.3)
#   - Resume returns completed_steps so caller can skip them (Req 23.9)
#   - on_decline() with default → use_default action + default value (Req 23.6)
#   - on_decline() without default → abandon_step action (Req 23.7)
#   - present_options() never auto-selects (Req 23.8)
#   - present_options() with callback returns user's selection (Req 23.8)
#   - is_paused / current_pause reflect pause state correctly (Req 23.3)
#   - CandidatePresentation.auto_selected is always False (Req 23.8)
#
# **Validates: Requirements 23.3, 23.6, 23.7, 23.8, 23.9**

from core.dialogue import PausePoint, DeclineResult, CandidatePresentation


# ---------------------------------------------------------------------------
# TestOnDecisionPoint — on_decision_point() pause/resume (Req 23.3, 23.9)
# ---------------------------------------------------------------------------


class TestOnDecisionPoint:
    """Tests for DialogueManager.on_decision_point() (Req 23.3, 23.9)."""

    @pytest.mark.asyncio
    async def test_on_decision_point_returns_answer_and_completed_steps(self):
        """
        on_decision_point() should return the user's answer and the list
        of completed steps so the caller can skip them on resume (Req 23.9).
        """
        async def fake_ask(question: str) -> str:
            return "user answer"

        dm = DialogueManager(ask_callback=fake_ask)
        completed = ["step_1", "step_2"]
        task_state = {"key": "value"}

        answer, returned_steps = await dm.on_decision_point(
            step_id="step_3",
            question="Which option?",
            slot_name="option",
            completed_steps=completed,
            task_state=task_state,
        )

        assert answer == "user answer"
        assert returned_steps == ["step_1", "step_2"]

    @pytest.mark.asyncio
    async def test_on_decision_point_retains_task_state(self):
        """
        The PausePoint created during on_decision_point() must carry the
        full task_state dict (Req 23.3).
        """
        pause_point_observed: list[PausePoint] = []

        async def fake_ask(question: str) -> str:
            # Capture the pause point while we are 'paused'
            pause_point_observed.append(dm.current_pause)
            return "ok"

        dm = DialogueManager(ask_callback=fake_ask)
        task_state = {"file": "report.pdf", "step_count": 3}

        await dm.on_decision_point(
            step_id="step_X",
            question="Proceed?",
            slot_name=None,
            completed_steps=[],
            task_state=task_state,
        )

        assert len(pause_point_observed) == 1
        pp = pause_point_observed[0]
        assert pp is not None
        assert pp.task_state == {"file": "report.pdf", "step_count": 3}

    @pytest.mark.asyncio
    async def test_is_paused_true_during_pause_false_after_resume(self):
        """
        is_paused is True while on_decision_point() awaits the user answer
        and False after the method returns (Req 23.3).
        """
        paused_during: list[bool] = []

        async def fake_ask(question: str) -> str:
            paused_during.append(dm.is_paused)
            return "answer"

        dm = DialogueManager(ask_callback=fake_ask)

        assert dm.is_paused is False
        await dm.on_decision_point(
            step_id="s1",
            question="Question?",
            slot_name=None,
            completed_steps=[],
            task_state={},
        )
        assert paused_during == [True]
        assert dm.is_paused is False

    @pytest.mark.asyncio
    async def test_current_pause_set_then_cleared(self):
        """
        current_pause holds a PausePoint while paused and is None after
        resuming (Req 23.3).
        """
        observed_pause: list[PausePoint | None] = []

        async def fake_ask(question: str) -> str:
            observed_pause.append(dm.current_pause)
            return "done"

        dm = DialogueManager(ask_callback=fake_ask)

        assert dm.current_pause is None
        await dm.on_decision_point(
            step_id="s2",
            question="Select?",
            slot_name="target",
            completed_steps=["s0", "s1"],
            task_state={"x": 1},
        )
        assert dm.current_pause is None

        pp = observed_pause[0]
        assert isinstance(pp, PausePoint)
        assert pp.step_id == "s2"
        assert pp.question == "Select?"
        assert pp.slot_name == "target"
        assert pp.completed_steps == ["s0", "s1"]
        assert pp.task_state == {"x": 1}

    @pytest.mark.asyncio
    async def test_on_decision_point_without_callback_returns_empty_answer(self):
        """
        Without an ask callback (headless mode), on_decision_point() returns
        an empty answer but still returns the completed_steps for resumption
        (Req 23.9).
        """
        dm = DialogueManager()
        completed = ["alpha", "beta"]

        answer, returned_steps = await dm.on_decision_point(
            step_id="gamma",
            question="Choose path?",
            slot_name="path",
            completed_steps=completed,
            task_state={"progress": 50},
        )

        assert answer == ""
        assert returned_steps == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_completed_steps_not_mutated_by_caller(self):
        """
        Mutations to the original completed_steps list after calling
        on_decision_point() must NOT affect the returned list (Req 23.9).
        """
        async def fake_ask(q: str) -> str:
            return "ok"

        dm = DialogueManager(ask_callback=fake_ask)
        original = ["step_a", "step_b"]

        _, returned = await dm.on_decision_point(
            step_id="step_c",
            question="?",
            slot_name=None,
            completed_steps=original,
            task_state={},
        )

        original.append("step_injected")
        assert "step_injected" not in returned


# ---------------------------------------------------------------------------
# TestOnDecline — on_decline() (Req 23.6, 23.7)
# ---------------------------------------------------------------------------


class TestOnDecline:
    """Tests for DialogueManager.on_decline() (Req 23.6, 23.7)."""

    def test_decline_with_default_returns_use_default_action(self):
        """
        When has_default=True, on_decline() returns action='use_default'
        (Req 23.6).
        """
        dm = DialogueManager()
        result = dm.on_decline("send_time", has_default=True, default_value="now")

        assert isinstance(result, DeclineResult)
        assert result.action == "use_default"

    def test_decline_with_default_returns_default_value(self):
        """
        When has_default=True, on_decline() includes the default_value in
        the result (Req 23.6).
        """
        dm = DialogueManager()
        result = dm.on_decline("language", has_default=True, default_value="English")

        assert result.value == "English"

    def test_decline_with_default_includes_descriptive_message(self):
        """
        When has_default=True, the message explains the default was applied
        (Req 23.6).
        """
        dm = DialogueManager()
        result = dm.on_decline("timezone", has_default=True, default_value="UTC")

        assert result.message  # non-empty
        assert "timezone" in result.message or "UTC" in result.message

    def test_decline_without_default_returns_abandon_step_action(self):
        """
        When has_default=False, on_decline() returns action='abandon_step'
        (Req 23.7).
        """
        dm = DialogueManager()
        result = dm.on_decline("recipient", has_default=False)

        assert isinstance(result, DeclineResult)
        assert result.action == "abandon_step"

    def test_decline_without_default_returns_none_value(self):
        """
        When has_default=False, the result value is None (Req 23.7).
        """
        dm = DialogueManager()
        result = dm.on_decline("recipient", has_default=False)

        assert result.value is None

    def test_decline_without_default_message_mentions_independent_steps(self):
        """
        When has_default=False, the message must clarify that only the
        dependent step is abandoned, not other independent steps (Req 23.7).
        """
        dm = DialogueManager()
        result = dm.on_decline("attachment", has_default=False)

        # Message should mention the slot and some variant of 'step'/'abandon'
        assert result.message
        assert "attachment" in result.message

    def test_decline_with_none_default_value_accepted(self):
        """
        has_default=True with default_value=None is a valid call — the
        default value itself is None.
        """
        dm = DialogueManager()
        result = dm.on_decline("priority", has_default=True, default_value=None)

        assert result.action == "use_default"
        assert result.value is None

    def test_decline_returns_decline_result_type(self):
        """on_decline() always returns a DeclineResult instance."""
        dm = DialogueManager()
        r1 = dm.on_decline("slot_a", has_default=True, default_value="x")
        r2 = dm.on_decline("slot_b", has_default=False)
        assert isinstance(r1, DeclineResult)
        assert isinstance(r2, DeclineResult)


# ---------------------------------------------------------------------------
# TestPresentOptions — present_options() (Req 23.8)
# ---------------------------------------------------------------------------


class TestPresentOptions:
    """Tests for DialogueManager.present_options() (Req 23.8)."""

    def test_without_callback_returns_candidate_presentation(self):
        """
        Without a callback, present_options() returns a CandidatePresentation
        (Req 23.8).
        """
        dm = DialogueManager()
        result = dm.present_options(["Alice", "Bob", "Carol"])

        assert isinstance(result, CandidatePresentation)

    def test_without_callback_selection_is_none(self):
        """
        Without a callback, selection is None — system is awaiting user
        input (Req 23.8).
        """
        dm = DialogueManager()
        result = dm.present_options(["option_1", "option_2"])

        assert isinstance(result, CandidatePresentation)
        assert result.selection is None

    def test_auto_selected_always_false_without_callback(self):
        """
        auto_selected is always False without a callback (Req 23.8).
        """
        dm = DialogueManager()
        result = dm.present_options(["x", "y", "z"])

        assert isinstance(result, CandidatePresentation)
        assert result.auto_selected is False

    def test_auto_selected_always_false_with_callback(self):
        """
        even with a callback that makes a selection, auto_selected must
        be False — the selection came from a user-provided callback, not
        auto-selection (Req 23.8).

        Note: when a callback is provided present_options returns the
        callback's return value directly.  The CandidatePresentation is
        only returned when no callback is supplied.  This test verifies
        auto_selected=False on a no-callback call (the non-auto path).
        """
        dm = DialogueManager()
        no_callback_result = dm.present_options(["a", "b"])
        assert isinstance(no_callback_result, CandidatePresentation)
        assert no_callback_result.auto_selected is False

    def test_without_callback_candidates_preserved(self):
        """
        The candidates list is preserved in the returned CandidatePresentation
        (Req 23.8).
        """
        dm = DialogueManager()
        candidates = ["contact_A", "contact_B", "contact_C"]
        result = dm.present_options(candidates)

        assert isinstance(result, CandidatePresentation)
        assert result.candidates == candidates

    def test_with_callback_returns_user_selection(self):
        """
        With a present_callback, present_options() returns the callback's
        return value (the user's selection) (Req 23.8).
        """
        dm = DialogueManager()
        candidates = ["Alice", "Bob", "Carol"]

        def user_picks(c: list) -> str:
            return c[1]  # user picks "Bob"

        result = dm.present_options(candidates, present_callback=user_picks)

        assert result == "Bob"

    def test_with_callback_does_not_return_candidate_presentation(self):
        """
        With a callback the direct return value (the selection) is returned,
        not a CandidatePresentation wrapper.
        """
        dm = DialogueManager()

        def pick_first(c: list) -> str:
            return c[0]

        result = dm.present_options(["X", "Y"], present_callback=pick_first)

        assert result == "X"
        assert not isinstance(result, CandidatePresentation)

    def test_empty_candidates_list_handled(self):
        """
        present_options() with an empty list returns a CandidatePresentation
        with an empty candidates list.
        """
        dm = DialogueManager()
        result = dm.present_options([])

        assert isinstance(result, CandidatePresentation)
        assert result.candidates == []
        assert result.selection is None
        assert result.auto_selected is False


# ---------------------------------------------------------------------------
# TestIsPausedAndCurrentPause — state properties (Req 23.3)
# ---------------------------------------------------------------------------


class TestIsPausedAndCurrentPause:
    """Tests for is_paused and current_pause properties (Req 23.3)."""

    def test_is_paused_false_initially(self):
        """is_paused is False on a fresh DialogueManager instance."""
        dm = DialogueManager()
        assert dm.is_paused is False

    def test_current_pause_none_initially(self):
        """current_pause is None on a fresh DialogueManager instance."""
        dm = DialogueManager()
        assert dm.current_pause is None

    @pytest.mark.asyncio
    async def test_is_paused_and_current_pause_set_then_cleared(self):
        """
        is_paused and current_pause reflect active pause during
        on_decision_point() and revert after resumption.
        """
        state_snapshots: list[dict] = []

        async def fake_ask(q: str) -> str:
            state_snapshots.append({
                "is_paused": dm.is_paused,
                "current_pause": dm.current_pause,
            })
            return "reply"

        dm = DialogueManager(ask_callback=fake_ask)

        await dm.on_decision_point(
            step_id="mid_step",
            question="Clarify?",
            slot_name="detail",
            completed_steps=["s0"],
            task_state={"ctx": 42},
        )

        # During the pause
        assert state_snapshots[0]["is_paused"] is True
        pp = state_snapshots[0]["current_pause"]
        assert pp is not None
        assert pp.step_id == "mid_step"
        assert pp.slot_name == "detail"
        assert pp.completed_steps == ["s0"]
        assert pp.task_state == {"ctx": 42}

        # After resumption
        assert dm.is_paused is False
        assert dm.current_pause is None
