"""
Integration-style tests for Dialogue_Manager wiring into ExecutionEngine
and IntentRouter (Task 25.3).

Covers:
  - ExecutionEngine: step with no required_slots proceeds without any dialogue call
  - ExecutionEngine: step with required_slots all already resolved proceeds without asking
  - ExecutionEngine: step with required_slots resolved from memory marks them and proceeds
  - ExecutionEngine: step with missing slots (no memory, no answer) → step abandoned
  - ExecutionEngine: slot resolution does not affect independent steps
  - IntentRouter: side-effecting intent with all slots resolved proceeds to handler
  - IntentRouter: side-effecting intent with memory-resolvable slots marks them and proceeds
  - IntentRouter: side-effecting intent with unresolvable slots yields clarification message

Design: Intent Routing, Execution loop.
Requirements: 23.1
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.dialogue import DialogueManager
from core.execution.execution_engine import (
    ExecutionEngine,
    StepEvent,
    StepEventType,
    collect_events,
)
from core.execution.safety_gate import SafetyGate
from core.orchestrator.intent_router import IntentRouter
from core.orchestrator.orchestrator import Intent, TurnContext
from core.planner import (
    Actuator,
    CommandPlan,
    Step,
    StepClassification,
    StepStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(
    step_id: str = "s1",
    intent: str = "do something",
    required_slots: list[str] | None = None,
    classification: StepClassification = StepClassification.REVERSIBLE,
    depends_on: list[str] | None = None,
) -> Step:
    return Step(
        id=step_id,
        intent=intent,
        actuator=Actuator.INTERNAL,
        classification=classification,
        depends_on=depends_on or [],
        required_slots=required_slots or [],
    )


def _plan(*steps: Step, command: str = "test") -> CommandPlan:
    return CommandPlan(origin_command=command, steps=list(steps))


def _make_memory_brain(slot_values: dict[str, str]) -> MagicMock:
    """Create a minimal MemoryBrain mock that resolves slot names to note bodies."""
    def fake_retrieve(query: str, k: int = 1) -> list:
        if query in slot_values:
            note = MagicMock()
            note.body = slot_values[query]
            return [note]
        return []

    brain = MagicMock()
    brain.retrieve = MagicMock(side_effect=fake_retrieve)
    return brain


def _intent_result(intent: Intent) -> Any:
    from core.orchestrator.intent_router import IntentResult
    return IntentResult(intent=intent)


def _turn_context(transcript: str = "test command") -> TurnContext:
    return TurnContext(transcript=transcript)


# ---------------------------------------------------------------------------
# ExecutionEngine + DialogueManager wiring tests
# ---------------------------------------------------------------------------


class TestExecutionEngineDialogueWiring:
    """
    Tests for ExecutionEngine per-step slot resolution via DialogueManager
    (Req 23.1).
    """

    @pytest.mark.asyncio
    async def test_step_with_no_required_slots_proceeds_without_dialogue(self):
        """
        A step with no required_slots must proceed without any
        DialogueManager call.
        """
        executed: list[str] = []

        async def actuator(step: Step) -> None:
            executed.append(step.id)

        dm = MagicMock(spec=DialogueManager)
        engine = ExecutionEngine(
            safety_gate=SafetyGate(),
            actuator_callback=actuator,
            dialogue_manager=dm,
        )
        plan = _plan(_step("s1", required_slots=[]))
        events = await collect_events(engine, plan)

        # dialogue_manager.assess should NOT be called for a step with no slots
        dm.assess.assert_not_called()
        assert "s1" in executed

    @pytest.mark.asyncio
    async def test_step_with_all_slots_pre_resolved_proceeds_without_asking(self):
        """
        A step whose required_slots are already resolved in the session
        proceeds without calling on_decision_point.
        """
        executed: list[str] = []

        async def actuator(step: Step) -> None:
            executed.append(step.id)

        dm = DialogueManager()
        dm.mark_resolved("target", "inbox")

        engine = ExecutionEngine(
            safety_gate=SafetyGate(),
            actuator_callback=actuator,
            dialogue_manager=dm,
        )
        step = _step("s1", required_slots=["target"])
        events = await collect_events(engine, _plan(step))

        # Step should have run
        assert "s1" in executed
        # No pause should have been created
        assert dm.is_paused is False

    @pytest.mark.asyncio
    async def test_step_with_slots_resolved_from_memory_marks_and_proceeds(self):
        """
        A step whose required_slots can be filled from memory:
        the slots are marked resolved and the step proceeds normally.
        """
        executed: list[str] = []

        async def actuator(step: Step) -> None:
            executed.append(step.id)

        brain = _make_memory_brain({"destination": "Mumbai"})
        dm = DialogueManager(memory_brain=brain)

        engine = ExecutionEngine(
            safety_gate=SafetyGate(),
            actuator_callback=actuator,
            dialogue_manager=dm,
        )
        step = _step("s1", required_slots=["destination"])
        await collect_events(engine, _plan(step))

        # Step should have run
        assert "s1" in executed
        # Slot should now be marked resolved
        assert dm.is_resolved("destination")
        assert dm.get_resolved("destination") == "Mumbai"

    @pytest.mark.asyncio
    async def test_step_with_missing_slot_no_memory_no_answer_is_abandoned(self):
        """
        A step with a required slot that:
        - is not pre-resolved in session
        - cannot be filled from memory
        - gets no answer from the user (empty answer from headless callback)
        → the step is abandoned (SKIPPED) and its dependents are also skipped.
        """
        actuator_called: list[str] = []

        async def actuator(step: Step) -> None:
            actuator_called.append(step.id)

        # No memory_brain → slot stays missing; no ask_callback → empty answer
        dm = DialogueManager(memory_brain=None)

        engine = ExecutionEngine(
            safety_gate=SafetyGate(),
            actuator_callback=actuator,
            dialogue_manager=dm,
        )
        step = _step("s1", required_slots=["recipient"])
        dep = _step("s2", depends_on=["s1"])
        events = await collect_events(engine, _plan(step, dep))

        # The step (and dependent) should NOT have been executed
        assert "s1" not in actuator_called
        assert "s2" not in actuator_called

        # Both should appear in the SKIPPED events
        skipped_ids = {e.step_id for e in events if e.event_type == StepEventType.SKIPPED}
        assert "s1" in skipped_ids
        assert "s2" in skipped_ids

    @pytest.mark.asyncio
    async def test_slot_resolution_does_not_affect_independent_steps(self):
        """
        When a step is abandoned due to a missing slot, independent steps
        (those that do NOT depend on the abandoned step) still run.

        Req 23.7 specifies that only the step requiring the slot (and its
        dependents) is abandoned — other independent steps proceed.
        """
        executed: list[str] = []

        async def actuator(step: Step) -> None:
            executed.append(step.id)

        dm = DialogueManager(memory_brain=None)

        engine = ExecutionEngine(
            safety_gate=SafetyGate(),
            actuator_callback=actuator,
            dialogue_manager=dm,
        )
        # s1 has a missing slot → will be abandoned
        # s_indep is independent → must still run
        s1 = _step("s1", required_slots=["missing_slot"])
        s_indep = _step("s_indep")
        events = await collect_events(engine, _plan(s1, s_indep))

        # Independent step must have executed
        assert "s_indep" in executed
        # Abandoned step must NOT have executed
        assert "s1" not in executed

    @pytest.mark.asyncio
    async def test_no_dialogue_manager_step_proceeds_normally(self):
        """
        When dialogue_manager=None (backwards-compatible), steps with
        required_slots proceed without any slot resolution (skipped).
        """
        executed: list[str] = []

        async def actuator(step: Step) -> None:
            executed.append(step.id)

        engine = ExecutionEngine(
            safety_gate=SafetyGate(),
            actuator_callback=actuator,
            dialogue_manager=None,
        )
        step = _step("s1", required_slots=["some_slot"])
        await collect_events(engine, _plan(step))

        # Without a dialogue_manager, the step runs regardless of required_slots
        assert "s1" in executed

    @pytest.mark.asyncio
    async def test_step_with_slot_filled_by_user_answer_proceeds(self):
        """
        When the user provides an answer via the ask_callback,
        the slot is marked resolved and the step runs.
        """
        executed: list[str] = []

        async def actuator(step: Step) -> None:
            executed.append(step.id)

        async def fake_ask(question: str) -> str:
            return "the_answer"

        dm = DialogueManager(memory_brain=None, ask_callback=fake_ask)

        engine = ExecutionEngine(
            safety_gate=SafetyGate(),
            actuator_callback=actuator,
            dialogue_manager=dm,
        )
        step = _step("s1", required_slots=["target_value"])
        await collect_events(engine, _plan(step))

        # Step must have executed after user answered
        assert "s1" in executed
        # Slot must be marked resolved
        assert dm.is_resolved("target_value")
        assert dm.get_resolved("target_value") == "the_answer"


# ---------------------------------------------------------------------------
# IntentRouter + DialogueManager wiring tests
# ---------------------------------------------------------------------------


class TestIntentRouterDialogueWiring:
    """
    Tests for IntentRouter.route() — side-effecting intent slot gating
    with memory-first resolution (Req 23.1, 23.2).
    """

    @pytest.mark.asyncio
    async def test_side_effecting_intent_with_all_slots_resolved_proceeds(self):
        """
        When all required slots for a side-effecting intent are already
        resolved in the session, route() proceeds to the handler and does
        NOT yield a clarification message.
        """
        dm = DialogueManager()
        # Pre-resolve all required slots for MAC_COMMAND
        dm.mark_resolved("command_target", "Safari")

        router = IntentRouter(dialogue_manager=dm)
        ctx = _turn_context("open Safari")

        chunks: list[str] = []
        async for chunk in router.route(_intent_result(Intent.MAC_COMMAND), ctx):
            chunks.append(chunk)

        response = "".join(chunks)
        # Should NOT be a clarification message
        assert "Missing" not in response
        assert "more information" not in response

    @pytest.mark.asyncio
    async def test_side_effecting_intent_with_memory_resolvable_slots_proceeds(self):
        """
        When slots are missing from session but can be resolved from memory,
        route() marks them resolved and proceeds to the handler — no
        clarification message is yielded.
        """
        brain = _make_memory_brain({"task_title": "Terminal"})
        dm = DialogueManager(memory_brain=brain)

        router = IntentRouter(dialogue_manager=dm)
        ctx = _turn_context("run a command")

        chunks: list[str] = []
        async for chunk in router.route(_intent_result(Intent.TASK), ctx):
            chunks.append(chunk)

        response = "".join(chunks)
        # Should NOT be a clarification message
        assert "Missing" not in response
        assert "more information" not in response

        # The slot should now be marked resolved in the session
        assert dm.is_resolved("task_title")
        assert dm.get_resolved("task_title") == "Terminal"

    @pytest.mark.asyncio
    async def test_side_effecting_intent_with_unresolvable_slots_yields_clarification(self):
        """
        When slots are missing and cannot be filled from memory,
        route() yields a clarification message and does NOT invoke
        the capability handler.
        """
        dm = DialogueManager(memory_brain=None)  # no memory

        router = IntentRouter(dialogue_manager=dm)
        ctx = _turn_context("schedule something")

        chunks: list[str] = []
        async for chunk in router.route(_intent_result(Intent.SCHEDULE), ctx):
            chunks.append(chunk)

        response = "".join(chunks)
        # Must be a clarification message
        assert "Missing" in response or "more information" in response

    @pytest.mark.asyncio
    async def test_read_only_intent_proceeds_without_slot_gate(self):
        """
        Read-only intents (chat, recall, etc.) must NOT be gated by
        the DialogueManager — they proceed directly to the handler.
        """
        dm = MagicMock(spec=DialogueManager)

        router = IntentRouter(dialogue_manager=dm)
        ctx = _turn_context("what time is it?")

        chunks: list[str] = []
        async for chunk in router.route(_intent_result(Intent.CHAT), ctx):
            chunks.append(chunk)

        # DialogueManager.assess should NOT be called for read-only intents
        dm.assess.assert_not_called()
        # Must have gotten a response
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_intent_router_uses_shared_dialogue_manager_instance(self):
        """
        The IntentRouter and the Orchestrator should share the SAME
        DialogueManager instance so slots resolved at intent-routing time
        are available during execution.
        """
        dm = DialogueManager()
        router = IntentRouter(dialogue_manager=dm)

        # The router's internal DM should be the exact same object we passed in
        assert router._dialogue_manager is dm

    @pytest.mark.asyncio
    async def test_memory_resolved_slots_marked_and_not_re_asked(self):
        """
        After a slot is resolved from memory in IntentRouter.route(),
        a second call with the same intent must not ask for the slot again
        (Req 23.4 — no re-ask for already-resolved slots).
        """
        brain = _make_memory_brain({"automation_name": "daily_report"})
        dm = DialogueManager(memory_brain=brain)
        router = IntentRouter(dialogue_manager=dm)

        ctx = _turn_context("run my automation")

        # First route — fills from memory
        chunks1: list[str] = []
        async for chunk in router.route(_intent_result(Intent.RUN_AUTOMATION), ctx):
            chunks1.append(chunk)

        # Slot should now be resolved
        assert dm.is_resolved("automation_name")

        # Second route — slot already resolved, must proceed without clarification
        # (Reset the brain to empty so memory would NOT find it again)
        empty_brain = _make_memory_brain({})
        dm._memory_brain = empty_brain

        chunks2: list[str] = []
        async for chunk in router.route(_intent_result(Intent.RUN_AUTOMATION), ctx):
            chunks2.append(chunk)

        response2 = "".join(chunks2)
        # Must NOT be a clarification message (slot was already resolved)
        assert "Missing" not in response2
        assert "more information" not in response2
