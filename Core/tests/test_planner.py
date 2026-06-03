"""
Unit and property-based tests for the Planner module (Tasks 21.1, 21.2).

Covers:
  - Task 21.1: CommandPlan / Step data model and dependency graph
  - Task 21.2: LLM planner with memory-backed slot filling

**Validates: Requirements 17.2, 21.1, 21.7**

Testing conventions:
  - Property tests use Hypothesis and run ≥100 iterations.
  - The LLM ModelProvider is stubbed so tests are cheap and deterministic.
  - Memory_Brain is either stubbed or backed by a real tmp vault.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from core.planner import (
    Actuator,
    CommandPlan,
    CyclicDependencyError,
    PlanGenerationError,
    Planner,
    Slot,
    SlotStatus,
    Step,
    StepClassification,
    StepStatus,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _step(**kwargs) -> Step:
    """Build a Step with sensible defaults."""
    defaults = dict(
        intent="do something",
        actuator=Actuator.INTERNAL,
        classification=StepClassification.REVERSIBLE,
    )
    defaults.update(kwargs)
    return Step(**defaults)


def _plan(*steps: Step, command: str = "test command") -> CommandPlan:
    """Build a CommandPlan from the given steps."""
    return CommandPlan(origin_command=command, steps=list(steps))


def _json_plan(*raw_steps: dict) -> str:
    """Serialize step dicts to the JSON format the LLM is expected to return."""
    return json.dumps({"steps": list(raw_steps)})


def _stub_llm_provider(response: str) -> MagicMock:
    """Return a mock ModelProvider whose invoke() returns *response*."""
    provider = MagicMock()
    provider.invoke.return_value = response
    return provider


def _note_with_body(body: str) -> MagicMock:
    """Return a minimal mock Note with the given body."""
    note = MagicMock()
    note.body = body
    note.topics = []
    note.tags = []
    return note


# ===========================================================================
# Task 21.1 — CommandPlan / Step data model and dependency graph
# ===========================================================================


class TestStepDataModel:
    """Unit tests for the Step dataclass (Task 21.1, Req 21.1)."""

    def test_default_status_is_pending(self):
        s = Step()
        assert s.status == StepStatus.PENDING

    def test_default_classification_is_unknown(self):
        s = Step()
        assert s.classification == StepClassification.UNKNOWN

    def test_id_is_auto_generated(self):
        s1, s2 = Step(), Step()
        assert s1.id != s2.id

    def test_explicit_id_preserved(self):
        s = Step(id="fixed-id")
        assert s.id == "fixed-id"

    def test_depends_on_defaults_to_empty(self):
        s = Step()
        assert s.depends_on == []

    def test_required_slots_defaults_to_empty(self):
        s = Step()
        assert s.required_slots == []

    def test_args_defaults_to_empty_dict(self):
        s = Step()
        assert s.args == {}

    def test_all_statuses_are_valid(self):
        for status in StepStatus:
            s = Step(status=status)
            assert s.status == status

    def test_all_actuators_are_valid(self):
        for act in Actuator:
            s = Step(actuator=act)
            assert s.actuator == act

    def test_all_classifications_are_valid(self):
        for cls in StepClassification:
            s = Step(classification=cls)
            assert s.classification == cls


# ===========================================================================
# Task 21.1 — Slot / SlotStatus data model
# ===========================================================================


class TestSlotStatusEnum:
    """Verify all SlotStatus enum values are present (Task 21.1, Req 23.1)."""

    def test_all_slot_statuses_present(self):
        expected = {"pending", "filled", "declined", "default_applied"}
        actual = {s.value for s in SlotStatus}
        assert actual == expected

    def test_slot_status_is_string_enum(self):
        assert isinstance(SlotStatus.PENDING, str)
        assert SlotStatus.FILLED == "filled"


class TestSlotDataModel:
    """Unit tests for the Slot dataclass (Task 21.1, Req 23.1, 23.2, 23.6)."""

    def test_default_status_is_pending(self):
        s = Slot(name="recipient")
        assert s.status == SlotStatus.PENDING

    def test_default_value_is_none(self):
        s = Slot(name="app")
        assert s.value is None

    def test_fill_sets_value_and_status(self):
        s = Slot(name="recipient")
        s.fill("alice@example.com")
        assert s.value == "alice@example.com"
        assert s.status == SlotStatus.FILLED

    def test_fill_marks_as_resolved(self):
        s = Slot(name="target")
        s.fill("Safari")
        assert s.is_resolved is True

    def test_decline_without_default_marks_declined(self):
        s = Slot(name="contact")
        s.decline()
        assert s.status == SlotStatus.DECLINED
        assert s.value is None
        assert s.is_resolved is False

    def test_decline_with_default_applies_default(self):
        s = Slot(name="app", default="Safari")
        s.decline()
        assert s.status == SlotStatus.DEFAULT_APPLIED
        assert s.value == "Safari"
        assert s.is_resolved is True

    def test_pending_slot_is_not_resolved(self):
        s = Slot(name="x")
        assert s.is_resolved is False

    def test_slot_name_stored(self):
        s = Slot(name="professor_email")
        assert s.name == "professor_email"

    def test_slot_with_explicit_value(self):
        s = Slot(name="url", value="https://github.com", status=SlotStatus.FILLED)
        assert s.is_resolved is True
        assert s.value == "https://github.com"


class TestEnumCompleteness:
    """Verify all enum values for every enum in the planner module (Task 21.1)."""

    def test_step_status_values(self):
        expected = {
            "pending", "awaiting_confirm", "awaiting_clarify",
            "running", "completed", "failed", "skipped",
        }
        actual = {s.value for s in StepStatus}
        assert actual == expected

    def test_step_classification_values(self):
        expected = {"consequential", "reversible", "unknown"}
        actual = {c.value for c in StepClassification}
        assert actual == expected

    def test_actuator_values(self):
        expected = {
            "applescript", "ax", "apple_events", "cdp",
            "vision", "calendar", "notifications", "internal",
        }
        actual = {a.value for a in Actuator}
        assert actual == expected

    def test_slot_status_values(self):
        expected = {"pending", "filled", "declined", "default_applied"}
        actual = {s.value for s in SlotStatus}
        assert actual == expected

    def test_all_enums_are_str_enums(self):
        for enum_cls in (StepStatus, StepClassification, Actuator, SlotStatus):
            for member in enum_cls:
                assert isinstance(member, str), (
                    f"{enum_cls.__name__}.{member.name} is not a str subclass"
                )



class TestCommandPlanDataModel:
    """Unit tests for CommandPlan dataclass (Task 21.1, Req 21.1)."""

    def test_id_auto_generated(self):
        p1, p2 = CommandPlan(), CommandPlan()
        assert p1.id != p2.id

    def test_steps_defaults_to_empty(self):
        assert CommandPlan().steps == []

    def test_origin_command_stored(self):
        p = CommandPlan(origin_command="open Safari")
        assert p.origin_command == "open Safari"

    def test_is_complete_all_completed(self):
        s1 = _step(status=StepStatus.COMPLETED)
        s2 = _step(status=StepStatus.COMPLETED)
        assert _plan(s1, s2).is_complete()

    def test_is_complete_with_skipped(self):
        s1 = _step(status=StepStatus.COMPLETED)
        s2 = _step(status=StepStatus.SKIPPED)
        assert _plan(s1, s2).is_complete()

    def test_is_complete_with_failed(self):
        s1 = _step(status=StepStatus.COMPLETED)
        s2 = _step(status=StepStatus.FAILED)
        assert _plan(s1, s2).is_complete()

    def test_is_not_complete_when_pending(self):
        s1 = _step(status=StepStatus.COMPLETED)
        s2 = _step(status=StepStatus.PENDING)
        assert not _plan(s1, s2).is_complete()

    def test_is_not_complete_when_running(self):
        s1 = _step(status=StepStatus.RUNNING)
        assert not _plan(s1).is_complete()

    def test_completed_steps_filtered(self):
        s1 = _step(status=StepStatus.COMPLETED)
        s2 = _step(status=StepStatus.PENDING)
        s3 = _step(status=StepStatus.FAILED)
        p = _plan(s1, s2, s3)
        assert p.completed_steps() == [s1]

    def test_failed_steps_filtered(self):
        s1 = _step(status=StepStatus.COMPLETED)
        s2 = _step(status=StepStatus.FAILED)
        assert _plan(s1, s2).failed_steps() == [s2]

    def test_skipped_steps_filtered(self):
        s1 = _step(status=StepStatus.SKIPPED)
        s2 = _step(status=StepStatus.PENDING)
        assert _plan(s1, s2).skipped_steps() == [s1]



class TestReadySteps:
    """Tests for CommandPlan.ready_steps() — the core of Req 17.2."""

    def test_no_deps_all_pending_are_ready(self):
        s1 = _step()
        s2 = _step()
        p = _plan(s1, s2)
        assert set(r.id for r in p.ready_steps()) == {s1.id, s2.id}

    def test_pending_step_with_completed_dep_is_ready(self):
        s1 = _step(id="a", status=StepStatus.COMPLETED)
        s2 = _step(id="b", depends_on=["a"])
        p = _plan(s1, s2)
        ready_ids = {r.id for r in p.ready_steps()}
        assert "b" in ready_ids

    def test_pending_step_with_pending_dep_is_not_ready(self):
        s1 = _step(id="a")
        s2 = _step(id="b", depends_on=["a"])
        p = _plan(s1, s2)
        ready_ids = {r.id for r in p.ready_steps()}
        assert "b" not in ready_ids
        assert "a" in ready_ids

    def test_empty_plan_returns_empty(self):
        assert _plan().ready_steps() == []

    def test_completed_steps_not_in_ready(self):
        s1 = _step(id="a", status=StepStatus.COMPLETED)
        p = _plan(s1)
        assert p.ready_steps() == []

    def test_failed_step_dep_blocks_dependent(self):
        """A step whose dep is FAILED is not ready (dep is not COMPLETED)."""
        s1 = _step(id="a", status=StepStatus.FAILED)
        s2 = _step(id="b", depends_on=["a"])
        p = _plan(s1, s2)
        ready_ids = {r.id for r in p.ready_steps()}
        assert "b" not in ready_ids

    def test_multiple_deps_all_completed(self):
        s1 = _step(id="a", status=StepStatus.COMPLETED)
        s2 = _step(id="b", status=StepStatus.COMPLETED)
        s3 = _step(id="c", depends_on=["a", "b"])
        p = _plan(s1, s2, s3)
        ready_ids = {r.id for r in p.ready_steps()}
        assert "c" in ready_ids

    def test_multiple_deps_one_not_completed(self):
        s1 = _step(id="a", status=StepStatus.COMPLETED)
        s2 = _step(id="b")  # PENDING
        s3 = _step(id="c", depends_on=["a", "b"])
        p = _plan(s1, s2, s3)
        ready_ids = {r.id for r in p.ready_steps()}
        assert "c" not in ready_ids



class TestDependentSteps:
    """Tests for CommandPlan.dependent_steps() and transitive_dependents()."""

    def test_direct_dependents_returned(self):
        s1 = _step(id="a")
        s2 = _step(id="b", depends_on=["a"])
        s3 = _step(id="c", depends_on=["a"])
        s4 = _step(id="d")  # no dep on "a"
        p = _plan(s1, s2, s3, s4)
        dep_ids = {s.id for s in p.dependent_steps("a")}
        assert dep_ids == {"b", "c"}
        assert "d" not in dep_ids

    def test_no_dependents_returns_empty(self):
        s1 = _step(id="a")
        s2 = _step(id="b")
        p = _plan(s1, s2)
        assert p.dependent_steps("a") == []

    def test_completed_step_not_in_dependents(self):
        s1 = _step(id="a")
        s2 = _step(id="b", depends_on=["a"], status=StepStatus.COMPLETED)
        p = _plan(s1, s2)
        assert p.dependent_steps("a") == []

    def test_transitive_dependents_chain(self):
        """a → b → c: transitive_dependents("a") must include b AND c."""
        s1 = _step(id="a")
        s2 = _step(id="b", depends_on=["a"])
        s3 = _step(id="c", depends_on=["b"])
        p = _plan(s1, s2, s3)
        trans_ids = {s.id for s in p.transitive_dependents("a")}
        assert "b" in trans_ids
        assert "c" in trans_ids

    def test_transitive_dependents_excludes_independent(self):
        s1 = _step(id="a")
        s2 = _step(id="b", depends_on=["a"])
        s3 = _step(id="c")  # independent of a
        p = _plan(s1, s2, s3)
        trans_ids = {s.id for s in p.transitive_dependents("a")}
        assert "c" not in trans_ids

    def test_transitive_dependents_excludes_non_pending(self):
        s1 = _step(id="a")
        s2 = _step(id="b", depends_on=["a"], status=StepStatus.RUNNING)
        s3 = _step(id="c", depends_on=["b"])
        p = _plan(s1, s2, s3)
        trans_ids = {s.id for s in p.transitive_dependents("a")}
        # s2 is not PENDING so it is excluded; s3 depends on s2 but s2 is not visited
        assert "b" not in trans_ids


class TestTopologicalOrder:
    """Tests for CommandPlan.topological_order() (Req 17.2)."""

    def test_linear_chain_ordered(self):
        s1 = _step(id="a")
        s2 = _step(id="b", depends_on=["a"])
        s3 = _step(id="c", depends_on=["b"])
        p = _plan(s1, s2, s3)
        order = p.topological_order()
        ids = [s.id for s in order]
        assert ids.index("a") < ids.index("b") < ids.index("c")

    def test_independent_steps_both_appear(self):
        s1 = _step(id="a")
        s2 = _step(id="b")
        p = _plan(s1, s2)
        ids = {s.id for s in p.topological_order()}
        assert ids == {"a", "b"}

    def test_empty_plan_returns_empty(self):
        assert _plan().topological_order() == []

    def test_cycle_raises_error(self):
        s1 = _step(id="a", depends_on=["b"])
        s2 = _step(id="b", depends_on=["a"])
        p = _plan(s1, s2)
        with pytest.raises(CyclicDependencyError):
            p.topological_order()

    def test_has_cycle_true(self):
        s1 = _step(id="a", depends_on=["b"])
        s2 = _step(id="b", depends_on=["a"])
        assert _plan(s1, s2).has_cycle()

    def test_has_cycle_false(self):
        s1 = _step(id="a")
        s2 = _step(id="b", depends_on=["a"])
        assert not _plan(s1, s2).has_cycle()

    def test_diamond_dependency_valid(self):
        """a → b, a → c, b+c → d (diamond) should not raise."""
        s_a = _step(id="a")
        s_b = _step(id="b", depends_on=["a"])
        s_c = _step(id="c", depends_on=["a"])
        s_d = _step(id="d", depends_on=["b", "c"])
        p = _plan(s_a, s_b, s_c, s_d)
        order = p.topological_order()
        ids = [s.id for s in order]
        assert ids.index("a") < ids.index("b")
        assert ids.index("a") < ids.index("c")
        assert ids.index("b") < ids.index("d")
        assert ids.index("c") < ids.index("d")



# ===========================================================================
# Task 21.2 — LLM Planner (stub + real)
# ===========================================================================


class TestPlannerStub:
    """Tests for Planner with no LLM provider (stub fallback)."""

    def test_returns_command_plan(self):
        planner = Planner()
        plan = planner.plan("open Safari")
        assert isinstance(plan, CommandPlan)

    def test_stub_plan_has_one_step(self):
        planner = Planner()
        plan = planner.plan("do something")
        assert len(plan.steps) == 1

    def test_stub_plan_intent_is_command(self):
        cmd = "open Messages"
        plan = Planner().plan(cmd)
        assert plan.steps[0].intent == cmd

    def test_stub_plan_origin_command(self):
        cmd = "search for flights"
        plan = Planner().plan(cmd)
        assert plan.origin_command == cmd

    def test_stub_plan_step_is_pending(self):
        plan = Planner().plan("launch app")
        assert plan.steps[0].status == StepStatus.PENDING

    def test_stub_plan_classification_is_unknown(self):
        plan = Planner().plan("do something")
        assert plan.steps[0].classification == StepClassification.UNKNOWN


class TestPlannerWithLLM:
    """Tests for Planner with a mocked LLM provider (Task 21.2, Req 21.1)."""

    def _two_step_json(self) -> str:
        return _json_plan(
            {
                "id": "step_1",
                "intent": "launch Safari",
                "actuator": "applescript",
                "args": {"app": "Safari"},
                "depends_on": [],
                "classification": "reversible",
                "required_slots": [],
            },
            {
                "id": "step_2",
                "intent": "navigate to GitHub",
                "actuator": "cdp",
                "args": {"url": "https://github.com"},
                "depends_on": ["step_1"],
                "classification": "reversible",
                "required_slots": [],
            },
        )

    def test_returns_command_plan(self):
        provider = _stub_llm_provider(self._two_step_json())
        plan = Planner(model_provider=provider).plan("open GitHub")
        assert isinstance(plan, CommandPlan)

    def test_two_steps_parsed(self):
        provider = _stub_llm_provider(self._two_step_json())
        plan = Planner(model_provider=provider).plan("open GitHub")
        assert len(plan.steps) == 2

    def test_dependency_preserved(self):
        provider = _stub_llm_provider(self._two_step_json())
        plan = Planner(model_provider=provider).plan("open GitHub")
        s2 = plan.steps[1]
        assert "step_1" in s2.depends_on

    def test_actuator_parsed(self):
        provider = _stub_llm_provider(self._two_step_json())
        plan = Planner(model_provider=provider).plan("open GitHub")
        assert plan.steps[0].actuator == Actuator.APPLESCRIPT

    def test_classification_parsed(self):
        provider = _stub_llm_provider(self._two_step_json())
        plan = Planner(model_provider=provider).plan("open GitHub")
        assert plan.steps[0].classification == StepClassification.REVERSIBLE

    def test_args_parsed(self):
        provider = _stub_llm_provider(self._two_step_json())
        plan = Planner(model_provider=provider).plan("open GitHub")
        assert plan.steps[0].args == {"app": "Safari"}

    def test_llm_failure_falls_back_to_stub(self):
        provider = MagicMock()
        provider.invoke.side_effect = RuntimeError("model offline")
        plan = Planner(model_provider=provider).plan("do something")
        # Stub plan: single step with intent == command
        assert len(plan.steps) == 1
        assert plan.steps[0].intent == "do something"

    def test_malformed_json_falls_back_to_stub(self):
        provider = _stub_llm_provider("not json at all")
        plan = Planner(model_provider=provider).plan("do something")
        assert len(plan.steps) == 1

    def test_json_wrapped_in_prose_parsed(self):
        prose_wrapped = (
            "Sure! Here is the plan:\n"
            + self._two_step_json()
            + "\nLet me know if you need changes."
        )
        provider = _stub_llm_provider(prose_wrapped)
        plan = Planner(model_provider=provider).plan("open GitHub")
        assert len(plan.steps) == 2

    def test_json_in_code_fence_parsed(self):
        code_fence = "```json\n" + self._two_step_json() + "\n```"
        provider = _stub_llm_provider(code_fence)
        plan = Planner(model_provider=provider).plan("open GitHub")
        assert len(plan.steps) == 2

    def test_unknown_actuator_defaults_to_internal(self):
        raw = _json_plan({
            "id": "s1",
            "intent": "do x",
            "actuator": "totally_unknown_backend",
            "args": {},
            "depends_on": [],
            "classification": "reversible",
            "required_slots": [],
        })
        provider = _stub_llm_provider(raw)
        plan = Planner(model_provider=provider).plan("x")
        assert plan.steps[0].actuator == Actuator.INTERNAL

    def test_unknown_classification_defaults_to_unknown(self):
        raw = _json_plan({
            "id": "s1",
            "intent": "do x",
            "actuator": "internal",
            "args": {},
            "depends_on": [],
            "classification": "totally_unknown",
            "required_slots": [],
        })
        provider = _stub_llm_provider(raw)
        plan = Planner(model_provider=provider).plan("x")
        assert plan.steps[0].classification == StepClassification.UNKNOWN

    def test_cyclic_plan_falls_back_to_stub(self):
        """A plan with a cycle cannot be executed; Planner falls back to stub."""
        raw = _json_plan(
            {
                "id": "a", "intent": "step a", "actuator": "internal",
                "args": {}, "depends_on": ["b"],
                "classification": "reversible", "required_slots": [],
            },
            {
                "id": "b", "intent": "step b", "actuator": "internal",
                "args": {}, "depends_on": ["a"],
                "classification": "reversible", "required_slots": [],
            },
        )
        provider = _stub_llm_provider(raw)
        # A cyclic LLM plan should raise CyclicDependencyError
        with pytest.raises(CyclicDependencyError):
            Planner(model_provider=provider).plan("cycle test")



class TestMemoryBackedSlotFilling:
    """Tests for memory-backed slot filling (Task 21.2, Req 21.7)."""

    def _plan_with_required_slot(self, slot: str) -> str:
        return _json_plan({
            "id": "s1",
            "intent": f"send message to {slot}",
            "actuator": "apple_events",
            "args": {},
            "depends_on": [],
            "classification": "consequential",
            "required_slots": [slot],
        })

    def test_slot_filled_from_memory_context(self):
        slot = "professor_email"
        raw = self._plan_with_required_slot(slot)
        provider = _stub_llm_provider(raw)
        memory = [_note_with_body("professor email is prof@university.edu")]

        plan = Planner(model_provider=provider).plan(
            "email my professor", memory_context=memory
        )
        step = plan.steps[0]
        # Slot should be resolved — no longer in required_slots
        assert slot not in step.required_slots

    def test_filled_slot_added_to_args(self):
        slot = "professor_email"
        raw = self._plan_with_required_slot(slot)
        provider = _stub_llm_provider(raw)
        memory = [_note_with_body("professor email is prof@university.edu")]

        plan = Planner(model_provider=provider).plan(
            "email my professor", memory_context=memory
        )
        step = plan.steps[0]
        assert slot in step.args

    def test_unresolvable_slot_stays_in_required_slots(self):
        slot = "unknown_contact"
        raw = self._plan_with_required_slot(slot)
        provider = _stub_llm_provider(raw)
        # Memory has no info about unknown_contact
        memory = [_note_with_body("I have a midterm on June 14")]

        plan = Planner(model_provider=provider).plan(
            "send to unknown contact", memory_context=memory
        )
        step = plan.steps[0]
        assert slot in step.required_slots

    def test_no_memory_context_slots_unchanged(self):
        slot = "recipient"
        raw = self._plan_with_required_slot(slot)
        provider = _stub_llm_provider(raw)

        plan = Planner(model_provider=provider).plan("send msg", memory_context=[])
        step = plan.steps[0]
        assert slot in step.required_slots

    def test_memory_brain_queried_when_no_context_provided(self):
        """When memory_context is None, the planner queries memory_brain."""
        slot = "app_name"
        raw = self._plan_with_required_slot(slot)
        provider = _stub_llm_provider(raw)

        mock_brain = MagicMock()
        mock_brain.retrieve.return_value = []

        Planner(model_provider=provider, memory_brain=mock_brain).plan("open app")
        mock_brain.retrieve.assert_called_once()

    def test_explicit_memory_context_bypasses_brain_retrieve(self):
        """Explicit memory_context skips calling memory_brain.retrieve."""
        raw = _json_plan({
            "id": "s1", "intent": "do x", "actuator": "internal",
            "args": {}, "depends_on": [], "classification": "reversible",
            "required_slots": [],
        })
        provider = _stub_llm_provider(raw)
        mock_brain = MagicMock()

        Planner(model_provider=provider, memory_brain=mock_brain).plan(
            "do x", memory_context=[]
        )
        mock_brain.retrieve.assert_not_called()

    def test_memory_brain_failure_doesnt_crash_planner(self):
        """Brain retrieval errors are swallowed; planning continues."""
        raw = _json_plan({
            "id": "s1", "intent": "x", "actuator": "internal",
            "args": {}, "depends_on": [], "classification": "reversible",
            "required_slots": [],
        })
        provider = _stub_llm_provider(raw)
        mock_brain = MagicMock()
        mock_brain.retrieve.side_effect = RuntimeError("db error")

        plan = Planner(model_provider=provider, memory_brain=mock_brain).plan("x")
        assert isinstance(plan, CommandPlan)



# ===========================================================================
# Property-based tests
# ===========================================================================


# ---------------------------------------------------------------------------
# Property 51: Execution respects step dependencies and order
# Feature: haki-personal-ai-assistant, Property 51: execution_respects_step_dependencies
# **Validates: Requirements 17.2**
# ---------------------------------------------------------------------------


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    chain_length=st.integers(min_value=1, max_value=8),
)
def test_property_51_ready_steps_respect_dependency_order(chain_length: int) -> None:
    """
    # Feature: haki-personal-ai-assistant, Property 51: execution_respects_step_dependencies
    **Validates: Requirements 17.2**

    Property: for a linear chain s1 → s2 → … → sN, at each point during
    simulated execution only the step whose predecessor is COMPLETED is
    returned by ready_steps().  Steps with unmet dependencies are never
    ready.
    """
    # Build a chain: each step depends on the previous
    steps = [_step(id=f"step_{i}") for i in range(chain_length)]
    for i in range(1, chain_length):
        steps[i].depends_on = [f"step_{i - 1}"]

    plan = _plan(*steps)

    # Simulate sequential execution respecting ready_steps()
    executed_order: list[str] = []
    for _ in range(chain_length):
        ready = plan.ready_steps()
        # At most one step is ready in a strict chain
        assert len(ready) == 1, (
            f"Expected exactly 1 ready step in chain, got {[r.id for r in ready]}"
        )
        step = ready[0]
        # Verify all dependencies are completed
        for dep_id in step.depends_on:
            assert dep_id in executed_order, (
                f"Step '{step.id}' became ready before its dep '{dep_id}' was completed"
            )
        step.status = StepStatus.COMPLETED
        executed_order.append(step.id)

    # All steps must have been executed
    assert len(executed_order) == chain_length


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_parallel=st.integers(min_value=2, max_value=6),
)
def test_property_51_independent_steps_all_ready(n_parallel: int) -> None:
    """
    # Feature: haki-personal-ai-assistant, Property 51: execution_respects_step_dependencies
    **Validates: Requirements 17.2**

    Property: all steps with no dependencies are simultaneously ready —
    they can run in parallel (Req 17.2).
    """
    steps = [_step(id=f"s{i}") for i in range(n_parallel)]
    plan = _plan(*steps)
    ready = plan.ready_steps()
    assert len(ready) == n_parallel, (
        f"Expected {n_parallel} independent steps to be ready, got {len(ready)}"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_deps=st.integers(min_value=1, max_value=5),
    n_independent=st.integers(min_value=1, max_value=5),
)
def test_property_51_dependent_step_not_ready_when_deps_pending(
    n_deps: int, n_independent: int
) -> None:
    """
    # Feature: haki-personal-ai-assistant, Property 51: execution_respects_step_dependencies
    **Validates: Requirements 17.2**

    Property: a step is never in ready_steps() when at least one of its
    depends_on steps is still PENDING.
    """
    dep_steps = [_step(id=f"dep_{i}") for i in range(n_deps)]
    dep_ids = [s.id for s in dep_steps]
    gated_step = _step(id="gated", depends_on=dep_ids)
    independent_steps = [_step(id=f"ind_{i}") for i in range(n_independent)]
    plan = _plan(*dep_steps, gated_step, *independent_steps)

    ready = plan.ready_steps()
    ready_ids = {s.id for s in ready}

    assert "gated" not in ready_ids, (
        "Gated step appeared in ready_steps() while its deps are still PENDING"
    )
    # Independent steps must all be ready
    for s in independent_steps:
        assert s.id in ready_ids


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_steps=st.integers(min_value=1, max_value=6),
)
def test_property_51_topological_order_satisfies_dependencies(n_steps: int) -> None:
    """
    # Feature: haki-personal-ai-assistant, Property 51: execution_respects_step_dependencies
    **Validates: Requirements 17.2**

    Property: for any acyclic plan, the topological order places every
    step after all of its depends_on predecessors.
    """
    # Build a valid DAG: each step may depend on the previous (no cycles)
    steps = [_step(id=f"t{i}") for i in range(n_steps)]
    for i in range(1, n_steps):
        # Only add a dependency half the time to vary the structure
        if i % 2 == 0:
            steps[i].depends_on = [f"t{i - 1}"]

    plan = _plan(*steps)
    order = plan.topological_order()
    position = {s.id: idx for idx, s in enumerate(order)}

    for step in steps:
        for dep_id in step.depends_on:
            if dep_id in position:
                assert position[dep_id] < position[step.id], (
                    f"Step '{step.id}' appears before its dep '{dep_id}' in topological order"
                )



# ---------------------------------------------------------------------------
# Property 65: Memory-backed slot filling
# Feature: haki-personal-ai-assistant, Property 65: memory_backed_slot_filling
# **Validates: Requirements 21.7**
# ---------------------------------------------------------------------------


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    slot_name=st.from_regex(r"[a-z][a-z_]{2,10}", fullmatch=True),
    slot_value=st.from_regex(r"[a-zA-Z0-9@.]{4,20}", fullmatch=True),
)
def test_property_65_memory_backed_slot_filling(
    slot_name: str, slot_value: str
) -> None:
    """
    # Feature: haki-personal-ai-assistant, Property 65: memory_backed_slot_filling
    **Validates: Requirements 21.7**

    Property: when a memory note contains "<slot_name> is <slot_value>",
    the planner fills the slot from memory (removes it from required_slots,
    adds the value to args) instead of leaving it unresolved.
    """
    # Build a plan whose only step requires the slot
    raw = _json_plan({
        "id": "s1",
        "intent": f"use {slot_name}",
        "actuator": "internal",
        "args": {},
        "depends_on": [],
        "classification": "reversible",
        "required_slots": [slot_name],
    })
    provider = _stub_llm_provider(raw)

    # Memory note that resolves the slot
    # Format: "<slot_key> is <value>" — matches the resolver's primary pattern
    slot_key = slot_name.replace("_", " ")
    note = _note_with_body(f"my {slot_key} is {slot_value}")

    plan = Planner(model_provider=provider).plan(
        f"use my {slot_key}", memory_context=[note]
    )
    step = plan.steps[0]

    # The slot must have been resolved from memory
    assert slot_name not in step.required_slots, (
        f"Slot '{slot_name}' should have been filled from memory "
        f"(note body: 'my {slot_key} is {slot_value}')"
    )
    assert slot_name in step.args, (
        f"Resolved slot '{slot_name}' should appear in step.args"
    )


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    num_slots=st.integers(min_value=1, max_value=4),
)
def test_property_65_unresolvable_slots_stay_required(num_slots: int) -> None:
    """
    # Feature: haki-personal-ai-assistant, Property 65: memory_backed_slot_filling
    **Validates: Requirements 21.7**

    Property: slots that cannot be resolved from memory stay in
    required_slots so the Dialogue_Manager can ask the user (Req 23.1).
    """
    slot_names = [f"unknown_slot_{i}" for i in range(num_slots)]
    raw = _json_plan({
        "id": "s1",
        "intent": "do something",
        "actuator": "internal",
        "args": {},
        "depends_on": [],
        "classification": "reversible",
        "required_slots": slot_names,
    })
    provider = _stub_llm_provider(raw)

    # Memory has nothing relevant to these slots
    memory = [_note_with_body("The weather is nice today outside")]

    plan = Planner(model_provider=provider).plan("do something", memory_context=memory)
    step = plan.steps[0]

    # All slots must still be required (none resolved)
    for slot in slot_names:
        assert slot in step.required_slots, (
            f"Slot '{slot}' should remain in required_slots when memory cannot resolve it"
        )
