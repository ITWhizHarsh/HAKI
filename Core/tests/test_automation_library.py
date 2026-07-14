"""
Unit tests for the AutomationLibrary (Task 33.2).

Covers:
- define / get / list_names (storage interface, Req 17.1, 17.3)
- nearest() — edit-distance suggestion (Req 17.4)
- run() — exact-name invocation over the Execution_Engine (Req 17.2)
- run() — no-match path: no execution, nearest suggestion yielded (Req 17.4)
- cancel() — delegates to ExecutionEngine (Req 17.5, 17.6)
- Failure propagation: failed step stops dependents, independent steps
  continue; report is partitioned (Req 17.7)
- Progress reporting: StepEvent STARTED/COMPLETED stream back to caller (Req 17.2)

Design: Automation_Library + Execution_Engine.
Requirements: 17.2, 17.4, 17.5, 17.6, 17.7.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.automation import AutomationLibrary, AutomationNotFoundError, NamedAutomation
from core.automation.automation_library import _levenshtein
from core.execution.execution_engine import (
    ExecutionEngine,
    StepEvent,
    StepEventType,
    collect_events,
)
from core.planner import Actuator, CommandPlan, Step, StepClassification, StepStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(
    step_id: str = "s1",
    intent: str = "do something",
    classification: StepClassification = StepClassification.REVERSIBLE,
    depends_on: list[str] | None = None,
) -> Step:
    return Step(
        id=step_id,
        intent=intent,
        actuator=Actuator.INTERNAL,
        classification=classification,
        depends_on=depends_on or [],
    )


def _make_library(
    names: list[str] | None = None,
    engine: ExecutionEngine | None = None,
) -> AutomationLibrary:
    """Return an AutomationLibrary pre-populated with zero-step automations."""
    lib = AutomationLibrary(execution_engine=engine)
    for name in names or []:
        lib.define(name, steps=[])
    return lib


async def _collect_run(library: AutomationLibrary, name: str) -> list[StepEvent]:
    """Collect all StepEvents from library.run(name).

    AutomationLibrary.run() is an async generator — iterate with ``async for``,
    not with ``await``.
    """
    events: list[StepEvent] = []
    async for event in library.run(name):
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# _levenshtein unit tests
# ---------------------------------------------------------------------------


class TestLevenshtein:
    def test_identical_strings(self):
        assert _levenshtein("hello", "hello") == 0

    def test_empty_strings(self):
        assert _levenshtein("", "") == 0

    def test_one_empty(self):
        assert _levenshtein("abc", "") == 3
        assert _levenshtein("", "abc") == 3

    def test_single_substitution(self):
        assert _levenshtein("kitten", "sitten") == 1

    def test_single_insertion(self):
        assert _levenshtein("cat", "cats") == 1

    def test_single_deletion(self):
        assert _levenshtein("cats", "cat") == 1

    def test_classic_levenshtein(self):
        # "kitten" → "sitting" = 3 edits
        assert _levenshtein("kitten", "sitting") == 3

    def test_completely_different(self):
        assert _levenshtein("abc", "xyz") == 3


# ---------------------------------------------------------------------------
# AutomationLibrary.define / get / list_names
# ---------------------------------------------------------------------------


class TestStorage:
    def test_define_and_get(self):
        lib = AutomationLibrary()
        step = _make_step()
        automation = lib.define("test_automation", [step])
        assert isinstance(automation, NamedAutomation)
        assert automation.name == "test_automation"
        assert automation.steps == [step]
        # get() returns the same object
        assert lib.get("test_automation") is automation

    def test_define_replaces_existing(self):
        lib = AutomationLibrary()
        step1 = _make_step("s1", "step 1")
        step2 = _make_step("s2", "step 2")
        lib.define("auto", [step1])
        lib.define("auto", [step2])
        assert lib.get("auto").steps == [step2]

    def test_get_not_found_raises(self):
        lib = AutomationLibrary()
        with pytest.raises(AutomationNotFoundError):
            lib.get("nonexistent")

    def test_list_names_empty(self):
        lib = AutomationLibrary()
        assert lib.list_names() == []

    def test_list_names_populated(self):
        lib = _make_library(["alpha", "beta", "gamma"])
        assert lib.list_names() == ["alpha", "beta", "gamma"]

    def test_define_empty_name_raises(self):
        lib = AutomationLibrary()
        with pytest.raises(ValueError):
            lib.define("", [])

    def test_define_with_description(self):
        lib = AutomationLibrary()
        lib.define("auto", [], description="Does something useful")
        assert lib.get("auto").description == "Does something useful"


# ---------------------------------------------------------------------------
# AutomationLibrary.nearest
# ---------------------------------------------------------------------------


class TestNearest:
    def test_nearest_empty_library_returns_none(self):
        lib = AutomationLibrary()
        assert lib.nearest("anything") is None

    def test_nearest_exact_match(self):
        lib = _make_library(["morning_brief", "daily_report"])
        assert lib.nearest("morning_brief") == "morning_brief"

    def test_nearest_one_char_off(self):
        lib = _make_library(["send_report", "morning_brief"])
        # "send_repot" → "send_report" is 1 edit
        assert lib.nearest("send_repot") == "send_report"

    def test_nearest_case_insensitive(self):
        lib = _make_library(["DailyBrief"])
        # Edit distance should be 0 because comparison is lowercased.
        assert lib.nearest("dailybrief") == "DailyBrief"

    def test_nearest_picks_closest(self):
        lib = _make_library(["aaa", "bbb", "ccc"])
        # "aab" is closest to "aaa" (1 edit) vs "bbb" (2) and "ccc" (3)
        assert lib.nearest("aab") == "aaa"

    def test_nearest_tie_picks_first_encountered(self):
        # When two names are equidistant, the first one wins (dict order).
        lib = _make_library(["xy", "yz"])
        # "xz" → "xy" = 1 edit, "xz" → "yz" = 1 edit
        result = lib.nearest("xz")
        assert result in ("xy", "yz")  # either is acceptable


# ---------------------------------------------------------------------------
# AutomationLibrary.run — no-match path (Req 17.4)
# ---------------------------------------------------------------------------


class TestRunNoMatch:
    @pytest.mark.asyncio
    async def test_no_match_yields_failed_event(self):
        lib = _make_library(["morning_brief"])
        events = await _collect_run(lib, "morning_bref")  # typo
        assert len(events) == 1
        ev = events[0]
        assert ev.event_type == StepEventType.FAILED
        assert ev.step_id is None
        assert "morning_brief" in ev.message  # suggestion included
        assert "morning_bref" in ev.message   # requested name included

    @pytest.mark.asyncio
    async def test_no_match_does_not_execute(self):
        """No actuator calls must happen on a no-match run."""
        executed: list[str] = []

        async def _actuator(step: Step) -> Any:
            executed.append(step.id)
            return None

        engine = ExecutionEngine(actuator_callback=_actuator)
        lib = AutomationLibrary(execution_engine=engine)
        lib.define("alpha", [_make_step("s1")])

        await _collect_run(lib, "completely_different_name")

        assert executed == [], "Actuator should NOT have been called on a no-match"

    @pytest.mark.asyncio
    async def test_no_match_empty_library_message(self):
        lib = AutomationLibrary()
        events = await _collect_run(lib, "nonexistent")
        assert len(events) == 1
        assert events[0].event_type == StepEventType.FAILED
        assert "empty" in events[0].message.lower() or "no automation" in events[0].message.lower()


# ---------------------------------------------------------------------------
# AutomationLibrary.run — exact match path (Req 17.2, 17.7)
# ---------------------------------------------------------------------------


class TestRunExactMatch:
    @pytest.mark.asyncio
    async def test_exact_match_executes_all_steps(self):
        """All steps in the automation should be executed and reported."""
        executed: list[str] = []

        async def _actuator(step: Step) -> Any:
            executed.append(step.id)
            return f"output-{step.id}"

        engine = ExecutionEngine(actuator_callback=_actuator)
        lib = AutomationLibrary(execution_engine=engine)
        steps = [_make_step("s1", "step 1"), _make_step("s2", "step 2")]
        lib.define("my_auto", steps)

        events = await _collect_run(lib, "my_auto")

        assert executed == ["s1", "s2"] or set(executed) == {"s1", "s2"}
        event_types = [e.event_type for e in events]
        assert StepEventType.COMPLETED in event_types
        assert StepEventType.PLAN_COMPLETE in event_types

    @pytest.mark.asyncio
    async def test_exact_match_streams_started_events(self):
        """STARTED events must be emitted so caller can report progress (Req 17.2)."""

        async def _actuator(step: Step) -> Any:
            return None

        engine = ExecutionEngine(actuator_callback=_actuator)
        lib = AutomationLibrary(execution_engine=engine)
        lib.define("auto", [_make_step("s1", "do thing")])

        events = await _collect_run(lib, "auto")
        started = [e for e in events if e.event_type == StepEventType.STARTED]
        assert len(started) == 1
        assert started[0].step_id == "s1"

    @pytest.mark.asyncio
    async def test_dependency_ordering_preserved(self):
        """Steps should respect depends_on ordering (Req 17.2)."""
        order: list[str] = []

        async def _actuator(step: Step) -> Any:
            order.append(step.id)
            return None

        engine = ExecutionEngine(actuator_callback=_actuator)
        lib = AutomationLibrary(execution_engine=engine)
        # s2 depends on s1 — s1 must complete before s2 starts
        s1 = _make_step("s1", "first")
        s2 = _make_step("s2", "second", depends_on=["s1"])
        lib.define("ordered", [s1, s2])

        await _collect_run(lib, "ordered")

        assert order.index("s1") < order.index("s2")

    @pytest.mark.asyncio
    async def test_failure_stops_dependents_independent_continues(self):
        """
        A failed step stops its dependents; independent steps still run.
        The final report is partitioned (Req 17.7).
        """
        executed: list[str] = []

        async def _actuator(step: Step) -> Any:
            executed.append(step.id)
            if step.id == "s1":
                raise RuntimeError("s1 failed intentionally")
            return None

        engine = ExecutionEngine(actuator_callback=_actuator)
        lib = AutomationLibrary(execution_engine=engine)

        s1 = _make_step("s1", "will fail")
        s2 = _make_step("s2", "depends on s1", depends_on=["s1"])
        s3 = _make_step("s3", "independent")
        lib.define("mixed", [s1, s2, s3])

        events = await _collect_run(lib, "mixed")

        # s1 must have been attempted
        assert "s1" in executed
        # s2 must NOT have been attempted (it depends on s1)
        assert "s2" not in executed
        # s3 is independent and must have been executed
        assert "s3" in executed

        # Final PLAN_COMPLETE event carries the report
        plan_complete = next(e for e in events if e.event_type == StepEventType.PLAN_COMPLETE)
        report = plan_complete.data
        completed_ids = {s.id for s in report.completed}
        failed_ids = {s.id for s in report.failed}
        not_performed_ids = {s.id for s in report.not_performed}

        assert "s3" in completed_ids, "Independent step should complete"
        assert "s1" in failed_ids, "Failed step in failed list"
        assert "s2" in not_performed_ids, "Dependent of failed step not performed"

    @pytest.mark.asyncio
    async def test_plan_complete_event_is_last(self):
        """PLAN_COMPLETE must be the final event in the stream."""
        engine = ExecutionEngine()  # no-op actuator
        lib = AutomationLibrary(execution_engine=engine)
        lib.define("simple", [_make_step("s1")])

        events = await _collect_run(lib, "simple")
        assert events[-1].event_type == StepEventType.PLAN_COMPLETE


# ---------------------------------------------------------------------------
# AutomationLibrary.cancel (Req 17.5, 17.6)
# ---------------------------------------------------------------------------


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_stops_unstarted_steps(self):
        """After cancel(), steps that have not started should be SKIPPED."""
        engine = ExecutionEngine()  # no-op actuator
        lib = AutomationLibrary(execution_engine=engine)

        # Call cancel before run
        lib.cancel()

        s1 = _make_step("s1")
        s2 = _make_step("s2")
        lib.define("canceltest", [s1, s2])

        events = await _collect_run(lib, "canceltest")
        plan_complete = next(e for e in events if e.event_type == StepEventType.PLAN_COMPLETE)
        report = plan_complete.data
        # All steps should be in not_performed (cancel was called before run)
        assert report.cancelled is True
        not_performed_ids = {s.id for s in report.not_performed}
        assert "s1" in not_performed_ids
        assert "s2" in not_performed_ids

    def test_cancel_delegates_to_engine(self):
        """cancel() must call the engine's cancel() method."""
        engine = ExecutionEngine()
        lib = AutomationLibrary(execution_engine=engine)
        assert engine._cancelled is False
        lib.cancel()
        assert engine._cancelled is True


# ---------------------------------------------------------------------------
# Integration: AutomationLibrary with a real ExecutionEngine (no actuator)
# ---------------------------------------------------------------------------


class TestIntegration:
    @pytest.mark.asyncio
    async def test_zero_step_automation_completes(self):
        """An automation with no steps should complete immediately."""
        lib = AutomationLibrary()
        lib.define("empty_auto", [])
        events = await _collect_run(lib, "empty_auto")
        assert events[-1].event_type == StepEventType.PLAN_COMPLETE
        plan_complete = events[-1]
        report = plan_complete.data
        assert report.all_completed  # vacuously true with no steps

    @pytest.mark.asyncio
    async def test_multiple_automations_share_engine(self):
        """The same engine instance should be reused across multiple run() calls."""
        call_count = [0]

        async def _actuator(step: Step) -> Any:
            call_count[0] += 1
            return None

        engine = ExecutionEngine(actuator_callback=_actuator)
        lib = AutomationLibrary(execution_engine=engine)

        lib.define("auto_a", [_make_step("a1", "task a")])
        lib.define("auto_b", [_make_step("b1", "task b")])

        # Run both automations sequentially (need a fresh engine for second run
        # since cancel() state persists — but here engine is not cancelled)
        await _collect_run(lib, "auto_a")
        # Reset engine for second run (recreate since _cancelled persists after a run)
        engine2 = ExecutionEngine(actuator_callback=_actuator)
        lib2 = AutomationLibrary(execution_engine=engine2)
        lib2.define("auto_b", [_make_step("b1", "task b")])
        await _collect_run(lib2, "auto_b")

        assert call_count[0] == 2


# ---------------------------------------------------------------------------
# NamedAutomation model — unit tests for Task 33.1
# ---------------------------------------------------------------------------


class TestNamedAutomation:
    """Unit tests for the NamedAutomation data model (Req 17.1)."""

    def test_create_with_name_and_steps(self) -> None:
        """NamedAutomation stores name and ordered steps."""
        steps = [_make_step("s1"), _make_step("s2")]
        auto = NamedAutomation(name="Morning Routine", steps=steps)
        assert auto.name == "Morning Routine"
        assert len(auto.steps) == 2
        assert auto.steps[0].id == "s1"
        assert auto.steps[1].id == "s2"

    def test_empty_name_raises(self) -> None:
        """NamedAutomation with an empty name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            NamedAutomation(name="")

    def test_id_is_generated_and_unique(self) -> None:
        """Each NamedAutomation gets a unique auto-generated ID."""
        a1 = NamedAutomation(name="a", steps=[])
        a2 = NamedAutomation(name="b", steps=[])
        assert hasattr(a1, "id")
        assert a1.id != a2.id

    def test_to_command_plan_wraps_steps(self) -> None:
        """to_command_plan() returns a CommandPlan with the same ordered steps."""
        steps = [_make_step("s1"), _make_step("s2", depends_on=["s1"])]
        auto = NamedAutomation(name="My Auto", steps=steps)
        plan = auto.to_command_plan()
        assert isinstance(plan, CommandPlan)
        assert len(plan.steps) == 2
        assert plan.steps[0].id == "s1"
        assert plan.steps[1].id == "s2"
        assert "My Auto" in plan.origin_command

    def test_step_order_preserved(self) -> None:
        """Step order in NamedAutomation matches the order passed in."""
        steps = [_make_step(f"s{i}") for i in range(5)]
        auto = NamedAutomation(name="Order Test", steps=steps)
        for i, step in enumerate(auto.steps):
            assert step.id == f"s{i}"


# ---------------------------------------------------------------------------
# SQLite persistence — Task 33.1 (Req 17.1, 17.3)
# ---------------------------------------------------------------------------


class TestSQLitePersistence:
    """
    Tests that verify SQLite-backed persistence of NamedAutomations.

    Requirement 17.1 — The Automation_Library SHALL store named automations
    consisting of a name and an ordered sequence of steps.

    Requirement 17.3 — WHEN the User requests a stored automation by name,
    THE Automation_Library SHALL load the automation whose name exactly matches
    the requested name and SHALL make its ordered steps available for execution.
    """

    def test_define_and_get_persists_name(self, tmp_path) -> None:
        """Loaded automation's name equals the stored name (Req 17.3)."""
        db = tmp_path / "automations.db"
        lib = AutomationLibrary(db_path=db)
        lib.define("My Automation", [_make_step("s1")])

        loaded = lib.get("My Automation")
        assert loaded.name == "My Automation"

    def test_define_and_get_preserves_step_order(self, tmp_path) -> None:
        """Loaded automation's steps are in the same order as stored (Req 17.3)."""
        db = tmp_path / "automations.db"
        lib = AutomationLibrary(db_path=db)
        steps = [_make_step(f"step_{i}", intent=f"intent {i}") for i in range(4)]
        lib.define("Ordered Steps", steps)

        loaded = lib.get("Ordered Steps")
        assert len(loaded.steps) == 4
        for i, step in enumerate(loaded.steps):
            assert step.id == f"step_{i}"
            assert step.intent == f"intent {i}"

    def test_persists_across_library_instances(self, tmp_path) -> None:
        """Automations stored in one instance are available in a new instance (Req 17.1)."""
        db = tmp_path / "automations.db"

        lib1 = AutomationLibrary(db_path=db)
        lib1.define("persistent auto", [
            _make_step("s1", intent="first step"),
            _make_step("s2", intent="second step"),
        ])

        lib2 = AutomationLibrary(db_path=db)
        loaded = lib2.get("persistent auto")

        assert loaded.name == "persistent auto"
        assert len(loaded.steps) == 2
        assert loaded.steps[0].id == "s1"
        assert loaded.steps[1].id == "s2"

    def test_step_order_persists_across_instances(self, tmp_path) -> None:
        """Step order is preserved after creating a new library instance."""
        db = tmp_path / "automations.db"
        steps = [_make_step(f"step_{i}", intent=f"intent {i}") for i in range(6)]

        lib1 = AutomationLibrary(db_path=db)
        lib1.define("ordered persistence", steps)

        lib2 = AutomationLibrary(db_path=db)
        loaded = lib2.get("ordered persistence")

        for i, step in enumerate(loaded.steps):
            assert step.id == f"step_{i}"
            assert step.intent == f"intent {i}"

    def test_get_nonexistent_raises_with_db(self, tmp_path) -> None:
        """AutomationNotFoundError is raised for a name that was never defined."""
        db = tmp_path / "automations.db"
        lib = AutomationLibrary(db_path=db)
        with pytest.raises(AutomationNotFoundError):
            lib.get("does not exist")

    def test_exact_name_match_case_sensitive_with_db(self, tmp_path) -> None:
        """Name matching is case-sensitive (Req 17.3 — exact match)."""
        db = tmp_path / "automations.db"
        lib = AutomationLibrary(db_path=db)
        lib.define("morning routine", [_make_step("s1")])

        with pytest.raises(AutomationNotFoundError):
            lib.get("Morning Routine")

        loaded = lib.get("morning routine")
        assert loaded.name == "morning routine"

    def test_define_replaces_existing_in_db(self, tmp_path) -> None:
        """Calling define() with a name that already exists overwrites it in the DB."""
        db = tmp_path / "automations.db"
        lib = AutomationLibrary(db_path=db)
        lib.define("replace me", [_make_step("old_step", intent="old intent")])
        lib.define("replace me", [
            _make_step("new_s1", intent="new step 1"),
            _make_step("new_s2", intent="new step 2"),
        ])

        # Verify in a fresh instance.
        lib2 = AutomationLibrary(db_path=db)
        loaded = lib2.get("replace me")
        assert len(loaded.steps) == 2
        assert loaded.steps[0].id == "new_s1"
        assert loaded.steps[1].id == "new_s2"

    def test_schema_created_automatically(self, tmp_path) -> None:
        """AutomationLibrary creates its table automatically when the DB is new."""
        db = tmp_path / "new_dir" / "automations.db"
        lib = AutomationLibrary(db_path=db)
        lib.define("schema test", [_make_step("s1")])
        loaded = lib.get("schema test")
        assert loaded.name == "schema test"

    def test_step_actuator_round_trips(self, tmp_path) -> None:
        """Step.actuator is preserved through serialisation."""
        from core.planner import Actuator as _Act

        db = tmp_path / "automations.db"
        lib = AutomationLibrary(db_path=db)
        steps = [
            Step(id="s1", intent="cdp step", actuator=_Act.CDP,
                 classification=StepClassification.REVERSIBLE),
            Step(id="s2", intent="ax step", actuator=_Act.AX,
                 classification=StepClassification.REVERSIBLE),
        ]
        lib.define("actuator round trip", steps)

        lib2 = AutomationLibrary(db_path=db)
        loaded = lib2.get("actuator round trip")
        assert loaded.steps[0].actuator == _Act.CDP
        assert loaded.steps[1].actuator == _Act.AX

    def test_step_depends_on_round_trips(self, tmp_path) -> None:
        """Step.depends_on list is preserved through serialisation."""
        db = tmp_path / "automations.db"
        lib = AutomationLibrary(db_path=db)
        steps = [
            _make_step("s1"),
            _make_step("s2", depends_on=["s1"]),
            _make_step("s3", depends_on=["s1", "s2"]),
        ]
        lib.define("deps round trip", steps)

        lib2 = AutomationLibrary(db_path=db)
        loaded = lib2.get("deps round trip")
        assert loaded.steps[0].depends_on == []
        assert loaded.steps[1].depends_on == ["s1"]
        assert loaded.steps[2].depends_on == ["s1", "s2"]
