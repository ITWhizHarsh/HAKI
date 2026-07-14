"""
Unit and property-based tests for the Execution Engine plan→gate→execute→verify
loop with parallelism and postcondition checks (Task 23.1).

Covers:
  - Postcondition verification: success, failure, exception
  - PlanCompletionEvent emission listing executed step IDs (Req 21.8)
  - Parallel execution of independent steps (Req 17.2)
  - Dependency-respecting sequential execution
  - Integration: safety gate + postconditions together
  - awaiting_confirmation events surface for consequential steps

**Validates: Requirements 17.2, 21.8**
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from core.planner import (
    Actuator,
    CommandPlan,
    Step,
    StepClassification,
    StepStatus,
)
from core.execution import (
    ConfirmationRequest,
    ConfirmationResult,
    ExecutionEngine,
    ExecutionReport,
    PlanCompletionEvent,
    SafetyGate,
    StepEvent,
    StepEventType,
)
from core.execution.execution_engine import collect_events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(
    step_id: str = "s1",
    intent: str = "do something",
    classification: StepClassification = StepClassification.REVERSIBLE,
    depends_on: list[str] | None = None,
    postcondition=None,
) -> Step:
    return Step(
        id=step_id,
        intent=intent,
        actuator=Actuator.INTERNAL,
        classification=classification,
        depends_on=depends_on or [],
        postcondition=postcondition,
    )


def _plan(*steps: Step, command: str = "test") -> CommandPlan:
    return CommandPlan(origin_command=command, steps=list(steps))


async def _always_confirm(req: ConfirmationRequest) -> ConfirmationResult:
    return ConfirmationResult.CONFIRMED


async def _always_reject(req: ConfirmationRequest) -> ConfirmationResult:
    return ConfirmationResult.REJECTED


def _get_completion_event(events: list[StepEvent]) -> PlanCompletionEvent:
    """Extract the PlanCompletionEvent from a list of StepEvents."""
    plan_complete = next(
        e for e in events if e.event_type == StepEventType.PLAN_COMPLETE
    )
    report: ExecutionReport = plan_complete.data
    assert report.completion_event is not None, (
        "PLAN_COMPLETE event should carry an ExecutionReport with completion_event set (Req 21.8)"
    )
    return report.completion_event


# ===========================================================================
# Postcondition verification
# ===========================================================================


class TestPostconditionVerification:
    """
    Tests that the execution engine verifies postconditions after each step
    and marks steps FAILED (stopping dependents) when postconditions fail.
    """

    @pytest.mark.asyncio
    async def test_passing_postcondition_step_completes(self):
        """A postcondition that returns True should not prevent completion."""
        executed: list[str] = []

        async def actuator(step: Step) -> str:
            executed.append(step.id)
            return "ok"

        step = _step("s1", postcondition=lambda r: r == "ok")
        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=actuator)
        events = await collect_events(engine, _plan(step))

        event_types = [e.event_type for e in events]
        assert StepEventType.COMPLETED in event_types
        assert StepEventType.FAILED not in event_types
        assert "s1" in executed

    @pytest.mark.asyncio
    async def test_failing_postcondition_marks_step_failed(self):
        """A postcondition returning False should mark the step FAILED."""
        async def actuator(step: Step) -> str:
            return "wrong_value"

        step = _step("s1", postcondition=lambda r: r == "expected_value")
        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=actuator)
        events = await collect_events(engine, _plan(step))

        event_types = [e.event_type for e in events]
        assert StepEventType.FAILED in event_types
        assert StepEventType.COMPLETED not in event_types

    @pytest.mark.asyncio
    async def test_postcondition_exception_marks_step_failed(self):
        """A postcondition that raises an exception should mark the step FAILED."""
        async def actuator(step: Step) -> None:
            return None

        def bad_postcondition(result: Any) -> bool:
            raise RuntimeError("postcondition blew up")

        step = _step("s1", postcondition=bad_postcondition)
        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=actuator)
        events = await collect_events(engine, _plan(step))

        event_types = [e.event_type for e in events]
        assert StepEventType.FAILED in event_types
        assert StepEventType.COMPLETED not in event_types

    @pytest.mark.asyncio
    async def test_postcondition_failure_stops_dependents(self):
        """
        When a step's postcondition fails, all transitive dependents
        must be SKIPPED (not started).
        """
        executed: list[str] = []

        async def actuator(step: Step) -> str:
            executed.append(step.id)
            return "bad"

        s1 = _step("s1", postcondition=lambda r: r == "good")
        s2 = _step("s2", depends_on=["s1"])  # dependent — must be skipped
        s3 = _step("s3", depends_on=["s2"])  # transitive dependent — must be skipped

        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=actuator)
        events = await collect_events(engine, _plan(s1, s2, s3))

        skipped = {e.step_id for e in events if e.event_type == StepEventType.SKIPPED}
        assert "s2" in skipped, "Direct dependent should be skipped on postcondition failure"
        assert "s3" in skipped, "Transitive dependent should be skipped on postcondition failure"
        assert "s2" not in executed
        assert "s3" not in executed

    @pytest.mark.asyncio
    async def test_no_postcondition_step_always_completes(self):
        """Steps without a postcondition always complete if actuator succeeds."""
        step = _step("s1")  # no postcondition
        engine = ExecutionEngine(safety_gate=SafetyGate())
        events = await collect_events(engine, _plan(step))
        event_types = [e.event_type for e in events]
        assert StepEventType.COMPLETED in event_types
        assert StepEventType.FAILED not in event_types

    @pytest.mark.asyncio
    async def test_independent_step_runs_despite_sibling_postcondition_failure(self):
        """
        A step independent of a failing step should still run even if
        its sibling's postcondition fails.
        """
        executed: list[str] = []

        async def actuator(step: Step) -> str:
            executed.append(step.id)
            return "result"

        s_fail = _step("fail", postcondition=lambda r: False)  # will fail
        s_indep = _step("indep")  # independent — should still run

        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=actuator)
        events = await collect_events(engine, _plan(s_fail, s_indep))

        assert "indep" in executed, "Independent step must run despite sibling postcondition failure"

    @pytest.mark.asyncio
    async def test_postcondition_receives_actuator_output(self):
        """The postcondition callable receives the actuator's return value."""
        received_results: list[Any] = []

        async def actuator(step: Step) -> dict:
            return {"status": "done", "count": 42}

        def capture_postcondition(result: Any) -> bool:
            received_results.append(result)
            return True

        step = _step("s1", postcondition=capture_postcondition)
        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=actuator)
        await collect_events(engine, _plan(step))

        assert len(received_results) == 1
        assert received_results[0] == {"status": "done", "count": 42}

    @pytest.mark.asyncio
    async def test_postcondition_failure_in_report(self):
        """Failed postcondition steps appear in report.failed."""
        async def actuator(step: Step) -> str:
            return "bad"

        step = _step("s1", postcondition=lambda r: r == "good")
        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=actuator)
        events = await collect_events(engine, _plan(step))

        plan_complete = next(e for e in events if e.event_type == StepEventType.PLAN_COMPLETE)
        report: ExecutionReport = plan_complete.data
        assert any(s.id == "s1" for s in report.failed)


# ===========================================================================
# PlanCompletionEvent emission (Req 21.8)
# ===========================================================================


class TestPlanCompletionEvent:
    """
    Tests that a PlanCompletionEvent is emitted at plan completion
    listing all executed, not-performed, and failed step IDs (Req 21.8).
    """

    @pytest.mark.asyncio
    async def test_plan_completion_event_emitted(self):
        """PLAN_COMPLETE event should carry a PlanCompletionEvent (Req 21.8)."""
        engine = ExecutionEngine(safety_gate=SafetyGate())
        plan = _plan(_step("s1"))
        events = await collect_events(engine, plan)
        completion = _get_completion_event(events)
        assert isinstance(completion, PlanCompletionEvent)

    @pytest.mark.asyncio
    async def test_executed_step_ids_in_completion_event(self):
        """Completed step IDs must appear in executed_step_ids (Req 21.8)."""
        engine = ExecutionEngine(safety_gate=SafetyGate())
        plan = _plan(
            _step("s1"),
            _step("s2"),
        )
        events = await collect_events(engine, plan)
        completion = _get_completion_event(events)
        assert "s1" in completion.executed_step_ids
        assert "s2" in completion.executed_step_ids

    @pytest.mark.asyncio
    async def test_not_performed_step_ids_in_completion_event(self):
        """Skipped step IDs must appear in not_performed_step_ids."""
        gate = SafetyGate(confirmation_callback=_always_reject)
        engine = ExecutionEngine(safety_gate=gate)
        plan = _plan(
            _step("bad", classification=StepClassification.CONSEQUENTIAL),
            _step("dep", depends_on=["bad"]),
        )
        events = await collect_events(engine, plan)
        completion = _get_completion_event(events)
        assert "bad" in completion.not_performed_step_ids
        assert "dep" in completion.not_performed_step_ids

    @pytest.mark.asyncio
    async def test_failed_steps_in_completion_event(self):
        """Failed step IDs must appear in failed_steps with reasons."""
        async def failing_actuator(step: Step):
            raise RuntimeError("oops")

        engine = ExecutionEngine(
            safety_gate=SafetyGate(),
            actuator_callback=failing_actuator,
        )
        plan = _plan(_step("f1"))
        events = await collect_events(engine, plan)
        completion = _get_completion_event(events)
        failed_ids = [step_id for step_id, _ in completion.failed_steps]
        assert "f1" in failed_ids

    @pytest.mark.asyncio
    async def test_postcondition_failed_step_in_completion_event(self):
        """Postcondition-failed steps appear in completion event's failed_steps."""
        async def actuator(step: Step) -> str:
            return "wrong"

        step = _step("s1", postcondition=lambda r: r == "right")
        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=actuator)
        events = await collect_events(engine, _plan(step))
        completion = _get_completion_event(events)
        failed_ids = [sid for sid, _ in completion.failed_steps]
        assert "s1" in failed_ids

    @pytest.mark.asyncio
    async def test_completion_event_all_completed_true(self):
        """all_completed is True when every step succeeds."""
        engine = ExecutionEngine(safety_gate=SafetyGate())
        plan = _plan(_step("s1"), _step("s2"))
        events = await collect_events(engine, plan)
        completion = _get_completion_event(events)
        assert completion.all_completed is True

    @pytest.mark.asyncio
    async def test_completion_event_all_completed_false_on_failure(self):
        """all_completed is False when a step fails."""
        async def fail(step: Step):
            raise RuntimeError("fail")

        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=fail)
        plan = _plan(_step("s1"))
        events = await collect_events(engine, plan)
        completion = _get_completion_event(events)
        assert completion.all_completed is False

    @pytest.mark.asyncio
    async def test_empty_plan_completion_event(self):
        """Empty plans produce a valid PlanCompletionEvent with empty lists."""
        engine = ExecutionEngine(safety_gate=SafetyGate())
        plan = CommandPlan(origin_command="empty")
        events = await collect_events(engine, plan)
        completion = _get_completion_event(events)
        assert completion.executed_step_ids == []
        assert completion.not_performed_step_ids == []
        assert completion.failed_steps == []


# ===========================================================================
# Parallel execution of independent steps (Req 17.2)
# ===========================================================================


class TestParallelExecution:
    """
    Tests that independent steps (no mutual dependency) run in parallel,
    while dependent steps wait for their prerequisites (Req 17.2).
    """

    @pytest.mark.asyncio
    async def test_independent_steps_run_concurrently(self):
        """
        Independent steps should overlap in time (run in parallel).
        We verify this by giving each step a short sleep and asserting
        the total elapsed time is closer to max(delays) than sum(delays).
        """
        start_times: dict[str, float] = {}
        end_times: dict[str, float] = {}

        async def timed_actuator(step: Step) -> None:
            start_times[step.id] = time.monotonic()
            await asyncio.sleep(0.05)  # 50ms delay per step
            end_times[step.id] = time.monotonic()

        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=timed_actuator)
        plan = _plan(
            _step("s1"),  # independent
            _step("s2"),  # independent
            _step("s3"),  # independent
        )

        wall_start = time.monotonic()
        events = await collect_events(engine, plan)
        wall_elapsed = time.monotonic() - wall_start

        # If they ran sequentially: ~150ms; in parallel: ~50ms.
        # Allow generous margin for test infrastructure overhead.
        assert wall_elapsed < 0.20, (
            f"Expected parallel execution (<200ms), but took {wall_elapsed:.3f}s. "
            "Independent steps must run concurrently (Req 17.2)."
        )

        # All three steps should have completed
        completion = _get_completion_event(events)
        assert set(completion.executed_step_ids) == {"s1", "s2", "s3"}

    @pytest.mark.asyncio
    async def test_dependent_step_waits_for_predecessor(self):
        """A step must not start until all its dependencies have COMPLETED."""
        execution_order: list[str] = []

        async def track_actuator(step: Step) -> None:
            execution_order.append(step.id)

        engine = ExecutionEngine(
            safety_gate=SafetyGate(),
            actuator_callback=track_actuator,
        )
        plan = _plan(
            _step("s1"),
            _step("s2", depends_on=["s1"]),
        )
        await collect_events(engine, plan)

        assert execution_order.index("s1") < execution_order.index("s2"), (
            "s2 must not execute before s1 (dependency ordering, Req 17.2)"
        )

    @pytest.mark.asyncio
    async def test_diamond_dependency_all_complete(self):
        """
        Diamond: a → (b, c) → d.
        b and c run in parallel; d waits for both.
        All four steps must complete.
        """
        executed: list[str] = []

        async def track_actuator(step: Step) -> None:
            executed.append(step.id)

        engine = ExecutionEngine(
            safety_gate=SafetyGate(),
            actuator_callback=track_actuator,
        )
        plan = _plan(
            _step("a"),
            _step("b", depends_on=["a"]),
            _step("c", depends_on=["a"]),
            _step("d", depends_on=["b", "c"]),
        )
        events = await collect_events(engine, plan)

        completion = _get_completion_event(events)
        assert set(completion.executed_step_ids) == {"a", "b", "c", "d"}

        # b and c must come after a
        assert executed.index("a") < executed.index("b")
        assert executed.index("a") < executed.index("c")
        # d must come after both b and c
        assert executed.index("b") < executed.index("d")
        assert executed.index("c") < executed.index("d")

    @pytest.mark.asyncio
    async def test_completed_events_for_all_parallel_steps(self):
        """COMPLETED events should be emitted for each independent step."""
        engine = ExecutionEngine(safety_gate=SafetyGate())
        plan = _plan(_step("a"), _step("b"), _step("c"))
        events = await collect_events(engine, plan)

        completed_ids = {
            e.step_id for e in events if e.event_type == StepEventType.COMPLETED
        }
        assert completed_ids == {"a", "b", "c"}


# ===========================================================================
# Awaiting-confirmation events for consequential steps
# ===========================================================================


class TestAwaitingConfirmationEvents:
    """
    Tests that awaiting_confirmation events are emitted when the safety gate
    pauses execution at consequential steps.
    """

    @pytest.mark.asyncio
    async def test_awaiting_confirmation_emitted_for_consequential(self):
        gate = SafetyGate(confirmation_callback=_always_confirm)
        engine = ExecutionEngine(safety_gate=gate)
        plan = _plan(_step("c1", classification=StepClassification.CONSEQUENTIAL))
        events = await collect_events(engine, plan)
        event_types = [e.event_type for e in events]
        assert StepEventType.AWAITING_CONFIRMATION in event_types

    @pytest.mark.asyncio
    async def test_no_awaiting_confirmation_for_reversible(self):
        gate = SafetyGate(confirmation_callback=_always_confirm)
        engine = ExecutionEngine(safety_gate=gate)
        plan = _plan(_step("r1", classification=StepClassification.REVERSIBLE))
        events = await collect_events(engine, plan)
        event_types = [e.event_type for e in events]
        assert StepEventType.AWAITING_CONFIRMATION not in event_types


# ===========================================================================
# Property-based tests (Req 17.2, 21.8)
# ===========================================================================


@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(n_steps=st.integers(min_value=1, max_value=6))
def test_property_plan_completion_event_always_emitted(n_steps: int) -> None:
    """
    **Validates: Requirements 17.2, 21.8**

    Property: for any plan with N independent reversible steps, a
    PlanCompletionEvent is always emitted and its executed_step_ids
    contains all N step IDs.
    """
    steps = [
        Step(
            id=f"s{i}",
            intent=f"step {i}",
            actuator=Actuator.INTERNAL,
            classification=StepClassification.REVERSIBLE,
        )
        for i in range(n_steps)
    ]
    plan = CommandPlan(origin_command="test", steps=steps)
    engine = ExecutionEngine(safety_gate=SafetyGate())

    events = asyncio.get_event_loop().run_until_complete(collect_events(engine, plan))

    plan_complete_events = [
        e for e in events if e.event_type == StepEventType.PLAN_COMPLETE
    ]
    assert len(plan_complete_events) >= 1, (
        "PLAN_COMPLETE event must always be emitted (Req 21.8)"
    )

    report: ExecutionReport = plan_complete_events[0].data
    assert report.completion_event is not None, (
        "completion_event must be set on ExecutionReport (Req 21.8)"
    )

    completion: PlanCompletionEvent = report.completion_event
    expected_ids = {s.id for s in steps}
    actual_ids = set(completion.executed_step_ids)
    assert actual_ids == expected_ids, (
        f"All {n_steps} step IDs must appear in executed_step_ids. "
        f"Expected {expected_ids}, got {actual_ids}."
    )


@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(chain_length=st.integers(min_value=2, max_value=6))
def test_property_dependency_ordering_respected(chain_length: int) -> None:
    """
    **Validates: Requirements 17.2**

    Property: for a linear chain of N steps (each depending on the
    previous), the steps execute in dependency order and all appear
    in executed_step_ids.
    """
    steps = [
        Step(
            id=f"s{i}",
            intent=f"step {i}",
            actuator=Actuator.INTERNAL,
            classification=StepClassification.REVERSIBLE,
            depends_on=[f"s{i-1}"] if i > 0 else [],
        )
        for i in range(chain_length)
    ]

    execution_order: list[str] = []

    async def track(step: Step) -> None:
        execution_order.append(step.id)

    async def run():
        plan = CommandPlan(origin_command="chain test", steps=steps)
        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=track)
        return await collect_events(engine, plan)

    events = asyncio.get_event_loop().run_until_complete(run())

    # Every step must have executed
    assert len(execution_order) == chain_length, (
        f"All {chain_length} chain steps must execute, got {execution_order}"
    )

    # The order must respect dependencies
    for i in range(1, chain_length):
        assert execution_order.index(f"s{i-1}") < execution_order.index(f"s{i}"), (
            f"s{i} must execute after s{i-1} (dependency chain)"
        )

    # All must be in completion event
    plan_complete = next(e for e in events if e.event_type == StepEventType.PLAN_COMPLETE)
    completion: PlanCompletionEvent = plan_complete.data.completion_event
    assert set(completion.executed_step_ids) == {s.id for s in steps}


@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(n_pass=st.integers(min_value=0, max_value=4), n_fail=st.integers(min_value=1, max_value=4))
def test_property_failed_postconditions_in_completion_event(
    n_pass: int, n_fail: int
) -> None:
    """
    **Validates: Requirements 17.2, 21.8**

    Property: for any mix of steps with passing and failing postconditions,
    every step with a failing postcondition appears in completion_event.failed_steps
    and every step with a passing postcondition appears in executed_step_ids
    (assuming no dependencies block them).
    """
    passing_steps = [
        Step(
            id=f"pass_{i}",
            intent="passing step",
            actuator=Actuator.INTERNAL,
            classification=StepClassification.REVERSIBLE,
            postcondition=lambda r: True,
        )
        for i in range(n_pass)
    ]
    failing_steps = [
        Step(
            id=f"fail_{i}",
            intent="failing step",
            actuator=Actuator.INTERNAL,
            classification=StepClassification.REVERSIBLE,
            postcondition=lambda r: False,
        )
        for i in range(n_fail)
    ]

    all_steps = passing_steps + failing_steps
    plan = CommandPlan(origin_command="mixed test", steps=all_steps)
    engine = ExecutionEngine(safety_gate=SafetyGate())

    events = asyncio.get_event_loop().run_until_complete(collect_events(engine, plan))

    plan_complete = next(e for e in events if e.event_type == StepEventType.PLAN_COMPLETE)
    completion: PlanCompletionEvent = plan_complete.data.completion_event

    # All failing steps must be in failed_steps
    failed_ids = {sid for sid, _ in completion.failed_steps}
    for step in failing_steps:
        assert step.id in failed_ids, (
            f"Failing step {step.id} must appear in completion_event.failed_steps"
        )

    # All passing steps must be in executed_step_ids
    for step in passing_steps:
        assert step.id in completion.executed_step_ids, (
            f"Passing step {step.id} must appear in completion_event.executed_step_ids"
        )
