"""
Unit and property-based tests for ExecutionEngine cancellation, named failure
propagation, and rejection propagation (Task 23.2).

Covers:
  - cancel() before execution: no steps run (Req 17.5)
  - cancel() mid-execution: report partitioned correctly (Req 17.6)
  - AppNotInstalledError stops transitive dependents (Req 21.9)
  - ElementNotFoundError stops transitive dependents (Req 21.12)
  - WebsiteUnreachableError stops transitive dependents (Req 21.13)
  - Generic actuator failure stops transitive dependents (Req 17.7)
  - Independent steps continue when an unrelated step fails (Req 17.7, 21.14)
  - Failure reason appears in completion_event.failed_steps (Req 21.8)
  - Rejection propagation (already in Task 23.1; verified here)

Property-based tests:
  Property 53 — Cancellation stops unstarted steps and partitions the report
    # Feature: haki-personal-ai-assistant, Property 53
    # Validates: Requirements 17.6

  Property 54 — Failure/rejection propagation
    # Feature: haki-personal-ai-assistant, Property 54
    # Validates: Requirements 17.7, 21.9, 21.12, 21.13, 21.14, 22.6
"""
from __future__ import annotations

import asyncio
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
    AppNotInstalledError,
    ElementNotFoundError,
    WebsiteUnreachableError,
)
from core.execution.execution_engine import collect_events


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _step(
    step_id: str,
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


def _plan(*steps: Step, command: str = "test") -> CommandPlan:
    return CommandPlan(origin_command=command, steps=list(steps))


def _get_report(events: list[StepEvent]) -> ExecutionReport:
    """Extract the ExecutionReport from the PLAN_COMPLETE event."""
    plan_complete = next(
        e for e in events if e.event_type == StepEventType.PLAN_COMPLETE
    )
    return plan_complete.data


def _get_completion(events: list[StepEvent]) -> PlanCompletionEvent:
    """Extract the PlanCompletionEvent from the PLAN_COMPLETE event."""
    report = _get_report(events)
    assert report.completion_event is not None
    return report.completion_event


async def _always_reject(req: ConfirmationRequest) -> ConfirmationResult:
    return ConfirmationResult.REJECTED


# ===========================================================================
# Cancellation Tests (Reqs 17.5, 17.6)
# ===========================================================================


class TestCancellation:
    """
    Tests for cancel() behavior: stopping unstarted steps and partitioning
    the final report into completed vs. not_performed (Reqs 17.5, 17.6).
    """

    @pytest.mark.asyncio
    async def test_cancel_before_execute_stops_all_steps(self):
        """
        Calling cancel() before execute() means no steps run at all.
        All steps land in not_performed_step_ids. (Req 17.5)
        """
        executed: list[str] = []

        async def actuator(step: Step) -> None:
            executed.append(step.id)

        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=actuator)
        engine.cancel()  # cancel BEFORE execute

        plan = _plan(_step("s1"), _step("s2"), _step("s3"))
        events = await collect_events(engine, plan)

        assert executed == [], "No steps should execute after pre-execution cancel"
        report = _get_report(events)
        assert report.cancelled is True
        completion = _get_completion(events)
        assert set(completion.not_performed_step_ids) == {"s1", "s2", "s3"}
        assert completion.executed_step_ids == []

    @pytest.mark.asyncio
    async def test_cancel_partitions_report(self):
        """
        cancel() mid-execution:
        - Steps that already completed appear in executed_step_ids.
        - Steps that were stopped appear in not_performed_step_ids.
        - report.cancelled == True. (Req 17.6)
        """
        # We use an event to synchronize: s1 runs, then we cancel,
        # then s2 should be stopped before it runs.
        s1_started = asyncio.Event()
        cancel_done = asyncio.Event()

        engine = ExecutionEngine(safety_gate=SafetyGate())

        async def actuator(step: Step) -> None:
            if step.id == "s1":
                s1_started.set()
                # Wait until cancel has been issued before completing s1
                await cancel_done.wait()
            # s2 would be a dependent of s1 — but we'll use independent
            # steps for simplicity to test the partition cleanly

        # s1 independent, s2 depends on s1 so it can't start until s1 finishes
        s1 = _step("s1")
        s2 = _step("s2", depends_on=["s1"])

        plan = _plan(s1, s2)

        async def run_and_cancel():
            stream = await engine.execute(plan)
            events = []
            async for event in stream:
                events.append(event)
                # Once s1 is in progress, trigger cancel
                if event.event_type == StepEventType.STARTED and event.step_id == "s1":
                    engine.cancel()
                    cancel_done.set()
            return events

        engine._async_actuator = actuator
        events = await run_and_cancel()

        report = _get_report(events)
        assert report.cancelled is True

        completion = _get_completion(events)
        # s1 started and completed
        assert "s1" in completion.executed_step_ids
        # s2 was never started — must be in not_performed
        assert "s2" in completion.not_performed_step_ids

    @pytest.mark.asyncio
    async def test_cancel_sets_cancelled_flag_true(self):
        """ExecutionReport.cancelled must be True after cancel()."""
        engine = ExecutionEngine(safety_gate=SafetyGate())
        engine.cancel()

        plan = _plan(_step("s1"))
        events = await collect_events(engine, plan)

        report = _get_report(events)
        assert report.cancelled is True

    @pytest.mark.asyncio
    async def test_cancel_stops_unstarted_independent_steps(self):
        """
        Pre-execution cancel stops ALL independent (no dependency) steps.
        """
        executed: list[str] = []

        async def actuator(step: Step) -> None:
            executed.append(step.id)

        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=actuator)
        engine.cancel()

        plan = _plan(
            _step("a"),
            _step("b"),
            _step("c"),
        )
        events = await collect_events(engine, plan)

        assert executed == []
        report = _get_report(events)
        assert report.cancelled is True
        assert len(_get_completion(events).not_performed_step_ids) == 3


# ===========================================================================
# Named failure propagation (Reqs 21.9, 21.12, 21.13)
# ===========================================================================


class TestNamedFailurePropagation:
    """
    Tests that named failure exceptions (AppNotInstalledError, ElementNotFoundError,
    WebsiteUnreachableError) stop transitive dependents while independent steps
    continue, and that reason strings appear in the report.
    """

    @pytest.mark.asyncio
    async def test_app_not_installed_stops_dependents_not_independent(self):
        """
        AppNotInstalledError on step A:
        - Transitive dependents of A are SKIPPED.
        - Independent steps (no path from A) still RUN.
        (Req 21.9)
        """
        executed: list[str] = []

        async def actuator(step: Step) -> None:
            if step.id == "a":
                raise AppNotInstalledError("WhatsApp")
            executed.append(step.id)

        # a → b → c (chain); d is independent
        plan = _plan(
            _step("a"),
            _step("b", depends_on=["a"]),
            _step("c", depends_on=["b"]),
            _step("d"),  # independent
        )
        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=actuator)
        events = await collect_events(engine, plan)

        completion = _get_completion(events)
        assert "b" in completion.not_performed_step_ids, "b depends on a → must be skipped"
        assert "c" in completion.not_performed_step_ids, "c transitively depends on a → must be skipped"
        assert "d" in completion.executed_step_ids, "d is independent → must still run"
        assert "d" in executed

    @pytest.mark.asyncio
    async def test_app_not_installed_reason_in_report(self):
        """
        AppNotInstalledError reason string appears in failed_steps. (Req 21.9)
        """
        async def actuator(step: Step) -> None:
            raise AppNotInstalledError("Safari")

        plan = _plan(_step("s1"))
        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=actuator)
        events = await collect_events(engine, plan)

        completion = _get_completion(events)
        failed_ids = [sid for sid, _ in completion.failed_steps]
        assert "s1" in failed_ids

        failed_reason = dict(completion.failed_steps)["s1"]
        assert "App not installed" in failed_reason
        assert "Safari" in failed_reason

    @pytest.mark.asyncio
    async def test_element_not_found_stops_dependents(self):
        """
        ElementNotFoundError on step A stops A's transitive dependents. (Req 21.12)
        """
        executed: list[str] = []

        async def actuator(step: Step) -> None:
            if step.id == "a":
                raise ElementNotFoundError("Submit button in checkout form")
            executed.append(step.id)

        plan = _plan(
            _step("a"),
            _step("b", depends_on=["a"]),
            _step("c"),  # independent
        )
        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=actuator)
        events = await collect_events(engine, plan)

        completion = _get_completion(events)
        assert "b" in completion.not_performed_step_ids
        assert "c" in completion.executed_step_ids
        assert "c" in executed

        failed_reason = dict(completion.failed_steps)["a"]
        assert "Element not found" in failed_reason
        assert "Submit button in checkout form" in failed_reason

    @pytest.mark.asyncio
    async def test_website_unreachable_stops_dependents(self):
        """
        WebsiteUnreachableError on step A stops A's transitive dependents. (Req 21.13)
        """
        executed: list[str] = []

        async def actuator(step: Step) -> None:
            if step.id == "a":
                raise WebsiteUnreachableError("https://example.com")
            executed.append(step.id)

        plan = _plan(
            _step("a"),
            _step("b", depends_on=["a"]),
            _step("c"),  # independent
        )
        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=actuator)
        events = await collect_events(engine, plan)

        completion = _get_completion(events)
        assert "b" in completion.not_performed_step_ids
        assert "c" in completion.executed_step_ids

        failed_reason = dict(completion.failed_steps)["a"]
        assert "Website unreachable" in failed_reason
        assert "https://example.com" in failed_reason

    @pytest.mark.asyncio
    async def test_failure_reason_in_report_generic(self):
        """
        Generic exception reason string appears in failed_steps. (Req 17.7)
        """
        async def actuator(step: Step) -> None:
            raise RuntimeError("disk is full")

        plan = _plan(_step("s1"))
        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=actuator)
        events = await collect_events(engine, plan)

        completion = _get_completion(events)
        assert len(completion.failed_steps) == 1
        step_id, reason = completion.failed_steps[0]
        assert step_id == "s1"
        assert "disk is full" in reason

    @pytest.mark.asyncio
    async def test_independent_steps_continue_on_failure(self):
        """
        Chain A→B, plus C independent.
        A fails → B skipped, C runs. (Req 17.7, 21.14)
        """
        executed: list[str] = []

        async def actuator(step: Step) -> None:
            if step.id == "a":
                raise RuntimeError("A failed")
            executed.append(step.id)

        plan = _plan(
            _step("a"),
            _step("b", depends_on=["a"]),
            _step("c"),
        )
        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=actuator)
        events = await collect_events(engine, plan)

        completion = _get_completion(events)
        assert "a" in [sid for sid, _ in completion.failed_steps]
        assert "b" in completion.not_performed_step_ids
        assert "c" in completion.executed_step_ids
        assert "c" in executed
        assert "b" not in executed

    @pytest.mark.asyncio
    async def test_transitive_dependent_chain_all_skipped(self):
        """
        When the root of a long chain fails, all downstream steps are skipped.
        """
        executed: list[str] = []

        async def actuator(step: Step) -> None:
            if step.id == "s1":
                raise AppNotInstalledError("TestApp")
            executed.append(step.id)

        plan = _plan(
            _step("s1"),
            _step("s2", depends_on=["s1"]),
            _step("s3", depends_on=["s2"]),
            _step("s4", depends_on=["s3"]),
        )
        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=actuator)
        events = await collect_events(engine, plan)

        completion = _get_completion(events)
        not_performed = set(completion.not_performed_step_ids)
        assert {"s2", "s3", "s4"} == not_performed
        assert executed == []

    @pytest.mark.asyncio
    async def test_multiple_independent_steps_all_run_despite_one_failure(self):
        """
        With 5 independent steps where 1 fails, the other 4 all complete.
        """
        executed: list[str] = []

        async def actuator(step: Step) -> None:
            if step.id == "fail":
                raise RuntimeError("intentional failure")
            executed.append(step.id)

        plan = _plan(
            _step("fail"),
            _step("ok1"),
            _step("ok2"),
            _step("ok3"),
            _step("ok4"),
        )
        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=actuator)
        events = await collect_events(engine, plan)

        completion = _get_completion(events)
        assert set(completion.executed_step_ids) == {"ok1", "ok2", "ok3", "ok4"}
        assert [sid for sid, _ in completion.failed_steps] == ["fail"]


# ===========================================================================
# Rejection propagation (Req 22.5, 22.6)
# ===========================================================================


class TestRejectionPropagation:
    """
    Verifies that rejection (from Safety_Gate) also stops transitive
    dependents while independent steps continue — i.e., the same
    propagation logic used for failures applies to rejections.
    (Task 23.1 confirmed this; we verify here too.)
    """

    @pytest.mark.asyncio
    async def test_rejection_stops_transitive_dependents(self):
        """
        Rejecting a CONSEQUENTIAL step skips its transitive dependents (Req 22.6).
        """
        gate = SafetyGate(confirmation_callback=_always_reject)
        engine = ExecutionEngine(safety_gate=gate)

        plan = _plan(
            _step("root", classification=StepClassification.CONSEQUENTIAL),
            _step("dep1", depends_on=["root"]),
            _step("dep2", depends_on=["dep1"]),
        )
        events = await collect_events(engine, plan)
        completion = _get_completion(events)

        assert "root" in completion.not_performed_step_ids
        assert "dep1" in completion.not_performed_step_ids
        assert "dep2" in completion.not_performed_step_ids

    @pytest.mark.asyncio
    async def test_rejection_does_not_stop_independent_steps(self):
        """
        Rejecting one step does not prevent unrelated (independent) steps
        from running (Req 22.6).
        """
        executed: list[str] = []

        async def actuator(step: Step) -> None:
            executed.append(step.id)

        gate = SafetyGate(confirmation_callback=_always_reject)
        engine = ExecutionEngine(safety_gate=gate, actuator_callback=actuator)

        plan = _plan(
            _step("bad", classification=StepClassification.CONSEQUENTIAL),
            _step("indep", classification=StepClassification.REVERSIBLE),
        )
        events = await collect_events(engine, plan)
        completion = _get_completion(events)

        assert "bad" in completion.not_performed_step_ids
        assert "indep" in completion.executed_step_ids
        assert "indep" in executed


# ===========================================================================
# Property-based tests
# ===========================================================================


# Feature: haki-personal-ai-assistant, Property 53: Cancellation stops unstarted steps and partitions the report
# Validates: Requirements 17.6
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(n_steps=st.integers(min_value=1, max_value=8))
def test_property_cancel_before_execute_no_steps_run(n_steps: int) -> None:
    """
    **Validates: Requirements 17.6**

    Property 53: For any plan with N steps, calling cancel() before execute()
    ensures ALL steps land in not_performed_step_ids and NONE in executed_step_ids.
    report.cancelled must be True.
    """
    executed: list[str] = []

    async def actuator(step: Step) -> None:
        executed.append(step.id)

    async def run() -> list[StepEvent]:
        eng = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=actuator)
        eng.cancel()
        steps = [
            Step(
                id=f"s{i}",
                intent=f"step {i}",
                actuator=Actuator.INTERNAL,
                classification=StepClassification.REVERSIBLE,
            )
            for i in range(n_steps)
        ]
        plan = CommandPlan(origin_command="cancel test", steps=steps)
        return await collect_events(eng, plan)

    events = asyncio.get_event_loop().run_until_complete(run())

    assert executed == [], (
        "No steps must execute when cancel() is called before execute()"
    )

    plan_complete = next(e for e in events if e.event_type == StepEventType.PLAN_COMPLETE)
    report: ExecutionReport = plan_complete.data
    assert report.cancelled is True, "report.cancelled must be True (Req 17.6)"

    completion: PlanCompletionEvent = report.completion_event
    assert completion.executed_step_ids == [], (
        "executed_step_ids must be empty after pre-execution cancel"
    )
    assert len(completion.not_performed_step_ids) == n_steps, (
        f"All {n_steps} steps must appear in not_performed_step_ids"
    )


# Feature: haki-personal-ai-assistant, Property 54: Failure and rejection propagation
# Validates: Requirements 17.7, 21.9, 21.12, 21.13, 21.14, 22.6
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_independent=st.integers(min_value=1, max_value=5),
    n_dependent=st.integers(min_value=1, max_value=4),
    failure_type=st.sampled_from(["generic", "app_not_installed", "element_not_found", "website_unreachable"]),
)
def test_property_failure_stops_only_dependents(
    n_independent: int, n_dependent: int, failure_type: str
) -> None:
    """
    **Validates: Requirements 17.7, 21.9, 21.12, 21.13, 21.14, 22.6**

    Property 54: For any plan with a failing root step, N dependent steps
    (forming a chain), and M independent steps:
    - All N dependent steps are in not_performed_step_ids.
    - All M independent steps are in executed_step_ids.
    - The root step appears in failed_steps with a non-empty reason.
    """
    dependent_ids = [f"dep_{i}" for i in range(n_dependent)]
    independent_ids = [f"ind_{i}" for i in range(n_independent)]

    def _make_failure() -> Exception:
        if failure_type == "app_not_installed":
            return AppNotInstalledError("TestApp")
        elif failure_type == "element_not_found":
            return ElementNotFoundError("test element")
        elif failure_type == "website_unreachable":
            return WebsiteUnreachableError("https://test.example.com")
        else:
            return RuntimeError("generic failure")

    async def actuator(step: Step) -> None:
        if step.id == "root":
            raise _make_failure()
        # all other steps succeed (no-op)

    # Build plan: root → dep_0 → dep_1 → ... chain, plus independent steps
    steps: list[Step] = [
        Step(
            id="root",
            intent="root step",
            actuator=Actuator.INTERNAL,
            classification=StepClassification.REVERSIBLE,
        )
    ]
    for i, dep_id in enumerate(dependent_ids):
        prev = "root" if i == 0 else dependent_ids[i - 1]
        steps.append(Step(
            id=dep_id,
            intent=f"dependent step {i}",
            actuator=Actuator.INTERNAL,
            classification=StepClassification.REVERSIBLE,
            depends_on=[prev],
        ))
    for ind_id in independent_ids:
        steps.append(Step(
            id=ind_id,
            intent=f"independent step",
            actuator=Actuator.INTERNAL,
            classification=StepClassification.REVERSIBLE,
        ))

    async def run() -> list[StepEvent]:
        eng = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=actuator)
        plan = CommandPlan(origin_command="failure propagation test", steps=steps)
        return await collect_events(eng, plan)

    events = asyncio.get_event_loop().run_until_complete(run())

    plan_complete = next(e for e in events if e.event_type == StepEventType.PLAN_COMPLETE)
    completion: PlanCompletionEvent = plan_complete.data.completion_event

    not_performed = set(completion.not_performed_step_ids)
    executed = set(completion.executed_step_ids)
    failed_dict = dict(completion.failed_steps)

    # root must be in failed_steps with a non-empty reason
    assert "root" in failed_dict, (
        "Failed root step must appear in failed_steps (Req 17.7)"
    )
    assert failed_dict["root"], "Failure reason must be non-empty"

    # All dependent steps must be in not_performed
    for dep_id in dependent_ids:
        assert dep_id in not_performed, (
            f"Dependent step '{dep_id}' must be in not_performed_step_ids "
            f"when root fails (Req 17.7, 21.14)"
        )

    # All independent steps must have been executed
    for ind_id in independent_ids:
        assert ind_id in executed, (
            f"Independent step '{ind_id}' must still be executed despite root failure "
            f"(Req 17.7, 21.14)"
        )
