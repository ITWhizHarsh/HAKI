"""
Unit tests for the Safety_Gate and ExecutionEngine (Tasks 22.1, 22.2).

Covers:
  - Task 22.1: classification, confirmation gating, reversible pass-through
  - Task 22.2: mid-plan pause/confirm/resume, rejection, timeout (no-response)

**Validates: Requirements 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 22.7, 22.8**
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

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
    actuator: Actuator = Actuator.INTERNAL,
) -> Step:
    return Step(
        id=step_id,
        intent=intent,
        actuator=actuator,
        classification=classification,
        depends_on=depends_on or [],
    )


def _plan(*steps: Step, command: str = "test") -> CommandPlan:
    return CommandPlan(origin_command=command, steps=list(steps))


async def _always_confirm(req: ConfirmationRequest) -> ConfirmationResult:
    return ConfirmationResult.CONFIRMED


async def _always_reject(req: ConfirmationRequest) -> ConfirmationResult:
    return ConfirmationResult.REJECTED


async def _always_timeout(req: ConfirmationRequest) -> ConfirmationResult:
    await asyncio.sleep(10)  # will be cut short by gate timeout
    return ConfirmationResult.CONFIRMED  # pragma: no cover


# ===========================================================================
# Task 22.1 — SafetyGate: classification and confirmation gating
# ===========================================================================


class TestSafetyGateClassification:
    """Test SafetyGate.requires_confirmation and is_reversible (Req 22.1, 22.4, 22.7)."""

    def test_consequential_requires_confirmation(self):
        s = _step(classification=StepClassification.CONSEQUENTIAL)
        assert SafetyGate.requires_confirmation(s) is True

    def test_unknown_requires_confirmation(self):
        """UNKNOWN is treated as CONSEQUENTIAL (Req 22.7)."""
        s = _step(classification=StepClassification.UNKNOWN)
        assert SafetyGate.requires_confirmation(s) is True

    def test_reversible_does_not_require_confirmation(self):
        """Reversible steps pass through without confirmation (Req 22.4)."""
        s = _step(classification=StepClassification.REVERSIBLE)
        assert SafetyGate.requires_confirmation(s) is False

    def test_is_reversible_true_for_reversible(self):
        s = _step(classification=StepClassification.REVERSIBLE)
        assert SafetyGate.is_reversible(s) is True

    def test_is_reversible_false_for_consequential(self):
        s = _step(classification=StepClassification.CONSEQUENTIAL)
        assert SafetyGate.is_reversible(s) is False

    def test_is_reversible_false_for_unknown(self):
        s = _step(classification=StepClassification.UNKNOWN)
        assert SafetyGate.is_reversible(s) is False

    def test_description_includes_intent(self):
        s = _step(intent="send email to bob")
        desc = SafetyGate._description_for_step(s)
        assert "send email to bob" in desc

    def test_description_includes_classification(self):
        s = _step(classification=StepClassification.CONSEQUENTIAL)
        desc = SafetyGate._description_for_step(s)
        assert "CONSEQUENTIAL" in desc.upper()

    def test_description_includes_args(self):
        s = Step(
            id="x", intent="open file",
            classification=StepClassification.REVERSIBLE,
            args={"path": "/tmp/test.txt"},
        )
        desc = SafetyGate._description_for_step(s)
        assert "path" in desc


class TestSafetyGateCheckReversible:
    """Reversible steps confirm immediately without calling the callback (Req 22.4)."""

    @pytest.mark.asyncio
    async def test_reversible_returns_confirmed_without_callback(self):
        callback = AsyncMock(return_value=ConfirmationResult.REJECTED)
        gate = SafetyGate(confirmation_callback=callback)
        s = _step(classification=StepClassification.REVERSIBLE)
        result = await gate.check(s)
        assert result == ConfirmationResult.CONFIRMED
        callback.assert_not_called()



class TestSafetyGateCheckConsequential:
    """Consequential/Unknown steps pause and require callback (Req 22.1, 22.7)."""

    @pytest.mark.asyncio
    async def test_consequential_calls_callback(self):
        callback = AsyncMock(return_value=ConfirmationResult.CONFIRMED)
        gate = SafetyGate(confirmation_callback=callback)
        s = _step(classification=StepClassification.CONSEQUENTIAL)
        result = await gate.check(s)
        callback.assert_called_once()
        assert result == ConfirmationResult.CONFIRMED

    @pytest.mark.asyncio
    async def test_unknown_calls_callback(self):
        """UNKNOWN treated as CONSEQUENTIAL (Req 22.7)."""
        callback = AsyncMock(return_value=ConfirmationResult.CONFIRMED)
        gate = SafetyGate(confirmation_callback=callback)
        s = _step(classification=StepClassification.UNKNOWN)
        result = await gate.check(s)
        callback.assert_called_once()
        assert result == ConfirmationResult.CONFIRMED

    @pytest.mark.asyncio
    async def test_rejection_propagated(self):
        gate = SafetyGate(confirmation_callback=_always_reject)
        s = _step(classification=StepClassification.CONSEQUENTIAL)
        result = await gate.check(s)
        assert result == ConfirmationResult.REJECTED

    @pytest.mark.asyncio
    async def test_confirmation_request_includes_description(self):
        received: list[ConfirmationRequest] = []

        async def capture(req: ConfirmationRequest) -> ConfirmationResult:
            received.append(req)
            return ConfirmationResult.CONFIRMED

        gate = SafetyGate(confirmation_callback=capture)
        s = _step(intent="delete all files", classification=StepClassification.CONSEQUENTIAL)
        await gate.check(s)
        assert len(received) == 1
        assert "delete all files" in received[0].description

    @pytest.mark.asyncio
    async def test_confirmation_request_includes_step(self):
        received: list[ConfirmationRequest] = []

        async def capture(req: ConfirmationRequest) -> ConfirmationResult:
            received.append(req)
            return ConfirmationResult.CONFIRMED

        gate = SafetyGate(confirmation_callback=capture)
        s = _step(step_id="my-step", classification=StepClassification.CONSEQUENTIAL)
        await gate.check(s)
        assert received[0].step_id == "my-step"
        assert received[0].step is s


class TestSafetyGateTimeout:
    """No-response is treated as rejection (Req 22.8)."""

    @pytest.mark.asyncio
    async def test_timeout_returns_timeout_result(self):
        gate = SafetyGate(
            confirmation_callback=_always_timeout,
            confirmation_timeout=0.05,  # 50ms timeout
        )
        s = _step(classification=StepClassification.CONSEQUENTIAL)
        result = await gate.check(s)
        assert result == ConfirmationResult.TIMEOUT

    @pytest.mark.asyncio
    async def test_timeout_does_not_execute(self):
        """When timeout occurs, no execution should happen — verified via ExecutionEngine."""
        executed: list[str] = []

        async def actuator(step: Step):
            executed.append(step.id)

        gate = SafetyGate(
            confirmation_callback=_always_timeout,
            confirmation_timeout=0.05,
        )
        engine = ExecutionEngine(safety_gate=gate, actuator_callback=actuator)
        plan = _plan(_step(classification=StepClassification.CONSEQUENTIAL))
        events = await collect_events(engine, plan)

        assert len(executed) == 0, "Step should not have been executed on timeout"



# ===========================================================================
# Task 22.2 — ExecutionEngine: mid-plan pause/confirm/resume, rejection, no-response
# ===========================================================================


class TestExecutionEngineReversibleSteps:
    """Reversible steps execute without asking for confirmation (Req 22.4)."""

    @pytest.mark.asyncio
    async def test_reversible_step_runs_without_confirmation(self):
        callback = AsyncMock(return_value=ConfirmationResult.REJECTED)
        gate = SafetyGate(confirmation_callback=callback)
        executed: list[str] = []

        async def actuator(step: Step):
            executed.append(step.id)

        engine = ExecutionEngine(safety_gate=gate, actuator_callback=actuator)
        plan = _plan(_step(step_id="rev", classification=StepClassification.REVERSIBLE))
        events = await collect_events(engine, plan)

        assert "rev" in executed
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_reversible_step_completed_event(self):
        engine = ExecutionEngine(safety_gate=SafetyGate())
        plan = _plan(_step(classification=StepClassification.REVERSIBLE))
        events = await collect_events(engine, plan)
        event_types = [e.event_type for e in events]
        assert StepEventType.COMPLETED in event_types

    @pytest.mark.asyncio
    async def test_multiple_reversible_steps_all_complete(self):
        executed: list[str] = []

        async def actuator(step: Step):
            executed.append(step.id)

        engine = ExecutionEngine(safety_gate=SafetyGate(), actuator_callback=actuator)
        plan = _plan(
            _step("r1", classification=StepClassification.REVERSIBLE),
            _step("r2", classification=StepClassification.REVERSIBLE),
        )
        events = await collect_events(engine, plan)
        assert set(executed) == {"r1", "r2"}


class TestExecutionEngineConfirmAndResume:
    """Mid-plan pause → confirm → resume (Reqs 22.2, 22.3)."""

    @pytest.mark.asyncio
    async def test_consequential_step_awaiting_confirmation_event_emitted(self):
        gate = SafetyGate(confirmation_callback=_always_confirm)
        engine = ExecutionEngine(safety_gate=gate)
        plan = _plan(_step(classification=StepClassification.CONSEQUENTIAL))
        events = await collect_events(engine, plan)
        event_types = [e.event_type for e in events]
        assert StepEventType.AWAITING_CONFIRMATION in event_types

    @pytest.mark.asyncio
    async def test_confirmed_step_executes(self):
        executed: list[str] = []

        async def actuator(step: Step):
            executed.append(step.id)

        gate = SafetyGate(confirmation_callback=_always_confirm)
        engine = ExecutionEngine(safety_gate=gate, actuator_callback=actuator)
        plan = _plan(_step("con", classification=StepClassification.CONSEQUENTIAL))
        await collect_events(engine, plan)
        assert "con" in executed

    @pytest.mark.asyncio
    async def test_confirmed_event_emitted(self):
        gate = SafetyGate(confirmation_callback=_always_confirm)
        engine = ExecutionEngine(safety_gate=gate)
        plan = _plan(_step(classification=StepClassification.CONSEQUENTIAL))
        events = await collect_events(engine, plan)
        event_types = [e.event_type for e in events]
        assert StepEventType.CONFIRMED in event_types

    @pytest.mark.asyncio
    async def test_mid_plan_pause_then_resume(self):
        """First step is reversible, second is consequential: should pause at second."""
        paused_at: list[str] = []
        executed: list[str] = []

        async def track_confirmation(req: ConfirmationRequest) -> ConfirmationResult:
            paused_at.append(req.step_id)
            return ConfirmationResult.CONFIRMED

        async def actuator(step: Step):
            executed.append(step.id)

        gate = SafetyGate(confirmation_callback=track_confirmation)
        engine = ExecutionEngine(safety_gate=gate, actuator_callback=actuator)
        plan = _plan(
            _step("r1", classification=StepClassification.REVERSIBLE),
            _step("c1", classification=StepClassification.CONSEQUENTIAL, depends_on=["r1"]),
        )
        await collect_events(engine, plan)

        assert "r1" in executed, "Reversible step should have run"
        assert "c1" in executed, "Consequential step should have run after confirmation"
        assert "c1" in paused_at, "Confirmation should have been requested for c1"



class TestExecutionEngineRejection:
    """On rejection: step not executed, transitive deps skipped, report produced (Reqs 22.3, 22.5, 22.6)."""

    @pytest.mark.asyncio
    async def test_rejected_step_not_executed(self):
        executed: list[str] = []

        async def actuator(step: Step):
            executed.append(step.id)

        gate = SafetyGate(confirmation_callback=_always_reject)
        engine = ExecutionEngine(safety_gate=gate, actuator_callback=actuator)
        plan = _plan(_step("bad", classification=StepClassification.CONSEQUENTIAL))
        await collect_events(engine, plan)
        assert "bad" not in executed

    @pytest.mark.asyncio
    async def test_rejected_step_skipped_event_emitted(self):
        gate = SafetyGate(confirmation_callback=_always_reject)
        engine = ExecutionEngine(safety_gate=gate)
        plan = _plan(_step("x", classification=StepClassification.CONSEQUENTIAL))
        events = await collect_events(engine, plan)
        event_types = [e.event_type for e in events]
        assert StepEventType.REJECTED in event_types

    @pytest.mark.asyncio
    async def test_dependent_skipped_after_rejection(self):
        """Transitive dependents of rejected step must be skipped (Req 22.5)."""
        executed: list[str] = []

        async def actuator(step: Step):
            executed.append(step.id)

        gate = SafetyGate(confirmation_callback=_always_reject)
        engine = ExecutionEngine(safety_gate=gate, actuator_callback=actuator)
        plan = _plan(
            _step("c1", classification=StepClassification.CONSEQUENTIAL),
            _step("dep1", classification=StepClassification.REVERSIBLE, depends_on=["c1"]),
            _step("dep2", classification=StepClassification.REVERSIBLE, depends_on=["dep1"]),
        )
        events = await collect_events(engine, plan)

        skipped_ids = {e.step_id for e in events if e.event_type == StepEventType.SKIPPED}
        assert "dep1" in skipped_ids
        assert "dep2" in skipped_ids
        assert "c1" not in executed
        assert "dep1" not in executed
        assert "dep2" not in executed

    @pytest.mark.asyncio
    async def test_independent_step_runs_despite_rejection(self):
        """Steps independent of rejected step must still run (Req 22.6)."""
        executed: list[str] = []

        async def actuator(step: Step):
            executed.append(step.id)

        async def selective_reject(req: ConfirmationRequest) -> ConfirmationResult:
            if req.step_id == "bad":
                return ConfirmationResult.REJECTED
            return ConfirmationResult.CONFIRMED

        gate = SafetyGate(confirmation_callback=selective_reject)
        engine = ExecutionEngine(safety_gate=gate, actuator_callback=actuator)
        plan = _plan(
            _step("bad", classification=StepClassification.CONSEQUENTIAL),
            _step("ind", classification=StepClassification.REVERSIBLE),  # independent
        )
        await collect_events(engine, plan)
        assert "ind" in executed, "Independent step should still run"

    @pytest.mark.asyncio
    async def test_report_contains_rejected_step_id(self):
        gate = SafetyGate(confirmation_callback=_always_reject)
        engine = ExecutionEngine(safety_gate=gate)
        plan = _plan(_step("rej", classification=StepClassification.CONSEQUENTIAL))
        events = await collect_events(engine, plan)
        plan_complete = next(e for e in events if e.event_type == StepEventType.PLAN_COMPLETE)
        report: ExecutionReport = plan_complete.data
        assert report.rejected_step_id == "rej"

    @pytest.mark.asyncio
    async def test_report_not_performed_includes_rejected_and_skipped(self):
        gate = SafetyGate(confirmation_callback=_always_reject)
        engine = ExecutionEngine(safety_gate=gate)
        plan = _plan(
            _step("c1", classification=StepClassification.CONSEQUENTIAL),
            _step("d1", classification=StepClassification.REVERSIBLE, depends_on=["c1"]),
        )
        events = await collect_events(engine, plan)
        report: ExecutionReport = next(
            e for e in events if e.event_type == StepEventType.PLAN_COMPLETE
        ).data
        not_performed_ids = {s.id for s in report.not_performed}
        assert "c1" in not_performed_ids
        assert "d1" in not_performed_ids



class TestExecutionEngineNoResponse:
    """No-response (timeout) treated as rejection (Req 22.8)."""

    @pytest.mark.asyncio
    async def test_timeout_step_not_executed(self):
        executed: list[str] = []

        async def actuator(step: Step):
            executed.append(step.id)

        gate = SafetyGate(
            confirmation_callback=_always_timeout,
            confirmation_timeout=0.05,
        )
        engine = ExecutionEngine(safety_gate=gate, actuator_callback=actuator)
        plan = _plan(_step("to", classification=StepClassification.CONSEQUENTIAL))
        await collect_events(engine, plan)
        assert "to" not in executed

    @pytest.mark.asyncio
    async def test_timeout_report_records_timeout_step(self):
        gate = SafetyGate(
            confirmation_callback=_always_timeout,
            confirmation_timeout=0.05,
        )
        engine = ExecutionEngine(safety_gate=gate)
        plan = _plan(_step("to", classification=StepClassification.CONSEQUENTIAL))
        events = await collect_events(engine, plan)
        report: ExecutionReport = next(
            e for e in events if e.event_type == StepEventType.PLAN_COMPLETE
        ).data
        assert report.timeout_step_id == "to"

    @pytest.mark.asyncio
    async def test_timeout_dependents_are_skipped(self):
        """Dependents of timed-out step must be skipped (Req 22.8 → treat as rejection)."""
        gate = SafetyGate(
            confirmation_callback=_always_timeout,
            confirmation_timeout=0.05,
        )
        engine = ExecutionEngine(safety_gate=gate)
        plan = _plan(
            _step("to", classification=StepClassification.CONSEQUENTIAL),
            _step("dep", classification=StepClassification.REVERSIBLE, depends_on=["to"]),
        )
        events = await collect_events(engine, plan)
        skipped_ids = {e.step_id for e in events if e.event_type == StepEventType.SKIPPED}
        assert "dep" in skipped_ids


class TestExecutionEngineReport:
    """ExecutionReport accurately reflects completed vs. not-performed (Req 22.6)."""

    @pytest.mark.asyncio
    async def test_completed_steps_in_report(self):
        engine = ExecutionEngine(safety_gate=SafetyGate())
        plan = _plan(
            _step("r1", classification=StepClassification.REVERSIBLE),
            _step("r2", classification=StepClassification.REVERSIBLE),
        )
        events = await collect_events(engine, plan)
        report: ExecutionReport = next(
            e for e in events if e.event_type == StepEventType.PLAN_COMPLETE
        ).data
        completed_ids = {s.id for s in report.completed}
        assert "r1" in completed_ids
        assert "r2" in completed_ids

    @pytest.mark.asyncio
    async def test_all_completed_flag_when_all_succeed(self):
        engine = ExecutionEngine(safety_gate=SafetyGate())
        plan = _plan(_step(classification=StepClassification.REVERSIBLE))
        events = await collect_events(engine, plan)
        report: ExecutionReport = next(
            e for e in events if e.event_type == StepEventType.PLAN_COMPLETE
        ).data
        assert report.all_completed is True

    @pytest.mark.asyncio
    async def test_all_completed_false_when_rejected(self):
        gate = SafetyGate(confirmation_callback=_always_reject)
        engine = ExecutionEngine(safety_gate=gate)
        plan = _plan(_step(classification=StepClassification.CONSEQUENTIAL))
        events = await collect_events(engine, plan)
        report: ExecutionReport = next(
            e for e in events if e.event_type == StepEventType.PLAN_COMPLETE
        ).data
        assert report.all_completed is False

    @pytest.mark.asyncio
    async def test_failed_step_in_report(self):
        async def failing_actuator(step: Step):
            raise RuntimeError("actuator error")

        engine = ExecutionEngine(
            safety_gate=SafetyGate(),
            actuator_callback=failing_actuator,
        )
        plan = _plan(_step("fail", classification=StepClassification.REVERSIBLE))
        events = await collect_events(engine, plan)
        report: ExecutionReport = next(
            e for e in events if e.event_type == StepEventType.PLAN_COMPLETE
        ).data
        assert any(s.id == "fail" for s in report.failed)

    @pytest.mark.asyncio
    async def test_dependents_skipped_on_failure(self):
        async def failing_actuator(step: Step):
            raise RuntimeError("error")

        engine = ExecutionEngine(
            safety_gate=SafetyGate(),
            actuator_callback=failing_actuator,
        )
        plan = _plan(
            _step("fail", classification=StepClassification.REVERSIBLE),
            _step("dep", classification=StepClassification.REVERSIBLE, depends_on=["fail"]),
        )
        events = await collect_events(engine, plan)
        skipped_ids = {e.step_id for e in events if e.event_type == StepEventType.SKIPPED}
        assert "dep" in skipped_ids



class TestExecutionEngineSyncCallback:
    """Synchronous confirmation callbacks should work too."""

    @pytest.mark.asyncio
    async def test_sync_confirm_callback(self):
        def sync_confirm(req: ConfirmationRequest) -> ConfirmationResult:
            return ConfirmationResult.CONFIRMED

        executed: list[str] = []

        async def actuator(step: Step):
            executed.append(step.id)

        gate = SafetyGate(sync_confirmation_callback=sync_confirm)
        engine = ExecutionEngine(safety_gate=gate, actuator_callback=actuator)
        plan = _plan(_step("c", classification=StepClassification.CONSEQUENTIAL))
        await collect_events(engine, plan)
        assert "c" in executed

    @pytest.mark.asyncio
    async def test_sync_reject_callback(self):
        def sync_reject(req: ConfirmationRequest) -> ConfirmationResult:
            return ConfirmationResult.REJECTED

        executed: list[str] = []

        async def actuator(step: Step):
            executed.append(step.id)

        gate = SafetyGate(sync_confirmation_callback=sync_reject)
        engine = ExecutionEngine(safety_gate=gate, actuator_callback=actuator)
        plan = _plan(_step("c", classification=StepClassification.CONSEQUENTIAL))
        await collect_events(engine, plan)
        assert "c" not in executed


class TestExecutionEngineCancel:
    """Cancellation stops pending steps."""

    @pytest.mark.asyncio
    async def test_cancel_before_execute_skips_all(self):
        engine = ExecutionEngine(safety_gate=SafetyGate())
        engine.cancel()
        plan = _plan(
            _step("s1", classification=StepClassification.REVERSIBLE),
            _step("s2", classification=StepClassification.REVERSIBLE),
        )
        events = await collect_events(engine, plan)
        report: ExecutionReport = next(
            e for e in events if e.event_type == StepEventType.PLAN_COMPLETE
        ).data
        assert report.cancelled is True


class TestSafetyGateMiscellaneous:
    """Edge cases and error handling."""

    def test_both_callbacks_raises(self):
        with pytest.raises(ValueError):
            SafetyGate(
                confirmation_callback=_always_confirm,
                sync_confirmation_callback=lambda r: ConfirmationResult.CONFIRMED,
            )

    @pytest.mark.asyncio
    async def test_no_callback_defaults_to_confirmed(self):
        """With no callback configured, consequential steps default to CONFIRMED."""
        gate = SafetyGate()
        s = _step(classification=StepClassification.CONSEQUENTIAL)
        result = await gate.check(s)
        assert result == ConfirmationResult.CONFIRMED

    @pytest.mark.asyncio
    async def test_callback_exception_treated_as_rejection(self):
        async def bad_callback(req: ConfirmationRequest) -> ConfirmationResult:
            raise RuntimeError("callback exploded")

        gate = SafetyGate(confirmation_callback=bad_callback)
        s = _step(classification=StepClassification.CONSEQUENTIAL)
        result = await gate.check(s)
        assert result == ConfirmationResult.REJECTED

    @pytest.mark.asyncio
    async def test_empty_plan_completes_immediately(self):
        engine = ExecutionEngine(safety_gate=SafetyGate())
        plan = CommandPlan(origin_command="empty")
        events = await collect_events(engine, plan)
        assert any(e.event_type == StepEventType.PLAN_COMPLETE for e in events)

