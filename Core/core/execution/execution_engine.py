"""
ExecutionEngine — dependency-aware plan execution with mid-plan
pause/confirm/resume and rejection/no-response handling.

The engine drives a :class:`~core.planner.CommandPlan` through the
following loop (per the design's execution flowchart):

    for each ready step (deps met):
        1. Resolve required slots (Dialogue_Manager integration point)
        2. Check Safety_Gate:
           - REVERSIBLE → execute immediately
           - CONSEQUENTIAL/UNKNOWN → pause, request confirmation
             - confirmed → execute
             - rejected → skip step + transitive dependents; report (22.3, 22.6)
             - timeout → treat as rejection (22.8)
        3. Execute via actuator callback
        4. Postcondition check (optional)
        5. Mark COMPLETED / FAILED; stop transitive dependents on failure

At the end an :class:`ExecutionReport` is returned describing completed
steps vs. not-performed steps.

Design: Safety_Gate, Execution loop.
Requirements: 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 22.7, 22.8.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable

from core.planner import (
    CommandPlan,
    Step,
    StepClassification,
    StepStatus,
)
from .safety_gate import (
    ConfirmationResult,
    SafetyGate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StepEventType & StepEvent (streaming result type)
# ---------------------------------------------------------------------------


class StepEventType(str, Enum):
    """
    Event types emitted by :class:`ExecutionEngine` during plan execution.

    These match the design's ``StepEvent`` interface:
    ``{started, completed, failed, awaitingConfirmation, skipped}``.
    """

    STARTED = "started"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    PLAN_COMPLETE = "plan_complete"


@dataclass
class StepEvent:
    """
    A single event emitted by the :class:`ExecutionEngine` as a step
    progresses through its lifecycle.

    Attributes
    ----------
    event_type:
        The type of event (see :class:`StepEventType`).
    step_id:
        The step this event relates to.  ``None`` for plan-level events
        (``PLAN_COMPLETE``).
    step:
        The full :class:`~core.planner.Step` object (may be ``None`` for
        plan-level events).
    message:
        Optional human-readable detail, e.g. the reason for failure or
        the confirmation description.
    data:
        Optional free-form payload (e.g. actuator output).
    """

    event_type: StepEventType
    step_id: str | None = None
    step: Step | None = None
    message: str | None = None
    data: Any = None


# ---------------------------------------------------------------------------
# ExecutionReport
# ---------------------------------------------------------------------------


@dataclass
class ExecutionReport:
    """
    Final summary of a plan execution.

    Attributes
    ----------
    completed:
        Steps that reached :attr:`~core.planner.StepStatus.COMPLETED`.
    not_performed:
        Steps that were SKIPPED (due to rejection or failed dependency)
        or never reached because of early termination.
    failed:
        Steps that reached :attr:`~core.planner.StepStatus.FAILED`.
    rejected_step_id:
        The ID of the step that was rejected by the user, if any.  This
        is the root cause of downstream skips (Req 22.6).
    timeout_step_id:
        The ID of the step whose confirmation timed out, if any (Req 22.8).
    cancelled:
        Whether the plan was externally cancelled via :meth:`ExecutionEngine.cancel`.
    """

    completed: list[Step] = field(default_factory=list)
    not_performed: list[Step] = field(default_factory=list)
    failed: list[Step] = field(default_factory=list)
    rejected_step_id: str | None = None
    timeout_step_id: str | None = None
    cancelled: bool = False

    @property
    def all_completed(self) -> bool:
        """``True`` when every step completed successfully."""
        return not self.not_performed and not self.failed and not self.cancelled

    def summary(self) -> str:
        """Return a brief human-readable summary."""
        parts = [f"Completed: {[s.id for s in self.completed]}"]
        if self.not_performed:
            parts.append(f"Not performed: {[s.id for s in self.not_performed]}")
        if self.failed:
            parts.append(f"Failed: {[s.id for s in self.failed]}")
        if self.rejected_step_id:
            parts.append(f"Rejected at: {self.rejected_step_id}")
        if self.timeout_step_id:
            parts.append(f"Timed out at: {self.timeout_step_id}")
        if self.cancelled:
            parts.append("(cancelled)")
        return "; ".join(parts)


# ---------------------------------------------------------------------------
# ActuatorCallback type alias
# ---------------------------------------------------------------------------

#: Async callback invoked to execute a step.
#: Receives the :class:`~core.planner.Step` and returns arbitrary output.
#: Should raise on failure.
ActuatorCallbackAsync = Callable[[Step], Awaitable[Any]]

#: Sync callback alternative.
ActuatorCallbackSync = Callable[[Step], Any]


# ---------------------------------------------------------------------------
# ExecutionEngine
# ---------------------------------------------------------------------------


class ExecutionEngine:
    """
    Dependency-aware, safety-gated plan executor.

    Executes a :class:`~core.planner.CommandPlan` step by step:

    - Runs independent steps in parallel (respects ``depends_on``).
    - Pauses at CONSEQUENTIAL/UNKNOWN steps to request confirmation
      (Safety_Gate integration, Req 22.1, 22.2).
    - On confirmation, resumes execution of the confirmed step and
      continues the plan (Req 22.3).
    - On rejection or timeout, skips the step and all transitive
      dependents, reports completed vs. not-performed (Reqs 22.5, 22.6,
      22.8).
    - Streams :class:`StepEvent` objects so callers can observe progress
      in real time.

    Parameters
    ----------
    safety_gate:
        A pre-configured :class:`~core.execution.SafetyGate`.  When
        ``None``, a default gate with no confirmation callback is created
        (all steps pass through — useful only for testing).
    actuator_callback:
        An **async** callable ``(Step) → Any`` that executes the step's
        actuator.  When ``None``, steps are "executed" as no-ops (useful
        for unit tests that test only the gating logic).
    sync_actuator_callback:
        A **sync** callable alternative to ``actuator_callback``.

    Design: Execution loop, ExecutionEngine interface.
    Requirements: 22.1–22.8.
    """

    def __init__(
        self,
        safety_gate: SafetyGate | None = None,
        actuator_callback: ActuatorCallbackAsync | None = None,
        sync_actuator_callback: ActuatorCallbackSync | None = None,
    ) -> None:
        self._gate = safety_gate or SafetyGate()
        self._async_actuator = actuator_callback
        self._sync_actuator = sync_actuator_callback
        self._cancelled = False
        self._cancel_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(self, plan: CommandPlan) -> AsyncIterator[StepEvent]:
        """
        Execute *plan* and yield :class:`StepEvent` objects as execution
        progresses.

        Usage::

            async for event in engine.execute(plan):
                print(event)

        The final event is always a ``PLAN_COMPLETE`` event carrying the
        :class:`ExecutionReport` in ``event.data``.

        Parameters
        ----------
        plan:
            The plan to execute.  Step statuses are mutated in-place so
            the caller can inspect the final plan state after iteration.

        Yields
        ------
        StepEvent
            Events for each step lifecycle transition.
        """
        return self._execute_stream(plan)

    def cancel(self) -> None:
        """
        Signal the engine to stop processing further steps.

        Already-running steps may complete (they are not interrupted),
        but no new steps will be started after this call.

        Calling ``cancel()`` before ``execute()`` will prevent any steps
        from running when execution starts.
        """
        self._cancelled = True
        logger.info("ExecutionEngine: cancellation requested.")

    # ------------------------------------------------------------------
    # Internal streaming implementation
    # ------------------------------------------------------------------

    async def _execute_stream(self, plan: CommandPlan) -> AsyncIterator[StepEvent]:
        """
        Core execution loop (async generator).

        Implements the design's plan→gate→execute→verify loop with
        dependency-aware parallelism and mid-plan pause/resume.
        """
        # Note: do NOT reset _cancelled here — if cancel() was called before
        # execute(), the cancellation should be honoured immediately.
        report = ExecutionReport()

        # We use a task-based approach: one asyncio task per ready step,
        # running concurrently when there are no shared dependencies.
        pending_tasks: set[asyncio.Task] = set()
        event_queue: asyncio.Queue[StepEvent | None] = asyncio.Queue()

        async def run_step(step: Step) -> None:
            """Coroutine that gates + executes a single step."""
            await self._run_one_step(step, plan, report, event_queue)

        try:
            while True:
                if self._cancelled:
                    # Cancel any remaining pending steps
                    for s in plan.steps:
                        if s.status == StepStatus.PENDING:
                            s.status = StepStatus.SKIPPED
                            report.not_performed.append(s)
                    report.cancelled = True
                    break

                # Collect steps that are ready and not already running
                running_ids = {
                    t.get_name() for t in pending_tasks if not t.done()
                }
                ready = [
                    s for s in plan.ready_steps()
                    if s.id not in running_ids
                ]

                if ready:
                    for step in ready:
                        # Mark as running so ready_steps() excludes it next
                        # iteration.  We use a separate RUNNING status for
                        # this purpose.
                        step.status = StepStatus.RUNNING
                        task = asyncio.create_task(run_step(step), name=step.id)
                        pending_tasks.add(task)

                # Drain available events from the queue (non-blocking)
                events_drained = 0
                while not event_queue.empty():
                    evt = event_queue.get_nowait()
                    if evt is not None:
                        yield evt
                        events_drained += 1

                # Clean up done tasks
                done = {t for t in pending_tasks if t.done()}
                for t in done:
                    pending_tasks.discard(t)
                    # Propagate task exceptions (should not occur; errors
                    # are handled inside run_step, but guard anyway).
                    try:
                        t.result()
                    except Exception as exc:
                        logger.error(
                            "ExecutionEngine: unexpected task exception: %s", exc
                        )

                # Check if plan is complete
                if plan.is_complete() and not pending_tasks:
                    break

                # If no tasks are running and no steps are ready but the
                # plan is not complete, something is stuck (e.g. all
                # remaining steps blocked by failed deps).  Break to avoid
                # infinite loop.
                if not pending_tasks and not plan.ready_steps():
                    # Mark any remaining PENDING steps as SKIPPED
                    for s in plan.steps:
                        if s.status == StepStatus.PENDING:
                            s.status = StepStatus.SKIPPED
                            report.not_performed.append(s)
                    break

                # Yield control to let tasks make progress
                await asyncio.sleep(0)

            # Drain remaining events after loop ends
            while not event_queue.empty():
                evt = event_queue.get_nowait()
                if evt is not None:
                    yield evt

        finally:
            # Cancel any remaining tasks on error / cancellation
            for t in pending_tasks:
                t.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)

        # Build final report from plan state (in case any steps were not
        # tracked due to early termination)
        report.completed = plan.completed_steps()
        report.not_performed = list(
            {s.id: s for s in report.not_performed + plan.skipped_steps()}.values()
        )
        report.failed = plan.failed_steps()

        yield StepEvent(
            event_type=StepEventType.PLAN_COMPLETE,
            message=report.summary(),
            data=report,
        )

    # ------------------------------------------------------------------
    # Single-step gating + execution
    # ------------------------------------------------------------------

    async def _run_one_step(
        self,
        step: Step,
        plan: CommandPlan,
        report: ExecutionReport,
        event_queue: asyncio.Queue[StepEvent | None],
    ) -> None:
        """
        Gate and execute a single step, emitting events to *event_queue*.

        Handles:
        - Safety_Gate confirmation (Reqs 22.1, 22.2, 22.5, 22.6, 22.7, 22.8)
        - Actuator execution
        - Dependency propagation on failure / rejection
        """
        # ----------------------------------------------------------------
        # 1. Safety gate
        # ----------------------------------------------------------------
        if self._gate.requires_confirmation(step):
            step.status = StepStatus.AWAITING_CONFIRM
            await event_queue.put(StepEvent(
                event_type=StepEventType.AWAITING_CONFIRMATION,
                step_id=step.id,
                step=step,
                message=(
                    f"Awaiting confirmation for: {step.intent} "
                    f"[{step.classification.value.upper()}]"
                ),
            ))

            # Mid-plan pause: await user confirmation (Req 22.2)
            confirmation = await self._gate.check(step)

            if confirmation == ConfirmationResult.CONFIRMED:
                # Confirmed: resume execution (Req 22.3)
                await event_queue.put(StepEvent(
                    event_type=StepEventType.CONFIRMED,
                    step_id=step.id,
                    step=step,
                    message=f"Step '{step.id}' confirmed.",
                ))
                step.status = StepStatus.RUNNING

            else:
                # Rejected or timed-out (Reqs 22.5, 22.6, 22.8)
                step.status = StepStatus.SKIPPED
                event_type = (
                    StepEventType.REJECTED
                    if confirmation == ConfirmationResult.REJECTED
                    else StepEventType.SKIPPED  # timeout treated as skip
                )
                msg = (
                    f"Step '{step.id}' rejected — not performed."
                    if confirmation == ConfirmationResult.REJECTED
                    else f"Step '{step.id}' timed out — treated as rejection (Req 22.8)."
                )
                await event_queue.put(StepEvent(
                    event_type=event_type,
                    step_id=step.id,
                    step=step,
                    message=msg,
                ))

                # Record rejection/timeout for the report
                if confirmation == ConfirmationResult.REJECTED:
                    report.rejected_step_id = step.id
                else:
                    report.timeout_step_id = step.id

                report.not_performed.append(step)

                # Cascade-stop transitive dependents (Reqs 22.5, 22.6)
                dependents = plan.transitive_dependents(step.id)
                for dep in dependents:
                    dep.status = StepStatus.SKIPPED
                    report.not_performed.append(dep)
                    await event_queue.put(StepEvent(
                        event_type=StepEventType.SKIPPED,
                        step_id=dep.id,
                        step=dep,
                        message=(
                            f"Step '{dep.id}' skipped because its "
                            f"predecessor '{step.id}' was not performed."
                        ),
                    ))
                return  # Do not execute this step

        else:
            # Reversible step: no confirmation needed (Req 22.4)
            step.status = StepStatus.RUNNING

        # ----------------------------------------------------------------
        # 2. Emit STARTED event
        # ----------------------------------------------------------------
        await event_queue.put(StepEvent(
            event_type=StepEventType.STARTED,
            step_id=step.id,
            step=step,
            message=f"Executing: {step.intent}",
        ))

        # ----------------------------------------------------------------
        # 3. Execute via actuator callback
        # ----------------------------------------------------------------
        try:
            output = await self._execute_actuator(step)
        except Exception as exc:
            # Actuator raised — mark FAILED and stop dependents
            step.status = StepStatus.FAILED
            await event_queue.put(StepEvent(
                event_type=StepEventType.FAILED,
                step_id=step.id,
                step=step,
                message=f"Step '{step.id}' failed: {exc}",
            ))
            report.failed.append(step)

            dependents = plan.transitive_dependents(step.id)
            for dep in dependents:
                dep.status = StepStatus.SKIPPED
                report.not_performed.append(dep)
                await event_queue.put(StepEvent(
                    event_type=StepEventType.SKIPPED,
                    step_id=dep.id,
                    step=dep,
                    message=(
                        f"Step '{dep.id}' skipped because its "
                        f"predecessor '{step.id}' failed."
                    ),
                ))
            return

        # ----------------------------------------------------------------
        # 4. Mark COMPLETED
        # ----------------------------------------------------------------
        step.status = StepStatus.COMPLETED
        await event_queue.put(StepEvent(
            event_type=StepEventType.COMPLETED,
            step_id=step.id,
            step=step,
            message=f"Step '{step.id}' completed.",
            data=output,
        ))

    # ------------------------------------------------------------------
    # Actuator dispatch
    # ------------------------------------------------------------------

    async def _execute_actuator(self, step: Step) -> Any:
        """
        Execute *step*'s actuator and return the output.

        Dispatches to either the async callback, the sync callback (in
        a thread executor), or a no-op stub when no callback is configured.
        """
        if self._async_actuator is not None:
            return await self._async_actuator(step)

        if self._sync_actuator is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._sync_actuator, step
            )

        # No actuator configured: no-op (useful for testing gating logic)
        logger.debug(
            "ExecutionEngine: no actuator callback configured for step '%s'; "
            "treating as no-op.",
            step.id,
        )
        return None


# ---------------------------------------------------------------------------
# Convenience: collect all events from an async generator into a list
# ---------------------------------------------------------------------------


async def collect_events(engine: ExecutionEngine, plan: CommandPlan) -> list[StepEvent]:
    """
    Helper that runs *plan* to completion and returns all emitted events.

    Useful in tests and for callers that want a simple non-streaming API.

    Parameters
    ----------
    engine:
        A pre-configured :class:`ExecutionEngine`.
    plan:
        The plan to execute.

    Returns
    -------
    list[StepEvent]
        All events emitted during plan execution, including the final
        ``PLAN_COMPLETE`` event.
    """
    events: list[StepEvent] = []
    stream = await engine.execute(plan)
    async for event in stream:
        events.append(event)
    return events
