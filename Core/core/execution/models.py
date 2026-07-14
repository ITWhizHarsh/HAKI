"""
Execution event and report models for the ExecutionEngine.

These dataclasses represent the streaming events emitted during plan
execution and the final completion report.

Design: Execution loop, ExecutionEngine interface.
Requirements: 17.2, 21.8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# StepEvent — single event emitted as a step progresses
# ---------------------------------------------------------------------------


@dataclass
class StepEvent:
    """
    An event emitted by the :class:`~core.execution.ExecutionEngine`
    during plan execution.

    Each event corresponds to one lifecycle transition of a single step.

    Attributes
    ----------
    type:
        The kind of event.  One of:

        - ``'started'`` — the step has begun executing.
        - ``'completed'`` — the step completed; ``result`` holds the
          actuator output.
        - ``'failed'`` — the step failed (actuator error or postcondition
          not met); ``reason`` describes why.
        - ``'awaiting_confirmation'`` — execution is paused at this step
          because the Safety_Gate requires user confirmation before the
          consequential action is performed (Req 22.1, 22.2).
        - ``'awaiting_clarification'`` — execution is paused because a
          required slot for this step has not yet been resolved
          (Req 23.1, 23.3).
        - ``'cancelled'`` — the step was skipped due to plan cancellation,
          upstream failure, or upstream rejection.

    step_id:
        Stable identifier of the step this event concerns.
    result:
        The actuator output when ``type == 'completed'``.
    reason:
        A human-readable explanation when ``type == 'failed'`` or
        ``type == 'cancelled'``.

    Design: ExecutionEngine interface.
    Requirements: 17.2, 21.8.
    """

    type: Literal[
        "started",
        "completed",
        "failed",
        "awaiting_confirmation",
        "awaiting_clarification",
        "cancelled",
    ]
    step_id: str
    result: Any = None
    reason: str | None = None


# ---------------------------------------------------------------------------
# PlanCompletionEvent — final summary emitted once the plan finishes
# ---------------------------------------------------------------------------


@dataclass
class PlanCompletionEvent:
    """
    Emitted once at the end of every plan execution to report the
    outcome to the caller.

    Satisfies Req 21.8: "WHEN the Mac_Controller completes execution of
    a Command_Plan, THE Mac_Controller SHALL report to the User which
    steps were completed."

    Attributes
    ----------
    executed_step_ids:
        IDs of steps that reached ``COMPLETED`` status — i.e. the steps
        whose actuator ran successfully and whose postcondition (if any)
        was satisfied.
    not_performed_step_ids:
        IDs of steps that were skipped (due to upstream failure,
        upstream rejection, or plan cancellation) and were therefore
        never attempted.
    failed_steps:
        Pairs of ``(step_id, reason)`` for steps that were attempted but
        failed (actuator error or postcondition failure).

    Design: ExecutionEngine interface, Execution loop.
    Requirements: 21.8.
    """

    executed_step_ids: list[str] = field(default_factory=list)
    not_performed_step_ids: list[str] = field(default_factory=list)
    failed_steps: list[tuple[str, str]] = field(default_factory=list)

    @property
    def all_completed(self) -> bool:
        """``True`` when every step completed and none failed or was skipped."""
        return not self.not_performed_step_ids and not self.failed_steps

    def summary(self) -> str:
        """Return a brief human-readable summary of plan execution."""
        parts: list[str] = []
        if self.executed_step_ids:
            parts.append(f"Completed: {self.executed_step_ids}")
        if self.failed_steps:
            parts.append(f"Failed: {[(s, r) for s, r in self.failed_steps]}")
        if self.not_performed_step_ids:
            parts.append(f"Not performed: {self.not_performed_step_ids}")
        return "; ".join(parts) if parts else "No steps executed."
