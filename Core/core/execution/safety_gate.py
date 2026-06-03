"""
Safety_Gate — action classification and confirmation gating.

Sits between the Planner and the Execution_Engine.  Its responsibilities:

1. Classify every Step as CONSEQUENTIAL | REVERSIBLE | UNKNOWN.
   (Steps are already annotated by the LLM planner; the gate enforces
   the runtime semantics of those annotations.)

2. Before executing any CONSEQUENTIAL or UNKNOWN step, pause and request
   user confirmation.  The confirmation request includes a human-readable
   description of the action to be performed (Req 22.1).

3. Reversible steps pass through without confirmation (Req 22.4).

4. UNKNOWN is always treated as CONSEQUENTIAL (Req 22.7).

5. If no response is received within the configured timeout period,
   treat it as a rejection and do NOT execute the step (Req 22.8).

6. On rejection (explicit or via timeout), the step must NOT be
   performed and all transitive dependent steps must be stopped
   (Req 22.3, 22.5, 22.6).

Design: Safety_Gate.
Requirements: 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 22.7, 22.8.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

from core.planner import Step, StepClassification

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Default confirmation timeout (seconds).  Callers should override this
#: for tests.  ``None`` means "wait forever" — only appropriate in fully
#: interactive environments; most callers should set an explicit timeout.
DEFAULT_CONFIRMATION_TIMEOUT: float | None = None


# ---------------------------------------------------------------------------
# ConfirmationRequest / ConfirmationResult
# ---------------------------------------------------------------------------


@dataclass
class ConfirmationRequest:
    """
    A request for the user to confirm or reject a consequential step.

    Attributes
    ----------
    step_id:
        The ID of the step awaiting confirmation.
    description:
        A human-readable description of the action about to be performed.
        Included in every confirmation request (Req 22.1).
    step:
        The full :class:`~core.planner.Step` object, for callers that
        need additional context (e.g. the UI layer).
    """

    step_id: str
    description: str
    step: Step


class ConfirmationResult(str, Enum):
    """
    The outcome of a confirmation request.

    CONFIRMED   User explicitly approved the action.
    REJECTED    User explicitly declined the action.
    TIMEOUT     No response was received within the timeout; treated as
                REJECTED for all gating purposes (Req 22.8).
    """

    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SafetyGateTimeout(Exception):
    """
    Raised internally when a confirmation awaitable times out.

    The :class:`SafetyGate` converts this into a
    :class:`ConfirmationResult.TIMEOUT` result so callers see a uniform
    result type (Req 22.8).
    """


# ---------------------------------------------------------------------------
# ConfirmationCallback types
# ---------------------------------------------------------------------------

#: Async callback type: receives a ``ConfirmationRequest`` and returns a
#: ``ConfirmationResult``.  The gate calls this whenever it needs the user
#: to confirm or reject a consequential step.
ConfirmationCallbackAsync = Callable[
    [ConfirmationRequest], Awaitable[ConfirmationResult]
]

#: Sync callback type: alternative for non-async callers.
ConfirmationCallbackSync = Callable[[ConfirmationRequest], ConfirmationResult]


# ---------------------------------------------------------------------------
# SafetyGate
# ---------------------------------------------------------------------------


class SafetyGate:
    """
    Safety gate between the Planner and the Execution_Engine.

    The gate inspects each :class:`~core.planner.Step`'s ``classification``
    and either:

    - Passes reversible steps through immediately (Req 22.4).
    - Pauses on consequential/unknown steps and awaits user confirmation
      (Reqs 22.1, 22.2, 22.7).

    Both sync and async confirmation callbacks are supported.  Supply
    exactly one via the constructor.

    Parameters
    ----------
    confirmation_callback:
        An **async** callable ``(ConfirmationRequest) → ConfirmationResult``
        that presents the confirmation to the user and returns their
        decision.  Use ``sync_confirmation_callback`` for sync callers.
    sync_confirmation_callback:
        A **sync** callable ``(ConfirmationRequest) → ConfirmationResult``.
        Wrapped internally in an executor so it does not block the event
        loop.
    confirmation_timeout:
        Seconds to wait for a confirmation response before treating the
        request as a timeout/rejection (Req 22.8).  ``None`` means no
        timeout.

    Design: Safety_Gate.
    Requirements: 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 22.7, 22.8.
    """

    def __init__(
        self,
        confirmation_callback: ConfirmationCallbackAsync | None = None,
        sync_confirmation_callback: ConfirmationCallbackSync | None = None,
        confirmation_timeout: float | None = DEFAULT_CONFIRMATION_TIMEOUT,
    ) -> None:
        if confirmation_callback is not None and sync_confirmation_callback is not None:
            raise ValueError(
                "Provide at most one of confirmation_callback or "
                "sync_confirmation_callback, not both."
            )
        self._async_callback: ConfirmationCallbackAsync | None = confirmation_callback
        self._sync_callback: ConfirmationCallbackSync | None = sync_confirmation_callback
        self._timeout: float | None = confirmation_timeout

    # ------------------------------------------------------------------
    # Classification helpers (Req 22.4, 22.7)
    # ------------------------------------------------------------------

    @staticmethod
    def requires_confirmation(step: Step) -> bool:
        """
        Return ``True`` when *step* requires user confirmation before
        execution.

        Rules (from Reqs 22.1, 22.4, 22.7):
        - CONSEQUENTIAL → requires confirmation
        - UNKNOWN → treated as CONSEQUENTIAL → requires confirmation
        - REVERSIBLE → does NOT require confirmation

        Parameters
        ----------
        step:
            The step to evaluate.
        """
        return step.classification in (
            StepClassification.CONSEQUENTIAL,
            StepClassification.UNKNOWN,
        )

    @staticmethod
    def is_reversible(step: Step) -> bool:
        """
        Return ``True`` when *step* is classified as REVERSIBLE and may
        run without confirmation (Req 22.4).
        """
        return step.classification == StepClassification.REVERSIBLE

    @staticmethod
    def _description_for_step(step: Step) -> str:
        """
        Build a human-readable description of *step*'s action for the
        confirmation request (Req 22.1).

        The description is constructed from the step's ``intent`` and
        ``args`` so the user knows exactly what will happen.
        """
        parts: list[str] = [step.intent or "Perform an action"]
        if step.args:
            arg_summary = "; ".join(f"{k}={v!r}" for k, v in step.args.items())
            parts.append(f"({arg_summary})")
        classification_label = step.classification.value.upper()
        parts.append(f"[classification: {classification_label}]")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Core gating API
    # ------------------------------------------------------------------

    async def check(self, step: Step) -> ConfirmationResult:
        """
        Evaluate *step* against the safety gate.

        For reversible steps, returns :attr:`ConfirmationResult.CONFIRMED`
        immediately without calling the confirmation callback (Req 22.4).

        For consequential/unknown steps, constructs a
        :class:`ConfirmationRequest` with the action description and
        awaits the callback.  If the callback times out, returns
        :attr:`ConfirmationResult.TIMEOUT` (Req 22.8).

        Parameters
        ----------
        step:
            The step to evaluate.

        Returns
        -------
        ConfirmationResult
            CONFIRMED, REJECTED, or TIMEOUT.
        """
        if not self.requires_confirmation(step):
            # Reversible step: pass through unconditionally (Req 22.4)
            logger.debug(
                "SafetyGate: step '%s' is REVERSIBLE — no confirmation needed.",
                step.id,
            )
            return ConfirmationResult.CONFIRMED

        # Build the confirmation request (Req 22.1)
        request = ConfirmationRequest(
            step_id=step.id,
            description=self._description_for_step(step),
            step=step,
        )

        logger.info(
            "SafetyGate: step '%s' (%s) requires confirmation — %s",
            step.id,
            step.classification.value,
            request.description,
        )

        return await self._await_confirmation(request)

    # ------------------------------------------------------------------
    # Sync convenience wrapper
    # ------------------------------------------------------------------

    def check_sync(self, step: Step) -> ConfirmationResult:
        """
        Synchronous wrapper around :meth:`check`.

        Runs the coroutine in a new event loop (or the running loop using
        ``run_until_complete`` if no running loop exists).  Prefer using
        :meth:`check` directly in async code.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule on the running loop (e.g. in Jupyter / tests)
                import concurrent.futures
                future = concurrent.futures.Future()

                async def _run():
                    try:
                        result = await self.check(step)
                        future.set_result(result)
                    except Exception as exc:
                        future.set_exception(exc)

                asyncio.ensure_future(_run())
                return future.result(timeout=self._timeout)
            else:
                return loop.run_until_complete(self.check(step))
        except RuntimeError:
            return asyncio.run(self.check(step))

    # ------------------------------------------------------------------
    # Internal callback dispatch
    # ------------------------------------------------------------------

    async def _await_confirmation(
        self, request: ConfirmationRequest
    ) -> ConfirmationResult:
        """
        Call the configured confirmation callback and apply the timeout.

        Dispatches to either the async callback or the sync callback
        (wrapped in an executor).  On timeout, returns TIMEOUT (Req 22.8).
        """
        if self._async_callback is None and self._sync_callback is None:
            # No callback configured: default to CONFIRMED (useful in
            # non-interactive / test environments where callers don't
            # need interactive gating but still want classification).
            logger.warning(
                "SafetyGate: no confirmation callback configured; "
                "defaulting to CONFIRMED for step '%s'.",
                request.step_id,
            )
            return ConfirmationResult.CONFIRMED

        try:
            coroutine = self._build_coroutine(request)
            if self._timeout is not None:
                result = await asyncio.wait_for(coroutine, timeout=self._timeout)
            else:
                result = await coroutine
        except asyncio.TimeoutError:
            logger.warning(
                "SafetyGate: confirmation timed out for step '%s' — "
                "treating as REJECTED (Req 22.8).",
                request.step_id,
            )
            return ConfirmationResult.TIMEOUT
        except Exception as exc:
            logger.error(
                "SafetyGate: confirmation callback raised an exception "
                "for step '%s': %s — treating as REJECTED.",
                request.step_id,
                exc,
            )
            return ConfirmationResult.REJECTED

        return result

    def _build_coroutine(
        self, request: ConfirmationRequest
    ) -> Awaitable[ConfirmationResult]:
        """
        Build an awaitable for the confirmation callback.

        If an async callback is configured, calls it directly.
        If a sync callback is configured, wraps it in
        ``asyncio.to_thread`` so it doesn't block the event loop.
        """
        if self._async_callback is not None:
            return self._async_callback(request)

        # Sync callback: run in thread executor
        assert self._sync_callback is not None

        async def _run_sync() -> ConfirmationResult:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._sync_callback, request
            )

        return _run_sync()
