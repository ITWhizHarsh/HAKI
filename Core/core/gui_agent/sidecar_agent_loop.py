"""
sidecar_agent_loop — SidecarAgentLoop state machine for Gemini-Sidecar GUI Agent.

Implements the core agent loop that drives GUI automation using Gemini vision
intelligence, macOS Quartz execution, and Human-in-the-Loop pause/resume.

State machine:
    IDLE → RUNNING → (each step) → CHECK_HITL → RUNNING
                                 → HITL_PAUSED → HITL_RESUMED → RUNNING
                                 → DONE        → IDLE
                                 → ERROR       → IDLE
                                 → MAX_STEPS   → IDLE

Requirements: 5.1, 5.5
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

__all__ = ["SidecarAgentLoop"]

if TYPE_CHECKING:
    from core.ipc.server import JSONIPCServer
    from core.gui_agent.gemini_vision_client import GeminiVisionClient, GeminiAction, GeminiAPIError
    from core.gui_agent.mac_quartz_executor import MacQuartzExecutor
    from core.gui_agent.hitl_bridge import HITLBridge

from core.ipc.server import AgentEventType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MAX_STEPS: int = 20
"""Maximum number of cognitive steps before the loop terminates with agent_max_steps_reached."""

RETRY_LIMIT: int = 3
"""Maximum number of consecutive Gemini API retries on retriable errors."""

RETRY_DELAY: float = 2.0
"""Seconds to wait between Gemini API retry attempts."""


# ---------------------------------------------------------------------------
# State enum (string constants)
# ---------------------------------------------------------------------------

class _State:
    """Internal state machine states for SidecarAgentLoop."""
    IDLE          = "IDLE"
    RUNNING       = "RUNNING"
    CHECK_HITL    = "CHECK_HITL"
    HITL_PAUSED   = "HITL_PAUSED"
    HITL_RESUMED  = "HITL_RESUMED"
    DONE          = "DONE"
    ERROR         = "ERROR"
    MAX_STEPS     = "MAX_STEPS"


# ---------------------------------------------------------------------------
# SidecarAgentLoop
# ---------------------------------------------------------------------------

class SidecarAgentLoop:
    """
    Async agent loop that drives the Gemini-Sidecar GUI automation workflow.

    Accepts injected collaborators (ipc_server, vision_client, executor,
    hitl_bridge) so all dependencies can be replaced with test doubles in
    unit tests, satisfying Req 5.1's requirement for isolation.

    Usage::

        loop = SidecarAgentLoop(ipc_server, vision_client, executor, hitl_bridge)
        await loop.run("Open Safari and navigate to example.com")

    The loop broadcasts IPC AGENT_EVENTs at each major lifecycle boundary:
    - ``agent_start``              — immediately on ``run()`` entry
    - ``agent_done``               — task completed successfully (Req 5.5)
    - ``agent_error``              — unhandled exception or fatal error
    - ``agent_max_steps_reached``  — step limit exceeded without task completion

    Requirements: 5.1, 5.5
    """

    def __init__(
        self,
        ipc_server: "JSONIPCServer",
        vision_client: "GeminiVisionClient | None" = None,
        executor: "MacQuartzExecutor | None" = None,
        hitl_bridge: "HITLBridge | None" = None,
    ) -> None:
        """
        Initialise the SidecarAgentLoop with injected collaborators.

        Args:
            ipc_server:    JSONIPCServer instance used to broadcast AGENT_EVENTs
                           to connected Swift clients.
            vision_client: GeminiVisionClient for screen capture and Gemini API
                           queries.  If None, will be instantiated lazily in
                           _run_loop (task 8.2).
            executor:      MacQuartzExecutor for dispatching click/type/scroll
                           actions via macOS Quartz.  If None, will be
                           instantiated lazily in _run_loop (task 8.2).
            hitl_bridge:   HITLBridge for detecting AXSecureTextField and
                           pausing/resuming the loop for HITL.  If None, HITL
                           detection is disabled.
        """
        self._ipc_server = ipc_server
        self._vision_client = vision_client
        self._executor = executor
        self._hitl_bridge = hitl_bridge

        # Internal state
        self._state: str = _State.IDLE
        self._abort: bool = False

        logger.debug(
            "SidecarAgentLoop: initialised (vision_client=%s, executor=%s, hitl_bridge=%s)",
            type(vision_client).__name__ if vision_client is not None else None,
            type(executor).__name__ if executor is not None else None,
            type(hitl_bridge).__name__ if hitl_bridge is not None else None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, task_description: str) -> None:
        """
        Entry point for the agent loop.

        Broadcasts ``agent_start`` with the task description, then calls
        ``_run_loop``.  On successful completion, ``_run_loop`` is expected to
        broadcast its own terminal event (``agent_done`` or
        ``agent_max_steps_reached``).  Any unhandled exception propagating out
        of ``_run_loop`` is caught here and broadcasts ``agent_error`` before
        re-raising.

        Args:
            task_description: Natural-language description of the GUI task to
                              complete (e.g. "Open Safari and navigate to
                              example.com").

        Requirements: 5.1 (isolated async execution), 5.5 (agent_done with summary)
        """
        logger.info("SidecarAgentLoop.run: starting task=%r", task_description[:80])
        self._state = _State.RUNNING
        self._abort = False

        # Broadcast agent_start as the very first IPC event (before any work).
        await self._ipc_server.broadcast_agent_event(
            AgentEventType.AGENT_START,
            {"task": task_description},
        )

        try:
            await self._run_loop(task_description)
        except Exception as exc:
            logger.exception(
                "SidecarAgentLoop.run: unhandled exception from _run_loop: %r", exc
            )
            self._state = _State.ERROR
            await self._ipc_server.broadcast_agent_event(
                AgentEventType.AGENT_ERROR,
                {"message": str(exc)},
            )
            raise
        finally:
            # Ensure state returns to IDLE regardless of outcome so the loop
            # is safe to reuse (or garbage-collect) after run() completes.
            if self._state not in (_State.IDLE,):
                self._state = _State.IDLE

    def abort(self) -> None:
        """
        Signal early termination of the running loop.

        Sets the internal ``_abort`` flag to ``True``.  The loop checks this
        flag at the start of each step iteration (implemented in task 8.2) and
        will stop gracefully on the next check.

        This method is synchronous and asyncio-safe: it only sets a boolean
        flag, making it safe to call from any thread or coroutine context.
        """
        logger.info("SidecarAgentLoop.abort: abort requested")
        self._abort = True

    # ------------------------------------------------------------------
    # Internal loop (task 8.2)
    # ------------------------------------------------------------------

    async def _run_loop(self, task_description: str) -> None:
        """
        Internal cognitive step loop: See→Think→Act→Verify, up to MAX_STEPS.

        Each iteration:
        1. See + Think  — ``_see_think()`` calls GeminiVisionClient (with retry)
        2. Act          — ``executor.dispatch(action)`` dispatches the HID event
        3. Broadcast    — ``agent_step`` IPC event with step number and action type
        4. Verify       — second ``_see_think(verify=True)`` call
        5. HITL check   — ``hitl_bridge.should_pause()`` / ``pause_and_wait()``
        6. Done check   — if ``action.action_type == "done"`` → broadcast ``agent_done``

        After MAX_STEPS iterations without a ``"done"`` action, broadcasts
        ``agent_max_steps_reached`` and returns.

        Args:
            task_description: Natural-language description of the GUI task.

        Requirements: 5.2, 5.4, 5.5
        """
        for step in range(1, MAX_STEPS + 1):
            # Abort check at the start of each step (set via abort())
            if self._abort:
                logger.info("SidecarAgentLoop._run_loop: abort flag set — stopping at step %d", step)
                return

            # ----------------------------------------------------------------
            # See + Think (with retry — full implementation in _see_think)
            # ----------------------------------------------------------------
            action = await self._see_think(task_description, step)
            if action is None:
                # _see_think exhausted all retries; error already logged inside
                logger.error(
                    "SidecarAgentLoop._run_loop: _see_think returned None at step %d — stopping", step
                )
                self._state = _State.ERROR
                await self._broadcast(AgentEventType.AGENT_ERROR, {"step": step})
                return

            # ----------------------------------------------------------------
            # Act — dispatch HID event via MacQuartzExecutor
            # ----------------------------------------------------------------
            if self._executor is not None:
                self._executor.dispatch(action)
            else:
                logger.warning(
                    "SidecarAgentLoop._run_loop: no executor configured — skipping dispatch at step %d",
                    step,
                )

            # Broadcast agent_step so UI/Swift clients can track progress
            await self._broadcast(
                AgentEventType.AGENT_STEP,
                {"step": step, "action": action.action_type},
            )

            # ----------------------------------------------------------------
            # Verify — a follow-up See+Think to confirm the action took effect
            # ----------------------------------------------------------------
            _verify_action = await self._see_think(task_description, step, verify=True)
            # Verification result is informational; a None result here does not
            # abort the loop (the main action has already been dispatched).

            # ----------------------------------------------------------------
            # HITL check — pause if a secure input field is detected
            # ----------------------------------------------------------------
            if self._hitl_bridge is not None and self._hitl_bridge.should_pause():
                self._state = _State.HITL_PAUSED
                logger.info("SidecarAgentLoop._run_loop: HITL pause triggered at step %d", step)
                await self._hitl_bridge.pause_and_wait()
                self._state = _State.RUNNING

            # ----------------------------------------------------------------
            # Done check — task declared complete by Gemini
            # ----------------------------------------------------------------
            if action.action_type == "done":
                self._state = _State.DONE
                logger.info("SidecarAgentLoop._run_loop: task done at step %d", step)
                await self._broadcast(
                    AgentEventType.AGENT_DONE,
                    {"success": True, "summary": action.summary},
                )
                return

        # Reached MAX_STEPS without a "done" action
        self._state = _State.MAX_STEPS
        logger.warning(
            "SidecarAgentLoop._run_loop: reached MAX_STEPS (%d) without completion", MAX_STEPS
        )
        await self._broadcast(
            AgentEventType.AGENT_MAX_STEPS_REACHED,
            {"steps": MAX_STEPS},
        )

    async def _see_think(
        self,
        task: str,
        step: int,
        verify: bool = False,
    ) -> "GeminiAction | None":
        """
        See + Think phase: query Gemini with a captured screen frame.

        Retries up to RETRY_LIMIT times (with RETRY_DELAY seconds between
        attempts) on retriable ``GeminiAPIError``.  Returns ``None`` when all
        retries are exhausted; propagates non-retriable errors immediately.

        This is a placeholder stub.  Full implementation (retry loop, error
        handling) is provided in task 8.3.

        Args:
            task:   Natural-language task description forwarded to Gemini.
            step:   Current loop step number (used for logging/context).
            verify: When True, this is the Verify phase; context hint only.

        Returns:
            A ``GeminiAction`` on success, or ``None`` on exhausted retries.

        Requirements: 5.6, 5.7 (implemented fully in task 8.3)
        """
        from core.gui_agent.gemini_vision_client import GeminiAPIError

        phase = "verify" if verify else "see_think"
        for attempt in range(RETRY_LIMIT):
            try:
                if self._vision_client is None:
                    logger.error(
                        "SidecarAgentLoop._see_think: no vision_client configured at step %d", step
                    )
                    return None
                return await self._vision_client.query_gemini(task)
            except GeminiAPIError as exc:
                logger.warning(
                    "SidecarAgentLoop._see_think: GeminiAPIError at step=%d phase=%s attempt=%d/%d — %r",
                    step, phase, attempt + 1, RETRY_LIMIT, exc,
                )
                if attempt < RETRY_LIMIT - 1:
                    await asyncio.sleep(RETRY_DELAY)
            except Exception as exc:
                # Non-retriable error — propagate immediately (Req 5.7)
                logger.error(
                    "SidecarAgentLoop._see_think: non-retriable error at step=%d phase=%s — %r",
                    step, phase, exc,
                )
                raise

        # All RETRY_LIMIT attempts exhausted
        logger.error(
            "SidecarAgentLoop._see_think: exhausted %d retries at step=%d phase=%s — returning None",
            RETRY_LIMIT, step, phase,
        )
        return None

    # ------------------------------------------------------------------
    # Private broadcast helper
    # ------------------------------------------------------------------

    async def _broadcast(self, event_type: str, payload: dict) -> None:
        """
        Thin wrapper around ``ipc_server.broadcast_agent_event``.

        Logs the event at DEBUG level and delegates to the IPC server.  Any
        exception from the IPC layer is caught and logged so that a broadcast
        failure never terminates the agent loop.

        Args:
            event_type: One of the ``AgentEventType`` string constants.
            payload:    Arbitrary JSON-serialisable dict sent with the event.
        """
        logger.debug(
            "SidecarAgentLoop._broadcast: event_type=%r payload=%r", event_type, payload
        )
        try:
            await self._ipc_server.broadcast_agent_event(event_type, payload)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "SidecarAgentLoop._broadcast: failed to broadcast %r — %r", event_type, exc
            )
