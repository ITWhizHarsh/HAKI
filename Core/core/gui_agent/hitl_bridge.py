"""
HITLBridge — Human-in-the-Loop secure field detection and pause/resume.

Detects AXSecureTextField in the current screen state (via macOS Accessibility
API or a Gemini metadata flag), pauses SidecarAgentLoop, emits AGENT_EVENT
messages over the JSON IPC socket, and handles spoken user responses.

Design: HITLBridge section of Gemini-Sidecar Architecture design document.
Requirements: 8.1, 8.3, 8.4, 8.5, 10.3
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

__all__ = ["HITLBridge", "HITLTimeoutError"]

if TYPE_CHECKING:
    from core.ipc.server import JSONIPCServer

from core.ipc.server import AgentEventType

logger = logging.getLogger(__name__)


class HITLTimeoutError(RuntimeError):
    """Raised when no HITL response is received within TIMEOUT_SECONDS."""


class HITLBridge:
    """
    Detects secure input fields and coordinates Human-in-the-Loop pauses.

    Responsibilities:
    - Detect `AXSecureTextField` in the current screen state (via Accessibility
      API or Gemini response metadata flag).
    - Pause SidecarAgentLoop via asyncio.Event.
    - Emit `agent_hitl_pause` over IPC.
    - Receive user's spoken answer from Pipecat STT and inject into the loop.
    - Emit `agent_hitl_resume` and set the resume event.
    - Implement a 60 s timeout; on expiry emit `agent_error` and terminate.
    """

    TIMEOUT_SECONDS: float = 60.0

    def __init__(self, ipc_server: "JSONIPCServer") -> None:
        self._ipc_server = ipc_server
        self._pause_event = asyncio.Event()
        self._resume_event = asyncio.Event()
        self._injected_text: str | None = None
        # Metadata flag set externally (e.g. from Gemini response metadata)
        # used as fallback when the Accessibility API is unavailable.
        self._ax_secure_field_detected: bool = False

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def should_pause(self) -> bool:
        """Return True if the current screen state requires HITL.

        Attempts to use the macOS Accessibility API (pyobjc) to inspect the
        focused UI element's AX role.  Falls back to the metadata flag
        ``_ax_secure_field_detected`` if the API is unavailable.

        Must be synchronous (called from SidecarAgentLoop's sync Verify step).
        Requirements: 8.1
        """
        try:
            # Attempt to use the macOS Accessibility API via pyobjc.
            from ApplicationServices import (  # type: ignore[import]
                AXUIElementCreateSystemWide,
                AXUIElementCopyAttributeValue,
                kAXFocusedUIElementAttribute,
                kAXRoleAttribute,
            )

            system_element = AXUIElementCreateSystemWide()

            # Get the focused UI element.
            err, focused = AXUIElementCopyAttributeValue(
                system_element, kAXFocusedUIElementAttribute, None
            )
            if err != 0 or focused is None:
                logger.debug(
                    "HITLBridge.should_pause: could not get focused element (err=%d)", err
                )
                return self._ax_secure_field_detected

            # Check the AX role of the focused element.
            err, role = AXUIElementCopyAttributeValue(
                focused, kAXRoleAttribute, None
            )
            if err != 0:
                logger.debug(
                    "HITLBridge.should_pause: could not get role (err=%d)", err
                )
                return self._ax_secure_field_detected

            is_secure = role == "AXSecureTextField"
            if is_secure:
                logger.debug("HITLBridge.should_pause: AXSecureTextField detected via AX API")
            return is_secure

        except ImportError:
            # ApplicationServices (pyobjc) not available — fall back to the
            # metadata flag set from Gemini response parsing.
            logger.debug(
                "HITLBridge.should_pause: ApplicationServices not available, "
                "falling back to _ax_secure_field_detected=%s",
                self._ax_secure_field_detected,
            )
            return self._ax_secure_field_detected

    def set_secure_field_flag(self, value: bool) -> None:
        """Set the metadata fallback flag for secure field detection.

        Called by external components (e.g. Gemini metadata parser) to signal
        that the current screen state involves a secure/password field when
        the Accessibility API is unavailable.

        Requirements: 8.1
        """
        self._ax_secure_field_detected = value
        logger.debug("HITLBridge.set_secure_field_flag: _ax_secure_field_detected=%s", value)

    # ------------------------------------------------------------------
    # Pause / resume  (stubs — implemented in tasks 6.2 and 6.3)
    # ------------------------------------------------------------------

    async def pause_and_wait(self) -> str | None:
        """Pause loop, emit agent_hitl_pause, wait for user answer (60 s timeout).

        Broadcasts agent_hitl_pause, then waits on _resume_event with a 60 s
        timeout. On timeout, broadcasts agent_error with the standard message
        and raises HITLTimeoutError.

        Requirements: 8.1, 8.5
        """
        await self._ipc_server.broadcast_agent_event(
            AgentEventType.AGENT_HITL_PAUSE, {}
        )
        try:
            await asyncio.wait_for(self._resume_event.wait(), timeout=self.TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            await self._ipc_server.broadcast_agent_event(
                AgentEventType.AGENT_ERROR,
                {"message": "HITL timeout: no user response"},
            )
            raise HITLTimeoutError("HITL timeout: no user response")
        return self._injected_text

    def inject_response(self, text: str) -> None:
        """Called by Pipecat STT handler when user speaks the HITL answer.

        Stores the text in memory only — never logged or persisted (Req 10.3).
        Sets the resume event to unblock pause_and_wait.
        Then broadcasts agent_hitl_resume over IPC.
        Requirements: 8.3, 8.4, 10.3
        """
        self._injected_text = text                    # held in-memory only, never logged
        self._resume_event.set()                       # unblock pause_and_wait
        # Schedule the async broadcast without blocking the caller's sync context.
        # inject_response is synchronous (called from Pipecat's STT callback thread),
        # so we must not await here. Use asyncio.get_event_loop().create_task if a
        # loop is running, otherwise fall back to asyncio.run_coroutine_threadsafe
        # with the IPC server's loop.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._ipc_server.broadcast_agent_event(
                    AgentEventType.AGENT_HITL_RESUME, {}
                )
            )
        except RuntimeError:
            # No running loop in this thread — use run_coroutine_threadsafe with
            # the loop from the IPC server if available, or log a warning.
            ipc_loop = getattr(self._ipc_server, "_loop", None)
            if ipc_loop is not None and ipc_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._ipc_server.broadcast_agent_event(
                        AgentEventType.AGENT_HITL_RESUME, {}
                    ),
                    ipc_loop,
                )
            else:
                logger.warning(
                    "HITLBridge.inject_response: no running event loop available "
                    "to broadcast agent_hitl_resume"
                )
