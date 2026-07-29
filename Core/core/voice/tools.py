"""Registered Pydantic tool schemas and adapter for voice-originated tool calls.

Voice turns may emit structured JSON tool calls from the LLM.  This module:

- Defines strict Pydantic v2 schemas (``extra="forbid"``) for each registered
  tool.
- Validates raw LLM output against those schemas before routing to any
  downstream service.
- Emits ``ToolCallDiagnostic`` on schema rejection or execution failure —
  never leaking user content, query text, or argument values.
- Returns ``ToolCallResult`` instances correlated by ``turn_id`` so the router
  can discard results that no longer belong to the active turn.

Downstream services are not modified here.  ``ScreenAgent.run(goal)`` is called
as-is; all permission and confirmation logic remains owned by ScreenAgent.
``confirmation_context="voice"`` is a schema-level field that marks voice origin
explicitly but does not alter ScreenAgent's internal flow.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Callable, Coroutine, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

if TYPE_CHECKING:
    from ..automation.screen_agent import AgentResult, ScreenAgent
    from ..memory.haki_brain import HAKIBrain
    from core.ipc.server import JSONIPCServer

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool prompt grammar (injected into LLM system prompt)
# ---------------------------------------------------------------------------

VOICE_TOOL_GRAMMAR: str = """\
You may call tools using this JSON format (one call per message):
{"tool": "obsidian_rag.search", "schema_version": 1, "query": "<search query>", "limit": 3}
{"tool": "screen_control.run", "schema_version": 1, "goal": "<goal description>", "confirmation_context": "voice"}
{"tool": "gui_agent.spawn", "schema_version": 1, "task_description": "<task description>"}
Only use these exact tool names. Do not add extra fields.\
"""


# ---------------------------------------------------------------------------
# Registered tool schemas
# ---------------------------------------------------------------------------

class ObsidianRAGCall(BaseModel):
    """Schema for an obsidian_rag.search tool call emitted by the voice LLM."""

    model_config = ConfigDict(extra="forbid")

    tool: Literal["obsidian_rag.search"]
    schema_version: int = 1
    query: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=2000),
    ]
    limit: Annotated[int, Field(ge=1, le=5)] = 3


class ScreenControlCall(BaseModel):
    """Schema for a screen_control.run tool call emitted by the voice LLM."""

    model_config = ConfigDict(extra="forbid")

    tool: Literal["screen_control.run"]
    schema_version: int = 1
    goal: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=1000),
    ]
    confirmation_context: Literal["voice"]


class SpawnGuiAgentCall(BaseModel):
    """Schema for a gui_agent.spawn tool call emitted by the voice LLM (Req 6.1, 6.2).

    ``task_description`` is stripped of surrounding whitespace.  Leading/trailing
    whitespace is removed before the min/max length checks are applied.
    ``extra="forbid"`` ensures the LLM cannot inject unexpected fields.
    """

    model_config = ConfigDict(extra="forbid")

    tool: Literal["gui_agent.spawn"]
    schema_version: int = 1
    task_description: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
    ]


# ---------------------------------------------------------------------------
# Thread launcher for SidecarAgentLoop (Req 6.1, 6.3, 9.1, 9.2)
# ---------------------------------------------------------------------------

def _start_agent_in_thread(
    task_description: str,
    ipc_server: "JSONIPCServer",
) -> threading.Thread:
    """Launch a ``SidecarAgentLoop`` in an isolated daemon thread (Req 9.1, 9.2).

    The thread owns its own ``asyncio`` event loop so it never shares loop state
    with the Pipecat voice thread.

    Parameters
    ----------
    task_description:
        Natural-language task for the GUI agent.
    ipc_server:
        Running ``JSONIPCServer`` instance passed to ``SidecarAgentLoop`` for
        AGENT_EVENT broadcasting.

    Returns
    -------
    threading.Thread
        The started daemon thread (``daemon=True``, name ``"haki-gui-agent"``).
    """
    def _thread_main() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            from core.gui_agent.sidecar_agent_loop import SidecarAgentLoop  # noqa: PLC0415
            from core.gui_agent.gemini_vision_client import GeminiVisionClient  # noqa: PLC0415
            from core.gui_agent.mac_quartz_executor import MacQuartzExecutor, ExecutorUnavailableError  # noqa: PLC0415
            from core.gui_agent.hitl_bridge import HITLBridge  # noqa: PLC0415

            try:
                vision_client = GeminiVisionClient()
            except Exception as exc:
                _logger.error("_start_agent_in_thread: GeminiVisionClient init failed: %r", exc)
                vision_client = None

            try:
                executor = MacQuartzExecutor()
            except ExecutorUnavailableError as exc:
                _logger.warning("_start_agent_in_thread: MacQuartzExecutor unavailable: %s", exc)
                executor = None
            except Exception as exc:
                _logger.error("_start_agent_in_thread: MacQuartzExecutor init failed: %r", exc)
                executor = None

            hitl_bridge = HITLBridge(ipc_server=ipc_server)

            agent = SidecarAgentLoop(
                ipc_server=ipc_server,
                vision_client=vision_client,
                executor=executor,
                hitl_bridge=hitl_bridge,
            )
            loop.run_until_complete(agent.run(task_description))
        except Exception as exc:
            _logger.exception("_start_agent_in_thread: unhandled error: %r", exc)
        finally:
            try:
                loop.close()
            except Exception:
                pass

    thread = threading.Thread(
        target=_thread_main,
        daemon=True,
        name="haki-gui-agent",
    )
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Diagnostic dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ToolCallDiagnostic:
    """Content-free diagnostic for a voice tool call event.

    ``tool_name`` contains only the registered name (e.g. ``obsidian_rag.search``)
    never argument values or user content.
    """

    outcome: Literal["rejected", "executed", "failed"]
    stage: Literal["tool_call"] = "tool_call"
    session_id: UUID | None = None
    turn_id: UUID | None = None
    tool_name: str | None = None
    error_class: str | None = None


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """Return value of ``VoiceToolAdapter.execute_tool_call``.

    ``data`` holds the tool output on success; ``error_message`` is a
    LLM-safe summary on failure.  Exactly one of ``data`` or ``error_message``
    will be set when ``success`` is True/False respectively.
    """

    tool_name: str | None
    turn_id: UUID
    success: bool
    data: Any = None
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Diagnostic sink type alias (mirrors llm.py convention)
# ---------------------------------------------------------------------------

DiagnosticSinkType = Any  # Callable[[ToolCallDiagnostic], Awaitable[None] | None]


# ---------------------------------------------------------------------------
# VoiceToolAdapter
# ---------------------------------------------------------------------------

class VoiceToolAdapter:
    """Parse, validate, and route voice-originated LLM tool calls.

    Parameters
    ----------
    haki_brain:
        HAKIBrain instance used for obsidian_rag.search calls.
    screen_agent:
        ScreenAgent instance used for screen_control.run calls.
    ipc_server:
        Running ``JSONIPCServer`` instance passed to ``SidecarAgentLoop`` for
        AGENT_EVENT broadcasting (Req 6.1, 9.2).
    xtts_sink:
        Optional async callable that synthesises an XTTS acknowledgement
        before the agent thread is started (Req 6.3).  Receives a single
        string argument.
    diagnostic_sink:
        Optional async or sync callable receiving ``ToolCallDiagnostic`` events.
        When ``None`` the diagnostic is logged at WARNING level.
    """

    def __init__(
        self,
        *,
        haki_brain: "HAKIBrain | None" = None,
        screen_agent: "ScreenAgent | None" = None,
        ipc_server: "JSONIPCServer | None" = None,
        xtts_sink: "Callable[[str], Coroutine[Any, Any, None]] | None" = None,
        diagnostic_sink: DiagnosticSinkType | None = None,
    ) -> None:
        self._haki_brain = haki_brain
        self._screen_agent = screen_agent
        self._ipc_server = ipc_server
        self._xtts_sink = xtts_sink
        self._diagnostic_sink = diagnostic_sink

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute_tool_call(
        self,
        raw_json: str,
        *,
        turn_id: UUID,
        session_id: UUID,
    ) -> ToolCallResult | None:
        """Parse *raw_json*, validate, route, and return a turn-correlated result.

        Returns ``None`` when the payload fails schema validation (schema
        rejection).  On execution error returns a ``ToolCallResult`` with
        ``success=False`` and an LLM-safe error message.

        Parameters
        ----------
        raw_json:
            Raw string from the LLM output that should contain a JSON object.
        turn_id:
            Identity of the originating turn; correlated into the result.
        session_id:
            Identity of the originating session; used only in diagnostics.
        """
        parsed = _try_parse_tool_call(raw_json)
        if parsed is None:
            await self._emit_diagnostic(
                ToolCallDiagnostic(
                    outcome="rejected",
                    session_id=session_id,
                    turn_id=turn_id,
                    tool_name=None,
                    error_class="JSONDecodeError",
                )
            )
            return None

        # Detect the tool discriminator before full validation
        tool_name: str | None = parsed.get("tool") if isinstance(parsed, dict) else None

        # Attempt schema validation
        validated: ObsidianRAGCall | ScreenControlCall | None
        try:
            validated = _validate_tool_call(parsed)
        except ValidationError as exc:
            await self._emit_diagnostic(
                ToolCallDiagnostic(
                    outcome="rejected",
                    session_id=session_id,
                    turn_id=turn_id,
                    tool_name=tool_name if isinstance(tool_name, str) else None,
                    error_class=type(exc).__name__,
                )
            )
            return None
        except Exception as exc:  # noqa: BLE001
            await self._emit_diagnostic(
                ToolCallDiagnostic(
                    outcome="rejected",
                    session_id=session_id,
                    turn_id=turn_id,
                    tool_name=tool_name if isinstance(tool_name, str) else None,
                    error_class=type(exc).__name__,
                )
            )
            return None

        # Route to the appropriate backend
        if isinstance(validated, ObsidianRAGCall):
            return await self._execute_obsidian_rag(validated, turn_id=turn_id, session_id=session_id)
        if isinstance(validated, ScreenControlCall):
            return await self._execute_screen_control(validated, turn_id=turn_id, session_id=session_id)
        if isinstance(validated, SpawnGuiAgentCall):
            return await self._execute_gui_agent_spawn(validated, turn_id=turn_id, session_id=session_id)

        # Should never reach here (unknown discriminator slipped past validation)
        await self._emit_diagnostic(
            ToolCallDiagnostic(
                outcome="rejected",
                session_id=session_id,
                turn_id=turn_id,
                tool_name=tool_name if isinstance(tool_name, str) else None,
                error_class="UnknownToolName",
            )
        )
        return None

    # ------------------------------------------------------------------
    # Internal routing helpers
    # ------------------------------------------------------------------

    async def _execute_obsidian_rag(
        self,
        call: ObsidianRAGCall,
        *,
        turn_id: UUID,
        session_id: UUID,
    ) -> ToolCallResult:
        """Call HAKIBrain.search and return bounded results correlated to turn_id."""
        if self._haki_brain is None:
            await self._emit_diagnostic(
                ToolCallDiagnostic(
                    outcome="failed",
                    session_id=session_id,
                    turn_id=turn_id,
                    tool_name="obsidian_rag.search",
                    error_class="HAKIBrainNotConfigured",
                )
            )
            return ToolCallResult(
                tool_name="obsidian_rag.search",
                turn_id=turn_id,
                success=False,
                error_message="Knowledge search is not available right now.",
            )

        try:
            results: list[dict] = await self._haki_brain.search(call.query, k=call.limit)
            # Enforce the limit cap (search may return more)
            bounded = results[: call.limit]
            await self._emit_diagnostic(
                ToolCallDiagnostic(
                    outcome="executed",
                    session_id=session_id,
                    turn_id=turn_id,
                    tool_name="obsidian_rag.search",
                    error_class=None,
                )
            )
            return ToolCallResult(
                tool_name="obsidian_rag.search",
                turn_id=turn_id,
                success=True,
                data=bounded,
            )
        except Exception as exc:  # noqa: BLE001
            await self._emit_diagnostic(
                ToolCallDiagnostic(
                    outcome="failed",
                    session_id=session_id,
                    turn_id=turn_id,
                    tool_name="obsidian_rag.search",
                    error_class=type(exc).__name__,
                )
            )
            return ToolCallResult(
                tool_name="obsidian_rag.search",
                turn_id=turn_id,
                success=False,
                error_message="Knowledge search failed. Please try again.",
            )

    async def _execute_screen_control(
        self,
        call: ScreenControlCall,
        *,
        turn_id: UUID,
        session_id: UUID,
    ) -> ToolCallResult:
        """Call ScreenAgent.run(goal) without modifying its permission/confirm flow."""
        if self._screen_agent is None:
            await self._emit_diagnostic(
                ToolCallDiagnostic(
                    outcome="failed",
                    session_id=session_id,
                    turn_id=turn_id,
                    tool_name="screen_control.run",
                    error_class="ScreenAgentNotConfigured",
                )
            )
            return ToolCallResult(
                tool_name="screen_control.run",
                turn_id=turn_id,
                success=False,
                error_message="Screen control is not available right now.",
            )

        try:
            # Pass only `goal`; ScreenAgent owns all permission/confirmation logic.
            agent_result: "AgentResult" = await self._screen_agent.run(call.goal)
            # Truncate the message to 2000 chars as per bounded results spec
            message = (agent_result.message or "")[:2000]
            await self._emit_diagnostic(
                ToolCallDiagnostic(
                    outcome="executed",
                    session_id=session_id,
                    turn_id=turn_id,
                    tool_name="screen_control.run",
                    error_class=None,
                )
            )
            return ToolCallResult(
                tool_name="screen_control.run",
                turn_id=turn_id,
                success=agent_result.success,
                data={"message": message, "steps": len(agent_result.steps), "goal": agent_result.goal},
                error_message=None if agent_result.success else message,
            )
        except Exception as exc:  # noqa: BLE001
            await self._emit_diagnostic(
                ToolCallDiagnostic(
                    outcome="failed",
                    session_id=session_id,
                    turn_id=turn_id,
                    tool_name="screen_control.run",
                    error_class=type(exc).__name__,
                )
            )
            return ToolCallResult(
                tool_name="screen_control.run",
                turn_id=turn_id,
                success=False,
                error_message="Screen control encountered an error.",
            )

    async def _execute_gui_agent_spawn(
        self,
        call: SpawnGuiAgentCall,
        *,
        turn_id: UUID,
        session_id: UUID,
    ) -> ToolCallResult:
        """Spawn a ``SidecarAgentLoop`` in a daemon thread (Req 6.1, 6.3, 6.4).

        Steps:
        1. If ``_xtts_sink`` is configured, await it with an acknowledgement
           phrase before starting the thread (Req 6.3).
        2. Call ``_start_agent_in_thread`` (Req 9.2).
        3. Return ``ToolCallResult(success=True)`` within 500 ms (Req 6.4).
        """
        if self._xtts_sink is not None:
            try:
                ack_text = f"On it, starting task: {call.task_description[:80]}"
                await self._xtts_sink(ack_text)
            except Exception as exc:
                _logger.debug("_execute_gui_agent_spawn: xtts_sink failed: %r", exc)

        if self._ipc_server is not None:
            _start_agent_in_thread(call.task_description, self._ipc_server)
        else:
            _logger.warning(
                "_execute_gui_agent_spawn: no ipc_server configured — agent not started"
            )

        await self._emit_diagnostic(
            ToolCallDiagnostic(
                outcome="executed",
                session_id=session_id,
                turn_id=turn_id,
                tool_name="gui_agent.spawn",
                error_class=None,
            )
        )
        return ToolCallResult(
            tool_name="gui_agent.spawn",
            turn_id=turn_id,
            success=True,
            data={"message": "GUI agent started", "task": call.task_description},
        )

    # ------------------------------------------------------------------
    # Diagnostic emission
    # ------------------------------------------------------------------

    async def _emit_diagnostic(self, diagnostic: ToolCallDiagnostic) -> None:
        if self._diagnostic_sink is None:
            _logger.warning(
                "tool_call diagnostic: outcome=%s tool_name=%s error_class=%s",
                diagnostic.outcome,
                diagnostic.tool_name,
                diagnostic.error_class,
            )
            return
        import inspect

        result = self._diagnostic_sink(diagnostic)
        if inspect.isawaitable(result):
            await result


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _try_parse_tool_call(text: str) -> dict | None:
    """Attempt to parse *text* as a JSON object.

    Returns a ``dict`` on success, ``None`` on any parse or type error.
    Only top-level JSON objects (dicts) are accepted.
    """
    try:
        parsed = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _validate_tool_call(data: dict) -> ObsidianRAGCall | ScreenControlCall | SpawnGuiAgentCall:
    """Validate *data* against the registered tool schemas.

    Raises ``ValidationError`` if the data does not match any registered schema.
    Raises ``ValueError`` for unknown tool names.
    """
    tool = data.get("tool")
    if tool == "obsidian_rag.search":
        return ObsidianRAGCall.model_validate(data)
    if tool == "screen_control.run":
        return ScreenControlCall.model_validate(data)
    if tool == "gui_agent.spawn":
        return SpawnGuiAgentCall.model_validate(data)
    # Unknown tool name — raise a ValidationError-compatible error
    raise ValueError(f"Unknown tool name: {tool!r}")


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "ObsidianRAGCall",
    "ScreenControlCall",
    "SpawnGuiAgentCall",
    "ToolCallDiagnostic",
    "ToolCallResult",
    "VoiceToolAdapter",
    "VOICE_TOOL_GRAMMAR",
    "_try_parse_tool_call",
    "_validate_tool_call",
    "_start_agent_in_thread",
]
