"""Tests for VoiceToolAdapter — schema validation and turn-correlated execution.

Validates:
- Requirements 6.3: ObsidianRAG schema validation before invocation
- Requirements 6.4: ScreenControl schema validation before invocation
- Requirements 6.5: Structured tool result correlated to originating turn
- Requirements 6.6: Schema rejection emits stage="tool_call" diagnostic, makes zero
  target calls
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from core.voice.tools import (
    ObsidianRAGCall,
    ScreenControlCall,
    ToolCallDiagnostic,
    ToolCallResult,
    VoiceToolAdapter,
    _try_parse_tool_call,
    _validate_tool_call,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(
    haki_brain=None,
    screen_agent=None,
    diagnostic_sink=None,
) -> VoiceToolAdapter:
    return VoiceToolAdapter(
        haki_brain=haki_brain,
        screen_agent=screen_agent,
        diagnostic_sink=diagnostic_sink,
    )


def _rag_payload(**overrides) -> str:
    data = {
        "tool": "obsidian_rag.search",
        "schema_version": 1,
        "query": "test query",
        "limit": 3,
    }
    data.update(overrides)
    return json.dumps(data)


def _screen_payload(**overrides) -> str:
    data = {
        "tool": "screen_control.run",
        "schema_version": 1,
        "goal": "open terminal",
        "confirmation_context": "voice",
    }
    data.update(overrides)
    return json.dumps(data)


# ---------------------------------------------------------------------------
# _try_parse_tool_call
# ---------------------------------------------------------------------------

class TestTryParseToolCall:
    def test_valid_json_object_returns_dict(self):
        result = _try_parse_tool_call('{"tool": "foo"}')
        assert result == {"tool": "foo"}

    def test_invalid_json_returns_none(self):
        assert _try_parse_tool_call("not json") is None

    def test_json_array_returns_none(self):
        assert _try_parse_tool_call("[1, 2, 3]") is None

    def test_json_string_returns_none(self):
        assert _try_parse_tool_call('"hello"') is None

    def test_empty_string_returns_none(self):
        assert _try_parse_tool_call("") is None

    def test_whitespace_stripped(self):
        result = _try_parse_tool_call('  {"tool": "x"}  ')
        assert result == {"tool": "x"}


# ---------------------------------------------------------------------------
# Schema: extra="forbid" enforcement
# ---------------------------------------------------------------------------

class TestExtraForbid:
    """Requirement 6.3 / 6.4: extra fields must raise ValidationError."""

    def test_obsidian_rag_extra_field_raises(self):
        with pytest.raises(ValidationError):
            ObsidianRAGCall(
                tool="obsidian_rag.search",
                schema_version=1,
                query="hello",
                limit=3,
                extra_field="bad",  # type: ignore[call-arg]
            )

    def test_screen_control_extra_field_raises(self):
        with pytest.raises(ValidationError):
            ScreenControlCall(
                tool="screen_control.run",
                schema_version=1,
                goal="do something",
                confirmation_context="voice",
                extra="bad",  # type: ignore[call-arg]
            )

    def test_obsidian_rag_extra_field_via_model_validate(self):
        with pytest.raises(ValidationError):
            ObsidianRAGCall.model_validate({
                "tool": "obsidian_rag.search",
                "schema_version": 1,
                "query": "test",
                "limit": 3,
                "unexpected": True,
            })

    def test_screen_control_extra_field_via_model_validate(self):
        with pytest.raises(ValidationError):
            ScreenControlCall.model_validate({
                "tool": "screen_control.run",
                "schema_version": 1,
                "goal": "open app",
                "confirmation_context": "voice",
                "extra_key": "oops",
            })


# ---------------------------------------------------------------------------
# Schema: field constraints
# ---------------------------------------------------------------------------

class TestObsidianRAGConstraints:
    def test_valid_minimal(self):
        call = ObsidianRAGCall(tool="obsidian_rag.search", query="hello")
        assert call.limit == 3  # default

    def test_query_min_length(self):
        with pytest.raises(ValidationError):
            ObsidianRAGCall(tool="obsidian_rag.search", query="   ")  # whitespace stripped → empty

    def test_query_max_length(self):
        with pytest.raises(ValidationError):
            ObsidianRAGCall(tool="obsidian_rag.search", query="x" * 2001)

    def test_limit_min(self):
        with pytest.raises(ValidationError):
            ObsidianRAGCall(tool="obsidian_rag.search", query="q", limit=0)

    def test_limit_max(self):
        with pytest.raises(ValidationError):
            ObsidianRAGCall(tool="obsidian_rag.search", query="q", limit=6)

    def test_limit_boundaries(self):
        for v in (1, 5):
            call = ObsidianRAGCall(tool="obsidian_rag.search", query="q", limit=v)
            assert call.limit == v


class TestScreenControlConstraints:
    def test_valid(self):
        call = ScreenControlCall(
            tool="screen_control.run",
            goal="open safari",
            confirmation_context="voice",
        )
        assert call.confirmation_context == "voice"

    def test_confirmation_context_must_be_voice(self):
        with pytest.raises(ValidationError):
            ScreenControlCall(
                tool="screen_control.run",
                goal="open safari",
                confirmation_context="auto",  # type: ignore[arg-type]
            )

    def test_goal_min_length(self):
        with pytest.raises(ValidationError):
            ScreenControlCall(
                tool="screen_control.run",
                goal="   ",  # whitespace stripped → empty
                confirmation_context="voice",
            )

    def test_goal_max_length(self):
        with pytest.raises(ValidationError):
            ScreenControlCall(
                tool="screen_control.run",
                goal="x" * 1001,
                confirmation_context="voice",
            )


# ---------------------------------------------------------------------------
# Invalid data → zero target calls (Requirement 6.6)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSchemaRejectionZeroCalls:
    async def test_malformed_json_makes_no_calls(self):
        brain = AsyncMock()
        agent = AsyncMock()
        diagnostics: list[ToolCallDiagnostic] = []
        adapter = _make_adapter(
            haki_brain=brain,
            screen_agent=agent,
            diagnostic_sink=diagnostics.append,
        )
        result = await adapter.execute_tool_call(
            "not valid json",
            turn_id=uuid4(),
            session_id=uuid4(),
        )
        assert result is None
        brain.search.assert_not_called()
        agent.run.assert_not_called()
        assert diagnostics[0].outcome == "rejected"
        assert diagnostics[0].stage == "tool_call"

    async def test_unknown_tool_name_makes_no_calls(self):
        brain = AsyncMock()
        agent = AsyncMock()
        diagnostics: list[ToolCallDiagnostic] = []
        adapter = _make_adapter(
            haki_brain=brain,
            screen_agent=agent,
            diagnostic_sink=diagnostics.append,
        )
        payload = json.dumps({"tool": "unknown.tool", "schema_version": 1})
        result = await adapter.execute_tool_call(payload, turn_id=uuid4(), session_id=uuid4())
        assert result is None
        brain.search.assert_not_called()
        agent.run.assert_not_called()
        assert diagnostics[0].outcome == "rejected"

    async def test_extra_fields_make_no_calls(self):
        brain = AsyncMock()
        agent = AsyncMock()
        diagnostics: list[ToolCallDiagnostic] = []
        adapter = _make_adapter(
            haki_brain=brain,
            screen_agent=agent,
            diagnostic_sink=diagnostics.append,
        )
        payload = json.dumps({
            "tool": "obsidian_rag.search",
            "schema_version": 1,
            "query": "test",
            "limit": 3,
            "extra": "bad",
        })
        result = await adapter.execute_tool_call(payload, turn_id=uuid4(), session_id=uuid4())
        assert result is None
        brain.search.assert_not_called()
        assert diagnostics[0].stage == "tool_call"
        assert diagnostics[0].outcome == "rejected"

    async def test_out_of_bound_limit_makes_no_calls(self):
        brain = AsyncMock()
        diagnostics: list[ToolCallDiagnostic] = []
        adapter = _make_adapter(
            haki_brain=brain,
            diagnostic_sink=diagnostics.append,
        )
        payload = json.dumps({
            "tool": "obsidian_rag.search",
            "schema_version": 1,
            "query": "hello",
            "limit": 10,  # out of bounds
        })
        result = await adapter.execute_tool_call(payload, turn_id=uuid4(), session_id=uuid4())
        assert result is None
        brain.search.assert_not_called()
        assert diagnostics[0].outcome == "rejected"

    async def test_screen_wrong_confirmation_context_makes_no_calls(self):
        agent = AsyncMock()
        diagnostics: list[ToolCallDiagnostic] = []
        adapter = _make_adapter(
            screen_agent=agent,
            diagnostic_sink=diagnostics.append,
        )
        payload = json.dumps({
            "tool": "screen_control.run",
            "schema_version": 1,
            "goal": "open terminal",
            "confirmation_context": "auto",  # not "voice"
        })
        result = await adapter.execute_tool_call(payload, turn_id=uuid4(), session_id=uuid4())
        assert result is None
        agent.run.assert_not_called()
        assert diagnostics[0].outcome == "rejected"


# ---------------------------------------------------------------------------
# Valid results are turn-correlated (Requirement 6.5)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestTurnCorrelation:
    async def test_obsidian_rag_result_has_correct_turn_id(self):
        brain = AsyncMock()
        brain.search.return_value = [
            {"title": "T1", "content": "C1", "source": "s1", "distance": 0.1},
            {"title": "T2", "content": "C2", "source": "s2", "distance": 0.2},
        ]
        turn_id = UUID("12345678-1234-5678-1234-567812345678")
        adapter = _make_adapter(haki_brain=brain)
        result = await adapter.execute_tool_call(
            _rag_payload(limit=2),
            turn_id=turn_id,
            session_id=uuid4(),
        )
        assert result is not None
        assert result.turn_id == turn_id
        assert result.tool_name == "obsidian_rag.search"
        assert result.success is True

    async def test_screen_control_result_has_correct_turn_id(self):
        agent = AsyncMock()
        agent_result = MagicMock()
        agent_result.success = True
        agent_result.message = "Done"
        agent_result.steps = []
        agent_result.goal = "open terminal"
        agent.run.return_value = agent_result

        turn_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        adapter = _make_adapter(screen_agent=agent)
        result = await adapter.execute_tool_call(
            _screen_payload(),
            turn_id=turn_id,
            session_id=uuid4(),
        )
        assert result is not None
        assert result.turn_id == turn_id
        assert result.tool_name == "screen_control.run"


# ---------------------------------------------------------------------------
# Voice-originated screen requests never auto-confirm (Requirement 6.4)
# ---------------------------------------------------------------------------

class TestVoiceConfirmationContext:
    def test_screen_control_schema_requires_voice_context(self):
        """confirmation_context must be 'voice'; any other value is rejected."""
        call = ScreenControlCall.model_validate({
            "tool": "screen_control.run",
            "schema_version": 1,
            "goal": "open finder",
            "confirmation_context": "voice",
        })
        assert call.confirmation_context == "voice"

    def test_screen_control_schema_rejects_non_voice(self):
        for bad_ctx in ("auto", "manual", "", "VOICE", "Voice"):
            with pytest.raises(ValidationError):
                ScreenControlCall.model_validate({
                    "tool": "screen_control.run",
                    "schema_version": 1,
                    "goal": "open finder",
                    "confirmation_context": bad_ctx,
                })

    @pytest.mark.asyncio
    async def test_adapter_passes_only_goal_to_screen_agent(self):
        """Adapter must call ScreenAgent.run(goal) only — no extra args."""
        agent = AsyncMock()
        agent_result = MagicMock()
        agent_result.success = True
        agent_result.message = "ok"
        agent_result.steps = []
        agent_result.goal = "open safari"
        agent.run.return_value = agent_result

        adapter = _make_adapter(screen_agent=agent)
        await adapter.execute_tool_call(
            _screen_payload(goal="open safari"),
            turn_id=uuid4(),
            session_id=uuid4(),
        )
        agent.run.assert_called_once_with("open safari")


# ---------------------------------------------------------------------------
# Bounded results
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBoundedResults:
    async def test_obsidian_rag_respects_limit_cap(self):
        """HAKIBrain returns more results than limit; adapter caps to limit."""
        brain = AsyncMock()
        brain.search.return_value = [
            {"title": f"T{i}", "content": f"C{i}", "source": "", "distance": float(i)}
            for i in range(10)
        ]
        adapter = _make_adapter(haki_brain=brain)
        result = await adapter.execute_tool_call(
            _rag_payload(limit=3),
            turn_id=uuid4(),
            session_id=uuid4(),
        )
        assert result is not None
        assert result.success is True
        assert len(result.data) == 3

    async def test_obsidian_rag_passes_limit_to_search(self):
        """The adapter passes the limit to HAKIBrain.search as k=limit."""
        brain = AsyncMock()
        brain.search.return_value = []
        adapter = _make_adapter(haki_brain=brain)
        await adapter.execute_tool_call(
            _rag_payload(limit=2),
            turn_id=uuid4(),
            session_id=uuid4(),
        )
        brain.search.assert_called_once_with("test query", k=2)

    async def test_screen_control_message_truncated_to_2000(self):
        agent = AsyncMock()
        agent_result = MagicMock()
        agent_result.success = True
        agent_result.message = "x" * 5000
        agent_result.steps = []
        agent_result.goal = "open terminal"
        agent.run.return_value = agent_result

        adapter = _make_adapter(screen_agent=agent)
        result = await adapter.execute_tool_call(
            _screen_payload(),
            turn_id=uuid4(),
            session_id=uuid4(),
        )
        assert result is not None
        assert result.success is True
        assert len(result.data["message"]) == 2000


# ---------------------------------------------------------------------------
# Diagnostic emission
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDiagnosticEmission:
    async def test_diagnostic_stage_is_tool_call_on_rejection(self):
        diagnostics: list[ToolCallDiagnostic] = []
        adapter = _make_adapter(diagnostic_sink=diagnostics.append)
        await adapter.execute_tool_call("bad json", turn_id=uuid4(), session_id=uuid4())
        assert len(diagnostics) == 1
        assert diagnostics[0].stage == "tool_call"
        assert diagnostics[0].outcome == "rejected"

    async def test_diagnostic_does_not_include_query_content(self):
        """Diagnostics must only include tool_name, never argument values."""
        diagnostics: list[ToolCallDiagnostic] = []
        brain = AsyncMock()
        brain.search.return_value = []
        adapter = _make_adapter(haki_brain=brain, diagnostic_sink=diagnostics.append)
        await adapter.execute_tool_call(
            _rag_payload(query="my private query"),
            turn_id=uuid4(),
            session_id=uuid4(),
        )
        assert len(diagnostics) == 1
        diag = diagnostics[0]
        # tool_name contains only the tool discriminator, never user content
        assert diag.tool_name == "obsidian_rag.search"
        assert "private query" not in (diag.tool_name or "")
        assert "private query" not in (diag.error_class or "")

    async def test_executed_diagnostic_on_success(self):
        diagnostics: list[ToolCallDiagnostic] = []
        brain = AsyncMock()
        brain.search.return_value = [{"title": "T", "content": "C", "source": "", "distance": 0.0}]
        adapter = _make_adapter(haki_brain=brain, diagnostic_sink=diagnostics.append)
        result = await adapter.execute_tool_call(
            _rag_payload(),
            turn_id=uuid4(),
            session_id=uuid4(),
        )
        assert result is not None
        assert result.success is True
        assert diagnostics[0].outcome == "executed"

    async def test_async_diagnostic_sink(self):
        """Diagnostic sink may be an async callable."""
        received: list[ToolCallDiagnostic] = []

        async def async_sink(d: ToolCallDiagnostic) -> None:
            received.append(d)

        adapter = _make_adapter(diagnostic_sink=async_sink)
        await adapter.execute_tool_call("not json", turn_id=uuid4(), session_id=uuid4())
        assert len(received) == 1
        assert received[0].stage == "tool_call"
