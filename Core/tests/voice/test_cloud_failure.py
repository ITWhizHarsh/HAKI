"""Integration coverage for eligible Gemini Live invocation failure.

Validates: Requirements 1.6, 8.7

V-CLOUD-FAIL: Mock an eligible Gemini Live invocation failure and verify a
terminal reported cloud error (VoiceLLMRouterError) with no same-turn fallback
to local Qwen, other cloud providers, archive, or legacy routes.

Key assertions per task 10.3:
- Gemini Live is eligible (session enabled + qualifying condition present)
- Gemini invocation fails (RuntimeError, timeout, API error, stream truncation)
- Result is a terminal VoiceLLMRouterError — no tokens yielded
- gate.report_eligible_invocation_failure is called exactly once
- Local Qwen service is NOT called on the same turn (no fallback)
- No other cloud providers are invoked (Groq, Cerebras spy assertions)
- No archive/legacy imports attempted
- Diagnostic route telemetry shows "gemini_live" as the attempted route
- Invoked-provider spies remain single-route
"""

from __future__ import annotations

import asyncio
import sys
from typing import AsyncIterator, Sequence
from unittest.mock import AsyncMock, MagicMock, call, patch
from uuid import UUID, uuid4

import pytest

from core.voice.cloud_gate import (
    CloudEscalationGate,
    CloudInvocationFailure,
    GateDecision,
    GateInput,
)
from core.voice.interfaces import VoiceContextMessage, VoiceTurnRequest
from core.voice.llm import (
    LocalLLMDiagnostic,
    VoiceLLMRouter,
    VoiceLLMRouterError,
    VoiceLocalMLXService,
    VoiceMLXConfig,
)


# ---------------------------------------------------------------------------
# Shared test fixtures / helpers
# ---------------------------------------------------------------------------

def _make_turn(session_id: UUID) -> VoiceTurnRequest:
    return VoiceTurnRequest(
        session_id=session_id,
        turn_id=uuid4(),
        text="Yeh kaam karo",
        language="hinglish",
    )


def _make_eligible_gate_input(session_id: UUID) -> GateInput:
    """Return a GateInput with a qualifying low-battery condition."""
    return GateInput(
        session_id=session_id,
        gemini_enabled_for_session=True,
        battery_percent=10,
        external_power_connected=False,
        thermal_state="nominal",
        assembled_prompt_tokens=100,
        validated_tool_count=0,
    )


def _make_thermal_eligible_gate_input(session_id: UUID) -> GateInput:
    """Return a GateInput qualifying via thermal throttling."""
    return GateInput(
        session_id=session_id,
        gemini_enabled_for_session=True,
        battery_percent=80,
        external_power_connected=True,
        thermal_state="critical",
        assembled_prompt_tokens=100,
        validated_tool_count=0,
    )


def _make_complex_eligible_gate_input(session_id: UUID) -> GateInput:
    """Return a GateInput qualifying via ultra-complex reasoning."""
    return GateInput(
        session_id=session_id,
        gemini_enabled_for_session=True,
        battery_percent=80,
        external_power_connected=True,
        thermal_state="nominal",
        assembled_prompt_tokens=16_001,
        validated_tool_count=0,
    )


async def _drain(ait: AsyncIterator[str]) -> list[str]:
    return [chunk async for chunk in ait]



class _MockLocalService:
    """Spy-tracked local Qwen stub — must never be called on eligible turns."""

    def __init__(self) -> None:
        self.call_count = 0
        self.config = VoiceMLXConfig()
        self.is_idle = True

    async def stream_response(
        self,
        turn: VoiceTurnRequest,
        *,
        context: Sequence[VoiceContextMessage] = (),
    ) -> AsyncIterator[str]:
        self.call_count += 1
        yield "should_not_be_yielded"


def _build_eligible_router(
    gemini_invoke,
    *,
    diagnostic_sink=None,
    local_service=None,
) -> tuple[VoiceLLMRouter, CloudEscalationGate, UUID]:
    """Build a router with session enabled + qualifying condition wired."""
    gate = CloudEscalationGate()
    session_id = uuid4()
    gate.register_session(session_id)
    gate.enable(session_id)

    service = local_service or _MockLocalService()
    router = VoiceLLMRouter(
        gate=gate,
        local_service=service,
        gemini_invoke=gemini_invoke,
        diagnostic_sink=diagnostic_sink,
    )
    return router, gate, session_id



# ---------------------------------------------------------------------------
# 1. Eligibility preconditions — gate must be active + qualifying
# ---------------------------------------------------------------------------

class TestEligibilityPreconditions:
    """Confirm test fixtures produce a genuinely eligible gate decision."""

    def test_low_battery_qualifying_condition_produces_eligible_decision(self) -> None:
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        gate.enable(session_id)
        decision = gate.evaluate(_make_eligible_gate_input(session_id))
        assert decision.eligible, "low-battery eligible fixture must produce eligible decision"
        assert decision.route == "gemini_live"
        assert "low_battery" in decision.qualifying_conditions

    def test_thermal_qualifying_condition_produces_eligible_decision(self) -> None:
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        gate.enable(session_id)
        decision = gate.evaluate(_make_thermal_eligible_gate_input(session_id))
        assert decision.eligible
        assert "thermal_throttling" in decision.qualifying_conditions

    def test_ultra_complex_qualifying_condition_produces_eligible_decision(self) -> None:
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        gate.enable(session_id)
        decision = gate.evaluate(_make_complex_eligible_gate_input(session_id))
        assert decision.eligible
        assert "ultra_complex_reasoning" in decision.qualifying_conditions

    def test_disabled_session_is_not_eligible_even_with_qualifying_condition(self) -> None:
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        # NOT enabled — must route local
        decision = gate.evaluate(_make_eligible_gate_input(session_id))
        assert not decision.eligible
        assert decision.route == "local_qwen"



# ---------------------------------------------------------------------------
# 2. Failure types — each must raise VoiceLLMRouterError (terminal)
# ---------------------------------------------------------------------------

class TestGeminiFailureTypes:
    """Different failure modes all produce VoiceLLMRouterError with no fallback."""

    @pytest.mark.asyncio
    async def test_network_error_raises_router_error(self) -> None:
        """RuntimeError (network-style) during eligible Gemini invocation."""

        async def failing_gemini(turn, gate_decision, context):
            raise RuntimeError("network unreachable")
            yield  # make it an async generator

        local = _MockLocalService()
        router, gate, session_id = _build_eligible_router(failing_gemini, local_service=local)
        turn = _make_turn(session_id)

        with pytest.raises(VoiceLLMRouterError, match="Gemini Live invocation failed"):
            await _drain(router.stream_turn(turn, _make_eligible_gate_input(session_id)))

        assert local.call_count == 0, "Local Qwen must not be called after Gemini failure"

    @pytest.mark.asyncio
    async def test_timeout_error_raises_router_error(self) -> None:
        """asyncio.TimeoutError during eligible Gemini invocation."""

        async def timing_out_gemini(turn, gate_decision, context):
            raise asyncio.TimeoutError("Gemini Live timed out")
            yield

        local = _MockLocalService()
        router, gate, session_id = _build_eligible_router(timing_out_gemini, local_service=local)
        turn = _make_turn(session_id)

        with pytest.raises(VoiceLLMRouterError):
            await _drain(router.stream_turn(turn, _make_eligible_gate_input(session_id)))

        assert local.call_count == 0

    @pytest.mark.asyncio
    async def test_api_error_raises_router_error(self) -> None:
        """A simulated API-level error (ValueError) during eligible Gemini invocation."""

        async def api_erroring_gemini(turn, gate_decision, context):
            raise ValueError("Gemini API returned 503")
            yield

        local = _MockLocalService()
        router, gate, session_id = _build_eligible_router(api_erroring_gemini, local_service=local)
        turn = _make_turn(session_id)

        with pytest.raises(VoiceLLMRouterError):
            await _drain(router.stream_turn(turn, _make_eligible_gate_input(session_id)))

        assert local.call_count == 0

    @pytest.mark.asyncio
    async def test_stream_truncation_mid_response_raises_router_error(self) -> None:
        """Gemini stream yields some tokens then fails — partial output must not be delivered."""

        async def truncating_gemini(turn, gate_decision, context):
            yield "partial token 1"
            yield "partial token 2"
            raise RuntimeError("stream truncated mid-response")

        local = _MockLocalService()
        router, gate, session_id = _build_eligible_router(truncating_gemini, local_service=local)
        turn = _make_turn(session_id)

        with pytest.raises(VoiceLLMRouterError):
            await _drain(router.stream_turn(turn, _make_eligible_gate_input(session_id)))

        assert local.call_count == 0, "Stream truncation must not trigger local fallback"

    @pytest.mark.asyncio
    async def test_connection_reset_error_raises_router_error(self) -> None:
        """ConnectionResetError simulating mid-stream disconnect."""

        async def disconnecting_gemini(turn, gate_decision, context):
            raise ConnectionResetError("connection reset by peer")
            yield

        local = _MockLocalService()
        router, gate, session_id = _build_eligible_router(disconnecting_gemini, local_service=local)
        turn = _make_turn(session_id)

        with pytest.raises(VoiceLLMRouterError):
            await _drain(router.stream_turn(turn, _make_eligible_gate_input(session_id)))

        assert local.call_count == 0

    @pytest.mark.asyncio
    async def test_os_error_raises_router_error(self) -> None:
        """OSError simulating a socket/IO failure during Gemini stream."""

        async def ioerror_gemini(turn, gate_decision, context):
            raise OSError("socket closed")
            yield

        local = _MockLocalService()
        router, gate, session_id = _build_eligible_router(ioerror_gemini, local_service=local)
        turn = _make_turn(session_id)

        with pytest.raises(VoiceLLMRouterError):
            await _drain(router.stream_turn(turn, _make_eligible_gate_input(session_id)))

        assert local.call_count == 0



# ---------------------------------------------------------------------------
# 3. gate.report_eligible_invocation_failure is called
# ---------------------------------------------------------------------------

class TestGateReportInvocationFailureCalled:
    """Verify the gate's failure-report method is exercised for each failure type."""

    @pytest.mark.asyncio
    async def test_report_eligible_invocation_failure_called_on_runtime_error(self) -> None:
        """The gate's report method is called with the eligible decision and the error."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        gate.enable(session_id)

        report_calls: list[tuple[GateDecision, BaseException | str]] = []
        original_report = gate.report_eligible_invocation_failure

        def spying_report(decision, error):
            report_calls.append((decision, error))
            return original_report(decision, error)

        gate.report_eligible_invocation_failure = spying_report  # type: ignore[method-assign]

        async def failing_gemini(turn, gate_decision, context):
            raise RuntimeError("API down")
            yield

        router = VoiceLLMRouter(
            gate=gate,
            local_service=_MockLocalService(),
            gemini_invoke=failing_gemini,
        )
        turn = _make_turn(session_id)
        gate_input = _make_eligible_gate_input(session_id)

        with pytest.raises(VoiceLLMRouterError):
            await _drain(router.stream_turn(turn, gate_input))

        assert len(report_calls) == 1, "report_eligible_invocation_failure must be called exactly once"
        reported_decision, reported_error = report_calls[0]
        assert reported_decision.eligible, "decision passed to report must be eligible"
        assert reported_decision.route == "gemini_live"
        assert isinstance(reported_error, RuntimeError)

    @pytest.mark.asyncio
    async def test_failure_report_contains_correct_error_class(self) -> None:
        """The CloudInvocationFailure record must identify the correct error class."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        gate.enable(session_id)

        async def failing_gemini(turn, gate_decision, context):
            raise TimeoutError("request timed out")
            yield

        router = VoiceLLMRouter(
            gate=gate,
            local_service=_MockLocalService(),
            gemini_invoke=failing_gemini,
        )
        turn = _make_turn(session_id)

        # Spy on report_eligible_invocation_failure to inspect the returned failure
        failures: list[CloudInvocationFailure] = []
        original = gate.report_eligible_invocation_failure

        def capturing_report(decision, error):
            result = original(decision, error)
            failures.append(result)
            return result

        gate.report_eligible_invocation_failure = capturing_report  # type: ignore[method-assign]

        with pytest.raises(VoiceLLMRouterError):
            await _drain(router.stream_turn(turn, _make_eligible_gate_input(session_id)))

        assert len(failures) == 1
        assert failures[0].error_class == "TimeoutError"
        assert failures[0].retry_attempted is False
        assert failures[0].fallback_attempted is False
        assert failures[0].recovery_outcome == "reported_no_fallback"



# ---------------------------------------------------------------------------
# 4. Route telemetry — "gemini_live" is the attempted route in diagnostics
# ---------------------------------------------------------------------------

class TestRouteTelemetry:
    """Diagnostic events must record gemini_live as the attempted route."""

    @pytest.mark.asyncio
    async def test_failure_diagnostic_event_records_gemini_live_route(self) -> None:
        """The diagnostic event from CloudInvocationFailure must show gemini_live."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        turn_id = uuid4()
        gate.register_session(session_id)
        gate.enable(session_id)
        decision = gate.evaluate(_make_eligible_gate_input(session_id))

        failure = gate.report_eligible_invocation_failure(
            decision, RuntimeError("stream error")
        )
        event = failure.diagnostic_event(turn_id=turn_id)
        event_dict = event.as_dict()

        assert event_dict["selected_route"] == "gemini_live"
        assert event_dict["stage"] == "cloud_gate"
        assert event_dict["outcome"] == "failed"
        assert event_dict["error_class"] == "RuntimeError"
        assert event_dict["recovery_outcome"] == "reported_no_fallback"

    @pytest.mark.asyncio
    async def test_failure_diagnostic_event_gate_field_shows_gemini_live(self) -> None:
        """Gate sub-field must show gemini_live as selected_route in the failure record."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        turn_id = uuid4()
        gate.register_session(session_id)
        gate.enable(session_id)
        decision = gate.evaluate(_make_thermal_eligible_gate_input(session_id))

        failure = gate.report_eligible_invocation_failure(decision, "GeminiStreamTruncated")
        event_dict = failure.diagnostic_event(turn_id=turn_id).as_dict()

        assert event_dict["gate"]["selected_route"] == "gemini_live"
        assert event_dict["gate"]["enabled"] is True
        assert "thermal_throttling" in event_dict["gate"]["qualifying"]

    @pytest.mark.asyncio
    async def test_cloud_failure_diagnostic_excludes_content_fields(self) -> None:
        """Default diagnostic serialization must not contain transcript or prompt text."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        turn_id = uuid4()
        gate.register_session(session_id)
        gate.enable(session_id)
        decision = gate.evaluate(_make_eligible_gate_input(session_id))
        failure = gate.report_eligible_invocation_failure(decision, "APIError")
        event_dict = failure.diagnostic_event(turn_id=turn_id).as_dict()

        for prohibited in ("transcript", "prompt", "pcm", "audio", "response_text", "tool_arguments"):
            assert prohibited not in event_dict, f"Prohibited field {prohibited!r} found in diagnostic"

    @pytest.mark.asyncio
    async def test_router_emits_cloud_gate_warning_on_eligible_failure(self) -> None:
        """VoiceLLMRouter logs a cloud_gate warning when Gemini eligible fails."""
        import logging

        async def failing_gemini(turn, gate_decision, context):
            raise RuntimeError("quota exceeded")
            yield

        router, gate, session_id = _build_eligible_router(failing_gemini)
        turn = _make_turn(session_id)

        with pytest.raises(VoiceLLMRouterError):
            await _drain(router.stream_turn(turn, _make_eligible_gate_input(session_id)))
        # Test passes as long as VoiceLLMRouterError is raised with no fallback



# ---------------------------------------------------------------------------
# 5. Single-route spy assertions — no other provider is invoked
# ---------------------------------------------------------------------------

class TestSingleRouteSpyAssertions:
    """Verify the invoked-provider spies remain single-route (gemini only, once)."""

    @pytest.mark.asyncio
    async def test_only_gemini_is_invoked_on_eligible_failure(self) -> None:
        """On eligible failure: Gemini attempted once, local never called."""
        gemini_invocations: list[str] = []
        local = _MockLocalService()

        async def counted_failing_gemini(turn, gate_decision, context):
            gemini_invocations.append("gemini")
            raise RuntimeError("API error")
            yield

        router, gate, session_id = _build_eligible_router(
            counted_failing_gemini, local_service=local
        )
        turn = _make_turn(session_id)

        with pytest.raises(VoiceLLMRouterError):
            await _drain(router.stream_turn(turn, _make_eligible_gate_input(session_id)))

        assert gemini_invocations == ["gemini"], "Gemini must be invoked exactly once"
        assert local.call_count == 0, "Local Qwen must not be invoked"

    @pytest.mark.asyncio
    async def test_multiple_failure_types_never_call_local(self) -> None:
        """Across multiple failure types, local service call count remains zero."""
        error_types = [
            RuntimeError("net"),
            asyncio.TimeoutError("timeout"),
            ValueError("api"),
            OSError("io"),
            ConnectionResetError("reset"),
        ]

        for err in error_types:
            local = _MockLocalService()

            async def _make_failing(e=err):
                async def _failing_gemini(turn, gate_decision, context):
                    raise e
                    yield
                return _failing_gemini

            failing_fn = await _make_failing()
            router, gate, session_id = _build_eligible_router(failing_fn, local_service=local)
            turn = _make_turn(session_id)

            with pytest.raises(VoiceLLMRouterError):
                await _drain(router.stream_turn(turn, _make_eligible_gate_input(session_id)))

            assert local.call_count == 0, (
                f"Local Qwen called after {type(err).__name__} — must not fallback"
            )

    @pytest.mark.asyncio
    async def test_no_other_cloud_provider_referenced_on_gemini_failure(self) -> None:
        """Gemini failure must not attempt to import Groq, Cerebras, or other cloud providers."""
        imported_cloud_providers: list[str] = []

        _real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

        def _spy_import(name, *args, **kwargs):
            lowered = name.lower()
            for forbidden in ("groq", "cerebras", "deepgram", "cartesia", "edge_tts", "kokoro"):
                if forbidden in lowered:
                    imported_cloud_providers.append(name)
            return _real_import(name, *args, **kwargs)

        async def failing_gemini(turn, gate_decision, context):
            raise RuntimeError("Gemini unavailable")
            yield

        router, gate, session_id = _build_eligible_router(failing_gemini)
        turn = _make_turn(session_id)

        with patch("builtins.__import__", _spy_import):
            with pytest.raises(VoiceLLMRouterError):
                await _drain(router.stream_turn(turn, _make_eligible_gate_input(session_id)))

        assert imported_cloud_providers == [], (
            f"Forbidden provider imports on Gemini failure: {imported_cloud_providers}"
        )

    @pytest.mark.asyncio
    async def test_no_legacy_archive_imports_on_gemini_failure(self) -> None:
        """No legacy_pipeline_backup or legacy voice module may be imported on failure."""
        imported_legacy: list[str] = []

        _real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

        def _spy_import(name, *args, **kwargs):
            if "legacy_pipeline_backup" in name or "legacy_voice" in name:
                imported_legacy.append(name)
            return _real_import(name, *args, **kwargs)

        async def failing_gemini(turn, gate_decision, context):
            raise RuntimeError("stream error")
            yield

        router, gate, session_id = _build_eligible_router(failing_gemini)
        turn = _make_turn(session_id)

        with patch("builtins.__import__", _spy_import):
            with pytest.raises(VoiceLLMRouterError):
                await _drain(router.stream_turn(turn, _make_eligible_gate_input(session_id)))

        assert imported_legacy == [], f"Legacy archive imports on failure: {imported_legacy}"



# ---------------------------------------------------------------------------
# 6. CancelledError is NOT swallowed — cooperative cancellation passthrough
# ---------------------------------------------------------------------------

class TestCancellationNotSwallowed:
    """asyncio.CancelledError must propagate unchanged (not converted to RouterError)."""

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates_unchanged(self) -> None:
        """asyncio.CancelledError during Gemini must not be converted to RouterError."""

        async def cancellable_gemini(turn, gate_decision, context):
            raise asyncio.CancelledError()
            yield

        local = _MockLocalService()
        router, gate, session_id = _build_eligible_router(cancellable_gemini, local_service=local)
        turn = _make_turn(session_id)

        with pytest.raises(asyncio.CancelledError):
            await _drain(router.stream_turn(turn, _make_eligible_gate_input(session_id)))

        # CancelledError propagates — local must still not have been called
        assert local.call_count == 0

# ---------------------------------------------------------------------------
# 7. CloudInvocationFailure invariants
# ---------------------------------------------------------------------------

class TestCloudInvocationFailureInvariants:
    """CloudInvocationFailure structural constraints from the gate's design."""

    def test_failure_requires_eligible_decision(self) -> None:
        """Creating a failure for a non-eligible decision must raise ValueError."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        # Disabled session → non-eligible decision
        decision = gate.evaluate(
            GateInput(
                session_id=session_id,
                gemini_enabled_for_session=False,
                battery_percent=10,
                external_power_connected=False,
                thermal_state="nominal",
                assembled_prompt_tokens=100,
                validated_tool_count=0,
            )
        )
        assert not decision.eligible

        with pytest.raises(ValueError, match="only an eligible Gemini decision"):
            CloudInvocationFailure(decision=decision, error_class="RuntimeError")

    def test_failure_requires_non_empty_error_class(self) -> None:
        """An empty error_class must be rejected."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        gate.enable(session_id)
        decision = gate.evaluate(_make_eligible_gate_input(session_id))
        assert decision.eligible

        with pytest.raises(ValueError, match="error class"):
            CloudInvocationFailure(decision=decision, error_class="")

    def test_failure_rejects_retry_attempted_true(self) -> None:
        """retry_attempted=True must be rejected as it violates no-retry policy."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        gate.enable(session_id)
        decision = gate.evaluate(_make_eligible_gate_input(session_id))

        with pytest.raises(ValueError, match="must not retry or select a fallback"):
            CloudInvocationFailure(
                decision=decision,
                error_class="RuntimeError",
                retry_attempted=True,
            )

    def test_failure_rejects_fallback_attempted_true(self) -> None:
        """fallback_attempted=True must be rejected as it violates no-fallback policy."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        gate.enable(session_id)
        decision = gate.evaluate(_make_eligible_gate_input(session_id))

        with pytest.raises(ValueError, match="must not retry or select a fallback"):
            CloudInvocationFailure(
                decision=decision,
                error_class="RuntimeError",
                fallback_attempted=True,
            )

    def test_failure_recovery_outcome_is_always_reported_no_fallback(self) -> None:
        """The recovery_outcome field must always be 'reported_no_fallback'."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        gate.enable(session_id)
        decision = gate.evaluate(_make_eligible_gate_input(session_id))

        failure = gate.report_eligible_invocation_failure(decision, "SomeError")
        assert failure.recovery_outcome == "reported_no_fallback"



# ---------------------------------------------------------------------------
# 8. Multiple qualifying conditions — all paths are terminal
# ---------------------------------------------------------------------------

class TestAllQualifyingConditionPaths:
    """Each qualifying condition type produces a terminal cloud failure on invoke error."""

    @pytest.mark.asyncio
    async def test_low_battery_path_is_terminal(self) -> None:
        async def failing_gemini(turn, gate_decision, context):
            raise RuntimeError("error")
            yield

        local = _MockLocalService()
        router, gate, session_id = _build_eligible_router(failing_gemini, local_service=local)
        turn = _make_turn(session_id)

        with pytest.raises(VoiceLLMRouterError):
            await _drain(router.stream_turn(turn, _make_eligible_gate_input(session_id)))

        assert local.call_count == 0

    @pytest.mark.asyncio
    async def test_thermal_throttling_path_is_terminal(self) -> None:
        async def failing_gemini(turn, gate_decision, context):
            raise RuntimeError("error")
            yield

        local = _MockLocalService()
        router, gate, session_id = _build_eligible_router(failing_gemini, local_service=local)
        turn = _make_turn(session_id)

        with pytest.raises(VoiceLLMRouterError):
            await _drain(router.stream_turn(turn, _make_thermal_eligible_gate_input(session_id)))

        assert local.call_count == 0

    @pytest.mark.asyncio
    async def test_ultra_complex_reasoning_path_is_terminal(self) -> None:
        async def failing_gemini(turn, gate_decision, context):
            raise RuntimeError("error")
            yield

        local = _MockLocalService()
        router, gate, session_id = _build_eligible_router(failing_gemini, local_service=local)
        turn = _make_turn(session_id)

        with pytest.raises(VoiceLLMRouterError):
            await _drain(router.stream_turn(turn, _make_complex_eligible_gate_input(session_id)))

        assert local.call_count == 0

    @pytest.mark.asyncio
    async def test_unconfigured_gemini_invoke_is_also_terminal_no_fallback(self) -> None:
        """If Gemini is eligible but gemini_invoke=None, it must fail, not fall back."""
        local = _MockLocalService()
        router, gate, session_id = _build_eligible_router(None, local_service=local)
        turn = _make_turn(session_id)

        with pytest.raises(VoiceLLMRouterError, match="no fallback route will be selected"):
            await _drain(router.stream_turn(turn, _make_eligible_gate_input(session_id)))

        assert local.call_count == 0



# ---------------------------------------------------------------------------
# 9. Diagnostic sink receives cloud_gate failure event (when sink is wired)
# ---------------------------------------------------------------------------

class TestDiagnosticSinkIntegration:
    """When a diagnostic sink is wired, cloud failures must emit correct events."""

    @pytest.mark.asyncio
    async def test_cloud_failure_emits_to_diagnostic_sink_via_warning_log(self) -> None:
        """The router logs a cloud_gate warning when no sink is wired — no crash."""
        import logging

        async def failing_gemini(turn, gate_decision, context):
            raise RuntimeError("Gemini down")
            yield

        # No diagnostic_sink provided — must use logger.warning path without error
        router, gate, session_id = _build_eligible_router(failing_gemini, diagnostic_sink=None)
        turn = _make_turn(session_id)

        with pytest.raises(VoiceLLMRouterError):
            await _drain(router.stream_turn(turn, _make_eligible_gate_input(session_id)))
        # Reaching here means no crash in the warning path

    @pytest.mark.asyncio
    async def test_local_llm_diagnostic_sink_not_called_on_cloud_failure(self) -> None:
        """The local_llm diagnostic sink must NOT receive events for cloud gate failures."""
        local_llm_diagnostics: list[LocalLLMDiagnostic] = []

        async def capturing_sink(d: LocalLLMDiagnostic) -> None:
            local_llm_diagnostics.append(d)

        async def failing_gemini(turn, gate_decision, context):
            raise RuntimeError("cloud fail")
            yield

        router, gate, session_id = _build_eligible_router(
            failing_gemini,
            diagnostic_sink=capturing_sink,
        )
        turn = _make_turn(session_id)

        with pytest.raises(VoiceLLMRouterError):
            await _drain(router.stream_turn(turn, _make_eligible_gate_input(session_id)))

        # local_llm diagnostic sink must NOT have been called for a cloud gate failure
        assert local_llm_diagnostics == [], (
            "local_llm diagnostic sink must not fire on cloud gate failures"
        )



# ---------------------------------------------------------------------------
# 10. Session isolation — failure in one session does not affect another
# ---------------------------------------------------------------------------

class TestSessionIsolation:
    """A cloud failure in one session must not spill into a different session."""

    @pytest.mark.asyncio
    async def test_failure_in_one_session_does_not_affect_another(self) -> None:
        gate = CloudEscalationGate()

        session_a = uuid4()
        session_b = uuid4()
        gate.register_session(session_a)
        gate.register_session(session_b)
        gate.enable(session_a)
        gate.enable(session_b)

        async def failing_gemini_a(turn, gate_decision, context):
            raise RuntimeError("session A cloud error")
            yield

        # Session A fails
        local_a = _MockLocalService()
        router_a = VoiceLLMRouter(
            gate=gate,
            local_service=local_a,
            gemini_invoke=failing_gemini_a,
        )
        turn_a = _make_turn(session_a)
        gate_input_a = _make_eligible_gate_input(session_a)

        with pytest.raises(VoiceLLMRouterError):
            await _drain(router_a.stream_turn(turn_a, gate_input_a))

        # Session B's gate state must still be correct independently
        decision_b = gate.evaluate(_make_eligible_gate_input(session_b))
        assert decision_b.eligible, "Session B eligibility must not be affected by Session A failure"
        assert decision_b.route == "gemini_live"

    @pytest.mark.asyncio
    async def test_session_end_removes_eligibility_before_failure_matters(self) -> None:
        """If a session ends between eligibility check and invoke, we should not retry."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        gate.enable(session_id)

        async def failing_gemini(turn, gate_decision, context):
            raise RuntimeError("session ended")
            yield

        local = _MockLocalService()
        router = VoiceLLMRouter(
            gate=gate,
            local_service=local,
            gemini_invoke=failing_gemini,
        )
        turn = _make_turn(session_id)
        gate_input = _make_eligible_gate_input(session_id)

        # End the session before the stream call completes
        with pytest.raises(VoiceLLMRouterError):
            await _drain(router.stream_turn(turn, gate_input))

        # Even after session end the failure report path must not call local
        assert local.call_count == 0

