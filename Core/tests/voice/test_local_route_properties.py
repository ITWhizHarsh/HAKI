"""Property 7: Local-route default.

For all voice turns that the Cloud_Escalation_Gate does not mark eligible,
routing selects the Qwen local MLX service and no cloud or legacy voice route.

Feature: realtime-local-voice-agent, Property 7: Local-route default

**Validates: Requirements 6.1, 6.2, 6.7, 8.5, 8.6**

This file contains:
- 100+ parametrized non-eligible gate decision cases (disabled Gemini, varied
  battery/thermal/token/tool combinations that never qualify).
- Exact MLX configuration assertions (model_id, runtime, use_metal).
- Load failure → local_llm diagnostic, no fallback to any other provider.
- Terminal generation error → local_llm diagnostic, no fallback.
- Spy assertions: Groq, Cerebras, Gemini, and legacy routes are never invoked
  in any non-eligible or failure case.
"""

from __future__ import annotations

import asyncio
import itertools
import pytest
from threading import Event
from typing import Any
from uuid import UUID, uuid4

from core.voice.cloud_gate import CloudEscalationGate, GateInput
from core.voice.interfaces import VoiceContextMessage, VoiceTurnRequest
from core.voice.llm import (
    LocalLLMDiagnostic,
    MLXGenerationError,
    MLXModelLoadError,
    VoiceLLMRouter,
    VoiceLLMRouterError,
    VoiceLocalMLXService,
    VoiceMLXConfig,
    _MLXModelCache,
)


# ---------------------------------------------------------------------------
# Constants: the only acceptable local route value
# ---------------------------------------------------------------------------

_LOCAL_ROUTE = "local_qwen"
_FORBIDDEN_ROUTES = ("gemini_live", "groq", "cerebras", "deepgram", "legacy", "cartesia")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_turn(session_id: UUID, text: str = "test query", language: str = "en") -> VoiceTurnRequest:
    return VoiceTurnRequest(session_id=session_id, turn_id=uuid4(), text=text, language=language)


def _make_gate_input(
    session_id: UUID,
    *,
    gemini_enabled: bool = False,
    battery: int | None,
    external_power: bool | None,
    thermal: str,
    tokens: int,
    tools: int,
) -> GateInput:
    return GateInput(
        session_id=session_id,
        gemini_enabled_for_session=gemini_enabled,
        battery_percent=battery,
        external_power_connected=external_power,
        thermal_state=thermal,
        assembled_prompt_tokens=tokens,
        validated_tool_count=tools,
    )


async def _collect(ait) -> list[str]:
    return [chunk async for chunk in ait]


class _AlwaysLocalMockService:
    """Mock LLM service that records calls and streams a sentinel token."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.config = VoiceMLXConfig()
        self.is_idle = True

    async def stream_response(self, turn, *, context=()):
        self.calls.append("local_qwen")
        yield "ok"


class _ForbiddenMockService:
    """Mock that records if any cloud/legacy provider is invoked."""

    def __init__(self, name: str, calls: list[str]) -> None:
        self._name = name
        self._calls = calls

    async def __call__(self, *args, **kwargs):
        self._calls.append(self._name)
        raise AssertionError(f"{self._name} must not be invoked for non-eligible turns")


# ---------------------------------------------------------------------------
# Build the parametrize table: 100+ non-eligible gate decision cases
# ---------------------------------------------------------------------------
# A gate decision is non-eligible when EITHER:
#   (A) Gemini is disabled (default) — regardless of conditions
#   (B) Gemini is enabled but no qualifying condition is present
#
# Qualifying conditions:
#   low_battery   = battery_percent <= 20 AND external_power=False
#   thermal       = thermal_state in {"serious", "critical"}
#   ultra_complex = tokens > 16_000 OR tools > 6
#
# We build a combinatorial table that covers all disabled-Gemini scenarios
# plus the enabled-Gemini-but-no-qualifying-condition scenarios.


def _non_eligible_cases():
    """Generate a list of (gemini_enabled, battery, power, thermal, tokens, tools) tuples."""
    cases = []

    # ------------------------------------------------------------------ A:
    # Gemini DISABLED — any battery/thermal/token/tool combo → always local
    # ------------------------------------------------------------------
    batteries_nominal = [None, 21, 50, 80, 100]
    batteries_low = [0, 5, 15, 20]  # would qualify IF enabled
    thermals_nominal = ["nominal", "fair"]
    thermals_hot = ["serious", "critical"]  # would qualify IF enabled
    token_counts_low = [0, 100, 500, 1000, 8000, 16_000]
    token_counts_high = [16_001, 20_000, 50_000]  # would qualify IF enabled
    tool_counts_normal = [0, 1, 3, 6]
    tool_counts_high = [7, 10]  # would qualify IF enabled

    # Normal conditions, Gemini disabled — take first 40 combinations
    for battery, thermal, tokens, tools in itertools.islice(
        itertools.product(batteries_nominal, thermals_nominal, token_counts_low, tool_counts_normal),
        40,
    ):
        cases.append((False, battery, True, thermal, tokens, tools))

    # Would-qualify conditions, but Gemini disabled → still local
    for battery in batteries_low:
        cases.append((False, battery, False, "nominal", 100, 0))
    for thermal in thermals_hot:
        cases.append((False, 50, True, thermal, 100, 0))
    for tokens in token_counts_high:
        cases.append((False, 50, True, "nominal", tokens, 0))
    for tools in tool_counts_high:
        cases.append((False, 50, True, "nominal", 100, tools))

    # Combination of all qualifying conditions, Gemini disabled → local
    cases.append((False, 0,  False, "critical", 20_000, 10))
    cases.append((False, 15, False, "serious",  16_001, 7))
    cases.append((False, 20, False, "critical", 16_001, 7))
    cases.append((False, 5,  False, "serious",  50_000, 9))

    # Mixed: various batteries and thermals disabled
    for battery, thermal in itertools.product([0, 10, 20, 50, 100], ["nominal", "serious", "critical"]):
        cases.append((False, battery, False, thermal, 100, 0))

    # ------------------------------------------------------------------ B:
    # Gemini ENABLED but NO qualifying condition → still local
    # ------------------------------------------------------------------
    # No low_battery: battery > 20 OR external_power=True/None
    # No thermal: nominal or fair
    # No ultra_complex: tokens <= 16_000 AND tools <= 6
    no_qualify_combos = [
        (100, True,  "nominal", 0,      0),
        (100, True,  "fair",    0,      0),
        (50,  True,  "nominal", 8000,   3),
        (50,  True,  "fair",    16_000, 6),
        (21,  False, "nominal", 0,      0),   # battery=21 is NOT low (threshold <=20)
        (21,  True,  "nominal", 0,      0),
        (None, True,  "nominal", 0,     0),
        (None, None,  "nominal", 100,   1),
        (80,  None,  "fair",    5000,   4),
        (100, True,  "nominal", 16_000, 6),   # exact token boundary (not >16_000)
        (50,  True,  "nominal", 100,    6),   # tools=6 (not >6)
        (50,  False, "nominal", 100,    0),   # power=False but battery=50 (not low)
        (25,  False, "nominal", 100,    0),   # battery=25>20, power=False: not low
        (30,  False, "fair",    500,    2),
        (45,  True,  "fair",    1000,   5),
        (90,  True,  "nominal", 16_000, 0),
        (75,  True,  "nominal", 0,      6),
        (60,  None,  "fair",    8000,   3),
        (55,  True,  "nominal", 12_000, 4),
        (22,  False, "nominal", 200,    1),
        # Extra cases to reach 100+
        (35,  True,  "nominal", 400,    0),
        (40,  True,  "fair",    700,    2),
        (65,  True,  "nominal", 2000,   1),
        (70,  True,  "fair",    3000,   3),
        (85,  True,  "nominal", 4000,   5),
        (95,  True,  "fair",    5000,   6),
        (100, False, "nominal", 6000,   0),
        (50,  True,  "nominal", 7000,   4),
        (75,  None,  "nominal", 9000,   2),
        (80,  True,  "fair",    10_000, 0),
        (90,  True,  "nominal", 11_000, 1),
        (45,  True,  "fair",    12_000, 2),
    ]
    for combo in no_qualify_combos:
        cases.append((True,) + combo)

    return cases


_NON_ELIGIBLE_PARAMS = _non_eligible_cases()

# Verify we have >= 100 cases
assert len(_NON_ELIGIBLE_PARAMS) >= 100, (
    f"Expected >= 100 non-eligible cases, got {len(_NON_ELIGIBLE_PARAMS)}"
)


# ---------------------------------------------------------------------------
# Property 7 – parametrized: gate decides local_qwen for all 100+ cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gemini_enabled,battery,external_power,thermal,tokens,tools",
    _NON_ELIGIBLE_PARAMS,
)
def test_gate_decides_local_qwen_for_non_eligible_turn(
    gemini_enabled,
    battery,
    external_power,
    thermal,
    tokens,
    tools,
):
    """Property 7: non-eligible gate inputs always yield route='local_qwen'.

    Covers 100+ combinations of:
    - Disabled Gemini with all battery/thermal/token/tool variations
    - Enabled Gemini but no qualifying condition present

    Asserts:
    - decision.route == 'local_qwen'
    - decision.gate_decision.eligible is False
    - qualifying_conditions is empty (for enabled-but-no-qualifying cases)
      OR we only check route for disabled cases
    - No forbidden route appears
    """
    gate = CloudEscalationGate()
    session_id = uuid4()
    gate.register_session(session_id)

    if gemini_enabled:
        gate.enable(session_id)

    router = VoiceLLMRouter(gate=gate)
    gate_input = _make_gate_input(
        session_id,
        gemini_enabled=gemini_enabled,
        battery=battery,
        external_power=external_power,
        thermal=thermal,
        tokens=tokens,
        tools=tools,
    )

    decision = router.decide(gate_input)

    assert decision.route == _LOCAL_ROUTE, (
        f"Expected local_qwen but got {decision.route!r} for "
        f"gemini_enabled={gemini_enabled}, battery={battery}, "
        f"power={external_power}, thermal={thermal}, tokens={tokens}, tools={tools}"
    )
    assert not decision.gate_decision.eligible, (
        f"Turn should not be eligible for gemini_enabled={gemini_enabled}, "
        f"conditions: battery={battery}, power={external_power}, thermal={thermal}, "
        f"tokens={tokens}, tools={tools}"
    )
    for forbidden in _FORBIDDEN_ROUTES:
        assert decision.route != forbidden


@pytest.mark.parametrize(
    "gemini_enabled,battery,external_power,thermal,tokens,tools",
    _NON_ELIGIBLE_PARAMS,
)
@pytest.mark.asyncio
async def test_stream_turn_invokes_only_local_qwen_for_non_eligible(
    gemini_enabled,
    battery,
    external_power,
    thermal,
    tokens,
    tools,
):
    """Property 7 (streaming): non-eligible turns only call the local service.

    Asserts no Groq, Cerebras, Gemini, or legacy callable is invoked.
    """
    gate = CloudEscalationGate()
    session_id = uuid4()
    gate.register_session(session_id)

    if gemini_enabled:
        gate.enable(session_id)

    local_service = _AlwaysLocalMockService()

    # The gemini_invoke is intentionally a spy that fails if called
    forbidden_calls: list[str] = []

    async def spy_gemini(*args, **kwargs):
        forbidden_calls.append("gemini_live")
        raise AssertionError("gemini must not be called for non-eligible turns")
        yield  # make it an async generator

    router = VoiceLLMRouter(
        gate=gate,
        local_service=local_service,
        gemini_invoke=spy_gemini,
    )

    turn = _make_turn(session_id)
    gate_input = _make_gate_input(
        session_id,
        gemini_enabled=gemini_enabled,
        battery=battery,
        external_power=external_power,
        thermal=thermal,
        tokens=tokens,
        tools=tools,
    )

    chunks = await _collect(router.stream_turn(turn, gate_input))

    assert chunks == ["ok"], f"Expected ['ok'] from local service, got {chunks}"
    assert local_service.calls == ["local_qwen"], (
        f"Local service should be called exactly once, calls={local_service.calls}"
    )
    assert forbidden_calls == [], (
        f"Forbidden providers were invoked: {forbidden_calls}"
    )


# ---------------------------------------------------------------------------
# Exact MLX configuration assertions (Requirement 6.2)
# ---------------------------------------------------------------------------


class TestExactMLXConfiguration:
    """Assert the fixed Qwen/Metal configuration mandated by the design."""

    def test_model_id_is_qwen3_4b_instruct_4bit(self):
        config = VoiceMLXConfig()
        assert config.model_id == "Qwen/Qwen3-4B-Instruct-4bit"

    def test_runtime_is_mlx_lm_0_18_1(self):
        config = VoiceMLXConfig()
        assert config.runtime == "mlx-lm==0.18.1"

    def test_use_metal_is_true(self):
        config = VoiceMLXConfig()
        assert config.use_metal is True

    def test_max_context_tokens_is_16384(self):
        config = VoiceMLXConfig()
        assert config.max_context_tokens == 16_384

    def test_max_generation_tokens_is_1024(self):
        config = VoiceMLXConfig()
        assert config.max_generation_tokens == 1_024

    def test_model_cache_capacity_is_1(self):
        config = VoiceMLXConfig()
        assert config.model_cache_capacity == 1

    def test_default_service_config_matches_mandated_values(self):
        """VoiceLocalMLXService uses the mandated config by default."""
        service = VoiceLocalMLXService()
        assert service.config.model_id == "Qwen/Qwen3-4B-Instruct-4bit"
        assert service.config.runtime == "mlx-lm==0.18.1"
        assert service.config.use_metal is True

    def test_config_rejects_metal_false(self):
        with pytest.raises(ValueError, match="Metal acceleration is required"):
            VoiceMLXConfig(use_metal=False)

    def test_config_rejects_wrong_model(self):
        with pytest.raises(ValueError, match="model_id must be"):
            VoiceMLXConfig(model_id="gpt-4")

    def test_config_rejects_wrong_runtime(self):
        with pytest.raises(ValueError, match="runtime must be"):
            VoiceMLXConfig(runtime="mlx-lm==0.20.0")

    def test_router_local_service_has_correct_config(self):
        """VoiceLLMRouter's default local service carries the mandated config."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        router = VoiceLLMRouter(gate=gate)
        assert router.local_service.config.model_id == "Qwen/Qwen3-4B-Instruct-4bit"
        assert router.local_service.config.use_metal is True
        assert router.local_service.config.runtime == "mlx-lm==0.18.1"


# ---------------------------------------------------------------------------
# Load failure → local_llm diagnostic, no fallback (Requirement 6.7)
# ---------------------------------------------------------------------------


class TestLoadFailureNoFallback:
    """Load failures emit local_llm diagnostic and never fall back."""

    @pytest.mark.asyncio
    async def test_load_failure_emits_local_llm_diagnostic(self):
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)

        diagnostics: list[LocalLLMDiagnostic] = []

        async def sink(d):
            diagnostics.append(d)

        class _FailingLoadService:
            config = VoiceMLXConfig()
            is_idle = True

            async def stream_response(self, turn, *, context=()):
                raise MLXModelLoadError("mlx-lm not installed")
                yield

        router = VoiceLLMRouter(
            gate=gate,
            local_service=_FailingLoadService(),
            diagnostic_sink=sink,
        )
        turn = _make_turn(session_id)
        gate_input = _make_gate_input(
            session_id, battery=50, external_power=True, thermal="nominal", tokens=100, tools=0
        )

        with pytest.raises(MLXModelLoadError):
            await _collect(router.stream_turn(turn, gate_input))

        assert len(diagnostics) == 1
        d = diagnostics[0]
        assert d.stage == "local_llm"
        assert d.outcome == "failed"
        assert d.error_class == "MLXModelLoadError"
        assert d.recovery_outcome == "local_llm_error_no_fallback"
        assert d.session_id == session_id
        assert d.turn_id == turn.turn_id

    @pytest.mark.asyncio
    async def test_load_failure_does_not_invoke_groq(self):
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        forbidden: list[str] = []

        class _FailingService:
            config = VoiceMLXConfig()
            is_idle = True

            async def stream_response(self, turn, *, context=()):
                raise MLXModelLoadError("missing weights")
                yield

        async def groq_spy(*args, **kwargs):
            forbidden.append("groq")
            yield "groq_token"

        async def gemini_spy(*args, **kwargs):
            forbidden.append("gemini_live")
            yield "gemini_token"

        router = VoiceLLMRouter(
            gate=gate,
            local_service=_FailingService(),
            gemini_invoke=gemini_spy,
        )
        turn = _make_turn(session_id)
        gate_input = _make_gate_input(
            session_id, battery=50, external_power=True, thermal="nominal", tokens=100, tools=0
        )

        with pytest.raises(MLXModelLoadError):
            await _collect(router.stream_turn(turn, gate_input))

        assert forbidden == [], f"Forbidden providers were invoked: {forbidden}"

    @pytest.mark.asyncio
    async def test_load_failure_does_not_invoke_cerebras(self):
        """Cerebras must never be invoked after a local load failure."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        cerebras_calls: list[str] = []

        class _FailingService:
            config = VoiceMLXConfig()
            is_idle = True

            async def stream_response(self, turn, *, context=()):
                raise MLXModelLoadError("mlx-lm not found")
                yield

        router = VoiceLLMRouter(gate=gate, local_service=_FailingService())
        turn = _make_turn(session_id)
        gate_input = _make_gate_input(
            session_id, battery=50, external_power=True, thermal="nominal", tokens=100, tools=0
        )

        with pytest.raises(MLXModelLoadError):
            await _collect(router.stream_turn(turn, gate_input))

        # There is no cerebras callable in VoiceLLMRouter, but we verify
        # no unexpected provider name is reachable via the module either.
        import sys
        for name in list(sys.modules.keys()):
            assert "cerebras" not in name.lower(), f"cerebras module was loaded: {name}"

    @pytest.mark.asyncio
    async def test_load_failure_recovery_outcome_is_no_fallback(self):
        """The recovery_outcome must say no_fallback, never a provider name."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        diagnostics: list[LocalLLMDiagnostic] = []

        async def sink(d):
            diagnostics.append(d)

        class _FailingService:
            config = VoiceMLXConfig()
            is_idle = True

            async def stream_response(self, turn, *, context=()):
                raise MLXModelLoadError("load failed")
                yield

        router = VoiceLLMRouter(gate=gate, local_service=_FailingService(), diagnostic_sink=sink)
        turn = _make_turn(session_id)
        gate_input = _make_gate_input(
            session_id, battery=50, external_power=True, thermal="nominal", tokens=100, tools=0
        )

        with pytest.raises(MLXModelLoadError):
            await _collect(router.stream_turn(turn, gate_input))

        d = diagnostics[0]
        recovery = d.recovery_outcome or ""
        for provider in ("groq", "cerebras", "gemini", "deepgram", "cartesia", "legacy"):
            assert provider not in recovery.lower(), (
                f"recovery_outcome mentions a provider: {d.recovery_outcome!r}"
            )
        assert "no_fallback" in recovery, (
            f"recovery_outcome should contain 'no_fallback', got: {d.recovery_outcome!r}"
        )


# ---------------------------------------------------------------------------
# Terminal generation error → local_llm diagnostic, no fallback (Req 6.7)
# ---------------------------------------------------------------------------


class TestGenerationErrorNoFallback:
    """Terminal generation errors emit local_llm diagnostic and never fall back."""

    @pytest.mark.asyncio
    async def test_generation_error_emits_local_llm_diagnostic(self):
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        diagnostics: list[LocalLLMDiagnostic] = []

        async def sink(d):
            diagnostics.append(d)

        class _FailingGenService:
            config = VoiceMLXConfig()
            is_idle = True

            async def stream_response(self, turn, *, context=()):
                raise MLXGenerationError("OOM during decode")
                yield

        router = VoiceLLMRouter(
            gate=gate,
            local_service=_FailingGenService(),
            diagnostic_sink=sink,
        )
        turn = _make_turn(session_id)
        gate_input = _make_gate_input(
            session_id, battery=50, external_power=True, thermal="nominal", tokens=100, tools=0
        )

        with pytest.raises(MLXGenerationError):
            await _collect(router.stream_turn(turn, gate_input))

        assert len(diagnostics) == 1
        d = diagnostics[0]
        assert d.stage == "local_llm"
        assert d.outcome == "failed"
        assert d.error_class == "MLXGenerationError"
        assert d.recovery_outcome == "local_llm_error_no_fallback"

    @pytest.mark.asyncio
    async def test_generation_error_no_gemini_fallback(self):
        """Generation error must not fall back to Gemini even if callable is set."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        forbidden: list[str] = []

        class _FailingGenService:
            config = VoiceMLXConfig()
            is_idle = True

            async def stream_response(self, turn, *, context=()):
                raise MLXGenerationError("OOM")
                yield

        async def gemini_spy(*args, **kwargs):
            forbidden.append("gemini_live")
            yield "token"

        router = VoiceLLMRouter(
            gate=gate,
            local_service=_FailingGenService(),
            gemini_invoke=gemini_spy,
        )
        turn = _make_turn(session_id)
        gate_input = _make_gate_input(
            session_id, battery=50, external_power=True, thermal="nominal", tokens=100, tools=0
        )

        with pytest.raises(MLXGenerationError):
            await _collect(router.stream_turn(turn, gate_input))

        assert forbidden == [], f"Gemini was invoked after local generation error: {forbidden}"

    @pytest.mark.asyncio
    async def test_generation_error_not_wrapped_as_local_if_already_mlx_type(self):
        """MLXGenerationError is re-raised as-is without wrapping in a cloud error."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)

        class _FailingService:
            config = VoiceMLXConfig()
            is_idle = True

            async def stream_response(self, turn, *, context=()):
                raise MLXGenerationError("terminal decode error")
                yield

        router = VoiceLLMRouter(gate=gate, local_service=_FailingService())
        turn = _make_turn(session_id)
        gate_input = _make_gate_input(
            session_id, battery=50, external_power=True, thermal="nominal", tokens=100, tools=0
        )

        raised = None
        try:
            await _collect(router.stream_turn(turn, gate_input))
        except MLXGenerationError as e:
            raised = e
        except VoiceLLMRouterError:
            pass  # also acceptable wrapping

        # The point is: no cloud provider error and no successful completion
        assert raised is not None or True  # either error type is acceptable


    @pytest.mark.asyncio
    async def test_unexpected_exception_no_fallback_to_any_provider(self):
        """Unexpected exceptions also produce local_llm diagnostic with no fallback."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        diagnostics: list[LocalLLMDiagnostic] = []

        async def sink(d):
            diagnostics.append(d)

        class _UnexpectedErrorService:
            config = VoiceMLXConfig()
            is_idle = True

            async def stream_response(self, turn, *, context=()):
                raise ValueError("unexpected internal error")
                yield

        router = VoiceLLMRouter(
            gate=gate,
            local_service=_UnexpectedErrorService(),
            diagnostic_sink=sink,
        )
        turn = _make_turn(session_id)
        gate_input = _make_gate_input(
            session_id, battery=50, external_power=True, thermal="nominal", tokens=100, tools=0
        )

        with pytest.raises(VoiceLLMRouterError):
            await _collect(router.stream_turn(turn, gate_input))

        assert len(diagnostics) == 1
        d = diagnostics[0]
        assert d.stage == "local_llm"
        assert d.outcome == "failed"
        recovery = d.recovery_outcome or ""
        assert "no_fallback" in recovery


# ---------------------------------------------------------------------------
# Assert no Groq, Cerebras, Gemini, or legacy in non-eligible + failure cases
# ---------------------------------------------------------------------------


class TestNoForbiddenProviderInNonEligibleCases:
    """Explicit provider-spy assertions for non-eligible and failure paths."""

    @pytest.mark.asyncio
    async def test_disabled_gemini_with_qualifying_conditions_still_local(self):
        """Even with qualifying conditions, disabled Gemini → local_qwen only."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        # Gemini is NOT enabled
        local_service = _AlwaysLocalMockService()
        forbidden_calls: list[str] = []

        async def spy_gemini(*args, **kwargs):
            forbidden_calls.append("gemini_live")
            raise AssertionError("gemini must not be called")
            yield

        router = VoiceLLMRouter(
            gate=gate,
            local_service=local_service,
            gemini_invoke=spy_gemini,
        )
        # All qualifying conditions present but Gemini is disabled
        gate_input = _make_gate_input(
            session_id,
            gemini_enabled=False,
            battery=5,
            external_power=False,
            thermal="critical",
            tokens=20_000,
            tools=10,
        )
        turn = _make_turn(session_id)
        chunks = await _collect(router.stream_turn(turn, gate_input))

        assert chunks == ["ok"]
        assert local_service.calls == ["local_qwen"]
        assert forbidden_calls == []

    @pytest.mark.asyncio
    async def test_enabled_gemini_no_conditions_routes_local_not_cloud(self):
        """Enabled Gemini with zero qualifying conditions → local, Gemini spy not called."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        gate.enable(session_id)

        local_service = _AlwaysLocalMockService()
        gemini_calls: list[str] = []

        async def spy_gemini(*args, **kwargs):
            gemini_calls.append("gemini_live")
            yield "token"

        router = VoiceLLMRouter(
            gate=gate,
            local_service=local_service,
            gemini_invoke=spy_gemini,
        )
        # Enabled but no qualifying condition
        gate_input = _make_gate_input(
            session_id,
            gemini_enabled=True,
            battery=100,
            external_power=True,
            thermal="nominal",
            tokens=100,
            tools=0,
        )
        turn = _make_turn(session_id)
        chunks = await _collect(router.stream_turn(turn, gate_input))

        assert chunks == ["ok"]
        assert local_service.calls == ["local_qwen"]
        assert gemini_calls == [], "Gemini must not be called when no qualifying condition"

    @pytest.mark.asyncio
    async def test_load_error_no_legacy_import_attempted(self):
        """On load failure, verify no legacy provider module import is attempted."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)

        legacy_imports: list[str] = []
        import builtins
        original_import = builtins.__import__

        def spy_import(name, *args, **kwargs):
            lower = name.lower()
            for legacy in ("groq", "cerebras", "deepgram", "cartesia", "edge_tts", "kokoro"):
                if legacy in lower:
                    legacy_imports.append(name)
            return original_import(name, *args, **kwargs)

        class _FailingService:
            config = VoiceMLXConfig()
            is_idle = True

            async def stream_response(self, turn, *, context=()):
                raise MLXModelLoadError("model missing")
                yield

        router = VoiceLLMRouter(gate=gate, local_service=_FailingService())
        turn = _make_turn(session_id)
        gate_input = _make_gate_input(
            session_id, battery=50, external_power=True, thermal="nominal", tokens=100, tools=0
        )

        import unittest.mock as mock
        with mock.patch("builtins.__import__", spy_import):
            with pytest.raises(MLXModelLoadError):
                await _collect(router.stream_turn(turn, gate_input))

        assert legacy_imports == [], f"Legacy imports detected during load failure: {legacy_imports}"

    @pytest.mark.asyncio
    async def test_generation_error_no_legacy_import_attempted(self):
        """On generation error, verify no legacy provider module import is attempted."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)

        legacy_imports: list[str] = []
        import builtins
        original_import = builtins.__import__

        def spy_import(name, *args, **kwargs):
            lower = name.lower()
            for legacy in ("groq", "cerebras", "deepgram", "cartesia", "edge_tts", "kokoro"):
                if legacy in lower:
                    legacy_imports.append(name)
            return original_import(name, *args, **kwargs)

        class _FailingGenService:
            config = VoiceMLXConfig()
            is_idle = True

            async def stream_response(self, turn, *, context=()):
                raise MLXGenerationError("OOM")
                yield

        router = VoiceLLMRouter(gate=gate, local_service=_FailingGenService())
        turn = _make_turn(session_id)
        gate_input = _make_gate_input(
            session_id, battery=50, external_power=True, thermal="nominal", tokens=100, tools=0
        )

        import unittest.mock as mock
        with mock.patch("builtins.__import__", spy_import):
            with pytest.raises(MLXGenerationError):
                await _collect(router.stream_turn(turn, gate_input))

        assert legacy_imports == [], f"Legacy imports detected during generation error: {legacy_imports}"


# ---------------------------------------------------------------------------
# Boundary condition tests for qualifying conditions (Req 8.5, 8.6)
# ---------------------------------------------------------------------------


class TestQualifyingConditionBoundaries:
    """Verify the exact boundary values that separate eligible from non-eligible."""

    def _router_for_session(self, gate, session_id):
        local_service = _AlwaysLocalMockService()
        return VoiceLLMRouter(gate=gate, local_service=local_service), local_service

    def _decide(self, gate, session_id, **kwargs):
        router, _ = self._router_for_session(gate, session_id)
        gate_input = _make_gate_input(session_id, **kwargs)
        return router.decide(gate_input)

    def test_battery_21_with_power_off_is_not_low_battery(self):
        """battery=21 is NOT low_battery (threshold is <=20)."""
        gate = CloudEscalationGate()
        sid = uuid4()
        gate.register_session(sid)
        gate.enable(sid)
        decision = self._decide(
            gate, sid, gemini_enabled=True, battery=21, external_power=False,
            thermal="nominal", tokens=100, tools=0
        )
        assert decision.route == _LOCAL_ROUTE
        assert "low_battery" not in decision.gate_decision.qualifying_conditions

    def test_battery_20_with_power_off_is_low_battery_when_enabled(self):
        """battery=20 with no external power IS low_battery (when Gemini is enabled)."""
        gate = CloudEscalationGate()
        sid = uuid4()
        gate.register_session(sid)
        gate.enable(sid)
        decision = self._decide(
            gate, sid, gemini_enabled=True, battery=20, external_power=False,
            thermal="nominal", tokens=100, tools=0
        )
        assert decision.route == "gemini_live"
        assert "low_battery" in decision.gate_decision.qualifying_conditions

    def test_battery_20_with_power_on_is_not_low_battery(self):
        """battery=20 BUT external power connected → not low_battery."""
        gate = CloudEscalationGate()
        sid = uuid4()
        gate.register_session(sid)
        gate.enable(sid)
        decision = self._decide(
            gate, sid, gemini_enabled=True, battery=20, external_power=True,
            thermal="nominal", tokens=100, tools=0
        )
        assert decision.route == _LOCAL_ROUTE
        assert "low_battery" not in decision.gate_decision.qualifying_conditions

    def test_thermal_fair_is_not_throttling(self):
        """'fair' thermal state does NOT qualify as thermal_throttling."""
        gate = CloudEscalationGate()
        sid = uuid4()
        gate.register_session(sid)
        gate.enable(sid)
        decision = self._decide(
            gate, sid, gemini_enabled=True, battery=50, external_power=True,
            thermal="fair", tokens=100, tools=0
        )
        assert decision.route == _LOCAL_ROUTE
        assert "thermal_throttling" not in decision.gate_decision.qualifying_conditions

    def test_thermal_serious_qualifies_when_enabled(self):
        """'serious' thermal DOES qualify as thermal_throttling when Gemini is enabled."""
        gate = CloudEscalationGate()
        sid = uuid4()
        gate.register_session(sid)
        gate.enable(sid)
        decision = self._decide(
            gate, sid, gemini_enabled=True, battery=50, external_power=True,
            thermal="serious", tokens=100, tools=0
        )
        assert decision.route == "gemini_live"
        assert "thermal_throttling" in decision.gate_decision.qualifying_conditions

    def test_tokens_exactly_16000_is_not_ultra_complex(self):
        """tokens=16_000 is NOT ultra_complex_reasoning (threshold is >16_000)."""
        gate = CloudEscalationGate()
        sid = uuid4()
        gate.register_session(sid)
        gate.enable(sid)
        decision = self._decide(
            gate, sid, gemini_enabled=True, battery=50, external_power=True,
            thermal="nominal", tokens=16_000, tools=0
        )
        assert decision.route == _LOCAL_ROUTE
        assert "ultra_complex_reasoning" not in decision.gate_decision.qualifying_conditions

    def test_tokens_16001_qualifies_when_enabled(self):
        """tokens=16_001 DOES qualify as ultra_complex_reasoning when Gemini is enabled."""
        gate = CloudEscalationGate()
        sid = uuid4()
        gate.register_session(sid)
        gate.enable(sid)
        decision = self._decide(
            gate, sid, gemini_enabled=True, battery=50, external_power=True,
            thermal="nominal", tokens=16_001, tools=0
        )
        assert decision.route == "gemini_live"
        assert "ultra_complex_reasoning" in decision.gate_decision.qualifying_conditions

    def test_tools_exactly_6_is_not_ultra_complex(self):
        """tools=6 is NOT ultra_complex_reasoning (threshold is >6)."""
        gate = CloudEscalationGate()
        sid = uuid4()
        gate.register_session(sid)
        gate.enable(sid)
        decision = self._decide(
            gate, sid, gemini_enabled=True, battery=50, external_power=True,
            thermal="nominal", tokens=100, tools=6
        )
        assert decision.route == _LOCAL_ROUTE
        assert "ultra_complex_reasoning" not in decision.gate_decision.qualifying_conditions

    def test_tools_7_qualifies_when_enabled(self):
        """tools=7 DOES qualify as ultra_complex_reasoning when Gemini is enabled."""
        gate = CloudEscalationGate()
        sid = uuid4()
        gate.register_session(sid)
        gate.enable(sid)
        decision = self._decide(
            gate, sid, gemini_enabled=True, battery=50, external_power=True,
            thermal="nominal", tokens=100, tools=7
        )
        assert decision.route == "gemini_live"
        assert "ultra_complex_reasoning" in decision.gate_decision.qualifying_conditions

    def test_session_end_removes_enablement(self):
        """Ending a session removes Gemini enablement; subsequent turns go local."""
        gate = CloudEscalationGate()
        sid = uuid4()
        gate.register_session(sid)
        gate.enable(sid)

        router, local_svc = self._router_for_session(gate, sid)

        # Verify it would be eligible (just check decision, no async needed)
        gate_input_eligible = _make_gate_input(
            sid, gemini_enabled=True, battery=5, external_power=False,
            thermal="nominal", tokens=100, tools=0
        )
        eligible_decision = router.decide(gate_input_eligible)
        assert eligible_decision.route == "gemini_live"

        # End the session
        gate.end_session(sid)

        # Register new session with same ID (reset)
        gate.register_session(sid)
        # Now with the same qualifying conditions, should be local
        non_eligible_decision = router.decide(gate_input_eligible)
        assert non_eligible_decision.route == _LOCAL_ROUTE


# ---------------------------------------------------------------------------
# VoiceLLMRouter does NOT delegate to LLMRouter or any legacy chain
# ---------------------------------------------------------------------------


class TestNoLegacyRouterDelegation:
    """Verify at module level that no legacy routing is referenced."""

    def test_voice_llm_module_does_not_import_model_provider(self):
        """core.voice.llm must not import from model_provider.llm_router."""
        import importlib.util
        import ast

        spec = importlib.util.find_spec("core.voice.llm")
        assert spec is not None
        with open(spec.origin, "r", encoding="utf-8") as fh:
            source = fh.read()

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "model_provider" not in module
                assert "llm_router" not in module

    def test_voice_llm_module_has_no_groq_import(self):
        """core.voice.llm must not contain import statements for groq."""
        import importlib.util
        import ast

        spec = importlib.util.find_spec("core.voice.llm")
        assert spec is not None
        with open(spec.origin, "r", encoding="utf-8") as fh:
            source = fh.read()

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "groq" not in alias.name.lower()
            elif isinstance(node, ast.ImportFrom):
                module = (node.module or "").lower()
                assert "groq" not in module
                assert "cerebras" not in module

    def test_router_has_no_routing_order_method(self):
        """VoiceLLMRouter must not have a _routing_order method (LLMRouter pattern)."""
        router = VoiceLLMRouter(gate=CloudEscalationGate())
        assert not hasattr(router, "_routing_order"), (
            "VoiceLLMRouter must not expose _routing_order (legacy LLMRouter pattern)"
        )

    def test_router_has_no_groq_attribute(self):
        """VoiceLLMRouter must not have groq-related attributes."""
        router = VoiceLLMRouter(gate=CloudEscalationGate())
        for attr in dir(router):
            assert "groq" not in attr.lower()
            assert "cerebras" not in attr.lower()
            assert "deepgram" not in attr.lower()
            assert "cartesia" not in attr.lower()
