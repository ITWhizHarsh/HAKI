"""Property 9: Session-scoped cloud gate truth table.

Feature: realtime-local-voice-agent, Property 9: Session-scoped cloud gate truth table

For all session identifiers, enablement actions, end events, battery/power states,
thermal states, prompt token counts, and validated tool counts, Gemini Live is
eligible exactly when it is explicitly enabled for the active session and at least
one qualifying condition holds; otherwise the route is local, and the gate
diagnostic records its enablement, evaluated conditions, qualifying conditions, and
selected route.

**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 10.3**

Design reference: §8, Property 9; V-GATE-PROP

Covers (100+ Hypothesis-generated cases plus focused boundary/session tests):
- Eligibility is True EXACTLY when enablement=True AND at least one condition holds
- Qualifying conditions (exact logic):
    low_battery:           battery_percent <= 20 AND external_power_connected == False
    thermal_throttling:    thermal_state in {"serious", "critical"}
    ultra_complex_reasoning: assembled_prompt_tokens > 16,000 OR validated_tool_count > 6
- Session isolation: enabling session A must not affect session B
- Session end: ending session removes enablement; subsequent evaluations route local
- Gate diagnostics contain: enabled, every evaluated condition, qualifying conditions,
  selected route (no prompt content)
- Disabled + any condition → local_qwen
- Enabled + no condition → local_qwen
- Enabled + qualifying → gemini_live
- Exact boundary values: 20% battery, external-power True/False/None,
  serious/critical thermal, 16,000-token, six-tool thresholds
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from core.voice.cloud_gate import (
    CloudEscalationGate,
    CloudEscalationSessionInactiveError,
    CloudInvocationFailure,
    GateDecision,
    GateInput,
    ThermalState,
)
from core.voice.diagnostics import GateDiagnostic


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALL_CONDITIONS = ("low_battery", "thermal_throttling", "ultra_complex_reasoning")
_THERMAL_STATES: list[ThermalState] = ["nominal", "fair", "serious", "critical"]
_QUALIFYING_THERMALS = frozenset({"serious", "critical"})


# ---------------------------------------------------------------------------
# Pure helper: compute expected qualifying conditions for given inputs
# ---------------------------------------------------------------------------

def _expected_qualifying(
    battery_percent: int | None,
    external_power_connected: bool | None,
    thermal_state: str,
    assembled_prompt_tokens: int,
    validated_tool_count: int,
) -> tuple[str, ...]:
    """Mirror the gate's condition logic for assertion purposes."""
    conditions: list[str] = []
    if (
        battery_percent is not None
        and battery_percent <= 20
        and external_power_connected is False
    ):
        conditions.append("low_battery")
    if thermal_state in _QUALIFYING_THERMALS:
        conditions.append("thermal_throttling")
    if assembled_prompt_tokens > 16_000 or validated_tool_count > 6:
        conditions.append("ultra_complex_reasoning")
    return tuple(conditions)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Battery: None (desktop/unknown) or 0-100 integer
_battery_st = st.one_of(
    st.none(),
    st.integers(min_value=0, max_value=100),
)

# External power: None (unknown) or True/False
_power_st = st.one_of(st.none(), st.booleans())

# Thermal state: all four valid values
_thermal_st = st.sampled_from(_THERMAL_STATES)

# Token counts: cover well below, at, and above the 16,000 boundary
_tokens_st = st.one_of(
    st.integers(min_value=0, max_value=16_000),       # at or below boundary
    st.integers(min_value=16_001, max_value=32_000),  # above boundary
)

# Tool counts: cover 0–6 (non-qualifying) and 7+ (qualifying)
_tools_st = st.one_of(
    st.integers(min_value=0, max_value=6),    # at or below boundary
    st.integers(min_value=7, max_value=20),   # above boundary
)

# Enablement: True or False
_enabled_st = st.booleans()


def _build_gate_input(
    session_id: UUID,
    enabled: bool,
    battery: int | None,
    power: bool | None,
    thermal: str,
    tokens: int,
    tools: int,
) -> GateInput:
    return GateInput(
        session_id=session_id,
        gemini_enabled_for_session=enabled,
        battery_percent=battery,
        external_power_connected=power,
        thermal_state=thermal,  # type: ignore[arg-type]
        assembled_prompt_tokens=tokens,
        validated_tool_count=tools,
    )


def _evaluate(
    enabled: bool,
    battery: int | None,
    power: bool | None,
    thermal: str,
    tokens: int,
    tools: int,
) -> GateDecision:
    """Helper: register a fresh session, optionally enable, and evaluate."""
    gate = CloudEscalationGate()
    sid = uuid4()
    gate.register_session(sid)
    if enabled:
        gate.enable(sid)
    gate_input = _build_gate_input(sid, enabled, battery, power, thermal, tokens, tools)
    return gate.evaluate(gate_input)


# ---------------------------------------------------------------------------
# Property 9 — core truth-table property (>= 100 cases via Hypothesis)
# ---------------------------------------------------------------------------

@given(
    enabled=_enabled_st,
    battery=_battery_st,
    power=_power_st,
    thermal=_thermal_st,
    tokens=_tokens_st,
    tools=_tools_st,
)
@settings(max_examples=200)
def test_property_9_eligibility_iff_enabled_and_qualifying(
    enabled: bool,
    battery: int | None,
    power: bool | None,
    thermal: str,
    tokens: int,
    tools: int,
) -> None:
    """Property 9: Gemini is eligible iff enabled AND at least one condition holds.

    Validates: Requirements 8.1, 8.2, 8.4, 8.5, 8.6, 10.3
    """
    decision = _evaluate(enabled, battery, power, thermal, tokens, tools)

    expected_qualifying = _expected_qualifying(battery, power, thermal, tokens, tools)
    expected_eligible = enabled and bool(expected_qualifying)
    expected_route = "gemini_live" if expected_eligible else "local_qwen"

    # --- Route assertion ---
    assert decision.route == expected_route, (
        f"route={decision.route!r}, expected={expected_route!r} "
        f"[enabled={enabled}, battery={battery}, power={power}, "
        f"thermal={thermal!r}, tokens={tokens}, tools={tools}]"
    )

    # --- Eligibility flag is consistent with route ---
    assert decision.eligible is expected_eligible

    # --- Qualifying conditions match the gate logic exactly ---
    assert decision.qualifying_conditions == expected_qualifying, (
        f"qualifying={decision.qualifying_conditions!r}, "
        f"expected={expected_qualifying!r}"
    )


@given(
    enabled=_enabled_st,
    battery=_battery_st,
    power=_power_st,
    thermal=_thermal_st,
    tokens=_tokens_st,
    tools=_tools_st,
)
@settings(max_examples=200)
def test_property_9_diagnostic_completeness(
    enabled: bool,
    battery: int | None,
    power: bool | None,
    thermal: str,
    tokens: int,
    tools: int,
) -> None:
    """Property 9: gate diagnostic records enabled, all conditions, qualifying, route.

    Validates: Requirements 10.3
    """
    decision = _evaluate(enabled, battery, power, thermal, tokens, tools)
    diag = decision.diagnostic

    # Must record the authoritative enabled state (from session, not caller snapshot)
    assert diag.enabled == enabled

    # Must evaluate EVERY condition (not just the ones that qualify)
    assert set(diag.evaluated) == set(_ALL_CONDITIONS)
    assert len(diag.evaluated) == len(_ALL_CONDITIONS)

    # Qualifying conditions in diagnostic must match the decision
    assert diag.qualifying == decision.qualifying_conditions

    # Selected route in diagnostic must match the decision route
    assert diag.selected_route == decision.route

    # Raw inputs are preserved for transparency
    assert diag.battery_percent == battery
    assert diag.external_power_connected == power
    assert diag.thermal_state == thermal
    assert diag.assembled_prompt_tokens == tokens
    assert diag.validated_tool_count == tools


@given(
    battery=_battery_st,
    power=_power_st,
    thermal=_thermal_st,
    tokens=_tokens_st,
    tools=_tools_st,
)
@settings(max_examples=150)
def test_property_9_disabled_always_routes_local(
    battery: int | None,
    power: bool | None,
    thermal: str,
    tokens: int,
    tools: int,
) -> None:
    """Property 9: disabled + any condition combination → local_qwen always.

    Validates: Requirements 8.1, 8.5
    """
    decision = _evaluate(False, battery, power, thermal, tokens, tools)
    assert decision.route == "local_qwen"
    assert decision.eligible is False


@given(
    battery=st.one_of(st.none(), st.integers(min_value=21, max_value=100)),
    power=st.one_of(st.booleans(), st.none()),
    tokens=st.integers(min_value=0, max_value=16_000),
    tools=st.integers(min_value=0, max_value=6),
)
@settings(max_examples=150)
def test_property_9_enabled_no_condition_routes_local(
    battery: int | None,
    power: bool | None,
    tokens: int,
    tools: int,
) -> None:
    """Property 9: enabled + no qualifying condition → local_qwen always.

    Uses inputs that guarantee all three conditions are absent:
    - battery > 20 or None (never low_battery regardless of power)
    - nominal or fair thermal (no thermal_throttling)
    - tokens <= 16,000 AND tools <= 6 (no ultra_complex_reasoning)

    Validates: Requirements 8.1, 8.6
    """
    for thermal in ("nominal", "fair"):
        decision = _evaluate(True, battery, power, thermal, tokens, tools)
        assert decision.route == "local_qwen", (
            f"enabled with no conditions should route local, got {decision.route!r} "
            f"[battery={battery}, power={power}, thermal={thermal!r}, "
            f"tokens={tokens}, tools={tools}]"
        )
        assert decision.eligible is False
        assert decision.qualifying_conditions == ()


# ---------------------------------------------------------------------------
# Boundary tests — exact threshold values (focused validation)
# ---------------------------------------------------------------------------

class TestBatteryPowerBoundary:
    """Exact boundary at battery=20% and external_power True/False/None."""

    def _decide(self, enabled: bool, battery: int | None, power: bool | None) -> GateDecision:
        return _evaluate(enabled, battery, power, "nominal", 100, 0)

    def test_battery_20_power_false_enabled_is_eligible(self) -> None:
        """battery=20 + power=False + enabled → low_battery qualifies."""
        d = self._decide(True, 20, False)
        assert d.route == "gemini_live"
        assert "low_battery" in d.qualifying_conditions

    def test_battery_20_power_false_disabled_is_local(self) -> None:
        """battery=20 + power=False but disabled → local_qwen."""
        d = self._decide(False, 20, False)
        assert d.route == "local_qwen"
        assert d.eligible is False

    def test_battery_20_power_true_is_not_low_battery(self) -> None:
        """battery=20 + power=True → external power cancels low_battery."""
        d = self._decide(True, 20, True)
        assert d.route == "local_qwen"
        assert "low_battery" not in d.qualifying_conditions

    def test_battery_20_power_none_is_not_low_battery(self) -> None:
        """battery=20 + power=None → unknown power is not False → no low_battery."""
        d = self._decide(True, 20, None)
        assert d.route == "local_qwen"
        assert "low_battery" not in d.qualifying_conditions

    def test_battery_21_power_false_is_not_low_battery(self) -> None:
        """battery=21 + power=False → 21 > 20, so not low_battery."""
        d = self._decide(True, 21, False)
        assert d.route == "local_qwen"
        assert "low_battery" not in d.qualifying_conditions

    def test_battery_0_power_false_enabled_is_eligible(self) -> None:
        """battery=0 + power=False → extreme low battery qualifies."""
        d = self._decide(True, 0, False)
        assert d.route == "gemini_live"
        assert "low_battery" in d.qualifying_conditions

    def test_battery_none_power_false_is_not_low_battery(self) -> None:
        """battery=None → unknown battery cannot satisfy low_battery."""
        d = self._decide(True, None, False)
        assert d.route == "local_qwen"
        assert "low_battery" not in d.qualifying_conditions

    def test_battery_19_power_false_enabled_is_eligible(self) -> None:
        """battery=19 (below 20) + power=False → low_battery qualifies."""
        d = self._decide(True, 19, False)
        assert d.route == "gemini_live"
        assert "low_battery" in d.qualifying_conditions


class TestThermalBoundary:
    """Exact boundaries for serious/critical thermal states."""

    def _decide(self, enabled: bool, thermal: str) -> GateDecision:
        return _evaluate(enabled, 100, True, thermal, 100, 0)

    def test_nominal_thermal_not_qualifying(self) -> None:
        d = self._decide(True, "nominal")
        assert "thermal_throttling" not in d.qualifying_conditions
        assert d.route == "local_qwen"

    def test_fair_thermal_not_qualifying(self) -> None:
        d = self._decide(True, "fair")
        assert "thermal_throttling" not in d.qualifying_conditions
        assert d.route == "local_qwen"

    def test_serious_thermal_enabled_qualifies(self) -> None:
        d = self._decide(True, "serious")
        assert "thermal_throttling" in d.qualifying_conditions
        assert d.route == "gemini_live"

    def test_critical_thermal_enabled_qualifies(self) -> None:
        d = self._decide(True, "critical")
        assert "thermal_throttling" in d.qualifying_conditions
        assert d.route == "gemini_live"

    def test_serious_thermal_disabled_stays_local(self) -> None:
        d = self._decide(False, "serious")
        assert d.route == "local_qwen"
        assert d.eligible is False

    def test_critical_thermal_disabled_stays_local(self) -> None:
        d = self._decide(False, "critical")
        assert d.route == "local_qwen"
        assert d.eligible is False


class TestTokenToolBoundary:
    """Exact 16,000-token and six-tool boundaries for ultra_complex_reasoning."""

    def _decide(self, enabled: bool, tokens: int, tools: int) -> GateDecision:
        return _evaluate(enabled, 100, True, "nominal", tokens, tools)

    def test_tokens_exactly_16000_not_qualifying(self) -> None:
        """16,000 tokens is NOT ultra_complex (threshold is strictly > 16,000)."""
        d = self._decide(True, 16_000, 0)
        assert "ultra_complex_reasoning" not in d.qualifying_conditions
        assert d.route == "local_qwen"

    def test_tokens_16001_enabled_qualifies(self) -> None:
        """16,001 tokens IS ultra_complex when enabled."""
        d = self._decide(True, 16_001, 0)
        assert "ultra_complex_reasoning" in d.qualifying_conditions
        assert d.route == "gemini_live"

    def test_tokens_16001_disabled_local(self) -> None:
        d = self._decide(False, 16_001, 0)
        assert d.route == "local_qwen"
        assert d.eligible is False

    def test_tools_exactly_6_not_qualifying(self) -> None:
        """6 tools is NOT ultra_complex (threshold is strictly > 6)."""
        d = self._decide(True, 0, 6)
        assert "ultra_complex_reasoning" not in d.qualifying_conditions
        assert d.route == "local_qwen"

    def test_tools_7_enabled_qualifies(self) -> None:
        """7 tools IS ultra_complex when enabled."""
        d = self._decide(True, 0, 7)
        assert "ultra_complex_reasoning" in d.qualifying_conditions
        assert d.route == "gemini_live"

    def test_tools_7_disabled_local(self) -> None:
        d = self._decide(False, 0, 7)
        assert d.route == "local_qwen"
        assert d.eligible is False

    def test_tokens_and_tools_both_qualifying_combined(self) -> None:
        """Both token and tool thresholds exceeded → single ultra_complex condition."""
        d = self._decide(True, 20_000, 10)
        assert "ultra_complex_reasoning" in d.qualifying_conditions
        # Condition appears exactly once (not twice)
        assert d.qualifying_conditions.count("ultra_complex_reasoning") == 1

    def test_all_three_conditions_qualify_simultaneously(self) -> None:
        """All three conditions can qualify at once when enabled."""
        d = _evaluate(True, 5, False, "critical", 20_000, 10)
        assert set(d.qualifying_conditions) == {
            "low_battery", "thermal_throttling", "ultra_complex_reasoning"
        }
        assert d.route == "gemini_live"
        assert d.eligible is True


# ---------------------------------------------------------------------------
# Session isolation: enabling session A must not affect session B
# ---------------------------------------------------------------------------

class TestSessionIsolation:
    """Requirement 8.2: enablement is per active session — other sessions unaffected."""

    def test_enable_session_a_does_not_affect_session_b(self) -> None:
        gate = CloudEscalationGate()
        sid_a = uuid4()
        sid_b = uuid4()
        gate.register_session(sid_a)
        gate.register_session(sid_b)

        gate.enable(sid_a)

        # A is enabled, B should remain disabled
        state_a = gate.ui_state(sid_a)
        state_b = gate.ui_state(sid_b)
        assert state_a.gemini_enabled is True
        assert state_b.gemini_enabled is False

    def test_eligible_turn_on_session_a_does_not_route_session_b_to_gemini(self) -> None:
        gate = CloudEscalationGate()
        sid_a = uuid4()
        sid_b = uuid4()
        gate.register_session(sid_a)
        gate.register_session(sid_b)
        gate.enable(sid_a)

        # A qualifies
        input_a = GateInput(
            session_id=sid_a, gemini_enabled_for_session=True,
            battery_percent=5, external_power_connected=False,
            thermal_state="nominal", assembled_prompt_tokens=100, validated_tool_count=0,
        )
        # B has same qualifying conditions but was not enabled
        input_b = GateInput(
            session_id=sid_b, gemini_enabled_for_session=False,
            battery_percent=5, external_power_connected=False,
            thermal_state="nominal", assembled_prompt_tokens=100, validated_tool_count=0,
        )

        decision_a = gate.evaluate(input_a)
        decision_b = gate.evaluate(input_b)

        assert decision_a.route == "gemini_live"
        assert decision_b.route == "local_qwen"
        assert decision_b.eligible is False

    def test_disable_session_a_does_not_affect_session_b(self) -> None:
        gate = CloudEscalationGate()
        sid_a = uuid4()
        sid_b = uuid4()
        gate.register_session(sid_a)
        gate.register_session(sid_b)
        gate.enable(sid_a)
        gate.enable(sid_b)

        gate.disable(sid_a)

        assert gate.ui_state(sid_a).gemini_enabled is False
        assert gate.ui_state(sid_b).gemini_enabled is True

    def test_multiple_concurrent_sessions_are_independently_tracked(self) -> None:
        """Ten sessions, half enabled, half disabled — each isolated."""
        gate = CloudEscalationGate()
        session_ids = [uuid4() for _ in range(10)]
        for sid in session_ids:
            gate.register_session(sid)

        # Enable even-indexed sessions
        for i, sid in enumerate(session_ids):
            if i % 2 == 0:
                gate.enable(sid)

        for i, sid in enumerate(session_ids):
            expected = (i % 2 == 0)
            assert gate.ui_state(sid).gemini_enabled is expected, (
                f"Session index {i}: expected gemini_enabled={expected}"
            )

    @given(n_sessions=st.integers(min_value=2, max_value=8))
    @settings(max_examples=50)
    def test_property_session_isolation_arbitrary_count(self, n_sessions: int) -> None:
        """Enabling one session in a pool never changes another session's state."""
        gate = CloudEscalationGate()
        sids = [uuid4() for _ in range(n_sessions)]
        for sid in sids:
            gate.register_session(sid)

        # Enable the first session only
        gate.enable(sids[0])

        assert gate.ui_state(sids[0]).gemini_enabled is True
        for sid in sids[1:]:
            assert gate.ui_state(sid).gemini_enabled is False, (
                "Enabling session 0 must not affect other sessions"
            )


# ---------------------------------------------------------------------------
# Session end: ending session removes enablement; subsequent turns route local
# ---------------------------------------------------------------------------

class TestSessionEndRemovesEnablement:
    """Requirement 8.3: session end disables Gemini for that session."""

    def test_end_session_removes_enablement_state(self) -> None:
        gate = CloudEscalationGate()
        sid = uuid4()
        gate.register_session(sid)
        gate.enable(sid)
        assert gate.ui_state(sid).gemini_enabled is True

        gate.end_session(sid)

        state = gate.ui_state(sid)
        assert state.active is False
        assert state.gemini_enabled is False

    def test_end_session_then_re_register_starts_disabled(self) -> None:
        """After ending a session, re-registering the same ID starts disabled."""
        gate = CloudEscalationGate()
        sid = uuid4()
        gate.register_session(sid)
        gate.enable(sid)
        gate.end_session(sid)

        # Re-register (simulating a new session with same UUID, e.g. in tests)
        gate.register_session(sid)
        assert gate.ui_state(sid).gemini_enabled is False

    def test_evaluation_after_end_routes_local_even_with_qualifying_conditions(self) -> None:
        """After session end then re-register (disabled), qualifying inputs → local."""
        gate = CloudEscalationGate()
        sid = uuid4()
        gate.register_session(sid)
        gate.enable(sid)
        gate.end_session(sid)
        gate.register_session(sid)  # new session, disabled by default

        gate_input = GateInput(
            session_id=sid,
            gemini_enabled_for_session=False,  # reflects new disabled session
            battery_percent=5,
            external_power_connected=False,
            thermal_state="critical",
            assembled_prompt_tokens=20_000,
            validated_tool_count=10,
        )
        decision = gate.evaluate(gate_input)
        assert decision.route == "local_qwen"
        assert decision.eligible is False

    def test_end_session_prevents_further_enable_actions(self) -> None:
        """After end_session, calling enable raises CloudEscalationSessionInactiveError."""
        gate = CloudEscalationGate()
        sid = uuid4()
        gate.register_session(sid)
        gate.end_session(sid)

        with pytest.raises(CloudEscalationSessionInactiveError):
            gate.enable(sid)

    @given(
        battery=st.integers(min_value=0, max_value=20),
        thermal=st.sampled_from(["serious", "critical"]),
        tokens=st.integers(min_value=16_001, max_value=32_000),
        tools=st.integers(min_value=7, max_value=20),
    )
    @settings(max_examples=50)
    def test_property_session_end_always_routes_local(
        self,
        battery: int,
        thermal: str,
        tokens: int,
        tools: int,
    ) -> None:
        """After end_session + re-register, all qualifying combinations → local."""
        gate = CloudEscalationGate()
        sid = uuid4()
        gate.register_session(sid)
        gate.enable(sid)
        gate.end_session(sid)
        gate.register_session(sid)  # fresh session — disabled

        gate_input = GateInput(
            session_id=sid,
            gemini_enabled_for_session=False,
            battery_percent=battery,
            external_power_connected=False,
            thermal_state=thermal,  # type: ignore[arg-type]
            assembled_prompt_tokens=tokens,
            validated_tool_count=tools,
        )
        decision = gate.evaluate(gate_input)
        assert decision.route == "local_qwen"
        assert decision.eligible is False


# ---------------------------------------------------------------------------
# Input snapshot cannot bypass gate state (Req 8.2)
# ---------------------------------------------------------------------------

class TestInputSnapshotCannotBypassGate:
    """The gemini_enabled_for_session snapshot field cannot manufacture eligibility."""

    def test_snapshot_true_on_disabled_session_still_routes_local(self) -> None:
        """The gate derives eligibility from session state, not the caller's snapshot."""
        gate = CloudEscalationGate()
        sid = uuid4()
        gate.register_session(sid)
        # Session is NOT enabled via gate.enable()

        gate_input = GateInput(
            session_id=sid,
            gemini_enabled_for_session=True,  # caller claims it's enabled — gate ignores this
            battery_percent=5,
            external_power_connected=False,
            thermal_state="critical",
            assembled_prompt_tokens=20_000,
            validated_tool_count=10,
        )
        decision = gate.evaluate(gate_input)

        # Gate uses its own tracking — session was not explicitly enabled
        assert decision.route == "local_qwen"
        assert decision.eligible is False
        assert decision.diagnostic.enabled is False  # diagnostic shows truth, not snapshot

    @given(
        battery=st.integers(min_value=0, max_value=20),
        tools=st.integers(min_value=7, max_value=15),
    )
    @settings(max_examples=50)
    def test_property_snapshot_bypass_impossible(self, battery: int, tools: int) -> None:
        """No matter how the caller sets the snapshot, the gate state is authoritative."""
        gate = CloudEscalationGate()
        sid = uuid4()
        gate.register_session(sid)
        # Deliberately NOT calling gate.enable(sid)

        gate_input = GateInput(
            session_id=sid,
            gemini_enabled_for_session=True,  # attacker-controlled snapshot
            battery_percent=battery,
            external_power_connected=False,
            thermal_state="critical",
            assembled_prompt_tokens=20_000,
            validated_tool_count=tools,
        )
        decision = gate.evaluate(gate_input)
        assert decision.route == "local_qwen"
        assert decision.eligible is False


# ---------------------------------------------------------------------------
# Diagnostic event content: no prompt text, all required fields present
# ---------------------------------------------------------------------------

class TestDiagnosticEventContent:
    """Requirement 10.3: gate diagnostic events record required fields, no prompt content."""

    def test_completed_event_has_all_required_fields(self) -> None:
        gate = CloudEscalationGate()
        sid = uuid4()
        turn_id = uuid4()
        gate.register_session(sid)
        gate.enable(sid)

        gate_input = GateInput(
            session_id=sid,
            gemini_enabled_for_session=True,
            battery_percent=5,
            external_power_connected=False,
            thermal_state="critical",
            assembled_prompt_tokens=20_000,
            validated_tool_count=10,
        )
        decision = gate.evaluate(gate_input)
        event = decision.diagnostic_event(turn_id=turn_id)
        record = event.as_dict()

        # Required identifiers
        assert record["session_id"] == str(sid)
        assert record["turn_id"] == str(turn_id)
        assert record["stage"] == "cloud_gate"
        assert record["outcome"] == "completed"
        assert record["selected_route"] == "gemini_live"

        # Gate sub-record
        gate_record = record["gate"]
        assert gate_record is not None
        assert gate_record["enabled"] is True
        assert set(gate_record["evaluated"]) == set(_ALL_CONDITIONS)
        assert set(gate_record["qualifying"]) == {
            "low_battery", "thermal_throttling", "ultra_complex_reasoning"
        }
        assert gate_record["selected_route"] == "gemini_live"

        # No prompt content
        assert "prompt" not in record
        assert "transcript" not in record
        assert "response" not in record

    def test_local_route_event_records_disabled_state(self) -> None:
        gate = CloudEscalationGate()
        sid = uuid4()
        turn_id = uuid4()
        gate.register_session(sid)
        # Not enabled

        gate_input = GateInput(
            session_id=sid,
            gemini_enabled_for_session=False,
            battery_percent=100,
            external_power_connected=True,
            thermal_state="nominal",
            assembled_prompt_tokens=0,
            validated_tool_count=0,
        )
        decision = gate.evaluate(gate_input)
        event = decision.diagnostic_event(turn_id=turn_id)
        record = event.as_dict()

        assert record["selected_route"] == "local_qwen"
        assert record["outcome"] == "completed"
        gate_record = record["gate"]
        assert gate_record["enabled"] is False
        assert gate_record["qualifying"] == []

    def test_failed_eligible_invocation_event_records_no_fallback(self) -> None:
        gate = CloudEscalationGate()
        sid = uuid4()
        turn_id = uuid4()
        gate.register_session(sid)
        gate.enable(sid)

        gate_input = GateInput(
            session_id=sid,
            gemini_enabled_for_session=True,
            battery_percent=10,
            external_power_connected=False,
            thermal_state="nominal",
            assembled_prompt_tokens=100,
            validated_tool_count=0,
        )
        decision = gate.evaluate(gate_input)
        assert decision.eligible

        failure = gate.report_eligible_invocation_failure(decision, "TimeoutError")
        failure_event = failure.diagnostic_event(turn_id=turn_id)
        record = failure_event.as_dict()

        assert record["outcome"] == "failed"
        assert record["selected_route"] == "gemini_live"
        assert record["error_class"] == "TimeoutError"
        assert record["recovery_outcome"] == "reported_no_fallback"

    @given(
        enabled=_enabled_st,
        battery=_battery_st,
        power=_power_st,
        thermal=_thermal_st,
        tokens=_tokens_st,
        tools=_tools_st,
    )
    @settings(max_examples=100)
    def test_property_diagnostic_never_contains_prompt_or_transcript(
        self,
        enabled: bool,
        battery: int | None,
        power: bool | None,
        thermal: str,
        tokens: int,
        tools: int,
    ) -> None:
        """Gate diagnostic records never contain prompt or transcript content."""
        decision = _evaluate(enabled, battery, power, thermal, tokens, tools)
        turn_id = uuid4()
        event = decision.diagnostic_event(turn_id=turn_id)
        record = event.as_dict()

        for forbidden_key in ("prompt", "transcript", "response", "pcm", "audio"):
            assert forbidden_key not in record, (
                f"Diagnostic record contains forbidden field: {forbidden_key!r}"
            )

        # content_capture_enabled must default False
        assert record.get("content_capture_enabled") is False


# ---------------------------------------------------------------------------
# CloudInvocationFailure contract: no retry, no fallback
# ---------------------------------------------------------------------------

class TestCloudInvocationFailure:
    """Requirement 8.7: Gemini failure is terminal — no retry, no fallback."""

    def test_failure_has_no_retry_and_no_fallback(self) -> None:
        gate = CloudEscalationGate()
        sid = uuid4()
        gate.register_session(sid)
        gate.enable(sid)

        gate_input = GateInput(
            session_id=sid,
            gemini_enabled_for_session=True,
            battery_percent=10,
            external_power_connected=False,
            thermal_state="nominal",
            assembled_prompt_tokens=100,
            validated_tool_count=0,
        )
        decision = gate.evaluate(gate_input)
        assert decision.eligible

        failure = gate.report_eligible_invocation_failure(decision, RuntimeError("503"))
        assert failure.retry_attempted is False
        assert failure.fallback_attempted is False
        assert failure.recovery_outcome == "reported_no_fallback"

    def test_failure_cannot_be_constructed_for_non_eligible_decision(self) -> None:
        """CloudInvocationFailure requires an eligible decision."""
        gate = CloudEscalationGate()
        sid = uuid4()
        gate.register_session(sid)
        # Not enabled → non-eligible decision

        gate_input = GateInput(
            session_id=sid,
            gemini_enabled_for_session=False,
            battery_percent=100,
            external_power_connected=True,
            thermal_state="nominal",
            assembled_prompt_tokens=0,
            validated_tool_count=0,
        )
        decision = gate.evaluate(gate_input)
        assert not decision.eligible

        with pytest.raises(ValueError):
            CloudInvocationFailure(decision=decision, error_class="SomeError")

    @given(
        battery=st.integers(min_value=0, max_value=20),
        thermal=st.sampled_from(["serious", "critical"]),
    )
    @settings(max_examples=50)
    def test_property_failure_is_always_terminal_no_fallback(
        self,
        battery: int,
        thermal: str,
    ) -> None:
        """For all eligible decisions, failure is always reported_no_fallback."""
        gate = CloudEscalationGate()
        sid = uuid4()
        gate.register_session(sid)
        gate.enable(sid)

        gate_input = GateInput(
            session_id=sid,
            gemini_enabled_for_session=True,
            battery_percent=battery,
            external_power_connected=False,
            thermal_state=thermal,  # type: ignore[arg-type]
            assembled_prompt_tokens=100,
            validated_tool_count=0,
        )
        decision = gate.evaluate(gate_input)
        assert decision.eligible

        failure = gate.report_eligible_invocation_failure(decision, "NetworkError")
        assert failure.retry_attempted is False
        assert failure.fallback_attempted is False
        assert failure.recovery_outcome == "reported_no_fallback"


# ---------------------------------------------------------------------------
# New session default: always starts disabled (Requirement 8.1)
# ---------------------------------------------------------------------------

class TestNewSessionStartsDisabled:
    """Every new voice session initializes with Gemini disabled."""

    @given(n=st.integers(min_value=1, max_value=20))
    @settings(max_examples=50)
    def test_property_every_new_session_starts_disabled(self, n: int) -> None:
        gate = CloudEscalationGate()
        for _ in range(n):
            sid = uuid4()
            state = gate.register_session(sid)
            assert state.gemini_enabled is False, (
                "Every new session must start with Gemini disabled"
            )
            assert state.active is True

    def test_register_session_resets_any_prior_enablement(self) -> None:
        """Re-registering a session ID resets enablement to False."""
        gate = CloudEscalationGate()
        sid = uuid4()
        gate.register_session(sid)
        gate.enable(sid)
        assert gate.ui_state(sid).gemini_enabled is True

        # Re-register (simulates a new session started with same UUID)
        state = gate.register_session(sid)
        assert state.gemini_enabled is False

