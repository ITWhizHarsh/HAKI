"""Focused truth-table coverage for explicit session-scoped cloud escalation.

Validates: Requirements 8.1–8.7, 10.3
"""

from __future__ import annotations

from itertools import product
from uuid import uuid4

import pytest

from core.voice.cloud_gate import (
    CloudEscalationGate,
    CloudEscalationSessionInactiveError,
    GateInput,
)
from core.voice.session import VoiceSession, VoiceSessionClosedError


def _input(
    session_id,
    *,
    enabled: bool,
    battery_percent: int | None = 100,
    external_power_connected: bool | None = True,
    thermal_state: str = "nominal",
    assembled_prompt_tokens: int = 0,
    validated_tool_count: int = 0,
) -> GateInput:
    return GateInput(
        session_id=session_id,
        gemini_enabled_for_session=enabled,
        battery_percent=battery_percent,
        external_power_connected=external_power_connected,
        thermal_state=thermal_state,  # type: ignore[arg-type]
        assembled_prompt_tokens=assembled_prompt_tokens,
        validated_tool_count=validated_tool_count,
    )


@pytest.mark.parametrize(
    (
        "enabled, battery_percent, external_power_connected, thermal_state, "
        "assembled_prompt_tokens, validated_tool_count"
    ),
    list(
        product(
            (False, True),
            (20, 21),
            (False, True),
            ("nominal", "serious"),
            (16_000, 16_001),
            (6, 7),
        )
    ),
)
def test_gate_truth_table_at_all_condition_boundaries(
    enabled: bool,
    battery_percent: int,
    external_power_connected: bool,
    thermal_state: str,
    assembled_prompt_tokens: int,
    validated_tool_count: int,
) -> None:
    """Gemini is eligible iff explicit enablement and one condition both hold."""
    gate = CloudEscalationGate()
    session_id = uuid4()
    gate.register_session(session_id)
    if enabled:
        gate.enable(session_id)

    decision = gate.evaluate(
        _input(
            session_id,
            enabled=enabled,
            battery_percent=battery_percent,
            external_power_connected=external_power_connected,
            thermal_state=thermal_state,
            assembled_prompt_tokens=assembled_prompt_tokens,
            validated_tool_count=validated_tool_count,
        )
    )

    low_battery = battery_percent <= 20 and not external_power_connected
    thermal = thermal_state == "serious"
    complex_reasoning = assembled_prompt_tokens > 16_000 or validated_tool_count > 6
    expected_conditions = tuple(
        condition
        for condition, present in (
            ("low_battery", low_battery),
            ("thermal_throttling", thermal),
            ("ultra_complex_reasoning", complex_reasoning),
        )
        if present
    )
    assert decision.qualifying_conditions == expected_conditions
    assert decision.eligible is (enabled and bool(expected_conditions))
    assert decision.route == ("gemini_live" if decision.eligible else "local_qwen")


@pytest.mark.parametrize("thermal_state", ("serious", "critical"))
def test_thermal_states_and_missing_battery_data_are_evaluated_safely(thermal_state: str) -> None:
    """Both throttling states qualify; incomplete power data never manufactures low battery."""
    gate = CloudEscalationGate()
    session_id = uuid4()
    gate.register_session(session_id)
    gate.enable(session_id)

    decision = gate.evaluate(
        _input(
            session_id,
            enabled=True,
            battery_percent=None,
            external_power_connected=None,
            thermal_state=thermal_state,
        )
    )

    assert decision.route == "gemini_live"
    assert decision.qualifying_conditions == ("thermal_throttling",)


def test_input_enablement_claim_cannot_bypass_explicit_active_session_action() -> None:
    """A caller cannot select cloud by setting the non-authoritative input snapshot."""
    gate = CloudEscalationGate()
    session_id = uuid4()
    gate.register_session(session_id)

    decision = gate.evaluate(
        _input(
            session_id,
            enabled=True,
            battery_percent=20,
            external_power_connected=False,
        )
    )

    assert decision.route == "local_qwen"
    assert decision.diagnostic.enabled is False
    assert decision.qualifying_conditions == ("low_battery",)


@pytest.mark.asyncio
async def test_session_enablement_is_isolated_displayed_and_removed_on_end() -> None:
    """Each VoiceSession starts disabled and close removes its UI-visible enablement."""
    first, second = VoiceSession(uuid4()), VoiceSession(uuid4())

    assert first.cloud_escalation_state.gemini_enabled is False
    assert second.cloud_escalation_state.gemini_enabled is False
    enabled = await first.set_gemini_live_enabled(True)
    assert enabled.active is True
    assert enabled.gemini_enabled is True
    assert second.cloud_escalation_state.gemini_enabled is False

    await first.close()
    assert first.cloud_escalation_state.active is False
    assert first.cloud_escalation_state.gemini_enabled is False
    with pytest.raises(VoiceSessionClosedError):
        await first.set_gemini_live_enabled(True)


def test_gate_diagnostic_records_all_inputs_conditions_and_selected_route() -> None:
    """Gate diagnostic values contain no prompt content while retaining all decisions."""
    gate = CloudEscalationGate()
    session_id, turn_id = uuid4(), uuid4()
    gate.register_session(session_id)
    gate.enable(session_id)
    decision = gate.evaluate(
        _input(
            session_id,
            enabled=True,
            battery_percent=20,
            external_power_connected=False,
            thermal_state="critical",
            assembled_prompt_tokens=16_001,
            validated_tool_count=7,
        )
    )

    event = decision.diagnostic_event(turn_id=turn_id).as_dict()
    assert event["stage"] == "cloud_gate"
    assert event["outcome"] == "completed"
    assert event["selected_route"] == "gemini_live"
    assert event["gate"] == {
        "enabled": True,
        "evaluated": [
            "low_battery",
            "thermal_throttling",
            "ultra_complex_reasoning",
        ],
        "battery_percent": 20,
        "external_power_connected": False,
        "thermal_state": "critical",
        "assembled_prompt_tokens": 16_001,
        "validated_tool_count": 7,
        "qualifying": [
            "low_battery",
            "thermal_throttling",
            "ultra_complex_reasoning",
        ],
        "selected_route": "gemini_live",
    }
    assert "prompt" not in event
    assert "transcript" not in event


def test_eligible_invocation_failure_is_terminal_and_has_no_retry_or_fallback() -> None:
    """A selected Gemini failure reports once without routing the turn elsewhere."""
    gate = CloudEscalationGate()
    session_id, turn_id = uuid4(), uuid4()
    gate.register_session(session_id)
    gate.enable(session_id)
    decision = gate.evaluate(
        _input(
            session_id,
            enabled=True,
            assembled_prompt_tokens=16_001,
        )
    )

    failure = gate.report_eligible_invocation_failure(decision, RuntimeError("unavailable"))
    event = failure.diagnostic_event(turn_id=turn_id).as_dict()
    assert failure.retry_attempted is False
    assert failure.fallback_attempted is False
    assert failure.recovery_outcome == "reported_no_fallback"
    assert event["selected_route"] == "gemini_live"
    assert event["outcome"] == "failed"
    assert event["error_class"] == "RuntimeError"
    assert event["recovery_outcome"] == "reported_no_fallback"


def test_enablement_rejects_non_active_sessions() -> None:
    """The UI action cannot enable Gemini after a session has ended."""
    gate = CloudEscalationGate()
    session_id = uuid4()
    gate.register_session(session_id)
    gate.end_session(session_id)

    with pytest.raises(CloudEscalationSessionInactiveError):
        gate.enable(session_id)
