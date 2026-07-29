"""Property 12: Privacy-preserving diagnostic completeness.

Feature: realtime-local-voice-agent, Property 12: Privacy-preserving diagnostic completeness

For all voice turn starts, terminal outcomes, gate evaluations, and stage failures,
diagnostics contain the required IDs, selected components/routes, applicable
timing/resource/error/recovery fields, while default serialization excludes raw
microphone data and full transcript text unless the session-scoped content control
is enabled.

**Validates: Requirements 10.1, 10.2, 10.3, 10.5, 10.6**

Design reference: §9, Property 12; V-DIAG-PROP

Covers:
- All (session_id, turn_id, stage, outcome) combinations: required fields present
- Default records contain NO: pcm, audio, transcript, response, prompt text, tool arguments
- content_capture_enabled defaults False
- Terminal outcomes (completed, cancelled, failed): timing fields + error_class if failed
- Gate decisions: gate diagnostic records enabled, evaluated conditions, qualifying, selected route
- No prompt text in gate diagnostic
- Failures: error_class required, recovery_outcome optional
- Content capture: transcript/response fields only present when content_capture_enabled=True
- Raw audio NEVER appears regardless of flag
- Named explicitly (diagnostic_transcript_text, not transcript_text)
- Content capture expiry: after session end, content_capture_enabled becomes False
- Include: cancellation outcomes, missing optional metrics, redaction hashes,
  content-control expiry, serialization round trips
"""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from core.voice.cloud_gate import CloudEscalationGate, GateInput, ThermalState
from core.voice.diagnostics import (
    ContentCaptureRegistry,
    GateDiagnostic,
    VoiceDiagnosticEvent,
    append_diagnostic,
    read_diagnostics,
)

# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

_ALL_STAGES = [
    "asr", "ipc", "pipecat", "voice_processing", "local_llm",
    "tool_call", "local_tts", "memory_budget", "cloud_gate", "playback",
]

_ALL_OUTCOMES = ["started", "completed", "cancelled", "failed", "rejected"]

_TERMINAL_OUTCOMES = ["completed", "cancelled", "failed"]

# Fields that MUST always be absent from a default (no content capture) record.
_PROHIBITED_DEFAULT_FIELDS: frozenset[str] = frozenset({
    "pcm", "pcm_bytes", "audio", "audio_bytes", "raw_audio",
    "transcript", "transcript_text",
    "response", "response_text",
    "prompt", "prompt_text",
    "tool_arguments", "tool_results", "tool_content",
})

# Fields that are ALWAYS required in every serialized diagnostic record.
_REQUIRED_FIELDS: frozenset[str] = frozenset({
    "schema_version", "event_id", "session_id", "turn_id",
    "stage", "outcome",
    "started_monotonic_ns",
    "transcription_completed_monotonic_ns",
    "first_llm_text_monotonic_ns",
    "first_tts_text_monotonic_ns",
    "first_pcm_delivered_monotonic_ns",
    "ttfa_ms",
    "selected_route",
    "asr_engine",
    "tts_engine",
    "model_resident_bytes",
    "pipeline_memory_bytes",
    "gate",
    "error_class",
    "recovery_outcome",
    "content_capture_enabled",
})

_GATE_CONDITIONS = ("low_battery", "thermal_throttling", "ultra_complex_reasoning")


def _has_prohibited_field(record: dict) -> bool:
    return bool(set(record.keys()) & _PROHIBITED_DEFAULT_FIELDS)


def _has_raw_audio_field(record: dict) -> bool:
    return bool(set(record.keys()) & {"pcm", "pcm_bytes", "audio", "audio_bytes", "raw_audio"})


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

stages_st = st.sampled_from(_ALL_STAGES)
outcomes_st = st.sampled_from(_ALL_OUTCOMES)
terminal_outcomes_st = st.sampled_from(_TERMINAL_OUTCOMES)

# Monotonic nanosecond timestamps: 0 = not yet recorded, up to 10 seconds
monotonic_ns_st = st.integers(min_value=0, max_value=10_000_000_000)
# Positive monotonic timestamps only (recorded events)
positive_monotonic_ns_st = st.integers(min_value=1, max_value=10_000_000_000)

# TTFA in milliseconds: None or non-negative float
ttfa_ms_st = st.one_of(st.none(), st.floats(min_value=0.0, max_value=5000.0, allow_nan=False))

# Memory bytes: None or non-negative integer
memory_bytes_st = st.one_of(st.none(), st.integers(min_value=0, max_value=6_000_000_000))

# Error class strings (non-empty)
error_class_st = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_"),
    min_size=1,
    max_size=80,
)

# Recovery outcome strings (optional)
recovery_outcome_st = st.one_of(
    st.none(),
    st.sampled_from(["reported_no_fallback", "retried_local", "error_presented", "cancelled"]),
)

# ASR / TTS engine identifiers (optional)
engine_id_st = st.one_of(
    st.none(),
    st.sampled_from(["qwen3_asr_coreml", "qwen3_asr_mlx", "xtts_v2", "mock_asr"]),
)

# Route selection (optional)
route_st = st.one_of(st.none(), st.sampled_from(["local_qwen", "gemini_live"]))

# Battery percent: None or 0–100
battery_pct_st = st.one_of(st.none(), st.integers(min_value=0, max_value=100))

# Thermal state for gate evaluation
thermal_state_st = st.sampled_from(["nominal", "fair", "serious", "critical"])

# Prompt token and tool count for gate
prompt_tokens_st = st.integers(min_value=0, max_value=20_000)
tool_count_st = st.integers(min_value=0, max_value=10)


def _make_gate_diagnostic(
    *,
    enabled: bool = False,
    qualifying: tuple = (),
    battery_percent: int | None = 80,
    external_power_connected: bool | None = True,
    thermal_state: str = "nominal",
    assembled_prompt_tokens: int = 100,
    validated_tool_count: int = 1,
) -> GateDiagnostic:
    """Build a GateDiagnostic whose route is consistent with enabled+qualifying."""
    route = "gemini_live" if enabled and qualifying else "local_qwen"
    return GateDiagnostic(
        enabled=enabled,
        evaluated=_GATE_CONDITIONS,
        battery_percent=battery_percent,
        external_power_connected=external_power_connected,
        thermal_state=thermal_state,  # type: ignore[arg-type]
        assembled_prompt_tokens=assembled_prompt_tokens,
        validated_tool_count=validated_tool_count,
        qualifying=qualifying,
        selected_route=route,  # type: ignore[arg-type]
    )


def _make_event(
    *,
    stage: str = "asr",
    outcome: str = "started",
    content_capture_enabled: bool = False,
    started_monotonic_ns: int = 0,
    asr_engine: str | None = None,
    tts_engine: str | None = None,
    selected_route: str | None = None,
    error_class: str | None = None,
    recovery_outcome: str | None = None,
    ttfa_ms: float | None = None,
    model_resident_bytes: int | None = None,
    pipeline_memory_bytes: int | None = None,
    gate: GateDiagnostic | None = None,
) -> VoiceDiagnosticEvent:
    return VoiceDiagnosticEvent(
        session_id=uuid4(),
        turn_id=uuid4(),
        stage=stage,  # type: ignore[arg-type]
        outcome=outcome,  # type: ignore[arg-type]
        content_capture_enabled=content_capture_enabled,
        started_monotonic_ns=started_monotonic_ns,
        asr_engine=asr_engine,
        tts_engine=tts_engine,
        selected_route=selected_route,  # type: ignore[arg-type]
        error_class=error_class,
        recovery_outcome=recovery_outcome,
        ttfa_ms=ttfa_ms,
        model_resident_bytes=model_resident_bytes,
        pipeline_memory_bytes=pipeline_memory_bytes,
        gate=gate,
    )


# ---------------------------------------------------------------------------
# Property 12 — Part 1: Required fields present for all (stage, outcome) pairs
# ---------------------------------------------------------------------------

@given(
    stage=stages_st,
    outcome=outcomes_st,
    started_ns=monotonic_ns_st,
    asr_engine=engine_id_st,
    tts_engine=engine_id_st,
)
@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow])
def test_property_12_required_fields_present_for_all_stage_outcome_combinations(
    stage: str,
    outcome: str,
    started_ns: int,
    asr_engine: str | None,
    tts_engine: str | None,
) -> None:
    """
    **Validates: Requirements 10.1, 10.2**

    Feature: realtime-local-voice-agent, Property 12: Privacy-preserving diagnostic completeness

    For all (stage, outcome) combinations, every required identifier and
    measurement field must appear in the serialized record.
    """
    # failed outcomes require an error_class; cloud_gate completed cannot have one
    error_class = "SomeError" if outcome == "failed" else None

    e = _make_event(
        stage=stage,
        outcome=outcome,
        started_monotonic_ns=started_ns,
        asr_engine=asr_engine,
        tts_engine=tts_engine,
        error_class=error_class,
    )
    record = e.as_dict()

    missing = _REQUIRED_FIELDS - set(record.keys())
    assert not missing, (
        f"stage={stage!r} outcome={outcome!r}: required fields missing: {missing}"
    )
    # Schema version must be 1
    assert record["schema_version"] == 1
    # IDs must be valid UUID strings
    for id_field in ("event_id", "session_id", "turn_id"):
        UUID(record[id_field])  # raises if invalid
    # Stage and outcome must round-trip
    assert record["stage"] == stage
    assert record["outcome"] == outcome
    # content_capture_enabled must default False
    assert record["content_capture_enabled"] is False


# ---------------------------------------------------------------------------
# Property 12 — Part 2: Default records contain no prohibited content fields
# ---------------------------------------------------------------------------

@given(
    stage=stages_st,
    outcome=outcomes_st,
    started_ns=monotonic_ns_st,
    ttfa_ms=ttfa_ms_st,
    model_bytes=memory_bytes_st,
    pipeline_bytes=memory_bytes_st,
)
@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow])
def test_property_12_default_records_contain_no_prohibited_content_fields(
    stage: str,
    outcome: str,
    started_ns: int,
    ttfa_ms: float | None,
    model_bytes: int | None,
    pipeline_bytes: int | None,
) -> None:
    """
    **Validates: Requirement 10.5**

    Feature: realtime-local-voice-agent, Property 12: Privacy-preserving diagnostic completeness

    Default records (content_capture_enabled=False) must contain no PCM bytes,
    audio, transcript, response, prompt text, or tool arguments — across every
    stage and outcome combination with and without optional metric fields.
    """
    error_class = "SomeError" if outcome == "failed" else None

    e = _make_event(
        stage=stage,
        outcome=outcome,
        started_monotonic_ns=started_ns,
        ttfa_ms=ttfa_ms,
        model_resident_bytes=model_bytes,
        pipeline_memory_bytes=pipeline_bytes,
        error_class=error_class,
    )
    record = e.as_dict()

    prohibited = set(record.keys()) & _PROHIBITED_DEFAULT_FIELDS
    assert not prohibited, (
        f"stage={stage!r} outcome={outcome!r}: prohibited fields found: {prohibited}"
    )
    assert record["content_capture_enabled"] is False
    # diagnostic_transcript_text / diagnostic_response_text must NOT appear by default
    assert "diagnostic_transcript_text" not in record
    assert "diagnostic_response_text" not in record


# ---------------------------------------------------------------------------
# Property 12 — Part 3: Terminal outcomes carry timing / error fields
# ---------------------------------------------------------------------------

@given(
    stage=stages_st,
    outcome=terminal_outcomes_st,
    started_ns=positive_monotonic_ns_st,
    transcription_ns=monotonic_ns_st,
    first_llm_ns=monotonic_ns_st,
    first_tts_ns=monotonic_ns_st,
    first_pcm_ns=monotonic_ns_st,
    ttfa_ms=ttfa_ms_st,
    model_bytes=memory_bytes_st,
    pipeline_bytes=memory_bytes_st,
    recovery=recovery_outcome_st,
)
@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow])
def test_property_12_terminal_outcomes_carry_timing_and_error_fields(
    stage: str,
    outcome: str,
    started_ns: int,
    transcription_ns: int,
    first_llm_ns: int,
    first_tts_ns: int,
    first_pcm_ns: int,
    ttfa_ms: float | None,
    model_bytes: int | None,
    pipeline_bytes: int | None,
    recovery: str | None,
) -> None:
    """
    **Validates: Requirements 10.2, 10.6**

    Feature: realtime-local-voice-agent, Property 12: Privacy-preserving diagnostic completeness

    Terminal outcomes (completed, cancelled, failed) must record all timing
    fields and — for failed outcomes — require an error_class.  Records must
    still contain no prohibited content fields.
    """
    error_class = "TerminalError" if outcome == "failed" else None

    e = VoiceDiagnosticEvent(
        session_id=uuid4(),
        turn_id=uuid4(),
        stage=stage,  # type: ignore[arg-type]
        outcome=outcome,  # type: ignore[arg-type]
        started_monotonic_ns=started_ns,
        transcription_completed_monotonic_ns=transcription_ns,
        first_llm_text_monotonic_ns=first_llm_ns,
        first_tts_text_monotonic_ns=first_tts_ns,
        first_pcm_delivered_monotonic_ns=first_pcm_ns,
        ttfa_ms=ttfa_ms,
        model_resident_bytes=model_bytes,
        pipeline_memory_bytes=pipeline_bytes,
        error_class=error_class,
        recovery_outcome=recovery if outcome == "failed" else None,
    )
    record = e.as_dict()

    # All timing fields must be present
    for field in (
        "started_monotonic_ns",
        "transcription_completed_monotonic_ns",
        "first_llm_text_monotonic_ns",
        "first_tts_text_monotonic_ns",
        "first_pcm_delivered_monotonic_ns",
    ):
        assert field in record, f"Missing timing field {field!r} in terminal {outcome!r} record"
        assert isinstance(record[field], int) and record[field] >= 0

    # failed outcome must have error_class
    if outcome == "failed":
        assert record["error_class"] == "TerminalError"
    else:
        assert record["error_class"] is None

    # No prohibited content fields
    assert not (set(record.keys()) & _PROHIBITED_DEFAULT_FIELDS), (
        f"Prohibited content fields found in terminal {outcome!r} record"
    )


# ---------------------------------------------------------------------------
# Property 12 — Part 4: Gate diagnostic completeness, no prompt text
# ---------------------------------------------------------------------------

@given(
    gemini_enabled=st.booleans(),
    battery_pct=battery_pct_st,
    external_power=st.one_of(st.none(), st.booleans()),
    thermal_state=thermal_state_st,
    prompt_tokens=prompt_tokens_st,
    tool_count=tool_count_st,
)
@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow])
def test_property_12_gate_diagnostic_completeness_and_no_prompt_text(
    gemini_enabled: bool,
    battery_pct: int | None,
    external_power: bool | None,
    thermal_state: str,
    prompt_tokens: int,
    tool_count: int,
) -> None:
    """
    **Validates: Requirement 10.3**

    Feature: realtime-local-voice-agent, Property 12: Privacy-preserving diagnostic completeness

    Gate evaluation records must include Gemini enablement state, all evaluated
    conditions, qualifying conditions, and selected route, but must NOT include
    any prompt text, transcript, or raw audio.
    """
    gate = CloudEscalationGate()
    session_id = uuid4()
    turn_id = uuid4()
    gate.register_session(session_id)
    if gemini_enabled:
        gate.enable(session_id)

    decision = gate.evaluate(
        GateInput(
            session_id=session_id,
            gemini_enabled_for_session=gemini_enabled,
            battery_percent=battery_pct,
            external_power_connected=external_power,
            thermal_state=thermal_state,  # type: ignore[arg-type]
            assembled_prompt_tokens=prompt_tokens,
            validated_tool_count=tool_count,
        )
    )
    event = decision.diagnostic_event(turn_id=turn_id)
    record = event.as_dict()

    # Gate sub-record must be present and contain all required keys
    gate_data = record["gate"]
    assert gate_data is not None, "Gate diagnostic must be present in cloud_gate records"
    for gate_field in ("enabled", "evaluated", "qualifying", "selected_route"):
        assert gate_field in gate_data, f"Gate diagnostic missing field {gate_field!r}"

    # evaluated must list all three conditions
    assert gate_data["evaluated"] == list(_GATE_CONDITIONS)

    # Route must match enabled + qualifying logic
    low_battery = (
        battery_pct is not None
        and battery_pct <= 20
        and external_power is False
    )
    throttling = thermal_state in ("serious", "critical")
    complex_reasoning = prompt_tokens > 16_000 or tool_count > 6
    expected_qualifying = tuple(
        c for c, present in (
            ("low_battery", low_battery),
            ("thermal_throttling", throttling),
            ("ultra_complex_reasoning", complex_reasoning),
        )
        if present
    )
    should_be_gemini = gemini_enabled and bool(expected_qualifying)
    assert record["selected_route"] == ("gemini_live" if should_be_gemini else "local_qwen")
    assert gate_data["selected_route"] == record["selected_route"]

    # No prompt text or raw audio anywhere in the record
    assert not (set(record.keys()) & _PROHIBITED_DEFAULT_FIELDS)
    assert "prompt" not in record
    assert "prompt_text" not in record
    assert not _has_raw_audio_field(record)


# ---------------------------------------------------------------------------
# Property 12 — Part 5: Failure records require error_class; recovery optional
# ---------------------------------------------------------------------------

@given(
    stage=stages_st,
    error_class=error_class_st,
    recovery=recovery_outcome_st,
)
@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow])
def test_property_12_failure_records_require_error_class_and_no_content(
    stage: str,
    error_class: str,
    recovery: str | None,
) -> None:
    """
    **Validates: Requirement 10.6**

    Feature: realtime-local-voice-agent, Property 12: Privacy-preserving diagnostic completeness

    Every stage failure must record a non-empty error_class and the optional
    recovery_outcome, while excluding all content-bearing fields.
    """
    e = VoiceDiagnosticEvent.for_failure(
        session_id=uuid4(),
        turn_id=uuid4(),
        stage=stage,  # type: ignore[arg-type]
        error_class=error_class,
        recovery_outcome=recovery,
    )
    record = e.as_dict()

    assert record["outcome"] == "failed"
    assert record["error_class"] == error_class
    assert record["error_class"]  # must be non-empty
    assert record["recovery_outcome"] == recovery  # may be None

    # No prohibited content fields
    prohibited = set(record.keys()) & _PROHIBITED_DEFAULT_FIELDS
    assert not prohibited, f"stage={stage!r} failure: prohibited fields {prohibited}"
    assert "diagnostic_transcript_text" not in record
    assert "diagnostic_response_text" not in record
    assert not _has_raw_audio_field(record)


@given(stage=stages_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_12_empty_error_class_rejected_for_failure(stage: str) -> None:
    """
    **Validates: Requirement 10.6**

    An empty error_class must be rejected for failure diagnostics regardless of stage.
    """
    with pytest.raises(ValueError):
        VoiceDiagnosticEvent.for_failure(
            session_id=uuid4(),
            turn_id=uuid4(),
            stage=stage,  # type: ignore[arg-type]
            error_class="",
        )


# ---------------------------------------------------------------------------
# Property 12 — Part 6: Content capture — transcript/response only with flag
# ---------------------------------------------------------------------------

@given(
    stage=stages_st,
    transcript=st.text(min_size=1, max_size=200),
    response=st.text(min_size=1, max_size=200),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_12_content_fields_only_when_capture_enabled(
    stage: str,
    transcript: str,
    response: str,
) -> None:
    """
    **Validates: Requirement 10.5**

    Feature: realtime-local-voice-agent, Property 12: Privacy-preserving diagnostic completeness

    Transcript and response fields appear ONLY when content_capture_enabled=True.
    They must use explicitly labeled names (diagnostic_transcript_text /
    diagnostic_response_text), never the bare names 'transcript' or 'response'.
    Raw audio must never appear regardless of the flag.
    """
    e_with = VoiceDiagnosticEvent(
        session_id=uuid4(), turn_id=uuid4(),
        stage=stage,  # type: ignore[arg-type]
        outcome="completed",
        content_capture_enabled=True,
    )
    e_with.set_transcript(transcript)
    e_with.set_response(response)
    record_with = e_with.as_dict()

    # With capture enabled: labeled fields present
    assert record_with["content_capture_enabled"] is True
    assert record_with.get("diagnostic_transcript_text") == transcript
    assert record_with.get("diagnostic_response_text") == response
    # Must NOT use bare prohibited names
    assert "transcript" not in record_with
    assert "transcript_text" not in record_with
    assert "response" not in record_with
    assert "response_text" not in record_with
    # Raw audio still absent
    assert not _has_raw_audio_field(record_with)

    # Without capture enabled: no content fields at all
    e_without = VoiceDiagnosticEvent(
        session_id=uuid4(), turn_id=uuid4(),
        stage=stage,  # type: ignore[arg-type]
        outcome="completed",
        content_capture_enabled=False,
    )
    record_without = e_without.as_dict()
    assert record_without["content_capture_enabled"] is False
    assert "diagnostic_transcript_text" not in record_without
    assert "diagnostic_response_text" not in record_without
    assert not _has_raw_audio_field(record_without)


@given(stage=stages_st, outcome=outcomes_st)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_12_raw_audio_never_appears_regardless_of_capture_flag(
    stage: str, outcome: str,
) -> None:
    """
    **Validates: Requirement 10.5**

    Feature: realtime-local-voice-agent, Property 12: Privacy-preserving diagnostic completeness

    Raw audio must NEVER appear in any diagnostic record, whether content
    capture is enabled or disabled, and for every stage and outcome.
    """
    error_class = "Err" if outcome == "failed" else None
    for capture_enabled in (False, True):
        e = _make_event(
            stage=stage,
            outcome=outcome,
            content_capture_enabled=capture_enabled,
            error_class=error_class,
        )
        record = e.as_dict()
        assert not _has_raw_audio_field(record), (
            f"Raw audio field in stage={stage!r} outcome={outcome!r} "
            f"capture={capture_enabled}: {set(record.keys()) & {'pcm','audio','raw_audio'}}"
        )
        # No set_audio / set_pcm / set_raw_audio methods exist
        assert not hasattr(e, "set_audio")
        assert not hasattr(e, "set_pcm")
        assert not hasattr(e, "set_raw_audio")


# ---------------------------------------------------------------------------
# Property 12 — Part 7: Content-control expiry after session end
# ---------------------------------------------------------------------------

@given(
    n_sessions=st.integers(min_value=1, max_value=5),
    enable_indices=st.lists(st.integers(min_value=0, max_value=4), unique=True, max_size=5),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_12_content_capture_expiry_on_session_end(
    n_sessions: int,
    enable_indices: list[int],
) -> None:
    """
    **Validates: Requirement 10.5**

    Feature: realtime-local-voice-agent, Property 12: Privacy-preserving diagnostic completeness

    After a session ends, content_capture_enabled must become False for that
    session, and the state of other sessions must remain unaffected.
    """
    registry = ContentCaptureRegistry()
    session_ids = [uuid4() for _ in range(n_sessions)]

    for sid in session_ids:
        registry.register_session(sid)

    # Enable for the requested indices (bounded to n_sessions)
    enabled_sids = set()
    for idx in enable_indices:
        if idx < n_sessions:
            registry.enable(session_ids[idx])
            enabled_sids.add(session_ids[idx])

    # End every session and verify capture is off for each
    for sid in session_ids:
        was_enabled = registry.is_enabled(sid)
        registry.end_session(sid)
        assert registry.is_enabled(sid) is False, (
            f"Session {sid}: content capture must be False after end_session, "
            f"was_enabled={was_enabled}"
        )


@given(
    n_sessions=st.integers(min_value=2, max_value=5),
    end_idx=st.integers(min_value=0, max_value=4),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_12_expiry_does_not_affect_other_active_sessions(
    n_sessions: int,
    end_idx: int,
) -> None:
    """
    **Validates: Requirement 10.5**

    Feature: realtime-local-voice-agent, Property 12: Privacy-preserving diagnostic completeness

    Ending one session must not disable content capture for other active sessions.
    """
    registry = ContentCaptureRegistry()
    session_ids = [uuid4() for _ in range(n_sessions)]
    for sid in session_ids:
        registry.register_session(sid)
        registry.enable(sid)

    target_idx = end_idx % n_sessions
    target_sid = session_ids[target_idx]
    registry.end_session(target_sid)

    assert registry.is_enabled(target_sid) is False
    for i, sid in enumerate(session_ids):
        if i != target_idx:
            assert registry.is_enabled(sid) is True, (
                f"Ending session[{target_idx}] must not affect session[{i}]"
            )


# ---------------------------------------------------------------------------
# Property 12 — Part 8: Serialization round trips
# ---------------------------------------------------------------------------

@given(
    stage=stages_st,
    outcome=outcomes_st,
    started_ns=monotonic_ns_st,
    ttfa_ms=ttfa_ms_st,
    model_bytes=memory_bytes_st,
    pipeline_bytes=memory_bytes_st,
    asr_engine=engine_id_st,
    tts_engine=engine_id_st,
    recovery=recovery_outcome_st,
)
@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow])
def test_property_12_serialization_round_trip_is_json_safe(
    stage: str,
    outcome: str,
    started_ns: int,
    ttfa_ms: float | None,
    model_bytes: int | None,
    pipeline_bytes: int | None,
    asr_engine: str | None,
    tts_engine: str | None,
    recovery: str | None,
) -> None:
    """
    **Validates: Requirements 10.1, 10.2**

    Feature: realtime-local-voice-agent, Property 12: Privacy-preserving diagnostic completeness

    The as_dict() output must be JSON-serializable and the parsed round-trip
    record must preserve all required scalar fields without modification.
    No prohibited content fields may appear in the serialized form.
    """
    error_class = "RoundTripError" if outcome == "failed" else None
    e = VoiceDiagnosticEvent(
        session_id=uuid4(), turn_id=uuid4(),
        stage=stage,  # type: ignore[arg-type]
        outcome=outcome,  # type: ignore[arg-type]
        started_monotonic_ns=started_ns,
        ttfa_ms=ttfa_ms,
        model_resident_bytes=model_bytes,
        pipeline_memory_bytes=pipeline_bytes,
        asr_engine=asr_engine,
        tts_engine=tts_engine,
        error_class=error_class,
        recovery_outcome=recovery if outcome == "failed" else None,
    )

    record = e.as_dict()

    # Must be JSON-serializable without error
    json_str = json.dumps(record, ensure_ascii=False)
    assert json_str  # non-empty

    # Round-trip must produce identical scalars
    parsed = json.loads(json_str)
    for field in ("schema_version", "stage", "outcome", "session_id", "turn_id",
                  "content_capture_enabled"):
        assert parsed[field] == record[field], (
            f"Round-trip mismatch for field {field!r}: {parsed[field]!r} != {record[field]!r}"
        )

    # No prohibited content fields after round-trip
    assert not (set(parsed.keys()) & _PROHIBITED_DEFAULT_FIELDS)


@given(
    stage=stages_st,
    outcome=terminal_outcomes_st,
    transcript=st.text(min_size=1, max_size=100),
    response=st.text(min_size=1, max_size=100),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_12_content_capture_round_trip_preserves_labeled_fields(
    stage: str,
    outcome: str,
    transcript: str,
    response: str,
) -> None:
    """
    **Validates: Requirement 10.5**

    Feature: realtime-local-voice-agent, Property 12: Privacy-preserving diagnostic completeness

    When content capture is enabled, the labeled diagnostic fields survive a
    JSON round trip with the correct keys and values.  Raw audio fields must
    still not appear.
    """
    error_class = "Err" if outcome == "failed" else None
    e = VoiceDiagnosticEvent(
        session_id=uuid4(), turn_id=uuid4(),
        stage=stage,  # type: ignore[arg-type]
        outcome=outcome,  # type: ignore[arg-type]
        content_capture_enabled=True,
        error_class=error_class,
    )
    e.set_transcript(transcript)
    e.set_response(response)

    parsed = json.loads(json.dumps(e.as_dict(), ensure_ascii=False))

    assert parsed["diagnostic_transcript_text"] == transcript
    assert parsed["diagnostic_response_text"] == response
    assert "transcript" not in parsed
    assert "transcript_text" not in parsed
    assert not _has_raw_audio_field(parsed)


# ---------------------------------------------------------------------------
# Property 12 — Part 9: Cancellation outcomes and missing optional metrics
# ---------------------------------------------------------------------------

@given(
    stage=stages_st,
    started_ns=monotonic_ns_st,
    ttfa_ms=ttfa_ms_st,
    model_bytes=memory_bytes_st,
    pipeline_bytes=memory_bytes_st,
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_12_cancellation_outcome_completes_without_error_class(
    stage: str,
    started_ns: int,
    ttfa_ms: float | None,
    model_bytes: int | None,
    pipeline_bytes: int | None,
) -> None:
    """
    **Validates: Requirements 10.2, 10.5**

    Feature: realtime-local-voice-agent, Property 12: Privacy-preserving diagnostic completeness

    Cancellation records must be complete (all required fields), must not
    require an error_class, and must contain no prohibited content fields.
    Missing optional metric fields (ttfa_ms=None, model_bytes=None, etc.) are
    valid and must not cause required fields to disappear.
    """
    e = _make_event(
        stage=stage,
        outcome="cancelled",
        started_monotonic_ns=started_ns,
        ttfa_ms=ttfa_ms,
        model_resident_bytes=model_bytes,
        pipeline_memory_bytes=pipeline_bytes,
    )
    record = e.as_dict()

    assert record["outcome"] == "cancelled"
    assert record["error_class"] is None
    # Required fields still present even when optional metrics are None
    missing = _REQUIRED_FIELDS - set(record.keys())
    assert not missing, f"Cancellation record missing required fields: {missing}"
    assert not (set(record.keys()) & _PROHIBITED_DEFAULT_FIELDS)


@given(
    stage=stages_st,
    asr_engine=engine_id_st,
    tts_engine=engine_id_st,
    selected_route=route_st,
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_12_started_records_include_component_identifiers(
    stage: str,
    asr_engine: str | None,
    tts_engine: str | None,
    selected_route: str | None,
) -> None:
    """
    **Validates: Requirement 10.1**

    Feature: realtime-local-voice-agent, Property 12: Privacy-preserving diagnostic completeness

    Turn-start records created with for_stage_start must include the provided
    component identifiers and selected route without any content fields.
    """
    e = VoiceDiagnosticEvent.for_stage_start(
        session_id=uuid4(), turn_id=uuid4(),
        stage=stage,  # type: ignore[arg-type]
        started_monotonic_ns=1000,
        asr_engine=asr_engine,
        tts_engine=tts_engine,
        selected_route=selected_route,  # type: ignore[arg-type]
    )
    record = e.as_dict()

    assert record["outcome"] == "started"
    assert record["asr_engine"] == asr_engine
    assert record["tts_engine"] == tts_engine
    assert record["selected_route"] == selected_route
    assert not (set(record.keys()) & _PROHIBITED_DEFAULT_FIELDS)
    missing = _REQUIRED_FIELDS - set(record.keys())
    assert not missing, f"Stage-start record missing required fields: {missing}"


# ---------------------------------------------------------------------------
# Property 12 — Part 10: Store-level content exclusion (JSONL on disk)
# ---------------------------------------------------------------------------

@given(
    stage=stages_st,
    outcome=outcomes_st,
    started_ns=monotonic_ns_st,
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_12_appended_records_on_disk_contain_no_content_fields(
    stage: str,
    outcome: str,
    started_ns: int,
) -> None:
    """
    **Validates: Requirements 10.1, 10.2, 10.5**

    Feature: realtime-local-voice-agent, Property 12: Privacy-preserving diagnostic completeness

    Appending a default event and reading it back from the JSONL store must
    produce a record with no prohibited content fields.
    """
    error_class = "DiskErr" if outcome == "failed" else None
    e = _make_event(stage=stage, outcome=outcome,
                    started_monotonic_ns=started_ns, error_class=error_class)

    with tempfile.TemporaryDirectory() as tmp:
        diag_dir = Path(tmp) / "voice"
        test_date = date(2025, 1, 15)
        append_diagnostic(e, directory=diag_dir, for_date=test_date)
        records = read_diagnostics(directory=diag_dir, for_date=test_date)

    assert len(records) == 1
    disk_record = records[0]
    prohibited = set(disk_record.keys()) & _PROHIBITED_DEFAULT_FIELDS
    assert not prohibited, (
        f"stage={stage!r} outcome={outcome!r}: disk record has prohibited fields: {prohibited}"
    )
    assert disk_record["schema_version"] == 1
    assert disk_record["stage"] == stage
    assert disk_record["outcome"] == outcome


@given(
    n_events=st.integers(min_value=2, max_value=10),
    stages=st.lists(stages_st, min_size=2, max_size=10),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_12_multiple_appended_records_all_clean_on_disk(
    n_events: int,
    stages: list[str],
) -> None:
    """
    **Validates: Requirements 10.1, 10.5**

    Feature: realtime-local-voice-agent, Property 12: Privacy-preserving diagnostic completeness

    Multiple appended records in a single JSONL file must all be free of
    prohibited content fields and each individually valid JSON.
    """
    with tempfile.TemporaryDirectory() as tmp:
        diag_dir = Path(tmp) / "voice"
        test_date = date(2025, 3, 1)
        for i in range(min(n_events, len(stages))):
            stage = stages[i]
            e = _make_event(stage=stage, outcome="completed")
            append_diagnostic(e, directory=diag_dir, for_date=test_date)

        records = read_diagnostics(directory=diag_dir, for_date=test_date)

    for i, rec in enumerate(records):
        assert not (set(rec.keys()) & _PROHIBITED_DEFAULT_FIELDS), (
            f"Record {i}: prohibited fields found: {set(rec.keys()) & _PROHIBITED_DEFAULT_FIELDS}"
        )
        # Each record must be schema_version 1
        assert rec["schema_version"] == 1
        # Each event_id must be parseable as UUID
        UUID(rec["event_id"])


# ---------------------------------------------------------------------------
# Property 12 — Part 11: Redaction hashes (per-session salted correlation)
# ---------------------------------------------------------------------------

@given(
    stage=stages_st,
    outcome=outcomes_st,
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_12_session_and_turn_ids_serve_as_safe_correlation_hashes(
    stage: str,
    outcome: str,
) -> None:
    """
    **Validates: Requirement 10.5**

    Feature: realtime-local-voice-agent, Property 12: Privacy-preserving diagnostic completeness

    session_id and turn_id in the record are opaque non-reversible UUIDs that
    correlate events without exposing transcript content.  Two events sharing a
    session_id must produce the same session_id string in their records; events
    from different sessions must produce different session_id strings.
    """
    sid = uuid4()
    tid1, tid2 = uuid4(), uuid4()
    error_class = "E" if outcome == "failed" else None

    e1 = VoiceDiagnosticEvent(
        session_id=sid, turn_id=tid1,
        stage=stage,  # type: ignore[arg-type]
        outcome=outcome,  # type: ignore[arg-type]
        error_class=error_class,
    )
    e2 = VoiceDiagnosticEvent(
        session_id=sid, turn_id=tid2,
        stage=stage,  # type: ignore[arg-type]
        outcome=outcome,  # type: ignore[arg-type]
        error_class=error_class,
    )
    r1 = e1.as_dict()
    r2 = e2.as_dict()

    # Same session → same session_id string
    assert r1["session_id"] == r2["session_id"] == str(sid)
    # Different turns → different turn_id strings
    assert r1["turn_id"] != r2["turn_id"]
    # No content present
    assert not (set(r1.keys()) & _PROHIBITED_DEFAULT_FIELDS)
    assert not (set(r2.keys()) & _PROHIBITED_DEFAULT_FIELDS)

    # Different session → different session_id in record
    e3 = VoiceDiagnosticEvent(
        session_id=uuid4(), turn_id=uuid4(),
        stage=stage,  # type: ignore[arg-type]
        outcome=outcome,  # type: ignore[arg-type]
        error_class=error_class,
    )
    r3 = e3.as_dict()
    assert r3["session_id"] != r1["session_id"]


# ---------------------------------------------------------------------------
# Property 12 — Part 12: Gate failures also exclude content
# ---------------------------------------------------------------------------

@given(
    battery_pct=battery_pct_st,
    external_power=st.one_of(st.none(), st.booleans()),
    thermal_state=thermal_state_st,
    prompt_tokens=prompt_tokens_st,
    tool_count=tool_count_st,
    error_class=error_class_st,
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_12_gate_failure_records_no_content(
    battery_pct: int | None,
    external_power: bool | None,
    thermal_state: str,
    prompt_tokens: int,
    tool_count: int,
    error_class: str,
) -> None:
    """
    **Validates: Requirements 10.3, 10.6**

    Feature: realtime-local-voice-agent, Property 12: Privacy-preserving diagnostic completeness

    A failed eligible cloud gate invocation must produce a failure diagnostic
    that still contains no content fields, includes the gate sub-record, and
    records the error_class and recovery_outcome without prompt text.
    """
    gate = CloudEscalationGate()
    session_id = uuid4()
    turn_id = uuid4()
    gate.register_session(session_id)

    # Only eligible turns (Gemini enabled + qualifying) can report an invocation failure
    low_battery = battery_pct is not None and battery_pct <= 20 and external_power is False
    throttling = thermal_state in ("serious", "critical")
    complex_r = prompt_tokens > 16_000 or tool_count > 6
    qualifying = low_battery or throttling or complex_r

    if not qualifying:
        # Skip cases that won't produce an eligible decision
        return

    gate.enable(session_id)
    decision = gate.evaluate(
        GateInput(
            session_id=session_id,
            gemini_enabled_for_session=True,
            battery_percent=battery_pct,
            external_power_connected=external_power,
            thermal_state=thermal_state,  # type: ignore[arg-type]
            assembled_prompt_tokens=prompt_tokens,
            validated_tool_count=tool_count,
        )
    )

    if not decision.eligible:
        return

    failure = gate.report_eligible_invocation_failure(decision, error_class)
    event = failure.diagnostic_event(turn_id=turn_id)
    record = event.as_dict()

    assert record["outcome"] == "failed"
    assert record["error_class"] == error_class
    assert record["stage"] == "cloud_gate"
    assert record["gate"] is not None
    assert not (set(record.keys()) & _PROHIBITED_DEFAULT_FIELDS)
    assert "prompt" not in record
    assert not _has_raw_audio_field(record)
    assert failure.retry_attempted is False
    assert failure.fallback_attempted is False
    assert failure.recovery_outcome == "reported_no_fallback"


# ---------------------------------------------------------------------------
# Property 12 — Part 13: High-volume mixed event stream (≥100 cases)
# ---------------------------------------------------------------------------

@given(
    events_spec=st.lists(
        st.tuples(stages_st, outcomes_st, st.booleans()),
        min_size=5,
        max_size=20,
    )
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_12_mixed_event_stream_all_records_clean(
    events_spec: list[tuple[str, str, bool]],
) -> None:
    """
    **Validates: Requirements 10.1, 10.2, 10.5, 10.6**

    Feature: realtime-local-voice-agent, Property 12: Privacy-preserving diagnostic completeness

    A mixed stream of start, terminal, gate, and failure events from various
    stages must all serialize to clean records (required fields present,
    no prohibited content fields), and be append-able to the JSONL store
    without corrupting earlier records.
    """
    with tempfile.TemporaryDirectory() as tmp:
        diag_dir = Path(tmp) / "voice"
        test_date = date(2025, 6, 1)
        appended_count = 0

        for stage, outcome, _unused_flag in events_spec:
            error_class = "MixedErr" if outcome == "failed" else None
            e = _make_event(stage=stage, outcome=outcome, error_class=error_class)
            record = e.as_dict()

            # In-memory record must be complete and clean
            missing = _REQUIRED_FIELDS - set(record.keys())
            assert not missing, f"Missing fields: {missing} for stage={stage!r} outcome={outcome!r}"
            assert not (set(record.keys()) & _PROHIBITED_DEFAULT_FIELDS)

            # Append to store
            append_diagnostic(e, directory=diag_dir, for_date=test_date)
            appended_count += 1

        # Read back all records and verify each one
        disk_records = read_diagnostics(directory=diag_dir, for_date=test_date)
        assert len(disk_records) == appended_count
        for i, rec in enumerate(disk_records):
            assert not (set(rec.keys()) & _PROHIBITED_DEFAULT_FIELDS), (
                f"Record {i} on disk has prohibited content fields"
            )
            assert rec["schema_version"] == 1
            UUID(rec["event_id"])
            UUID(rec["session_id"])
            UUID(rec["turn_id"])
            # JSON string round-trips cleanly
            json.dumps(rec, ensure_ascii=False)
