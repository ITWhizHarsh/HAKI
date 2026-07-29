"""Unit tests for the versioned voice diagnostics schema, redaction, and store.

Validates: Requirements 10.1–10.6
Focused on: serialize start/completion/cancellation/failure paths; assert default
fields exclude every content-bearing/raw-byte field; atomic local store; session-
scoped content capture defaults false and expires at session end; never raw audio.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from datetime import date
from pathlib import Path
from uuid import uuid4

import pytest

from core.voice.diagnostics import (
    ContentCaptureRegistry,
    DiagnosticStoreError,
    GateDiagnostic,
    VoiceDiagnosticEvent,
    append_diagnostic,
    read_diagnostics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    stage: str = "asr",
    outcome: str = "started",
    content_capture_enabled: bool = False,
    **kwargs,
) -> VoiceDiagnosticEvent:
    return VoiceDiagnosticEvent(
        session_id=uuid4(),
        turn_id=uuid4(),
        stage=stage,  # type: ignore[arg-type]
        outcome=outcome,  # type: ignore[arg-type]
        content_capture_enabled=content_capture_enabled,
        **kwargs,
    )


def _all_stages() -> list[str]:
    return [
        "asr", "ipc", "pipecat", "voice_processing", "local_llm",
        "tool_call", "local_tts", "memory_budget", "cloud_gate", "playback",
    ]


_CONTENT_FIELD_NAMES = {
    "pcm", "pcm_bytes", "audio", "audio_bytes", "raw_audio",
    "transcript", "transcript_text", "response", "response_text",
    "prompt", "prompt_text", "tool_arguments", "tool_results", "tool_content",
}


def _has_content_field(record: dict) -> bool:
    """Return True if any prohibited content field is present in the serialized record."""
    return bool(set(record.keys()) & _CONTENT_FIELD_NAMES)


# ---------------------------------------------------------------------------
# Schema version and required identifier fields
# ---------------------------------------------------------------------------

class TestSchemaStructure:
    def test_schema_version_is_1(self) -> None:
        e = _make_event()
        assert e.as_dict()["schema_version"] == 1

    def test_event_id_is_a_uuid_string(self) -> None:
        e = _make_event()
        d = e.as_dict()
        assert isinstance(d["event_id"], str)
        # Must be parseable as UUID
        import uuid
        uuid.UUID(d["event_id"])

    def test_session_and_turn_ids_are_strings(self) -> None:
        sid, tid = uuid4(), uuid4()
        e = VoiceDiagnosticEvent(session_id=sid, turn_id=tid, stage="asr", outcome="started")
        d = e.as_dict()
        assert d["session_id"] == str(sid)
        assert d["turn_id"] == str(tid)

    def test_all_required_fields_present_in_default_record(self) -> None:
        required = {
            "schema_version", "event_id", "session_id", "turn_id",
            "stage", "outcome", "started_monotonic_ns",
            "transcription_completed_monotonic_ns", "first_llm_text_monotonic_ns",
            "first_tts_text_monotonic_ns", "first_pcm_delivered_monotonic_ns",
            "ttfa_ms", "selected_route", "asr_engine", "tts_engine",
            "model_resident_bytes", "pipeline_memory_bytes", "gate",
            "error_class", "recovery_outcome", "content_capture_enabled",
        }
        e = _make_event()
        assert required <= set(e.as_dict().keys())


# ---------------------------------------------------------------------------
# All pipeline stages are accepted
# ---------------------------------------------------------------------------

class TestAllStages:
    @pytest.mark.parametrize("stage", _all_stages())
    def test_every_stage_is_valid(self, stage: str) -> None:
        e = _make_event(stage=stage)
        assert e.as_dict()["stage"] == stage

    def test_invalid_stage_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid stage"):
            _make_event(stage="unknown_stage")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# All outcome values are accepted
# ---------------------------------------------------------------------------

class TestOutcomes:
    @pytest.mark.parametrize("outcome", ["started", "completed", "cancelled", "failed", "rejected"])
    def test_every_outcome_is_valid(self, outcome: str) -> None:
        e = _make_event(outcome=outcome)
        assert e.as_dict()["outcome"] == outcome

    def test_invalid_outcome_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid outcome"):
            _make_event(outcome="unknown")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Default redaction: no content-bearing fields in default records
# ---------------------------------------------------------------------------

class TestDefaultRedaction:
    @pytest.mark.parametrize("stage", _all_stages())
    def test_default_record_contains_no_content_fields(self, stage: str) -> None:
        """Default serialization must exclude every content-bearing/raw-byte field."""
        e = _make_event(stage=stage, outcome="completed")
        d = e.as_dict()
        assert not _has_content_field(d), (
            f"Stage {stage!r} record contains prohibited field(s): "
            f"{set(d.keys()) & _CONTENT_FIELD_NAMES}"
        )

    def test_start_path_excludes_content(self) -> None:
        e = VoiceDiagnosticEvent.for_stage_start(
            session_id=uuid4(), turn_id=uuid4(), stage="asr",
            started_monotonic_ns=1000, asr_engine="qwen3_asr_coreml",
        )
        assert not _has_content_field(e.as_dict())

    def test_completion_path_excludes_content(self) -> None:
        e = _make_event(
            stage="playback", outcome="completed",
            started_monotonic_ns=100,
            first_pcm_delivered_monotonic_ns=200,
            ttfa_ms=50.5,
            model_resident_bytes=1_000_000,
            pipeline_memory_bytes=2_000_000,
        )
        assert not _has_content_field(e.as_dict())

    def test_cancellation_path_excludes_content(self) -> None:
        e = _make_event(stage="local_llm", outcome="cancelled")
        assert not _has_content_field(e.as_dict())

    def test_failure_path_excludes_content(self) -> None:
        e = VoiceDiagnosticEvent.for_failure(
            session_id=uuid4(), turn_id=uuid4(), stage="local_tts",
            error_class="TTSSynthesisError", recovery_outcome="reported_no_fallback",
        )
        assert not _has_content_field(e.as_dict())

    def test_content_capture_enabled_defaults_false(self) -> None:
        e = _make_event()
        assert e.as_dict()["content_capture_enabled"] is False

    def test_set_transcript_without_flag_raises(self) -> None:
        e = _make_event(stage="asr", outcome="completed")
        with pytest.raises(ValueError, match="content_capture_enabled"):
            e.set_transcript("hello world")

    def test_set_response_without_flag_raises(self) -> None:
        e = _make_event(stage="local_llm", outcome="completed")
        with pytest.raises(ValueError, match="content_capture_enabled"):
            e.set_response("I can help you with that.")


# ---------------------------------------------------------------------------
# Session-scoped content capture
# ---------------------------------------------------------------------------

class TestContentCapture:
    def test_content_fields_appear_when_capture_enabled(self) -> None:
        e = _make_event(stage="asr", outcome="completed", content_capture_enabled=True)
        e.set_transcript("Kal meeting reschedule kar do")
        d = e.as_dict()
        assert d["content_capture_enabled"] is True
        assert d["diagnostic_transcript_text"] == "Kal meeting reschedule kar do"

    def test_response_field_appears_when_capture_enabled(self) -> None:
        e = _make_event(stage="local_llm", outcome="completed", content_capture_enabled=True)
        e.set_response("Meeting rescheduled for 3pm.")
        d = e.as_dict()
        assert d["diagnostic_response_text"] == "Meeting rescheduled for 3pm."

    def test_content_fields_absent_when_capture_disabled(self) -> None:
        e = _make_event(stage="asr", outcome="completed", content_capture_enabled=False)
        d = e.as_dict()
        assert "diagnostic_transcript_text" not in d
        assert "diagnostic_response_text" not in d

    def test_raw_audio_cannot_be_set_regardless_of_flag(self) -> None:
        """Raw audio capture must never be possible — even with content capture enabled."""
        e = _make_event(stage="asr", outcome="completed", content_capture_enabled=True)
        # No raw audio method should exist
        assert not hasattr(e, "set_audio")
        assert not hasattr(e, "set_pcm")
        assert not hasattr(e, "set_raw_audio")
        d = e.as_dict()
        # Verify raw audio fields are never in dict
        raw_fields = {"pcm", "pcm_bytes", "audio", "audio_bytes", "raw_audio"}
        assert not (set(d.keys()) & raw_fields)

    def test_transcript_set_without_flag_in_init_raises(self) -> None:
        with pytest.raises(ValueError):
            VoiceDiagnosticEvent(
                session_id=uuid4(), turn_id=uuid4(), stage="asr", outcome="completed",
                content_capture_enabled=False,
                _transcript_text="hello",
            )


# ---------------------------------------------------------------------------
# ContentCaptureRegistry
# ---------------------------------------------------------------------------

class TestContentCaptureRegistry:
    def test_new_session_defaults_false(self) -> None:
        reg = ContentCaptureRegistry()
        sid = uuid4()
        reg.register_session(sid)
        assert reg.is_enabled(sid) is False

    def test_enable_returns_true(self) -> None:
        reg = ContentCaptureRegistry()
        sid = uuid4()
        reg.register_session(sid)
        result = reg.enable(sid)
        assert result is True
        assert reg.is_enabled(sid) is True

    def test_disable_returns_false(self) -> None:
        reg = ContentCaptureRegistry()
        sid = uuid4()
        reg.register_session(sid)
        reg.enable(sid)
        result = reg.disable(sid)
        assert result is False
        assert reg.is_enabled(sid) is False

    def test_session_isolation(self) -> None:
        """Enabling for one session must not affect another."""
        reg = ContentCaptureRegistry()
        sid1, sid2 = uuid4(), uuid4()
        reg.register_session(sid1)
        reg.register_session(sid2)
        reg.enable(sid1)
        assert reg.is_enabled(sid1) is True
        assert reg.is_enabled(sid2) is False

    def test_end_session_expires_content_capture(self) -> None:
        """Content capture must expire (become false) when the session ends."""
        reg = ContentCaptureRegistry()
        sid = uuid4()
        reg.register_session(sid)
        reg.enable(sid)
        assert reg.is_enabled(sid) is True
        reg.end_session(sid)
        assert reg.is_enabled(sid) is False

    def test_enable_on_inactive_session_raises(self) -> None:
        reg = ContentCaptureRegistry()
        with pytest.raises(ValueError, match="inactive session"):
            reg.enable(uuid4())

    def test_multiple_sessions_independent(self) -> None:
        reg = ContentCaptureRegistry()
        sids = [uuid4() for _ in range(5)]
        for sid in sids:
            reg.register_session(sid)
        reg.enable(sids[2])
        for i, sid in enumerate(sids):
            assert reg.is_enabled(sid) is (i == 2)

    def test_re_register_resets_to_false(self) -> None:
        reg = ContentCaptureRegistry()
        sid = uuid4()
        reg.register_session(sid)
        reg.enable(sid)
        reg.register_session(sid)  # re-register (new session same ID)
        assert reg.is_enabled(sid) is False


# ---------------------------------------------------------------------------
# for_stage_start factory — Requirement 10.1
# ---------------------------------------------------------------------------

class TestStageStartFactory:
    def test_creates_started_outcome(self) -> None:
        e = VoiceDiagnosticEvent.for_stage_start(
            session_id=uuid4(), turn_id=uuid4(), stage="asr",
            started_monotonic_ns=9999,
        )
        assert e.outcome == "started"
        assert e.started_monotonic_ns == 9999

    def test_records_asr_and_tts_engine(self) -> None:
        e = VoiceDiagnosticEvent.for_stage_start(
            session_id=uuid4(), turn_id=uuid4(), stage="asr",
            started_monotonic_ns=0,
            asr_engine="qwen3_asr_coreml",
            tts_engine="xtts_v2",
        )
        d = e.as_dict()
        assert d["asr_engine"] == "qwen3_asr_coreml"
        assert d["tts_engine"] == "xtts_v2"

    def test_records_selected_route(self) -> None:
        e = VoiceDiagnosticEvent.for_stage_start(
            session_id=uuid4(), turn_id=uuid4(), stage="cloud_gate",
            started_monotonic_ns=0, selected_route="local_qwen",
        )
        assert e.as_dict()["selected_route"] == "local_qwen"

    def test_no_content_in_start_record(self) -> None:
        e = VoiceDiagnosticEvent.for_stage_start(
            session_id=uuid4(), turn_id=uuid4(), stage="local_llm",
            started_monotonic_ns=1234,
        )
        assert not _has_content_field(e.as_dict())


# ---------------------------------------------------------------------------
# for_failure factory — Requirement 10.6
# ---------------------------------------------------------------------------

class TestFailureFactory:
    def test_creates_failed_outcome(self) -> None:
        e = VoiceDiagnosticEvent.for_failure(
            session_id=uuid4(), turn_id=uuid4(), stage="local_llm",
            error_class="MLXLoadError",
        )
        assert e.outcome == "failed"
        assert e.error_class == "MLXLoadError"

    def test_records_recovery_outcome(self) -> None:
        e = VoiceDiagnosticEvent.for_failure(
            session_id=uuid4(), turn_id=uuid4(), stage="local_tts",
            error_class="TTSError", recovery_outcome="reported_no_fallback",
        )
        assert e.recovery_outcome == "reported_no_fallback"

    def test_failure_with_empty_error_class_raises(self) -> None:
        with pytest.raises(ValueError):
            VoiceDiagnosticEvent.for_failure(
                session_id=uuid4(), turn_id=uuid4(), stage="asr",
                error_class="",
            )

    @pytest.mark.parametrize("stage", _all_stages())
    def test_failure_on_every_stage_excludes_content(self, stage: str) -> None:
        e = VoiceDiagnosticEvent.for_failure(
            session_id=uuid4(), turn_id=uuid4(), stage=stage,  # type: ignore[arg-type]
            error_class="SomeError",
        )
        assert not _has_content_field(e.as_dict())


# ---------------------------------------------------------------------------
# Timing and resource fields — Requirement 10.2
# ---------------------------------------------------------------------------

class TestTimingAndResourceFields:
    def test_all_timing_fields_present_and_serializable(self) -> None:
        e = VoiceDiagnosticEvent(
            session_id=uuid4(), turn_id=uuid4(), stage="playback", outcome="completed",
            started_monotonic_ns=100,
            transcription_completed_monotonic_ns=200,
            first_llm_text_monotonic_ns=300,
            first_tts_text_monotonic_ns=400,
            first_pcm_delivered_monotonic_ns=500,
            ttfa_ms=123.45,
        )
        d = e.as_dict()
        assert d["started_monotonic_ns"] == 100
        assert d["transcription_completed_monotonic_ns"] == 200
        assert d["first_llm_text_monotonic_ns"] == 300
        assert d["first_tts_text_monotonic_ns"] == 400
        assert d["first_pcm_delivered_monotonic_ns"] == 500
        assert d["ttfa_ms"] == 123.45

    def test_memory_fields_serialized(self) -> None:
        e = _make_event(
            stage="memory_budget", outcome="failed",
            model_resident_bytes=2_500_000_000,
            pipeline_memory_bytes=5_000_000_000,
        )
        d = e.as_dict()
        assert d["model_resident_bytes"] == 2_500_000_000
        assert d["pipeline_memory_bytes"] == 5_000_000_000

    def test_negative_timing_raises(self) -> None:
        with pytest.raises(ValueError):
            _make_event(started_monotonic_ns=-1)

    def test_negative_ttfa_raises(self) -> None:
        with pytest.raises(ValueError):
            _make_event(ttfa_ms=-1.0)


# ---------------------------------------------------------------------------
# Gate diagnostic integration — Requirement 10.3
# ---------------------------------------------------------------------------

class TestGateDiagnosticIntegration:
    def _make_gate(self, route="local_qwen") -> GateDiagnostic:
        qualifying = ("low_battery",) if route == "gemini_live" else ()
        return GateDiagnostic(
            enabled=route == "gemini_live",
            evaluated=("low_battery", "thermal_throttling", "ultra_complex_reasoning"),
            battery_percent=20 if route == "gemini_live" else 80,
            external_power_connected=False if route == "gemini_live" else True,
            thermal_state="nominal",
            assembled_prompt_tokens=100,
            validated_tool_count=1,
            qualifying=qualifying,
            selected_route=route,  # type: ignore[arg-type]
        )

    def test_gate_field_included_in_record(self) -> None:
        gate = self._make_gate("local_qwen")
        e = VoiceDiagnosticEvent(
            session_id=uuid4(), turn_id=uuid4(), stage="cloud_gate", outcome="completed",
            selected_route="local_qwen", gate=gate,
        )
        d = e.as_dict()
        assert d["gate"] is not None
        assert d["gate"]["enabled"] is False
        assert d["gate"]["selected_route"] == "local_qwen"

    def test_gate_route_mismatch_raises(self) -> None:
        gate = self._make_gate("local_qwen")
        with pytest.raises(ValueError, match="route must match"):
            VoiceDiagnosticEvent(
                session_id=uuid4(), turn_id=uuid4(), stage="cloud_gate", outcome="completed",
                selected_route="gemini_live",  # mismatches gate
                gate=gate,
            )

    def test_failed_cloud_gate_requires_error_class(self) -> None:
        gate = self._make_gate("local_qwen")
        with pytest.raises(ValueError, match="error class"):
            VoiceDiagnosticEvent(
                session_id=uuid4(), turn_id=uuid4(), stage="cloud_gate", outcome="failed",
                selected_route="local_qwen", gate=gate,
            )


# ---------------------------------------------------------------------------
# Atomic local store — Requirement 10.4
# ---------------------------------------------------------------------------

class TestAtomicLocalStore:
    def test_append_creates_file_with_0600_mode(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        e = _make_event()
        path = append_diagnostic(e, directory=diag_dir)
        file_mode = oct(stat.S_IMODE(os.stat(path).st_mode))
        assert file_mode == oct(0o600), f"Expected 0600, got {file_mode}"

    def test_directory_has_0700_mode(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        e = _make_event()
        append_diagnostic(e, directory=diag_dir)
        dir_mode = oct(stat.S_IMODE(os.stat(diag_dir).st_mode))
        assert dir_mode == oct(0o700), f"Expected 0700, got {dir_mode}"

    def test_append_returns_correct_path(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        e = _make_event()
        today = date.today()
        path = append_diagnostic(e, directory=diag_dir, for_date=today)
        assert path == diag_dir / f"{today.isoformat()}.jsonl"

    def test_multiple_appends_produce_multiple_jsonl_lines(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        today = date.today()
        for stage in ("asr", "local_llm", "playback"):
            append_diagnostic(_make_event(stage=stage), directory=diag_dir, for_date=today)
        records = read_diagnostics(directory=diag_dir, for_date=today)
        assert len(records) == 3
        stages = [r["stage"] for r in records]
        assert stages == ["asr", "local_llm", "playback"]

    def test_each_record_is_valid_json(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        today = date.today()
        e = _make_event(stage="ipc", outcome="failed", error_class="IPCError")
        append_diagnostic(e, directory=diag_dir, for_date=today)
        jsonl_path = diag_dir / f"{today.isoformat()}.jsonl"
        for raw_line in jsonl_path.read_text(encoding="utf-8").splitlines():
            parsed = json.loads(raw_line)
            assert parsed["stage"] == "ipc"

    def test_date_rotation(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        d1 = date(2025, 1, 1)
        d2 = date(2025, 1, 2)
        append_diagnostic(_make_event(stage="asr"), directory=diag_dir, for_date=d1)
        append_diagnostic(_make_event(stage="playback"), directory=diag_dir, for_date=d2)
        r1 = read_diagnostics(directory=diag_dir, for_date=d1)
        r2 = read_diagnostics(directory=diag_dir, for_date=d2)
        assert len(r1) == 1 and r1[0]["stage"] == "asr"
        assert len(r2) == 1 and r2[0]["stage"] == "playback"

    def test_read_returns_empty_list_when_no_file(self, tmp_path: Path) -> None:
        records = read_diagnostics(directory=tmp_path / "nonexistent", for_date=date.today())
        assert records == []

    def test_appended_record_excludes_content_fields(self, tmp_path: Path) -> None:
        """On-disk records must never contain content-bearing fields by default."""
        diag_dir = tmp_path / "voice"
        e = VoiceDiagnosticEvent.for_stage_start(
            session_id=uuid4(), turn_id=uuid4(), stage="asr",
            started_monotonic_ns=100, asr_engine="qwen3_asr_coreml",
        )
        append_diagnostic(e, directory=diag_dir)
        records = read_diagnostics(directory=diag_dir)
        assert len(records) == 1
        assert not _has_content_field(records[0])

    def test_appended_record_schema_version_is_1(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        e = _make_event()
        append_diagnostic(e, directory=diag_dir)
        records = read_diagnostics(directory=diag_dir)
        assert records[0]["schema_version"] == 1


# ---------------------------------------------------------------------------
# Privacy: content-bearing and raw-byte field exclusion enforced
# ---------------------------------------------------------------------------

class TestPrivacyEnforcement:
    """Requirement 10.5: default records must not contain raw audio or full transcript."""

    @pytest.mark.parametrize("stage", _all_stages())
    def test_no_content_field_in_default_serialization_for_every_stage(
        self, stage: str
    ) -> None:
        for outcome in ("started", "completed", "cancelled"):
            e = _make_event(stage=stage, outcome=outcome)
            d = e.as_dict()
            found = set(d.keys()) & _CONTENT_FIELD_NAMES
            assert not found, (
                f"stage={stage!r} outcome={outcome!r}: found prohibited field(s) {found}"
            )

    def test_content_in_record_only_with_explicit_session_label(self) -> None:
        """Content fields must use explicitly-labeled names to prevent confusion."""
        e = _make_event(stage="asr", outcome="completed", content_capture_enabled=True)
        e.set_transcript("Namaste")
        d = e.as_dict()
        # Must use "diagnostic_transcript_text", NOT "transcript" or "transcript_text"
        assert "transcript" not in d
        assert "transcript_text" not in d
        assert "diagnostic_transcript_text" in d

    def test_raw_audio_never_appears_in_any_record(self) -> None:
        """Raw audio bytes cannot appear even with content_capture_enabled=True."""
        for capture_enabled in (True, False):
            e = _make_event(stage="asr", outcome="started", content_capture_enabled=capture_enabled)
            d = e.as_dict()
            raw_audio_fields = {"pcm", "pcm_bytes", "audio", "audio_bytes", "raw_audio"}
            found = set(d.keys()) & raw_audio_fields
            assert not found, f"Raw audio field(s) {found} found in record"
