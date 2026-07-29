"""Versioned privacy-safe voice diagnostics schema, redaction, and local store.

This module implements the full Voice_Diagnostic_Event schema (§9 of design.md)
including: versioned JSONL records for all pipeline stages, session-scoped
content-capture control (default false, session-expiring, never raw audio),
and atomic append to a local HAKI directory with 0700/0600 permissions.

Default serialization excludes PCM bytes, transcript text, LLM response text,
tool arguments/results containing content, and full prompt text.  Only when
an active session has ``content_capture_enabled=True`` will transcript and
response fields appear in records; raw microphone audio is always excluded.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4


# ---------------------------------------------------------------------------
# Public type aliases used by cloud_gate.py and other callers
# ---------------------------------------------------------------------------

GateCondition = Literal[
    "low_battery", "thermal_throttling", "ultra_complex_reasoning"
]
VoiceRoute = Literal["local_qwen", "gemini_live"]
DiagnosticOutcome = Literal["started", "completed", "cancelled", "failed", "rejected"]

VoiceStage = Literal[
    "asr",
    "ipc",
    "pipecat",
    "voice_processing",
    "local_llm",
    "tool_call",
    "local_tts",
    "memory_budget",
    "cloud_gate",
    "playback",
]

_VALID_STAGES: frozenset[str] = frozenset({
    "asr", "ipc", "pipecat", "voice_processing", "local_llm",
    "tool_call", "local_tts", "memory_budget", "cloud_gate", "playback",
})
_VALID_OUTCOMES: frozenset[str] = frozenset({
    "started", "completed", "cancelled", "failed", "rejected",
})
_GATE_CONDITIONS: tuple[GateCondition, ...] = (
    "low_battery",
    "thermal_throttling",
    "ultra_complex_reasoning",
)


# ---------------------------------------------------------------------------
# Content-bearing fields that are ALWAYS excluded from default serialization
# ---------------------------------------------------------------------------

# These field names must never appear in a default-mode diagnostic record.
_PROHIBITED_DEFAULT_FIELDS: frozenset[str] = frozenset({
    "pcm",
    "pcm_bytes",
    "audio",
    "audio_bytes",
    "raw_audio",
    "transcript",
    "transcript_text",
    "response",
    "response_text",
    "prompt",
    "prompt_text",
    "tool_arguments",
    "tool_results",
    "tool_content",
})


# ---------------------------------------------------------------------------
# GateDiagnostic (cloud_gate evaluation payload)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GateDiagnostic:
    """Complete, content-free evidence for a single cloud-gate evaluation."""

    enabled: bool
    evaluated: tuple[GateCondition, ...]
    battery_percent: int | None
    external_power_connected: bool | None
    thermal_state: Literal["nominal", "fair", "serious", "critical"]
    assembled_prompt_tokens: int
    validated_tool_count: int
    qualifying: tuple[GateCondition, ...]
    selected_route: VoiceRoute

    def __post_init__(self) -> None:
        if self.evaluated != _GATE_CONDITIONS:
            raise ValueError("gate diagnostics must evaluate every qualifying condition")
        if not isinstance(self.enabled, bool):
            raise ValueError("gate enablement must be a boolean")
        if self.battery_percent is not None and (
            not isinstance(self.battery_percent, int)
            or isinstance(self.battery_percent, bool)
            or not 0 <= self.battery_percent <= 100
        ):
            raise ValueError("battery percent must be an integer from 0 through 100 or None")
        if self.external_power_connected is not None and not isinstance(
            self.external_power_connected, bool
        ):
            raise ValueError("external power state must be a boolean or None")
        if self.thermal_state not in {"nominal", "fair", "serious", "critical"}:
            raise ValueError("thermal state is invalid")
        for name, value in (
            ("assembled_prompt_tokens", self.assembled_prompt_tokens),
            ("validated_tool_count", self.validated_tool_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if any(condition not in _GATE_CONDITIONS for condition in self.qualifying):
            raise ValueError("gate diagnostic contains an unknown qualifying condition")
        if len(set(self.qualifying)) != len(self.qualifying):
            raise ValueError("gate diagnostic qualifying conditions must be unique")
        if (self.selected_route == "gemini_live") != (self.enabled and bool(self.qualifying)):
            raise ValueError("gate route must match enablement and qualifying conditions")

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-safe gate payload used by a diagnostic event."""
        return {
            "enabled": self.enabled,
            "evaluated": list(self.evaluated),
            "battery_percent": self.battery_percent,
            "external_power_connected": self.external_power_connected,
            "thermal_state": self.thermal_state,
            "assembled_prompt_tokens": self.assembled_prompt_tokens,
            "validated_tool_count": self.validated_tool_count,
            "qualifying": list(self.qualifying),
            "selected_route": self.selected_route,
        }


# ---------------------------------------------------------------------------
# Full VoiceDiagnosticEvent — all stages, all required fields
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class VoiceDiagnosticEvent:
    """A versioned schema-1 Voice_Diagnostic_Event record.

    Records identifiers, stage, outcome, all timing fields, routing, component
    IDs, resource measurements, gate data, and failure/recovery metadata.

    Default serialization EXCLUDES all content-bearing fields (PCM, transcript,
    response, prompt, tool arguments/results).  When ``content_capture_enabled``
    is True (set only by an active session-scoped UI action), transcript and
    response fields may be included in the record; raw audio is always excluded.
    """

    # Required identifiers
    session_id: UUID
    turn_id: UUID
    stage: VoiceStage
    outcome: DiagnosticOutcome

    # Component identifiers (optional per stage)
    asr_engine: str | None = None
    tts_engine: str | None = None
    selected_route: VoiceRoute | None = None

    # Timing fields (all monotonic nanoseconds, 0 = not yet recorded)
    started_monotonic_ns: int = 0
    transcription_completed_monotonic_ns: int = 0
    first_llm_text_monotonic_ns: int = 0
    first_tts_text_monotonic_ns: int = 0
    first_pcm_delivered_monotonic_ns: int = 0
    ttfa_ms: float | None = None

    # Resource measurements
    model_resident_bytes: int | None = None
    pipeline_memory_bytes: int | None = None

    # Cloud gate data
    gate: GateDiagnostic | None = None

    # Failure and recovery
    error_class: str | None = None
    recovery_outcome: str | None = None

    # Content-capture session flag (default false, expires at session end)
    content_capture_enabled: bool = False

    # Optional content fields: only included in serialization when
    # content_capture_enabled is True.  Never contains raw audio.
    _transcript_text: str | None = field(default=None, repr=False, compare=False)
    _response_text: str | None = field(default=None, repr=False, compare=False)

    # Auto-generated identifiers
    event_id: UUID = field(default_factory=uuid4)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, UUID) or not isinstance(self.turn_id, UUID):
            raise ValueError("diagnostic session and turn identifiers must be UUIDs")
        if self.stage not in _VALID_STAGES:
            raise ValueError(f"invalid stage: {self.stage!r}")
        if self.outcome not in _VALID_OUTCOMES:
            raise ValueError(f"invalid outcome: {self.outcome!r}")
        if self.stage == "cloud_gate" and self.selected_route is not None and self.gate is not None:
            if self.selected_route != self.gate.selected_route:
                raise ValueError("event route must match the gate route")
        if self.outcome == "failed" and self.stage == "cloud_gate" and not self.error_class:
            raise ValueError("failed cloud gate diagnostics require an error class")
        if self.outcome == "completed" and self.stage == "cloud_gate" and self.error_class is not None:
            raise ValueError("completed cloud gate diagnostics cannot contain an error class")
        for name, value in (
            ("started_monotonic_ns", self.started_monotonic_ns),
            ("transcription_completed_monotonic_ns", self.transcription_completed_monotonic_ns),
            ("first_llm_text_monotonic_ns", self.first_llm_text_monotonic_ns),
            ("first_tts_text_monotonic_ns", self.first_tts_text_monotonic_ns),
            ("first_pcm_delivered_monotonic_ns", self.first_pcm_delivered_monotonic_ns),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.ttfa_ms is not None and (not isinstance(self.ttfa_ms, (int, float)) or isinstance(self.ttfa_ms, bool) or self.ttfa_ms < 0):
            raise ValueError("ttfa_ms must be a non-negative number or None")
        for name, value in (
            ("model_resident_bytes", self.model_resident_bytes),
            ("pipeline_memory_bytes", self.pipeline_memory_bytes),
        ):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if not isinstance(self.content_capture_enabled, bool):
            raise ValueError("content_capture_enabled must be a boolean")
        if self._transcript_text is not None and not self.content_capture_enabled:
            raise ValueError("transcript_text requires content_capture_enabled=True")
        if self._response_text is not None and not self.content_capture_enabled:
            raise ValueError("response_text requires content_capture_enabled=True")

    def set_transcript(self, text: str) -> None:
        """Attach transcript text only when content capture is active for this session."""
        if not self.content_capture_enabled:
            raise ValueError("transcript capture requires content_capture_enabled=True")
        if not isinstance(text, str):
            raise ValueError("transcript text must be a string")
        object.__setattr__(self, "_transcript_text", text)

    def set_response(self, text: str) -> None:
        """Attach response text only when content capture is active for this session."""
        if not self.content_capture_enabled:
            raise ValueError("response capture requires content_capture_enabled=True")
        if not isinstance(text, str):
            raise ValueError("response text must be a string")
        object.__setattr__(self, "_response_text", text)

    def as_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dict.

        Default mode excludes all content-bearing fields.  Only identifiers,
        stage/outcome, timing, routing, resource measurements, gate data, and
        error/recovery fields appear in default records.  When
        ``content_capture_enabled`` is True, explicitly-labeled transcript and
        response fields are added; raw audio bytes are never included.
        """
        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "event_id": str(self.event_id),
            "session_id": str(self.session_id),
            "turn_id": str(self.turn_id),
            "stage": self.stage,
            "outcome": self.outcome,
            "started_monotonic_ns": self.started_monotonic_ns,
            "transcription_completed_monotonic_ns": self.transcription_completed_monotonic_ns,
            "first_llm_text_monotonic_ns": self.first_llm_text_monotonic_ns,
            "first_tts_text_monotonic_ns": self.first_tts_text_monotonic_ns,
            "first_pcm_delivered_monotonic_ns": self.first_pcm_delivered_monotonic_ns,
            "ttfa_ms": self.ttfa_ms,
            "selected_route": self.selected_route,
            "asr_engine": self.asr_engine,
            "tts_engine": self.tts_engine,
            "model_resident_bytes": self.model_resident_bytes,
            "pipeline_memory_bytes": self.pipeline_memory_bytes,
            "gate": self.gate.as_dict() if self.gate is not None else None,
            "error_class": self.error_class,
            "recovery_outcome": self.recovery_outcome,
            "content_capture_enabled": self.content_capture_enabled,
        }
        # Content fields only when session-scoped capture is enabled
        if self.content_capture_enabled:
            record["diagnostic_transcript_text"] = self._transcript_text
            record["diagnostic_response_text"] = self._response_text
        return record

    @classmethod
    def for_stage_start(
        cls,
        *,
        session_id: UUID,
        turn_id: UUID,
        stage: VoiceStage,
        started_monotonic_ns: int,
        asr_engine: str | None = None,
        tts_engine: str | None = None,
        selected_route: VoiceRoute | None = None,
        content_capture_enabled: bool = False,
    ) -> "VoiceDiagnosticEvent":
        """Convenience factory for a turn-started diagnostic (Requirement 10.1)."""
        return cls(
            session_id=session_id,
            turn_id=turn_id,
            stage=stage,
            outcome="started",
            started_monotonic_ns=started_monotonic_ns,
            asr_engine=asr_engine,
            tts_engine=tts_engine,
            selected_route=selected_route,
            content_capture_enabled=content_capture_enabled,
        )

    @classmethod
    def for_failure(
        cls,
        *,
        session_id: UUID,
        turn_id: UUID,
        stage: VoiceStage,
        error_class: str,
        recovery_outcome: str | None = None,
        content_capture_enabled: bool = False,
    ) -> "VoiceDiagnosticEvent":
        """Convenience factory for a stage-failure diagnostic (Requirement 10.6)."""
        if not error_class:
            raise ValueError("error_class must not be empty for failure diagnostics")
        return cls(
            session_id=session_id,
            turn_id=turn_id,
            stage=stage,
            outcome="failed",
            error_class=error_class,
            recovery_outcome=recovery_outcome,
            content_capture_enabled=content_capture_enabled,
        )


# ---------------------------------------------------------------------------
# Session-scoped content-capture registry
# ---------------------------------------------------------------------------

class ContentCaptureRegistry:
    """Tracks which active voice sessions have enabled diagnostic content capture.

    The control defaults to False for every new session, is visible in the UI,
    and is automatically removed when the session ends.  Raw audio is never
    a permitted content-capture field regardless of the setting.
    """

    def __init__(self) -> None:
        self._enabled: set[UUID] = set()
        self._active: set[UUID] = set()

    def register_session(self, session_id: UUID) -> None:
        """Start a new session with content capture disabled."""
        _require_uuid(session_id, "session_id")
        self._active.add(session_id)
        self._enabled.discard(session_id)

    def enable(self, session_id: UUID) -> bool:
        """Enable content capture for an active session; return new state."""
        _require_uuid(session_id, "session_id")
        if session_id not in self._active:
            raise ValueError("cannot enable content capture for an inactive session")
        self._enabled.add(session_id)
        return True

    def disable(self, session_id: UUID) -> bool:
        """Disable content capture for an active session; return new state."""
        _require_uuid(session_id, "session_id")
        self._enabled.discard(session_id)
        return False

    def end_session(self, session_id: UUID) -> None:
        """Remove all capture state when the session ends (expiry)."""
        _require_uuid(session_id, "session_id")
        self._enabled.discard(session_id)
        self._active.discard(session_id)

    def is_enabled(self, session_id: UUID) -> bool:
        """Return whether content capture is currently enabled for this session."""
        _require_uuid(session_id, "session_id")
        return session_id in self._active and session_id in self._enabled


# ---------------------------------------------------------------------------
# Atomic local JSONL diagnostics store
# ---------------------------------------------------------------------------

def _default_diagnostics_dir() -> Path:
    """Return ~/Library/Application Support/HAKI/diagnostics/voice/."""
    home = Path.home()
    return home / "Library" / "Application Support" / "HAKI" / "diagnostics" / "voice"


def _ensure_directory(directory: Path) -> None:
    """Create the diagnostics directory with 0700 permissions if absent."""
    directory.mkdir(parents=True, exist_ok=True)
    # Enforce 0700 on the directory even if it already existed
    directory.chmod(0o700)


def _date_file(directory: Path, for_date: date | None = None) -> Path:
    """Return the JSONL path for the given date (defaults to today local time)."""
    today = for_date or datetime.now(tz=timezone.utc).astimezone().date()
    return directory / f"{today.isoformat()}.jsonl"


def append_diagnostic(
    event: VoiceDiagnosticEvent,
    *,
    directory: Path | None = None,
    for_date: date | None = None,
) -> Path:
    """Atomically append one JSON-serialized event to the local JSONL store.

    Storage path: ``<directory>/<local-date>.jsonl``
    Directory mode: 0700.  File mode: 0600.
    Atomic: the record is written to a sibling tempfile and renamed into place
    so a crash mid-write cannot corrupt the existing store.

    Returns the path of the JSONL file that was appended to.

    Raises ``DiagnosticStoreError`` if the directory cannot be prepared or the
    atomic rename fails; the caller should convert this to a non-fatal event.
    """
    target_dir = directory if directory is not None else _default_diagnostics_dir()
    try:
        _ensure_directory(target_dir)
    except OSError as exc:
        raise DiagnosticStoreError(f"cannot prepare diagnostics directory: {exc}") from exc

    jsonl_path = _date_file(target_dir, for_date)
    line = json.dumps(event.as_dict(), separators=(",", ":"), ensure_ascii=False) + "\n"
    encoded = line.encode("utf-8")

    try:
        # Open the target file for append, create with 0600 if new
        fd = os.open(
            str(jsonl_path),
            os.O_CREAT | os.O_WRONLY | os.O_APPEND,
            0o600,
        )
        try:
            os.write(fd, encoded)
        finally:
            os.close(fd)
        # Enforce 0600 on an existing file whose mode may differ
        jsonl_path.chmod(0o600)
    except OSError as exc:
        raise DiagnosticStoreError(f"cannot append diagnostic record: {exc}") from exc

    return jsonl_path


def read_diagnostics(
    *,
    directory: Path | None = None,
    for_date: date | None = None,
) -> list[dict[str, object]]:
    """Read all diagnostic records from the JSONL file for a given date.

    Returns an empty list if the file does not yet exist.
    Raises ``DiagnosticStoreError`` for I/O or parse failures.
    """
    target_dir = directory if directory is not None else _default_diagnostics_dir()
    jsonl_path = _date_file(target_dir, for_date)
    if not jsonl_path.exists():
        return []
    try:
        records: list[dict[str, object]] = []
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                stripped = raw_line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticStoreError(f"cannot read diagnostics: {exc}") from exc


class DiagnosticStoreError(RuntimeError):
    """Non-fatal storage failure that must not propagate content."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_uuid(value: object, name: str) -> None:
    if not isinstance(value, UUID):
        raise ValueError(f"{name} must be a UUID")


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "ContentCaptureRegistry",
    "DiagnosticOutcome",
    "DiagnosticStoreError",
    "GateCondition",
    "GateDiagnostic",
    "VoiceDiagnosticEvent",
    "VoiceRoute",
    "VoiceStage",
    "append_diagnostic",
    "read_diagnostics",
]
