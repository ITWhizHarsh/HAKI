"""Strict v1 protocol for the local Swift/Python voice UNIX socket.

The text/control transport is JSON Lines.  It intentionally accepts only the
schemas defined below and **never** accepts microphone bytes, samples, or a
microphone-audio encoding.  PCM is permitted only in the Python-to-Swift
``PCM_CHUNK`` output frame: its JSON metadata line is immediately followed by
a four-byte unsigned big-endian byte length and exactly that many S16LE bytes.

The transport server owns socket creation and peer-credential extraction.  This
module supplies the pure schema, framing, session-ordering, and ownership
validation primitives it needs, keeping the contract independently testable.

Protocol v1 message types
=========================

Swift -> Python:
  * ``TRANSCRIPT_EVENT`` -- normalized text only, ordered per turn.
  * ``CAPTURE_INTERRUPTED`` -- invalidates a capture turn.
  * ``EVENT_ACK`` -- acknowledgement for an event identifier.
  * ``PCM_ACCEPTED`` / ``PLAYBACK_*`` -- renderer acknowledgements and terminal
    playback events.

Python -> Swift:
  * ``EVENT_ACK`` -- accepted or discarded transcript acknowledgement.
  * ``STOP_PLAYBACK`` / ``STOP_PLAYBACK_ACK`` -- idempotent cancellation control.
  * ``PCM_CHUNK`` -- output PCM metadata plus its binary payload.

A transcript turn accepts zero or more non-final events followed by one final
event.  Event sequence values must increase by exactly one after the first
accepted event for that turn.  A final or playback terminal event is terminal;
all later events of that kind are rejected.  Session-bound validation rejects
messages for an older/replaced session before they can affect state.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import stat
import struct
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping


PROTOCOL_VERSION: Final = 1
MAX_JSON_LINE_BYTES: Final = 64 * 1024
MAX_PCM_CHUNK_BYTES: Final = 4 * 1024 * 1024
MAX_EVENT_SEQUENCE: Final = (1 << 63) - 1
SESSION_CAPABILITY_HEX_LENGTH: Final = 32  # 128 random bits encoded as hex.
PCM_LENGTH_PREFIX_BYTES: Final = 4

TRANSCRIPT_EVENT: Final = "TRANSCRIPT_EVENT"
EVENT_ACK: Final = "EVENT_ACK"
CAPTURE_INTERRUPTED: Final = "CAPTURE_INTERRUPTED"
STOP_PLAYBACK: Final = "STOP_PLAYBACK"
STOP_PLAYBACK_ACK: Final = "STOP_PLAYBACK_ACK"
PCM_CHUNK: Final = "PCM_CHUNK"
PCM_ACCEPTED: Final = "PCM_ACCEPTED"
PLAYBACK_CONFIRMED: Final = "PLAYBACK_CONFIRMED"
PLAYBACK_CANCELLED: Final = "PLAYBACK_CANCELLED"
PLAYBACK_FAILED: Final = "PLAYBACK_FAILED"

TRANSCRIPT_LANGUAGES: Final = frozenset({"hi", "en", "hinglish"})
PLAYBACK_TERMINAL_TYPES: Final = frozenset(
    {PLAYBACK_CONFIRMED, PLAYBACK_CANCELLED, PLAYBACK_FAILED}
)

# This is intentionally available to Swift fixture generators and Python tests.
# It documents the complete v1 wire contract without exposing microphone data.
PROTOCOL_V1_DOCUMENTATION: Final = """
v1 JSONL schema summary
-----------------------
TRANSCRIPT_EVENT:
  {version, type, event_id, session_id, turn_id, event_seq, text, is_final,
   language, capture_started_monotonic_ns, capture_ended_monotonic_ns}
EVENT_ACK:
  {version, type, event_id, status[, reason]}; status is accepted|discarded.
CAPTURE_INTERRUPTED:
  {version, type, event_id, session_id, turn_id, reason}
STOP_PLAYBACK / STOP_PLAYBACK_ACK:
  {version, type, session_id, turn_id, generation}
PCM_CHUNK metadata:
  {version, type, session_id, turn_id, sentence_id, sequence, sample_rate_hz,
   channels, format, byte_length}, then u32-be byte_length and raw S16LE PCM.
PCM_ACCEPTED:
  {version, type, session_id, turn_id, sentence_id, sequence}
PLAYBACK_CONFIRMED / PLAYBACK_CANCELLED:
  {version, type, event_id, session_id, turn_id, sentence_id}
PLAYBACK_FAILED:
  {version, type, event_id, session_id, turn_id, sentence_id, error_class}

All object keys are exact. Unknown keys and names that could carry microphone
content (for example audio, microphone, samples, pcm, waveform, bytes, or
base64 fields) are rejected. IDs are lowercase canonical UUID strings.
""".strip()

_FIXTURE_SESSION_ID: Final = "00000000-0000-4000-8000-000000000001"
_FIXTURE_TURN_ID: Final = "00000000-0000-4000-8000-000000000002"
_FIXTURE_PARTIAL_EVENT_ID: Final = "00000000-0000-4000-8000-000000000003"
_FIXTURE_FINAL_EVENT_ID: Final = "00000000-0000-4000-8000-000000000004"
_FIXTURE_SENTENCE_ID: Final = "00000000-0000-4000-8000-000000000005"

PROTOCOL_V1_FIXTURES: Final[Mapping[str, Mapping[str, Any]]] = {
    "transcript_partial": {
        "version": 1,
        "type": TRANSCRIPT_EVENT,
        "event_id": _FIXTURE_PARTIAL_EVENT_ID,
        "session_id": _FIXTURE_SESSION_ID,
        "turn_id": _FIXTURE_TURN_ID,
        "event_seq": 17,
        "text": "Kal meeting",
        "is_final": False,
        "language": "hinglish",
        "capture_started_monotonic_ns": 123,
        "capture_ended_monotonic_ns": 456,
    },
    "transcript_final": {
        "version": 1,
        "type": TRANSCRIPT_EVENT,
        "event_id": _FIXTURE_FINAL_EVENT_ID,
        "session_id": _FIXTURE_SESSION_ID,
        "turn_id": _FIXTURE_TURN_ID,
        "event_seq": 18,
        "text": "Kal meeting reschedule kar do",
        "is_final": True,
        "language": "hinglish",
        "capture_started_monotonic_ns": 123,
        "capture_ended_monotonic_ns": 789,
    },
    "pcm_chunk_metadata": {
        "version": 1,
        "type": PCM_CHUNK,
        "session_id": _FIXTURE_SESSION_ID,
        "turn_id": _FIXTURE_TURN_ID,
        "sentence_id": _FIXTURE_SENTENCE_ID,
        "sequence": 0,
        "sample_rate_hz": 24_000,
        "channels": 1,
        "format": "s16le",
        "byte_length": 4,
    },
    "playback_confirmed": {
        "version": 1,
        "type": PLAYBACK_CONFIRMED,
        "event_id": "00000000-0000-4000-8000-000000000006",
        "session_id": _FIXTURE_SESSION_ID,
        "turn_id": _FIXTURE_TURN_ID,
        "sentence_id": _FIXTURE_SENTENCE_ID,
    },
}

_UUID_FIELDS: Final = frozenset({"event_id", "session_id", "turn_id", "sentence_id"})
_MICROPHONE_FIELD_TOKENS: Final = (
    "audio",
    "microphone",
    "mic",
    "sample",
    "pcm",
    "waveform",
    "wave",
    "buffer",
    "base64",
    "binary",
    "bytes",
)
_CAPABILITY_RE: Final = re.compile(rf"[0-9a-f]{{{SESSION_CAPABILITY_HEX_LENGTH}}}")


class VoiceProtocolError(ValueError):
    """A stable, content-free rejection returned by protocol validation."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class PeerSecurityError(VoiceProtocolError):
    """A socket owner, mode, peer UID, or capability validation failure."""


@dataclass(frozen=True)
class ValidatedMessage:
    """An immutable, schema-validated JSONL message."""

    data: Mapping[str, Any]

    @property
    def message_type(self) -> str:
        return self.data["type"]

    def as_dict(self) -> dict[str, Any]:
        """Return a detached copy suitable for a transport writer."""
        return dict(self.data)


@dataclass
class _TurnState:
    last_event_seq: int | None = None
    finalized: bool = False


@dataclass
class VoiceProtocolSession:
    """Stateful v1 validator for one active voice-session identifier.

    Instances deliberately contain no socket or peer state.  A server should
    create one only after validating the UDS owner, peer UID, and session
    capability, then use :meth:`accept_jsonl` for each JSONL control record.
    """

    session_id: str | uuid.UUID
    _turns: dict[str, _TurnState] = field(default_factory=dict, init=False)
    _pcm_sequences: dict[tuple[str, str], int] = field(default_factory=dict, init=False)
    _terminal_sentences: set[tuple[str, str]] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.session_id = _canonical_uuid("session_id", self.session_id)

    def accept_jsonl(self, line: bytes | str) -> ValidatedMessage:
        """Decode, schema-validate, session-check, and sequence-check JSONL."""
        return self.accept(decode_jsonl(line))

    def accept(self, message: Mapping[str, Any] | ValidatedMessage) -> ValidatedMessage:
        """Accept a valid message only if it belongs to this live session."""
        validated = (
            message if isinstance(message, ValidatedMessage) else validate_message(message)
        )
        data = validated.data
        session_id = data.get("session_id")
        if session_id is not None and session_id != self.session_id:
            raise VoiceProtocolError("stale_session")

        if data["type"] == TRANSCRIPT_EVENT:
            self._accept_transcript(data)
        elif data["type"] == PCM_CHUNK:
            self._accept_pcm_sequence(data)
        elif data["type"] in PLAYBACK_TERMINAL_TYPES:
            self._accept_playback_terminal(data)
        return validated

    def _accept_transcript(self, message: Mapping[str, Any]) -> None:
        turn_id = message["turn_id"]
        state = self._turns.setdefault(turn_id, _TurnState())
        if state.finalized:
            raise VoiceProtocolError("duplicate_final")

        event_seq = message["event_seq"]
        if state.last_event_seq is not None and event_seq != state.last_event_seq + 1:
            raise VoiceProtocolError("invalid_event_sequence")

        state.last_event_seq = event_seq
        state.finalized = message["is_final"]

    def _accept_pcm_sequence(self, message: Mapping[str, Any]) -> None:
        key = (message["turn_id"], message["sentence_id"])
        last_sequence = self._pcm_sequences.get(key)
        sequence = message["sequence"]
        if last_sequence is not None and sequence != last_sequence + 1:
            raise VoiceProtocolError("invalid_pcm_sequence")
        self._pcm_sequences[key] = sequence

    def _accept_playback_terminal(self, message: Mapping[str, Any]) -> None:
        key = (message["turn_id"], message["sentence_id"])
        if key in self._terminal_sentences:
            raise VoiceProtocolError("duplicate_playback_terminal")
        self._terminal_sentences.add(key)


def protocol_fixture(name: str) -> dict[str, Any]:
    """Return a mutable copy of one documented, valid v1 fixture."""
    try:
        return dict(PROTOCOL_V1_FIXTURES[name])
    except KeyError as exc:
        raise KeyError(f"unknown voice protocol fixture: {name}") from exc


def encode_jsonl(message: Mapping[str, Any] | ValidatedMessage) -> bytes:
    """Validate and serialize a single JSONL message without content logging."""
    validated = message if isinstance(message, ValidatedMessage) else validate_message(message)
    return (
        json.dumps(validated.data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def decode_jsonl(line: bytes | str) -> ValidatedMessage:
    """Decode exactly one UTF-8 JSONL object and validate its strict v1 schema."""
    if isinstance(line, bytes):
        if len(line) > MAX_JSON_LINE_BYTES:
            raise VoiceProtocolError("json_line_too_large")
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise VoiceProtocolError("invalid_utf8") from exc
    elif isinstance(line, str):
        if len(line.encode("utf-8")) > MAX_JSON_LINE_BYTES:
            raise VoiceProtocolError("json_line_too_large")
        text = line
    else:
        raise VoiceProtocolError("invalid_jsonl_type")

    if text.endswith("\n"):
        text = text[:-1]
    if not text or "\n" in text or "\r" in text:
        raise VoiceProtocolError("invalid_jsonl_framing")

    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    except VoiceProtocolError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VoiceProtocolError("malformed_json") from exc
    return validate_message(value)


def validate_message(message: Mapping[str, Any] | ValidatedMessage) -> ValidatedMessage:
    """Validate one complete v1 JSON object without applying session state."""
    if isinstance(message, ValidatedMessage):
        return message
    if not isinstance(message, Mapping):
        raise VoiceProtocolError("message_must_be_object")

    data = dict(message)
    _validate_common_fields(data)
    message_type = data["type"]

    if message_type == TRANSCRIPT_EVENT:
        _validate_transcript_event(data)
    elif message_type == EVENT_ACK:
        _validate_event_ack(data)
    elif message_type == CAPTURE_INTERRUPTED:
        _validate_capture_interrupted(data)
    elif message_type in {STOP_PLAYBACK, STOP_PLAYBACK_ACK}:
        _validate_stop_playback(data)
    elif message_type == PCM_CHUNK:
        _validate_pcm_chunk_metadata(data)
    elif message_type == PCM_ACCEPTED:
        _validate_pcm_accepted(data)
    elif message_type in PLAYBACK_TERMINAL_TYPES:
        _validate_playback_terminal(data)
    else:
        raise VoiceProtocolError("unknown_message_type")
    return ValidatedMessage(data)


def encode_pcm_chunk(
    metadata: Mapping[str, Any] | ValidatedMessage,
    payload: bytes,
) -> bytes:
    """Return a JSONL PCM header followed by its exact binary length frame."""
    if not isinstance(payload, bytes):
        raise VoiceProtocolError("invalid_binary_payload")
    validated = metadata if isinstance(metadata, ValidatedMessage) else validate_message(metadata)
    if validated.message_type != PCM_CHUNK:
        raise VoiceProtocolError("pcm_metadata_type_required")
    _validate_pcm_payload_length(validated.data["byte_length"], payload)
    return encode_jsonl(validated) + struct.pack(">I", len(payload)) + payload


def decode_pcm_chunk(frame: bytes) -> tuple[ValidatedMessage, bytes]:
    """Decode one complete PCM output frame and reject all length ambiguities."""
    if not isinstance(frame, bytes):
        raise VoiceProtocolError("invalid_binary_frame")
    newline_index = frame.find(b"\n")
    if newline_index < 0 or newline_index + 1 > MAX_JSON_LINE_BYTES:
        raise VoiceProtocolError("invalid_binary_framing")

    metadata = decode_jsonl(frame[: newline_index + 1])
    if metadata.message_type != PCM_CHUNK:
        raise VoiceProtocolError("pcm_metadata_type_required")

    payload_start = newline_index + 1 + PCM_LENGTH_PREFIX_BYTES
    if len(frame) < payload_start:
        raise VoiceProtocolError("invalid_binary_length")
    declared_prefix_length = struct.unpack(">I", frame[newline_index + 1 : payload_start])[0]
    payload = frame[payload_start:]
    if declared_prefix_length != len(payload):
        raise VoiceProtocolError("invalid_binary_length")
    _validate_pcm_payload_length(metadata.data["byte_length"], payload)
    return metadata, payload


def generate_session_capability() -> str:
    """Generate the 128-bit capability passed only through inherited launch config."""
    return secrets.token_hex(16)


def validate_session_capability(candidate: str, expected: str) -> None:
    """Validate a capability's representation and compare it in constant time."""
    if not isinstance(candidate, str) or not isinstance(expected, str):
        raise PeerSecurityError("invalid_session_capability")
    if not _CAPABILITY_RE.fullmatch(candidate) or not _CAPABILITY_RE.fullmatch(expected):
        raise PeerSecurityError("invalid_session_capability")
    if not hmac.compare_digest(candidate, expected):
        raise PeerSecurityError("session_capability_mismatch")


def validate_peer_uid(peer_uid: int, *, owner_uid: int | None = None) -> None:
    """Reject a UDS peer that is not the HAKI process owner."""
    expected_uid = os.getuid() if owner_uid is None else owner_uid
    if not _is_int(peer_uid) or not _is_int(expected_uid) or peer_uid != expected_uid:
        raise PeerSecurityError("peer_uid_mismatch")


def validate_owner_only_directory(path: str | Path, *, owner_uid: int | None = None) -> None:
    """Require an existing non-symlink runtime directory owned at mode 0700."""
    status = _safe_lstat(path)
    if not stat.S_ISDIR(status.st_mode):
        raise PeerSecurityError("runtime_directory_not_directory")
    _validate_owner_and_mode(status, owner_uid=owner_uid, required_mode=0o700, kind="runtime_directory")


def validate_owner_only_socket(path: str | Path, *, owner_uid: int | None = None) -> None:
    """Require an existing non-symlink UDS socket owned at mode 0600."""
    status = _safe_lstat(path)
    if not stat.S_ISSOCK(status.st_mode):
        raise PeerSecurityError("socket_not_unix_socket")
    _validate_owner_and_mode(status, owner_uid=owner_uid, required_mode=0o600, kind="socket")


def _safe_lstat(path: str | Path) -> os.stat_result:
    try:
        status = os.lstat(path)
    except OSError as exc:
        raise PeerSecurityError("socket_path_unavailable") from exc
    if stat.S_ISLNK(status.st_mode):
        raise PeerSecurityError("socket_path_symlink")
    return status


def _validate_owner_and_mode(
    status: os.stat_result,
    *,
    owner_uid: int | None,
    required_mode: int,
    kind: str,
) -> None:
    expected_uid = os.getuid() if owner_uid is None else owner_uid
    if status.st_uid != expected_uid:
        raise PeerSecurityError(f"{kind}_owner_mismatch")
    if stat.S_IMODE(status.st_mode) != required_mode:
        raise PeerSecurityError(f"{kind}_mode_insecure")


def _validate_common_fields(data: Mapping[str, Any]) -> None:
    # Type-specific validators own exact-field checks; doing it here would
    # incorrectly reject every type-specific field before dispatch.
    if not {"version", "type"}.issubset(data):
        raise VoiceProtocolError("missing_required_field")
    if not _is_int(data["version"]):
        raise VoiceProtocolError("invalid_protocol_version")
    if data["version"] != PROTOCOL_VERSION:
        raise VoiceProtocolError("protocol_version_incompatible")
    if not isinstance(data["type"], str):
        raise VoiceProtocolError("invalid_message_type")


def _validate_transcript_event(data: Mapping[str, Any]) -> None:
    _ensure_exact_fields(
        data,
        required={
            "version", "type", "event_id", "session_id", "turn_id", "event_seq", "text",
            "is_final", "language", "capture_started_monotonic_ns", "capture_ended_monotonic_ns",
        },
    )
    _validate_uuid_fields(data, {"event_id", "session_id", "turn_id"})
    _validate_nonnegative_int(data, "event_seq", maximum=MAX_EVENT_SEQUENCE)
    if not isinstance(data["text"], str) or not data["text"].strip():
        raise VoiceProtocolError("invalid_transcript_text")
    if not isinstance(data["is_final"], bool):
        raise VoiceProtocolError("invalid_finality")
    if data["language"] not in TRANSCRIPT_LANGUAGES:
        raise VoiceProtocolError("invalid_language")
    _validate_nonnegative_int(data, "capture_started_monotonic_ns")
    _validate_nonnegative_int(data, "capture_ended_monotonic_ns")
    if data["capture_ended_monotonic_ns"] < data["capture_started_monotonic_ns"]:
        raise VoiceProtocolError("invalid_capture_timestamps")


def _validate_event_ack(data: Mapping[str, Any]) -> None:
    _ensure_exact_fields(
        data,
        required={"version", "type", "event_id", "status"},
        optional={"reason"},
    )
    _validate_uuid_fields(data, {"event_id"})
    if data["status"] not in {"accepted", "discarded"}:
        raise VoiceProtocolError("invalid_ack_status")
    if "reason" in data and (not isinstance(data["reason"], str) or not data["reason"].strip()):
        raise VoiceProtocolError("invalid_ack_reason")


def _validate_capture_interrupted(data: Mapping[str, Any]) -> None:
    _ensure_exact_fields(
        data,
        required={"version", "type", "event_id", "session_id", "turn_id", "reason"},
    )
    _validate_uuid_fields(data, {"event_id", "session_id", "turn_id"})
    _validate_nonempty_string(data, "reason", reason="invalid_control_reason")


def _validate_stop_playback(data: Mapping[str, Any]) -> None:
    _ensure_exact_fields(
        data,
        required={"version", "type", "session_id", "turn_id", "generation"},
    )
    _validate_uuid_fields(data, {"session_id", "turn_id"})
    _validate_nonnegative_int(data, "generation")


def _validate_pcm_chunk_metadata(data: Mapping[str, Any]) -> None:
    _ensure_exact_fields(
        data,
        required={
            "version", "type", "session_id", "turn_id", "sentence_id", "sequence",
            "sample_rate_hz", "channels", "format", "byte_length",
        },
    )
    _validate_uuid_fields(data, {"session_id", "turn_id", "sentence_id"})
    _validate_nonnegative_int(data, "sequence", maximum=MAX_EVENT_SEQUENCE)
    _validate_nonnegative_int(data, "sample_rate_hz", minimum=8_000, maximum=192_000)
    _validate_nonnegative_int(data, "channels", minimum=1, maximum=2)
    if data["format"] != "s16le":
        raise VoiceProtocolError("invalid_pcm_format")
    _validate_nonnegative_int(data, "byte_length", minimum=1, maximum=MAX_PCM_CHUNK_BYTES)
    if data["byte_length"] % 2:
        raise VoiceProtocolError("invalid_binary_length")


def _validate_pcm_accepted(data: Mapping[str, Any]) -> None:
    _ensure_exact_fields(
        data,
        required={"version", "type", "session_id", "turn_id", "sentence_id", "sequence"},
    )
    _validate_uuid_fields(data, {"session_id", "turn_id", "sentence_id"})
    _validate_nonnegative_int(data, "sequence", maximum=MAX_EVENT_SEQUENCE)


def _validate_playback_terminal(data: Mapping[str, Any]) -> None:
    required = {"version", "type", "event_id", "session_id", "turn_id", "sentence_id"}
    if data["type"] == PLAYBACK_FAILED:
        required.add("error_class")
    _ensure_exact_fields(data, required=required)
    _validate_uuid_fields(data, {"event_id", "session_id", "turn_id", "sentence_id"})
    if data["type"] == PLAYBACK_FAILED:
        _validate_nonempty_string(data, "error_class", reason="invalid_playback_error_class")


def _ensure_exact_fields(
    data: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    keys = set(data)
    unknown = keys - allowed
    if unknown:
        if any(_is_microphone_payload_field(key) for key in unknown):
            raise VoiceProtocolError("microphone_payload_forbidden")
        raise VoiceProtocolError("unknown_field")
    if keys != required and not required.issubset(keys):
        raise VoiceProtocolError("missing_required_field")


def _is_microphone_payload_field(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.lower().replace("_", "").replace("-", "")
    return any(token in normalized for token in _MICROPHONE_FIELD_TOKENS)


def _validate_uuid_fields(data: Mapping[str, Any], fields: set[str]) -> None:
    for field_name in fields:
        _canonical_uuid(field_name, data[field_name])


def _canonical_uuid(field_name: str, value: str | uuid.UUID) -> str:
    if isinstance(value, uuid.UUID):
        return str(value)
    if not isinstance(value, str):
        raise VoiceProtocolError(f"malformed_id:{field_name}")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError, TypeError) as exc:
        raise VoiceProtocolError(f"malformed_id:{field_name}") from exc
    if str(parsed) != value:
        raise VoiceProtocolError(f"malformed_id:{field_name}")
    return value


def _validate_nonnegative_int(
    data: Mapping[str, Any],
    field_name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> None:
    value = data[field_name]
    if not _is_int(value) or value < minimum or (maximum is not None and value > maximum):
        raise VoiceProtocolError(f"invalid_{field_name}")


def _validate_nonempty_string(data: Mapping[str, Any], field_name: str, *, reason: str) -> None:
    value = data[field_name]
    if not isinstance(value, str) or not value.strip():
        raise VoiceProtocolError(reason)


def _validate_pcm_payload_length(expected_length: int, payload: bytes) -> None:
    if len(payload) != expected_length or len(payload) > MAX_PCM_CHUNK_BYTES:
        raise VoiceProtocolError("invalid_binary_length")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VoiceProtocolError("duplicate_json_field")
        result[key] = value
    return result


__all__ = [
    "CAPTURE_INTERRUPTED",
    "EVENT_ACK",
    "MAX_PCM_CHUNK_BYTES",
    "PCM_ACCEPTED",
    "PCM_CHUNK",
    "PLAYBACK_CANCELLED",
    "PLAYBACK_CONFIRMED",
    "PLAYBACK_FAILED",
    "PLAYBACK_TERMINAL_TYPES",
    "PROTOCOL_V1_DOCUMENTATION",
    "PROTOCOL_V1_FIXTURES",
    "PROTOCOL_VERSION",
    "PeerSecurityError",
    "STOP_PLAYBACK",
    "STOP_PLAYBACK_ACK",
    "TRANSCRIPT_EVENT",
    "TRANSCRIPT_LANGUAGES",
    "ValidatedMessage",
    "VoiceProtocolError",
    "VoiceProtocolSession",
    "decode_jsonl",
    "decode_pcm_chunk",
    "encode_jsonl",
    "encode_pcm_chunk",
    "generate_session_capability",
    "protocol_fixture",
    "validate_message",
    "validate_owner_only_directory",
    "validate_owner_only_socket",
    "validate_peer_uid",
    "validate_session_capability",
]
