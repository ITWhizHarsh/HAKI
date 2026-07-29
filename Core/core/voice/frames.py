"""Typed metadata and wrappers for frames in a local voice session.

Voice pipeline metadata is carried separately from text payloads so turn IDs,
ordering, capture timing, and cancellation generations remain available to every
processor without being encoded into user-visible content.  These types do not
import Pipecat; the pipeline adapter can attach them to the corresponding
Pipecat frames while this module remains independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Mapping, TypeVar
from uuid import UUID

from .interfaces import VoiceLanguage


class VoiceFrameType(str, Enum):
    """The frame categories used by the replacement voice pipeline."""

    INPUT_AUDIO = "input_audio"
    TRANSCRIPTION = "transcription"
    LLM_TEXT = "llm_text"
    TTS_TEXT = "tts_text"
    PCM_CHUNK = "pcm_chunk"
    PLAYBACK_EVENT = "playback_event"
    CONTROL = "control"


class FrameMetadataError(ValueError):
    """Raised when frame metadata is malformed before it reaches a pipeline."""


@dataclass(frozen=True, slots=True)
class VoiceFrameMetadata:
    """Ordering and cancellation identity common to every voice frame.

    ``sequence`` is monotonically gap-free within a turn once its first frame
    is accepted.  ``cancellation_generation`` identifies the session generation
    that produced the frame; producers must not emit work from an older
    generation after interruption or cancellation.
    """

    session_id: UUID
    turn_id: UUID
    sequence: int
    cancellation_generation: int

    def __post_init__(self) -> None:
        _require_uuid("session_id", self.session_id)
        _require_uuid("turn_id", self.turn_id)
        _require_nonnegative_int("sequence", self.sequence)
        _require_nonnegative_int("cancellation_generation", self.cancellation_generation)


@dataclass(frozen=True, slots=True)
class AudioFrameMetadata(VoiceFrameMetadata):
    """Metadata for an in-memory, ring-backed audio-input frame."""

    captured_monotonic_ns: int
    sample_rate_hz: int
    channels: int

    def __post_init__(self) -> None:
        super(AudioFrameMetadata, self).__post_init__()
        _require_nonnegative_int("captured_monotonic_ns", self.captured_monotonic_ns)
        _require_positive_int("sample_rate_hz", self.sample_rate_hz)
        _require_positive_int("channels", self.channels)


@dataclass(frozen=True, slots=True)
class TranscriptionFrameMetadata(VoiceFrameMetadata):
    """Metadata attached to a normalized ASR partial or final transcript."""

    event_seq: int
    is_final: bool
    language: VoiceLanguage
    capture_started_monotonic_ns: int
    capture_ended_monotonic_ns: int

    def __post_init__(self) -> None:
        super(TranscriptionFrameMetadata, self).__post_init__()
        _require_nonnegative_int("event_seq", self.event_seq)
        if self.sequence != self.event_seq:
            raise FrameMetadataError("transcription sequence must equal event_seq")
        if not isinstance(self.is_final, bool):
            raise FrameMetadataError("is_final must be a bool")
        if self.language not in {"hi", "en", "hinglish"}:
            raise FrameMetadataError("language must be hi, en, or hinglish")
        _require_nonnegative_int("capture_started_monotonic_ns", self.capture_started_monotonic_ns)
        _require_nonnegative_int("capture_ended_monotonic_ns", self.capture_ended_monotonic_ns)
        if self.capture_ended_monotonic_ns < self.capture_started_monotonic_ns:
            raise FrameMetadataError("capture end must not precede capture start")


@dataclass(frozen=True, slots=True)
class SentenceFrameMetadata(VoiceFrameMetadata):
    """Metadata for a sentence-oriented TTS or playback frame."""

    sentence_id: UUID

    def __post_init__(self) -> None:
        super(SentenceFrameMetadata, self).__post_init__()
        _require_uuid("sentence_id", self.sentence_id)


@dataclass(frozen=True, slots=True)
class PCMChunkFrameMetadata(SentenceFrameMetadata):
    """Metadata for one ordered PCM chunk belonging to a sentence."""

    chunk_sequence: int

    def __post_init__(self) -> None:
        super(PCMChunkFrameMetadata, self).__post_init__()
        _require_nonnegative_int("chunk_sequence", self.chunk_sequence)


PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True, slots=True)
class TypedVoiceFrame(Generic[PayloadT]):
    """A pipeline-neutral payload plus its typed, non-text metadata."""

    frame_type: VoiceFrameType
    metadata: VoiceFrameMetadata
    payload: PayloadT

    def __post_init__(self) -> None:
        if not isinstance(self.frame_type, VoiceFrameType):
            raise FrameMetadataError("frame_type must be a VoiceFrameType")
        if not isinstance(self.metadata, VoiceFrameMetadata):
            raise FrameMetadataError("metadata must be VoiceFrameMetadata")


def metadata_from_transcript_event(
    event: Mapping[str, Any],
    *,
    cancellation_generation: int,
) -> TranscriptionFrameMetadata:
    """Convert a previously schema-validated v1 transcript event to metadata.

    The strict UDS protocol owns complete wire validation.  This helper only
    extracts the fields needed by downstream frames, retaining them outside the
    transcript text and assigning the active session cancellation generation.
    """

    try:
        return TranscriptionFrameMetadata(
            session_id=UUID(_required_string(event, "session_id")),
            turn_id=UUID(_required_string(event, "turn_id")),
            sequence=_required_nonnegative_int(event, "event_seq"),
            cancellation_generation=cancellation_generation,
            event_seq=_required_nonnegative_int(event, "event_seq"),
            is_final=_required_bool(event, "is_final"),
            language=_required_language(event, "language"),
            capture_started_monotonic_ns=_required_nonnegative_int(
                event, "capture_started_monotonic_ns"
            ),
            capture_ended_monotonic_ns=_required_nonnegative_int(
                event, "capture_ended_monotonic_ns"
            ),
        )
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise FrameMetadataError("invalid transcript event metadata") from exc


def _require_uuid(name: str, value: object) -> None:
    if not isinstance(value, UUID):
        raise FrameMetadataError(f"{name} must be a UUID")


def _require_nonnegative_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FrameMetadataError(f"{name} must be a non-negative integer")


def _require_positive_int(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise FrameMetadataError(f"{name} must be a positive integer")


def _required_string(event: Mapping[str, Any], name: str) -> str:
    value = event[name]
    if not isinstance(value, str):
        raise FrameMetadataError(f"{name} must be a string")
    return value


def _required_nonnegative_int(event: Mapping[str, Any], name: str) -> int:
    value = event[name]
    _require_nonnegative_int(name, value)
    return value


def _required_bool(event: Mapping[str, Any], name: str) -> bool:
    value = event[name]
    if not isinstance(value, bool):
        raise FrameMetadataError(f"{name} must be a bool")
    return value


def _required_language(event: Mapping[str, Any], name: str) -> VoiceLanguage:
    value = event[name]
    if value not in {"hi", "en", "hinglish"}:
        raise FrameMetadataError(f"{name} must be hi, en, or hinglish")
    return value


__all__ = [
    "AudioFrameMetadata",
    "FrameMetadataError",
    "PCMChunkFrameMetadata",
    "SentenceFrameMetadata",
    "TranscriptionFrameMetadata",
    "TypedVoiceFrame",
    "VoiceFrameMetadata",
    "VoiceFrameType",
    "metadata_from_transcript_event",
]
