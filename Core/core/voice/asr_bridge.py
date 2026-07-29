"""Secure local ring and transcript ingress for the Pipecat voice graph.

This module is deliberately an adapter boundary: microphone bytes are acquired
only from an authenticated same-UID ring reader, copied into an
``InputAudioRawFrame`` for Silero, and released before downstream processing.
Transcript ingress accepts the text-only UDS contract and keeps turn metadata
in typed wrappers instead of serialising it into transcript text.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Mapping, Protocol, runtime_checkable
from uuid import UUID

from core.ipc.voice_protocol import (
    EVENT_ACK,
    TRANSCRIPT_EVENT,
    ValidatedMessage,
    VoiceProtocolError,
    validate_message,
)

from .frames import (
    AudioFrameMetadata,
    TranscriptionFrameMetadata,
    TypedVoiceFrame,
    VoiceFrameType,
    metadata_from_transcript_event,
)
from .session import (
    DuplicateTurnError,
    FrameOrderingError,
    LateFrameRejected,
    TurnQueueName,
    UnknownTurnError,
    VoiceSession,
    VoiceSessionError,
)

SILERO_SAMPLE_RATE_HZ = 16_000
SILERO_CHANNELS = 1


class VoiceIngressError(RuntimeError):
    """A safe, content-free local voice ingress failure."""


class CaptureVADUnavailable(VoiceIngressError):
    """The authenticated ring cannot safely provide input to Silero."""


class RingSlotRejected(VoiceIngressError):
    """A local ring descriptor is malformed, stale, or out of order."""


class TranscriptIngressRejected(VoiceIngressError):
    """A transcript cannot be turned into a sequenced Pipecat frame."""


class CaptureVADState(str, Enum):
    """Availability of the ring-backed capture/VAD path."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class RingSlotDescriptor:
    """Metadata-only reference to one authenticated shared-memory ring slot.

    The inherited ring capability and owner validation stay inside the ring
    reader. This descriptor intentionally has no PCM, encoded audio, socket
    payload, or diagnostic-content field.
    """

    session_id: UUID
    turn_id: UUID
    slot_index: int
    sequence: int
    captured_monotonic_ns: int
    sample_rate_hz: int
    channels: int
    byte_length: int

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, UUID) or not isinstance(self.turn_id, UUID):
            raise RingSlotRejected("ring_descriptor_identity_invalid")
        for name, value in (
            ("slot_index", self.slot_index),
            ("sequence", self.sequence),
            ("captured_monotonic_ns", self.captured_monotonic_ns),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RingSlotRejected(f"ring_descriptor_{name}_invalid")
        if self.sample_rate_hz != SILERO_SAMPLE_RATE_HZ:
            raise RingSlotRejected("ring_descriptor_sample_rate_invalid")
        if self.channels != SILERO_CHANNELS:
            raise RingSlotRejected("ring_descriptor_channels_invalid")
        if (
            not isinstance(self.byte_length, int)
            or isinstance(self.byte_length, bool)
            or self.byte_length <= 0
            or self.byte_length % 2
        ):
            raise RingSlotRejected("ring_descriptor_length_invalid")


@runtime_checkable
class AuthenticatedRingSlotReader(Protocol):
    """Same-UID ring mapping port supplied through inherited session config.

    Implementations must validate the ring descriptor's owner UID, mode, shared
    memory identity, and launch-inherited capability before returning bytes.
    They must never send the mapped bytes through the transcript socket.
    """

    session_id: UUID

    async def map_slot(self, descriptor: RingSlotDescriptor) -> bytes | bytearray | memoryview:
        """Temporarily map one authenticated slot."""

    async def release_slot(self, descriptor: RingSlotDescriptor) -> None:
        """Release a slot immediately after its Pipecat frame is copied."""


@runtime_checkable
class PipecatIngressFrameFactory(Protocol):
    """Small compatibility port around Pipecat's mandatory frame constructors."""

    def create_input_audio_frame(
        self, *, audio: bytes, sample_rate_hz: int, channels: int
    ) -> object:
        """Create an ``InputAudioRawFrame`` for Silero only."""

    def create_transcription_frame(self, *, text: str) -> object:
        """Create a ``TranscriptionFrame`` without modifying ``text``."""


@dataclass(frozen=True, slots=True)
class RingIngressResult:
    """One ring-derived Pipecat input frame and an explicit dropped-frame gap."""

    frame: TypedVoiceFrame[object]
    gap_before: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class TranscriptIngressResult:
    """A text-only protocol acknowledgement plus an optional accepted wrapper."""

    acknowledgement: Mapping[str, object]
    frame: TypedVoiceFrame[object] | None

    @property
    def accepted(self) -> bool:
        return self.acknowledgement["status"] == "accepted"


@dataclass(slots=True)
class _TranscriptSequenceState:
    next_event_seq: int | None = None
    finalized: bool = False


class AudioRingIngress:
    """Map/release authenticated slots into Silero ``InputAudioRawFrame`` values.

    A slot is copied into an immutable ``bytes`` payload before frame creation,
    then released before this method returns. Any mapping, validation, release,
    or frame-construction failure makes capture/VAD unavailable; it never
    diverts microphone data to the text/control UDS.
    """

    def __init__(
        self,
        *,
        session: VoiceSession,
        ring_reader: AuthenticatedRingSlotReader,
        frame_factory: PipecatIngressFrameFactory,
    ) -> None:
        self._session = session
        self._ring_reader = ring_reader
        self._frame_factory = frame_factory
        self._next_sequence_by_turn: dict[UUID, int] = {}
        self._state = CaptureVADState.AVAILABLE
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CaptureVADState:
        return self._state

    async def ingest(self, descriptor: RingSlotDescriptor) -> RingIngressResult:
        """Create one typed ``InputAudioRawFrame`` and immediately release its slot."""
        async with self._lock:
            if self._state is CaptureVADState.UNAVAILABLE:
                raise CaptureVADUnavailable("capture_vad_unavailable")
            try:
                self._validate_descriptor_identity(descriptor)
                record = await _ensure_turn(self._session, descriptor.turn_id)
                if record.is_terminal:
                    raise RingSlotRejected("ring_descriptor_terminal_turn")

                expected = self._next_sequence_by_turn.get(descriptor.turn_id)
                if expected is not None and descriptor.sequence < expected:
                    raise RingSlotRejected("ring_descriptor_out_of_order")
                gap_before = (
                    (expected, descriptor.sequence - 1)
                    if expected is not None and descriptor.sequence > expected
                    else None
                )

                # The `finally` makes release happen before any caller can send
                # the resulting frame to Silero or another processor.
                mapped = await self._ring_reader.map_slot(descriptor)
                try:
                    audio = bytes(mapped)
                    if len(audio) != descriptor.byte_length:
                        raise RingSlotRejected("ring_slot_length_mismatch")
                    pipecat_frame = self._frame_factory.create_input_audio_frame(
                        audio=audio,
                        sample_rate_hz=descriptor.sample_rate_hz,
                        channels=descriptor.channels,
                    )
                finally:
                    await self._ring_reader.release_slot(descriptor)

                self._next_sequence_by_turn[descriptor.turn_id] = descriptor.sequence + 1
                metadata = AudioFrameMetadata(
                    session_id=descriptor.session_id,
                    turn_id=descriptor.turn_id,
                    sequence=descriptor.sequence,
                    cancellation_generation=record.cancellation_generation,
                    captured_monotonic_ns=descriptor.captured_monotonic_ns,
                    sample_rate_hz=descriptor.sample_rate_hz,
                    channels=descriptor.channels,
                )
                return RingIngressResult(
                    frame=TypedVoiceFrame(
                        frame_type=VoiceFrameType.INPUT_AUDIO,
                        metadata=metadata,
                        payload=pipecat_frame,
                    ),
                    gap_before=gap_before,
                )
            except Exception as exc:
                self._state = CaptureVADState.UNAVAILABLE
                if isinstance(exc, CaptureVADUnavailable):
                    raise
                raise CaptureVADUnavailable("ring_input_unavailable") from exc

    def _validate_descriptor_identity(self, descriptor: RingSlotDescriptor) -> None:
        if descriptor.session_id != self._session.session_id:
            raise RingSlotRejected("ring_descriptor_stale_session")
        if getattr(self._ring_reader, "session_id", None) != self._session.session_id:
            raise RingSlotRejected("ring_reader_session_mismatch")


class TranscriptSocketIngress:
    """Validate, sequence, wrap, and acknowledge text-only transcript events.

    A returned ``accepted`` acknowledgement proves the event passed strict wire
    validation *and* the session's ordering/queue acceptance. A discarded ACK
    is returned for a valid event that cannot be sequenced; malformed events
    raise so the secure UDS server can apply its protocol-level rejection.
    """

    def __init__(
        self,
        *,
        session: VoiceSession,
        frame_factory: PipecatIngressFrameFactory,
    ) -> None:
        self._session = session
        self._frame_factory = frame_factory
        self._sequence_by_turn: dict[UUID, _TranscriptSequenceState] = {}
        self._lock = asyncio.Lock()

    async def ingest(
        self, message: Mapping[str, Any] | ValidatedMessage
    ) -> TranscriptIngressResult:
        """Accept a transcript only after UDS and session sequencing succeed."""
        validated = validate_message(message)
        if validated.message_type != TRANSCRIPT_EVENT:
            raise TranscriptIngressRejected("transcript_event_required")
        data = validated.data
        event_id = data["event_id"]

        async with self._lock:
            try:
                if UUID(data["session_id"]) != self._session.session_id:
                    raise VoiceProtocolError("stale_session")
                turn_id = UUID(data["turn_id"])
                sequence_state = self._sequence_by_turn.setdefault(turn_id, _TranscriptSequenceState())
                _validate_transcript_sequence(sequence_state, data)

                record = await _ensure_turn(self._session, turn_id)
                metadata = metadata_from_transcript_event(
                    data,
                    cancellation_generation=record.cancellation_generation,
                )
                # Text remains exactly the normalized UDS text. Turn/session and
                # sequencing data live in `TranscriptionFrameMetadata` above.
                pipecat_frame = self._frame_factory.create_transcription_frame(text=data["text"])
                frame = TypedVoiceFrame(
                    frame_type=VoiceFrameType.TRANSCRIPTION,
                    metadata=metadata,
                    payload=pipecat_frame,
                )
                queue = TurnQueueName.CONTROL if metadata.is_final else TurnQueueName.PARTIAL
                await self._session.accept_frame(frame, queue=queue)
                _commit_transcript_sequence(sequence_state, metadata)
                return TranscriptIngressResult(
                    acknowledgement=_acknowledgement(event_id, "accepted"),
                    frame=frame,
                )
            except (VoiceProtocolError, FrameOrderingError, LateFrameRejected, VoiceSessionError) as exc:
                return TranscriptIngressResult(
                    acknowledgement=_acknowledgement(event_id, "discarded", _safe_reason(exc)),
                    frame=None,
                )
            except Exception as exc:
                return TranscriptIngressResult(
                    acknowledgement=_acknowledgement(event_id, "discarded", "ingress_rejected"),
                    frame=None,
                )


def _validate_transcript_sequence(
    state: _TranscriptSequenceState, data: Mapping[str, Any]
) -> None:
    if state.finalized:
        raise VoiceProtocolError("duplicate_final")
    event_seq = data["event_seq"]
    if state.next_event_seq is not None and event_seq != state.next_event_seq:
        raise VoiceProtocolError("invalid_event_sequence")


def _commit_transcript_sequence(
    state: _TranscriptSequenceState, metadata: TranscriptionFrameMetadata
) -> None:
    state.next_event_seq = metadata.event_seq + 1
    state.finalized = metadata.is_final


async def _ensure_turn(session: VoiceSession, turn_id: UUID):
    try:
        return session.turns.get(turn_id)
    except UnknownTurnError:
        try:
            return await session.start_turn(turn_id)
        except DuplicateTurnError:
            # A concurrent audio/transcript ingress may have registered it.
            return session.turns.get(turn_id)


def _acknowledgement(
    event_id: str,
    status: Literal["accepted", "discarded"],
    reason: str | None = None,
) -> dict[str, object]:
    acknowledgement: dict[str, object] = {
        "version": 1,
        "type": EVENT_ACK,
        "event_id": event_id,
        "status": status,
    }
    if reason is not None:
        acknowledgement["reason"] = reason
    return acknowledgement


def _safe_reason(error: BaseException) -> str:
    if isinstance(error, VoiceProtocolError):
        return error.reason
    if isinstance(error, FrameOrderingError):
        return "invalid_event_sequence"
    if isinstance(error, LateFrameRejected):
        return "late_turn"
    return "ingress_rejected"


__all__ = [
    "AuthenticatedRingSlotReader",
    "AudioRingIngress",
    "CaptureVADState",
    "CaptureVADUnavailable",
    "PipecatIngressFrameFactory",
    "RingIngressResult",
    "RingSlotDescriptor",
    "RingSlotRejected",
    "SILERO_CHANNELS",
    "SILERO_SAMPLE_RATE_HZ",
    "TranscriptIngressRejected",
    "TranscriptIngressResult",
    "TranscriptSocketIngress",
    "VoiceIngressError",
]
