"""Focused ingress coverage for Pipecat audio and transcription adapters.

Validates: Requirements 3.4–3.6, 4.2–4.3, 4.6
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from core.ipc.voice_protocol import TRANSCRIPT_EVENT
from core.voice.asr_bridge import (
    CaptureVADState,
    CaptureVADUnavailable,
    RingSlotDescriptor,
)
from core.voice.frames import AudioFrameMetadata, TranscriptionFrameMetadata, VoiceFrameType
from core.voice.pipeline import PipecatFrameAdapter, VoiceIngressProcessors
from core.voice.session import VoiceSession


@dataclass(frozen=True)
class MockInputAudioRawFrame:
    audio: bytes
    sample_rate: int
    num_channels: int


@dataclass(frozen=True)
class MockTranscriptionFrame:
    text: str
    user_id: str


class MockAuthenticatedRing:
    def __init__(self, session_id: UUID, *, payloads: dict[int, bytes] | None = None, fail: bool = False) -> None:
        self.session_id = session_id
        self.payloads = payloads or {}
        self.fail = fail
        self.map_calls: list[RingSlotDescriptor] = []
        self.release_calls: list[RingSlotDescriptor] = []

    async def map_slot(self, descriptor: RingSlotDescriptor) -> bytes:
        self.map_calls.append(descriptor)
        if self.fail:
            raise OSError("ring unavailable")
        return self.payloads[descriptor.slot_index]

    async def release_slot(self, descriptor: RingSlotDescriptor) -> None:
        self.release_calls.append(descriptor)


def _adapter() -> PipecatFrameAdapter:
    return PipecatFrameAdapter(
        input_audio_frame_type=MockInputAudioRawFrame,
        transcription_frame_type=MockTranscriptionFrame,
    )


def _slot(*, session_id: UUID, turn_id: UUID, slot: int, sequence: int, length: int) -> RingSlotDescriptor:
    return RingSlotDescriptor(
        session_id=session_id,
        turn_id=turn_id,
        slot_index=slot,
        sequence=sequence,
        captured_monotonic_ns=123 + sequence,
        sample_rate_hz=16_000,
        channels=1,
        byte_length=length,
    )


def _transcript(*, session_id: UUID, turn_id: UUID, sequence: int, final: bool = False) -> dict[str, object]:
    return {
        "version": 1,
        "type": TRANSCRIPT_EVENT,
        "event_id": str(uuid4()),
        "session_id": str(session_id),
        "turn_id": str(turn_id),
        "event_seq": sequence,
        "text": "Kal meeting reschedule kar do",
        "is_final": final,
        "language": "hinglish",
        "capture_started_monotonic_ns": 10,
        "capture_ended_monotonic_ns": 20,
    }


@pytest.mark.asyncio
async def test_ring_unavailable_marks_capture_vad_unavailable_without_socket_fallback() -> None:
    """Ring mapping failure leaves capture/VAD unavailable and exposes no microphone route."""
    session = VoiceSession(uuid4())
    ring = MockAuthenticatedRing(session.session_id, fail=True)
    ingress = VoiceIngressProcessors(session=session, ring_reader=ring, frame_adapter=_adapter())

    with pytest.raises(CaptureVADUnavailable, match="ring_input_unavailable"):
        await ingress.ingest_ring_slot(
            _slot(session_id=session.session_id, turn_id=uuid4(), slot=0, sequence=0, length=2)
        )

    assert ingress.audio_ring.state is CaptureVADState.UNAVAILABLE
    assert ring.release_calls == []
    assert not hasattr(ingress, "send_socket_audio")


@pytest.mark.asyncio
async def test_ring_descriptor_gap_is_explicit_and_slot_is_released_before_result() -> None:
    """Dropped descriptors remain an explicit gap while the mapped slot releases immediately."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    ring = MockAuthenticatedRing(session.session_id, payloads={0: b"\x01\x00", 1: b"\x02\x00"})
    ingress = VoiceIngressProcessors(session=session, ring_reader=ring, frame_adapter=_adapter())

    first = await ingress.ingest_ring_slot(
        _slot(session_id=session.session_id, turn_id=turn_id, slot=0, sequence=4, length=2)
    )
    gapped = await ingress.ingest_ring_slot(
        _slot(session_id=session.session_id, turn_id=turn_id, slot=1, sequence=6, length=2)
    )

    assert first.gap_before is None
    assert gapped.gap_before == (5, 5)
    assert ring.release_calls == ring.map_calls
    assert gapped.frame.frame_type is VoiceFrameType.INPUT_AUDIO
    assert isinstance(gapped.frame.metadata, AudioFrameMetadata)
    assert gapped.frame.metadata.sequence == 6
    assert gapped.frame.payload == MockInputAudioRawFrame(b"\x02\x00", 16_000, 1)


@pytest.mark.asyncio
async def test_socket_rejection_returns_discarded_ack_only_after_sequence_check() -> None:
    """A valid-but-gapped transcript receives a discarded ACK and never reaches Pipecat."""
    session = VoiceSession(uuid4())
    ring = MockAuthenticatedRing(session.session_id)
    ingress = VoiceIngressProcessors(session=session, ring_reader=ring, frame_adapter=_adapter())
    turn_id = uuid4()

    accepted = await ingress.ingest_transcript_message(
        _transcript(session_id=session.session_id, turn_id=turn_id, sequence=4)
    )
    rejected_message = _transcript(session_id=session.session_id, turn_id=turn_id, sequence=6)
    rejected = await ingress.ingest_transcript_message(rejected_message)

    assert accepted.accepted
    assert rejected.acknowledgement == {
        "version": 1,
        "type": "EVENT_ACK",
        "event_id": rejected_message["event_id"],
        "status": "discarded",
        "reason": "invalid_event_sequence",
    }
    record = session.turns.get(turn_id)
    assert record.queues.partial.qsize() == 1


@pytest.mark.asyncio
async def test_transcript_frame_keeps_typed_metadata_outside_normalized_text() -> None:
    """Accepted text becomes a TranscriptionFrame with turn metadata in its wrapper."""
    session = VoiceSession(uuid4())
    ring = MockAuthenticatedRing(session.session_id)
    ingress = VoiceIngressProcessors(session=session, ring_reader=ring, frame_adapter=_adapter())
    turn_id = uuid4()
    message = _transcript(session_id=session.session_id, turn_id=turn_id, sequence=11, final=True)

    result = await ingress.ingest_transcript_message(message)

    assert result.accepted
    assert result.frame is not None
    assert result.frame.frame_type is VoiceFrameType.TRANSCRIPTION
    assert isinstance(result.frame.metadata, TranscriptionFrameMetadata)
    assert result.frame.metadata.turn_id == turn_id
    assert result.frame.metadata.event_seq == 11
    assert result.frame.payload == MockTranscriptionFrame(
        text="Kal meeting reschedule kar do", user_id="local-user"
    )
    assert str(turn_id) not in result.frame.payload.text
    assert "event_seq" not in result.frame.payload.text
