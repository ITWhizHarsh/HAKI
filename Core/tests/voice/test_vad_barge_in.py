"""Focused deterministic coverage for Task 6 smart-turn and interruption logic.

Validates: Requirements 4.7, 5.1–5.8
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from core.ipc.voice_protocol import PLAYBACK_CONFIRMED, TRANSCRIPT_EVENT
from core.voice.asr_bridge import RingSlotDescriptor
from core.voice.frames import AudioFrameMetadata, TranscriptionFrameMetadata, TypedVoiceFrame, VoiceFrameType
from core.voice.pipeline import PipecatFrameAdapter, VoiceIngressProcessors, VoicePipelineSinks, VoiceSessionPipeline
from core.voice.session import LateFrameRejected, ProvisionalSentenceState, TurnState, VoiceSession
from core.voice.vad import (
    BargeInCoordinator,
    SileroVADSmartTurnProcessor,
    SmartTurnVADConfig,
    TurnJoinProcessor,
    VADTransition,
    VADTransitionKind,
)


@dataclass(frozen=True)
class SpeechInput:
    speech_probability: float


@dataclass(frozen=True)
class TranscriptPayload:
    text: str


@dataclass(frozen=True)
class MockInputAudioRawFrame:
    audio: bytes
    sample_rate: int
    num_channels: int


@dataclass(frozen=True)
class MockTranscriptionFrame:
    text: str
    user_id: str


@dataclass(frozen=True)
class MockLLMTextFrame:
    text: str


@dataclass(frozen=True)
class MockTTSTextFrame:
    text: str


class Ring:
    def __init__(self, session_id: UUID, payloads: dict[int, bytes]) -> None:
        self.session_id = session_id
        self.payloads = payloads

    async def map_slot(self, descriptor: RingSlotDescriptor) -> bytes:
        return self.payloads[descriptor.slot_index]

    async def release_slot(self, descriptor: RingSlotDescriptor) -> None:
        return None


def _audio(session: VoiceSession, turn_id: UUID, timestamp_ns: int, probability: float) -> TypedVoiceFrame[object]:
    return TypedVoiceFrame(
        VoiceFrameType.INPUT_AUDIO,
        AudioFrameMetadata(
            session_id=session.session_id,
            turn_id=turn_id,
            sequence=timestamp_ns // 100_000_000,
            cancellation_generation=session.cancellation_generation,
            captured_monotonic_ns=timestamp_ns,
            sample_rate_hz=16_000,
            channels=1,
        ),
        SpeechInput(probability),
    )


def _final_transcript(session: VoiceSession, turn_id: UUID, text: str = "Kal meeting reschedule kar do") -> TypedVoiceFrame[object]:
    return TypedVoiceFrame(
        VoiceFrameType.TRANSCRIPTION,
        TranscriptionFrameMetadata(
            session_id=session.session_id,
            turn_id=turn_id,
            sequence=0,
            cancellation_generation=session.cancellation_generation,
            event_seq=0,
            is_final=True,
            language="hinglish",
            capture_started_monotonic_ns=0,
            capture_ended_monotonic_ns=0,
        ),
        TranscriptPayload(text),
    )


def _adapter() -> PipecatFrameAdapter:
    return PipecatFrameAdapter(
        input_audio_frame_type=MockInputAudioRawFrame,
        transcription_frame_type=MockTranscriptionFrame,
        llm_text_frame_type=MockLLMTextFrame,
        tts_text_frame_type=MockTTSTextFrame,
    )


def _slot(session_id: UUID, turn_id: UUID, slot: int, sequence: int, timestamp_ns: int) -> RingSlotDescriptor:
    return RingSlotDescriptor(
        session_id=session_id,
        turn_id=turn_id,
        slot_index=slot,
        sequence=sequence,
        captured_monotonic_ns=timestamp_ns,
        sample_rate_hz=16_000,
        channels=1,
        byte_length=2,
    )


def _transcript_message(session_id: UUID, turn_id: UUID) -> dict[str, object]:
    return {
        "version": 1,
        "type": TRANSCRIPT_EVENT,
        "event_id": str(uuid4()),
        "session_id": str(session_id),
        "turn_id": str(turn_id),
        "event_seq": 0,
        "text": "Kal meeting reschedule kar do",
        "is_final": True,
        "language": "hinglish",
        "capture_started_monotonic_ns": 0,
        "capture_ended_monotonic_ns": 0,
    }


@pytest.mark.asyncio
async def test_smart_turn_silence_resets_and_finalizes_exactly_at_threshold() -> None:
    """Speech resets silence; only the exact 800 ms post-speech boundary finalizes."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    processor = SileroVADSmartTurnProcessor(config=SmartTurnVADConfig())

    assert processor.process(_audio(session, turn_id, 0, 0.8), playback_active=False) == ()
    assert processor.process(_audio(session, turn_id, 199_000_000, 0.8), playback_active=False) == ()
    assert processor.process(_audio(session, turn_id, 200_000_000, 0.8), playback_active=False)[0].kind is VADTransitionKind.SPEECH_STARTED
    assert processor.process(_audio(session, turn_id, 300_000_000, 0.0), playback_active=False) == ()
    assert processor.process(_audio(session, turn_id, 1_099_000_000, 0.0), playback_active=False) == ()
    assert processor.process(_audio(session, turn_id, 1_100_000_000, 0.8), playback_active=False)[0].kind is VADTransitionKind.SPEECH_RESUMED
    assert processor.process(_audio(session, turn_id, 1_200_000_000, 0.0), playback_active=False) == ()
    assert processor.process(_audio(session, turn_id, 1_999_000_000, 0.0), playback_active=False) == ()
    final = processor.process(_audio(session, turn_id, 2_000_000_000, 0.0), playback_active=False)
    assert [transition.kind for transition in final] == [VADTransitionKind.SMART_TURN_FINALIZED]


@pytest.mark.asyncio
async def test_turn_join_never_starts_from_transcript_only_and_updates_partial_ui() -> None:
    """ASR finality waits for VAD finality while partial text remains UI-only."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)
    ready = []
    partials = []
    join = TurnJoinProcessor(session=session, on_turn_ready=ready.append, on_partial_ui=partials.append)
    partial = TypedVoiceFrame(
        VoiceFrameType.TRANSCRIPTION,
        TranscriptionFrameMetadata(
            session_id=session.session_id,
            turn_id=turn_id,
            sequence=0,
            cancellation_generation=0,
            event_seq=0,
            is_final=False,
            language="hinglish",
            capture_started_monotonic_ns=0,
            capture_ended_monotonic_ns=0,
        ),
        TranscriptPayload("Kal meeting"),
    )

    assert not await join.process_transcription(partial)
    assert partials == [partial]
    assert not await join.process_transcription(_final_transcript(session, turn_id))
    assert ready == []
    assert (await session.context_snapshot()).messages == ()
    assert session.turns.get(turn_id).state is TurnState.FINAL_PENDING_SILENCE

    assert await join.process_vad_transition(
        VADTransition(turn_id, VADTransitionKind.SMART_TURN_FINALIZED, 800_000_000, 0)
    )
    assert [request.text for request in ready] == ["Kal meeting reschedule kar do"]
    assert [message.role for message in (await session.context_snapshot()).messages] == ["user"]
    assert session.turns.get(turn_id).state is TurnState.REASONING


@pytest.mark.asyncio
async def test_pipeline_turn_join_uses_16khz_timeline_and_requires_final_silence() -> None:
    """A final transcript alone does not reach the ready sink before the 800 ms boundary."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    ready = []
    ingress = VoiceIngressProcessors(
        session=session,
        ring_reader=Ring(
            session.session_id,
            {0: b"\x01\x00", 1: b"\x01\x00", 2: b"\x00\x00", 3: b"\x00\x00"},
        ),
        frame_adapter=_adapter(),
    )
    pipeline = VoiceSessionPipeline(
        session=session,
        ingress=ingress,
        task_factory=object,
        sinks=VoicePipelineSinks(turn_ready=ready.append),
        vad_processor=SileroVADSmartTurnProcessor(
            probability_provider=lambda payload: 0.9 if payload.audio == b"\x01\x00" else 0.0
        ),
    )

    await pipeline.start()
    try:
        assert (await pipeline.ingest_transcript_message(_transcript_message(session.session_id, turn_id))).accepted
        assert ready == []
        await pipeline.ingest_ring_slot(_slot(session.session_id, turn_id, 0, 0, 0))
        await pipeline.ingest_ring_slot(_slot(session.session_id, turn_id, 1, 1, 200_000_000))
        assert ready == []
        await pipeline.ingest_ring_slot(_slot(session.session_id, turn_id, 2, 2, 300_000_000))
        await pipeline.ingest_ring_slot(_slot(session.session_id, turn_id, 3, 3, 1_100_000_000))
        assert [request.turn_id for request in ready] == [turn_id]
    finally:
        await pipeline.close()


@pytest.mark.asyncio
async def test_barge_in_cancels_generation_and_late_output_or_confirmation_cannot_revive_it() -> None:
    """A 200 ms playback interruption drains work and excludes cancelled text from context."""
    session = VoiceSession(uuid4())
    assistant_turn, user_turn, sentence_id = uuid4(), uuid4(), uuid4()
    stopped = asyncio.Event()
    resumed = asyncio.Event()
    ingress = VoiceIngressProcessors(
        session=session,
        ring_reader=Ring(session.session_id, {0: b"\x01\x00", 1: b"\x01\x00"}),
        frame_adapter=_adapter(),
    )

    async def stop_renderer(turn_id: UUID, generation: int) -> None:
        assert turn_id == assistant_turn
        assert generation == 1
        stopped.set()

    async def resume_capture() -> None:
        resumed.set()

    pipeline = VoiceSessionPipeline(
        session=session,
        ingress=ingress,
        task_factory=object,
        sinks=VoicePipelineSinks(stop_playback=stop_renderer, capture_resumed=resume_capture),
        vad_processor=SileroVADSmartTurnProcessor(probability_provider=lambda _payload: 0.9),
    )
    await pipeline.start()
    try:
        await session.start_turn(assistant_turn)
        await pipeline.emit_tts_text(turn_id=assistant_turn, sentence_id=sentence_id, sequence=0, text="I can do that.")
        generation_zero_task = asyncio.create_task(asyncio.sleep(60))
        pipeline.register_cancellable_task(turn_id=assistant_turn, generation=0, task=generation_zero_task)

        await pipeline.ingest_ring_slot(_slot(session.session_id, user_turn, 0, 0, 0))
        await pipeline.ingest_ring_slot(_slot(session.session_id, user_turn, 1, 1, 200_000_000))
        for _ in range(10):
            if stopped.is_set() and resumed.is_set() and generation_zero_task.cancelled():
                break
            await asyncio.sleep(0)

        assert session.turns.get(assistant_turn).state is TurnState.CANCELLED
        assert session.turns.get(user_turn).cancellation_generation == 1
        assert stopped.is_set() and resumed.is_set()
        assert generation_zero_task.cancelled()
        assert await session.playback_ledger.state_for(sentence_id) is ProvisionalSentenceState.CANCELLED
        with pytest.raises(LateFrameRejected):
            await pipeline.emit_llm_text(turn_id=assistant_turn, sequence=1, text="late token")
        with pytest.raises(LateFrameRejected):
            await pipeline.emit_pcm_chunk(
                turn_id=assistant_turn,
                sentence_id=sentence_id,
                sequence=1,
                chunk_sequence=0,
                pcm=b"\x01\x00",
            )
        assert not await pipeline.process_playback_event(
            {
                "version": 1,
                "type": PLAYBACK_CONFIRMED,
                "event_id": str(uuid4()),
                "session_id": str(session.session_id),
                "turn_id": str(assistant_turn),
                "sentence_id": str(sentence_id),
            },
            playback_completed_monotonic_ns=1,
        )
        assert (await session.context_snapshot()).assistant_sentences == ()
    finally:
        if not generation_zero_task.done():
            generation_zero_task.cancel()
        await pipeline.close()


@pytest.mark.asyncio
async def test_playback_ledger_appends_only_confirmed_sentences_in_confirmation_order() -> None:
    """Unconfirmed/cancelled text is never included in the later LLM context."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    first, second, unconfirmed = uuid4(), uuid4(), uuid4()
    await session.start_turn(turn_id)
    for sentence_id, text in ((first, "First."), (second, "Second."), (unconfirmed, "Never heard.")):
        assert await session.playback_ledger.register(
            turn_id=turn_id,
            sentence_id=sentence_id,
            text=text,
            cancellation_generation=0,
        )

    assert await session.playback_ledger.confirm(turn_id=turn_id, sentence_id=second, playback_completed_monotonic_ns=20)
    assert await session.playback_ledger.confirm(turn_id=turn_id, sentence_id=first, playback_completed_monotonic_ns=10)
    await session.cancel_turn(turn_id)
    assert not await session.playback_ledger.confirm(
        turn_id=turn_id,
        sentence_id=unconfirmed,
        playback_completed_monotonic_ns=30,
    )

    context = await session.context_snapshot()
    assert [sentence.text for sentence in context.assistant_sentences] == ["Second.", "First."]
    assert [message.text for message in context.messages] == ["Second.", "First."]
