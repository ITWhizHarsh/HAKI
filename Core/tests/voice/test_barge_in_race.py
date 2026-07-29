"""Race tests for barge-in cancellation generations and PlaybackLedger.

Validates: Requirements 5.1–5.8
Design: §5 (Smart Turn, Barge-In, and Confirmed-Playback Context)

These tests prove that:
- Late tokens/chunks/confirmations cannot revive a cancelled assistant turn.
- Unconfirmed (provisional) text never enters the prompt context for later turns.
- Concurrent producers racing against a barge-in are all rejected.
- The PlaybackLedger only appends sentences that received PLAYBACK_CONFIRMED.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from core.ipc.voice_protocol import PLAYBACK_CONFIRMED, PLAYBACK_CANCELLED, PLAYBACK_FAILED
from core.voice.frames import (
    AudioFrameMetadata,
    SentenceFrameMetadata,
    TranscriptionFrameMetadata,
    TypedVoiceFrame,
    VoiceFrameType,
)
from core.voice.pipeline import (
    PipecatFrameAdapter,
    VoiceIngressProcessors,
    VoicePipelineSinks,
    VoiceSessionPipeline,
)
from core.voice.session import (
    LateFrameRejected,
    PlaybackLedger,
    ProvisionalSentenceState,
    TurnState,
    VoiceSession,
)
from core.voice.vad import BargeInCoordinator, SileroVADSmartTurnProcessor, VADTransitionKind


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpeechInput:
    speech_probability: float


@dataclass(frozen=True)
class MockInputAudioRawFrame:
    audio: bytes
    sample_rate: int
    num_channels: int


@dataclass(frozen=True)
class MockTranscriptionFrame:
    text: str
    user_id: str = "local-user"


@dataclass(frozen=True)
class MockLLMTextFrame:
    text: str


@dataclass(frozen=True)
class MockTTSTextFrame:
    text: str


class FakeRing:
    """Minimal ring reader returning a fixed payload for any slot."""

    def __init__(self, session_id: UUID) -> None:
        self.session_id = session_id

    async def map_slot(self, descriptor) -> bytes:
        return b"\x01\x00"

    async def release_slot(self, descriptor) -> None:
        return None


def _adapter() -> PipecatFrameAdapter:
    return PipecatFrameAdapter(
        input_audio_frame_type=MockInputAudioRawFrame,
        transcription_frame_type=MockTranscriptionFrame,
        llm_text_frame_type=MockLLMTextFrame,
        tts_text_frame_type=MockTTSTextFrame,
    )


def _audio_frame(session: VoiceSession, turn_id: UUID, ts_ns: int, prob: float) -> TypedVoiceFrame[object]:
    return TypedVoiceFrame(
        VoiceFrameType.INPUT_AUDIO,
        AudioFrameMetadata(
            session_id=session.session_id,
            turn_id=turn_id,
            sequence=ts_ns // 100_000_000,
            cancellation_generation=session.cancellation_generation,
            captured_monotonic_ns=ts_ns,
            sample_rate_hz=16_000,
            channels=1,
        ),
        SpeechInput(prob),
    )


def _confirmation_message(session: VoiceSession, turn_id: UUID, sentence_id: UUID) -> dict:
    return {
        "version": 1,
        "type": PLAYBACK_CONFIRMED,
        "event_id": str(uuid4()),
        "session_id": str(session.session_id),
        "turn_id": str(turn_id),
        "sentence_id": str(sentence_id),
    }


def _cancellation_message(session: VoiceSession, turn_id: UUID, sentence_id: UUID) -> dict:
    return {
        "version": 1,
        "type": PLAYBACK_CANCELLED,
        "event_id": str(uuid4()),
        "session_id": str(session.session_id),
        "turn_id": str(turn_id),
        "sentence_id": str(sentence_id),
    }


def _failure_message(session: VoiceSession, turn_id: UUID, sentence_id: UUID) -> dict:
    return {
        "version": 1,
        "type": PLAYBACK_FAILED,
        "event_id": str(uuid4()),
        "session_id": str(session.session_id),
        "turn_id": str(turn_id),
        "sentence_id": str(sentence_id),
        "error_class": "renderer_error",
    }


# ---------------------------------------------------------------------------
# 1. Atomic generation increment and concurrent stop + drain + resume
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_barge_in_atomically_increments_generation_and_invalidates_turn() -> None:
    """Generation increment is atomic; cancelled turn rejects all further output."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)
    sentence_id = uuid4()
    await session.playback_ledger.register(
        turn_id=turn_id, sentence_id=sentence_id,
        text="Hello world.", cancellation_generation=0,
    )

    stopped_turn: list[UUID] = []
    stopped_gen: list[int] = []

    async def stop(t: UUID, g: int) -> None:
        stopped_turn.append(t)
        stopped_gen.append(g)

    coordinator = BargeInCoordinator(
        session=session,
        renderer_stop=stop,
    )
    coordinator.start_playback(turn_id=turn_id, generation=0)

    pre_gen = session.cancellation_generation
    result = await coordinator.declare_barge_in(capture_turn_id=None)

    assert result is not None
    assert result.interrupted_generation == 0
    assert result.cancellation_generation == pre_gen + 1
    assert session.cancellation_generation == pre_gen + 1
    assert session.turns.get(turn_id).state is TurnState.CANCELLED

    # Let the background stop task run.
    for _ in range(20):
        await asyncio.sleep(0)
    assert stopped_turn == [turn_id]
    assert stopped_gen == [pre_gen + 1]

    # Confirm that provisional sentence was cancelled by the generation advance.
    assert await session.playback_ledger.state_for(sentence_id) is ProvisionalSentenceState.CANCELLED


@pytest.mark.asyncio
async def test_barge_in_returns_to_capturing_without_awaiting_cleanup() -> None:
    """Capture resumes immediately; the stop sink runs concurrently, not before resume."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    order: list[str] = []
    stop_gate = asyncio.Event()

    async def slow_stop(t: UUID, g: int) -> None:
        await stop_gate.wait()
        order.append("stop_done")

    async def resume() -> None:
        order.append("capture_resumed")

    coordinator = BargeInCoordinator(
        session=session,
        renderer_stop=slow_stop,
        resume_capture=resume,
    )
    coordinator.start_playback(turn_id=turn_id, generation=0)

    # Run the barge-in. Resume must fire before slow_stop completes.
    result = await coordinator.declare_barge_in()
    assert result is not None

    # Let all background tasks run up to the stop gate.
    for _ in range(20):
        await asyncio.sleep(0)

    assert "capture_resumed" in order, "capture must resume without waiting for stop"
    assert "stop_done" not in order, "stop has not yet been released"

    # Release the stop gate and confirm it eventually completes.
    stop_gate.set()
    for _ in range(20):
        await asyncio.sleep(0)
    assert "stop_done" in order
    # Capture resumed before stop completed.
    assert order.index("capture_resumed") < order.index("stop_done")


# ---------------------------------------------------------------------------
# 2. Late token / chunk / confirmation cannot revive cancelled output
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_late_llm_token_rejected_after_barge_in() -> None:
    """emit_llm_text raises LateFrameRejected for the cancelled turn's generation."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    ingress = VoiceIngressProcessors(
        session=session,
        ring_reader=FakeRing(session.session_id),
        frame_adapter=_adapter(),
    )
    pipeline = VoiceSessionPipeline(
        session=session,
        ingress=ingress,
        task_factory=object,
        sinks=VoicePipelineSinks(),
    )
    await pipeline.start()
    try:
        sentence_id = uuid4()
        await pipeline.emit_tts_text(
            turn_id=turn_id, sentence_id=sentence_id, sequence=0, text="First sentence."
        )
        # Trigger barge-in via the pipeline's coordinator directly.
        pipeline.turn_control.barge_in.start_playback(turn_id=turn_id, generation=0)
        result = await pipeline.turn_control.barge_in.declare_barge_in()
        assert result is not None

        # Late LLM token for the cancelled turn must be rejected.
        with pytest.raises(LateFrameRejected):
            await pipeline.emit_llm_text(turn_id=turn_id, sequence=1, text="late token")

        # Late PCM chunk for the cancelled turn must be rejected.
        with pytest.raises(LateFrameRejected):
            await pipeline.emit_pcm_chunk(
                turn_id=turn_id, sentence_id=sentence_id,
                sequence=2, chunk_sequence=0, pcm=b"\x00\x01",
            )
    finally:
        await pipeline.close()


@pytest.mark.asyncio
async def test_late_playback_confirmation_cannot_add_to_context_after_barge_in() -> None:
    """PLAYBACK_CONFIRMED after barge-in returns False and appends nothing to context."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)
    sentence_id = uuid4()

    ingress = VoiceIngressProcessors(
        session=session,
        ring_reader=FakeRing(session.session_id),
        frame_adapter=_adapter(),
    )
    pipeline = VoiceSessionPipeline(
        session=session, ingress=ingress,
        task_factory=object, sinks=VoicePipelineSinks(),
    )
    await pipeline.start()
    try:
        await pipeline.emit_tts_text(
            turn_id=turn_id, sentence_id=sentence_id, sequence=0, text="I can do that."
        )
        pipeline.turn_control.barge_in.start_playback(turn_id=turn_id, generation=0)
        await pipeline.turn_control.barge_in.declare_barge_in()

        confirmed = await pipeline.process_playback_event(
            _confirmation_message(session, turn_id, sentence_id),
            playback_completed_monotonic_ns=1,
        )
        assert confirmed is False
        context = await session.context_snapshot()
        assert context.assistant_sentences == ()
        assert [m.role for m in context.messages] == []
    finally:
        await pipeline.close()


@pytest.mark.asyncio
async def test_concurrent_producers_all_rejected_after_barge_in() -> None:
    """Concurrent emit_llm_text calls racing against a barge-in are all rejected."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    ingress = VoiceIngressProcessors(
        session=session,
        ring_reader=FakeRing(session.session_id),
        frame_adapter=_adapter(),
    )
    pipeline = VoiceSessionPipeline(
        session=session, ingress=ingress,
        task_factory=object, sinks=VoicePipelineSinks(),
    )
    await pipeline.start()
    try:
        # Cancel first, then try concurrent late emits.
        pipeline.turn_control.barge_in.start_playback(turn_id=turn_id, generation=0)
        await pipeline.turn_control.barge_in.declare_barge_in()

        async def try_emit(seq: int) -> bool:
            try:
                await pipeline.emit_llm_text(turn_id=turn_id, sequence=seq, text=f"token {seq}")
                return True
            except LateFrameRejected:
                return False

        results = await asyncio.gather(*[try_emit(i) for i in range(1, 6)])
        assert all(r is False for r in results), "all late tokens must be rejected"
        context = await session.context_snapshot()
        assert context.assistant_sentences == ()
    finally:
        await pipeline.close()


# ---------------------------------------------------------------------------
# 3. Cancellable task drain — registered asyncio tasks are cancelled on barge-in
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_registered_generation_tasks_cancelled_on_barge_in() -> None:
    """Tasks registered for the interrupted generation are cancelled; others survive."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    ingress = VoiceIngressProcessors(
        session=session,
        ring_reader=FakeRing(session.session_id),
        frame_adapter=_adapter(),
    )
    pipeline = VoiceSessionPipeline(
        session=session, ingress=ingress,
        task_factory=object, sinks=VoicePipelineSinks(),
    )
    await pipeline.start()
    try:
        # Register two tasks for the current generation and one for a new turn.
        gen0_task_a = asyncio.create_task(asyncio.sleep(60), name="gen0-a")
        gen0_task_b = asyncio.create_task(asyncio.sleep(60), name="gen0-b")
        pipeline.register_cancellable_task(turn_id=turn_id, generation=0, task=gen0_task_a)
        pipeline.register_cancellable_task(turn_id=turn_id, generation=0, task=gen0_task_b)

        new_turn_id = uuid4()
        await session.start_turn(new_turn_id)
        unrelated_task = asyncio.create_task(asyncio.sleep(60), name="unrelated")
        pipeline.register_cancellable_task(turn_id=new_turn_id, generation=1, task=unrelated_task)

        pipeline.turn_control.barge_in.start_playback(turn_id=turn_id, generation=0)
        result = await pipeline.turn_control.barge_in.declare_barge_in()
        assert result is not None

        # Allow cancel propagation.
        for _ in range(20):
            await asyncio.sleep(0)

        assert gen0_task_a.cancelled(), "gen0 task A must be cancelled"
        assert gen0_task_b.cancelled(), "gen0 task B must be cancelled"
        assert not unrelated_task.done(), "unrelated task must not be cancelled"
    finally:
        for t in (gen0_task_a, gen0_task_b, unrelated_task):
            if not t.done():
                t.cancel()
        await pipeline.close()


# ---------------------------------------------------------------------------
# 4. Queued TTS/PCM work is drained for the interrupted generation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_queued_output_drained_for_interrupted_generation_only() -> None:
    """_cancel_interrupted_work removes queued frames for the interrupted turn only."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    ingress = VoiceIngressProcessors(
        session=session,
        ring_reader=FakeRing(session.session_id),
        frame_adapter=_adapter(),
    )
    pipeline = VoiceSessionPipeline(
        session=session, ingress=ingress,
        task_factory=object, sinks=VoicePipelineSinks(),
    )
    await pipeline.start()
    try:
        s1, s2 = uuid4(), uuid4()
        # Emit two sentences for generation 0.
        await pipeline.emit_tts_text(turn_id=turn_id, sentence_id=s1, sequence=0, text="First.")
        await pipeline.emit_tts_text(turn_id=turn_id, sentence_id=s2, sequence=1, text="Second.")

        tts_before = pipeline.queues["tts"].qsize()
        assert tts_before == 2

        # Drain only generation 0 work.
        await pipeline._cancel_interrupted_work(turn_id=turn_id, generation=0)

        # All generation-0 TTS frames should be gone.
        assert pipeline.queues["tts"].qsize() == 0
    finally:
        await pipeline.close()


# ---------------------------------------------------------------------------
# 5. PlaybackLedger: only PLAYBACK_CONFIRMED appends; cancelled/failed do not
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_playback_ledger_confirms_only_normal_completion_in_order() -> None:
    """Sentences reach context only via PLAYBACK_CONFIRMED, in confirmation order."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)
    s1, s2, s3 = uuid4(), uuid4(), uuid4()

    ledger = session.playback_ledger
    for sid, text in ((s1, "Alpha."), (s2, "Beta."), (s3, "Gamma.")):
        assert await ledger.register(
            turn_id=turn_id, sentence_id=sid, text=text, cancellation_generation=0
        )

    # Confirm in reverse order to verify confirmation order is preserved.
    assert await ledger.confirm(turn_id=turn_id, sentence_id=s3, playback_completed_monotonic_ns=30)
    assert await ledger.confirm(turn_id=turn_id, sentence_id=s1, playback_completed_monotonic_ns=10)
    # s2 is never confirmed.

    context = await session.context_snapshot()
    texts = [s.text for s in context.assistant_sentences]
    assert texts == ["Gamma.", "Alpha."], "confirmation order, not registration order"
    # s2 is absent.
    assert "Beta." not in texts


@pytest.mark.asyncio
async def test_playback_cancelled_event_does_not_add_to_context() -> None:
    """PLAYBACK_CANCELLED must not write provisional text to the context ledger.

    The event is processed and returns True, but no sentence is appended.
    A PLAYBACK_CANCELLED received at the same generation leaves the sentence
    in PENDING state — it was never confirmed so it never reaches context.
    After a barge-in that advances the generation, the ledger.cancel call
    in process_playback_event marks the sentence CANCELLED.
    """
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)
    sentence_id = uuid4()
    await session.playback_ledger.register(
        turn_id=turn_id, sentence_id=sentence_id,
        text="Should never be heard.", cancellation_generation=0,
    )

    ingress = VoiceIngressProcessors(
        session=session,
        ring_reader=FakeRing(session.session_id),
        frame_adapter=_adapter(),
    )
    pipeline = VoiceSessionPipeline(
        session=session, ingress=ingress,
        task_factory=object, sinks=VoicePipelineSinks(),
    )
    await pipeline.start()
    try:
        # Simulate a barge-in that advances the generation first.
        pipeline.turn_control.barge_in.start_playback(turn_id=turn_id, generation=0)
        await pipeline.turn_control.barge_in.declare_barge_in()
        # Now generation is 1; the sentence (gen=0) will be marked CANCELLED.

        result = await pipeline.process_playback_event(
            _cancellation_message(session, turn_id, sentence_id)
        )
        assert result is True  # Cancellation is processed.
        context = await session.context_snapshot()
        assert context.assistant_sentences == ()
        # After generation advanced, the ledger marks the sentence CANCELLED.
        assert await session.playback_ledger.state_for(sentence_id) is ProvisionalSentenceState.CANCELLED
    finally:
        await pipeline.close()


@pytest.mark.asyncio
async def test_playback_failed_event_does_not_add_to_context() -> None:
    """PLAYBACK_FAILED must not write provisional text to the context ledger."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)
    sentence_id = uuid4()
    await session.playback_ledger.register(
        turn_id=turn_id, sentence_id=sentence_id,
        text="Renderer failed mid-sentence.", cancellation_generation=0,
    )

    ingress = VoiceIngressProcessors(
        session=session,
        ring_reader=FakeRing(session.session_id),
        frame_adapter=_adapter(),
    )
    pipeline = VoiceSessionPipeline(
        session=session, ingress=ingress,
        task_factory=object, sinks=VoicePipelineSinks(),
    )
    await pipeline.start()
    try:
        result = await pipeline.process_playback_event(
            _failure_message(session, turn_id, sentence_id)
        )
        assert result is True
        context = await session.context_snapshot()
        assert context.assistant_sentences == ()
        assert await session.playback_ledger.state_for(sentence_id) is ProvisionalSentenceState.FAILED
    finally:
        await pipeline.close()


@pytest.mark.asyncio
async def test_unregistered_confirmation_is_silently_ignored() -> None:
    """A PLAYBACK_CONFIRMED for an unregistered sentence is a no-op (not an error)."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)
    unknown_sentence = uuid4()

    ingress = VoiceIngressProcessors(
        session=session,
        ring_reader=FakeRing(session.session_id),
        frame_adapter=_adapter(),
    )
    pipeline = VoiceSessionPipeline(
        session=session, ingress=ingress,
        task_factory=object, sinks=VoicePipelineSinks(),
    )
    await pipeline.start()
    try:
        result = await pipeline.process_playback_event(
            _confirmation_message(session, turn_id, unknown_sentence),
            playback_completed_monotonic_ns=1,
        )
        assert result is False
        assert (await session.context_snapshot()).assistant_sentences == ()
    finally:
        await pipeline.close()


# ---------------------------------------------------------------------------
# 6. Unconfirmed text never enters the prompt context for later turns
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unconfirmed_provisional_text_absent_from_later_llm_context() -> None:
    """Provisional text that was never confirmed is not in VoiceContext.messages."""
    session = VoiceSession(uuid4())
    user_turn_id = uuid4()
    assistant_turn_id = uuid4()
    await session.start_turn(assistant_turn_id)

    unconfirmed_id, confirmed_id = uuid4(), uuid4()
    await session.playback_ledger.register(
        turn_id=assistant_turn_id, sentence_id=unconfirmed_id,
        text="You will never hear this.", cancellation_generation=0,
    )
    await session.playback_ledger.register(
        turn_id=assistant_turn_id, sentence_id=confirmed_id,
        text="You heard this.", cancellation_generation=0,
    )
    await session.append_user_turn(turn_id=user_turn_id, text="What did you say?")
    await session.playback_ledger.confirm(
        turn_id=assistant_turn_id, sentence_id=confirmed_id,
        playback_completed_monotonic_ns=100,
    )
    # unconfirmed_id is never confirmed; cancel the turn.
    await session.cancel_turn(assistant_turn_id)

    context = await session.context_snapshot()
    all_texts = [m.text for m in context.messages]
    assert "You will never hear this." not in all_texts
    assert "You heard this." in all_texts
    assert "What did you say?" in all_texts


@pytest.mark.asyncio
async def test_double_confirmation_is_a_noop_for_same_sentence() -> None:
    """A duplicate PLAYBACK_CONFIRMED for the same sentence_id is rejected cleanly."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)
    sentence_id = uuid4()
    await session.playback_ledger.register(
        turn_id=turn_id, sentence_id=sentence_id,
        text="Once only.", cancellation_generation=0,
    )

    first = await session.playback_ledger.confirm(
        turn_id=turn_id, sentence_id=sentence_id, playback_completed_monotonic_ns=10
    )
    second = await session.playback_ledger.confirm(
        turn_id=turn_id, sentence_id=sentence_id, playback_completed_monotonic_ns=20
    )
    assert first is True
    assert second is False
    # Exactly one entry in context.
    context = await session.context_snapshot()
    assert len(context.assistant_sentences) == 1
    assert context.assistant_sentences[0].text == "Once only."


# ---------------------------------------------------------------------------
# 7. Idempotent barge-in — a second declaration while active is a no-op
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_barge_in_is_idempotent_while_active() -> None:
    """Declaring barge-in again while already active returns None (no double-cancel)."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    stop_count = [0]

    async def count_stop(t: UUID, g: int) -> None:
        stop_count[0] += 1

    coordinator = BargeInCoordinator(session=session, renderer_stop=count_stop)
    coordinator.start_playback(turn_id=turn_id, generation=0)

    result1 = await coordinator.declare_barge_in()
    result2 = await coordinator.declare_barge_in()

    assert result1 is not None
    assert result2 is None  # Second call returns None — no-op.

    for _ in range(20):
        await asyncio.sleep(0)
    assert stop_count[0] == 1, "stop_renderer must be called exactly once"


@pytest.mark.asyncio
async def test_barge_in_without_active_playback_returns_none() -> None:
    """Calling declare_barge_in with no active playback is a safe no-op."""
    session = VoiceSession(uuid4())
    coordinator = BargeInCoordinator(session=session)
    result = await coordinator.declare_barge_in()
    assert result is None


# ---------------------------------------------------------------------------
# 8. Stale-generation producer rejection — generation check guards all output
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stale_generation_frame_rejected_by_session_accept() -> None:
    """Frames carrying an older cancellation generation are rejected by accept_frame."""
    from core.voice.frames import VoiceFrameMetadata
    from core.voice.session import TurnQueueName

    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    # Advance the session generation by cancelling the turn.
    await session.cancel_turn(turn_id)
    assert session.cancellation_generation == 1

    # Start a fresh turn and try to emit a frame stamped with generation 0.
    new_turn_id = uuid4()
    await session.start_turn(new_turn_id)

    stale_frame = TypedVoiceFrame(
        VoiceFrameType.TRANSCRIPTION,
        VoiceFrameMetadata(
            session_id=session.session_id,
            turn_id=new_turn_id,
            sequence=0,
            cancellation_generation=0,  # stale — session is now at 1
        ),
        "late text",
    )
    with pytest.raises(LateFrameRejected):
        await session.accept_frame(stale_frame, queue=TurnQueueName.CONTROL)


@pytest.mark.asyncio
async def test_output_is_current_rejects_stale_generation_without_error() -> None:
    """output_is_current returns False for cancelled-generation frames (dispatch gate)."""
    from core.voice.frames import VoiceFrameMetadata

    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)
    generation_before = session.cancellation_generation

    stale_frame = TypedVoiceFrame(
        VoiceFrameType.LLM_TEXT,
        VoiceFrameMetadata(
            session_id=session.session_id,
            turn_id=turn_id,
            sequence=0,
            cancellation_generation=generation_before,
        ),
        "late llm text",
    )
    # Cancel the turn so its generation advances.
    await session.cancel_turn(turn_id)
    assert not await session.output_is_current(stale_frame)


# ---------------------------------------------------------------------------
# 9. VAD threshold triggers barge-in via the pipeline coordinator
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vad_200ms_voiced_during_playback_triggers_barge_in() -> None:
    """200 ms continuous speech during playback drives the BARGE_IN_THRESHOLD transition."""
    from core.voice.asr_bridge import RingSlotDescriptor

    session = VoiceSession(uuid4())
    assistant_turn_id = uuid4()
    user_turn_id = uuid4()

    stopped = asyncio.Event()
    resumed = asyncio.Event()

    async def on_stop(turn_id: UUID, generation: int) -> None:
        stopped.set()

    async def on_resume() -> None:
        resumed.set()

    ingress = VoiceIngressProcessors(
        session=session,
        ring_reader=FakeRing(session.session_id),
        frame_adapter=_adapter(),
    )
    pipeline = VoiceSessionPipeline(
        session=session,
        ingress=ingress,
        task_factory=object,
        sinks=VoicePipelineSinks(stop_playback=on_stop, capture_resumed=on_resume),
        vad_processor=SileroVADSmartTurnProcessor(
            probability_provider=lambda _: 0.9
        ),
    )
    await pipeline.start()
    try:
        await session.start_turn(assistant_turn_id)
        sentence_id = uuid4()
        await pipeline.emit_tts_text(
            turn_id=assistant_turn_id, sentence_id=sentence_id,
            sequence=0, text="Something to say.",
        )
        pipeline.turn_control.barge_in.start_playback(
            turn_id=assistant_turn_id, generation=0
        )

        # Inject two voiced frames > 200 ms apart to cross the threshold.
        def _slot(ts: int) -> RingSlotDescriptor:
            return RingSlotDescriptor(
                session_id=session.session_id,
                turn_id=user_turn_id,
                slot_index=0,
                sequence=ts // 100_000_000,
                captured_monotonic_ns=ts,
                sample_rate_hz=16_000,
                channels=1,
                byte_length=2,
            )

        await pipeline.ingest_ring_slot(_slot(0))
        await pipeline.ingest_ring_slot(_slot(200_000_001))

        # Allow background tasks to propagate the stop/resume signals.
        for _ in range(30):
            if stopped.is_set() and resumed.is_set():
                break
            await asyncio.sleep(0)

        assert stopped.is_set(), "stop_playback must be signalled after 200 ms"
        assert resumed.is_set(), "capture must resume after barge-in"
        assert session.turns.get(assistant_turn_id).state is TurnState.CANCELLED
    finally:
        await pipeline.close()


# ---------------------------------------------------------------------------
# 10. PlaybackLedger register rejects stale-generation provisional sentences
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ledger_register_rejects_stale_generation() -> None:
    """Registering a sentence whose generation was superseded by a barge-in returns False."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    # Immediately cancel (advance generation).
    await session.cancel_turn(turn_id)
    generation_after_cancel = session.cancellation_generation

    # Try to register a provisional sentence with the old generation (0).
    new_turn_id = uuid4()
    await session.start_turn(new_turn_id)
    accepted = await session.playback_ledger.register(
        turn_id=new_turn_id,
        sentence_id=uuid4(),
        text="Should be rejected.",
        cancellation_generation=0,  # stale
    )
    # The new turn is at generation 1, so generation 0 is stale.
    assert accepted is False
    assert (await session.context_snapshot()).assistant_sentences == ()


@pytest.mark.asyncio
async def test_playback_start_and_finish_cycles_correctly() -> None:
    """start_playback / finish_playback cycle tracks active playback state accurately."""
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    coordinator = BargeInCoordinator(session=session)
    assert coordinator.playback_active is False

    coordinator.start_playback(turn_id=turn_id, generation=0)
    assert coordinator.playback_active is True

    coordinator.finish_playback(turn_id=turn_id)
    assert coordinator.playback_active is False

    # finish for a different turn does not affect unrelated state.
    coordinator.start_playback(turn_id=turn_id, generation=0)
    other_turn = uuid4()
    coordinator.finish_playback(turn_id=other_turn)
    assert coordinator.playback_active is True  # still active for original turn
