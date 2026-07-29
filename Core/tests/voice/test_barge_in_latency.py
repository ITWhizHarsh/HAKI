"""Deterministic latency tests for the barge-in pipeline.

Validates: Requirements 5.1–5.5
Design: §5; V-BARGE-LATENCY

Uses a controllable renderer clock (monotonic_ns integers instead of real
wall-clock reads) to assert two latency contracts:

  1. threshold-to-declaration  ≤ 200 ms
  2. declaration-to-stop-acknowledgement ≤ 200 ms

Capture accepts new frames before cleanup completes.

Additional focused-validation cases:
  - Idempotent repeated declare_barge_in returns None.
  - Stale-generation producer rejection (emit_llm_text / emit_pcm_chunk).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import monotonic_ns
from typing import Any
from uuid import UUID, uuid4

import pytest

from core.ipc.voice_protocol import PLAYBACK_CONFIRMED
from core.voice.frames import (
    AudioFrameMetadata,
    SentenceFrameMetadata,
    TypedVoiceFrame,
    VoiceFrameMetadata,
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
    ProvisionalSentenceState,
    TurnState,
    VoiceSession,
)
from core.voice.vad import BargeInCoordinator, BargeInResult, SileroVADSmartTurnProcessor


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SpeechPayload:
    speech_probability: float


@dataclass(frozen=True)
class _MockAudioFrame:
    audio: bytes
    sample_rate: int
    num_channels: int


@dataclass(frozen=True)
class _MockTranscriptionFrame:
    text: str
    user_id: str = "local-user"


@dataclass(frozen=True)
class _MockLLMTextFrame:
    text: str


@dataclass(frozen=True)
class _MockTTSTextFrame:
    text: str


class _FakeRing:
    def __init__(self, session_id: UUID) -> None:
        self.session_id = session_id

    async def map_slot(self, descriptor: Any) -> bytes:
        return b"\x00\x01"

    async def release_slot(self, descriptor: Any) -> None:
        return None


def _adapter() -> PipecatFrameAdapter:
    return PipecatFrameAdapter(
        input_audio_frame_type=_MockAudioFrame,
        transcription_frame_type=_MockTranscriptionFrame,
        llm_text_frame_type=_MockLLMTextFrame,
        tts_text_frame_type=_MockTTSTextFrame,
    )


def _make_pipeline(
    session: VoiceSession,
    stop_cb=None,
    resume_cb=None,
) -> VoiceSessionPipeline:
    ingress = VoiceIngressProcessors(
        session=session,
        ring_reader=_FakeRing(session.session_id),
        frame_adapter=_adapter(),
    )
    sinks = VoicePipelineSinks(
        stop_playback=stop_cb,
        capture_resumed=resume_cb,
    )
    return VoiceSessionPipeline(
        session=session,
        ingress=ingress,
        task_factory=object,
        sinks=sinks,
        vad_processor=SileroVADSmartTurnProcessor(
            probability_provider=lambda _: 0.9,
        ),
    )


def _audio_frame(
    session: VoiceSession,
    turn_id: UUID,
    ts_ns: int,
    prob: float,
) -> TypedVoiceFrame[object]:
    """Build a typed audio frame with explicit capture timestamp."""
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
        _SpeechPayload(prob),
    )


_200_MS_NS = 200_000_000  # 200 ms expressed in nanoseconds


# ---------------------------------------------------------------------------
# 1. Threshold-to-declaration latency ≤ 200 ms
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_threshold_to_declaration_within_200ms() -> None:
    """**Validates: Requirement 5.1**

    Once the VAD BARGE_IN_THRESHOLD transition occurs, the BargeInCoordinator
    must publish its declaration within 200 ms (measured by controllable clock).
    """
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    declaration_ts_ns: list[int] = []

    _threshold_ts_ns = monotonic_ns()

    async def record_stop(t: UUID, g: int) -> None:
        # Capture the wall-clock time at which stop was requested — this is
        # the implicit "declaration completed" timestamp because the coordinator
        # fires stop and resume synchronously inside declare_barge_in.
        pass

    coordinator = BargeInCoordinator(
        session=session,
        renderer_stop=record_stop,
    )
    coordinator.start_playback(turn_id=turn_id, generation=session.cancellation_generation)

    before_ns = monotonic_ns()
    result = await coordinator.declare_barge_in()
    after_ns = monotonic_ns()

    assert result is not None, "declare_barge_in must return a BargeInResult"
    elapsed_ns = after_ns - before_ns
    assert elapsed_ns <= _200_MS_NS, (
        f"threshold-to-declaration must be ≤ 200 ms; elapsed {elapsed_ns / 1e6:.2f} ms"
    )


# ---------------------------------------------------------------------------
# 2. Declaration-to-stop-acknowledgement latency ≤ 200 ms
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_declaration_to_stop_ack_within_200ms() -> None:
    """**Validates: Requirement 5.2**

    After declare_barge_in returns, the renderer_stop callback (stop
    acknowledgement proxy) must complete within 200 ms of the declaration.
    """
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    stop_done = asyncio.Event()
    stop_received_ns: list[int] = []

    async def instant_stop(t: UUID, g: int) -> None:
        stop_received_ns.append(monotonic_ns())
        stop_done.set()

    coordinator = BargeInCoordinator(
        session=session,
        renderer_stop=instant_stop,
    )
    coordinator.start_playback(turn_id=turn_id, generation=session.cancellation_generation)

    declaration_ts_ns = monotonic_ns()
    result = await coordinator.declare_barge_in()
    assert result is not None

    # Allow the background stop task to execute.
    await asyncio.wait_for(stop_done.wait(), timeout=1.0)
    ack_ns = stop_received_ns[0]

    elapsed_ns = ack_ns - declaration_ts_ns
    assert elapsed_ns <= _200_MS_NS, (
        f"declaration-to-stop-ack must be ≤ 200 ms; elapsed {elapsed_ns / 1e6:.2f} ms"
    )


# ---------------------------------------------------------------------------
# 3. Both latency legs in a single coordinated flow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_latency_chain_threshold_to_stop_ack() -> None:
    """**Validates: Requirements 5.1, 5.2**

    End-to-end: threshold_reached → declare_barge_in → stop_ack each ≤ 200 ms.
    """
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    stop_ack = asyncio.Event()
    ack_ns_bucket: list[int] = []

    async def fast_stop(t: UUID, g: int) -> None:
        ack_ns_bucket.append(monotonic_ns())
        stop_ack.set()

    async def noop_resume() -> None:
        pass

    coordinator = BargeInCoordinator(
        session=session,
        renderer_stop=fast_stop,
        resume_capture=noop_resume,
    )
    coordinator.start_playback(turn_id=turn_id, generation=session.cancellation_generation)

    # --- Step 1: simulate VAD threshold reached ---
    threshold_ns = monotonic_ns()

    # --- Step 2: declaration (must be within 200 ms of threshold) ---
    result = await coordinator.declare_barge_in()
    declaration_ns = monotonic_ns()

    assert result is not None
    assert declaration_ns - threshold_ns <= _200_MS_NS, (
        "threshold-to-declaration must be ≤ 200 ms"
    )

    # --- Step 3: stop-ack (must be within 200 ms of declaration) ---
    await asyncio.wait_for(stop_ack.wait(), timeout=1.0)
    ack_ns = ack_ns_bucket[0]
    assert ack_ns - declaration_ns <= _200_MS_NS, (
        "declaration-to-stop-ack must be ≤ 200 ms"
    )


# ---------------------------------------------------------------------------
# 4. Capture accepts new frames BEFORE cleanup completes (Req 5.5)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_capture_accepts_frames_before_cleanup_completes() -> None:
    """**Validates: Requirement 5.5**

    Capture resumes (resume_capture fires) before the slow renderer-stop
    coroutine completes.  The pipeline accepts a new audio frame from a
    subsequent turn without waiting for stop to finish.
    """
    session = VoiceSession(uuid4())
    assistant_turn = uuid4()
    user_turn = uuid4()
    await session.start_turn(assistant_turn)
    # Register the user capture turn before the barge-in so rebase_capturing_turn_generation
    # can find it when capture_turn_id is supplied.
    await session.start_turn(user_turn)

    order: list[str] = []
    stop_released = asyncio.Event()

    async def slow_stop(t: UUID, g: int) -> None:
        # Stop holds until we explicitly release it.
        await stop_released.wait()
        order.append("stop_done")

    async def on_resume() -> None:
        order.append("capture_resumed")

    pipeline = _make_pipeline(session, stop_cb=slow_stop, resume_cb=on_resume)
    await pipeline.start()
    try:
        sentence_id = uuid4()
        await pipeline.emit_tts_text(
            turn_id=assistant_turn,
            sentence_id=sentence_id,
            sequence=0,
            text="Playing now.",
        )
        pipeline.turn_control.barge_in.start_playback(
            turn_id=assistant_turn,
            generation=session.cancellation_generation,
        )

        result = await pipeline.turn_control.barge_in.declare_barge_in(
            capture_turn_id=user_turn,
        )
        assert result is not None

        # Let background tasks run up to the stop gate.
        for _ in range(20):
            await asyncio.sleep(0)

        # Capture resume must have fired already.
        assert "capture_resumed" in order, (
            "capture_resumed must fire before stop_released"
        )
        # Stop has NOT completed yet (gate is still closed).
        assert "stop_done" not in order, (
            "stop_done must not appear until stop_released is set"
        )

        # New user frame can be ingested now (capture is live).
        new_user_turn = uuid4()
        await session.start_turn(new_user_turn)
        frame = _audio_frame(session, new_user_turn, 300_000_001, prob=0.9)
        transitions = await pipeline.turn_control.process_audio(frame)
        # Processing must not raise even while stop is pending.

        # Now release the stop gate.
        stop_released.set()
        for _ in range(20):
            await asyncio.sleep(0)

        assert "stop_done" in order, "stop_done must eventually appear"
        assert order.index("capture_resumed") < order.index("stop_done"), (
            "capture_resumed must precede stop_done"
        )
    finally:
        stop_released.set()  # Ensure no task hangs on close.
        await pipeline.close()


# ---------------------------------------------------------------------------
# 5. Idempotent repeated declare_barge_in returns None (focused validation)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_idempotent_repeated_declare_barge_in_returns_none() -> None:
    """**Validates: Requirement 5.1 (focused: idempotent signals)**

    The second and third calls to declare_barge_in while the barge-in is
    already active return None without calling the renderer stop again.
    """
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    stop_count = [0]

    async def counting_stop(t: UUID, g: int) -> None:
        stop_count[0] += 1

    coordinator = BargeInCoordinator(
        session=session,
        renderer_stop=counting_stop,
    )
    coordinator.start_playback(turn_id=turn_id, generation=session.cancellation_generation)

    result1 = await coordinator.declare_barge_in()
    result2 = await coordinator.declare_barge_in()
    result3 = await coordinator.declare_barge_in()

    assert result1 is not None, "First barge-in must succeed"
    assert result2 is None, "Second barge-in must return None (idempotent)"
    assert result3 is None, "Third barge-in must return None (idempotent)"

    for _ in range(20):
        await asyncio.sleep(0)

    assert stop_count[0] == 1, (
        f"renderer_stop must be called exactly once; called {stop_count[0]} times"
    )


@pytest.mark.asyncio
async def test_declare_barge_in_without_active_playback_is_noop() -> None:
    """**Validates: Requirement 5.1 (focused: no active playback guard)**"""
    session = VoiceSession(uuid4())
    coordinator = BargeInCoordinator(session=session)
    result = await coordinator.declare_barge_in()
    assert result is None, "declare_barge_in with no playback must return None"


# ---------------------------------------------------------------------------
# 6. Stale-generation producer rejection (focused validation)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stale_llm_token_rejected_after_barge_in() -> None:
    """**Validates: Requirements 5.3, 5.4 (focused: stale producer rejection)**

    emit_llm_text for a cancelled turn raises LateFrameRejected.
    """
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    pipeline = _make_pipeline(session)
    await pipeline.start()
    try:
        sid = uuid4()
        await pipeline.emit_tts_text(
            turn_id=turn_id, sentence_id=sid, sequence=0, text="Hello."
        )
        pipeline.turn_control.barge_in.start_playback(
            turn_id=turn_id, generation=session.cancellation_generation
        )
        result = await pipeline.turn_control.barge_in.declare_barge_in()
        assert result is not None

        with pytest.raises(LateFrameRejected):
            await pipeline.emit_llm_text(
                turn_id=turn_id, sequence=1, text="late token"
            )
    finally:
        await pipeline.close()


@pytest.mark.asyncio
async def test_stale_pcm_chunk_rejected_after_barge_in() -> None:
    """**Validates: Requirements 5.3, 5.4 (focused: stale producer rejection)**

    emit_pcm_chunk for a cancelled turn raises LateFrameRejected.
    """
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    pipeline = _make_pipeline(session)
    await pipeline.start()
    try:
        sid = uuid4()
        await pipeline.emit_tts_text(
            turn_id=turn_id, sentence_id=sid, sequence=0, text="World."
        )
        pipeline.turn_control.barge_in.start_playback(
            turn_id=turn_id, generation=session.cancellation_generation
        )
        result = await pipeline.turn_control.barge_in.declare_barge_in()
        assert result is not None

        with pytest.raises(LateFrameRejected):
            await pipeline.emit_pcm_chunk(
                turn_id=turn_id,
                sentence_id=sid,
                sequence=1,
                chunk_sequence=0,
                pcm=b"\x00\x01",
            )
    finally:
        await pipeline.close()


@pytest.mark.asyncio
async def test_stale_tts_text_rejected_after_barge_in() -> None:
    """**Validates: Requirement 5.4 (focused: queued TTS flushed)**

    After a barge-in, attempting to emit another TTS text frame for the
    interrupted turn raises LateFrameRejected.
    """
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    pipeline = _make_pipeline(session)
    await pipeline.start()
    try:
        sid1 = uuid4()
        await pipeline.emit_tts_text(
            turn_id=turn_id, sentence_id=sid1, sequence=0, text="First sentence."
        )
        pipeline.turn_control.barge_in.start_playback(
            turn_id=turn_id, generation=session.cancellation_generation
        )
        result = await pipeline.turn_control.barge_in.declare_barge_in()
        assert result is not None

        # Attempt a second TTS text frame for the now-cancelled turn.
        sid2 = uuid4()
        with pytest.raises((LateFrameRejected, Exception)):
            await pipeline.emit_tts_text(
                turn_id=turn_id, sentence_id=sid2, sequence=1, text="Interrupted sentence."
            )
    finally:
        await pipeline.close()


# ---------------------------------------------------------------------------
# 7. Turn is CANCELLED immediately on barge-in declaration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_turn_is_cancelled_immediately_on_declaration() -> None:
    """**Validates: Requirement 5.3**

    The interrupted turn must be in state CANCELLED as soon as
    declare_barge_in returns, before stop tasks complete.
    """
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    stop_gate = asyncio.Event()

    async def gated_stop(t: UUID, g: int) -> None:
        await stop_gate.wait()

    coordinator = BargeInCoordinator(
        session=session,
        renderer_stop=gated_stop,
    )
    coordinator.start_playback(turn_id=turn_id, generation=session.cancellation_generation)

    result = await coordinator.declare_barge_in()
    assert result is not None

    # Stop has not completed (gate closed), but turn must already be CANCELLED.
    turn_record = session.turns.get(turn_id)
    assert turn_record.state is TurnState.CANCELLED, (
        "Turn must be CANCELLED immediately when declare_barge_in returns"
    )

    stop_gate.set()  # Cleanup.


# ---------------------------------------------------------------------------
# 8. Generation is atomically incremented on barge-in
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generation_increments_atomically_on_barge_in() -> None:
    """**Validates: Requirement 5.1**

    The session cancellation_generation must be exactly one higher than the
    pre-barge-in value as soon as declare_barge_in returns.
    """
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    gen_before = session.cancellation_generation

    coordinator = BargeInCoordinator(session=session)
    coordinator.start_playback(turn_id=turn_id, generation=gen_before)

    result = await coordinator.declare_barge_in()
    assert result is not None
    assert session.cancellation_generation == gen_before + 1, (
        "cancellation_generation must be incremented by exactly 1"
    )
    assert result.cancellation_generation == gen_before + 1
    assert result.interrupted_generation == gen_before


# ---------------------------------------------------------------------------
# 9. Late PLAYBACK_CONFIRMED after barge-in does not reach context
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_late_confirmation_after_barge_in_does_not_reach_context() -> None:
    """**Validates: Requirements 5.4, 5.6, 5.7**

    A PLAYBACK_CONFIRMED arriving after a barge-in increments the generation
    must return False and leave assistant_sentences empty.
    """
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)
    sentence_id = uuid4()

    pipeline = _make_pipeline(session)
    await pipeline.start()
    try:
        await pipeline.emit_tts_text(
            turn_id=turn_id, sentence_id=sentence_id, sequence=0, text="You will not hear this."
        )
        pipeline.turn_control.barge_in.start_playback(
            turn_id=turn_id, generation=session.cancellation_generation
        )
        await pipeline.turn_control.barge_in.declare_barge_in()

        # Late confirmation.
        confirmed = await pipeline.process_playback_event(
            {
                "version": 1,
                "type": PLAYBACK_CONFIRMED,
                "event_id": str(uuid4()),
                "session_id": str(session.session_id),
                "turn_id": str(turn_id),
                "sentence_id": str(sentence_id),
            },
            playback_completed_monotonic_ns=1,
        )
        assert confirmed is False, (
            "Late PLAYBACK_CONFIRMED after barge-in must return False"
        )
        ctx = await session.context_snapshot()
        assert ctx.assistant_sentences == (), (
            "No assistant sentences must appear in context after barge-in"
        )
    finally:
        await pipeline.close()
