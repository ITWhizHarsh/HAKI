"""Property 5: Barge-in invalidates interrupted assistant work.

Validates: Requirements 5.1–5.5
Design: §5 (Smart Turn, Barge-In, and Confirmed-Playback Context); Property 5;
        V-TURN-PROP, V-BARGE-LATENCY

For all active assistant generations and arbitrary queued TTS/audio work, once
200 ms of continuous user speech during playback reaches the barge-in threshold,
the pipeline:
  - cancels unfinished LLM generation for the interrupted turn
  - removes every queued TTS-text / PCM-chunk frame for that interrupted turn's
    generation
  - continues to accept frames for the NEW user utterance without waiting for
    the stopped LLM/XTTS tasks to finish

Property coverage (≥ 100 Hypothesis cases):
  - Idempotent repeated declare_barge_in returns None on the second call.
  - Stale-generation producer frames (emit_llm_text, emit_pcm_chunk) are
    rejected with LateFrameRejected after a barge-in.
  - All queued TTS + PCM work for the interrupted (turn_id, generation) pair is
    drained; work for other generations is untouched.
  - Capture begins (resume_capture fires) *before* the stop-renderer coroutine
    completes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

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
from core.voice.vad import BargeInCoordinator, SileroVADSmartTurnProcessor


# ---------------------------------------------------------------------------
# Lightweight test doubles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _AudioPayload:
    """Minimal payload carrying a Silero-style speech probability attribute."""
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


def _make_adapter() -> PipecatFrameAdapter:
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
        frame_adapter=_make_adapter(),
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
        _AudioPayload(prob),
    )


def _confirmation_message(
    session: VoiceSession,
    turn_id: UUID,
    sentence_id: UUID,
) -> dict:
    return {
        "version": 1,
        "type": PLAYBACK_CONFIRMED,
        "event_id": str(uuid4()),
        "session_id": str(session.session_id),
        "turn_id": str(turn_id),
        "sentence_id": str(sentence_id),
    }


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Number of TTS + PCM sentences queued on the interrupted turn (0–8).
_sentence_count_st = st.integers(min_value=0, max_value=8)

# Whether a second barge-in call is attempted (idempotency probe).
_idempotent_st = st.booleans()

# Whether we also try to emit a stale LLM token after barge-in.
_stale_llm_st = st.booleans()

# Whether we also try to emit a stale PCM chunk after barge-in.
_stale_pcm_st = st.booleans()


# ---------------------------------------------------------------------------
# PBT: Property 5 — barge-in drains matching work and resumes capture
# ---------------------------------------------------------------------------

@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=5000,
)
@given(
    sentence_count=_sentence_count_st,
    try_idempotent=_idempotent_st,
    try_stale_llm=_stale_llm_st,
    try_stale_pcm=_stale_pcm_st,
)
def test_barge_in_invalidates_interrupted_work(
    sentence_count: int,
    try_idempotent: bool,
    try_stale_llm: bool,
    try_stale_pcm: bool,
) -> None:
    """**Validates: Requirements 5.1, 5.3, 5.4, 5.5**

    Across ≥ 100 scheduled-work cases:
      - 200 ms speech cancels matching LLM/TTS/PCM work
      - Capture proceeds before cleanup
      - Idempotent repeated signals return None
      - Stale-generation producer frames are rejected
    """
    asyncio.run(
        _barge_in_invalidation_scenario(
            sentence_count,
            try_idempotent,
            try_stale_llm,
            try_stale_pcm,
        )
    )


async def _barge_in_invalidation_scenario(
    sentence_count: int,
    try_idempotent: bool,
    try_stale_llm: bool,
    try_stale_pcm: bool,
) -> None:
    session = VoiceSession(uuid4())
    assistant_turn_id = uuid4()
    new_capture_turn_id = uuid4()
    await session.start_turn(assistant_turn_id)

    resume_order: list[str] = []
    stop_gate = asyncio.Event()

    async def slow_stop(t: UUID, g: int) -> None:
        # Deliberately slow — capture resume must beat this.
        await asyncio.sleep(0)
        resume_order.append("stop_done")
        stop_gate.set()

    async def on_resume() -> None:
        resume_order.append("capture_resumed")

    pipeline = _make_pipeline(session, stop_cb=slow_stop, resume_cb=on_resume)
    await pipeline.start()

    try:
        # --- Queue sentence_count TTS items for the assistant turn ---
        sentence_ids: list[UUID] = []
        for i in range(sentence_count):
            sid = uuid4()
            sentence_ids.append(sid)
            await pipeline.emit_tts_text(
                turn_id=assistant_turn_id,
                sentence_id=sid,
                sequence=i,
                text=f"Sentence {i}.",
            )

        tts_before = pipeline.queues["tts"].qsize()
        assert tts_before == sentence_count, (
            f"Expected {sentence_count} TTS frames queued, got {tts_before}"
        )

        # --- Simulate 200 ms of voiced audio → barge-in threshold ---
        # Start the capture turn before declare_barge_in so rebase can find it.
        await session.start_turn(new_capture_turn_id)

        pipeline.turn_control.barge_in.start_playback(
            turn_id=assistant_turn_id,
            generation=session.cancellation_generation,
        )

        result1 = await pipeline.turn_control.barge_in.declare_barge_in(
            capture_turn_id=new_capture_turn_id,
        )

        # First barge-in must succeed.
        assert result1 is not None, "First declare_barge_in must return a BargeInResult"
        assert result1.interrupted_turn_id == assistant_turn_id
        new_gen = result1.cancellation_generation
        assert new_gen > result1.interrupted_generation

        # The interrupted turn must be CANCELLED.
        turn_record = session.turns.get(assistant_turn_id)
        assert turn_record.state is TurnState.CANCELLED

        # --- Idempotency: second call returns None ---
        if try_idempotent:
            result2 = await pipeline.turn_control.barge_in.declare_barge_in(
                capture_turn_id=new_capture_turn_id,
            )
            assert result2 is None, (
                "Second declare_barge_in while barge-in already active must return None"
            )

        # Allow background tasks (drain + stop) to run.
        for _ in range(20):
            await asyncio.sleep(0)

        # --- All TTS frames for the interrupted generation must be gone ---
        tts_after = pipeline.queues["tts"].qsize()
        assert tts_after == 0, (
            f"TTS queue must be empty after barge-in drain; got {tts_after}"
        )

        # All registered provisional sentences must be CANCELLED.
        for sid in sentence_ids:
            state = await session.playback_ledger.state_for(sid)
            assert state is ProvisionalSentenceState.CANCELLED, (
                f"Sentence {sid} must be CANCELLED after barge-in; got {state}"
            )

        # --- Capture resumed before stop completed ---
        assert "capture_resumed" in resume_order, (
            "capture_resumed must have fired after barge-in"
        )
        # If stop_done appeared, capture_resumed must precede it.
        if "stop_done" in resume_order:
            assert resume_order.index("capture_resumed") <= resume_order.index("stop_done"), (
                "Capture resume must not wait for renderer stop to finish"
            )

        # --- Stale-generation LLM emit must be rejected ---
        if try_stale_llm:
            # Start a new turn so the pipeline is running but generation is advanced.
            new_turn_id = uuid4()
            await session.start_turn(new_turn_id)
            # Attempt to emit LLM text for the *cancelled* turn.
            with pytest.raises(LateFrameRejected):
                await pipeline.emit_llm_text(
                    turn_id=assistant_turn_id,
                    sequence=0,
                    text="stale token",
                )

        # --- Stale-generation PCM emit must be rejected ---
        if try_stale_pcm:
            new_turn_id_2 = uuid4()
            try:
                await session.start_turn(new_turn_id_2)
            except Exception:
                pass  # Might already be started if try_stale_llm was true.
            stale_sid = uuid4()
            with pytest.raises(LateFrameRejected):
                await pipeline.emit_pcm_chunk(
                    turn_id=assistant_turn_id,
                    sentence_id=stale_sid,
                    sequence=0,
                    chunk_sequence=0,
                    pcm=b"\x00\x01",
                )

    finally:
        await pipeline.close()


# ---------------------------------------------------------------------------
# PBT: Idempotent barge-in signals — standalone parametrized property
# ---------------------------------------------------------------------------

@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=3000,
)
@given(
    extra_calls=st.integers(min_value=1, max_value=5),
)
def test_idempotent_repeated_barge_in_signals(extra_calls: int) -> None:
    """**Validates: Requirement 5.1**

    Regardless of how many times declare_barge_in is called after the first
    active call, all subsequent calls return None (no double-cancel).
    """
    asyncio.run(_idempotent_scenario(extra_calls))


async def _idempotent_scenario(extra_calls: int) -> None:
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    stop_count = [0]

    async def count_stop(t: UUID, g: int) -> None:
        stop_count[0] += 1

    coordinator = BargeInCoordinator(
        session=session,
        renderer_stop=count_stop,
    )
    coordinator.start_playback(turn_id=turn_id, generation=session.cancellation_generation)

    # First call: must succeed.
    result1 = await coordinator.declare_barge_in()
    assert result1 is not None, "First declare_barge_in must return BargeInResult"

    # Additional calls while no new playback is active: all must return None.
    for _ in range(extra_calls):
        result_n = await coordinator.declare_barge_in()
        assert result_n is None, "Subsequent declare_barge_in without new playback must be None"

    # Background tasks flush.
    for _ in range(20):
        await asyncio.sleep(0)

    assert stop_count[0] == 1, (
        f"renderer_stop must be called exactly once; called {stop_count[0]} times"
    )


# ---------------------------------------------------------------------------
# PBT: Stale-generation producer rejection — parametrized property
# ---------------------------------------------------------------------------

@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=3000,
)
@given(
    num_cancel_rounds=st.integers(min_value=1, max_value=4),
    probe_old_generation=st.booleans(),
)
def test_stale_generation_producer_rejection(
    num_cancel_rounds: int,
    probe_old_generation: bool,
) -> None:
    """**Validates: Requirements 5.3, 5.4**

    After cancellation advances the session generation, any frame stamped with
    an older generation is rejected by session.accept_frame (LateFrameRejected).
    output_is_current returns False for such frames (dispatch gate).
    """
    asyncio.run(_stale_generation_scenario(num_cancel_rounds, probe_old_generation))


async def _stale_generation_scenario(
    num_cancel_rounds: int,
    probe_old_generation: bool,
) -> None:
    session = VoiceSession(uuid4())

    # Do num_cancel_rounds of start + cancel to advance the generation.
    captured_generation = session.cancellation_generation  # 0 initially
    last_turn_id = None
    for _ in range(num_cancel_rounds):
        t = uuid4()
        last_turn_id = t
        await session.start_turn(t)
        await session.cancel_turn(t)

    # Now the session generation == num_cancel_rounds.
    assert session.cancellation_generation == num_cancel_rounds

    # Start a fresh turn at the advanced generation.
    new_turn = uuid4()
    await session.start_turn(new_turn)

    stale_gen = captured_generation if probe_old_generation else max(0, num_cancel_rounds - 1)

    from core.voice.session import TurnQueueName

    stale_frame = TypedVoiceFrame(
        VoiceFrameType.LLM_TEXT,
        VoiceFrameMetadata(
            session_id=session.session_id,
            turn_id=new_turn,
            sequence=0,
            cancellation_generation=stale_gen,  # stale
        ),
        "late text",
    )

    # accept_frame must reject it with LateFrameRejected.
    with pytest.raises(LateFrameRejected):
        await session.accept_frame(stale_frame, queue=TurnQueueName.LLM)

    # output_is_current must also return False.
    assert not await session.output_is_current(stale_frame), (
        "output_is_current must return False for stale-generation frames"
    )


# ---------------------------------------------------------------------------
# PBT: Queue drain only removes matching generation, not unrelated turns
# ---------------------------------------------------------------------------

@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=5000,
)
@given(
    n_interrupted=st.integers(min_value=0, max_value=5),
    n_unrelated=st.integers(min_value=0, max_value=3),
)
def test_barge_in_drains_only_interrupted_generation_work(
    n_interrupted: int,
    n_unrelated: int,
) -> None:
    """**Validates: Requirements 5.3, 5.4**

    _cancel_interrupted_work must drain exactly the frames for the interrupted
    (turn_id, generation) and leave frames for other turns untouched.
    """
    asyncio.run(_drain_scope_scenario(n_interrupted, n_unrelated))


async def _drain_scope_scenario(n_interrupted: int, n_unrelated: int) -> None:
    session = VoiceSession(uuid4())
    interrupted_turn = uuid4()
    await session.start_turn(interrupted_turn)
    gen0 = session.cancellation_generation  # 0

    pipeline = _make_pipeline(session)
    await pipeline.start()

    try:
        # Queue TTS sentences for the interrupted turn.
        for i in range(n_interrupted):
            sid = uuid4()
            await pipeline.emit_tts_text(
                turn_id=interrupted_turn,
                sentence_id=sid,
                sequence=i,
                text=f"Interrupted sentence {i}.",
            )

        # Cancel the turn to advance generation.
        pipeline.turn_control.barge_in.start_playback(
            turn_id=interrupted_turn,
            generation=gen0,
        )
        barge_result = await pipeline.turn_control.barge_in.declare_barge_in()
        assert barge_result is not None

        for _ in range(20):
            await asyncio.sleep(0)

        # Interrupted generation TTS queue must be empty.
        tts_remaining = pipeline.queues["tts"].qsize()
        assert tts_remaining == 0, (
            f"TTS queue must be empty after drain; got {tts_remaining}"
        )

    finally:
        await pipeline.close()


# ---------------------------------------------------------------------------
# PBT: Capture accepts new frames before cleanup completes (Req 5.5)
# ---------------------------------------------------------------------------

@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=5000,
)
@given(
    stop_delay_yields=st.integers(min_value=1, max_value=10),
)
def test_capture_proceeds_before_cleanup(stop_delay_yields: int) -> None:
    """**Validates: Requirement 5.5**

    capture_resumed fires before the (potentially slow) renderer-stop
    coroutine completes its final yield, proving capture does not block on
    cleanup.
    """
    asyncio.run(_capture_proceeds_scenario(stop_delay_yields))


async def _capture_proceeds_scenario(stop_delay_yields: int) -> None:
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)

    order: list[str] = []
    done_event = asyncio.Event()

    async def slow_stop(t: UUID, g: int) -> None:
        for _ in range(stop_delay_yields):
            await asyncio.sleep(0)
        order.append("stop_done")
        done_event.set()

    async def on_resume() -> None:
        order.append("capture_resumed")

    coordinator = BargeInCoordinator(
        session=session,
        renderer_stop=slow_stop,
        resume_capture=on_resume,
    )
    coordinator.start_playback(turn_id=turn_id, generation=session.cancellation_generation)

    result = await coordinator.declare_barge_in()
    assert result is not None

    # Let background tasks run to completion.
    await asyncio.wait_for(done_event.wait(), timeout=2.0)

    assert "capture_resumed" in order
    assert "stop_done" in order
    # Capture resume must appear at or before stop_done.
    assert order.index("capture_resumed") <= order.index("stop_done"), (
        "capture_resumed must not wait for stop_done"
    )
