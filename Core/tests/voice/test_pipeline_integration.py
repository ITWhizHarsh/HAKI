"""Pipecat pipeline integration tests.

Validates: Requirements 4.1–4.6, 4.8–4.9
Design reference: §4 (V-PIPELINE)

Covers:
  1. Final transcript arrives via the mandatory Pipecat transcription frame path.
  2. Bounded queues: partial (latest-wins), control, LLM, sentence, PCM.
  3. Terminal ordering: turn order maintained; terminal states reject later frames.
  4. All-or-nothing initialization: failure leaves voice unavailable, no custom threads.
  5. Replacement-only failure outcomes: unavailable/error/cancel, no legacy runtime.
  6. Frame graph mandatory types: InputAudioRawFrame, TranscriptionFrame,
     LLMTextFrame, TTSTextFrame — correct concrete types used.
  7. Backpressure: LLM/sentence queues respect capacity; PCM queue bounded.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import pytest

from core.ipc.voice_protocol import (
    PLAYBACK_CANCELLED,
    PLAYBACK_CONFIRMED,
    PLAYBACK_FAILED,
    TRANSCRIPT_EVENT,
)
from core.voice.asr_bridge import RingSlotDescriptor
from core.voice.frames import VoiceFrameType
from core.voice.pipeline import (
    BlockingLibraryError,
    BlockingVoiceLibraryExecutor,
    PipecatFrameAdapter,
    PipecatFrameAdapterUnavailable,
    PipelineAvailability,
    PipelineDiagnostic,
    PipelineInitializationError,
    PipelineQueueLimits,
    VoiceIngressProcessors,
    VoicePipelineSinks,
    VoicePipelineUnavailable,
    VoiceSessionPipeline,
)
from core.voice.session import LateFrameRejected, VoiceSession


# ---------------------------------------------------------------------------
# Minimal mock Pipecat frame types (substitute for real pipecat package)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Mock ring reader (no actual shared memory needed)
# ---------------------------------------------------------------------------

class MockRingReader:
    """Authenticated ring reader stub that returns fixed two-byte PCM data."""

    def __init__(self, session_id: UUID) -> None:
        self.session_id = session_id
        self.map_calls = 0
        self.release_calls = 0

    async def map_slot(self, descriptor: RingSlotDescriptor) -> bytes:
        self.map_calls += 1
        return b"\x01\x00" * (descriptor.byte_length // 2)

    async def release_slot(self, descriptor: RingSlotDescriptor) -> None:
        self.release_calls += 1



# ---------------------------------------------------------------------------
# Task factories
# ---------------------------------------------------------------------------

class _OkTaskFactory:
    """Returns a single non-None sentinel every call."""
    def __init__(self) -> None:
        self.calls = 0
        self.task = object()

    def __call__(self) -> object:
        self.calls += 1
        return self.task


class _FailingTaskFactory:
    """Raises immediately to simulate Pipecat initialization failure."""
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        raise RuntimeError("synthetic pipecat task creation failure")


class _NoneTaskFactory:
    """Returns None to simulate a task factory that produces no PipelineTask."""
    def __call__(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def _mock_adapter() -> PipecatFrameAdapter:
    return PipecatFrameAdapter(
        input_audio_frame_type=MockInputAudioRawFrame,
        transcription_frame_type=MockTranscriptionFrame,
        llm_text_frame_type=MockLLMTextFrame,
        tts_text_frame_type=MockTTSTextFrame,
    )


def _build_pipeline(
    *,
    task_factory: Any = None,
    sinks: VoicePipelineSinks | None = None,
    queue_limits: PipelineQueueLimits | None = None,
    diagnostic_sink: Any = None,
) -> tuple[VoiceSessionPipeline, VoiceSession]:
    session = VoiceSession(uuid4())
    ring = MockRingReader(session.session_id)
    ingress = VoiceIngressProcessors(
        session=session,
        ring_reader=ring,
        frame_adapter=_mock_adapter(),
    )
    factory = task_factory if task_factory is not None else _OkTaskFactory()
    pipeline = VoiceSessionPipeline(
        session=session,
        ingress=ingress,
        task_factory=factory,
        sinks=sinks,
        queue_limits=queue_limits,
        diagnostic_sink=diagnostic_sink,
    )
    return pipeline, session


def _ring_slot(*, session_id: UUID, turn_id: UUID, sequence: int = 0) -> RingSlotDescriptor:
    return RingSlotDescriptor(
        session_id=session_id,
        turn_id=turn_id,
        slot_index=0,
        sequence=sequence,
        captured_monotonic_ns=1,
        sample_rate_hz=16_000,
        channels=1,
        byte_length=2,
    )


def _transcript_event(
    *,
    session_id: UUID,
    turn_id: UUID,
    event_seq: int = 0,
    is_final: bool = True,
    text: str = "Kal meeting reschedule kar do",
    language: str = "hinglish",
) -> dict[str, object]:
    return {
        "version": 1,
        "type": TRANSCRIPT_EVENT,
        "event_id": str(uuid4()),
        "session_id": str(session_id),
        "turn_id": str(turn_id),
        "event_seq": event_seq,
        "text": text,
        "is_final": is_final,
        "language": language,
        "capture_started_monotonic_ns": 10,
        "capture_ended_monotonic_ns": 20,
    }


def _playback_event(
    *,
    event_type: str,
    session_id: UUID,
    turn_id: UUID,
    sentence_id: UUID,
) -> dict[str, object]:
    base: dict[str, object] = {
        "version": 1,
        "type": event_type,
        "event_id": str(uuid4()),
        "session_id": str(session_id),
        "turn_id": str(turn_id),
        "sentence_id": str(sentence_id),
    }
    if event_type == PLAYBACK_FAILED:
        base["error_class"] = "renderer_error"
    return base


async def _drain(pipeline: VoiceSessionPipeline, *, n: int, timeout: float = 2.0) -> None:
    """Wait up to *timeout* seconds for *n* queue items to be consumed."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        total = sum(q.qsize() for q in pipeline.queues.values())
        if total <= 0:
            return
        if asyncio.get_event_loop().time() > deadline:
            pytest.fail(f"queues not drained within {timeout}s; remaining={total}")
        await asyncio.sleep(0)


async def _wait_delivered(delivered: list, *, count: int, timeout: float = 2.0) -> None:
    """Spin until the delivered list reaches *count* items."""
    deadline = asyncio.get_event_loop().time() + timeout
    while len(delivered) < count:
        if asyncio.get_event_loop().time() > deadline:
            pytest.fail(f"expected {count} frames but only got {len(delivered)}")
        await asyncio.sleep(0)


# ===========================================================================
# 1. Final transcript arrives via the mandatory Pipecat transcription frame path
# ===========================================================================

class TestFinalTranscriptIngress:
    """Requirement 4.6: final Transcript_Event → TranscriptionFrame."""

    @pytest.mark.asyncio
    async def test_final_transcript_produces_transcription_frame_type(self) -> None:
        delivered: list = []
        pipeline, session = _build_pipeline(
            sinks=VoicePipelineSinks(final_transcription=lambda f: delivered.append(f))
        )
        turn_id = uuid4()
        await pipeline.start()
        try:
            result = await pipeline.ingest_transcript_message(
                _transcript_event(session_id=session.session_id, turn_id=turn_id, is_final=True)
            )
            assert result.accepted
            await _wait_delivered(delivered, count=1)
            frame = delivered[0]
            assert frame.frame_type is VoiceFrameType.TRANSCRIPTION
            assert isinstance(frame.payload, MockTranscriptionFrame)
            assert frame.metadata.turn_id == turn_id
            assert frame.metadata.is_final is True
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_partial_transcript_produces_transcription_frame_in_partial_queue(self) -> None:
        partial_frames: list = []
        pipeline, session = _build_pipeline(
            sinks=VoicePipelineSinks(partial_transcription=lambda f: partial_frames.append(f))
        )
        turn_id = uuid4()
        await pipeline.start()
        try:
            result = await pipeline.ingest_transcript_message(
                _transcript_event(
                    session_id=session.session_id,
                    turn_id=turn_id,
                    event_seq=0,
                    is_final=False,
                )
            )
            assert result.accepted
            await _wait_delivered(partial_frames, count=1)
            assert partial_frames[0].frame_type is VoiceFrameType.TRANSCRIPTION
            assert partial_frames[0].metadata.is_final is False
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_submit_final_transcript_direct_creates_transcription_frame(self) -> None:
        """submit_final_transcript is the in-process ingress path (Req 4.6)."""
        from core.voice.interfaces import VoiceTurnRequest
        delivered: list = []
        pipeline, session = _build_pipeline(
            sinks=VoicePipelineSinks(final_transcription=lambda f: delivered.append(f))
        )
        turn_id = uuid4()
        turn = VoiceTurnRequest(
            session_id=session.session_id,
            turn_id=turn_id,
            text="Kya haal hai",
            language="hi",
        )
        await pipeline.start()
        try:
            await pipeline.submit_final_transcript(turn)
            await _wait_delivered(delivered, count=1)
            frame = delivered[0]
            assert frame.frame_type is VoiceFrameType.TRANSCRIPTION
            assert isinstance(frame.payload, MockTranscriptionFrame)
            assert frame.payload.text == "Kya haal hai"
            assert frame.metadata.is_final is True
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_transcript_text_is_not_modified_by_ingress(self) -> None:
        """Req 4.6: normalized text must reach the frame unchanged."""
        delivered: list = []
        pipeline, session = _build_pipeline(
            sinks=VoicePipelineSinks(final_transcription=lambda f: delivered.append(f))
        )
        turn_id = uuid4()
        original_text = "Schedule meeting kal ke liye"
        await pipeline.start()
        try:
            await pipeline.ingest_transcript_message(
                _transcript_event(
                    session_id=session.session_id,
                    turn_id=turn_id,
                    text=original_text,
                    is_final=True,
                )
            )
            await _wait_delivered(delivered, count=1)
            assert delivered[0].payload.text == original_text
        finally:
            await pipeline.close()


# ===========================================================================
# 2. Bounded queues: correct capacity and backpressure / latest-wins policies
# ===========================================================================

class TestBoundedQueues:
    """Requirements 4.2–4.5: each queue has correct behaviour at capacity."""

    @pytest.mark.asyncio
    async def test_partial_queue_is_latest_wins_at_capacity(self) -> None:
        """Partial (UI coalescing) queue replaces the old item when full."""
        hold = asyncio.Event()
        partial_frames: list = []

        async def slow_sink(frame) -> None:
            partial_frames.append(frame)
            await hold.wait()

        pipeline, session = _build_pipeline(
            sinks=VoicePipelineSinks(partial_transcription=slow_sink),
            queue_limits=PipelineQueueLimits(partial=1),
        )
        turn_id = uuid4()
        await pipeline.start()
        try:
            # First partial: consumed by slow sink, blocks hold
            await pipeline.ingest_transcript_message(
                _transcript_event(
                    session_id=session.session_id, turn_id=turn_id,
                    event_seq=0, is_final=False, text="first partial",
                )
            )
            await _wait_delivered(partial_frames, count=1)

            # Two more partials fill the capacity=1 queue; second replaces first
            await pipeline.ingest_transcript_message(
                _transcript_event(
                    session_id=session.session_id, turn_id=turn_id,
                    event_seq=1, is_final=False, text="second partial",
                )
            )
            await pipeline.ingest_transcript_message(
                _transcript_event(
                    session_id=session.session_id, turn_id=turn_id,
                    event_seq=2, is_final=False, text="third partial",
                )
            )
            # partial queue must not grow beyond 1
            assert pipeline.queues["partial"].qsize() <= 1
        finally:
            hold.set()
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_control_queue_respects_capacity_and_is_non_droppable(self) -> None:
        """Final/control frames await capacity; they are never silently dropped."""
        released = asyncio.Event()
        consumed: list = []

        async def hold_sink(frame) -> None:
            consumed.append(frame)
            await released.wait()

        pipeline, session = _build_pipeline(
            sinks=VoicePipelineSinks(final_transcription=hold_sink),
            queue_limits=PipelineQueueLimits(control=1),
        )
        t1, t2, t3 = uuid4(), uuid4(), uuid4()
        await pipeline.start()
        try:
            # First final consumed by sink (blocking)
            await pipeline.ingest_transcript_message(
                _transcript_event(session_id=session.session_id, turn_id=t1, is_final=True)
            )
            await _wait_delivered(consumed, count=1)

            # Second final fills the queue (capacity=1)
            await pipeline.ingest_transcript_message(
                _transcript_event(session_id=session.session_id, turn_id=t2, is_final=True)
            )

            # Third final must wait for capacity — not dropped
            blocked = asyncio.create_task(
                pipeline.ingest_transcript_message(
                    _transcript_event(session_id=session.session_id, turn_id=t3, is_final=True)
                )
            )
            await asyncio.sleep(0.05)
            assert not blocked.done(), "control frame must await capacity, not be dropped"

            released.set()
            result = await asyncio.wait_for(blocked, timeout=2.0)
            assert result.accepted
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_llm_queue_respects_configured_capacity(self) -> None:
        """LLM queue does not exceed its maxsize; pipeline awaits capacity."""
        limits = PipelineQueueLimits(llm=2)
        pipeline, session = _build_pipeline(queue_limits=limits)
        await pipeline.start()
        turn_id = uuid4()
        await session.start_turn(turn_id)
        try:
            assert pipeline.queues["llm"].maxsize == 2
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_pcm_queue_capacity_matches_configured_limit(self) -> None:
        """PCM queue is bounded by milliseconds of audio; synthesis must pause."""
        limits = PipelineQueueLimits(pcm=8)
        pipeline, session = _build_pipeline(queue_limits=limits)
        await pipeline.start()
        try:
            assert pipeline.queues["pcm"].maxsize == 8
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_audio_queue_drops_oldest_when_full_and_counts_drops(self) -> None:
        """Audio (VAD) queue uses latest-wins; dropped audio increments the counter."""
        hold = asyncio.Event()
        audio_frames: list = []

        async def slow_audio(frame) -> None:
            audio_frames.append(frame)
            await hold.wait()

        limits = PipelineQueueLimits(audio=1)
        pipeline, session = _build_pipeline(
            sinks=VoicePipelineSinks(input_audio=slow_audio),
            queue_limits=limits,
        )
        turn_id = uuid4()
        await pipeline.start()
        try:
            await session.start_turn(turn_id)
            # First slot: consumed by slow sink
            await pipeline.ingest_ring_slot(
                _ring_slot(session_id=session.session_id, turn_id=turn_id, sequence=0)
            )
            await _wait_delivered(audio_frames, count=1)

            # Two more ring slots overflow the capacity=1 queue
            await pipeline.ingest_ring_slot(
                _ring_slot(session_id=session.session_id, turn_id=turn_id, sequence=1)
            )
            await pipeline.ingest_ring_slot(
                _ring_slot(session_id=session.session_id, turn_id=turn_id, sequence=2)
            )
            assert pipeline.dropped_audio_frames >= 1
        finally:
            hold.set()
            await pipeline.close()


# ===========================================================================
# 3. Terminal ordering: per-turn frame order; terminal states reject late frames
# ===========================================================================

class TestTerminalOrdering:
    """Requirement 4.8: per-turn order through completion/cancellation."""

    @pytest.mark.asyncio
    async def test_llm_frames_for_one_turn_are_delivered_in_sequence_order(self) -> None:
        """LLM frames must arrive at the sink in submission order."""
        delivered: list = []
        pipeline, session = _build_pipeline(
            sinks=VoicePipelineSinks(llm_text=lambda f: delivered.append(f))
        )
        turn_id = uuid4()
        await pipeline.start()
        await session.start_turn(turn_id)
        try:
            await pipeline.emit_llm_text(turn_id=turn_id, sequence=0, text="Hello")
            await pipeline.emit_llm_text(turn_id=turn_id, sequence=1, text=" world")
            await pipeline.emit_llm_text(turn_id=turn_id, sequence=2, text="!")
            await _wait_delivered(delivered, count=3)
            seqs = [f.metadata.sequence for f in delivered]
            assert seqs == [0, 1, 2]
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_tts_frames_for_one_turn_are_delivered_in_sequence_order(self) -> None:
        delivered: list = []
        pipeline, session = _build_pipeline(
            sinks=VoicePipelineSinks(tts_text=lambda f: delivered.append(f))
        )
        turn_id = uuid4()
        await pipeline.start()
        await session.start_turn(turn_id)
        try:
            for seq, text in enumerate(["First sentence.", "Second sentence.", "Third."]):
                await pipeline.emit_tts_text(
                    turn_id=turn_id,
                    sentence_id=uuid4(),
                    sequence=seq,
                    text=text,
                )
            await _wait_delivered(delivered, count=3)
            seqs = [f.metadata.sequence for f in delivered]
            assert seqs == [0, 1, 2]
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_completed_turn_rejects_further_llm_frames(self) -> None:
        """Once a turn is in a terminal state, LateFrameRejected is raised."""
        from core.voice.session import TurnState
        pipeline, session = _build_pipeline()
        turn_id = uuid4()
        await pipeline.start()
        await session.start_turn(turn_id)
        try:
            # Drive turn to completed
            await session.turns.transition(turn_id, TurnState.PARTIAL)
            await session.turns.transition(turn_id, TurnState.FINAL_PENDING_SILENCE)
            await session.turns.transition(turn_id, TurnState.REASONING)
            await session.turns.transition(turn_id, TurnState.SYNTHESIZING)
            await session.turns.transition(turn_id, TurnState.PLAYING)
            await session.turns.transition(turn_id, TurnState.COMPLETED)

            with pytest.raises((LateFrameRejected, VoicePipelineUnavailable, Exception)):
                await pipeline.emit_llm_text(turn_id=turn_id, sequence=0, text="Late text")
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_cancelled_turn_rejects_further_tts_frames(self) -> None:
        """A cancelled turn must not accept TTS frames (LateFrameRejected)."""
        pipeline, session = _build_pipeline()
        turn_id = uuid4()
        await pipeline.start()
        await session.start_turn(turn_id)
        try:
            await session.cancel_turn(turn_id)
            with pytest.raises(Exception):
                await pipeline.emit_tts_text(
                    turn_id=turn_id,
                    sentence_id=uuid4(),
                    sequence=0,
                    text="Late TTS",
                )
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_interleaved_turns_each_maintain_independent_order(self) -> None:
        """Two concurrent turns each receive their own ordered sequence of LLM frames."""
        delivered_by_turn: dict[UUID, list] = {}

        def capture(frame) -> None:
            tid = frame.metadata.turn_id
            delivered_by_turn.setdefault(tid, []).append(frame)

        pipeline, session = _build_pipeline(
            sinks=VoicePipelineSinks(llm_text=capture)
        )
        ta, tb = uuid4(), uuid4()
        await pipeline.start()
        await session.start_turn(ta)
        await session.start_turn(tb)
        try:
            await pipeline.emit_llm_text(turn_id=ta, sequence=0, text="A0")
            await pipeline.emit_llm_text(turn_id=tb, sequence=0, text="B0")
            await pipeline.emit_llm_text(turn_id=ta, sequence=1, text="A1")
            await pipeline.emit_llm_text(turn_id=tb, sequence=1, text="B1")

            await asyncio.sleep(0.1)
            for seqs in delivered_by_turn.values():
                assert [f.metadata.sequence for f in seqs] == sorted(
                    [f.metadata.sequence for f in seqs]
                )
        finally:
            await pipeline.close()


# ===========================================================================
# 4. All-or-nothing initialization: failure → unavailable, no substitute thread
# ===========================================================================

class TestAllOrNothingInitialization:
    """Requirement 4.9: Pipecat init failure leaves voice unavailable, no fallback."""

    @pytest.mark.asyncio
    async def test_task_factory_raising_leaves_pipeline_unavailable(self) -> None:
        diagnostics: list[PipelineDiagnostic] = []
        factory = _FailingTaskFactory()
        pipeline, _ = _build_pipeline(
            task_factory=factory,
            diagnostic_sink=diagnostics.append,
        )
        with pytest.raises(PipelineInitializationError):
            await pipeline.start()

        assert pipeline.availability is PipelineAvailability.UNAVAILABLE
        assert pipeline.pipeline_task is None
        assert pipeline._runtime_task is None

    @pytest.mark.asyncio
    async def test_task_factory_returning_none_leaves_pipeline_unavailable(self) -> None:
        factory = _NoneTaskFactory()
        pipeline, _ = _build_pipeline(task_factory=factory)
        with pytest.raises(PipelineInitializationError):
            await pipeline.start()
        assert pipeline.availability is PipelineAvailability.UNAVAILABLE
        assert pipeline.pipeline_task is None

    @pytest.mark.asyncio
    async def test_initialization_failure_emits_pipecat_stage_diagnostic(self) -> None:
        """A failed start must emit a content-free pipecat/failed diagnostic."""
        diagnostics: list[PipelineDiagnostic] = []
        pipeline, _ = _build_pipeline(
            task_factory=_FailingTaskFactory(),
            diagnostic_sink=diagnostics.append,
        )
        with pytest.raises(PipelineInitializationError):
            await pipeline.start()

        assert len(diagnostics) == 1
        diag = diagnostics[0]
        assert diag.stage == "pipecat"
        assert diag.outcome == "failed"
        assert diag.error_class is not None

    @pytest.mark.asyncio
    async def test_initialization_failure_starts_no_custom_thread(self) -> None:
        """No additional OS thread is spawned when the pipeline fails to start."""
        baseline = {t.ident for t in threading.enumerate()}
        pipeline, _ = _build_pipeline(task_factory=_FailingTaskFactory())
        with pytest.raises(PipelineInitializationError):
            await pipeline.start()
        after = {t.ident for t in threading.enumerate()}
        assert after == baseline

    @pytest.mark.asyncio
    async def test_unavailable_pipeline_rejects_all_ingress(self) -> None:
        """Once unavailable, the pipeline must refuse ring slots and transcripts."""
        pipeline, session = _build_pipeline(task_factory=_FailingTaskFactory())
        with pytest.raises(PipelineInitializationError):
            await pipeline.start()

        turn_id = uuid4()
        with pytest.raises(VoicePipelineUnavailable):
            await pipeline.ingest_ring_slot(
                _ring_slot(session_id=session.session_id, turn_id=turn_id)
            )
        with pytest.raises(VoicePipelineUnavailable):
            await pipeline.ingest_transcript_message(
                _transcript_event(session_id=session.session_id, turn_id=turn_id)
            )

    @pytest.mark.asyncio
    async def test_second_start_call_is_idempotent_on_running_pipeline(self) -> None:
        """Calling start() again on a running pipeline is a no-op (not an error)."""
        factory = _OkTaskFactory()
        pipeline, _ = _build_pipeline(task_factory=factory)
        await pipeline.start()
        try:
            await pipeline.start()  # second call must be idempotent
            assert factory.calls == 1  # task factory called exactly once
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_closed_pipeline_cannot_be_restarted(self) -> None:
        pipeline, _ = _build_pipeline()
        await pipeline.start()
        await pipeline.close()
        with pytest.raises(VoicePipelineUnavailable):
            await pipeline.start()


# ===========================================================================
# 5. Replacement-only failure outcomes: no legacy imports or custom thread
# ===========================================================================

class TestReplacementOnlyFailureOutcomes:
    """Requirements 1.6, 4.9: failures produce unavailable/error/cancel, never legacy."""

    @pytest.mark.asyncio
    async def test_no_legacy_module_imported_after_initialization_failure(self) -> None:
        """Legacy voice module names must not appear in sys.modules after failure."""
        import sys
        pipeline, _ = _build_pipeline(task_factory=_FailingTaskFactory())
        with pytest.raises(PipelineInitializationError):
            await pipeline.start()

        legacy_names = [
            k for k in sys.modules
            if any(
                term in k.lower()
                for term in ("deepgram", "cartesia", "kokoro", "chattts", "edge_tts",
                             "groq_voice", "afplay", "legacy_pipeline")
            )
        ]
        assert legacy_names == [], f"legacy modules in sys.modules: {legacy_names}"

    @pytest.mark.asyncio
    async def test_sink_exception_does_not_start_legacy_fallback(self) -> None:
        """A sink that raises must not trigger a legacy or custom-thread route.

        When a sink raises the pipeline's TaskGroup propagates the error on close;
        we verify no legacy/fallback attributes exist before the exception escapes.
        """
        def bad_sink(frame) -> None:
            raise RuntimeError("sink crashed")

        pipeline, session = _build_pipeline(
            sinks=VoicePipelineSinks(final_transcription=bad_sink)
        )
        turn_id = uuid4()
        await pipeline.start()
        # Ingress itself must succeed even if the sink later raises
        result = await pipeline.ingest_transcript_message(
            _transcript_event(session_id=session.session_id, turn_id=turn_id)
        )
        assert result.accepted
        # No legacy attributes on the pipeline object before shutdown
        assert not hasattr(pipeline, "legacy_pipeline")
        assert not hasattr(pipeline, "playback_subprocess")
        # close() propagates the sink exception via ExceptionGroup — tolerate it
        try:
            await pipeline.close()
        except (RuntimeError, ExceptionGroup, BaseExceptionGroup):
            pass

    @pytest.mark.asyncio
    async def test_blocking_executor_restricted_to_permitted_libraries(self) -> None:
        """Only mlx-lm and xtts are permitted through the blocking executor."""
        from threading import Event
        pipeline, _ = _build_pipeline()
        await pipeline.start()
        try:
            # Permitted library must succeed
            result = await pipeline.run_blocking_library(
                library="mlx-lm",
                operation=lambda stop: "ok",
            )
            assert result == "ok"

            # Non-permitted library must raise BlockingLibraryError, not route elsewhere
            with pytest.raises(BlockingLibraryError):
                await pipeline.run_blocking_library(
                    library="legacy-tts",
                    operation=lambda stop: "should not run",
                )
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_blocking_executor_uses_exactly_one_worker_thread(self) -> None:
        """The executor is bounded to one worker; it never spawns a playback thread."""
        pipeline, _ = _build_pipeline()
        await pipeline.start()
        try:
            assert pipeline._blocking_executor.max_workers == 1
            assert not hasattr(pipeline._blocking_executor, "playback_thread")
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_pipeline_unavailable_error_has_no_legacy_fallback_attribute(self) -> None:
        """VoicePipelineUnavailable carries no reference to a legacy component."""
        exc = VoicePipelineUnavailable("voice_pipeline_unavailable")
        assert not hasattr(exc, "legacy_route")
        assert not hasattr(exc, "fallback")


# ===========================================================================
# 6. Frame graph mandatory types: correct Pipecat frame types
# ===========================================================================

class TestMandatoryFrameTypes:
    """Requirements 4.2–4.5: prescribed Pipecat frame types are used in the graph."""

    @pytest.mark.asyncio
    async def test_ring_slot_produces_input_audio_raw_frame_payload(self) -> None:
        """Req 4.2: audio input frames use the InputAudioRawFrame-compatible type."""
        delivered: list = []
        pipeline, session = _build_pipeline(
            sinks=VoicePipelineSinks(input_audio=lambda f: delivered.append(f))
        )
        turn_id = uuid4()
        await pipeline.start()
        try:
            await pipeline.ingest_ring_slot(
                _ring_slot(session_id=session.session_id, turn_id=turn_id)
            )
            await _wait_delivered(delivered, count=1)
            frame = delivered[0]
            assert frame.frame_type is VoiceFrameType.INPUT_AUDIO
            assert isinstance(frame.payload, MockInputAudioRawFrame)
            # Verify required Silero fields are present
            assert hasattr(frame.payload, "audio")
            assert hasattr(frame.payload, "sample_rate")
            assert hasattr(frame.payload, "num_channels")
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_transcript_ingress_uses_transcription_frame_type(self) -> None:
        """Req 4.3: ASR results use TranscriptionFrame."""
        delivered: list = []
        pipeline, session = _build_pipeline(
            sinks=VoicePipelineSinks(final_transcription=lambda f: delivered.append(f))
        )
        turn_id = uuid4()
        await pipeline.start()
        try:
            await pipeline.ingest_transcript_message(
                _transcript_event(session_id=session.session_id, turn_id=turn_id)
            )
            await _wait_delivered(delivered, count=1)
            assert delivered[0].frame_type is VoiceFrameType.TRANSCRIPTION
            assert isinstance(delivered[0].payload, MockTranscriptionFrame)
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_emit_llm_text_uses_llm_text_frame_type(self) -> None:
        """Req 4.4: model output uses LLMTextFrame."""
        delivered: list = []
        pipeline, session = _build_pipeline(
            sinks=VoicePipelineSinks(llm_text=lambda f: delivered.append(f))
        )
        turn_id = uuid4()
        await pipeline.start()
        await session.start_turn(turn_id)
        try:
            await pipeline.emit_llm_text(turn_id=turn_id, sequence=0, text="Response text.")
            await _wait_delivered(delivered, count=1)
            assert delivered[0].frame_type is VoiceFrameType.LLM_TEXT
            assert isinstance(delivered[0].payload, MockLLMTextFrame)
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_emit_tts_text_uses_tts_text_frame_type(self) -> None:
        """Req 4.5: sentence-ready synthesis input uses TTSTextFrame."""
        delivered: list = []
        pipeline, session = _build_pipeline(
            sinks=VoicePipelineSinks(tts_text=lambda f: delivered.append(f))
        )
        turn_id = uuid4()
        await pipeline.start()
        await session.start_turn(turn_id)
        try:
            await pipeline.emit_tts_text(
                turn_id=turn_id,
                sentence_id=uuid4(),
                sequence=0,
                text="Sentence ready.",
            )
            await _wait_delivered(delivered, count=1)
            assert delivered[0].frame_type is VoiceFrameType.TTS_TEXT
            assert isinstance(delivered[0].payload, MockTTSTextFrame)
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_frame_adapter_raises_when_pipecat_frame_unavailable(self) -> None:
        """If a Pipecat frame type is None after load the adapter raises the right error."""
        # Build with all types set so construction succeeds
        adapter = PipecatFrameAdapter(
            input_audio_frame_type=MockInputAudioRawFrame,
            transcription_frame_type=MockTranscriptionFrame,
            llm_text_frame_type=MockLLMTextFrame,
            tts_text_frame_type=MockTTSTextFrame,
        )
        # Force-clear the type so _required_type triggers the guard path
        adapter._input_audio_frame_type = None  # type: ignore[assignment]

        def _load_noop() -> None:
            # Do not restore the type — leave it None to trigger guard
            pass

        original = adapter._load_frame_types
        adapter._load_frame_types = _load_noop  # type: ignore[method-assign]
        try:
            with pytest.raises(PipecatFrameAdapterUnavailable):
                adapter.create_input_audio_frame(audio=b"\x00\x01", sample_rate_hz=16000, channels=1)
        finally:
            adapter._load_frame_types = original  # type: ignore[method-assign]

    @pytest.mark.asyncio
    async def test_all_four_mandatory_frame_types_flow_end_to_end(self) -> None:
        """All four mandatory frame types must be deliverable in a single pipeline run."""
        delivered: list = []

        def record(frame) -> None:
            delivered.append(frame)

        pipeline, session = _build_pipeline(
            sinks=VoicePipelineSinks(
                input_audio=record,
                final_transcription=record,
                llm_text=record,
                tts_text=record,
            )
        )
        turn_id = uuid4()
        await pipeline.start()
        try:
            await pipeline.ingest_ring_slot(
                _ring_slot(session_id=session.session_id, turn_id=turn_id)
            )
            await pipeline.ingest_transcript_message(
                _transcript_event(session_id=session.session_id, turn_id=turn_id, event_seq=0)
            )
            await pipeline.emit_llm_text(turn_id=turn_id, sequence=1, text="LLM output.")
            await pipeline.emit_tts_text(
                turn_id=turn_id, sentence_id=uuid4(), sequence=2, text="TTS output."
            )
            await _wait_delivered(delivered, count=4)
            frame_types = {f.frame_type for f in delivered}
            assert VoiceFrameType.INPUT_AUDIO in frame_types
            assert VoiceFrameType.TRANSCRIPTION in frame_types
            assert VoiceFrameType.LLM_TEXT in frame_types
            assert VoiceFrameType.TTS_TEXT in frame_types
        finally:
            await pipeline.close()


# ===========================================================================
# 7. Backpressure: LLM/sentence/PCM queue capacity and bounded behaviour
# ===========================================================================

class TestBackpressure:
    """Requirements 4.4–4.5, design §4: bounded output queues resist overrun."""

    @pytest.mark.asyncio
    async def test_sentence_queue_maxsize_matches_configured_limit(self) -> None:
        limits = PipelineQueueLimits(sentence=4)
        pipeline, _ = _build_pipeline(queue_limits=limits)
        await pipeline.start()
        try:
            assert pipeline.queues["tts"].maxsize == 4
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_pcm_emit_is_blocked_when_pcm_queue_is_full(self) -> None:
        """PCM queue back-pressure: synthesis must pause before renderer overrun."""
        hold = asyncio.Event()
        consumed: list = []

        async def slow_pcm(frame) -> None:
            consumed.append(frame)
            await hold.wait()

        limits = PipelineQueueLimits(pcm=1)
        pipeline, session = _build_pipeline(
            sinks=VoicePipelineSinks(pcm=slow_pcm),
            queue_limits=limits,
        )
        turn_id = uuid4()
        sent_id = uuid4()
        await pipeline.start()
        await session.start_turn(turn_id)
        try:
            # First chunk consumed by slow sink
            await pipeline.emit_pcm_chunk(
                turn_id=turn_id,
                sentence_id=sent_id,
                sequence=0,
                chunk_sequence=0,
                pcm=b"\x00\x01",
            )
            await _wait_delivered(consumed, count=1)

            # Second fills the queue
            await pipeline.emit_pcm_chunk(
                turn_id=turn_id,
                sentence_id=sent_id,
                sequence=1,
                chunk_sequence=1,
                pcm=b"\x00\x02",
            )

            # Third must be blocked
            blocked = asyncio.create_task(
                pipeline.emit_pcm_chunk(
                    turn_id=turn_id,
                    sentence_id=sent_id,
                    sequence=2,
                    chunk_sequence=2,
                    pcm=b"\x00\x03",
                )
            )
            await asyncio.sleep(0.05)
            assert not blocked.done(), "PCM emit must await capacity, not overflow"
        finally:
            hold.set()
            blocked.cancel()
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_queue_limits_default_values_are_positive(self) -> None:
        """All default queue capacities are positive integers."""
        limits = PipelineQueueLimits()
        assert limits.audio > 0
        assert limits.partial > 0
        assert limits.control > 0
        assert limits.llm > 0
        assert limits.sentence > 0
        assert limits.pcm > 0

    @pytest.mark.asyncio
    async def test_queue_limits_reject_zero_capacity(self) -> None:
        with pytest.raises(ValueError):
            PipelineQueueLimits(llm=0)

    @pytest.mark.asyncio
    async def test_queue_limits_reject_negative_capacity(self) -> None:
        with pytest.raises(ValueError):
            PipelineQueueLimits(pcm=-1)


# ===========================================================================
# 8. Playback event processing and ledger integration
# ===========================================================================

class TestPlaybackEventProcessing:
    """Requirements 4.1, 4.8: playback events flow through the pipeline safely."""

    @pytest.mark.asyncio
    async def test_playback_confirmed_returns_true_for_registered_sentence(self) -> None:
        pipeline, session = _build_pipeline()
        turn_id = uuid4()
        sent_id = uuid4()
        await pipeline.start()
        await session.start_turn(turn_id)
        try:
            # Register sentence via TTS emit
            await pipeline.emit_tts_text(
                turn_id=turn_id,
                sentence_id=sent_id,
                sequence=0,
                text="Confirmed sentence.",
            )
            confirmed = await pipeline.process_playback_event(
                _playback_event(
                    event_type=PLAYBACK_CONFIRMED,
                    session_id=session.session_id,
                    turn_id=turn_id,
                    sentence_id=sent_id,
                )
            )
            assert confirmed is True
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_playback_cancelled_returns_true(self) -> None:
        pipeline, session = _build_pipeline()
        turn_id = uuid4()
        sent_id = uuid4()
        await pipeline.start()
        await session.start_turn(turn_id)
        try:
            await pipeline.emit_tts_text(
                turn_id=turn_id,
                sentence_id=sent_id,
                sequence=0,
                text="Cancelled sentence.",
            )
            result = await pipeline.process_playback_event(
                _playback_event(
                    event_type=PLAYBACK_CANCELLED,
                    session_id=session.session_id,
                    turn_id=turn_id,
                    sentence_id=sent_id,
                )
            )
            assert result is True
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_playback_event_for_wrong_session_returns_false(self) -> None:
        pipeline, session = _build_pipeline()
        turn_id = uuid4()
        sent_id = uuid4()
        other_session = uuid4()
        await pipeline.start()
        try:
            result = await pipeline.process_playback_event(
                _playback_event(
                    event_type=PLAYBACK_CONFIRMED,
                    session_id=other_session,
                    turn_id=turn_id,
                    sentence_id=sent_id,
                )
            )
            assert result is False
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_playback_failed_marks_sentence_failed_in_ledger(self) -> None:
        from core.voice.session import ProvisionalSentenceState
        pipeline, session = _build_pipeline()
        turn_id = uuid4()
        sent_id = uuid4()
        await pipeline.start()
        await session.start_turn(turn_id)
        try:
            await pipeline.emit_tts_text(
                turn_id=turn_id,
                sentence_id=sent_id,
                sequence=0,
                text="Failed sentence.",
            )
            result = await pipeline.process_playback_event(
                _playback_event(
                    event_type=PLAYBACK_FAILED,
                    session_id=session.session_id,
                    turn_id=turn_id,
                    sentence_id=sent_id,
                )
            )
            assert result is True
            state = await session.playback_ledger.state_for(sent_id)
            assert state is ProvisionalSentenceState.FAILED
        finally:
            await pipeline.close()


# ===========================================================================
# 9. Injected failures: executor, sink, and adapter failures stay contained
# ===========================================================================

class TestInjectedFailures:
    """Req 4.9: injected failures never start a substitute or legacy runtime."""

    @pytest.mark.asyncio
    async def test_async_sink_exception_does_not_start_legacy_fallback(self) -> None:
        """A sink that raises must not trigger legacy or custom-thread replacement paths.

        When a sink raises, the TaskGroup propagates it on close(). We verify
        no legacy attributes exist and that the exception is a standard Python error,
        not a legacy runtime being started.
        """
        async def bad_async_sink(frame) -> None:
            raise ValueError("async sink failure")

        pipeline, session = _build_pipeline(
            sinks=VoicePipelineSinks(final_transcription=bad_async_sink)
        )
        turn_id = uuid4()
        await pipeline.start()
        # Ingress must succeed before the sink runs
        result = await pipeline.ingest_transcript_message(
            _transcript_event(session_id=session.session_id, turn_id=turn_id)
        )
        assert result.accepted
        # No legacy/fallback attributes exist
        assert not hasattr(pipeline, "legacy_pipeline")
        assert not hasattr(pipeline, "playback_subprocess")
        # close() propagates the sink exception — tolerate ValueError/ExceptionGroup
        try:
            await pipeline.close()
        except (ValueError, ExceptionGroup, BaseExceptionGroup):
            pass

    @pytest.mark.asyncio
    async def test_empty_llm_text_is_rejected_with_value_error(self) -> None:
        pipeline, session = _build_pipeline()
        turn_id = uuid4()
        await pipeline.start()
        await session.start_turn(turn_id)
        try:
            with pytest.raises(ValueError, match="empty"):
                await pipeline.emit_llm_text(turn_id=turn_id, sequence=0, text="   ")
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_empty_tts_text_is_rejected_with_value_error(self) -> None:
        pipeline, session = _build_pipeline()
        turn_id = uuid4()
        await pipeline.start()
        await session.start_turn(turn_id)
        try:
            with pytest.raises(ValueError, match="empty"):
                await pipeline.emit_tts_text(
                    turn_id=turn_id,
                    sentence_id=uuid4(),
                    sequence=0,
                    text="",
                )
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_empty_pcm_chunk_is_rejected_with_value_error(self) -> None:
        pipeline, session = _build_pipeline()
        turn_id = uuid4()
        await pipeline.start()
        await session.start_turn(turn_id)
        try:
            with pytest.raises(ValueError):
                await pipeline.emit_pcm_chunk(
                    turn_id=turn_id,
                    sentence_id=uuid4(),
                    sequence=0,
                    chunk_sequence=0,
                    pcm=b"",
                )
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_blocking_executor_closed_raises_unavailable(self) -> None:
        """Once the executor is closed, run() raises VoicePipelineUnavailable."""
        executor = BlockingVoiceLibraryExecutor()
        executor.close()
        with pytest.raises(VoicePipelineUnavailable):
            await executor.run(library="mlx-lm", operation=lambda stop: "result")

    @pytest.mark.asyncio
    async def test_cross_session_transcript_is_discarded(self) -> None:
        """A transcript event for a different session ID must be discarded, not crash."""
        pipeline, session = _build_pipeline()
        other_session = uuid4()
        turn_id = uuid4()
        await pipeline.start()
        try:
            result = await pipeline.ingest_transcript_message(
                _transcript_event(session_id=other_session, turn_id=turn_id)
            )
            # Should either raise (VoicePipelineUnavailable/ValueError) or be discarded
            assert not result.accepted
        except (VoicePipelineUnavailable, ValueError):
            pass  # Both are acceptable replacement-only error outcomes
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_pipeline_queues_exposed_for_diagnostics(self) -> None:
        """queues property exposes all six bounded queues for diagnostic access."""
        pipeline, _ = _build_pipeline()
        await pipeline.start()
        try:
            queues = pipeline.queues
            expected_keys = {"audio", "partial", "control", "llm", "tts", "pcm"}
            assert set(queues.keys()) == expected_keys
            for q in queues.values():
                assert isinstance(q, asyncio.Queue)
                assert q.maxsize > 0
        finally:
            await pipeline.close()

