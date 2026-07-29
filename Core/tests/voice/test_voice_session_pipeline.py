"""Focused async coverage for the owned Pipecat voice-session graph.

Validates: Requirements 4.1–4.6, 4.8–4.9
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from core.ipc.voice_protocol import TRANSCRIPT_EVENT
from core.voice.asr_bridge import RingSlotDescriptor
from core.voice.frames import VoiceFrameType
from core.voice.pipeline import (
    PipecatFrameAdapter,
    PipelineAvailability,
    PipelineDiagnostic,
    PipelineInitializationError,
    PipelineQueueLimits,
    VoiceIngressProcessors,
    VoicePipelineSinks,
    VoiceSessionPipeline,
)
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


@dataclass(frozen=True)
class MockLLMTextFrame:
    text: str


@dataclass(frozen=True)
class MockTTSTextFrame:
    text: str


class MockAuthenticatedRing:
    def __init__(self, session_id: UUID) -> None:
        self.session_id = session_id

    async def map_slot(self, descriptor: RingSlotDescriptor) -> bytes:
        return b"\x01\x00"

    async def release_slot(self, descriptor: RingSlotDescriptor) -> None:
        return None


class CountingTaskFactory:
    def __init__(self) -> None:
        self.calls = 0
        self.task = object()

    def __call__(self) -> object:
        self.calls += 1
        return self.task


class FailingTaskFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        raise RuntimeError("synthetic pipecat initialization failure")


def _adapter() -> PipecatFrameAdapter:
    return PipecatFrameAdapter(
        input_audio_frame_type=MockInputAudioRawFrame,
        transcription_frame_type=MockTranscriptionFrame,
        llm_text_frame_type=MockLLMTextFrame,
        tts_text_frame_type=MockTTSTextFrame,
    )


def _pipeline(
    *,
    task_factory=CountingTaskFactory(),
    sinks: VoicePipelineSinks | None = None,
    queue_limits: PipelineQueueLimits | None = None,
    diagnostic_sink=None,
) -> tuple[VoiceSessionPipeline, VoiceSession, object]:
    session = VoiceSession(uuid4())
    ingress = VoiceIngressProcessors(
        session=session,
        ring_reader=MockAuthenticatedRing(session.session_id),
        frame_adapter=_adapter(),
    )
    pipeline = VoiceSessionPipeline(
        session=session,
        ingress=ingress,
        task_factory=task_factory,
        sinks=sinks,
        queue_limits=queue_limits,
        diagnostic_sink=diagnostic_sink,
    )
    return pipeline, session, task_factory


def _slot(*, session_id: UUID, turn_id: UUID) -> RingSlotDescriptor:
    return RingSlotDescriptor(
        session_id=session_id,
        turn_id=turn_id,
        slot_index=0,
        sequence=0,
        captured_monotonic_ns=1,
        sample_rate_hz=16_000,
        channels=1,
        byte_length=2,
    )


def _final_transcript(*, session_id: UUID, turn_id: UUID, sequence: int = 0) -> dict[str, object]:
    return {
        "version": 1,
        "type": TRANSCRIPT_EVENT,
        "event_id": str(uuid4()),
        "session_id": str(session_id),
        "turn_id": str(turn_id),
        "event_seq": sequence,
        "text": "Kal meeting reschedule kar do",
        "is_final": True,
        "language": "hinglish",
        "capture_started_monotonic_ns": 10,
        "capture_ended_monotonic_ns": 20,
    }


async def _wait_for(predicate) -> None:
    for _ in range(50):
        if predicate():
            return
        await asyncio.sleep(0)
    pytest.fail("expected pipeline graph work was not dispatched")


@pytest.mark.asyncio
async def test_initialization_failure_leaves_voice_unavailable_without_substitute_runtime() -> None:
    """A Pipecat failure reports pipecat and allocates no task or fallback graph."""
    diagnostics: list[PipelineDiagnostic] = []
    factory = FailingTaskFactory()
    pipeline, _, _ = _pipeline(task_factory=factory, diagnostic_sink=diagnostics.append)

    with pytest.raises(PipelineInitializationError, match="pipecat_initialization_failed"):
        await pipeline.start()

    assert factory.calls == 1
    assert pipeline.availability is PipelineAvailability.UNAVAILABLE
    assert pipeline.pipeline_task is None
    assert diagnostics == [
        PipelineDiagnostic(stage="pipecat", outcome="failed", error_class="RuntimeError")
    ]
    assert pipeline._runtime_task is None


@pytest.mark.asyncio
async def test_single_pipeline_task_routes_mandatory_frames_in_graph_order() -> None:
    """Audio, final transcript, LLM text, and TTS text use their mandatory frame path."""
    delivered: list[object] = []

    async def record(frame) -> None:
        delivered.append(frame)

    factory = CountingTaskFactory()
    sinks = VoicePipelineSinks(
        input_audio=record,
        final_transcription=record,
        llm_text=record,
        tts_text=record,
    )
    pipeline, session, _ = _pipeline(task_factory=factory, sinks=sinks)
    turn_id, sentence_id = uuid4(), uuid4()

    await pipeline.start()
    try:
        await pipeline.ingest_ring_slot(_slot(session_id=session.session_id, turn_id=turn_id))
        accepted = await pipeline.ingest_transcript_message(
            _final_transcript(session_id=session.session_id, turn_id=turn_id)
        )
        assert accepted.accepted
        await pipeline.emit_llm_text(turn_id=turn_id, sequence=1, text="I can do that.")
        await pipeline.emit_tts_text(
            turn_id=turn_id,
            sentence_id=sentence_id,
            sequence=2,
            text="I can do that.",
        )
        await _wait_for(lambda: len(delivered) == 4)

        assert factory.calls == 1
        assert pipeline.pipeline_task is factory.task
        assert [frame.frame_type for frame in delivered] == [
            VoiceFrameType.INPUT_AUDIO,
            VoiceFrameType.TRANSCRIPTION,
            VoiceFrameType.LLM_TEXT,
            VoiceFrameType.TTS_TEXT,
        ]
        assert isinstance(delivered[0].payload, MockInputAudioRawFrame)
        assert isinstance(delivered[1].payload, MockTranscriptionFrame)
        assert isinstance(delivered[2].payload, MockLLMTextFrame)
        assert isinstance(delivered[3].payload, MockTTSTextFrame)
        assert [frame.metadata.sequence for frame in delivered[1:]] == [0, 1, 2]
    finally:
        await pipeline.close()


@pytest.mark.asyncio
async def test_final_control_frames_wait_for_capacity_and_preserve_fifo_order() -> None:
    """Final transcripts are non-droppable even when the bounded control queue fills."""
    delivered_turns: list[UUID] = []
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def hold_first(frame) -> None:
        delivered_turns.append(frame.metadata.turn_id)
        if len(delivered_turns) == 1:
            first_started.set()
            await release_first.wait()

    pipeline, session, _ = _pipeline(
        sinks=VoicePipelineSinks(final_transcription=hold_first),
        queue_limits=PipelineQueueLimits(control=1),
    )
    first, second, third = uuid4(), uuid4(), uuid4()

    await pipeline.start()
    try:
        assert (await pipeline.ingest_transcript_message(
            _final_transcript(session_id=session.session_id, turn_id=first)
        )).accepted
        await first_started.wait()
        assert (await pipeline.ingest_transcript_message(
            _final_transcript(session_id=session.session_id, turn_id=second)
        )).accepted

        blocked = asyncio.create_task(
            pipeline.ingest_transcript_message(
                _final_transcript(session_id=session.session_id, turn_id=third)
            )
        )
        await asyncio.sleep(0)
        assert not blocked.done(), "final control work must await capacity rather than be dropped"

        release_first.set()
        assert (await blocked).accepted
        await _wait_for(lambda: delivered_turns == [first, second, third])
        assert pipeline.queues["control"].maxsize == 1
    finally:
        await pipeline.close()


@pytest.mark.asyncio
async def test_starting_graph_instantiates_no_custom_playback_thread() -> None:
    """The graph remains asyncio-owned until an allowed blocking model call is requested."""
    baseline = {thread.ident for thread in threading.enumerate()}
    pipeline, _, _ = _pipeline()

    await pipeline.start()
    try:
        assert pipeline._blocking_executor._executor is None
        assert {thread.ident for thread in threading.enumerate()} == baseline
        assert not hasattr(pipeline, "playback_subprocess")
        assert not hasattr(pipeline, "legacy_pipeline")
    finally:
        await pipeline.close()
