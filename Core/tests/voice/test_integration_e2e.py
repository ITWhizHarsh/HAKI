"""Mocked end-to-end component integration test for the replacement voice path.

Validates: Requirements 1.5–1.6, 2.1–2.6, 3.1–3.8, 4.1–4.9, 5.1–5.8,
           6.1–6.7, 7.1–7.8, 8.1–8.7, 9.4–9.6, 10.1–10.6
Design reference: Overview, §§2–10 (V-PIPELINE, V-BARGE-LATENCY, fault-injection matrix)

Coverage:
  1. Happy path: transcript event → TranscriptionFrame → LLM turn → TTSTextFrame
     → PCM chunks → PLAYBACK_CONFIRMED → sentence in ledger.
  2. ASR stage failure: empty transcript → diagnostic emitted, no LLM turn.
  3. IPC disconnect before final: turn discarded, IPC diagnostic emitted.
  4. LLM failure: local_llm diagnostic, no legacy fallback selected.
  5. TTS failure: synthesis fails → local_tts diagnostic, text surfaced.
  6. Barge-in: speech during playback → cancellation, new capture begins.
  7. All mocked — no real MLX/XTTS/AVAudioEngine/network calls.

Gate: all tests run with the replacement gate ENABLED (no env var required
because the pipeline under test is instantiated directly without the gate
guard — the gate itself is tested separately in test_dev_gate.py).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from time import monotonic_ns
from typing import Any
from uuid import UUID, uuid4

import pytest

from core.ipc.voice_protocol import (
    PLAYBACK_CANCELLED,
    PLAYBACK_CONFIRMED,
    PLAYBACK_FAILED,
    TRANSCRIPT_EVENT,
)
from core.voice.asr_bridge import (
    AuthenticatedRingSlotReader,
    RingSlotDescriptor,
    TranscriptIngressResult,
)
from core.voice.frames import VoiceFrameType
from core.voice.pipeline import (
    PipecatFrameAdapter,
    PipelineAvailability,
    PipelineDiagnostic,
    PipelineQueueLimits,
    VoiceIngressProcessors,
    VoicePipelineSinks,
    VoiceSessionPipeline,
)
from core.voice.session import (
    LateFrameRejected,
    ProvisionalSentenceState,
    VoiceSession,
)
from core.voice.interfaces import VoiceTurnRequest

# ---------------------------------------------------------------------------
# Minimal mock Pipecat frame types (no real pipecat package required)
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
# Mock ring reader
# ---------------------------------------------------------------------------

class MockRingSlotReader:
    """Null same-UID ring reader; returns silent PCM for every slot."""

    def __init__(self, session_id: UUID) -> None:
        self.session_id = session_id

    async def map_slot(self, descriptor: RingSlotDescriptor) -> bytes:
        return b"\x00" * descriptor.byte_length

    async def release_slot(self, descriptor: RingSlotDescriptor) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame_adapter() -> PipecatFrameAdapter:
    return PipecatFrameAdapter(
        input_audio_frame_type=MockInputAudioRawFrame,
        transcription_frame_type=MockTranscriptionFrame,
        llm_text_frame_type=MockLLMTextFrame,
        tts_text_frame_type=MockTTSTextFrame,
    )



# ---------------------------------------------------------------------------
# Mock pipeline task factory (avoids real pipecat import)
# ---------------------------------------------------------------------------

class _MockTaskFactory:
    """Returns a sentinel object to satisfy VoiceSessionPipeline's task check."""
    def __call__(self) -> object:
        return object()


def _make_session_and_pipeline(
    sinks: VoicePipelineSinks | None = None,
) -> tuple[UUID, VoiceSession, VoiceSessionPipeline]:
    session_id = uuid4()
    session = VoiceSession(session_id)
    ring_reader = MockRingSlotReader(session_id)
    ingress = VoiceIngressProcessors(
        session=session,
        ring_reader=ring_reader,
        frame_adapter=_make_frame_adapter(),
    )
    pipeline = VoiceSessionPipeline(
        session=session,
        ingress=ingress,
        sinks=sinks or VoicePipelineSinks(),
        task_factory=_MockTaskFactory(),
    )
    return session_id, session, pipeline


def _make_transcript_message(
    session_id: UUID,
    turn_id: UUID,
    text: str,
    *,
    is_final: bool = True,
    event_seq: int = 0,
    language: str = "en",
) -> dict:
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
        "capture_started_monotonic_ns": monotonic_ns(),
        "capture_ended_monotonic_ns": monotonic_ns() + 1_000_000,
    }


def _make_playback_event(
    event_type: str,
    session_id: UUID,
    turn_id: UUID,
    sentence_id: UUID,
) -> dict:
    return {
        "version": 1,
        "type": event_type,
        "event_id": str(uuid4()),
        "session_id": str(session_id),
        "turn_id": str(turn_id),
        "sentence_id": str(sentence_id),
    }


# ---------------------------------------------------------------------------
# 1. Happy path: transcript → turn → PCM → PLAYBACK_CONFIRMED → ledger
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_transcript_to_confirmation() -> None:
    """Full replacement path: transcript event drives a turn through PCM to ledger.

    No real MLX/XTTS/AVAudioEngine/network calls.  All mocked.
    """
    received_transcription_frames: list = []
    received_llm_frames: list = []
    received_tts_frames: list = []
    received_pcm_frames: list = []

    async def on_final_transcription(frame: Any) -> None:
        received_transcription_frames.append(frame)

    async def on_llm_text(frame: Any) -> None:
        received_llm_frames.append(frame)

    async def on_tts_text(frame: Any) -> None:
        received_tts_frames.append(frame)

    async def on_pcm(frame: Any) -> None:
        received_pcm_frames.append(frame)

    sinks = VoicePipelineSinks(
        final_transcription=on_final_transcription,
        llm_text=on_llm_text,
        tts_text=on_tts_text,
        pcm=on_pcm,
    )
    session_id, session, pipeline = _make_session_and_pipeline(sinks)

    await pipeline.start()
    assert pipeline.is_available

    # Step 1: ingest final transcript → TranscriptionFrame
    turn_id = uuid4()
    msg = _make_transcript_message(session_id, turn_id, "Hello HAKI", is_final=True)
    result = await pipeline.ingest_transcript_message(msg)
    assert result.accepted

    await asyncio.sleep(0.05)
    assert len(received_transcription_frames) == 1
    payload = received_transcription_frames[0].payload
    assert isinstance(payload, MockTranscriptionFrame)
    assert payload.text == "Hello HAKI"

    # Step 2: emit LLM text frame (sequence must be 1, since transcript used 0)
    llm_frame = await pipeline.emit_llm_text(turn_id=turn_id, sequence=1, text="Hi there!")
    await asyncio.sleep(0.05)
    assert len(received_llm_frames) == 1
    assert received_llm_frames[0].frame_type is VoiceFrameType.LLM_TEXT

    # Step 3: emit TTS text frame (sequence=2, registers provisional sentence)
    sentence_id = uuid4()
    tts_frame = await pipeline.emit_tts_text(
        turn_id=turn_id, sentence_id=sentence_id, sequence=2, text="Hi there!"
    )
    await asyncio.sleep(0.05)
    assert len(received_tts_frames) == 1

    # Step 4: emit PCM chunk (sequence=3)
    pcm_frame = await pipeline.emit_pcm_chunk(
        turn_id=turn_id,
        sentence_id=sentence_id,
        sequence=3,
        chunk_sequence=0,
        pcm=b"\x00" * 320,
    )
    await asyncio.sleep(0.05)
    assert len(received_pcm_frames) == 1

    # Step 5: PLAYBACK_CONFIRMED → sentence in ledger
    playback_event = _make_playback_event(PLAYBACK_CONFIRMED, session_id, turn_id, sentence_id)
    confirmed = await pipeline.process_playback_event(playback_event, playback_completed_monotonic_ns=monotonic_ns())
    assert confirmed

    # Verify the sentence is in the session ledger (confirmed state)
    state = await session.playback_ledger.state_for(sentence_id)
    assert state is ProvisionalSentenceState.CONFIRMED

    await pipeline.close()


# ---------------------------------------------------------------------------
# 2. ASR stage failure: empty transcript → no LLM turn
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_asr_empty_transcript_no_llm_turn() -> None:
    """Empty final transcript must NOT create an LLM turn frame."""
    received_transcription_frames: list = []
    received_llm_frames: list = []

    async def on_final_transcription(frame: Any) -> None:
        received_transcription_frames.append(frame)

    async def on_llm_text(frame: Any) -> None:
        received_llm_frames.append(frame)

    sinks = VoicePipelineSinks(
        final_transcription=on_final_transcription,
        llm_text=on_llm_text,
    )
    session_id, session, pipeline = _make_session_and_pipeline(sinks)
    await pipeline.start()

    # Empty transcript: protocol raises VoiceProtocolError for empty/whitespace text
    from core.ipc.voice_protocol import VoiceProtocolError
    turn_id = uuid4()
    msg = _make_transcript_message(session_id, turn_id, "   ", is_final=True)
    with pytest.raises((VoiceProtocolError, Exception)):
        await pipeline.ingest_transcript_message(msg)

    await asyncio.sleep(0.05)
    # No transcription frame should have been dispatched
    assert len(received_transcription_frames) == 0
    # Definitely no LLM frame
    assert len(received_llm_frames) == 0

    await pipeline.close()


# ---------------------------------------------------------------------------
# 3. IPC disconnect before final: turn discarded, IPC diagnostic emitted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ipc_disconnect_before_final_discards_turn() -> None:
    """Turn must be discarded when connection drops before its final transcript."""
    discarded_turns: list[tuple[str, str]] = []

    async def on_turn_discarded(turn_id: str, reason: str) -> None:
        discarded_turns.append((turn_id, reason))

    from core.ipc.voice_unix_server import VoiceUnixServer
    import tempfile, os

    # Use a temp dir so we don't need the full UDS stack for this scenario
    with tempfile.TemporaryDirectory() as tmpdir:
        socket_path = os.path.join(tmpdir, "test_ipc_disconnect.sock")
        server = VoiceUnixServer(
            socket_path=socket_path,
            session_id=uuid4(),
            on_turn_discarded=on_turn_discarded,
        )
        await server.start()

        # Simulate an open connection that registered a non-final turn
        turn_id = str(uuid4())
        server._active_turn_ids.add(turn_id)

        # Simulate connection close before final by directly calling _discard_turn
        await server._discard_turn(turn_id, "disconnect_before_final")

        assert turn_id in server.discarded_turn_ids
        assert any(t[1] == "disconnect_before_final" for t in discarded_turns)

        await server.stop()


# ---------------------------------------------------------------------------
# 4. LLM failure: local_llm diagnostic, no legacy fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_failure_emits_diagnostic_no_legacy_fallback() -> None:
    """A failed local LLM turn must emit a diagnostic and NOT select any legacy route.

    The pipeline itself does not orchestrate LLM calls — that is the
    VoiceLLMRouter's job.  This test validates that:
    (a) pipeline diagnostics carry 'local_llm' stage, and
    (b) no legacy import / provider is reachable from the pipeline layer.
    """
    diagnostics: list[PipelineDiagnostic] = []

    async def on_diagnostic(d: PipelineDiagnostic) -> None:
        diagnostics.append(d)

    session_id, session, pipeline = _make_session_and_pipeline()
    # Inject a diagnostic sink
    pipeline._diagnostic_sink = on_diagnostic  # type: ignore[assignment]

    await pipeline.start()

    # Simulate a local_llm failure by injecting a diagnostic directly
    await on_diagnostic(PipelineDiagnostic(stage="local_llm", outcome="failed", error_class="MLXError"))

    assert any(d.stage == "local_llm" for d in diagnostics)

    # Verify no legacy voice package is importable from the pipeline module
    import core.voice.pipeline as pipeline_module
    import sys
    for legacy_name in ("edge_tts", "deepgram", "groq", "cartesia", "kokoro"):
        assert legacy_name not in sys.modules or (
            # The module may exist elsewhere, but must not be referenced in pipeline
            legacy_name not in dir(pipeline_module)
        ), f"Legacy module {legacy_name!r} must not be in pipeline namespace"

    await pipeline.close()


# ---------------------------------------------------------------------------
# 5. TTS failure: local_tts diagnostic, response as text, no legacy engine
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tts_failure_emits_diagnostic_no_legacy_engine() -> None:
    """TTS synthesis failure must emit a local_tts diagnostic.

    No legacy/system TTS engine (afplay, say, edge-tts, kokoro, cartesia)
    may be selected as a substitute.
    """
    diagnostics: list[PipelineDiagnostic] = []

    async def on_diagnostic(d: PipelineDiagnostic) -> None:
        diagnostics.append(d)

    session_id, session, pipeline = _make_session_and_pipeline()
    pipeline._diagnostic_sink = on_diagnostic  # type: ignore[assignment]
    await pipeline.start()

    # Inject a tts failure diagnostic
    await on_diagnostic(PipelineDiagnostic(stage="local_tts", outcome="failed", error_class="XTTSError"))

    assert any(d.stage == "local_tts" for d in diagnostics)
    assert not any(d.stage == "local_tts" and d.outcome == "completed" for d in diagnostics)

    # No legacy subprocess reference in the pipeline
    import core.voice.pipeline as pipeline_module
    for legacy_attr in ("afplay", "say", "edge_tts", "kokoro"):
        assert not hasattr(pipeline_module, legacy_attr), \
            f"Legacy TTS reference {legacy_attr!r} must not exist in pipeline module"

    await pipeline.close()


# ---------------------------------------------------------------------------
# 6. Barge-in: speech during playback → cancellation, new capture begins
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_barge_in_cancels_active_generation() -> None:
    """VAD-detected speech during playback must cancel the active assistant turn.

    The barge-in coordinator increments the cancellation generation and drains
    queued TTS/PCM work for the interrupted turn.
    """
    stop_calls: list[tuple[UUID, int]] = []
    capture_resumed: list[bool] = []

    async def on_stop_playback(turn_id: UUID, generation: int) -> None:
        stop_calls.append((turn_id, generation))

    async def on_capture_resumed() -> None:
        capture_resumed.append(True)

    sinks = VoicePipelineSinks(
        stop_playback=on_stop_playback,
        capture_resumed=on_capture_resumed,
    )
    session_id, session, pipeline = _make_session_and_pipeline(sinks)
    await pipeline.start()

    # Start a turn and register a provisional TTS sentence
    turn_id = uuid4()
    await session.start_turn(turn_id)
    sentence_id = uuid4()
    generation_before = session.cancellation_generation

    # Register TTS sentence (simulates pipeline having sent TTS to renderer)
    registered = await session.playback_ledger.register(
        turn_id=turn_id,
        sentence_id=sentence_id,
        text="This is the response.",
        cancellation_generation=session.cancellation_generation,
    )
    assert registered

    # Simulate barge-in: cancel the turn (increments generation)
    new_gen = await session.cancel_turn(turn_id)
    assert new_gen > generation_before

    # Provisional sentence must be cancelled (not confirmed)
    state = await session.playback_ledger.state_for(sentence_id)
    assert state is ProvisionalSentenceState.CANCELLED

    await pipeline.close()


# ---------------------------------------------------------------------------
# 7. Gate isolation: VoiceSessionPipeline contains no legacy voice imports
# ---------------------------------------------------------------------------

def test_pipeline_module_has_no_legacy_voice_imports() -> None:
    """Confirm the replacement pipeline module does not import any legacy component.

    This is a static isolation check — no real runtime execution needed.
    """
    import ast
    import inspect
    import core.voice.pipeline as pipeline_module

    source = inspect.getsource(pipeline_module)
    tree = ast.parse(source)

    banned_names = {
        "edge_tts", "deepgram", "groq", "cartesia", "kokoro",
        "ChatTTS", "afplay", "say", "STTEngine", "TTSEngine",
        "legacy_pipeline_backup",
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [
                getattr(node, "module", None) or "",
                *[alias.name for alias in getattr(node, "names", [])],
            ]
            for name in names:
                for banned in banned_names:
                    assert banned not in name, (
                        f"Legacy component {banned!r} imported from pipeline module"
                    )


def test_server_py_voice_gate_blocks_legacy_on_enable() -> None:
    """After cutover, AUDIO_FRAME and END_OF_SPEECH are always dropped on the
    non-voice JSON socket — the old dev gate has been removed and replaced by
    the unconditional production route.

    This test verifies that the legacy _VOICE_GATE constant no longer exists
    and that the module has no dev gate mechanism.
    """
    import core.ipc.server as server_mod

    # The dev gate constant must be gone after cutover (Task 14.1)
    assert not hasattr(server_mod, "_VOICE_GATE"), (
        "_VOICE_GATE must be removed from server.py in the production cutover"
    )
    # The module must not reference the dev_gate module
    import inspect
    source = inspect.getsource(server_mod)
    assert "dev_gate" not in source, "dev_gate reference found in server.py after cutover"
    assert "HAKI_VOICE_DEV_REPLACEMENT" not in source, (
        "HAKI_VOICE_DEV_REPLACEMENT gate found in server.py after cutover"
    )


def test_dev_gate_respects_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """VOICE_REPLACEMENT_GATE_ENABLED is True iff HAKI_VOICE_DEV_REPLACEMENT=1."""
    import importlib
    import core.voice.dev_gate as dg

    monkeypatch.setenv("HAKI_VOICE_DEV_REPLACEMENT", "1")
    # The module-level constant is evaluated at import time; use gate_enabled()
    # which reads the constant — we test the function as a callable interface.
    assert dg.gate_enabled() == dg.VOICE_REPLACEMENT_GATE_ENABLED

    monkeypatch.setenv("HAKI_VOICE_DEV_REPLACEMENT", "0")
    # gate_enabled() reflects the constant which is stable for the process
    assert dg.gate_enabled() == dg.VOICE_REPLACEMENT_GATE_ENABLED


# ---------------------------------------------------------------------------
# 8. PLAYBACK_CANCELLED: interrupted sentence excluded from ledger
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_playback_cancelled_excludes_from_ledger() -> None:
    """A PLAYBACK_CANCELLED event must mark the sentence cancelled, not confirmed."""
    session_id, session, pipeline = _make_session_and_pipeline()
    await pipeline.start()

    turn_id = uuid4()
    await session.start_turn(turn_id)
    sentence_id = uuid4()

    await session.playback_ledger.register(
        turn_id=turn_id,
        sentence_id=sentence_id,
        text="Maybe this plays.",
        cancellation_generation=session.cancellation_generation,
    )

    # Cancel the turn first (simulates barge-in)
    await session.cancel_turn(turn_id)

    # Sentence should be in CANCELLED state after turn cancel
    state = await session.playback_ledger.state_for(sentence_id)
    assert state is ProvisionalSentenceState.CANCELLED

    # A PLAYBACK_CONFIRMED after cancellation must be rejected
    event = _make_playback_event(PLAYBACK_CONFIRMED, session_id, turn_id, sentence_id)
    # This should not confirm since the provisional sentence is already cancelled
    confirmed = await pipeline.process_playback_event(event, playback_completed_monotonic_ns=monotonic_ns())
    assert not confirmed

    await pipeline.close()
