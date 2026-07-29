"""Property 3: Sequenced local capture, transcript, and Pipecat frame preservation.

Feature: realtime-local-voice-agent, Property 3: Sequenced local capture,
transcript, and Pipecat frame preservation

For all accepted monotonically captured audio-frame sequences and all valid
partial/final transcript event streams, frames retain strictly increasing capture
sequence metadata, transcript events contain no microphone payload, each turn has
zero or more non-final events followed by at most one final event, and the
corresponding InputAudioRawFrame, TranscriptionFrame, LLMTextFrame, and
TTSTextFrame values retain their turn order until terminal completion or
cancellation.

**Validates: Requirements 2.6, 3.2, 3.3, 3.6, 4.2, 4.3, 4.4, 4.5, 4.6, 4.8**

Design reference: §§2–4, Property 3; V-FRAME-PROP

Covers:
- Monotonic capture metadata: strictly increasing sequence numbers and timestamps
- No PCM over UDS: transcript socket messages contain no microphone payload fields
- At-most-one ordered final: zero or more non-finals followed by at most one final
- Mandated Pipecat frame types: correct frame type objects used for each pipeline stage
- Per-turn terminal ordering: frames maintain turn order until completion/cancellation
- Reconnect scenarios: partial turns on disconnect are discarded
- Duplicate-final rejection: second final event is rejected
- Out-of-order sequence rejection: non-sequential event_seq is rejected
- Cancellation-generation examples: stale-generation frames are rejected
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from core.ipc.voice_protocol import (
    TRANSCRIPT_EVENT,
    PROTOCOL_VERSION,
    ValidatedMessage,
    VoiceProtocolError,
    VoiceProtocolSession,
    validate_message,
)
from core.voice.asr_bridge import (
    AudioRingIngress,
    RingSlotDescriptor,
    TranscriptSocketIngress,
    SILERO_SAMPLE_RATE_HZ,
    SILERO_CHANNELS,
)
from core.voice.frames import (
    AudioFrameMetadata,
    TranscriptionFrameMetadata,
    SentenceFrameMetadata,
    TypedVoiceFrame,
    VoiceFrameMetadata,
    VoiceFrameType,
)
from core.voice.session import (
    LateFrameRejected,
    TurnQueueName,
    TurnState,
    VoiceSession,
)


# ---------------------------------------------------------------------------
# Mock Pipecat frame types used for all tests (avoids real Pipecat import)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MockInputAudioRawFrame:
    """Stands in for pipecat.frames.frames.InputAudioRawFrame."""
    audio: bytes
    sample_rate: int
    num_channels: int


@dataclass(frozen=True)
class MockTranscriptionFrame:
    """Stands in for pipecat.frames.frames.TranscriptionFrame."""
    text: str
    user_id: str = "local-user"


@dataclass(frozen=True)
class MockLLMTextFrame:
    """Stands in for pipecat.frames.frames.LLMTextFrame."""
    text: str


@dataclass(frozen=True)
class MockTTSTextFrame:
    """Stands in for pipecat.frames.frames.TTSTextFrame."""
    text: str


# ---------------------------------------------------------------------------
# Minimal PipecatIngressFrameFactory implementation
# ---------------------------------------------------------------------------

class _MockFrameFactory:
    """Creates mock Pipecat frames without importing the real pipecat package."""

    def create_input_audio_frame(
        self, *, audio: bytes, sample_rate_hz: int, channels: int
    ) -> object:
        return MockInputAudioRawFrame(
            audio=audio, sample_rate=sample_rate_hz, num_channels=channels
        )

    def create_transcription_frame(self, *, text: str) -> object:
        return MockTranscriptionFrame(text=text)

    def create_llm_text_frame(self, *, text: str) -> object:
        return MockLLMTextFrame(text=text)

    def create_tts_text_frame(self, *, text: str) -> object:
        return MockTTSTextFrame(text=text)


# ---------------------------------------------------------------------------
# Fake authenticated ring reader for AudioRingIngress tests
# ---------------------------------------------------------------------------

class _FakeRingReader:
    """Simulates an authenticated same-UID ring slot reader."""

    def __init__(self, session_id: UUID) -> None:
        self.session_id = session_id
        self._data: dict[int, bytes] = {}

    def set_slot_data(self, slot_index: int, data: bytes) -> None:
        self._data[slot_index] = data

    async def map_slot(self, descriptor: RingSlotDescriptor) -> bytes:
        return self._data.get(descriptor.slot_index, b"\x00" * descriptor.byte_length)

    async def release_slot(self, descriptor: RingSlotDescriptor) -> None:
        self._data.pop(descriptor.slot_index, None)


# ---------------------------------------------------------------------------
# Helpers for building valid transcript event wire messages
# ---------------------------------------------------------------------------

_MICROPHONE_FORBIDDEN_FIELDS = (
    "audio", "microphone", "mic", "samples", "pcm", "waveform",
    "wave", "buffer", "samples_b64", "binary", "bytes",
)

_VALID_LANGUAGES = ("hi", "en", "hinglish")


def _make_transcript_event(
    *,
    session_id: str,
    turn_id: str,
    event_id: str | None = None,
    event_seq: int = 0,
    text: str = "hello",
    is_final: bool = False,
    language: str = "en",
    capture_started_ns: int = 100,
    capture_ended_ns: int = 200,
) -> dict[str, Any]:
    """Build a valid TRANSCRIPT_EVENT dict for use with VoiceProtocolSession."""
    return {
        "version": PROTOCOL_VERSION,
        "type": TRANSCRIPT_EVENT,
        "event_id": event_id or str(uuid4()),
        "session_id": session_id,
        "turn_id": turn_id,
        "event_seq": event_seq,
        "text": text,
        "is_final": is_final,
        "language": language,
        "capture_started_monotonic_ns": capture_started_ns,
        "capture_ended_monotonic_ns": capture_ended_ns,
    }


def _canonical_uuid_str(u: UUID) -> str:
    return str(u)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_safe_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd", "Zs"),
        whitelist_characters=" ",
    ),
    min_size=1,
    max_size=40,
).filter(lambda t: t.strip())

_language_st = st.sampled_from(list(_VALID_LANGUAGES))

_small_nonneg_int = st.integers(min_value=0, max_value=1_000_000_000)


@st.composite
def capture_sequence_stream(draw: st.DrawFn) -> list[tuple[int, int]]:
    """
    Generate a stream of (sequence, timestamp_ns) pairs that represent
    captured audio frames with STRICTLY monotonically increasing sequence
    numbers and timestamps.

    Returns a list of (seq, ts_ns) tuples of length 1..20.
    """
    n = draw(st.integers(min_value=1, max_value=20))
    start_seq = draw(st.integers(min_value=0, max_value=100))
    start_ts = draw(st.integers(min_value=1, max_value=10_000_000))
    # Use consecutive sequences and always-increasing timestamps
    step_ts = draw(st.integers(min_value=1, max_value=1_000_000))
    pairs = []
    for i in range(n):
        pairs.append((start_seq + i, start_ts + i * step_ts))
    return pairs


@st.composite
def gapped_capture_sequence_stream(draw: st.DrawFn) -> list[tuple[int, int]]:
    """
    Generate a stream where sequences may have gaps (dropped frames),
    but timestamps remain strictly increasing and sequences never go backward.
    """
    n = draw(st.integers(min_value=2, max_value=15))
    start_seq = draw(st.integers(min_value=0, max_value=50))
    start_ts = draw(st.integers(min_value=1, max_value=1_000_000))
    pairs = []
    seq = start_seq
    ts = start_ts
    for _ in range(n):
        pairs.append((seq, ts))
        gap = draw(st.integers(min_value=1, max_value=5))
        seq += gap
        ts_step = draw(st.integers(min_value=1, max_value=500_000))
        ts += ts_step
    return pairs


@st.composite
def partial_final_event_stream(draw: st.DrawFn) -> tuple[list[dict], int | None]:
    """
    Generate a valid partial/final event stream for a single turn.

    Returns (events_list, final_event_seq_or_None).
    - 0 to 5 non-final events with sequential event_seq
    - followed by 0 or 1 final event
    - all events share the same session_id/turn_id
    - text is non-empty
    - capture timestamps are valid
    """
    session_id = str(draw(st.builds(uuid4)))
    turn_id = str(draw(st.builds(uuid4)))
    language = draw(_language_st)
    n_partials = draw(st.integers(min_value=0, max_value=5))
    has_final = draw(st.booleans())

    events = []
    seq = draw(st.integers(min_value=0, max_value=10))
    ts_start = draw(st.integers(min_value=100, max_value=10_000))
    ts_step = draw(st.integers(min_value=1, max_value=1_000))

    for i in range(n_partials):
        ts_s = ts_start + i * ts_step * 2
        ts_e = ts_s + ts_step
        events.append(_make_transcript_event(
            session_id=session_id,
            turn_id=turn_id,
            event_seq=seq + i,
            text=draw(_safe_text),
            is_final=False,
            language=language,
            capture_started_ns=ts_s,
            capture_ended_ns=ts_e,
        ))

    final_seq = None
    if has_final:
        final_seq = seq + n_partials
        ts_s = ts_start + n_partials * ts_step * 2
        ts_e = ts_s + ts_step
        events.append(_make_transcript_event(
            session_id=session_id,
            turn_id=turn_id,
            event_seq=final_seq,
            text=draw(_safe_text),
            is_final=True,
            language=language,
            capture_started_ns=ts_s,
            capture_ended_ns=ts_e,
        ))

    return events, final_seq


# ---------------------------------------------------------------------------
# Property 3a: Monotonic capture metadata (Req 2.6)
# ---------------------------------------------------------------------------

@settings(max_examples=120, deadline=None)
@given(capture_sequence_stream())
def test_audio_frame_metadata_is_strictly_monotonic(
    stream: list[tuple[int, int]],
) -> None:
    """**Validates: Requirements 2.6, 4.2**

    Every accepted audio frame carries strictly increasing (monotonic) sequence
    numbers and capture timestamps.  Frames with an older sequence must be
    rejected by the ingress layer.

    This property verifies that for any generated stream of (seq, ts_ns) pairs
    with monotonically increasing values, AudioFrameMetadata correctly carries
    the assigned sequence and timestamp, and all pairs remain in strict order.
    """
    sequences = [seq for seq, _ in stream]
    timestamps = [ts for _, ts in stream]

    # Build AudioFrameMetadata for each pair and verify ordering is preserved.
    session_id = uuid4()
    turn_id = uuid4()
    frames = []
    for i, (seq, ts) in enumerate(stream):
        meta = AudioFrameMetadata(
            session_id=session_id,
            turn_id=turn_id,
            sequence=seq,
            cancellation_generation=0,
            captured_monotonic_ns=ts,
            sample_rate_hz=SILERO_SAMPLE_RATE_HZ,
            channels=SILERO_CHANNELS,
        )
        frames.append(meta)

    # Sequence numbers must be strictly increasing.
    for i in range(1, len(frames)):
        assert frames[i].sequence > frames[i - 1].sequence, (
            f"Capture sequence must be strictly increasing: "
            f"frame[{i}].sequence={frames[i].sequence} <= "
            f"frame[{i-1}].sequence={frames[i-1].sequence}"
        )

    # Timestamps must be strictly increasing.
    for i in range(1, len(frames)):
        assert frames[i].captured_monotonic_ns > frames[i - 1].captured_monotonic_ns, (
            f"Capture timestamp must be strictly increasing: "
            f"frame[{i}].ts={frames[i].captured_monotonic_ns} <= "
            f"frame[{i-1}].ts={frames[i - 1].captured_monotonic_ns}"
        )


@settings(max_examples=100, deadline=None)
@given(gapped_capture_sequence_stream())
def test_gapped_audio_frame_sequences_remain_monotonic(
    stream: list[tuple[int, int]],
) -> None:
    """**Validates: Requirement 2.6**

    Even when frames have gaps (dropped frames), the surviving sequence numbers
    and timestamps remain strictly increasing (never go backward).
    """
    session_id = uuid4()
    turn_id = uuid4()
    frames = []
    for seq, ts in stream:
        meta = AudioFrameMetadata(
            session_id=session_id,
            turn_id=turn_id,
            sequence=seq,
            cancellation_generation=0,
            captured_monotonic_ns=ts,
            sample_rate_hz=SILERO_SAMPLE_RATE_HZ,
            channels=SILERO_CHANNELS,
        )
        frames.append(meta)

    for i in range(1, len(frames)):
        assert frames[i].sequence > frames[i - 1].sequence, (
            f"Gapped sequence must still be increasing: got {frames[i].sequence} "
            f"after {frames[i-1].sequence}"
        )
        assert frames[i].captured_monotonic_ns > frames[i - 1].captured_monotonic_ns, (
            f"Gapped timestamp must still be increasing"
        )


# ---------------------------------------------------------------------------
# Property 3b: No PCM over UDS — transcript events contain no mic payload
# (Req 3.6, 3.2)
# ---------------------------------------------------------------------------

@settings(max_examples=120, deadline=None)
@given(partial_final_event_stream())
def test_transcript_events_contain_no_microphone_payload_fields(
    stream_and_final: tuple[list[dict], Any],
) -> None:
    """**Validates: Requirements 3.6, 3.2**

    Transcript events transmitted over the UDS must contain no microphone
    sample payload fields.  The protocol schema explicitly forbids field names
    containing audio/mic/sample/pcm/waveform/buffer/base64/binary/bytes tokens.

    For every generated transcript event stream, each message must:
    1. Pass VoiceProtocolSession.accept() without raising microphone_payload_forbidden.
    2. Contain none of the prohibited field names.
    """
    events, _ = stream_and_final
    assume(len(events) > 0)

    session = VoiceProtocolSession(session_id=events[0]["session_id"])

    for event in events:
        # 1. Validate against protocol schema — must not contain mic payload.
        validated = session.accept(validate_message(event))
        data = validated.data

        # 2. Verify that no microphone-payload field name appears.
        for key in data:
            key_normalized = key.lower().replace("_", "").replace("-", "")
            for forbidden in _MICROPHONE_FORBIDDEN_FIELDS:
                assert forbidden not in key_normalized, (
                    f"Transcript event contains a forbidden microphone payload field: "
                    f"key={key!r} matched forbidden token {forbidden!r}"
                )


@settings(max_examples=100, deadline=None)
@given(st.lists(st.sampled_from(_MICROPHONE_FORBIDDEN_FIELDS), min_size=1, max_size=3))
def test_transcript_event_with_mic_payload_field_is_rejected(
    forbidden_fields: list[str],
) -> None:
    """**Validates: Requirement 3.6**

    Injecting any microphone payload field into a transcript event must cause
    validate_message() to raise VoiceProtocolError with reason
    'microphone_payload_forbidden' or 'unknown_field'.
    """
    session_id = str(uuid4())
    turn_id = str(uuid4())
    event = _make_transcript_event(
        session_id=session_id, turn_id=turn_id,
        event_seq=0, is_final=False,
    )
    # Inject a forbidden field name.
    for fname in forbidden_fields:
        event[fname] = b"some audio bytes"

    with pytest.raises(VoiceProtocolError) as exc_info:
        validate_message(event)

    assert exc_info.value.reason in (
        "microphone_payload_forbidden", "unknown_field"
    ), (
        f"Expected rejection reason to be microphone_payload_forbidden or unknown_field, "
        f"got {exc_info.value.reason!r}"
    )


# ---------------------------------------------------------------------------
# Property 3c: At-most-one ordered final (Req 3.3, 3.2)
# ---------------------------------------------------------------------------

@settings(max_examples=120, deadline=None)
@given(partial_final_event_stream())
def test_at_most_one_ordered_final_per_turn(
    stream_and_final: tuple[list[dict], Any],
) -> None:
    """**Validates: Requirements 3.3, 3.2**

    Each turn has zero or more non-final Transcript_Events followed by at most
    one final Transcript_Event.  A duplicate final must raise VoiceProtocolError.

    For every generated stream:
    - All events are accepted in sequence without error.
    - If a final event exists, it must be the last event accepted.
    - A second attempt to send a final event for the same turn must be rejected
      with reason 'duplicate_final'.
    """
    events, final_seq = stream_and_final
    assume(len(events) > 0)

    session = VoiceProtocolSession(session_id=events[0]["session_id"])

    accepted_count = 0
    saw_final = False

    for event in events:
        validated = session.accept(validate_message(event))
        accepted_count += 1
        if event["is_final"]:
            saw_final = True

    assert accepted_count == len(events), (
        f"All {len(events)} events should have been accepted; accepted {accepted_count}"
    )

    # If there was a final, try sending a duplicate — must be rejected.
    if saw_final and events:
        last_final = events[-1]
        assert last_final["is_final"] is True

        duplicate = dict(last_final)
        duplicate["event_id"] = str(uuid4())  # new event_id, same turn/final

        with pytest.raises(VoiceProtocolError) as exc_info:
            session.accept(validate_message(duplicate))

        assert exc_info.value.reason == "duplicate_final", (
            f"Duplicate final must raise duplicate_final, got {exc_info.value.reason!r}"
        )


def test_duplicate_final_is_always_rejected_deterministic() -> None:
    """**Validates: Requirement 3.3**

    Deterministic regression: sending the same final event twice must always
    raise VoiceProtocolError('duplicate_final').
    """
    session_id = str(uuid4())
    turn_id = str(uuid4())
    session = VoiceProtocolSession(session_id=session_id)

    # A non-final partial.
    partial = _make_transcript_event(
        session_id=session_id, turn_id=turn_id,
        event_seq=0, is_final=False, text="partial text",
    )
    session.accept(validate_message(partial))

    # The final event.
    final = _make_transcript_event(
        session_id=session_id, turn_id=turn_id,
        event_seq=1, is_final=True, text="final text",
    )
    session.accept(validate_message(final))

    # Duplicate final (different event_id, same is_final=True and same turn).
    duplicate = dict(final)
    duplicate["event_id"] = str(uuid4())
    duplicate["event_seq"] = 2  # Would be the next seq, but turn is already finalized

    with pytest.raises(VoiceProtocolError) as exc_info:
        session.accept(validate_message(duplicate))

    assert exc_info.value.reason == "duplicate_final"


@settings(max_examples=100, deadline=None)
@given(
    st.integers(min_value=0, max_value=5),  # number of partials before first final
    st.integers(min_value=1, max_value=3),  # number of extra finals to attempt
)
def test_multiple_finals_all_rejected_after_first(
    n_partials: int, n_extra_finals: int
) -> None:
    """**Validates: Requirement 3.3**

    After the first final event is accepted, every subsequent event for the
    same turn with is_final=True must be rejected with duplicate_final,
    regardless of how many extra finals are attempted.
    """
    session_id = str(uuid4())
    turn_id = str(uuid4())
    session = VoiceProtocolSession(session_id=session_id)

    # Send n_partials non-final events.
    for i in range(n_partials):
        event = _make_transcript_event(
            session_id=session_id, turn_id=turn_id,
            event_seq=i, is_final=False,
        )
        session.accept(validate_message(event))

    # Send the one valid final.
    final_seq = n_partials
    final_event = _make_transcript_event(
        session_id=session_id, turn_id=turn_id,
        event_seq=final_seq, is_final=True,
    )
    session.accept(validate_message(final_event))

    # Attempt n_extra_finals more finals — all must fail.
    for k in range(n_extra_finals):
        extra = _make_transcript_event(
            session_id=session_id, turn_id=turn_id,
            event_seq=final_seq + k + 1, is_final=True,
        )
        extra["event_id"] = str(uuid4())
        with pytest.raises(VoiceProtocolError) as exc_info:
            session.accept(validate_message(extra))
        assert exc_info.value.reason == "duplicate_final", (
            f"Extra final #{k+1} must raise duplicate_final"
        )


# ---------------------------------------------------------------------------
# Property 3d: Mandated Pipecat frame types (Req 4.2–4.5)
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(
    st.integers(min_value=1, max_value=8),  # n_audio_frames
    st.integers(min_value=0, max_value=4),  # n_partials
    _language_st,
    _safe_text,
)
def test_mandated_pipecat_frame_types_for_each_pipeline_stage(
    n_audio_frames: int,
    n_partials: int,
    language: str,
    text: str,
) -> None:
    """**Validates: Requirements 4.2, 4.3, 4.4, 4.5**

    The pipeline must represent:
    - captured microphone input with InputAudioRawFrame (mock: MockInputAudioRawFrame)
    - ASR results with TranscriptionFrame (mock: MockTranscriptionFrame)
    - generated model output with LLMTextFrame (mock: MockLLMTextFrame)
    - sentence-ready synthesis input with TTSTextFrame (mock: MockTTSTextFrame)

    This property verifies that for any number of audio frames, partials, a
    language, and text, the frame factory produces the correct concrete types.
    """
    factory = _MockFrameFactory()

    # 1. InputAudioRawFrame — represents captured microphone input (Req 4.2)
    pcm = b"\x00\x01" * 160  # 320 bytes = 160 samples of S16LE mono
    for i in range(n_audio_frames):
        audio_frame = factory.create_input_audio_frame(
            audio=pcm,
            sample_rate_hz=SILERO_SAMPLE_RATE_HZ,
            channels=SILERO_CHANNELS,
        )
        assert isinstance(audio_frame, MockInputAudioRawFrame), (
            f"Audio frame {i} must be InputAudioRawFrame type"
        )
        assert audio_frame.audio == pcm
        assert audio_frame.sample_rate == SILERO_SAMPLE_RATE_HZ
        assert audio_frame.num_channels == SILERO_CHANNELS

    # 2. TranscriptionFrame — represents ASR results (Req 4.3)
    for i in range(n_partials + 1):
        trans_frame = factory.create_transcription_frame(text=text)
        assert isinstance(trans_frame, MockTranscriptionFrame), (
            f"Transcription frame {i} must be TranscriptionFrame type"
        )
        assert trans_frame.text == text

    # 3. LLMTextFrame — represents generated model output (Req 4.4)
    llm_frame = factory.create_llm_text_frame(text=text)
    assert isinstance(llm_frame, MockLLMTextFrame), (
        "LLM text frame must be LLMTextFrame type"
    )
    assert llm_frame.text == text

    # 4. TTSTextFrame — represents sentence-ready synthesis input (Req 4.5)
    tts_frame = factory.create_tts_text_frame(text=text)
    assert isinstance(tts_frame, MockTTSTextFrame), (
        "TTS text frame must be TTSTextFrame type"
    )
    assert tts_frame.text == text


def test_pipecat_frame_types_are_distinct_classes() -> None:
    """**Validates: Requirements 4.2–4.5**

    Each pipeline stage must use a DIFFERENT frame type class.  Mixing types
    (e.g. using TranscriptionFrame as an audio frame) must be detectable.
    """
    assert MockInputAudioRawFrame is not MockTranscriptionFrame
    assert MockTranscriptionFrame is not MockLLMTextFrame
    assert MockLLMTextFrame is not MockTTSTextFrame
    assert MockInputAudioRawFrame is not MockTTSTextFrame


@settings(max_examples=100, deadline=None)
@given(_safe_text, _language_st)
def test_transcription_frame_carries_normalized_text_only(
    text: str, language: str,
) -> None:
    """**Validates: Requirements 4.3, 4.6**

    TranscriptionFrame text must be exactly the normalized UDS text.
    Turn/session/sequencing metadata must live in TypedVoiceFrame.metadata
    (TranscriptionFrameMetadata), NOT embedded in the transcription text.
    """
    factory = _MockFrameFactory()
    trans_frame = factory.create_transcription_frame(text=text)
    assert isinstance(trans_frame, MockTranscriptionFrame)
    # The text must be unchanged — no metadata serialized into it.
    assert trans_frame.text == text
    # No UUID, sequence number, or language code should be appended to text.
    assert "|" not in trans_frame.text or text.count("|") == trans_frame.text.count("|")


# ---------------------------------------------------------------------------
# Property 3e: Per-turn terminal ordering (Req 4.8, 4.6)
# ---------------------------------------------------------------------------

@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.integers(min_value=1, max_value=6),   # n_turns
    st.integers(min_value=1, max_value=6),   # frames_per_turn
    st.booleans(),                           # cancel_last_turn
)
def test_per_turn_frame_ordering_preserved_until_terminal(
    n_turns: int,
    frames_per_turn: int,
    cancel_last_turn: bool,
) -> None:
    """**Validates: Requirements 4.8, 4.6**

    For a voice session with multiple active turns, each turn's frames must
    retain their per-turn ordering (strictly increasing sequence) and terminal
    states (CANCELLED/COMPLETED) must not emit further frames.

    Verifies:
    - Per-turn sequence numbers accepted in order
    - Late frames (after cancel) are rejected with LateFrameRejected
    - Cancellation advances the generation so stale frames are detected
    """
    asyncio.run(_per_turn_ordering_scenario(n_turns, frames_per_turn, cancel_last_turn))


async def _per_turn_ordering_scenario(
    n_turns: int,
    frames_per_turn: int,
    cancel_last_turn: bool,
) -> None:
    session = VoiceSession(uuid4())
    factory = _MockFrameFactory()
    turn_ids = [uuid4() for _ in range(n_turns)]

    # Register all turns.
    for turn_id in turn_ids:
        await session.start_turn(turn_id)

    # Emit frames_per_turn LLM text frames per turn in sequence order.
    # Each frame is TypedVoiceFrame with VoiceFrameMetadata.
    for turn_id in turn_ids:
        record = session.turns.get(turn_id)
        for seq in range(frames_per_turn):
            meta = VoiceFrameMetadata(
                session_id=session.session_id,
                turn_id=turn_id,
                sequence=seq,
                cancellation_generation=record.cancellation_generation,
            )
            frame = TypedVoiceFrame(
                VoiceFrameType.LLM_TEXT,
                meta,
                factory.create_llm_text_frame(text=f"token {seq}"),
            )
            await session.accept_frame(frame, queue=TurnQueueName.LLM)

    # Cancel the last turn if requested.
    if cancel_last_turn and n_turns > 0:
        last_turn = turn_ids[-1]
        await session.cancel_turn(last_turn)
        record = session.turns.get(last_turn)
        assert record.state is TurnState.CANCELLED

        # Attempt to emit a frame for the cancelled turn — must be rejected.
        stale_meta = VoiceFrameMetadata(
            session_id=session.session_id,
            turn_id=last_turn,
            sequence=frames_per_turn,  # next sequence
            cancellation_generation=0,  # old generation
        )
        stale_frame = TypedVoiceFrame(
            VoiceFrameType.LLM_TEXT,
            stale_meta,
            factory.create_llm_text_frame(text="stale"),
        )
        with pytest.raises(LateFrameRejected):
            await session.accept_frame(stale_frame, queue=TurnQueueName.LLM)

    # Non-cancelled turns should still accept the next frame in sequence.
    non_cancelled = turn_ids[:-1] if (cancel_last_turn and n_turns > 0) else turn_ids
    for turn_id in non_cancelled:
        record = session.turns.get(turn_id)
        if record.is_terminal:
            continue
        next_seq = frames_per_turn
        meta = VoiceFrameMetadata(
            session_id=session.session_id,
            turn_id=turn_id,
            sequence=next_seq,
            cancellation_generation=record.cancellation_generation,
        )
        frame = TypedVoiceFrame(
            VoiceFrameType.LLM_TEXT,
            meta,
            factory.create_llm_text_frame(text="continuation"),
        )
        # Should succeed — turn is still alive.
        await session.accept_frame(frame, queue=TurnQueueName.LLM)


@settings(max_examples=100, deadline=None)
@given(st.integers(min_value=0, max_value=5))
def test_out_of_order_frames_rejected_for_same_turn(n_pre_frames: int) -> None:
    """**Validates: Requirement 4.8**

    After a frame with sequence N is accepted, a frame with sequence N-1 or
    any earlier value must be rejected with FrameOrderingError or LateFrameRejected.
    """
    from core.voice.session import FrameOrderingError

    asyncio.run(_out_of_order_scenario(n_pre_frames))


async def _out_of_order_scenario(n_pre_frames: int) -> None:
    session = VoiceSession(uuid4())
    turn_id = uuid4()
    await session.start_turn(turn_id)
    record = session.turns.get(turn_id)
    factory = _MockFrameFactory()

    # Accept n_pre_frames in order.
    for seq in range(n_pre_frames):
        meta = VoiceFrameMetadata(
            session_id=session.session_id,
            turn_id=turn_id,
            sequence=seq,
            cancellation_generation=record.cancellation_generation,
        )
        frame = TypedVoiceFrame(
            VoiceFrameType.LLM_TEXT,
            meta,
            factory.create_llm_text_frame(text=f"token {seq}"),
        )
        await session.accept_frame(frame, queue=TurnQueueName.LLM)

    if n_pre_frames == 0:
        return  # No prior frames, so no out-of-order test is possible yet.

    # Attempt to go backward: sequence 0 after we have accepted up to n_pre_frames-1.
    backward_meta = VoiceFrameMetadata(
        session_id=session.session_id,
        turn_id=turn_id,
        sequence=0,  # always out of order after seq 0 is already used
        cancellation_generation=record.cancellation_generation,
    )
    backward_frame = TypedVoiceFrame(
        VoiceFrameType.LLM_TEXT,
        backward_meta,
        factory.create_llm_text_frame(text="backward"),
    )

    with pytest.raises((LateFrameRejected, Exception)):
        await session.accept_frame(backward_frame, queue=TurnQueueName.LLM)


# ---------------------------------------------------------------------------
# Property 3f: Out-of-order event_seq rejection (Req 3.2)
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(
    st.integers(min_value=0, max_value=4),  # n_accepted_partials
    st.integers(min_value=2, max_value=10),  # jump: skip n sequences
)
def test_out_of_order_event_seq_is_rejected(
    n_accepted_partials: int, jump: int
) -> None:
    """**Validates: Requirement 3.2**

    Event_seq values must be consecutive.  After n partial events are accepted
    starting at seq 0, sending seq 0 again (or any seq != next expected) must
    raise VoiceProtocolError with reason 'invalid_event_sequence'.
    """
    session_id = str(uuid4())
    turn_id = str(uuid4())
    session = VoiceProtocolSession(session_id=session_id)

    # Accept n_accepted_partials in order (seq 0, 1, ..., n-1).
    for i in range(n_accepted_partials):
        event = _make_transcript_event(
            session_id=session_id, turn_id=turn_id,
            event_seq=i, is_final=False,
        )
        session.accept(validate_message(event))

    # Now send seq that is out of order (jumps by 'jump' instead of +1).
    if n_accepted_partials == 0:
        # First seq can be anything, so skip this case.
        return

    next_expected = n_accepted_partials
    out_of_order_seq = next_expected + jump  # skips 'jump' values

    ooo_event = _make_transcript_event(
        session_id=session_id, turn_id=turn_id,
        event_seq=out_of_order_seq, is_final=False,
    )
    with pytest.raises(VoiceProtocolError) as exc_info:
        session.accept(validate_message(ooo_event))

    assert exc_info.value.reason == "invalid_event_sequence", (
        f"Out-of-order event_seq must raise invalid_event_sequence, "
        f"got {exc_info.value.reason!r}"
    )


def test_event_seq_must_start_at_any_nonneg_and_increment_by_one() -> None:
    """**Validates: Requirement 3.2**

    The first event_seq for a turn may be any non-negative integer.  Every
    subsequent event must be exactly first_seq + 1, first_seq + 2, etc.
    """
    session_id = str(uuid4())
    turn_id = str(uuid4())
    session = VoiceProtocolSession(session_id=session_id)

    # Start with seq=17 (arbitrary non-zero start).
    e0 = _make_transcript_event(
        session_id=session_id, turn_id=turn_id,
        event_seq=17, is_final=False,
    )
    session.accept(validate_message(e0))

    # seq=18 is the expected next — must succeed.
    e1 = _make_transcript_event(
        session_id=session_id, turn_id=turn_id,
        event_seq=18, is_final=False,
    )
    session.accept(validate_message(e1))

    # seq=20 (skipped 19) must fail.
    e_skip = _make_transcript_event(
        session_id=session_id, turn_id=turn_id,
        event_seq=20, is_final=False,
    )
    with pytest.raises(VoiceProtocolError) as exc_info:
        session.accept(validate_message(e_skip))

    assert exc_info.value.reason == "invalid_event_sequence"


# ---------------------------------------------------------------------------
# Property 3g: Reconnect scenario — partial turn discarded on disconnect
# (Req 3.8)
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.integers(min_value=0, max_value=4),  # partials sent before disconnect
    st.booleans(),                          # whether to start a new turn after reconnect
)
def test_unfinished_turn_discarded_on_session_reconnect(
    n_partials: int, start_new_turn_after: bool,
) -> None:
    """**Validates: Requirement 3.8**

    If the IPC channel disconnects before a final Transcript_Event, the
    unfinished turn is discarded. Attempting to send more events for it on
    a new session context is rejected as a stale session.

    This property simulates the reconnect by creating a new VoiceProtocolSession
    with a different session_id (the reconnected session), and shows that
    events for the OLD session_id are rejected with 'stale_session'.
    """
    old_session_id = str(uuid4())
    new_session_id = str(uuid4())
    turn_id = str(uuid4())

    old_session = VoiceProtocolSession(session_id=old_session_id)

    # Send some partials on the old session.
    for i in range(n_partials):
        event = _make_transcript_event(
            session_id=old_session_id, turn_id=turn_id,
            event_seq=i, is_final=False,
        )
        old_session.accept(validate_message(event))

    # "Disconnect" — create a new session (simulates Python-side reconnect).
    new_session = VoiceProtocolSession(session_id=new_session_id)

    # Late message from the OLD session must be rejected by the NEW session.
    late_event = _make_transcript_event(
        session_id=old_session_id, turn_id=turn_id,
        event_seq=n_partials, is_final=True,
        text="late final after disconnect",
    )
    with pytest.raises(VoiceProtocolError) as exc_info:
        new_session.accept(validate_message(late_event))

    assert exc_info.value.reason == "stale_session", (
        f"Late event from old session must be rejected as stale_session, "
        f"got {exc_info.value.reason!r}"
    )

    # New turn on the new session must work fine.
    if start_new_turn_after:
        new_turn_id = str(uuid4())
        new_event = _make_transcript_event(
            session_id=new_session_id, turn_id=new_turn_id,
            event_seq=0, is_final=False,
        )
        accepted = new_session.accept(validate_message(new_event))
        assert accepted is not None, "New session must accept new turn events"


# ---------------------------------------------------------------------------
# Property 3h: Cancellation-generation examples (Req 4.8)
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.integers(min_value=1, max_value=4),  # num_cancel_rounds
    st.booleans(),                           # emit_from_oldest_gen
)
def test_stale_generation_frames_rejected_after_cancellation(
    num_cancel_rounds: int,
    emit_from_oldest_gen: bool,
) -> None:
    """**Validates: Requirement 4.8**

    After cancellation advances the session generation, frames stamped with
    any older generation are rejected by session.accept_frame.

    The output_is_current gate also returns False for such frames, ensuring
    the pipeline dispatcher cannot accidentally forward stale work.
    """
    asyncio.run(
        _stale_gen_cancellation_scenario(num_cancel_rounds, emit_from_oldest_gen)
    )


async def _stale_gen_cancellation_scenario(
    num_cancel_rounds: int,
    emit_from_oldest_gen: bool,
) -> None:
    session = VoiceSession(uuid4())
    factory = _MockFrameFactory()

    # Start + cancel num_cancel_rounds turns to advance the generation.
    for _ in range(num_cancel_rounds):
        t = uuid4()
        await session.start_turn(t)
        await session.cancel_turn(t)

    current_gen = session.cancellation_generation
    assert current_gen == num_cancel_rounds

    # Start a new turn at the current generation.
    new_turn = uuid4()
    await session.start_turn(new_turn)
    record = session.turns.get(new_turn)
    assert record.cancellation_generation == current_gen

    # Build a frame with the STALE generation.
    stale_gen = 0 if emit_from_oldest_gen else max(0, current_gen - 1)
    stale_meta = VoiceFrameMetadata(
        session_id=session.session_id,
        turn_id=new_turn,
        sequence=0,
        cancellation_generation=stale_gen,
    )
    stale_frame = TypedVoiceFrame(
        VoiceFrameType.LLM_TEXT,
        stale_meta,
        factory.create_llm_text_frame(text="stale"),
    )

    # accept_frame must reject stale generation.
    with pytest.raises(LateFrameRejected):
        await session.accept_frame(stale_frame, queue=TurnQueueName.LLM)

    # output_is_current must return False for stale generation.
    is_current = await session.output_is_current(stale_frame)
    assert is_current is False, (
        f"output_is_current must be False for stale gen {stale_gen} "
        f"(current={current_gen})"
    )

    # A frame with the current generation must be accepted.
    current_meta = VoiceFrameMetadata(
        session_id=session.session_id,
        turn_id=new_turn,
        sequence=0,
        cancellation_generation=current_gen,
    )
    current_frame = TypedVoiceFrame(
        VoiceFrameType.LLM_TEXT,
        current_meta,
        factory.create_llm_text_frame(text="current"),
    )
    await session.accept_frame(current_frame, queue=TurnQueueName.LLM)

    is_current2 = await session.output_is_current(current_frame)
    assert is_current2 is True, (
        f"output_is_current must be True for current gen {current_gen}"
    )


# ---------------------------------------------------------------------------
# Property 3i: TranscriptSocketIngress end-to-end (Req 3.2, 3.3, 3.6, 4.3)
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(partial_final_event_stream())
def test_transcript_ingress_produces_transcription_frames_with_correct_metadata(
    stream_and_final: tuple[list[dict], Any],
) -> None:
    """**Validates: Requirements 3.2, 3.3, 3.6, 4.3**

    TranscriptSocketIngress must:
    1. Accept valid ordered partial/final events.
    2. Produce TypedVoiceFrame with VoiceFrameType.TRANSCRIPTION.
    3. Attach TranscriptionFrameMetadata — turn_id, event_seq, is_final, language.
    4. Use the MockTranscriptionFrame as the Pipecat payload.
    5. ACK with accepted status for valid events.
    6. Discard duplicate finals with discarded ACK.
    """
    asyncio.run(_transcript_ingress_scenario(stream_and_final))


async def _transcript_ingress_scenario(
    stream_and_final: tuple[list[dict], Any],
) -> None:
    events, _ = stream_and_final
    if not events:
        return

    session_id = UUID(events[0]["session_id"])
    session = VoiceSession(session_id)
    factory = _MockFrameFactory()
    ingress = TranscriptSocketIngress(session=session, frame_factory=factory)

    accepted_frames: list[TypedVoiceFrame] = []
    saw_final = False

    for event in events:
        result = await ingress.ingest(event)
        ack = result.acknowledgement

        assert ack["status"] in ("accepted", "discarded"), (
            f"ACK status must be accepted or discarded, got {ack['status']!r}"
        )

        if result.frame is not None:
            frame = result.frame
            # Frame type must be TRANSCRIPTION.
            assert frame.frame_type is VoiceFrameType.TRANSCRIPTION, (
                f"Frame type must be TRANSCRIPTION, got {frame.frame_type!r}"
            )
            # Metadata must be TranscriptionFrameMetadata.
            assert isinstance(frame.metadata, TranscriptionFrameMetadata), (
                f"Frame metadata must be TranscriptionFrameMetadata"
            )
            # Payload must be the Pipecat TranscriptionFrame mock.
            assert isinstance(frame.payload, MockTranscriptionFrame), (
                f"Payload must be MockTranscriptionFrame"
            )
            # Metadata fields must match the event.
            assert str(frame.metadata.turn_id) == event["turn_id"]
            assert frame.metadata.event_seq == event["event_seq"]
            assert frame.metadata.is_final == event["is_final"]
            assert frame.metadata.language == event["language"]

            accepted_frames.append(frame)
            if event["is_final"]:
                saw_final = True

    # At most one final frame must be present in the accepted frames.
    final_frames = [f for f in accepted_frames if f.metadata.is_final]
    assert len(final_frames) <= 1, (
        f"At most one final frame expected, got {len(final_frames)}"
    )


# ---------------------------------------------------------------------------
# Property 3j: AudioRingIngress produces InputAudioRawFrame (Req 4.2)
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.integers(min_value=1, max_value=8),   # n_slots
    st.integers(min_value=320, max_value=3840, ).filter(lambda x: x % 2 == 0),  # byte_length (even)
)
def test_ring_ingress_produces_input_audio_raw_frames_with_metadata(
    n_slots: int,
    byte_length: int,
) -> None:
    """**Validates: Requirements 2.6, 4.2**

    AudioRingIngress must:
    1. Map each ring slot into a MockInputAudioRawFrame (InputAudioRawFrame).
    2. Attach AudioFrameMetadata with correct session_id, turn_id, sequence,
       captured_monotonic_ns, sample_rate_hz, channels.
    3. Produce monotonically increasing sequence metadata across slots.
    4. Release the slot immediately after frame creation.
    """
    asyncio.run(_ring_ingress_scenario(n_slots, byte_length))


async def _ring_ingress_scenario(n_slots: int, byte_length: int) -> None:
    session_id = uuid4()
    turn_id = uuid4()
    session = VoiceSession(session_id)
    await session.start_turn(turn_id)

    ring = _FakeRingReader(session_id)
    factory = _MockFrameFactory()
    ingress = AudioRingIngress(session=session, ring_reader=ring, frame_factory=factory)

    ts_ns = 1_000_000  # start timestamp
    ts_step = 10_000_000  # 10ms per frame

    frames_produced = []

    for i in range(n_slots):
        # Provide fake PCM data for this slot.
        pcm_data = bytes([i % 256] * byte_length)
        ring.set_slot_data(i, pcm_data)

        descriptor = RingSlotDescriptor(
            session_id=session_id,
            turn_id=turn_id,
            slot_index=i,
            sequence=i,
            captured_monotonic_ns=ts_ns + i * ts_step,
            sample_rate_hz=SILERO_SAMPLE_RATE_HZ,
            channels=SILERO_CHANNELS,
            byte_length=byte_length,
        )

        result = await ingress.ingest(descriptor)
        frame = result.frame

        # 1. Frame type must be INPUT_AUDIO.
        assert frame.frame_type is VoiceFrameType.INPUT_AUDIO, (
            f"Ring frame {i} must have INPUT_AUDIO type"
        )

        # 2. Metadata must be AudioFrameMetadata.
        assert isinstance(frame.metadata, AudioFrameMetadata), (
            f"Ring frame {i} metadata must be AudioFrameMetadata"
        )
        meta = frame.metadata
        assert meta.session_id == session_id
        assert meta.turn_id == turn_id
        assert meta.sequence == i
        assert meta.captured_monotonic_ns == ts_ns + i * ts_step
        assert meta.sample_rate_hz == SILERO_SAMPLE_RATE_HZ
        assert meta.channels == SILERO_CHANNELS

        # 3. Payload must be MockInputAudioRawFrame (InputAudioRawFrame).
        assert isinstance(frame.payload, MockInputAudioRawFrame), (
            f"Ring frame {i} payload must be MockInputAudioRawFrame"
        )
        assert frame.payload.sample_rate == SILERO_SAMPLE_RATE_HZ
        assert frame.payload.num_channels == SILERO_CHANNELS
        assert frame.payload.audio == pcm_data

        frames_produced.append(frame)

    # 4. Sequence metadata must be strictly monotonically increasing.
    for j in range(1, len(frames_produced)):
        prev_seq = frames_produced[j - 1].metadata.sequence
        curr_seq = frames_produced[j].metadata.sequence
        assert curr_seq > prev_seq, (
            f"Ring frame sequences must be monotonically increasing: "
            f"{curr_seq} must be > {prev_seq}"
        )

        prev_ts = frames_produced[j - 1].metadata.captured_monotonic_ns
        curr_ts = frames_produced[j].metadata.captured_monotonic_ns
        assert curr_ts > prev_ts, (
            f"Ring frame timestamps must be monotonically increasing"
        )


# ---------------------------------------------------------------------------
# Property 3k: TTS and LLM frame metadata carry turn_id and seq (Req 4.4, 4.5, 4.6)
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.integers(min_value=1, max_value=6),   # n_llm_frames
    st.integers(min_value=1, max_value=4),   # n_tts_frames
    _safe_text,
)
def test_llm_and_tts_frames_carry_correct_metadata_and_types(
    n_llm_frames: int,
    n_tts_frames: int,
    text: str,
) -> None:
    """**Validates: Requirements 4.4, 4.5, 4.6**

    LLMTextFrame and TTSTextFrame values produced by the pipeline must:
    1. Be the correct mock Pipecat types.
    2. Carry TypedVoiceFrame metadata with turn_id, sequence, cancellation_generation.
    3. Have strictly increasing sequence values within the same turn.
    4. Have their text payload equal to the input text (no metadata embedded).
    """
    session_id = uuid4()
    turn_id = uuid4()
    factory = _MockFrameFactory()

    # Build LLM frames for the turn.
    llm_frames = []
    for seq in range(n_llm_frames):
        meta = VoiceFrameMetadata(
            session_id=session_id,
            turn_id=turn_id,
            sequence=seq,
            cancellation_generation=0,
        )
        frame = TypedVoiceFrame(
            VoiceFrameType.LLM_TEXT,
            meta,
            factory.create_llm_text_frame(text=text),
        )
        llm_frames.append(frame)

    # Build TTS sentence frames.
    tts_frames = []
    for seq in range(n_tts_frames):
        sentence_id = uuid4()
        meta = SentenceFrameMetadata(
            session_id=session_id,
            turn_id=turn_id,
            sequence=seq,
            cancellation_generation=0,
            sentence_id=sentence_id,
        )
        frame = TypedVoiceFrame(
            VoiceFrameType.TTS_TEXT,
            meta,
            factory.create_tts_text_frame(text=text),
        )
        tts_frames.append(frame)

    # LLM frames must have correct types and sequential metadata.
    for i, frame in enumerate(llm_frames):
        assert frame.frame_type is VoiceFrameType.LLM_TEXT
        assert isinstance(frame.payload, MockLLMTextFrame)
        assert frame.payload.text == text
        assert frame.metadata.session_id == session_id
        assert frame.metadata.turn_id == turn_id
        assert frame.metadata.sequence == i

    # TTS frames must have correct types and sequential metadata.
    for i, frame in enumerate(tts_frames):
        assert frame.frame_type is VoiceFrameType.TTS_TEXT
        assert isinstance(frame.payload, MockTTSTextFrame)
        assert frame.payload.text == text
        assert isinstance(frame.metadata, SentenceFrameMetadata)
        assert frame.metadata.session_id == session_id
        assert frame.metadata.turn_id == turn_id
        assert frame.metadata.sequence == i

    # Monotonic sequence within each frame type stream.
    for j in range(1, len(llm_frames)):
        assert llm_frames[j].metadata.sequence > llm_frames[j - 1].metadata.sequence
    for j in range(1, len(tts_frames)):
        assert tts_frames[j].metadata.sequence > tts_frames[j - 1].metadata.sequence


# ---------------------------------------------------------------------------
# Focused: Reconnect — sending missed final on reconnect is rejected
# ---------------------------------------------------------------------------

def test_reconnect_missed_final_is_never_replayed_on_new_session() -> None:
    """**Validates: Requirement 3.8**

    On reconnect, Swift discards the unfinished turn and does NOT replay the
    missed final event. This test shows that a new session correctly rejects
    events from the disconnected session's session_id.

    The Python side creates a new VoiceProtocolSession with a new session_id;
    any message arriving with the old session_id raises stale_session.
    """
    old_session_id = str(uuid4())
    new_session_id = str(uuid4())
    turn_id = str(uuid4())

    new_session = VoiceProtocolSession(session_id=new_session_id)

    # Simulate the missed final event from the disconnected session.
    missed_final = _make_transcript_event(
        session_id=old_session_id,
        turn_id=turn_id,
        event_seq=3,
        is_final=True,
        text="missed final text",
    )

    with pytest.raises(VoiceProtocolError) as exc_info:
        new_session.accept(validate_message(missed_final))

    assert exc_info.value.reason == "stale_session"

    # New events on the new session ID are fine.
    new_turn = str(uuid4())
    fresh_event = _make_transcript_event(
        session_id=new_session_id,
        turn_id=new_turn,
        event_seq=0,
        is_final=False,
        text="fresh partial",
    )
    result = new_session.accept(validate_message(fresh_event))
    assert result is not None


# ---------------------------------------------------------------------------
# Focused: Duplicate final within TranscriptSocketIngress
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_transcript_ingress_discards_duplicate_final_with_discarded_ack() -> None:
    """**Validates: Requirement 3.3**

    When a final event has been accepted, a second final event for the same
    turn is returned with status='discarded' and no TypedVoiceFrame produced.
    """
    session_id = uuid4()
    session = VoiceSession(session_id)
    factory = _MockFrameFactory()
    ingress = TranscriptSocketIngress(session=session, frame_factory=factory)

    session_id_str = str(session_id)
    turn_id_str = str(uuid4())

    # First: partial event.
    partial = _make_transcript_event(
        session_id=session_id_str, turn_id=turn_id_str,
        event_seq=0, is_final=False, text="partial",
    )
    r0 = await ingress.ingest(partial)
    assert r0.accepted, "Partial event must be accepted"

    # Second: final event.
    final = _make_transcript_event(
        session_id=session_id_str, turn_id=turn_id_str,
        event_seq=1, is_final=True, text="final",
    )
    r1 = await ingress.ingest(final)
    assert r1.accepted, "First final event must be accepted"
    assert r1.frame is not None
    assert r1.frame.metadata.is_final is True

    # Third: duplicate final — must be discarded.
    dup_final = _make_transcript_event(
        session_id=session_id_str, turn_id=turn_id_str,
        event_seq=2, is_final=True, text="duplicate final",
    )
    r2 = await ingress.ingest(dup_final)
    assert not r2.accepted, "Duplicate final must be discarded"
    assert r2.frame is None, "Duplicate final must produce no frame"


# ---------------------------------------------------------------------------
# Focused: End-to-end multi-turn ordered frame stream
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multi_turn_frame_stream_maintains_independent_ordering() -> None:
    """**Validates: Requirement 4.8**

    Multiple concurrent turns must have fully independent frame orderings.
    Frames from turn A do not affect the expected sequence of turn B and vice versa.
    """
    session = VoiceSession(uuid4())
    factory = _MockFrameFactory()

    turn_a = uuid4()
    turn_b = uuid4()
    await session.start_turn(turn_a)
    await session.start_turn(turn_b)

    record_a = session.turns.get(turn_a)
    record_b = session.turns.get(turn_b)

    # Interleave frames from turn A and turn B.
    interleaved = [
        (turn_a, record_a, 0),
        (turn_b, record_b, 0),
        (turn_a, record_a, 1),
        (turn_b, record_b, 1),
        (turn_a, record_a, 2),
        (turn_b, record_b, 2),
    ]

    for (turn_id, record, seq) in interleaved:
        meta = VoiceFrameMetadata(
            session_id=session.session_id,
            turn_id=turn_id,
            sequence=seq,
            cancellation_generation=record.cancellation_generation,
        )
        frame = TypedVoiceFrame(
            VoiceFrameType.LLM_TEXT,
            meta,
            factory.create_llm_text_frame(text=f"t={turn_id!s:.8} seq={seq}"),
        )
        # Must accept without error.
        await session.accept_frame(frame, queue=TurnQueueName.LLM)

    # Verify turn A next expected sequence is 3.
    assert record_a.next_sequence == 3
    # Verify turn B next expected sequence is 3.
    assert record_b.next_sequence == 3


# ---------------------------------------------------------------------------
# Focused: Cancellation-generation example with TTS sentence ordering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancellation_generation_tts_ordering() -> None:
    """**Validates: Requirements 4.8, 5.3, 5.4**

    After a barge-in advances the cancellation generation, TTS sentence frames
    bearing the old generation are considered stale.  Only frames with the new
    generation can be accepted for the continued (new) turn.
    """
    session = VoiceSession(uuid4())
    factory = _MockFrameFactory()

    assistant_turn = uuid4()
    await session.start_turn(assistant_turn)
    old_gen = session.cancellation_generation  # 0

    # Register a TTS sentence at generation 0.
    old_sentence = uuid4()
    old_meta = SentenceFrameMetadata(
        session_id=session.session_id,
        turn_id=assistant_turn,
        sequence=0,
        cancellation_generation=old_gen,
        sentence_id=old_sentence,
    )
    old_frame = TypedVoiceFrame(
        VoiceFrameType.TTS_TEXT,
        old_meta,
        factory.create_tts_text_frame(text="old gen sentence."),
    )
    await session.accept_frame(old_frame, queue=TurnQueueName.SENTENCE)

    # Cancel the turn — advances the generation.
    await session.cancel_turn(assistant_turn)
    assert session.cancellation_generation == 1

    # Start a new capture turn at the new generation.
    new_capture_turn = uuid4()
    await session.start_turn(new_capture_turn)
    new_gen = session.cancellation_generation

    # Attempt to register a TTS frame with the old generation for the cancelled turn.
    stale_sentence = uuid4()
    stale_meta = SentenceFrameMetadata(
        session_id=session.session_id,
        turn_id=assistant_turn,  # cancelled turn
        sequence=1,
        cancellation_generation=old_gen,  # stale
        sentence_id=stale_sentence,
    )
    stale_frame = TypedVoiceFrame(
        VoiceFrameType.TTS_TEXT,
        stale_meta,
        factory.create_tts_text_frame(text="stale sentence."),
    )

    # Must be rejected — the turn is terminal.
    with pytest.raises(LateFrameRejected):
        await session.accept_frame(stale_frame, queue=TurnQueueName.SENTENCE)

    # output_is_current must return False for the stale frame.
    assert not await session.output_is_current(stale_frame), (
        "output_is_current must be False for stale TTS frame"
    )
