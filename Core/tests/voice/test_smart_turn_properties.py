"""Property 4: Smart-turn silence rule.

Feature: realtime-local-voice-agent, Property 4: Smart-turn silence rule

For all Silero VAD speech/silence timelines and barge-in states, an active user
utterance is finalized if and only if it has 800 milliseconds of continuous
post-speech silence while no barge-in is active.

**Validates: Requirements 4.7**

Design reference: §5, Property 4; V-TURN-PROP

Covers:
- Clock boundaries (799 ms vs 800 ms silence)
- Recurrent speech (silence resets when speech resumes)
- VAD hysteresis changes (configurable thresholds)
- Active-barge cases (barge-in suppresses finalization)
- At least 100 generated cases via Hypothesis
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from core.voice.frames import AudioFrameMetadata, TypedVoiceFrame, VoiceFrameType
from core.voice.vad import (
    SileroVADSmartTurnProcessor,
    SmartTurnVADConfig,
    VADActivity,
    VADTransitionKind,
)


# ---------------------------------------------------------------------------
# Test data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpeechInput:
    """Minimal payload compatible with the default _payload_probability adapter."""
    speech_probability: float


# VAD silence threshold is 800 ms; speech start threshold is 200 ms.
SILENCE_NS = 800_000_000
SPEECH_START_NS = 200_000_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(
    session_id: UUID,
    turn_id: UUID,
    timestamp_ns: int,
    probability: float,
    sequence: int,
    cancellation_generation: int = 0,
) -> TypedVoiceFrame[object]:
    """Build an InputAudio-typed frame suitable for SileroVADSmartTurnProcessor."""
    return TypedVoiceFrame(
        VoiceFrameType.INPUT_AUDIO,
        AudioFrameMetadata(
            session_id=session_id,
            turn_id=turn_id,
            sequence=sequence,
            cancellation_generation=cancellation_generation,
            captured_monotonic_ns=timestamp_ns,
            sample_rate_hz=16_000,
            channels=1,
        ),
        SpeechInput(probability),
    )


def _feed_timeline(
    processor: SileroVADSmartTurnProcessor,
    turn_id: UUID,
    session_id: UUID,
    events: list[tuple[int, float]],
    *,
    playback_active: bool = False,
) -> list[VADTransitionKind]:
    """Feed a list of (timestamp_ns, probability) frames and collect all transition kinds."""
    transitions: list[VADTransitionKind] = []
    for seq, (ts_ns, prob) in enumerate(events):
        frame = _make_frame(session_id, turn_id, ts_ns, prob, sequence=seq)
        result = processor.process(frame, playback_active=playback_active)
        transitions.extend(t.kind for t in result)
    return transitions


def _finalized(transitions: list[VADTransitionKind]) -> bool:
    return VADTransitionKind.SMART_TURN_FINALIZED in transitions


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# A single speech segment: voiced start + voiced duration + silence duration.
# Timestamps are generated in nanoseconds.

@st.composite
def speech_then_silence_timeline(draw: st.DrawFn) -> tuple[list[tuple[int, float]], int]:
    """
    Generate a timeline of (timestamp_ns, probability) pairs representing:
    - A period of silence before speech starts (0 to 500 ms)
    - Speech accumulation (between 200 ms and 2000 ms)
    - A post-speech silence period (0 to 2000 ms)

    Returns (events, silence_duration_ns).

    The silence duration is measured as the delta from the first silence
    frame to the last silence frame, which is what the processor actually
    computes (now_ns - silence_since_ns >= threshold).
    """
    pre_silence_ms = draw(st.integers(min_value=0, max_value=500))
    voiced_ms = draw(st.integers(min_value=200, max_value=2000))
    post_silence_ms = draw(st.integers(min_value=0, max_value=2000))
    step_ms = 10  # 10 ms frames

    events: list[tuple[int, float]] = []
    t_ms = 0

    # Pre-speech silence
    for _ in range(pre_silence_ms // step_ms):
        events.append((t_ms * 1_000_000, 0.0))
        t_ms += step_ms

    # Voiced speech: use voiced_ms+10ms frames to ensure we CROSS the threshold
    # (voiced_ms frames only reach voiced_ms - step_ms timestamp, not voiced_ms itself)
    for _ in range(voiced_ms // step_ms + 1):
        events.append((t_ms * 1_000_000, 0.8))
        t_ms += step_ms

    # Post-speech silence: the silence_since_ns is set when the first 0.0 frame arrives
    # The check is: now_ns - silence_since_ns >= threshold
    # With n frames at step_ms intervals: delta = (n-1)*step_ms
    # To get delta >= post_silence_ms: need post_silence_ms // step_ms + 1 frames
    silence_start_ms = t_ms
    silence_frames = post_silence_ms // step_ms + 1
    for _ in range(silence_frames):
        events.append((t_ms * 1_000_000, 0.0))
        t_ms += step_ms

    # Actual delta from first to last silence frame = (silence_frames - 1) * step_ms
    actual_silence_ns = (silence_frames - 1) * step_ms * 1_000_000

    return events, actual_silence_ns


@st.composite
def recurrent_speech_timeline(draw: st.DrawFn) -> tuple[list[tuple[int, float]], bool]:
    """
    Generate a timeline with a speech period, a partial silence (< 800 ms),
    then speech resumes, then a final silence.
    Returns (events, should_finalize).

    The processor sets silence_since_ns at the first 0.0 frame after speech
    and checks: now_ns - silence_since_ns >= threshold.
    With 10ms frames, final_silence_ms frames span (n-1)*10ms from first to last.
    We need at least (threshold_ms // step_ms + 1) frames to cross 800ms.
    """
    voiced_ms_1 = draw(st.integers(min_value=200, max_value=800))
    partial_silence_ms = draw(st.integers(min_value=10, max_value=790))
    voiced_ms_2 = draw(st.integers(min_value=200, max_value=500))
    # Use frame counts directly: n frames span (n-1)*10ms delta
    # To cross 800ms: need at least 81 frames → 800ms delta
    final_silence_frames = draw(st.integers(min_value=0, max_value=150))
    step_ms = 10

    events: list[tuple[int, float]] = []
    t_ms = 0

    # voiced_ms_1+10ms frames to cross speech_start_ms=200ms threshold
    for _ in range(voiced_ms_1 // step_ms + 1):
        events.append((t_ms * 1_000_000, 0.9))
        t_ms += step_ms

    for _ in range(partial_silence_ms // step_ms):
        events.append((t_ms * 1_000_000, 0.0))
        t_ms += step_ms

    # voiced_ms_2+10ms frames to resume speech state
    for _ in range(voiced_ms_2 // step_ms + 1):
        events.append((t_ms * 1_000_000, 0.9))
        t_ms += step_ms

    silence_start_ms = t_ms
    for _ in range(final_silence_frames):
        events.append((t_ms * 1_000_000, 0.0))
        t_ms += step_ms

    # Actual delta from first to last silence frame
    if final_silence_frames >= 2:
        actual_delta_ms = (final_silence_frames - 1) * step_ms
    elif final_silence_frames == 1:
        actual_delta_ms = 0
    else:
        actual_delta_ms = 0

    should_finalize = actual_delta_ms >= 800
    return events, should_finalize


@st.composite
def hysteresis_config(draw: st.DrawFn) -> SmartTurnVADConfig:
    """Generate valid hysteresis configurations with varying thresholds."""
    release = draw(st.floats(min_value=0.10, max_value=0.45))
    speech = draw(st.floats(min_value=release + 0.05, max_value=0.99))
    silence_ms = draw(st.integers(min_value=400, max_value=1200))
    speech_start_ms = draw(st.integers(min_value=100, max_value=500))
    return SmartTurnVADConfig(
        speech_probability_threshold=round(speech, 3),
        release_probability_threshold=round(release, 3),
        post_speech_silence_ms=silence_ms,
        speech_start_ms=speech_start_ms,
    )


# ---------------------------------------------------------------------------
# Property 4 — core finalization rule (iff)
# ---------------------------------------------------------------------------

@settings(max_examples=120, deadline=None)
@given(speech_then_silence_timeline())
def test_smart_turn_finalizes_iff_silence_reaches_800ms(
    data: tuple[list[tuple[int, float]], int],
) -> None:
    """
    **Validates: Requirements 4.7**

    An active user utterance is finalized if and only if it has >=800 ms of
    continuous post-speech silence while no barge-in is active.
    """
    events, silence_ns = data
    # Only test timelines with actual speech (voiced frames above threshold)
    has_speech = any(prob >= 0.8 for _, prob in events)
    assume(has_speech)

    session_id = uuid4()
    turn_id = uuid4()
    processor = SileroVADSmartTurnProcessor(config=SmartTurnVADConfig())
    transitions = _feed_timeline(processor, turn_id, session_id, events, playback_active=False)

    if silence_ns >= SILENCE_NS:
        assert _finalized(transitions), (
            f"Expected SMART_TURN_FINALIZED with {silence_ns / 1_000_000:.0f} ms silence, "
            f"but transitions were: {transitions}"
        )
    else:
        assert not _finalized(transitions), (
            f"Unexpected SMART_TURN_FINALIZED with only {silence_ns / 1_000_000:.0f} ms silence "
            f"(threshold is 800 ms). Transitions: {transitions}"
        )


# ---------------------------------------------------------------------------
# Clock boundary cases
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(st.integers(min_value=0, max_value=79))
def test_boundary_799ms_never_finalizes(frame_count: int) -> None:
    """
    **Validates: Requirements 4.7**

    Silence frames that produce a delta of at most 790ms (first-to-last) must
    never finalize the turn. Since finalization requires now_ns - silence_since_ns >= 800ms,
    and silence_since_ns is the FIRST 0.0 frame timestamp, we need: delta < 800ms.

    With 10ms steps: n frames → delta = (n-1)*10ms. For n <= 80: delta <= 790ms < 800ms.
    """
    session_id = uuid4()
    turn_id = uuid4()
    processor = SileroVADSmartTurnProcessor(config=SmartTurnVADConfig())

    step_ns = 10_000_000  # 10ms
    events: list[tuple[int, float]] = []
    t_ns = 0

    # Speech phase: 300 ms at 0.9 (30 frames)
    for _ in range(30):
        events.append((t_ns, 0.9))
        t_ns += step_ns

    # Silence phase: frame_count + 1 frames (delta = frame_count * 10ms, max 790ms)
    for _ in range(frame_count + 1):
        events.append((t_ns, 0.0))
        t_ns += step_ns

    transitions = _feed_timeline(processor, turn_id, session_id, events, playback_active=False)
    delta_ms = frame_count * 10
    assert not _finalized(transitions), (
        f"Silence delta of {delta_ms}ms (< 800ms) must not finalize. "
        f"transitions={transitions}"
    )


def test_boundary_exactly_800ms_finalizes() -> None:
    """
    **Validates: Requirements 4.7**

    Silence that crosses exactly 800 ms (>=800_000_000 ns) must finalize the turn.

    The processor sets silence_since_ns at the first 0.0 frame, and checks:
        now_ns - silence_since_ns >= 800_000_000
    So we need a frame at silence_start_ns + 800_000_000 to trigger finalization.
    With 10ms frames, 81 frames give a delta of 80*10ms = 800ms from first to last.
    """
    session_id = uuid4()
    turn_id = uuid4()
    processor = SileroVADSmartTurnProcessor(config=SmartTurnVADConfig())

    step_ns = 10_000_000  # 10ms
    events: list[tuple[int, float]] = []
    t_ns = 0

    # Speech phase: 21 frames × 10ms; last frame at t=200ms crosses speech_start threshold
    for _ in range(21):
        events.append((t_ns, 0.9))
        t_ns += step_ns

    silence_start_ns = t_ns
    # Silence phase: 81 frames; delta from first (silence_start_ns) to last = 80*10ms = 800ms
    for _ in range(81):
        events.append((t_ns, 0.0))
        t_ns += step_ns

    transitions = _feed_timeline(processor, turn_id, session_id, events, playback_active=False)
    assert _finalized(transitions), (
        f"Exactly 800ms silence delta must finalize. Silence started at {silence_start_ns}ns. "
        f"Last frame at {t_ns - step_ns}ns, "
        f"delta={(t_ns - step_ns - silence_start_ns) // 1_000_000}ms. "
        f"Transitions: {transitions}"
    )


# ---------------------------------------------------------------------------
# Recurrent speech — silence resets when speech resumes
# ---------------------------------------------------------------------------

@settings(max_examples=120, deadline=None)
@given(recurrent_speech_timeline())
def test_recurrent_speech_resets_silence_window(
    data: tuple[list[tuple[int, float]], bool],
) -> None:
    """
    **Validates: Requirements 4.7**

    When speech resumes during a silence window, the 800 ms timer resets.
    Finalization is only triggered by 800 ms of silence AFTER the most recent
    speech ends.
    """
    events, should_finalize = data
    has_speech = any(prob >= 0.8 for _, prob in events)
    assume(has_speech)

    session_id = uuid4()
    turn_id = uuid4()
    processor = SileroVADSmartTurnProcessor(config=SmartTurnVADConfig())
    transitions = _feed_timeline(processor, turn_id, session_id, events, playback_active=False)

    if should_finalize:
        assert _finalized(transitions), (
            f"Expected finalization after 800ms silence following recurrent speech. "
            f"Transitions: {transitions}"
        )
    else:
        assert not _finalized(transitions), (
            f"Should not finalize: final silence was < 800ms after speech resumed. "
            f"Transitions: {transitions}"
        )


def test_speech_resumption_resets_silence_counter_deterministically() -> None:
    """
    **Validates: Requirements 4.7**

    Deterministic regression: 400ms silence → speech resumes → 400ms silence must
    NOT finalize; only 800ms of uninterrupted silence after speech ends finalizes.

    Key: the silence_since_ns is reset when speech resumes, so the 800ms must
    accumulate from the LAST speech end. With 10ms frames, 81 frames give 800ms delta.
    """
    session_id = uuid4()
    turn_id = uuid4()
    processor = SileroVADSmartTurnProcessor(config=SmartTurnVADConfig())

    events: list[tuple[int, float]] = []
    step_ns = 10_000_000  # 10 ms
    t_ns = 0

    # 21 frames of speech at 0.9 — last frame at t=200ms crosses speech_start threshold
    for _ in range(21):
        events.append((t_ns, 0.9))
        t_ns += step_ns

    # 40 frames of silence: delta = 39*10ms = 390ms (< 800ms — should not finalize)
    for _ in range(40):
        events.append((t_ns, 0.0))
        t_ns += step_ns

    # Speech resumes for 21 frames (enough to keep USER_SPEAKING)
    for _ in range(21):
        events.append((t_ns, 0.9))
        t_ns += step_ns

    # 40 frames of silence again: delta = 39*10ms = 390ms (resets counter — must not finalize yet)
    for _ in range(40):
        events.append((t_ns, 0.0))
        t_ns += step_ns

    transitions = _feed_timeline(processor, turn_id, session_id, events, playback_active=False)
    assert not _finalized(transitions), (
        "Silence after speech resumption must restart the 800ms counter. "
        "390ms before + 390ms after speech resumption must not finalize."
    )

    # Now add 42 more silence frames to cross 800ms (total 82 frames from last speech end,
    # delta = 81*10ms = 810ms ≥ 800ms)
    extended_events = list(events)
    for _ in range(42):
        extended_events.append((t_ns, 0.0))
        t_ns += step_ns

    session_id2 = uuid4()
    turn_id2 = uuid4()
    processor2 = SileroVADSmartTurnProcessor(config=SmartTurnVADConfig())
    transitions2 = _feed_timeline(processor2, turn_id2, session_id2, extended_events, playback_active=False)
    assert _finalized(transitions2), (
        "800ms of uninterrupted silence after the last speech should finalize."
    )


# ---------------------------------------------------------------------------
# VAD hysteresis changes
# ---------------------------------------------------------------------------

@settings(max_examples=120, deadline=None)
@given(hysteresis_config())
def test_custom_hysteresis_silence_threshold_respected(
    config: SmartTurnVADConfig,
) -> None:
    """
    **Validates: Requirements 4.7**

    The finalization condition uses the configured silence threshold, not a
    hard-coded 800ms constant. A timeline with silence delta >= config.post_speech_silence_ms
    must finalize; one with less must not.

    With 10ms frames, n frames give delta = (n-1)*10ms from first to last frame.
    To get delta >= threshold_ms: need (threshold_ms//10 + 1 + 1) frames.
    To get delta < threshold_ms: need at most threshold_ms//10 frames.
    """
    session_id = uuid4()
    step_ns = 10_000_000  # 10ms

    # Build a voiced phase long enough to cross speech_start_ms
    # Frame index speech_start_ms//10 is at speech_start_ms - (speech_start_ms % 10) ms
    # Need speech_start_ms//10 + 2 frames to reliably cross the threshold
    voiced_frames = config.speech_start_ms // 10 + 2
    threshold_ms = config.post_speech_silence_ms

    # Timeline with enough silence to finalize: delta = threshold_ms exactly
    # need (threshold_ms//10 + 1) frames for delta = threshold_ms
    silence_frames_enough = threshold_ms // 10 + 2  # gives delta >= threshold_ms

    turn_id_yes = uuid4()
    events_enough: list[tuple[int, float]] = []
    t_ns = 0
    for _ in range(voiced_frames):
        events_enough.append((t_ns, config.speech_probability_threshold))
        t_ns += step_ns
    for _ in range(silence_frames_enough):
        events_enough.append((t_ns, 0.0))
        t_ns += step_ns

    # Timeline with silence strictly shorter: delta = (threshold_ms - 10ms)
    # need threshold_ms//10 frames for delta = (threshold_ms//10 - 1)*10ms < threshold_ms
    silence_frames_short = max(1, threshold_ms // 10 - 1)

    turn_id_no = uuid4()
    events_short: list[tuple[int, float]] = []
    t_ns2 = 0
    for _ in range(voiced_frames):
        events_short.append((t_ns2, config.speech_probability_threshold))
        t_ns2 += step_ns
    for _ in range(silence_frames_short):
        events_short.append((t_ns2, 0.0))
        t_ns2 += step_ns

    proc_yes = SileroVADSmartTurnProcessor(config=config)
    proc_no = SileroVADSmartTurnProcessor(config=config)

    transitions_yes = _feed_timeline(proc_yes, turn_id_yes, session_id, events_enough, playback_active=False)
    transitions_no = _feed_timeline(proc_no, turn_id_no, session_id, events_short, playback_active=False)

    delta_enough_ms = (silence_frames_enough - 1) * 10
    delta_short_ms = (silence_frames_short - 1) * 10

    assert _finalized(transitions_yes), (
        f"Expected finalization with {delta_enough_ms}ms silence delta "
        f"(threshold={threshold_ms}ms). Transitions: {transitions_yes}"
    )
    assert not _finalized(transitions_no), (
        f"Did not expect finalization with {delta_short_ms}ms silence delta "
        f"(threshold={threshold_ms}ms). Transitions: {transitions_no}"
    )


@settings(max_examples=100, deadline=None)
@given(
    st.floats(min_value=0.05, max_value=0.35),
    st.floats(min_value=0.61, max_value=0.99),
)
def test_hysteresis_mid_range_probability_follows_prior_state(
    release: float, speech: float
) -> None:
    """
    **Validates: Requirements 4.7**

    Probabilities between release_threshold and speech_threshold must retain
    the prior voiced/unvoiced state (hysteresis), not artificially trigger
    speech or silence transitions.

    Timeline: 300ms above speech threshold → 40 frames at mid-range (hysteresis keeps
    voiced=True) → 900ms below release threshold (unvoiced, starts silence counter).
    With 90 frames of 0.0, delta from first to last = 89*10ms = 890ms >= 800ms.
    """
    assume(release < speech - 0.05)
    config = SmartTurnVADConfig(
        speech_probability_threshold=round(speech, 3),
        release_probability_threshold=round(release, 3),
    )
    mid = round((release + speech) / 2.0, 3)
    speech_prob = min(0.999, round(speech + 0.01, 3))

    session_id = uuid4()
    turn_id = uuid4()
    processor = SileroVADSmartTurnProcessor(config=config)
    step_ns = 10_000_000

    events: list[tuple[int, float]] = []
    t_ns = 0

    # Speech onset: enough frames to cross the default speech_start_ms=200ms threshold
    # (default config is used - speech_start_ms=200, so 21 frames needed)
    for _ in range(30):
        events.append((t_ns, speech_prob))
        t_ns += step_ns

    # Mid-range: should remain voiced (hysteresis holds speech active)
    for _ in range(40):
        events.append((t_ns, mid))
        t_ns += step_ns

    # Now drop below release: start silence counter
    # 90 frames: delta from first to last = 89*10ms = 890ms >= 800ms
    for _ in range(90):
        events.append((t_ns, 0.0))
        t_ns += step_ns

    transitions = _feed_timeline(processor, turn_id, session_id, events, playback_active=False)
    # Should finalize since we got 890ms silence delta after dropping below release
    assert _finalized(transitions), (
        f"Expected finalization after mid-range hysteresis then 890ms silence. "
        f"config: release={config.release_probability_threshold:.3f}, "
        f"speech={config.speech_probability_threshold:.3f}, mid={mid:.3f}. "
        f"Transitions: {transitions}"
    )


# ---------------------------------------------------------------------------
# Active-barge cases — barge-in suppresses finalization
# ---------------------------------------------------------------------------

@settings(max_examples=120, deadline=None)
@given(
    st.integers(min_value=800, max_value=3000),  # silence_ms — well above 800ms
)
def test_barge_in_active_suppresses_finalization(silence_ms: int) -> None:
    """
    **Validates: Requirements 4.7**

    When barge-in is active, the smart-turn silence rule must NOT finalize
    the turn, even if silence exceeds 800 ms.
    """
    session_id = uuid4()
    turn_id = uuid4()
    processor = SileroVADSmartTurnProcessor(config=SmartTurnVADConfig())

    step_ns = 10_000_000
    events: list[tuple[int, float]] = []
    t_ns = 0

    # 21 frames of speech — last frame at 200ms crosses speech_start threshold
    for _ in range(21):
        events.append((t_ns, 0.9))
        t_ns += step_ns

    # Post-speech silence >= 800ms: use frame count to ensure delta >= silence_ms
    # silence_ms frames spaced 10ms → delta = (silence_ms - 1) * 10ms... we just need
    # (silence_ms // 10 + 1) frames for delta = silence_ms ms
    for _ in range(silence_ms // 10 + 2):
        events.append((t_ns, 0.0))
        t_ns += step_ns

    # Mark barge-in active BEFORE feeding frames
    processor.set_barge_in_active(turn_id, True)

    transitions = _feed_timeline(processor, turn_id, session_id, events, playback_active=False)
    assert not _finalized(transitions), (
        f"Barge-in must suppress finalization even with {silence_ms}ms silence. "
        f"Transitions: {transitions}"
    )


def test_barge_in_inactive_allows_finalization_and_active_suppresses() -> None:
    """
    **Validates: Requirements 4.7**

    Using the same timeline: without barge-in it finalizes; with barge-in it doesn't.
    This directly tests the iff condition for barge-in state.
    """
    session_id = uuid4()
    turn_id_no_barge = uuid4()
    turn_id_with_barge = uuid4()

    step_ns = 10_000_000
    events: list[tuple[int, float]] = []
    t_ns = 0

    # 21 voiced frames — last frame at 200ms crosses speech_start threshold
    for _ in range(21):
        events.append((t_ns, 0.9))
        t_ns += step_ns

    # 82 silence frames — delta = 81*10ms = 810ms > 800ms
    for _ in range(82):
        events.append((t_ns, 0.0))
        t_ns += step_ns

    # Without barge-in: should finalize
    proc_no_barge = SileroVADSmartTurnProcessor(config=SmartTurnVADConfig())
    transitions_no_barge = _feed_timeline(
        proc_no_barge, turn_id_no_barge, session_id, events, playback_active=False
    )
    assert _finalized(transitions_no_barge), (
        "Expected SMART_TURN_FINALIZED when barge-in is inactive with 1000ms silence."
    )

    # With barge-in: must NOT finalize
    proc_with_barge = SileroVADSmartTurnProcessor(config=SmartTurnVADConfig())
    proc_with_barge.set_barge_in_active(turn_id_with_barge, True)
    transitions_with_barge = _feed_timeline(
        proc_with_barge, turn_id_with_barge, session_id, events, playback_active=False
    )
    assert not _finalized(transitions_with_barge), (
        "SMART_TURN_FINALIZED must be suppressed when barge-in is active."
    )


@settings(max_examples=100, deadline=None)
@given(
    st.integers(min_value=0, max_value=700),   # silence before barge cleared (ms)
    st.integers(min_value=82, max_value=150),  # silence frames after barge cleared (>= 81 → delta >= 800ms)
)
def test_finalization_after_barge_in_cleared(
    pre_clear_silence_ms: int, post_clear_frames: int
) -> None:
    """
    **Validates: Requirements 4.7**

    After barge-in is cleared, the 800ms silence rule applies again.
    This tests that deactivating barge-in re-enables the finalization logic.

    post_clear_frames frames → delta = (post_clear_frames - 1) * 10ms
    For post_clear_frames >= 82: delta >= 810ms > 800ms.
    """
    session_id = uuid4()
    turn_id = uuid4()
    processor = SileroVADSmartTurnProcessor(config=SmartTurnVADConfig())

    step_ns = 10_000_000
    events_phase1: list[tuple[int, float]] = []
    t_ns = 0

    # 21 frames of speech — last frame at 200ms crosses speech_start threshold
    for _ in range(21):
        events_phase1.append((t_ns, 0.9))
        t_ns += step_ns

    # Silence while barge-in is active
    for _ in range(max(pre_clear_silence_ms, 10) // 10):
        events_phase1.append((t_ns, 0.0))
        t_ns += step_ns

    # Feed phase 1 with barge-in active
    processor.set_barge_in_active(turn_id, True)
    transitions_1 = _feed_timeline(processor, turn_id, session_id, events_phase1, playback_active=False)
    assert not _finalized(transitions_1), (
        "Must not finalize while barge-in is active."
    )

    # Clear barge-in
    processor.set_barge_in_active(turn_id, False)

    # More silence after barge-in is cleared
    # post_clear_frames frames → delta = (post_clear_frames - 1) * 10ms >= 810ms
    events_phase2: list[tuple[int, float]] = []
    base_ns = t_ns
    for i in range(post_clear_frames):
        events_phase2.append((base_ns + i * step_ns, 0.0))

    transitions_2 = _feed_timeline(processor, turn_id, session_id, events_phase2, playback_active=False)
    delta_ms = (post_clear_frames - 1) * 10
    assert _finalized(transitions_2), (
        f"Expected finalization after barge-in cleared with {delta_ms}ms silence delta. "
        f"Transitions: {transitions_2}"
    )


# ---------------------------------------------------------------------------
# No speech — never finalize without prior voiced activity
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(st.integers(min_value=800, max_value=5000))
def test_pure_silence_without_speech_never_finalizes(silence_ms: int) -> None:
    """
    **Validates: Requirements 4.7**

    Silence alone (without prior speech activity crossing the 200ms start
    threshold) must never trigger finalization. The state machine starts in
    LISTENING and requires USER_SPEAKING before the silence counter can run.
    """
    session_id = uuid4()
    turn_id = uuid4()
    processor = SileroVADSmartTurnProcessor(config=SmartTurnVADConfig())

    step_ms = 10
    events: list[tuple[int, float]] = []
    for i in range(silence_ms // step_ms):
        events.append((i * step_ms * 1_000_000, 0.0))

    transitions = _feed_timeline(processor, turn_id, session_id, events, playback_active=False)
    assert not _finalized(transitions), (
        f"Silence without prior speech must never finalize. silence_ms={silence_ms}, "
        f"transitions={transitions}"
    )


# ---------------------------------------------------------------------------
# Finalization is one-shot — processor rejects frames after FINALIZED
# ---------------------------------------------------------------------------

def test_finalization_is_one_shot_and_processor_ignores_subsequent_frames() -> None:
    """
    **Validates: Requirements 4.7**

    Once SMART_TURN_FINALIZED is emitted, the processor emits no further
    transitions for the same turn.
    """
    session_id = uuid4()
    turn_id = uuid4()
    processor = SileroVADSmartTurnProcessor(config=SmartTurnVADConfig())

    step_ns = 10_000_000
    t_ns = 0

    # Bring to USER_SPEAKING
    for seq in range(30):
        frame = _make_frame(session_id, turn_id, t_ns, 0.9, sequence=seq)
        processor.process(frame, playback_active=False)
        t_ns += step_ns

    # Cross the 800ms silence boundary
    finalize_transitions: list[VADTransitionKind] = []
    for seq in range(30, 120):
        frame = _make_frame(session_id, turn_id, t_ns, 0.0, sequence=seq)
        result = processor.process(frame, playback_active=False)
        finalize_transitions.extend(t.kind for t in result)
        t_ns += step_ns

    assert VADTransitionKind.SMART_TURN_FINALIZED in finalize_transitions

    # Additional frames after finalization must produce no transitions
    post_finalize: list[VADTransitionKind] = []
    for seq in range(120, 140):
        frame = _make_frame(session_id, turn_id, t_ns, 0.9, sequence=seq)
        result = processor.process(frame, playback_active=False)
        post_finalize.extend(t.kind for t in result)
        t_ns += step_ns

    assert post_finalize == [], (
        f"After finalization, no transitions should be emitted. Got: {post_finalize}"
    )


# ---------------------------------------------------------------------------
# Multi-turn isolation — each turn has independent state
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(st.integers(min_value=2, max_value=5))
def test_independent_turns_have_isolated_silence_state(num_turns: int) -> None:
    """
    **Validates: Requirements 4.7**

    Multiple concurrent turn IDs fed to the same processor must have
    fully independent silence counters and finalization states.
    """
    session_id = uuid4()
    processor = SileroVADSmartTurnProcessor(config=SmartTurnVADConfig())

    turn_ids = [uuid4() for _ in range(num_turns)]
    finalized_count = 0

    step_ms = 10

    for turn_id in turn_ids:
        events: list[tuple[int, float]] = []
        t_ms = 0
        # 300ms speech
        for _ in range(30):
            events.append((t_ms * 1_000_000, 0.9))
            t_ms += step_ms
        # 900ms silence — enough to finalize
        for _ in range(90):
            events.append((t_ms * 1_000_000, 0.0))
            t_ms += step_ms

        transitions = _feed_timeline(processor, turn_id, session_id, events, playback_active=False)
        if _finalized(transitions):
            finalized_count += 1

    assert finalized_count == num_turns, (
        f"All {num_turns} independent turns must finalize independently. "
        f"Only {finalized_count} finalized."
    )
