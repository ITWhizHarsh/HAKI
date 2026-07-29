"""Smart-turn VAD, turn joining, and immediate barge-in coordination.

The module deliberately keeps Silero inference behind a small probability port. It
owns the deterministic timing/state rules while a production adapter provides
probabilities from the local Silero model. This makes timing and cancellation
behaviour independently testable without recording microphone audio.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable
from uuid import UUID

from .frames import AudioFrameMetadata, TranscriptionFrameMetadata, TypedVoiceFrame
from .interfaces import VoiceTurnRequest
from .session import InvalidTurnTransition, TurnState, UnknownTurnError, VoiceSession


SILERO_SAMPLE_RATE_HZ = 16_000
SILERO_CHANNELS = 1


class VADConfigurationError(ValueError):
    """The local VAD timing or hysteresis configuration is invalid."""


class VADFrameRejected(ValueError):
    """An input frame is not a 16 kHz mono Pipecat audio frame."""


class VADActivity(str, Enum):
    """The stable speech state for one user turn."""

    LISTENING = "listening"
    USER_SPEAKING = "user_speaking"
    FINALIZED = "finalized"


class VADTransitionKind(str, Enum):
    """Internal, content-free state transitions emitted from audio frames."""

    SPEECH_STARTED = "speech_started"
    SPEECH_RESUMED = "speech_resumed"
    SMART_TURN_FINALIZED = "smart_turn_finalized"
    BARGE_IN_THRESHOLD = "barge_in_threshold"


@dataclass(frozen=True, slots=True)
class SmartTurnVADConfig:
    """Configurable Silero hysteresis and continuous-duration thresholds."""

    speech_probability_threshold: float = 0.60
    release_probability_threshold: float = 0.35
    speech_start_ms: int = 200
    post_speech_silence_ms: int = 800

    def __post_init__(self) -> None:
        for name, value in (
            ("speech_probability_threshold", self.speech_probability_threshold),
            ("release_probability_threshold", self.release_probability_threshold),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
                raise VADConfigurationError(f"{name} must be between zero and one")
        if self.release_probability_threshold >= self.speech_probability_threshold:
            raise VADConfigurationError("release probability must be below speech probability")
        for name, value in (("speech_start_ms", self.speech_start_ms), ("post_speech_silence_ms", self.post_speech_silence_ms)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise VADConfigurationError(f"{name} must be a positive integer")

    @property
    def speech_start_ns(self) -> int:
        return self.speech_start_ms * 1_000_000

    @property
    def post_speech_silence_ns(self) -> int:
        return self.post_speech_silence_ms * 1_000_000


@dataclass(frozen=True, slots=True)
class VADTransition:
    """A turn-correlated VAD state transition with a monotonic timestamp."""

    turn_id: UUID
    kind: VADTransitionKind
    occurred_monotonic_ns: int
    cancellation_generation: int


@dataclass(slots=True)
class _TurnVADState:
    activity: VADActivity = VADActivity.LISTENING
    voiced: bool = False
    voiced_since_ns: int | None = None
    silence_since_ns: int | None = None
    last_captured_monotonic_ns: int | None = None
    barge_in_active: bool = False
    barge_threshold_reported: bool = False


@runtime_checkable
class VoiceActivityProbabilityProvider(Protocol):
    """Produces a local Silero probability for one Pipecat raw-audio payload."""

    def probability(self, input_audio_frame: object) -> float:
        """Return a finite speech probability in the inclusive range [0, 1]."""


ProbabilityProvider = VoiceActivityProbabilityProvider | Callable[[object], float]


class SileroVADSmartTurnProcessor:
    """Apply Silero hysteresis and smart-turn timing to 16 kHz mono frames.

    It emits no transcript and never starts the LLM.  A final transcript and a
    ``SMART_TURN_FINALIZED`` transition must later be joined by
    :class:`TurnJoinProcessor` before a user turn becomes eligible.
    """

    def __init__(
        self,
        *,
        probability_provider: ProbabilityProvider | None = None,
        config: SmartTurnVADConfig | None = None,
    ) -> None:
        self.config = config or SmartTurnVADConfig()
        self._probability_provider = probability_provider or _payload_probability
        self._turns: dict[UUID, _TurnVADState] = {}

    def set_barge_in_active(self, turn_id: UUID, active: bool) -> None:
        """Suppress smart-turn finalization only while the interruption is active."""
        state = self._turns.setdefault(turn_id, _TurnVADState())
        state.barge_in_active = active

    def reset_turn(self, turn_id: UUID) -> None:
        """Discard only local timing state after a terminal turn."""
        self._turns.pop(turn_id, None)

    def process(
        self,
        frame: TypedVoiceFrame[object],
        *,
        playback_active: bool,
    ) -> tuple[VADTransition, ...]:
        """Consume one mandatory ``InputAudioRawFrame`` wrapper.

        Hysteresis treats values between release and speech thresholds as the
        prior state.  Any released frame resets the continuous voiced window;
        any resumed speech resets the post-speech silence window.
        """
        metadata = frame.metadata
        if not isinstance(metadata, AudioFrameMetadata):
            raise VADFrameRejected("vad_audio_metadata_required")
        if metadata.sample_rate_hz != SILERO_SAMPLE_RATE_HZ or metadata.channels != SILERO_CHANNELS:
            raise VADFrameRejected("vad_requires_16khz_mono_input")
        probability = self._probability(frame.payload)
        now_ns = metadata.captured_monotonic_ns
        state = self._turns.setdefault(metadata.turn_id, _TurnVADState())
        if (
            state.last_captured_monotonic_ns is not None
            and now_ns < state.last_captured_monotonic_ns
        ):
            raise VADFrameRejected("vad_timestamp_not_monotonic")
        state.last_captured_monotonic_ns = now_ns
        if state.activity is VADActivity.FINALIZED:
            return ()

        is_voiced = self._apply_hysteresis(state, probability)
        transitions: list[VADTransition] = []
        if is_voiced:
            resumed = state.voiced is False and state.silence_since_ns is not None
            if not state.voiced:
                state.voiced = True
                state.voiced_since_ns = now_ns
                state.silence_since_ns = None
                state.barge_threshold_reported = False
                if state.activity is VADActivity.USER_SPEAKING and resumed:
                    transitions.append(self._transition(metadata, VADTransitionKind.SPEECH_RESUMED, now_ns))
            if (
                state.activity is VADActivity.LISTENING
                and state.voiced_since_ns is not None
                and now_ns - state.voiced_since_ns >= self.config.speech_start_ns
            ):
                state.activity = VADActivity.USER_SPEAKING
                transitions.append(self._transition(metadata, VADTransitionKind.SPEECH_STARTED, now_ns))
            if (
                state.activity is VADActivity.USER_SPEAKING
                and playback_active
                and not state.barge_in_active
                and not state.barge_threshold_reported
                and state.voiced_since_ns is not None
                and now_ns - state.voiced_since_ns >= self.config.speech_start_ns
            ):
                state.barge_threshold_reported = True
                transitions.append(self._transition(metadata, VADTransitionKind.BARGE_IN_THRESHOLD, now_ns))
            return tuple(transitions)

        if state.voiced:
            state.voiced = False
            state.voiced_since_ns = None
            state.barge_threshold_reported = False
            if state.activity is VADActivity.USER_SPEAKING:
                state.silence_since_ns = now_ns
        if (
            state.activity is VADActivity.USER_SPEAKING
            and state.silence_since_ns is not None
            and not state.barge_in_active
            and now_ns - state.silence_since_ns >= self.config.post_speech_silence_ns
        ):
            state.activity = VADActivity.FINALIZED
            transitions.append(self._transition(metadata, VADTransitionKind.SMART_TURN_FINALIZED, now_ns))
        return tuple(transitions)

    def _probability(self, payload: object) -> float:
        provider = self._probability_provider
        value = provider.probability(payload) if isinstance(provider, VoiceActivityProbabilityProvider) else provider(payload)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
            raise VADFrameRejected("vad_probability_invalid")
        return float(value)

    def _apply_hysteresis(self, state: _TurnVADState, probability: float) -> bool:
        if probability >= self.config.speech_probability_threshold:
            return True
        if probability < self.config.release_probability_threshold:
            return False
        return state.voiced

    @staticmethod
    def _transition(
        metadata: AudioFrameMetadata,
        kind: VADTransitionKind,
        timestamp_ns: int,
    ) -> VADTransition:
        return VADTransition(
            turn_id=metadata.turn_id,
            kind=kind,
            occurred_monotonic_ns=timestamp_ns,
            cancellation_generation=metadata.cancellation_generation,
        )


TurnReadySink = Callable[[VoiceTurnRequest], Awaitable[None] | None]
PartialUISink = Callable[[TypedVoiceFrame[object]], Awaitable[None] | None]


@dataclass(slots=True)
class _JoinedTurnState:
    final_transcript: TypedVoiceFrame[object] | None = None
    finalized_generation: int | None = None
    eligible: bool = False


class TurnJoinProcessor:
    """Join ASR finality and VAD silence finality by the same turn identifier."""

    def __init__(
        self,
        *,
        session: VoiceSession,
        on_turn_ready: TurnReadySink | None = None,
        on_partial_ui: PartialUISink | None = None,
    ) -> None:
        self._session = session
        self._on_turn_ready = on_turn_ready
        self._on_partial_ui = on_partial_ui
        self._turns: dict[UUID, _JoinedTurnState] = {}
        self._lock = asyncio.Lock()

    async def process_transcription(self, frame: TypedVoiceFrame[object]) -> bool:
        """Record ASR state; partials update UI but cannot trigger generation."""
        metadata = frame.metadata
        if not isinstance(metadata, TranscriptionFrameMetadata):
            raise ValueError("turn_join_transcription_metadata_required")
        if not metadata.is_final:
            await self._transition_if_needed(metadata.turn_id, TurnState.PARTIAL)
            await _invoke(self._on_partial_ui, frame)
            return False

        async with self._lock:
            state = self._turns.setdefault(metadata.turn_id, _JoinedTurnState())
            if state.final_transcript is not None:
                return False
            state.final_transcript = frame
            await self._transition_if_needed(metadata.turn_id, TurnState.FINAL_PENDING_SILENCE)
            return await self._make_eligible_if_ready(metadata.turn_id, state)

    async def process_vad_transition(self, transition: VADTransition) -> bool:
        """Record VAD finality and release a turn only when ASR is also final."""
        if transition.kind is not VADTransitionKind.SMART_TURN_FINALIZED:
            return False
        try:
            record = self._session.turns.get(transition.turn_id)
        except UnknownTurnError:
            return False
        if record.is_terminal or record.cancellation_generation != transition.cancellation_generation:
            return False
        async with self._lock:
            state = self._turns.setdefault(transition.turn_id, _JoinedTurnState())
            if state.finalized_generation is not None:
                return False
            state.finalized_generation = transition.cancellation_generation
            return await self._make_eligible_if_ready(transition.turn_id, state)

    async def _make_eligible_if_ready(self, turn_id: UUID, state: _JoinedTurnState) -> bool:
        if state.eligible or state.finalized_generation is None or state.final_transcript is None:
            return False
        metadata = state.final_transcript.metadata
        if not isinstance(metadata, TranscriptionFrameMetadata):  # defensive type narrowing
            raise ValueError("turn_join_transcription_metadata_required")
        if metadata.cancellation_generation != state.finalized_generation:
            return False
        try:
            record = self._session.turns.get(turn_id)
        except UnknownTurnError:
            return False
        if record.is_terminal or record.cancellation_generation != metadata.cancellation_generation:
            return False
        payload = state.final_transcript.payload
        text = getattr(payload, "text", payload if isinstance(payload, str) else None)
        if not isinstance(text, str) or not text.strip():
            return False
        state.eligible = True
        await self._session.append_user_turn(turn_id=turn_id, text=text)
        await self._transition_if_needed(turn_id, TurnState.REASONING)
        await _invoke(
            self._on_turn_ready,
            VoiceTurnRequest(
                session_id=metadata.session_id,
                turn_id=turn_id,
                text=text,
                language=metadata.language,
            ),
        )
        return True

    async def _transition_if_needed(self, turn_id: UUID, target: TurnState) -> None:
        record = self._session.turns.get(turn_id)
        if record.state is target:
            return
        try:
            await self._session.turns.transition(turn_id, target)
        except InvalidTurnTransition:
            # Terminal/cancelled turns cannot be revived by a late transcript or VAD frame.
            return


RendererStopSink = Callable[[UUID, int], Awaitable[None] | None]
CancellationSink = Callable[[UUID, int], Awaitable[None] | None]
CaptureResumeSink = Callable[[], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class BargeInResult:
    """The atomic cancellation decision, not completion of background cleanup."""

    interrupted_turn_id: UUID
    interrupted_generation: int
    cancellation_generation: int


class BargeInCoordinator:
    """Declare interruption and invalidate an assistant generation immediately."""

    def __init__(
        self,
        *,
        session: VoiceSession,
        renderer_stop: RendererStopSink | None = None,
        cancel_work: CancellationSink | None = None,
        resume_capture: CaptureResumeSink | None = None,
    ) -> None:
        self._session = session
        self._renderer_stop = renderer_stop
        self._cancel_work = cancel_work
        self._resume_capture = resume_capture
        self._active_playback: tuple[UUID, int] | None = None
        self._active = False
        self._lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task[None]] = set()

    @property
    def playback_active(self) -> bool:
        return self._active_playback is not None

    @property
    def barge_in_active(self) -> bool:
        return self._active

    def start_playback(self, *, turn_id: UUID, generation: int) -> None:
        """Mark the currently audible assistant generation for VAD interruption."""
        self._active_playback = (turn_id, generation)

    def finish_playback(self, *, turn_id: UUID) -> None:
        """Remove completed playback from future barge-in detection."""
        if self._active_playback is not None and self._active_playback[0] == turn_id:
            self._active_playback = None

    async def process_vad_transition(self, transition: VADTransition) -> BargeInResult | None:
        """Cancel the audible assistant turn once continuous speech reaches 200 ms."""
        if transition.kind is not VADTransitionKind.BARGE_IN_THRESHOLD:
            return None
        return await self.declare_barge_in(capture_turn_id=transition.turn_id)

    async def declare_barge_in(self, *, capture_turn_id: UUID | None = None) -> BargeInResult | None:
        """Atomically advance generation and launch stop/cleanup without waiting."""
        async with self._lock:
            active = self._active_playback
            if active is None or self._active:
                return None
            turn_id, interrupted_generation = active
            record = self._session.turns.get(turn_id)
            if record.is_terminal or record.cancellation_generation != interrupted_generation:
                self._active_playback = None
                return None
            self._active = True
            try:
                cancellation_generation = await self._session.cancel_turn(turn_id)
                if capture_turn_id is not None and capture_turn_id != turn_id:
                    await self._session.rebase_capturing_turn_generation(capture_turn_id)
                self._active_playback = None
                result = BargeInResult(turn_id, interrupted_generation, cancellation_generation)
                self._schedule(self._renderer_stop, turn_id, cancellation_generation)
                self._schedule(self._cancel_work, turn_id, interrupted_generation)
                # Capture starts immediately; cancellation tasks continue independently.
                self._schedule(self._resume_capture)
                return result
            finally:
                self._active = False

    def _schedule(self, callback: Callable[..., Awaitable[None] | None] | None, *args: object) -> None:
        if callback is None:
            return
        result = callback(*args)
        if inspect.isawaitable(result):
            task = asyncio.create_task(result)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)


async def _invoke(callback: Callable[..., Awaitable[None] | None] | None, *args: object) -> None:
    if callback is None:
        return
    result = callback(*args)
    if inspect.isawaitable(result):
        await result


def _payload_probability(payload: object) -> float:
    """Default adapter for test/production wrappers that expose a probability."""
    value = getattr(payload, "speech_probability", 0.0)
    return float(value)


__all__ = [
    "BargeInCoordinator",
    "BargeInResult",
    "SILERO_CHANNELS",
    "SILERO_SAMPLE_RATE_HZ",
    "SileroVADSmartTurnProcessor",
    "SmartTurnVADConfig",
    "TurnJoinProcessor",
    "VADActivity",
    "VADConfigurationError",
    "VADFrameRejected",
    "VADTransition",
    "VADTransitionKind",
    "VoiceActivityProbabilityProvider",
]
