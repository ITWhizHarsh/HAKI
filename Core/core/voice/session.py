"""Session-owned voice turn lifecycle, cancellation, queues, and context.

A :class:`VoiceSession` represents exactly one active local voice session.  It
owns all turn records, cancellation generations, bounded queues, and the
confirmed-playback context supplied to future voice turns.  It intentionally
has no link to the process-wide orchestrator or its conversation history.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeVar
from uuid import UUID

from .cloud_gate import CloudEscalationGate, CloudEscalationState
from .frames import TypedVoiceFrame, VoiceFrameMetadata
from .interfaces import VoiceContextMessage


class TurnState(str, Enum):
    """Linear lifecycle states for one turn."""

    CAPTURING = "capturing"
    PARTIAL = "partial"
    FINAL_PENDING_SILENCE = "final_pending_silence"
    REASONING = "reasoning"
    SYNTHESIZING = "synthesizing"
    PLAYING = "playing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


TERMINAL_TURN_STATES = frozenset({TurnState.COMPLETED, TurnState.CANCELLED, TurnState.FAILED})

_ALLOWED_TRANSITIONS: dict[TurnState, frozenset[TurnState]] = {
    TurnState.CAPTURING: frozenset({TurnState.PARTIAL, TurnState.FINAL_PENDING_SILENCE, TurnState.CANCELLED, TurnState.FAILED}),
    TurnState.PARTIAL: frozenset({TurnState.FINAL_PENDING_SILENCE, TurnState.CANCELLED, TurnState.FAILED}),
    TurnState.FINAL_PENDING_SILENCE: frozenset({TurnState.REASONING, TurnState.CANCELLED, TurnState.FAILED}),
    TurnState.REASONING: frozenset({TurnState.SYNTHESIZING, TurnState.CANCELLED, TurnState.FAILED}),
    TurnState.SYNTHESIZING: frozenset({TurnState.PLAYING, TurnState.CANCELLED, TurnState.FAILED}),
    TurnState.PLAYING: frozenset({TurnState.COMPLETED, TurnState.CANCELLED, TurnState.FAILED}),
    TurnState.COMPLETED: frozenset(),
    TurnState.CANCELLED: frozenset(),
    TurnState.FAILED: frozenset(),
}


class VoiceSessionError(RuntimeError):
    """Base error for safe, content-free voice session rejections."""


class UnknownTurnError(VoiceSessionError):
    """A caller referenced a turn that is not registered in this session."""


class DuplicateTurnError(VoiceSessionError):
    """A second active record was requested for the same turn identifier."""


class InvalidTurnTransition(VoiceSessionError):
    """A turn attempted to skip, reopen, or otherwise violate its lifecycle."""


class LateFrameRejected(VoiceSessionError):
    """A terminal, stale-generation, or cross-session frame was rejected."""


class FrameOrderingError(VoiceSessionError):
    """A frame did not use the next accepted sequence for its turn."""


class VoiceSessionClosedError(VoiceSessionError):
    """A closed session cannot accept new turns, frames, or context."""


class TurnQueueName(str, Enum):
    """Bounded queue destinations with their design-specific backpressure policy."""

    PARTIAL = "partial"
    CONTROL = "control"
    LLM = "llm"
    SENTENCE = "sentence"
    PCM = "pcm"


@dataclass(frozen=True, slots=True)
class VoiceQueueLimits:
    """Capacities for a session turn's independently bounded frame queues."""

    partial: int = 1
    control: int = 32
    llm: int = 32
    sentence: int = 16
    pcm: int = 64

    def __post_init__(self) -> None:
        for name, capacity in (("partial", self.partial), ("control", self.control), ("llm", self.llm), ("sentence", self.sentence), ("pcm", self.pcm)):
            if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
                raise ValueError(f"{name} queue capacity must be a positive integer")


FrameT = TypeVar("FrameT", bound=TypedVoiceFrame[object])


@dataclass(slots=True)
class TurnQueues:
    """Queues owned by exactly one turn with explicit backpressure policies."""

    partial: asyncio.Queue[TypedVoiceFrame[object]]
    control: asyncio.Queue[TypedVoiceFrame[object]]
    llm: asyncio.Queue[TypedVoiceFrame[object]]
    sentence: asyncio.Queue[TypedVoiceFrame[object]]
    pcm: asyncio.Queue[TypedVoiceFrame[object]]

    @classmethod
    def create(cls, limits: VoiceQueueLimits) -> "TurnQueues":
        return cls(
            partial=asyncio.Queue(maxsize=limits.partial),
            control=asyncio.Queue(maxsize=limits.control),
            llm=asyncio.Queue(maxsize=limits.llm),
            sentence=asyncio.Queue(maxsize=limits.sentence),
            pcm=asyncio.Queue(maxsize=limits.pcm),
        )

    def queue_for(self, name: TurnQueueName) -> asyncio.Queue[TypedVoiceFrame[object]]:
        return getattr(self, name.value)

    async def put(self, name: TurnQueueName, frame: TypedVoiceFrame[object]) -> None:
        queue = self.queue_for(name)
        if name is TurnQueueName.PARTIAL:
            while queue.full():
                queue.get_nowait()
                queue.task_done()
            queue.put_nowait(frame)
            return
        await queue.put(frame)

    def clear_output_work(self) -> None:
        """Discard unscheduled output after cancellation without touching control."""
        for queue in (self.llm, self.sentence, self.pcm):
            while not queue.empty():
                queue.get_nowait()
                queue.task_done()


@dataclass(slots=True)
class TurnRecord:
    """Mutable session-local state guarded by ``lock`` for one turn."""

    turn_id: UUID
    cancellation_generation: int
    queues: TurnQueues
    state: TurnState = TurnState.CAPTURING
    next_sequence: int | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_TURN_STATES


@dataclass(frozen=True, slots=True)
class PlayedSentence:
    """One assistant sentence that the renderer confirmed as fully played."""

    turn_id: UUID
    sentence_id: UUID
    text: str
    playback_completed_monotonic_ns: int

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("played sentence text must not be empty")
        if not isinstance(self.playback_completed_monotonic_ns, int) or isinstance(self.playback_completed_monotonic_ns, bool) or self.playback_completed_monotonic_ns < 0:
            raise ValueError("playback completion timestamp must be a non-negative integer")


class ProvisionalSentenceState(str, Enum):
    """Renderer-visible lifecycle for a generated assistant sentence."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(slots=True)
class ProvisionalSentence:
    """Assistant text that is intentionally absent from context until confirmed."""

    turn_id: UUID
    sentence_id: UUID
    text: str
    cancellation_generation: int
    state: ProvisionalSentenceState = ProvisionalSentenceState.PENDING


@dataclass(frozen=True, slots=True)
class VoiceContext:
    """An immutable snapshot of context owned only by one ``VoiceSession``."""

    user_turns: tuple[VoiceContextMessage, ...]
    assistant_sentences: tuple[PlayedSentence, ...]
    messages: tuple[VoiceContextMessage, ...]


class TurnRegistry:
    """Per-session turn registry with serial ordering for each identifier."""

    def __init__(self, session_id: UUID, *, queue_limits: VoiceQueueLimits | None = None) -> None:
        if not isinstance(session_id, UUID):
            raise ValueError("session_id must be a UUID")
        self._session_id = session_id
        self._queue_limits = queue_limits or VoiceQueueLimits()
        self._turns: dict[UUID, TurnRecord] = {}
        self._creation_lock = asyncio.Lock()

    @property
    def session_id(self) -> UUID:
        return self._session_id

    async def create_turn(self, turn_id: UUID, *, cancellation_generation: int) -> TurnRecord:
        if not isinstance(turn_id, UUID):
            raise ValueError("turn_id must be a UUID")
        _require_generation(cancellation_generation)
        async with self._creation_lock:
            if turn_id in self._turns:
                raise DuplicateTurnError("turn is already registered")
            record = TurnRecord(turn_id=turn_id, cancellation_generation=cancellation_generation, queues=TurnQueues.create(self._queue_limits))
            self._turns[turn_id] = record
            return record

    def get(self, turn_id: UUID) -> TurnRecord:
        try:
            return self._turns[turn_id]
        except KeyError as exc:
            raise UnknownTurnError("turn is not registered for this voice session") from exc

    async def transition(self, turn_id: UUID, target: TurnState) -> TurnState:
        record = self.get(turn_id)
        async with record.lock:
            if record.state is target:
                return record.state
            if target not in _ALLOWED_TRANSITIONS[record.state]:
                raise InvalidTurnTransition(f"cannot transition {record.state.value} to {target.value}")
            record.state = target
            return record.state

    async def cancel(self, turn_id: UUID, *, cancellation_generation: int) -> int:
        _require_generation(cancellation_generation)
        record = self.get(turn_id)
        async with record.lock:
            if record.is_terminal:
                raise LateFrameRejected("terminal turn cannot be cancelled again")
            if cancellation_generation <= record.cancellation_generation:
                raise LateFrameRejected("cancellation generation is stale")
            record.cancellation_generation = cancellation_generation
            record.state = TurnState.CANCELLED
            record.queues.clear_output_work()
            return record.cancellation_generation

    async def accept_frame(self, frame: TypedVoiceFrame[object], *, queue: TurnQueueName) -> None:
        metadata = frame.metadata
        if metadata.session_id != self._session_id:
            raise LateFrameRejected("frame belongs to a different voice session")
        record = self.get(metadata.turn_id)
        async with record.lock:
            if record.is_terminal:
                raise LateFrameRejected("terminal turn rejects late frames")
            if metadata.cancellation_generation != record.cancellation_generation:
                raise LateFrameRejected("frame cancellation generation is stale")
            if record.next_sequence is not None and metadata.sequence != record.next_sequence:
                raise FrameOrderingError("frame sequence is not the next sequence for this turn")
            record.next_sequence = metadata.sequence + 1
            await record.queues.put(queue, frame)


class PlaybackLedger:
    """The only writer of renderer-confirmed assistant context for a session."""

    def __init__(self, session: "VoiceSession") -> None:
        self._session = session
        self._sentences: dict[UUID, ProvisionalSentence] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        *,
        turn_id: UUID,
        sentence_id: UUID,
        text: str,
        cancellation_generation: int,
    ) -> bool:
        """Record queued text as provisional only if its generation remains live."""
        if not text.strip():
            raise ValueError("provisional sentence text must not be empty")
        record = self._session.turns.get(turn_id)
        async with self._lock:
            async with record.lock:
                if record.is_terminal or record.cancellation_generation != cancellation_generation:
                    return False
            if sentence_id in self._sentences:
                return False
            self._sentences[sentence_id] = ProvisionalSentence(turn_id, sentence_id, text, cancellation_generation)
            return True

    async def confirm(
        self,
        *,
        turn_id: UUID,
        sentence_id: UUID,
        playback_completed_monotonic_ns: int,
    ) -> bool:
        """Append only a pending registered sentence in renderer-confirmation order."""
        async with self._lock:
            provisional = self._sentences.get(sentence_id)
            if provisional is None or provisional.turn_id != turn_id or provisional.state is not ProvisionalSentenceState.PENDING:
                return False
            record = self._session.turns.get(turn_id)
            async with record.lock:
                if record.is_terminal or record.cancellation_generation != provisional.cancellation_generation:
                    provisional.state = ProvisionalSentenceState.CANCELLED
                    return False
                provisional.state = ProvisionalSentenceState.CONFIRMED
                await self._session._append_confirmed_sentence(
                    turn_id=turn_id,
                    sentence_id=sentence_id,
                    text=provisional.text,
                    playback_completed_monotonic_ns=playback_completed_monotonic_ns,
                )
                return True

    async def cancel(self, *, turn_id: UUID, cancellation_generation: int) -> None:
        """Make unconfirmed sentences from the interrupted generation ineligible forever."""
        async with self._lock:
            for provisional in self._sentences.values():
                if provisional.turn_id == turn_id and provisional.cancellation_generation < cancellation_generation and provisional.state is ProvisionalSentenceState.PENDING:
                    provisional.state = ProvisionalSentenceState.CANCELLED

    async def fail(self, *, turn_id: UUID, sentence_id: UUID) -> bool:
        async with self._lock:
            provisional = self._sentences.get(sentence_id)
            if provisional is None or provisional.turn_id != turn_id or provisional.state is not ProvisionalSentenceState.PENDING:
                return False
            provisional.state = ProvisionalSentenceState.FAILED
            return True

    async def state_for(self, sentence_id: UUID) -> ProvisionalSentenceState | None:
        async with self._lock:
            provisional = self._sentences.get(sentence_id)
            return None if provisional is None else provisional.state


class VoiceSession:
    """Own one active voice session's turns, generations, queues, and context."""

    def __init__(self, session_id: UUID, *, queue_limits: VoiceQueueLimits | None = None) -> None:
        if not isinstance(session_id, UUID):
            raise ValueError("session_id must be a UUID")
        self.session_id = session_id
        self.cloud_gate = CloudEscalationGate()
        self.cloud_gate.register_session(session_id)
        self.turns = TurnRegistry(session_id, queue_limits=queue_limits)
        self._cancellation_generation = 0
        self._closed = False
        self._session_lock = asyncio.Lock()
        self._context_lock = asyncio.Lock()
        self._user_turns: list[VoiceContextMessage] = []
        self._assistant_sentences: list[PlayedSentence] = []
        self._messages: list[VoiceContextMessage] = []
        self.playback_ledger = PlaybackLedger(self)

    @property
    def cancellation_generation(self) -> int:
        return self._cancellation_generation

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def cloud_escalation_state(self) -> CloudEscalationState:
        """Expose the active session's Gemini state for the voice UI."""
        return self.cloud_gate.ui_state(self.session_id)

    async def set_gemini_live_enabled(self, enabled: bool) -> CloudEscalationState:
        """Apply an explicit UI action to Gemini for this active session only."""
        async with self._session_lock:
            self._ensure_open()
            return self.cloud_gate.set_enabled(self.session_id, enabled=enabled)

    async def start_turn(self, turn_id: UUID) -> TurnRecord:
        async with self._session_lock:
            self._ensure_open()
            return await self.turns.create_turn(turn_id, cancellation_generation=self._cancellation_generation)

    async def cancel_turn(self, turn_id: UUID) -> int:
        """Atomically advance generation, drain matching queues, and cancel ledger text."""
        async with self._session_lock:
            self._ensure_open()
            record = self.turns.get(turn_id)
            if record.is_terminal:
                raise LateFrameRejected("terminal turn cannot be cancelled again")
            previous_generation = record.cancellation_generation
            self._cancellation_generation += 1
            generation = await self.turns.cancel(turn_id, cancellation_generation=self._cancellation_generation)
            await self.playback_ledger.cancel(turn_id=turn_id, cancellation_generation=generation)
            assert previous_generation < generation
            return generation

    async def rebase_capturing_turn_generation(self, turn_id: UUID) -> int:
        """Attach a just-started user capture to the latest barge-in generation."""
        async with self._session_lock:
            self._ensure_open()
            record = self.turns.get(turn_id)
            async with record.lock:
                if record.is_terminal:
                    raise LateFrameRejected("terminal turn cannot resume capture")
                record.cancellation_generation = self._cancellation_generation
                return record.cancellation_generation

    async def accept_frame(self, frame: TypedVoiceFrame[object], *, queue: TurnQueueName) -> None:
        self._ensure_open()
        await self.turns.accept_frame(frame, queue=queue)

    async def output_is_current(self, frame: TypedVoiceFrame[object]) -> bool:
        """Gate async output dispatch so cancelled queued work cannot reach a sink."""
        if self._closed or frame.metadata.session_id != self.session_id:
            return False
        try:
            record = self.turns.get(frame.metadata.turn_id)
        except UnknownTurnError:
            return False
        async with record.lock:
            return not record.is_terminal and record.cancellation_generation == frame.metadata.cancellation_generation

    async def append_user_turn(self, *, turn_id: UUID, text: str) -> None:
        self._ensure_open()
        if not text.strip():
            raise ValueError("user turn text must not be empty")
        async with self._context_lock:
            if any(item.turn_id == turn_id for item in self._user_turns):
                raise ValueError("user turn already exists in voice context")
            message = VoiceContextMessage(turn_id=turn_id, role="user", text=text)
            self._user_turns.append(message)
            self._messages.append(message)

    async def confirm_played_sentence(
        self,
        *,
        turn_id: UUID,
        sentence_id: UUID,
        text: str,
        playback_completed_monotonic_ns: int,
    ) -> None:
        """Compatibility API for an already-normal renderer confirmation.

        Pipeline code must instead register provisional text and use
        ``playback_ledger.confirm`` so late confirmations after cancellation are
        rejected.  This direct method keeps prior session-only callers working.
        """
        self._ensure_open()
        await self._append_confirmed_sentence(
            turn_id=turn_id,
            sentence_id=sentence_id,
            text=text,
            playback_completed_monotonic_ns=playback_completed_monotonic_ns,
        )

    async def _append_confirmed_sentence(
        self,
        *,
        turn_id: UUID,
        sentence_id: UUID,
        text: str,
        playback_completed_monotonic_ns: int,
    ) -> None:
        sentence = PlayedSentence(turn_id, sentence_id, text, playback_completed_monotonic_ns)
        async with self._context_lock:
            if any(item.sentence_id == sentence_id for item in self._assistant_sentences):
                raise ValueError("sentence already exists in voice context")
            self._assistant_sentences.append(sentence)
            self._messages.append(VoiceContextMessage(turn_id=turn_id, role="assistant", text=text))

    async def context_snapshot(self) -> VoiceContext:
        async with self._context_lock:
            return VoiceContext(tuple(self._user_turns), tuple(self._assistant_sentences), tuple(self._messages))

    async def close(self) -> None:
        async with self._session_lock:
            if self._closed:
                return
            self.cloud_gate.end_session(self.session_id)
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise VoiceSessionClosedError("voice session is closed")


def _require_generation(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("cancellation_generation must be a non-negative integer")


__all__ = [
    "DuplicateTurnError",
    "FrameOrderingError",
    "InvalidTurnTransition",
    "LateFrameRejected",
    "PlaybackLedger",
    "PlayedSentence",
    "ProvisionalSentence",
    "ProvisionalSentenceState",
    "TERMINAL_TURN_STATES",
    "TurnQueueName",
    "TurnQueues",
    "TurnRecord",
    "TurnRegistry",
    "TurnState",
    "UnknownTurnError",
    "VoiceContext",
    "VoiceQueueLimits",
    "VoiceSession",
    "VoiceSessionClosedError",
    "VoiceSessionError",
]
