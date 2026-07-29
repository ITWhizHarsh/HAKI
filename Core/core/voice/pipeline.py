"""Pipecat frame compatibility, ingress, and the session-owned voice graph.

Only this module imports Pipecat's concrete frame classes.  It builds one owned
``PipelineTask`` for a voice session, while the surrounding graph uses bounded
``asyncio.Queue`` instances and an ``asyncio.TaskGroup`` in the existing event
loop.  It intentionally contains no legacy routing, custom playback thread, or
subprocess fallback.

Sentence boundary → TTS integration pattern
--------------------------------------------
The ``SentenceBoundaryProcessor`` (``core.voice.tts``) and
``XTTSSentenceAdapter`` connect to this pipeline via the ``tts_text`` and
``pcm`` sinks in :class:`VoicePipelineSinks`.  The typical wiring is::

    from core.voice.tts import SentenceBoundaryProcessor, XTTSSentenceAdapter

    boundary = SentenceBoundaryProcessor()
    tts_adapter = XTTSSentenceAdapter(
        voice_asset_path=voice_asset,
        pipeline=pipeline,
        text_fallback_sink=my_text_fallback,
    )
    await tts_adapter.initialize_session()

    sentence_seq = 0

    async def llm_text_sink(frame: TypedVoiceFrame) -> None:
        # frame.payload is a Pipecat LLMTextFrame; extract its text attribute.
        text = getattr(frame.payload, "text", "") or ""
        for sentence_text in boundary.feed(text):
            sentence = VoiceSentence(
                turn_id=frame.metadata.turn_id,
                sentence_id=uuid4(),
                text=sentence_text,
                language=detected_language,
            )
            await tts_adapter.synthesize_sentence(sentence, sequence=sentence_seq)
            sentence_seq += 1

    async def on_llm_stream_end(turn_id: UUID) -> None:
        # Flush any trailing sentence at end of LLM stream.
        for sentence_text in boundary.flush():
            sentence = VoiceSentence(
                turn_id=turn_id,
                sentence_id=uuid4(),
                text=sentence_text,
                language=detected_language,
            )
            await tts_adapter.synthesize_sentence(sentence, sequence=sentence_seq)

    sinks = VoicePipelineSinks(llm_text=llm_text_sink)

The ``XTTSSentenceAdapter`` handles all PCM chunking
(``pipeline.emit_pcm_chunk``), TTFA measurement (recorded in
:class:`~.diagnostics.VoiceDiagnosticEvent`), and failure reporting without
ever selecting an alternate speech engine.  ``emit_tts_text`` is called by the
adapter before synthesis begins so the ordered ``TTSTextFrame`` is always
placed in the graph queue before the first PCM chunk.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from threading import Event
from time import monotonic_ns
from typing import Any, Protocol, TypeVar, runtime_checkable
from uuid import UUID

from core.ipc.voice_protocol import (
    PLAYBACK_CANCELLED,
    PLAYBACK_CONFIRMED,
    PLAYBACK_FAILED,
    STOP_PLAYBACK,
    validate_message,
)

from .asr_bridge import (
    AuthenticatedRingSlotReader,
    AudioRingIngress,
    PipecatIngressFrameFactory,
    RingIngressResult,
    RingSlotDescriptor,
    TranscriptIngressResult,
    TranscriptSocketIngress,
)
from .frames import (
    PCMChunkFrameMetadata,
    SentenceFrameMetadata,
    TranscriptionFrameMetadata,
    TypedVoiceFrame,
    VoiceFrameMetadata,
    VoiceFrameType,
)
from .interfaces import VoiceTurnRequest
from .session import (
    DuplicateTurnError,
    LateFrameRejected,
    TurnQueueName,
    UnknownTurnError,
    VoiceQueueLimits,
    VoiceSession,
)
from .vad import BargeInCoordinator, SileroVADSmartTurnProcessor, TurnJoinProcessor, VADTransition


class PipecatFrameAdapterUnavailable(RuntimeError):
    """Pipecat is unavailable or exposes incompatible mandatory frame APIs."""


class PipelineInitializationError(RuntimeError):
    """The replacement Pipecat graph could not be constructed atomically."""


class VoicePipelineUnavailable(RuntimeError):
    """The voice graph is not available to accept new work."""


class BlockingLibraryError(RuntimeError):
    """A caller tried to use the blocking executor for a non-voice library."""


class PipelineAvailability(str, Enum):
    """Lifecycle state used by the voice service to gate turn admission."""

    NEW = "new"
    RUNNING = "running"
    UNAVAILABLE = "unavailable"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class PipelineQueueLimits:
    """Explicit bounds for the session graph's independent frame queues."""

    audio: int = 32
    partial: int = 1
    control: int = 32
    llm: int = 32
    sentence: int = 16
    pcm: int = 64

    def __post_init__(self) -> None:
        for name, capacity in (
            ("audio", self.audio),
            ("partial", self.partial),
            ("control", self.control),
            ("llm", self.llm),
            ("sentence", self.sentence),
            ("pcm", self.pcm),
        ):
            if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
                raise ValueError(f"{name} queue capacity must be a positive integer")

    def session_limits(self) -> VoiceQueueLimits:
        """Use matching per-turn bounds when the pipeline owns its session."""
        return VoiceQueueLimits(
            partial=self.partial,
            control=self.control,
            llm=self.llm,
            sentence=self.sentence,
            pcm=self.pcm,
        )


@dataclass(frozen=True, slots=True)
class PipelineDiagnostic:
    """A minimal content-free initialization/runtime diagnostic boundary."""

    stage: str
    outcome: str
    error_class: str | None = None


FrameSink = Callable[[TypedVoiceFrame[object]], Awaitable[None] | None]
TurnReadySink = Callable[[VoiceTurnRequest], Awaitable[None] | None]
RendererStopSink = Callable[[UUID, int], Awaitable[None] | None]
CaptureResumeSink = Callable[[], Awaitable[None] | None]
DiagnosticSink = Callable[[PipelineDiagnostic], Awaitable[None] | None]
PipelineTaskFactory = Callable[[], object]
BlockingResult = TypeVar("BlockingResult")
BlockingOperation = Callable[[Event], BlockingResult]


@dataclass(frozen=True, slots=True)
class VoicePipelineSinks:
    """Optional asynchronous graph endpoints supplied by later voice stages.

    ``turn_ready`` is deliberately distinct from final-transcript display.  It
    fires only after the final transcript and 800 ms VAD condition join.  The
    renderer-stop and capture-resume callbacks are non-blocking control paths
    scheduled by :class:`BargeInCoordinator`.
    """

    input_audio: FrameSink | None = None
    partial_transcription: FrameSink | None = None
    final_transcription: FrameSink | None = None
    turn_ready: TurnReadySink | None = None
    stop_playback: RendererStopSink | None = None
    capture_resumed: CaptureResumeSink | None = None
    llm_text: FrameSink | None = None
    tts_text: FrameSink | None = None
    pcm: FrameSink | None = None


class PipelineTurnCoordinator:
    """Task 6 processors connecting VAD, turn joining, barge-in, and ledger state."""

    def __init__(
        self,
        *,
        session: VoiceSession,
        sinks: VoicePipelineSinks,
        drain_interrupted_work: Callable[[UUID, int], Awaitable[None]],
        vad_processor: SileroVADSmartTurnProcessor | None = None,
    ) -> None:
        self._session = session
        self._sinks = sinks
        self.vad = vad_processor or SileroVADSmartTurnProcessor()
        self.turn_join = TurnJoinProcessor(
            session=session,
            on_turn_ready=sinks.turn_ready,
            on_partial_ui=None,
        )
        self.barge_in = BargeInCoordinator(
            session=session,
            renderer_stop=self._stop_renderer,
            cancel_work=drain_interrupted_work,
            resume_capture=self._resume_capture,
        )

    async def process_audio(self, frame: TypedVoiceFrame[object]) -> tuple[VADTransition, ...]:
        transitions = self.vad.process(frame, playback_active=self.barge_in.playback_active)
        for transition in transitions:
            await self.turn_join.process_vad_transition(transition)
            if transition.kind.value == "barge_in_threshold":
                await self.barge_in.process_vad_transition(transition)
        return transitions

    async def process_transcription(self, frame: TypedVoiceFrame[object]) -> bool:
        return await self.turn_join.process_transcription(frame)

    async def register_tts_sentence(self, frame: TypedVoiceFrame[object], *, text: str) -> bool:
        metadata = frame.metadata
        if not isinstance(metadata, SentenceFrameMetadata):
            raise ValueError("tts_sentence_metadata_required")
        accepted = await self._session.playback_ledger.register(
            turn_id=metadata.turn_id,
            sentence_id=metadata.sentence_id,
            text=text,
            cancellation_generation=metadata.cancellation_generation,
        )
        if accepted:
            self.barge_in.start_playback(
                turn_id=metadata.turn_id,
                generation=metadata.cancellation_generation,
            )
        return accepted

    async def process_playback_event(
        self,
        message: Mapping[str, object],
        *,
        playback_completed_monotonic_ns: int | None = None,
    ) -> bool:
        validated = validate_message(message).data
        if UUID(validated["session_id"]) != self._session.session_id:
            return False
        turn_id = UUID(validated["turn_id"])
        sentence_id = UUID(validated["sentence_id"])
        event_type = validated["type"]
        if event_type == PLAYBACK_CONFIRMED:
            confirmed = await self._session.playback_ledger.confirm(
                turn_id=turn_id,
                sentence_id=sentence_id,
                playback_completed_monotonic_ns=(
                    monotonic_ns()
                    if playback_completed_monotonic_ns is None
                    else playback_completed_monotonic_ns
                ),
            )
            self.barge_in.finish_playback(turn_id=turn_id)
            return confirmed
        if event_type == PLAYBACK_CANCELLED:
            await self._session.playback_ledger.cancel(
                turn_id=turn_id,
                cancellation_generation=self._session.cancellation_generation,
            )
            self.barge_in.finish_playback(turn_id=turn_id)
            return True
        if event_type == PLAYBACK_FAILED:
            self.barge_in.finish_playback(turn_id=turn_id)
            return await self._session.playback_ledger.fail(turn_id=turn_id, sentence_id=sentence_id)
        return False

    async def _stop_renderer(self, turn_id: UUID, generation: int) -> None:
        await _invoke_callback(self._sinks.stop_playback, turn_id, generation)

    async def _resume_capture(self) -> None:
        await _invoke_callback(self._sinks.capture_resumed)


class PipecatFrameAdapter(PipecatIngressFrameFactory):
    """Construct the mandatory Pipecat frame values behind one adapter.

    HAKI metadata remains in ``TypedVoiceFrame.metadata`` instead of text
    payloads.  Imports and constructor compatibility are confined here so a
    Pipecat pin update does not leak through the rest of ``core.voice``.
    """

    def __init__(
        self,
        *,
        input_audio_frame_type: type[Any] | None = None,
        transcription_frame_type: type[Any] | None = None,
        llm_text_frame_type: type[Any] | None = None,
        tts_text_frame_type: type[Any] | None = None,
    ) -> None:
        self._input_audio_frame_type = input_audio_frame_type
        self._transcription_frame_type = transcription_frame_type
        self._llm_text_frame_type = llm_text_frame_type
        self._tts_text_frame_type = tts_text_frame_type
        if input_audio_frame_type is None or transcription_frame_type is None:
            self._load_frame_types()

    def validate_mandatory_frames(self) -> None:
        """Fail initialization before a session starts if any required type is absent."""
        self._load_frame_types()

    def create_input_audio_frame(
        self, *, audio: bytes, sample_rate_hz: int, channels: int
    ) -> object:
        """Create ``InputAudioRawFrame(audio, sample_rate, num_channels)``."""
        frame_type = self._required_type("_input_audio_frame_type")
        try:
            return frame_type(audio=audio, sample_rate=sample_rate_hz, num_channels=channels)
        except TypeError:
            try:
                return frame_type(audio, sample_rate_hz, channels)
            except TypeError as exc:
                raise PipecatFrameAdapterUnavailable("input_audio_frame_constructor_invalid") from exc

    def create_transcription_frame(self, *, text: str) -> object:
        """Create a ``TranscriptionFrame`` using normalized text unchanged."""
        frame_type = self._required_type("_transcription_frame_type")
        try:
            return frame_type(text=text, user_id="local-user")
        except TypeError:
            try:
                return frame_type(text, "local-user")
            except TypeError:
                try:
                    return frame_type(text=text)
                except TypeError as exc:
                    raise PipecatFrameAdapterUnavailable(
                        "transcription_frame_constructor_invalid"
                    ) from exc

    def create_llm_text_frame(self, *, text: str) -> object:
        """Create an ``LLMTextFrame`` without embedding HAKI metadata in text."""
        return self._create_text_frame("_llm_text_frame_type", text, "llm_text_frame_constructor_invalid")

    def create_tts_text_frame(self, *, text: str) -> object:
        """Create a ``TTSTextFrame`` without embedding HAKI metadata in text."""
        return self._create_text_frame("_tts_text_frame_type", text, "tts_text_frame_constructor_invalid")

    def _create_text_frame(self, attribute: str, text: str, error: str) -> object:
        frame_type = self._required_type(attribute)
        try:
            return frame_type(text=text)
        except TypeError:
            try:
                return frame_type(text)
            except TypeError as exc:
                raise PipecatFrameAdapterUnavailable(error) from exc

    def _required_type(self, attribute: str) -> type[Any]:
        frame_type = getattr(self, attribute)
        if frame_type is None:
            self._load_frame_types()
            frame_type = getattr(self, attribute)
        if frame_type is None:  # Defensive guard for malformed third-party packages.
            raise PipecatFrameAdapterUnavailable("mandatory_pipecat_frame_unavailable")
        return frame_type

    def _load_frame_types(self) -> None:
        if all(
            frame_type is not None
            for frame_type in (
                self._input_audio_frame_type,
                self._transcription_frame_type,
                self._llm_text_frame_type,
                self._tts_text_frame_type,
            )
        ):
            return
        try:
            from pipecat.frames.frames import (
                InputAudioRawFrame,
                LLMTextFrame,
                TTSTextFrame,
                TranscriptionFrame,
            )
        except (ImportError, AttributeError) as exc:
            raise PipecatFrameAdapterUnavailable("pipecat_frames_unavailable") from exc
        self._input_audio_frame_type = self._input_audio_frame_type or InputAudioRawFrame
        self._transcription_frame_type = self._transcription_frame_type or TranscriptionFrame
        self._llm_text_frame_type = self._llm_text_frame_type or LLMTextFrame
        self._tts_text_frame_type = self._tts_text_frame_type or TTSTextFrame


class VoiceIngressProcessors:
    """Session-owned authenticated ring and text-only transcript ingress."""

    def __init__(
        self,
        *,
        session: VoiceSession,
        ring_reader: AuthenticatedRingSlotReader,
        frame_adapter: PipecatIngressFrameFactory | None = None,
    ) -> None:
        self.frame_adapter = frame_adapter or PipecatFrameAdapter()
        self.audio_ring = AudioRingIngress(
            session=session,
            ring_reader=ring_reader,
            frame_factory=self.frame_adapter,
        )
        self.transcripts = TranscriptSocketIngress(
            session=session,
            frame_factory=self.frame_adapter,
        )

    async def ingest_ring_slot(self, descriptor: RingSlotDescriptor) -> RingIngressResult:
        """Map/release a ring slot into the mandatory Silero input frame."""
        return await self.audio_ring.ingest(descriptor)

    async def ingest_transcript_message(self, message: object) -> TranscriptIngressResult:
        """Validate, sequence, wrap, and ACK a text-only transcript message."""
        return await self.transcripts.ingest(message)  # type: ignore[arg-type]


class BlockingVoiceLibraryExecutor:
    """Owned one-worker executor for unavoidable MLX/XTTS blocking calls only.

    It is lazy: no worker thread exists unless a permitted blocking local-model
    operation is actually scheduled.  Cancellation sets a cooperative signal,
    cancels queued work, and never opens a playback subprocess/thread route.
    """

    _PERMITTED_LIBRARIES = frozenset({"mlx-lm", "xtts"})

    def __init__(self) -> None:
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False

    @property
    def max_workers(self) -> int:
        return 1

    async def run(self, *, library: str, operation: BlockingOperation[BlockingResult]) -> BlockingResult:
        if library not in self._PERMITTED_LIBRARIES:
            raise BlockingLibraryError("blocking_executor_restricted_to_local_model_libraries")
        if self._closed:
            raise VoicePipelineUnavailable("blocking_executor_closed")
        if asyncio.current_task() is not None and asyncio.current_task().cancelling():
            raise asyncio.CancelledError

        cancellation = Event()
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="haki-voice-local-model",
            )
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, operation, cancellation)
        try:
            return await future
        except asyncio.CancelledError:
            cancellation.set()
            future.cancel()
            raise

    def close(self) -> None:
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


class VoiceSessionPipeline:
    """One all-or-nothing Pipecat graph for one active ``VoiceSession``.

    ``InputAudioRawFrame`` flows to the VAD endpoint, partial/final
    ``TranscriptionFrame`` values flow through their policy-specific queues,
    and local model/sentence stages explicitly emit ``LLMTextFrame`` then
    ``TTSTextFrame``.  Each path is bounded; only audio and partial UI updates
    may be coalesced.  Final transcript/control work always waits for capacity.
    """

    def __init__(
        self,
        *,
        session: VoiceSession,
        ingress: VoiceIngressProcessors,
        task_factory: PipelineTaskFactory | None = None,
        queue_limits: PipelineQueueLimits | None = None,
        sinks: VoicePipelineSinks | None = None,
        diagnostic_sink: DiagnosticSink | None = None,
        vad_processor: SileroVADSmartTurnProcessor | None = None,
    ) -> None:
        # Lazily set by wire_ipc_server(); exposes the JSONIPCServer to any
        # VoiceToolAdapter that is subsequently wired into this pipeline
        # (Req 5.1, 7.3, 8.1, 13.1).
        self._ipc_server: Any | None = None
        self._tool_adapter: Any | None = None
        if ingress.audio_ring._session is not session or ingress.transcripts._session is not session:
            raise ValueError("ingress processors must belong to the pipeline session")
        self.session = session
        self.ingress = ingress
        self.queue_limits = queue_limits or PipelineQueueLimits()
        self.sinks = sinks or VoicePipelineSinks()
        self._task_factory = task_factory or _create_default_pipeline_task
        self._diagnostic_sink = diagnostic_sink
        self._availability = PipelineAvailability.NEW
        self._pipeline_task: object | None = None
        self._runtime_task: asyncio.Task[None] | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._started = asyncio.Event()
        self._stopping = asyncio.Event()
        self._blocking_executor = BlockingVoiceLibraryExecutor()
        self._audio_queue: asyncio.Queue[TypedVoiceFrame[object]] = asyncio.Queue(
            maxsize=self.queue_limits.audio
        )
        self._partial_queue: asyncio.Queue[TypedVoiceFrame[object]] = asyncio.Queue(
            maxsize=self.queue_limits.partial
        )
        self._control_queue: asyncio.Queue[TypedVoiceFrame[object]] = asyncio.Queue(
            maxsize=self.queue_limits.control
        )
        self._llm_queue: asyncio.Queue[TypedVoiceFrame[object]] = asyncio.Queue(
            maxsize=self.queue_limits.llm
        )
        self._tts_queue: asyncio.Queue[TypedVoiceFrame[object]] = asyncio.Queue(
            maxsize=self.queue_limits.sentence
        )
        self._pcm_queue: asyncio.Queue[TypedVoiceFrame[object]] = asyncio.Queue(
            maxsize=self.queue_limits.pcm
        )
        self._cancellable_tasks: dict[tuple[UUID, int], set[asyncio.Task[object]]] = {}
        self.turn_control = PipelineTurnCoordinator(
            session=self.session,
            sinks=self.sinks,
            drain_interrupted_work=self._cancel_interrupted_work,
            vad_processor=vad_processor,
        )
        self.dropped_audio_frames = 0

    @property
    def availability(self) -> PipelineAvailability:
        return self._availability

    @property
    def is_available(self) -> bool:
        return self._availability is PipelineAvailability.RUNNING

    @property
    def pipeline_task(self) -> object | None:
        """The sole Pipecat ``PipelineTask`` allocated for this session."""
        return self._pipeline_task

    @property
    def queues(self) -> dict[str, asyncio.Queue[TypedVoiceFrame[object]]]:
        """Expose bounded graph queues for diagnostics and focused integration tests."""
        return {
            "audio": self._audio_queue,
            "partial": self._partial_queue,
            "control": self._control_queue,
            "llm": self._llm_queue,
            "tts": self._tts_queue,
            "pcm": self._pcm_queue,
        }

    def wire_ipc_server(self, ipc_server: Any) -> None:
        """Wire a running ``JSONIPCServer`` into this pipeline (Req 5.1, 7.3, 8.1).

        Stores *ipc_server* on this pipeline instance and immediately propagates
        it into any ``VoiceToolAdapter`` already attached via ``_tool_adapter``.
        If no tool adapter is attached yet, the server reference is stored and
        will be picked up the next time ``_tool_adapter`` is set.

        This method is the single authoritative wiring point used by
        ``haki_core_service.py`` to satisfy task 13.1.  It replaces the
        previous ``hasattr(_pipeline, "_tool_adapter")`` guard which silently
        failed because ``VoiceSessionPipeline`` never had a ``_tool_adapter``
        attribute.

        Args:
            ipc_server: Running ``JSONIPCServer`` instance that exposes
                ``broadcast_agent_event`` and ``_connected_writers``.
        """
        self._ipc_server = ipc_server
        if self._tool_adapter is not None:
            self._tool_adapter._ipc_server = ipc_server

    async def start(self) -> None:
        """Atomically create the one Pipecat task and start graph workers.

        A construction or frame-compatibility failure creates no fallback
        runtime.  The pipeline remains unavailable and reports one content-free
        ``pipecat`` failure diagnostic.
        """
        async with self._lifecycle_lock:
            if self._availability is PipelineAvailability.RUNNING:
                return
            if self._availability is PipelineAvailability.CLOSED:
                raise VoicePipelineUnavailable("voice_pipeline_closed")
            if self._availability is PipelineAvailability.UNAVAILABLE:
                raise VoicePipelineUnavailable("voice_pipeline_unavailable")

            try:
                self._validate_frame_adapter()
                task = self._task_factory()
                if task is None:
                    raise PipelineInitializationError("pipecat_pipeline_task_missing")
                self._pipeline_task = task
                self._runtime_task = asyncio.get_running_loop().create_task(
                    self._run_graph(), name=f"voice-pipeline-{self.session.session_id}"
                )
                await self._started.wait()
                if self._runtime_task.done():
                    await self._runtime_task
                self._availability = PipelineAvailability.RUNNING
            except BaseException as exc:
                await self._stop_runtime_task()
                self._pipeline_task = None
                self._availability = PipelineAvailability.UNAVAILABLE
                await self._report_pipecat_failure(exc)
                if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                    raise
                raise PipelineInitializationError("pipecat_initialization_failed") from exc

    async def close(self) -> None:
        """Cancel the owned task group and release only local-model resources."""
        async with self._lifecycle_lock:
            if self._availability is PipelineAvailability.CLOSED:
                return
            self._stopping.set()
            await self._stop_runtime_task()
            self._blocking_executor.close()
            await self.session.close()
            self._availability = PipelineAvailability.CLOSED

    async def ingest_ring_slot(self, descriptor: RingSlotDescriptor) -> RingIngressResult:
        """Route a mandatory ``InputAudioRawFrame`` into the bounded VAD path."""
        self._require_running()
        result = await self.ingress.ingest_ring_slot(descriptor)
        await self.turn_control.process_audio(result.frame)
        await self._put_latest(self._audio_queue, result.frame, count_audio_drop=True)
        return result

    async def ingest_transcript_message(self, message: object) -> TranscriptIngressResult:
        """Route sequenced ``TranscriptionFrame`` values by their drop policy."""
        self._require_running()
        result = await self.ingress.ingest_transcript_message(message)
        if result.frame is not None:
            await self.turn_control.process_transcription(result.frame)
            queue = TurnQueueName.CONTROL if result.frame.metadata.is_final else TurnQueueName.PARTIAL
            await self._transfer_from_turn_queue(result.frame, queue)
        return result

    async def submit_final_transcript(self, turn: VoiceTurnRequest) -> None:
        """Direct async ingress for an already validated final local transcript.

        UDS callers should use ``ingest_transcript_message``.  This method
        exists for in-process replacement composition and never accepts audio.
        """
        self._require_running()
        if turn.session_id != self.session.session_id or not turn.text.strip():
            raise ValueError("final transcript must target this session and contain text")
        try:
            record = self.session.turns.get(turn.turn_id)
        except UnknownTurnError:
            try:
                record = await self.session.start_turn(turn.turn_id)
            except DuplicateTurnError:
                record = self.session.turns.get(turn.turn_id)
        metadata = TranscriptionFrameMetadata(
            session_id=self.session.session_id,
            turn_id=turn.turn_id,
            sequence=0 if record.next_sequence is None else record.next_sequence,
            cancellation_generation=record.cancellation_generation,
            event_seq=0 if record.next_sequence is None else record.next_sequence,
            is_final=True,
            language=turn.language,
            capture_started_monotonic_ns=0,
            capture_ended_monotonic_ns=0,
        )
        payload = self._frame_adapter().create_transcription_frame(text=turn.text)
        frame = TypedVoiceFrame(VoiceFrameType.TRANSCRIPTION, metadata, payload)
        await self.session.accept_frame(frame, queue=TurnQueueName.CONTROL)
        await self.turn_control.process_transcription(frame)
        await self._transfer_from_turn_queue(frame, TurnQueueName.CONTROL)

    async def emit_llm_text(self, *, turn_id: UUID, sequence: int, text: str) -> TypedVoiceFrame[object]:
        """Place one ordered mandatory ``LLMTextFrame`` into the graph."""
        self._require_running()
        if not text.strip():
            raise ValueError("LLM text frame must not be empty")
        record = self.session.turns.get(turn_id)
        metadata = VoiceFrameMetadata(
            session_id=self.session.session_id,
            turn_id=turn_id,
            sequence=sequence,
            cancellation_generation=record.cancellation_generation,
        )
        frame = TypedVoiceFrame(
            VoiceFrameType.LLM_TEXT,
            metadata,
            self._frame_adapter().create_llm_text_frame(text=text),
        )
        await self.session.accept_frame(frame, queue=TurnQueueName.LLM)
        await self._transfer_from_turn_queue(frame, TurnQueueName.LLM)
        return frame

    async def emit_tts_text(
        self,
        *,
        turn_id: UUID,
        sentence_id: UUID,
        sequence: int,
        text: str,
    ) -> TypedVoiceFrame[object]:
        """Place one ordered mandatory sentence-ready ``TTSTextFrame`` into the graph."""
        self._require_running()
        if not text.strip():
            raise ValueError("TTS text frame must not be empty")
        record = self.session.turns.get(turn_id)
        metadata = SentenceFrameMetadata(
            session_id=self.session.session_id,
            turn_id=turn_id,
            sequence=sequence,
            cancellation_generation=record.cancellation_generation,
            sentence_id=sentence_id,
        )
        frame = TypedVoiceFrame(
            VoiceFrameType.TTS_TEXT,
            metadata,
            self._frame_adapter().create_tts_text_frame(text=text),
        )
        await self.session.accept_frame(frame, queue=TurnQueueName.SENTENCE)
        if not await self.turn_control.register_tts_sentence(frame, text=text):
            raise LateFrameRejected("tts sentence generation is no longer current")
        await self._transfer_from_turn_queue(frame, TurnQueueName.SENTENCE)
        return frame

    async def emit_pcm_chunk(
        self,
        *,
        turn_id: UUID,
        sentence_id: UUID,
        sequence: int,
        chunk_sequence: int,
        pcm: bytes,
    ) -> TypedVoiceFrame[bytes]:
        """Queue current-generation PCM so cancellation can drain it before rendering."""
        self._require_running()
        if not pcm:
            raise ValueError("PCM chunk must not be empty")
        record = self.session.turns.get(turn_id)
        frame = TypedVoiceFrame(
            VoiceFrameType.PCM_CHUNK,
            PCMChunkFrameMetadata(
                session_id=self.session.session_id,
                turn_id=turn_id,
                sequence=sequence,
                cancellation_generation=record.cancellation_generation,
                sentence_id=sentence_id,
                chunk_sequence=chunk_sequence,
            ),
            pcm,
        )
        await self.session.accept_frame(frame, queue=TurnQueueName.PCM)
        await self._transfer_from_turn_queue(frame, TurnQueueName.PCM)
        return frame

    def register_cancellable_task(
        self,
        *,
        turn_id: UUID,
        generation: int,
        task: asyncio.Task[object],
    ) -> None:
        """Track LLM/synthesis work so a barge-in can cancel it without waiting."""
        key = (turn_id, generation)
        tasks = self._cancellable_tasks.setdefault(key, set())
        tasks.add(task)

        def discard(completed: asyncio.Task[object]) -> None:
            pending = self._cancellable_tasks.get(key)
            if pending is not None:
                pending.discard(completed)
                if not pending:
                    self._cancellable_tasks.pop(key, None)

        task.add_done_callback(discard)

    async def process_playback_event(
        self,
        message: Mapping[str, object],
        *,
        playback_completed_monotonic_ns: int | None = None,
    ) -> bool:
        """Write only normal renderer confirmations to the session ledger."""
        self._require_running()
        return await self.turn_control.process_playback_event(
            message,
            playback_completed_monotonic_ns=playback_completed_monotonic_ns,
        )

    async def run_blocking_library(
        self, *, library: str, operation: BlockingOperation[BlockingResult]
    ) -> BlockingResult:
        """Run an unavoidable local MLX/XTTS call on the owned one-worker executor."""
        self._require_running()
        return await self._blocking_executor.run(library=library, operation=operation)

    async def _run_graph(self) -> None:
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(self._dispatch(self._audio_queue, self.sinks.input_audio))
                group.create_task(self._dispatch(self._partial_queue, self.sinks.partial_transcription))
                group.create_task(self._dispatch(self._control_queue, self.sinks.final_transcription))
                group.create_task(self._dispatch(self._llm_queue, self.sinks.llm_text))
                group.create_task(self._dispatch(self._tts_queue, self.sinks.tts_text))
                group.create_task(self._dispatch(self._pcm_queue, self.sinks.pcm))
                group.create_task(self._stop_task_group_when_requested())
                self._started.set()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            if _is_pipeline_stop(exc):
                return
            self._availability = PipelineAvailability.UNAVAILABLE
            await self._report_pipecat_failure(exc)
            raise

    async def _stop_task_group_when_requested(self) -> None:
        await self._stopping.wait()
        raise _PipelineStop()

    async def _dispatch(
        self,
        queue: asyncio.Queue[TypedVoiceFrame[object]],
        sink: FrameSink | None,
    ) -> None:
        while True:
            frame = await queue.get()
            try:
                if frame.frame_type in {
                    VoiceFrameType.LLM_TEXT,
                    VoiceFrameType.TTS_TEXT,
                    VoiceFrameType.PCM_CHUNK,
                } and not await self.session.output_is_current(frame):
                    continue
                if sink is not None:
                    result = sink(frame)
                    if inspect.isawaitable(result):
                        await result
            finally:
                queue.task_done()

    async def _cancel_interrupted_work(self, turn_id: UUID, generation: int) -> None:
        """Cancel matching LLM/TTS tasks and purge queued stale output immediately."""
        for task in tuple(self._cancellable_tasks.pop((turn_id, generation), set())):
            task.cancel()
        for queue in (self._llm_queue, self._tts_queue, self._pcm_queue):
            retained: list[TypedVoiceFrame[object]] = []
            while not queue.empty():
                frame = queue.get_nowait()
                queue.task_done()
                metadata = frame.metadata
                if not (
                    metadata.turn_id == turn_id
                    and metadata.cancellation_generation == generation
                ):
                    retained.append(frame)
            for frame in retained:
                queue.put_nowait(frame)

    async def _transfer_from_turn_queue(
        self,
        accepted: TypedVoiceFrame[object],
        queue_name: TurnQueueName,
    ) -> None:
        """Move a session-validated frame into its graph queue immediately.

        The session queue performs per-turn sequencing/policy validation.  This
        transfer prevents a passive per-turn queue from becoming an unconsumed
        second buffer while retaining non-droppable control and ordered output
        backpressure at the graph boundary.
        """
        record = self.session.turns.get(accepted.metadata.turn_id)
        turn_queue = record.queues.queue_for(queue_name)
        try:
            frame = turn_queue.get_nowait()
        except asyncio.QueueEmpty as exc:
            raise VoicePipelineUnavailable("accepted_turn_frame_missing") from exc
        turn_queue.task_done()
        if queue_name is TurnQueueName.PARTIAL:
            await self._put_latest(self._partial_queue, frame)
        elif queue_name is TurnQueueName.CONTROL:
            await self._control_queue.put(frame)
        elif queue_name is TurnQueueName.LLM:
            await self._llm_queue.put(frame)
        elif queue_name is TurnQueueName.SENTENCE:
            await self._tts_queue.put(frame)
        else:  # pragma: no cover - the graph never routes PCM through this helper.
            await self._pcm_queue.put(frame)

    async def _put_latest(
        self,
        queue: asyncio.Queue[TypedVoiceFrame[object]],
        frame: TypedVoiceFrame[object],
        *,
        count_audio_drop: bool = False,
    ) -> None:
        while queue.full():
            queue.get_nowait()
            queue.task_done()
            if count_audio_drop:
                self.dropped_audio_frames += 1
        queue.put_nowait(frame)

    def _frame_adapter(self) -> "PipecatGraphFrameFactory":
        frame_adapter = self.ingress.frame_adapter
        if not isinstance(frame_adapter, PipecatGraphFrameFactory):
            raise PipecatFrameAdapterUnavailable("pipecat_text_frame_factory_unavailable")
        return frame_adapter

    def _validate_frame_adapter(self) -> None:
        frame_adapter = self._frame_adapter()
        if isinstance(frame_adapter, PipecatFrameAdapter):
            frame_adapter.validate_mandatory_frames()

    async def _stop_runtime_task(self) -> None:
        if self._runtime_task is None:
            return
        task = self._runtime_task
        self._runtime_task = None
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._started.clear()

    async def _report_pipecat_failure(self, error: BaseException) -> None:
        if self._diagnostic_sink is None:
            return
        result = self._diagnostic_sink(
            PipelineDiagnostic(stage="pipecat", outcome="failed", error_class=type(error).__name__)
        )
        if inspect.isawaitable(result):
            await result

    def _require_running(self) -> None:
        if self._availability is not PipelineAvailability.RUNNING:
            raise VoicePipelineUnavailable("voice_pipeline_unavailable")


@runtime_checkable
class PipecatGraphFrameFactory(PipecatIngressFrameFactory, Protocol):
    """Pipecat compatibility port for all mandatory frames in this graph."""

    def create_llm_text_frame(self, *, text: str) -> object:
        """Create an ``LLMTextFrame``."""

    def create_tts_text_frame(self, *, text: str) -> object:
        """Create a ``TTSTextFrame``."""


async def _invoke_callback(
    callback: Callable[..., Awaitable[None] | None] | None,
    *args: object,
) -> None:
    if callback is None:
        return
    result = callback(*args)
    if inspect.isawaitable(result):
        await result


class _PipelineStop(Exception):
    """Private signal that exits a TaskGroup and cancels all queue workers."""


def _is_pipeline_stop(error: BaseException) -> bool:
    """Recognize the exception group emitted when normal shutdown cancels workers."""
    if isinstance(error, _PipelineStop):
        return True
    nested = getattr(error, "exceptions", None)
    return isinstance(nested, tuple) and bool(nested) and all(
        _is_pipeline_stop(item) for item in nested
    )


def _create_default_pipeline_task() -> object:
    """Construct one empty Pipecat task; processors are represented by graph sinks.

    This adapter deliberately does not use a runner, thread, subprocess, or
    alternative playback engine.  The task object is retained for the active
    session and later processor tasks extend the same graph rather than create
    additional replacement pipelines.
    """
    try:
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.task import PipelineTask
    except ImportError as exc:
        raise PipecatFrameAdapterUnavailable("pipecat_pipeline_task_unavailable") from exc
    try:
        return PipelineTask(Pipeline([]))
    except (TypeError, ValueError) as exc:
        raise PipecatFrameAdapterUnavailable("pipecat_pipeline_task_constructor_invalid") from exc


__all__ = [
    "BlockingLibraryError",
    "BlockingVoiceLibraryExecutor",
    "PipecatFrameAdapter",
    "PipecatFrameAdapterUnavailable",
    "PipelineAvailability",
    "PipelineDiagnostic",
    "PipelineInitializationError",
    "PipelineQueueLimits",
    "PipelineTurnCoordinator",
    "VoiceIngressProcessors",
    "VoicePipelineSinks",
    "VoicePipelineUnavailable",
    "VoiceSessionPipeline",
]
