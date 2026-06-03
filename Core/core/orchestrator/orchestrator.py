"""
Orchestrator — central router for a conversational turn.

Sequence:
    Voice_Engine (IPC) → parallel(Mood_Detector, Language_Engine, Memory_Brain)
    → intent classification (Model_Provider LLM)
    → capability dispatch
    → Persona_Engine shaping
    → TTS token stream → IPC layer

Every await point is cancellable via asyncio.CancelledError so that a barge-in
or explicit cancel() call can abort the turn at any stage.  The parallel
enrichment phase uses asyncio.gather with return_exceptions=True; if any
individual task fails or times out, the orchestrator proceeds with whatever
results are available (Requirement 6.5).

Memory retrieval is bounded by a 2-second timeout (asyncio.wait_for) as
required by Requirement 7.3.

Design: The Orchestrator, Intent Routing.
Requirements: 3.1, 4.8, 5.1, 6.5.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, AsyncIterator, TYPE_CHECKING

from core.dialogue import DialogueManager
from core.language import LanguageEngine, UninterpretableInputError
from core.model_provider import Capability, ModelProviderRegistry, StubModelProvider
from core.mood import MoodDetector, MoodResult
from core.persona import PersonaContext, PersonaEngine

if TYPE_CHECKING:
    from core.orchestrator.intent_router import IntentRouter, IntentResult

logger = logging.getLogger(__name__)

# Maximum time (seconds) allowed for Memory_Brain retrieval (Req 7.3).
MEMORY_TIMEOUT_SECS: float = 2.0


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------


class Intent(str, Enum):
    """Supported intent classifications produced by the LLM router."""

    CHAT = "chat"
    RECALL = "recall"
    REMEMBER = "remember"
    READ_ALOUD = "read_aloud"
    MAC_COMMAND = "mac_command"
    RUN_AUTOMATION = "run_automation"
    IMAGE = "image"
    SCHEDULE = "schedule"
    TASK = "task"
    META = "meta"
    UNKNOWN = "unknown"


# Mapping from LLM-returned string labels to Intent values.
_INTENT_LABEL_MAP: dict[str, Intent] = {i.value: i for i in Intent}


# ---------------------------------------------------------------------------
# Turn context
# ---------------------------------------------------------------------------


@dataclass
class TurnContext:
    """Snapshot of all inputs assembled for a single conversational turn."""

    transcript: str
    audio_features: dict[str, Any] = field(default_factory=dict)

    # Populated by parallel enrichment phase.
    mood: MoodResult | None = None
    language_composition: str | None = None
    language_constraints: dict[str, Any] = field(default_factory=dict)
    memory_context: list[Any] = field(default_factory=list)

    # Populated by intent routing phase.
    intent: Intent = Intent.UNKNOWN


# ---------------------------------------------------------------------------
# Memory_Brain stub (placeholder until Task 7.x implements the real brain)
# ---------------------------------------------------------------------------


class _MemoryBrainStub:
    """
    Placeholder for the Memory_Brain subsystem.

    Returns an empty list for every retrieval query and logs a debug
    message.  The Orchestrator treats an empty list as "memory unavailable"
    and proceeds without it (Req 6.5).
    """

    async def retrieve(self, query: str, k: int = 5) -> list[Any]:
        logger.debug("MemoryBrainStub.retrieve(query=%r, k=%d) → []", query, k)
        return []


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """
    Central router for a conversational turn.

    Wires together:
    - Mood_Detector (prosodic mood classification — Req 4.8)
    - Language_Engine (composition analysis — Req 5.1)
    - Memory_Brain stub (relevant-context retrieval — Req 7.3 / 6.5)
    - Model_Provider LLM (intent routing)
    - Capability dispatcher (stub — future tasks add real handlers)
    - Persona_Engine (tone shaping — Req 6.5)

    Turn lifecycle
    --------------
    1. ``run_turn()`` is a coroutine — schedule it as an asyncio Task so that
       ``cancel()`` can cancel the task at any await point.
    2. ``cancel()`` cancels the underlying asyncio Task (if one is running) *and*
       sets a cancel event so that poll loops also abort cleanly.

    Barge-in integration
    --------------------
    The IPC server should call ``orchestrator.cancel()`` on a BARGE_IN or
    CANCEL control event, then immediately call ``run_turn()`` with the new
    transcript once the cancelled task has been awaited / cleaned up.

    Parameters
    ----------
    mood_detector : MoodDetector | None
        Live Mood_Detector instance.  A default (stub-backed) instance is
        created when None is supplied.
    language_engine : LanguageEngine | None
        Live Language_Engine instance.  A default instance is created when None.
    memory_brain : object | None
        Memory_Brain instance with an ``async retrieve(query, k)`` method.
        A stub is used when None is supplied.
    persona_engine : PersonaEngine | None
        Live Persona_Engine instance.  A default instance is created when None.
    dialogue_manager : DialogueManager | None
        Live Dialogue_Manager instance for ambiguity detection.
    llm_provider : object | None
        Model Provider for the LLM capability (intent classification).  A
        StubModelProvider is used when None is supplied.
    """

    def __init__(
        self,
        mood_detector: MoodDetector | None = None,
        language_engine: LanguageEngine | None = None,
        memory_brain: Any | None = None,
        persona_engine: PersonaEngine | None = None,
        dialogue_manager: DialogueManager | None = None,
        llm_provider: Any | None = None,
        intent_router: "IntentRouter | None" = None,
    ) -> None:
        self._mood_detector: MoodDetector = mood_detector or MoodDetector()
        self._language_engine: LanguageEngine = language_engine or LanguageEngine()
        self._memory_brain: Any = memory_brain or _MemoryBrainStub()
        self._persona_engine: PersonaEngine = persona_engine or PersonaEngine()
        self._dialogue_manager: DialogueManager = dialogue_manager or DialogueManager()

        if llm_provider is None:
            _registry = ModelProviderRegistry()
            self._llm_provider: Any = StubModelProvider(Capability.LLM, _registry)
        else:
            self._llm_provider = llm_provider

        # Lazily-imported IntentRouter to avoid circular import at module level.
        if intent_router is None:
            from core.orchestrator.intent_router import IntentRouter as _IR  # noqa: PLC0415
            self._intent_router: "IntentRouter" = _IR(
                dialogue_manager=self._dialogue_manager,
                llm_provider=self._llm_provider,
            )
        else:
            self._intent_router = intent_router

        # The most recent IntentResult (populated after classify()).
        # Stored so tests and subsystems can inspect the classification.
        self._last_intent_result: "IntentResult | None" = None

        # Cancellation state.
        self._cancel_event: asyncio.Event = asyncio.Event()
        # The asyncio.Task wrapping the current run_turn coroutine, if any.
        self._current_task: asyncio.Task[Any] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_turn(
        self,
        transcript: str,
        audio_features: dict[str, Any] | None = None,
    ) -> str:
        """
        Execute one conversational turn end-to-end and return shaped text.

        The coroutine is cancellable at every ``await`` point — an
        ``asyncio.CancelledError`` from any phase propagates upward,
        letting the caller (or the IPC layer) handle barge-in.

        Parameters
        ----------
        transcript : str
            Final STT transcript for the user's utterance.
        audio_features : dict | None
            Prosodic features extracted by the Voice_Engine alongside the
            transcript (pitch, volume, …), forwarded to Mood_Detector.

        Returns
        -------
        str
            Shaped response text, ready to stream to the TTS subsystem.
        """
        # Reset cancellation flag for this fresh turn.
        self._cancel_event.clear()

        ctx = TurnContext(
            transcript=transcript,
            audio_features=audio_features or {},
        )

        # ----------------------------------------------------------------
        # Phase 1 — Parallel enrichment: Mood | Language | Memory
        # Each sub-coroutine is independently awaitable; if one raises or
        # times out, return_exceptions=True ensures the others continue.
        # After gather, we unpack and proceed with whatever is available
        # (Requirement 6.5).
        # ----------------------------------------------------------------
        logger.debug("Turn [%r]: starting parallel enrichment", transcript[:40])

        results = await asyncio.gather(
            self._classify_mood(ctx),
            self._detect_language(ctx),
            self._retrieve_memory(ctx),
            return_exceptions=True,
        )

        mood_result, lang_result, mem_result = results

        # Unpack mood (Req 4.8)
        if isinstance(mood_result, BaseException):
            logger.warning("Mood classification failed: %r — proceeding without it", mood_result)
            ctx.mood = None
        else:
            ctx.mood = mood_result  # MoodResult | None

        # Unpack language composition (Req 5.1)
        if isinstance(lang_result, BaseException):
            logger.warning(
                "Language detection failed: %r — proceeding without it", lang_result
            )
            ctx.language_composition = None
            ctx.language_constraints = {}
        else:
            ctx.language_composition, ctx.language_constraints = lang_result  # type: ignore[misc]

        # Unpack memory context (Req 6.5, 7.3)
        if isinstance(mem_result, BaseException):
            logger.warning(
                "Memory retrieval failed: %r — proceeding without it", mem_result
            )
            ctx.memory_context = []
        else:
            ctx.memory_context = mem_result or []  # type: ignore[assignment]

        # Cancellation checkpoint — honour barge-in after parallel phase.
        await self._checkpoint()

        # ----------------------------------------------------------------
        # Phase 2 — Intent classification via LLM (Model_Provider)
        # ----------------------------------------------------------------
        logger.debug("Turn [%r]: routing intent", transcript[:40])
        ctx.intent = await self._route_intent(ctx)

        await self._checkpoint()

        # ----------------------------------------------------------------
        # Phase 3 — Capability dispatch
        # ----------------------------------------------------------------
        logger.debug("Turn [%r]: dispatching to capability (intent=%s)", transcript[:40], ctx.intent)
        raw_response = await self._dispatch(ctx)

        await self._checkpoint()

        # ----------------------------------------------------------------
        # Phase 4 — Persona shaping (Req 6.5 — proceeds with whatever
        # mood / memory is available)
        # ----------------------------------------------------------------
        logger.debug("Turn [%r]: shaping response via PersonaEngine", transcript[:40])
        shaped = await self._shape_response(raw_response, ctx)

        return shaped

    async def stream_turn(
        self,
        transcript: str,
        audio_features: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """
        Execute one conversational turn and yield response tokens one at a
        time so the IPC layer can stream them to the TTS subsystem
        (Requirement 3.1 — first audio ≤ 300 ms).

        This generator wraps ``run_turn``; a real implementation would
        stream tokens directly from the LLM via ``invoke_stream`` and yield
        each token as it arrives.  The current stub yields the full shaped
        response as a single token.

        Yields
        ------
        str
            Individual response tokens (words / subword pieces).
        """
        shaped = await self.run_turn(transcript, audio_features)
        # Yield word-by-word so TTS can start playback immediately
        # (Req 3.1 — streaming playback of earliest available words).
        for word in shaped.split():
            yield word
            # Cancellation checkpoint between tokens.
            await self._checkpoint()

    def cancel(self) -> None:
        """
        Signal cancellation for barge-in or explicit user stop.

        Sets an internal cancel event (checked at every checkpoint) and
        cancels the asyncio Task that is currently running ``run_turn``
        or ``stream_turn``, if any.  If no task is running this is a no-op.
        """
        self._cancel_event.set()
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()
            logger.debug("Orchestrator.cancel(): cancelled current turn task")

    def set_current_task(self, task: asyncio.Task[Any]) -> None:
        """
        Register the asyncio Task wrapping the current ``run_turn`` call.

        The IPC layer should call this immediately after scheduling the
        coroutine as a Task so that ``cancel()`` can cancel it.
        """
        self._current_task = task

    # ------------------------------------------------------------------
    # Private: parallel enrichment helpers
    # ------------------------------------------------------------------

    async def _classify_mood(self, ctx: TurnContext) -> MoodResult | None:
        """
        Classify mood from audio features via Mood_Detector (Req 4.8).

        Returns None when audio_features is missing the minimum duration
        key or when the detector returns unclassifiable.
        """
        duration_ms: float = float(ctx.audio_features.get("duration_ms", 0.0))
        try:
            result: MoodResult = self._mood_detector.classify(
                ctx.audio_features, duration_ms
            )
            return result
        except Exception as exc:
            logger.warning("_classify_mood raised %r", exc)
            raise

    async def _detect_language(
        self, ctx: TurnContext
    ) -> tuple[str | None, dict[str, Any]]:
        """
        Detect language composition and generate LLM constraints (Req 5.1).

        Returns a tuple of (composition_label, constraints_dict).
        When the input is uninterpretable, logs the event and returns
        (None, {}) so the caller can proceed without language constraints.
        """
        try:
            analysis = self._language_engine.analyze(ctx.transcript)
            constraints = self._language_engine.generate_constraints(analysis)
            return analysis.composition, constraints
        except UninterpretableInputError as exc:
            logger.info("Language uninterpretable for transcript %r: %s", ctx.transcript[:40], exc)
            # Req 5.1 / 5.5: language engine cannot interpret → caller should
            # ask user to rephrase, but we still return gracefully so the
            # enrichment gather does not block the other tasks.
            return None, {}
        except Exception as exc:
            logger.warning("_detect_language raised %r", exc)
            raise

    async def _retrieve_memory(self, ctx: TurnContext) -> list[Any]:
        """
        Query Memory_Brain for relevant context, bounded by a 2-second
        timeout (Requirement 7.3).  Returns an empty list on timeout or error.

        Dispatch strategy (backward-compatible):
        1. If the brain has an ``aretrieve`` method (real MemoryBrain), call
           it — it wraps the synchronous ``retrieve()`` in a thread executor.
        2. Otherwise, call ``retrieve()`` directly (legacy stubs that already
           expose an ``async def retrieve()`` coroutine function).
        """
        import inspect  # noqa: PLC0415 — import here to keep module-level imports minimal

        try:
            if hasattr(self._memory_brain, "aretrieve"):
                coro = self._memory_brain.aretrieve(ctx.transcript)
            elif inspect.iscoroutinefunction(self._memory_brain.retrieve):
                coro = self._memory_brain.retrieve(ctx.transcript)
            else:
                # Synchronous retrieve() — run in thread executor to avoid
                # blocking the event loop.
                loop = asyncio.get_event_loop()
                coro = loop.run_in_executor(None, self._memory_brain.retrieve, ctx.transcript)

            notes: list[Any] = await asyncio.wait_for(
                coro,
                timeout=MEMORY_TIMEOUT_SECS,
            )
            return notes
        except asyncio.TimeoutError:
            logger.info("Memory retrieval timed out after %.1fs — proceeding without it", MEMORY_TIMEOUT_SECS)
            return []
        except Exception as exc:
            logger.warning("_retrieve_memory raised %r", exc)
            raise

    # ------------------------------------------------------------------
    # Private: intent routing
    # ------------------------------------------------------------------

    async def _route_intent(self, ctx: TurnContext) -> Intent:
        """
        Classify the user's intent via IntentRouter.classify() (Req 6.1).

        Passes language_result from the enriched TurnContext so the router
        can leverage it for more accurate classification.  The result is
        stored in ``self._last_intent_result`` for downstream inspection.
        Falls back to Intent.CHAT when the router cannot return a valid label.
        """
        from core.orchestrator.intent_router import IntentResult  # noqa: PLC0415

        # Build the language_result dict that classify() expects.
        language_result: dict[str, Any] | None = None
        if ctx.language_composition is not None:
            language_result = {
                "composition": ctx.language_composition,
                **ctx.language_constraints,
            }

        try:
            intent_result: IntentResult = await self._intent_router.classify(
                ctx.transcript,
                language_result=language_result,
            )
            self._last_intent_result = intent_result
            return intent_result.intent
        except Exception as exc:
            logger.warning("Intent routing via IntentRouter failed: %r — defaulting to CHAT", exc)
            return Intent.CHAT

    @staticmethod
    def _build_routing_prompt(ctx: TurnContext) -> str:
        """Construct the intent-classification prompt from TurnContext."""
        lang_hint = (
            f" [language: {ctx.language_composition}]"
            if ctx.language_composition
            else ""
        )
        return (
            f"Classify the user intent for the following transcript{lang_hint}.\n"
            f"Return exactly one label from: "
            f"chat, recall, remember, read_aloud, mac_command, run_automation, "
            f"image, schedule, task, meta, unknown.\n"
            f"Transcript: {ctx.transcript!r}"
        )

    @staticmethod
    def _parse_intent(raw: Any) -> Intent:
        """
        Extract an Intent label from a (potentially stub) LLM response dict.

        Falls back to Intent.CHAT when no recognizable label is found.
        """
        if isinstance(raw, dict):
            # A real LLM would embed the label in "text" or "content".
            candidate = (
                raw.get("intent")
                or raw.get("text")
                or raw.get("content")
                or ""
            )
            if isinstance(candidate, str):
                # Strip surrounding whitespace / punctuation and lower-case.
                label = candidate.strip().strip('"').lower()
                if label in _INTENT_LABEL_MAP:
                    return _INTENT_LABEL_MAP[label]
        elif isinstance(raw, str):
            label = raw.strip().lower()
            if label in _INTENT_LABEL_MAP:
                return _INTENT_LABEL_MAP[label]
        return Intent.CHAT

    # ------------------------------------------------------------------
    # Private: capability dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, ctx: TurnContext) -> str:
        """
        Route the turn to the owning capability subsystem via IntentRouter.route().

        IntentRouter.route() is an async generator that handles the
        DialogueManager gate for side-effecting intents internally.  If
        required slots are missing it yields a clarification message rather
        than executing the capability (Req 23.1).

        All handlers are stubs for now; future tasks replace each with a real
        subsystem call.
        """
        from core.orchestrator.intent_router import IntentResult  # noqa: PLC0415

        # Reconstruct an IntentResult from the stored classification or build
        # a minimal one from ctx.intent for backwards compatibility.
        if self._last_intent_result is not None:
            intent_result: IntentResult = self._last_intent_result
        else:
            intent_result = IntentResult(intent=ctx.intent)

        # Collect all chunks from the async generator returned by route().
        chunks: list[str] = []
        async for chunk in self._intent_router.route(intent_result, ctx):
            chunks.append(chunk)
        return "".join(chunks)

    async def _handle_chat(self, ctx: TurnContext) -> str:
        """
        Chat / general-response handler stub.

        A real implementation calls the LLM with the composed system prompt
        (language constraints + persona + memory) and streams tokens back.
        """
        lang_instruction = ctx.language_constraints.get("language_instruction", "")
        memory_summary = (
            f"[{len(ctx.memory_context)} memory notes available]"
            if ctx.memory_context
            else "[no memory context]"
        )
        return (
            f"[LLM stub — chat] "
            f"Transcript: {ctx.transcript!r} | "
            f"{lang_instruction} | {memory_summary}"
        )

    async def _handle_recall(self, ctx: TurnContext) -> str:
        """Recall handler — surfaces memory notes related to the transcript."""
        if ctx.memory_context:
            notes_text = "; ".join(str(n) for n in ctx.memory_context)
            return f"Here is what I remember: {notes_text}"
        return "I don't have any relevant memories for that."

    async def _handle_remember(self, ctx: TurnContext) -> str:
        """Remember handler stub — would persist new information."""
        return f"[Memory_Brain stub] Would store: {ctx.transcript!r}"

    # ------------------------------------------------------------------
    # Private: persona shaping
    # ------------------------------------------------------------------

    async def _shape_response(self, response: str, ctx: TurnContext) -> str:
        """
        Shape *response* through the Persona_Engine using whatever mood and
        memory context is available (Requirement 6.5).
        """
        memory_context_str: str | None = None
        if ctx.memory_context:
            memory_context_str = "\n".join(str(note) for note in ctx.memory_context)

        persona_ctx = PersonaContext(
            mood_result=ctx.mood,
            memory_context=memory_context_str,
        )

        # PersonaEngine.shape() is synchronous but wrapped here in a
        # coroutine so the caller can await it uniformly.
        shaped: str = self._persona_engine.shape(response, persona_ctx)
        return shaped

    # ------------------------------------------------------------------
    # Private: cancellation checkpoint
    # ------------------------------------------------------------------

    async def _checkpoint(self) -> None:
        """
        Yield control to the event loop and raise CancelledError if the
        cancel event has been set (barge-in / explicit stop).

        Using a short asyncio.sleep(0) ensures pending cancellations from
        Task.cancel() are delivered even if the orchestrator is in a tight
        synchronous section.
        """
        await asyncio.sleep(0)
        if self._cancel_event.is_set():
            raise asyncio.CancelledError(
                "Turn cancelled by barge-in or user request."
            )
