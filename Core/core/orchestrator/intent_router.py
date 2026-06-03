"""
IntentRouter — intent classification and capability routing.

Classifies each conversational turn into one of the nine HAKI intents
and routes execution to the correct capability handler stub.
Side-effecting intents pass through the DialogueManager gate before
any execution begins; read-only / conversational intents proceed
directly.

Design: Intent Routing.
Requirements: 6.1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable

from core.dialogue import DialogueManager, SlotFillResult
from core.model_provider import Capability, ModelProviderRegistry, StubModelProvider
from core.orchestrator.orchestrator import Intent, TurnContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IntentResult — the output of classify()
# ---------------------------------------------------------------------------


@dataclass
class IntentResult:
    """
    The result of intent classification for a single conversational turn.

    Attributes
    ----------
    intent : Intent
        The classified intent.
    confidence : float
        Confidence score in [0.0, 1.0].  1.0 for now (stub LLM); real
        backends may return calibrated scores.
    raw_label : str
        The raw label string returned by the LLM before parsing.  Useful
        for debugging and future logging.
    language_hint : str | None
        The language composition detected for this turn, if available.
    """

    intent: Intent
    confidence: float = 1.0
    raw_label: str = ""
    language_hint: str | None = None


# ---------------------------------------------------------------------------
# Type alias for capability handlers
# ---------------------------------------------------------------------------

# A capability handler is an async generator that accepts a TurnContext
# and yields zero or more response tokens / chunks.
CapabilityHandler = Callable[[TurnContext], AsyncGenerator[str, None]]


# ---------------------------------------------------------------------------
# Intent → required slots (for DialogueManager gate on side-effecting intents)
# ---------------------------------------------------------------------------

# Side-effecting intents — must pass through the DialogueManager gate.
_SIDE_EFFECTING_INTENTS: frozenset[Intent] = frozenset(
    [
        Intent.MAC_COMMAND,
        Intent.SCHEDULE,
        Intent.TASK,
        Intent.RUN_AUTOMATION,
    ]
)

# Read-only / conversational intents — proceed without dialogue gating.
_READ_ONLY_INTENTS: frozenset[Intent] = frozenset(
    [
        Intent.CHAT,
        Intent.RECALL,
        Intent.REMEMBER,
        Intent.READ_ALOUD,
        Intent.IMAGE,
        Intent.META,
    ]
)

# Minimum slots required per side-effecting intent.
# An empty list means no pre-execution slot check beyond intent classification.
_REQUIRED_SLOTS: dict[Intent, list[str]] = {
    Intent.MAC_COMMAND: ["command_target"],
    Intent.SCHEDULE: ["event_title", "event_datetime"],
    Intent.TASK: ["task_title"],
    Intent.RUN_AUTOMATION: ["automation_name"],
}


# ---------------------------------------------------------------------------
# Handler stubs (async generators)
# ---------------------------------------------------------------------------


async def _chat_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    TODO: wire Persona_Engine + Memory_Brain for conversational chat (Req 6, 7).
    Passes context to the LLM for a freeform response shaped by personality.
    """
    yield f"[chat stub] '{ctx.transcript}'"


async def _recall_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    TODO: wire Memory_Brain.retrieve() for memory recall queries (Req 7.3, 7.7).
    Searches the vault and returns matching notes as a response.
    """
    yield f"[recall stub] Searching memory for: '{ctx.transcript}'"


async def _remember_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    TODO: wire Memory_Brain.remember() to store new information (Req 7.1).
    Confirms the store to the user only after durable write completes.
    """
    yield f"[remember stub] Storing: '{ctx.transcript}'"


async def _read_aloud_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    TODO: wire Screen_Reader.capture_focused() + Voice_Engine for read-aloud (Req 1).
    Captures frontmost window text and streams it to TTS.
    """
    yield "[read_aloud stub] Reading screen content aloud."


async def _mac_command_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    TODO: wire Mac_Controller planner + Execution_Engine for ad-hoc Mac control (Req 21).
    Generates a CommandPlan and executes it step-by-step with safety gating.
    """
    yield f"[mac_command stub] Executing command: '{ctx.transcript}'"


async def _run_automation_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    TODO: wire Automation_Library.run() to execute a named automation (Req 17–19).
    Resolves the automation name and dispatches to the Execution_Engine.
    """
    yield f"[run_automation stub] Running automation: '{ctx.transcript}'"


async def _image_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    TODO: wire Image_Studio.generate() / Image_Studio.edit() for image tasks (Req 15).
    Generates or edits an image from the voice description and presents the result.
    """
    yield f"[image stub] Generating image for: '{ctx.transcript}'"


async def _schedule_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    TODO: wire Scheduler.propose_event() for calendar event creation (Req 11).
    Proposes a calendar event and waits for explicit user confirmation.
    """
    yield f"[schedule stub] Creating schedule entry for: '{ctx.transcript}'"


async def _task_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    TODO: wire Task_Tracker.add() for task creation (Req 13).
    Persists the task and assigns severity; no partial write on failure.
    """
    yield f"[task stub] Adding task: '{ctx.transcript}'"


async def _meta_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    TODO: wire Clock / Settings / Privacy_Manager for meta requests (Req 2, 9, 14, 20).
    Handles queries about time, settings, permissions, and privacy toggles.
    """
    yield f"[meta stub] Handling meta request: '{ctx.transcript}'"


async def _unknown_handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
    """
    Fallback handler for unclassified intents.
    Routes back to chat as a safe default.
    """
    yield "[unknown stub] Could not classify intent; defaulting to chat."


# ---------------------------------------------------------------------------
# Handler for missing-slot gate (inline async generator)
# ---------------------------------------------------------------------------


async def _missing_slots_handler(
    missing: list[str],
) -> Callable[[TurnContext], AsyncGenerator[str, None]]:
    """Return a handler that informs the user about missing required slots."""

    async def _handler(ctx: TurnContext) -> AsyncGenerator[str, None]:
        missing_str = ", ".join(missing)
        yield (
            f"I need a bit more information before I can do that. "
            f"Missing: {missing_str}. Could you provide those details?"
        )

    return _handler


# ---------------------------------------------------------------------------
# IntentRouter
# ---------------------------------------------------------------------------

# Mapping from intent to its handler stub.
_HANDLER_MAP: dict[Intent, CapabilityHandler] = {
    Intent.CHAT: _chat_handler,
    Intent.RECALL: _recall_handler,
    Intent.REMEMBER: _remember_handler,
    Intent.READ_ALOUD: _read_aloud_handler,
    Intent.MAC_COMMAND: _mac_command_handler,
    Intent.RUN_AUTOMATION: _run_automation_handler,
    Intent.IMAGE: _image_handler,
    Intent.SCHEDULE: _schedule_handler,
    Intent.TASK: _task_handler,
    Intent.META: _meta_handler,
    Intent.UNKNOWN: _unknown_handler,
}

# System prompt template for intent classification.
_CLASSIFY_SYSTEM_PROMPT = """You are the intent classifier for HAKI, a personal AI assistant.
Classify the user's request into exactly one of these intents:
  chat           — general conversation, questions, or statements
  recall         — asking HAKI to recall or look up something it knows
  remember       — asking HAKI to remember or store a piece of information
  read_aloud     — asking HAKI to read on-screen content aloud
  mac_command    — ad-hoc control of the Mac (open app, send message, etc.)
  run_automation — running a named/saved automation
  image          — generating or editing an image
  schedule       — creating a calendar event or reminder
  task           — creating or managing a task
  meta           — time, settings, privacy, or permission queries

Reply with a single intent keyword and nothing else."""


class IntentRouter:
    """
    Classifies a conversational turn into an intent and routes it to the
    appropriate capability handler stub.

    classify()
    ----------
    Uses the LLM (via Model_Provider) with a structured classification prompt
    to map the transcript to one of the nine HAKI intents.  Returns an
    :class:`IntentResult` with the intent and metadata.

    route()
    -------
    Accepts an :class:`IntentResult` and a :class:`TurnContext`, runs the
    DialogueManager gate for side-effecting intents, and returns an async
    generator that yields response chunks for the turn.  If required slots
    are missing, the generator yields an informative clarification message
    instead of executing the capability.

    Side-effecting intents (mac_command, schedule, task, run_automation)
    pass through the DialogueManager gate: ``dialogue_manager.assess()``
    is called with the required slots; if slots are insufficient the
    capability handler is NOT invoked.

    Read-only / conversational intents (chat, recall, remember, read_aloud,
    image, meta) proceed without dialogue gating.

    Requirements: 6.1.
    """

    def __init__(
        self,
        dialogue_manager: DialogueManager | None = None,
        llm_provider: Any | None = None,
    ) -> None:
        """
        Parameters
        ----------
        dialogue_manager:
            Injected :class:`~core.dialogue.DialogueManager` instance used to
            gate side-effecting intents.  If None a default instance is
            created (no memory context).
        llm_provider:
            A :class:`~core.model_provider.ModelProvider` for LLM calls.
            Must support ``.invoke(prompt, ...)`` returning a dict with an
            ``"output"`` key.  If None a :class:`~core.model_provider.StubModelProvider`
            is used (returns ``"chat"`` intent for any input).
        """
        self._dialogue_manager: DialogueManager = dialogue_manager or DialogueManager()

        # Set up a default LLM provider stub if none is supplied.
        if llm_provider is None:
            registry = ModelProviderRegistry()
            self._llm = StubModelProvider(Capability.LLM, registry)
        else:
            self._llm = llm_provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def classify(
        self,
        transcript: str,
        language_result: dict | None = None,
    ) -> IntentResult:
        """
        Classify the transcript into one of the nine HAKI intents.

        Calls the LLM provider (via Model_Provider abstraction) with a
        structured classification prompt.  Falls back to ``Intent.CHAT``
        if the LLM response cannot be mapped to a known intent.

        The call to ``self._llm.invoke()`` is synchronous but wrapped in
        ``asyncio.to_thread`` so it does not block the event loop when a
        real (potentially slow) model backend is in use.

        Parameters
        ----------
        transcript:
            The STT transcript of the user's utterance.
        language_result:
            Optional dict from the Language_Engine (e.g.
            ``{"composition": "hinglish", "tokens": [...]}``).  Used to
            enrich the classification prompt with a language hint.

        Returns
        -------
        IntentResult
            The classified intent along with confidence and raw LLM output.
        """
        # Build the language-aware classification prompt.
        lang_hint = ""
        language_hint: str | None = None
        if language_result and isinstance(language_result, dict):
            composition = language_result.get("composition")
            if composition:
                lang_hint = f" [language: {composition}]"
                language_hint = str(composition)

        prompt = (
            f"{_CLASSIFY_SYSTEM_PROMPT}\n\n"
            f"User{lang_hint}: {transcript}"
        )

        try:
            # Offload to thread so a blocking real LLM backend does not
            # stall the async event loop.
            import asyncio
            result = await asyncio.to_thread(self._llm.invoke, prompt)

            # Extract the raw label string from the provider response.
            if isinstance(result, dict):
                raw = str(
                    result.get("output", result.get("intent", result.get("input", "")))
                ).strip().lower()
            else:
                raw = str(result).strip().lower()

            intent = self._parse_intent(raw)
            logger.debug(
                "IntentRouter.classify: transcript=%r → intent=%s (raw=%r)",
                transcript[:40], intent.value, raw[:40],
            )
            return IntentResult(
                intent=intent,
                confidence=1.0,
                raw_label=raw,
                language_hint=language_hint,
            )

        except Exception as exc:
            logger.warning(
                "IntentRouter.classify failed (%r) — defaulting to CHAT", exc
            )
            return IntentResult(
                intent=Intent.CHAT,
                confidence=0.0,
                raw_label="",
                language_hint=language_hint,
            )

    async def route(
        self,
        intent_result: IntentResult,
        turn_context: TurnContext,
    ) -> AsyncGenerator[str, None]:
        """
        Route the turn to the owning capability subsystem and return an async
        generator that yields response chunks.

        For side-effecting intents (mac_command, schedule, task,
        run_automation) the DialogueManager gate is checked first.  If
        required slots are missing the returned generator yields a
        clarification prompt rather than executing the capability (Req 23.1).

        For read-only / conversational intents the capability handler is
        returned directly without any slot check.

        Parameters
        ----------
        intent_result:
            The :class:`IntentResult` returned by :py:meth:`classify`.
        turn_context:
            The current :class:`~core.orchestrator.orchestrator.TurnContext`.

        Yields
        ------
        str
            Response chunks / tokens from the capability handler.
        """
        intent = intent_result.intent
        handler = _HANDLER_MAP.get(intent, _unknown_handler)

        if intent in _SIDE_EFFECTING_INTENTS:
            needed_slots = _REQUIRED_SLOTS.get(intent, [])
            slot_result: SlotFillResult = self._dialogue_manager.assess(
                turn_context.transcript,
                needed_slots,
            )

            if not slot_result.sufficient:
                # Defer to the dialogue gate — yield clarification message.
                missing = slot_result.missing
                missing_str = ", ".join(missing)
                logger.debug(
                    "IntentRouter.route: intent=%s missing slots=%r",
                    intent.value, missing,
                )
                yield (
                    f"I need a bit more information before I can do that. "
                    f"Missing: {missing_str}. Could you provide those details?"
                )
                return

        # Execute the capability handler.
        async for chunk in handler(turn_context):
            yield chunk

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_intent(raw: str) -> Intent:
        """
        Map a raw LLM output string to an :class:`~core.orchestrator.orchestrator.Intent`.

        Performs a fuzzy prefix match so that extra punctuation or whitespace
        from the LLM does not cause a hard failure.
        """
        # Strip surrounding quotes, punctuation, and whitespace.
        cleaned = raw.strip("\"'.,!? \t\n").lower()

        _ALIAS_MAP: dict[str, Intent] = {
            "chat": Intent.CHAT,
            "recall": Intent.RECALL,
            "remember": Intent.REMEMBER,
            "read_aloud": Intent.READ_ALOUD,
            "readaloud": Intent.READ_ALOUD,
            "read aloud": Intent.READ_ALOUD,
            "mac_command": Intent.MAC_COMMAND,
            "maccommand": Intent.MAC_COMMAND,
            "mac command": Intent.MAC_COMMAND,
            "run_automation": Intent.RUN_AUTOMATION,
            "runautomation": Intent.RUN_AUTOMATION,
            "run automation": Intent.RUN_AUTOMATION,
            "image": Intent.IMAGE,
            "schedule": Intent.SCHEDULE,
            "task": Intent.TASK,
            "meta": Intent.META,
            "unknown": Intent.UNKNOWN,
        }

        if cleaned in _ALIAS_MAP:
            return _ALIAS_MAP[cleaned]

        # Prefix match fallback — pick the first alias whose key starts with
        # the cleaned string or that the cleaned string starts with.
        for key, intent_val in _ALIAS_MAP.items():
            if cleaned.startswith(key) or key.startswith(cleaned):
                return intent_val

        return Intent.CHAT
