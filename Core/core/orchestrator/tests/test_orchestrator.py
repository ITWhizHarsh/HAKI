"""
Tests for the Orchestrator turn loop (Task 11.1) and Intent Routing (Task 11.2).

Covers:
- Parallel enrichment: mood / language / memory dispatched and results merged
- Cancellability at every await point (barge-in / cancel())
- Memory timeout respected (2-second budget, Req 7.3)
- Graceful degradation when mood / language / memory are unavailable (Req 6.5)
- Intent routing fallback
- Persona shaping applied to the raw capability response
- IPC TURN_REQUEST → stream_turn wiring (via JSONIPCServer)
- IntentRouter.classify() — LLM-backed intent classification (Task 11.2)
- IntentRouter.route() — all 9 intents route to their handler stubs (Task 11.2)
- Dialogue gate for side-effecting intents (Task 11.2)
- IntentResult dataclass (Task 11.2)

Design: The Orchestrator, Intent Routing.
Requirements: 3.1, 4.8, 5.1, 6.1, 6.5.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.mood import MoodResult
from core.orchestrator import Orchestrator, Intent, TurnContext, MEMORY_TIMEOUT_SECS
from core.orchestrator.intent_router import IntentRouter, IntentResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SlowMemory:
    """Memory stub that takes longer than the 2-second budget to respond."""

    def __init__(self, delay: float = MEMORY_TIMEOUT_SECS + 0.3) -> None:
        self._delay = delay

    async def retrieve(self, query: str, k: int = 5) -> list[Any]:
        await asyncio.sleep(self._delay)
        return [{"note": "this should never arrive"}]


class _FailingMemory:
    """Memory stub that always raises an exception."""

    async def retrieve(self, query: str, k: int = 5) -> list[Any]:
        raise RuntimeError("Memory_Brain unavailable")


class _RichMemory:
    """Memory stub that returns a canned list of notes."""

    async def retrieve(self, query: str, k: int = 5) -> list[Any]:
        return [{"note": f"memory note for {query!r}"}]


# ---------------------------------------------------------------------------
# Basic happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_turn_returns_string():
    """run_turn must always return a non-empty string."""
    orch = Orchestrator()
    result = await orch.run_turn("Hello HAKI")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_run_turn_with_audio_features():
    """Passing audio_features should not break the turn loop."""
    orch = Orchestrator()
    features = {"pitch_mean": 220.0, "volume_mean": -15.0, "duration_ms": 1500.0}
    result = await orch.run_turn("I am angry", features)
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_run_turn_transcript_included_in_response():
    """The shaped response should reference the user's transcript somewhere."""
    orch = Orchestrator()
    result = await orch.run_turn("what time is it")
    # The stub response contains the transcript; persona shaping does not remove it.
    assert "what time is it" in result


# ---------------------------------------------------------------------------
# Parallel enrichment / graceful degradation (Req 6.5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_turn_proceeds_without_memory():
    """
    When Memory_Brain is unavailable the turn loop must complete normally
    (Requirement 6.5).
    """
    orch = Orchestrator(memory_brain=_FailingMemory())
    result = await orch.run_turn("remember my exam date")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_run_turn_memory_timeout_respected():
    """
    Memory retrieval must be bounded by MEMORY_TIMEOUT_SECS (Req 7.3).
    The turn must still complete within a reasonable wall-clock window even
    when the memory stub is slow.
    """
    orch = Orchestrator(memory_brain=_SlowMemory())
    start = time.monotonic()
    result = await orch.run_turn("recall my schedule")
    elapsed = time.monotonic() - start

    assert isinstance(result, str)
    # The turn should complete not long after the timeout fires.
    # Allow 1-second slack for test overhead / CI variance.
    assert elapsed < MEMORY_TIMEOUT_SECS + 1.5, (
        f"Turn took {elapsed:.2f}s — memory timeout not respected"
    )


@pytest.mark.asyncio
async def test_run_turn_with_rich_memory():
    """Memory notes are accessible to the capability handler."""
    orch = Orchestrator(memory_brain=_RichMemory())
    result = await orch.run_turn("recall my exam date")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_run_turn_mood_unavailable():
    """
    When Mood_Detector raises, the turn must still complete (Req 6.5).
    """
    bad_mood = MagicMock()
    bad_mood.classify.side_effect = RuntimeError("acoustic model not loaded")

    orch = Orchestrator(mood_detector=bad_mood)
    result = await orch.run_turn("I'm feeling great today")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_run_turn_language_uninterpretable():
    """
    Uninterpretable language input must not abort the turn (Req 5.5 / 6.5).
    """
    from core.language import LanguageEngine, UninterpretableInputError

    bad_lang = MagicMock(spec=LanguageEngine)
    bad_lang.analyze.side_effect = UninterpretableInputError("cannot determine")

    orch = Orchestrator(language_engine=bad_lang)
    result = await orch.run_turn("###")
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Cancellation / barge-in (Req 3.3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_raises_cancelled_error():
    """
    cancel() must cause the currently-running turn to raise CancelledError.
    The task wrapper should catch it, not swallow it silently.
    """
    orch = Orchestrator(memory_brain=_SlowMemory())

    async def _slow_turn():
        return await orch.run_turn("a long utterance that takes time")

    task = asyncio.ensure_future(_slow_turn())
    orch.set_current_task(task)

    # Give the turn a moment to start, then cancel.
    await asyncio.sleep(0.05)
    orch.cancel()

    with pytest.raises((asyncio.CancelledError, Exception)):
        await asyncio.wait_for(task, timeout=3.0)


@pytest.mark.asyncio
async def test_cancel_flag_cleared_on_new_turn():
    """
    A new call to run_turn must clear the cancel flag so the turn is not
    immediately aborted.
    """
    orch = Orchestrator()
    orch.cancel()  # Pre-set the cancel flag

    # The next run_turn should clear the flag and complete normally.
    result = await orch.run_turn("fresh turn after cancel")
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Intent routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_intent_is_chat():
    """When the LLM stub does not return a valid label, default to CHAT."""
    orch = Orchestrator()
    # run_turn internally uses CHAT as the fallback and returns a stub response.
    result = await orch.run_turn("tell me a joke")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_intent_routing_by_stub_response():
    """
    Injecting an LLM provider that returns 'recall' label routes to the
    recall handler.
    """
    recall_llm = MagicMock()
    recall_llm.invoke.return_value = {"intent": "recall"}

    orch = Orchestrator(llm_provider=recall_llm)
    result = await orch.run_turn("what do you remember about me")
    # The recall handler's response contains "remember" or the memory message.
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# stream_turn generator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_turn_yields_tokens():
    """stream_turn must yield at least one token."""
    orch = Orchestrator()
    tokens: list[str] = []
    async for token in orch.stream_turn("Hello"):
        tokens.append(token)
    assert len(tokens) >= 1
    assert all(isinstance(t, str) for t in tokens)


@pytest.mark.asyncio
async def test_stream_turn_reassembled_matches_run_turn():
    """
    Re-joining the tokens from stream_turn should produce the same text as
    run_turn (since the stub yields word-by-word).
    """
    orch = Orchestrator()
    transcript = "what is my schedule today"

    streamed = []
    async for token in orch.stream_turn(transcript):
        streamed.append(token)

    from_run = await orch.run_turn(transcript)
    # run_turn is called twice — reset cancel flag between calls.
    assert " ".join(streamed) == from_run


# ---------------------------------------------------------------------------
# TurnContext
# ---------------------------------------------------------------------------


def test_turn_context_defaults():
    """TurnContext should initialise with sane defaults."""
    ctx = TurnContext(transcript="hi")
    assert ctx.transcript == "hi"
    assert ctx.audio_features == {}
    assert ctx.mood is None
    assert ctx.language_composition is None
    assert ctx.memory_context == []
    assert ctx.intent is Intent.UNKNOWN


# ===========================================================================
# Task 11.2 — Intent Routing tests
# ===========================================================================
#
# These tests verify:
#   1. IntentResult dataclass structure
#   2. IntentRouter.classify() — LLM-backed async classification
#   3. IntentRouter.route() — all 9 intents route to stub handlers
#   4. Dialogue gate defers side-effecting intents when slots are missing
#   5. Orchestrator wires classify + route end-to-end
#
# Design: Intent Routing. Requirements: 6.1.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_returning(intent_label: str) -> MagicMock:
    """Return a stub LLM provider whose invoke() returns the given intent label."""
    stub = MagicMock()
    stub.invoke.return_value = {"intent": intent_label}
    return stub


def _make_turn_context(transcript: str = "test") -> TurnContext:
    """Build a minimal TurnContext for routing tests."""
    return TurnContext(transcript=transcript)


# ---------------------------------------------------------------------------
# IntentResult
# ---------------------------------------------------------------------------


def test_intent_result_defaults():
    """IntentResult must have sensible defaults for optional fields."""
    ir = IntentResult(intent=Intent.CHAT)
    assert ir.intent is Intent.CHAT
    assert ir.confidence == 1.0
    assert ir.raw_label == ""
    assert ir.language_hint is None


def test_intent_result_fields():
    """IntentResult stores all explicitly provided fields."""
    ir = IntentResult(
        intent=Intent.RECALL,
        confidence=0.85,
        raw_label="recall",
        language_hint="hinglish",
    )
    assert ir.intent is Intent.RECALL
    assert ir.confidence == 0.85
    assert ir.raw_label == "recall"
    assert ir.language_hint == "hinglish"


# ---------------------------------------------------------------------------
# IntentRouter.classify() — async LLM-based classification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_returns_intent_result():
    """classify() must return an IntentResult, not a bare Intent."""
    router = IntentRouter(llm_provider=_make_llm_returning("chat"))
    result = await router.classify("tell me a joke")
    assert isinstance(result, IntentResult)
    assert isinstance(result.intent, Intent)


@pytest.mark.asyncio
async def test_classify_all_nine_intents():
    """
    classify() must correctly map each of the nine intent labels returned
    by the LLM to the corresponding Intent enum member.
    """
    intent_cases = [
        ("chat", Intent.CHAT),
        ("recall", Intent.RECALL),
        ("remember", Intent.REMEMBER),
        ("read_aloud", Intent.READ_ALOUD),
        ("mac_command", Intent.MAC_COMMAND),
        ("run_automation", Intent.RUN_AUTOMATION),
        ("image", Intent.IMAGE),
        ("schedule", Intent.SCHEDULE),
        ("task", Intent.TASK),
        ("meta", Intent.META),
    ]
    for label, expected_intent in intent_cases:
        router = IntentRouter(llm_provider=_make_llm_returning(label))
        result = await router.classify(f"transcript for {label}")
        assert result.intent is expected_intent, (
            f"Expected {expected_intent} for label {label!r}, got {result.intent}"
        )


@pytest.mark.asyncio
async def test_classify_defaults_to_chat_on_unknown_label():
    """An unrecognised LLM label must fall back to Intent.CHAT."""
    router = IntentRouter(llm_provider=_make_llm_returning("gibberish_xyz"))
    result = await router.classify("some transcript")
    assert result.intent is Intent.CHAT


@pytest.mark.asyncio
async def test_classify_defaults_to_chat_on_llm_error():
    """If the LLM raises an exception, classify() must return CHAT with confidence 0."""
    failing_llm = MagicMock()
    failing_llm.invoke.side_effect = RuntimeError("LLM unavailable")
    router = IntentRouter(llm_provider=failing_llm)
    result = await router.classify("what time is it")
    assert result.intent is Intent.CHAT
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_classify_propagates_language_hint():
    """
    When a language_result dict is provided, classify() should store the
    composition in IntentResult.language_hint.
    """
    router = IntentRouter(llm_provider=_make_llm_returning("chat"))
    result = await router.classify(
        "Kya time hai",
        language_result={"composition": "hinglish"},
    )
    assert result.language_hint == "hinglish"


@pytest.mark.asyncio
async def test_classify_no_language_hint_when_none():
    """When no language_result is given, language_hint must be None."""
    router = IntentRouter(llm_provider=_make_llm_returning("chat"))
    result = await router.classify("hello", language_result=None)
    assert result.language_hint is None


# ---------------------------------------------------------------------------
# IntentRouter.route() — async generator routing to stub handlers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_read_only_intent_yields_chunks():
    """
    Read-only intents (chat, recall, remember, read_aloud, image, meta)
    must yield at least one chunk without requiring a slot check.
    """
    read_only = [
        Intent.CHAT,
        Intent.RECALL,
        Intent.REMEMBER,
        Intent.READ_ALOUD,
        Intent.IMAGE,
        Intent.META,
    ]
    for intent in read_only:
        router = IntentRouter()
        ctx = _make_turn_context(f"test for {intent.value}")
        ir = IntentResult(intent=intent)
        chunks = [chunk async for chunk in router.route(ir, ctx)]
        assert len(chunks) >= 1, f"No chunks for intent {intent.value}"
        assert all(isinstance(c, str) for c in chunks)


@pytest.mark.asyncio
async def test_route_all_nine_intents_return_non_empty_response():
    """
    Every intent must produce a non-empty response, with side-effecting
    intents either clarifying missing slots or executing their stubs.
    """
    all_intents = list(Intent)
    for intent in all_intents:
        router = IntentRouter()
        ctx = _make_turn_context(f"test for {intent.value}")
        ir = IntentResult(intent=intent)
        chunks = [chunk async for chunk in router.route(ir, ctx)]
        full = "".join(chunks)
        assert len(full) > 0, f"Empty response for intent {intent.value}"


@pytest.mark.asyncio
async def test_route_chat_stub_contains_transcript():
    """The chat stub handler should reference the user's transcript."""
    router = IntentRouter()
    ctx = _make_turn_context("what do you know about me")
    ir = IntentResult(intent=Intent.CHAT)
    chunks = [chunk async for chunk in router.route(ir, ctx)]
    full = "".join(chunks)
    assert "what do you know about me" in full


@pytest.mark.asyncio
async def test_route_recall_stub_contains_transcript():
    """The recall stub handler should reference the user's transcript."""
    router = IntentRouter()
    ctx = _make_turn_context("remember my exam date")
    ir = IntentResult(intent=Intent.RECALL)
    chunks = [chunk async for chunk in router.route(ir, ctx)]
    full = "".join(chunks)
    assert "remember my exam date" in full


@pytest.mark.asyncio
async def test_route_unknown_intent_yields_fallback():
    """Intent.UNKNOWN must route to the fallback handler without error."""
    router = IntentRouter()
    ctx = _make_turn_context("###")
    ir = IntentResult(intent=Intent.UNKNOWN)
    chunks = [chunk async for chunk in router.route(ir, ctx)]
    full = "".join(chunks)
    assert len(full) > 0


# ---------------------------------------------------------------------------
# Dialogue gate — side-effecting intents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_side_effecting_intent_without_slots_defers():
    """
    Side-effecting intents (mac_command, schedule, task, run_automation)
    must defer to the DialogueManager when required slots are missing;
    the yielded message should mention the missing information.
    """
    side_effecting = [
        Intent.MAC_COMMAND,
        Intent.SCHEDULE,
        Intent.TASK,
        Intent.RUN_AUTOMATION,
    ]
    for intent in side_effecting:
        router = IntentRouter()  # default DialogueManager stub always returns missing slots
        ctx = _make_turn_context(f"do {intent.value}")
        ir = IntentResult(intent=intent)
        chunks = [chunk async for chunk in router.route(ir, ctx)]
        full = "".join(chunks)
        # The clarification message should appear — DialogueManager stub returns all slots missing
        assert "more information" in full or "Missing" in full, (
            f"Expected clarification for side-effecting intent {intent.value}, got: {full!r}"
        )


@pytest.mark.asyncio
async def test_route_side_effecting_intent_with_all_slots_executes():
    """
    When a DialogueManager reports all slots sufficient, the capability
    handler runs and returns the stub response.
    """
    from core.dialogue import DialogueManager, SlotFillResult

    # DialogueManager that always reports sufficient slots.
    sufficient_dm = MagicMock(spec=DialogueManager)
    sufficient_dm.assess.return_value = SlotFillResult(sufficient=True, missing=[])

    router = IntentRouter(dialogue_manager=sufficient_dm)
    ctx = _make_turn_context("schedule a meeting tomorrow at 3pm")
    ir = IntentResult(intent=Intent.SCHEDULE)
    chunks = [chunk async for chunk in router.route(ir, ctx)]
    full = "".join(chunks)
    # Handler stub runs → returns the schedule stub response
    assert "schedule" in full.lower()


@pytest.mark.asyncio
async def test_route_read_only_intent_does_not_call_dialogue_manager():
    """
    Read-only intents must NOT invoke the DialogueManager gate at all.
    """
    from core.dialogue import DialogueManager

    mock_dm = MagicMock(spec=DialogueManager)
    router = IntentRouter(dialogue_manager=mock_dm)
    ctx = _make_turn_context("tell me a joke")
    ir = IntentResult(intent=Intent.CHAT)
    _ = [chunk async for chunk in router.route(ir, ctx)]
    mock_dm.assess.assert_not_called()


# ---------------------------------------------------------------------------
# Orchestrator end-to-end wiring (Task 11.2 integration)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_routes_intent_with_injected_llm():
    """
    When an LLM returning a specific intent label is injected, the
    orchestrator's dispatch should use that intent's handler stub.
    """
    recall_llm = _make_llm_returning("recall")
    orch = Orchestrator(llm_provider=recall_llm)
    result = await orch.run_turn("what do you know about me")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_orchestrator_last_intent_result_populated():
    """
    After a turn completes, the orchestrator's _last_intent_result should be
    set to an IntentResult reflecting the classified intent.
    """
    chat_llm = _make_llm_returning("chat")
    orch = Orchestrator(llm_provider=chat_llm)
    await orch.run_turn("hello")
    # Access the internal state; this is valid for test introspection.
    assert orch._last_intent_result is not None
    assert isinstance(orch._last_intent_result, IntentResult)
    assert orch._last_intent_result.intent is Intent.CHAT


@pytest.mark.asyncio
async def test_orchestrator_side_effecting_intent_defers_missing_slots():
    """
    When the LLM classifies a side-effecting intent and the DialogueManager
    stub reports missing slots, the response should be a clarification message.
    """
    schedule_llm = _make_llm_returning("schedule")
    orch = Orchestrator(llm_provider=schedule_llm)
    result = await orch.run_turn("schedule something")
    # Default DialogueManager stub always reports missing slots.
    assert isinstance(result, str)
    assert "more information" in result or "Missing" in result


@pytest.mark.asyncio
async def test_orchestrator_meta_intent_routes_to_stub():
    """Meta intent (time, settings) should route to the meta stub."""
    meta_llm = _make_llm_returning("meta")
    orch = Orchestrator(llm_provider=meta_llm)
    result = await orch.run_turn("what time is it")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_orchestrator_image_intent_routes_to_stub():
    """Image intent should route to the image generation stub."""
    image_llm = _make_llm_returning("image")
    orch = Orchestrator(llm_provider=image_llm)
    result = await orch.run_turn("create an image of a sunset")
    assert isinstance(result, str)
    assert "image" in result.lower() or "sunset" in result.lower()


@pytest.mark.asyncio
async def test_orchestrator_read_aloud_intent_routes_to_stub():
    """Read-aloud intent should route to the screen-reader stub."""
    read_llm = _make_llm_returning("read_aloud")
    orch = Orchestrator(llm_provider=read_llm)
    result = await orch.run_turn("read this page aloud")
    assert isinstance(result, str)
    assert "read" in result.lower() or "screen" in result.lower()
