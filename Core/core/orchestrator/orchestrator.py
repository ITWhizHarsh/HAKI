"""
Orchestrator — central router for a conversational turn.

Sequences: Voice_Engine → parallel(Mood, Language, Memory) → intent
classification → capability dispatch → Persona shaping → TTS.
Every await point is cancellable to support barge-in and
clarifying-dialogue pauses.

Design: The Orchestrator, Intent Routing.
Requirements: 3.1, 4.8, 5.1, 6.5.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


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


@dataclass
class TurnContext:
    """Snapshot of all inputs assembled for a single conversational turn."""

    transcript: str
    audio_features: dict[str, Any] = field(default_factory=dict)
    mood: Any | None = None
    language_composition: str | None = None
    memory_context: list[Any] = field(default_factory=list)
    intent: Intent = Intent.UNKNOWN


class Orchestrator:
    """
    Central router for a conversational turn.

    This is a stub implementation.  The full implementation wires up
    Voice_Engine, Mood_Detector, Language_Engine, Memory_Brain,
    Dialogue_Manager, capability subsystems, and Persona_Engine.
    """

    def __init__(self) -> None:
        self._cancel_event: asyncio.Event = asyncio.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_turn(self, transcript: str, audio_features: dict[str, Any] | None = None) -> str:
        """
        Execute one conversational turn end-to-end.

        Parameters
        ----------
        transcript:
            Final STT transcript for the user's utterance.
        audio_features:
            Prosodic features (pitch, energy, …) extracted alongside the
            transcript, forwarded to Mood_Detector.

        Returns
        -------
        str
            Shaped response text ready for TTS.
        """
        self._cancel_event.clear()
        ctx = TurnContext(
            transcript=transcript,
            audio_features=audio_features or {},
        )

        # Phase 1 — parallel enrichment (mood / language / memory)
        ctx.mood, ctx.language_composition, ctx.memory_context = await asyncio.gather(
            self._classify_mood(ctx),
            self._detect_language(ctx),
            self._retrieve_memory(ctx),
        )
        self._check_cancelled()

        # Phase 2 — intent routing
        ctx.intent = await self._route_intent(ctx)
        self._check_cancelled()

        # Phase 3 — capability dispatch
        raw_response = await self._dispatch(ctx)
        self._check_cancelled()

        # Phase 4 — persona shaping
        shaped = await self._shape_response(raw_response, ctx)
        return shaped

    def cancel(self) -> None:
        """Signal cancellation (barge-in or explicit stop)."""
        self._cancel_event.set()

    # ------------------------------------------------------------------
    # Private helpers — stubs for downstream subsystems
    # ------------------------------------------------------------------

    def _check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise asyncio.CancelledError("Turn cancelled by barge-in or user request.")

    async def _classify_mood(self, ctx: TurnContext) -> Any:
        """Stub: forward audio_features to Mood_Detector."""
        return None

    async def _detect_language(self, ctx: TurnContext) -> str | None:
        """Stub: forward transcript to Language_Engine."""
        return None

    async def _retrieve_memory(self, ctx: TurnContext) -> list[Any]:
        """Stub: query Memory_Brain with transcript as query."""
        return []

    async def _route_intent(self, ctx: TurnContext) -> Intent:
        """Stub: LLM-based intent classification."""
        return Intent.CHAT

    async def _dispatch(self, ctx: TurnContext) -> str:
        """Stub: route to owning subsystem based on intent."""
        return f"[Orchestrator stub] Received: {ctx.transcript!r}"

    async def _shape_response(self, response: str, ctx: TurnContext) -> str:
        """Stub: forward response + context to Persona_Engine."""
        return response
