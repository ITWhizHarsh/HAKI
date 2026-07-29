"""Typed ports for the isolated local voice runtime.

The voice package owns session-scoped turn processing. These protocols are
intentionally small and do not expose retired routing services or process-wide
orchestration conversation state.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable
from uuid import UUID


VoiceLanguage = Literal["hi", "en", "hinglish"]


@dataclass(frozen=True)
class VoiceTurnRequest:
    """A normalized final transcript accepted by one voice session."""

    session_id: UUID
    turn_id: UUID
    text: str
    language: VoiceLanguage


@dataclass(frozen=True)
class VoiceContextMessage:
    """An immutable message supplied from session-owned voice context."""

    turn_id: UUID
    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True)
class VoiceSentence:
    """A complete response sentence ready for local speech synthesis."""

    turn_id: UUID
    sentence_id: UUID
    text: str
    language: VoiceLanguage


@runtime_checkable
class VoiceTurnPipeline(Protocol):
    """Session-owned asynchronous voice turn ingress and shutdown port."""

    async def submit_final_transcript(self, turn: VoiceTurnRequest) -> None:
        """Accept one final transcript without delegating to legacy routing."""

    async def close(self) -> None:
        """Release the session-owned pipeline resources."""


@runtime_checkable
class LocalVoiceLLM(Protocol):
    """Local-only token stream port for a single voice turn."""

    def stream_response(
        self,
        turn: VoiceTurnRequest,
        *,
        context: Sequence[VoiceContextMessage],
    ) -> AsyncIterator[str]:
        """Yield response text from the voice-specific local LLM service."""


@runtime_checkable
class LocalVoiceTTS(Protocol):
    """Local-only PCM stream port for one complete response sentence."""

    def stream_pcm(self, sentence: VoiceSentence) -> AsyncIterator[bytes]:
        """Yield PCM chunks without selecting a system or legacy TTS engine."""
