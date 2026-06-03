"""
Persona Engine — HAKI personality identity, intensity shaping, and mood-to-tone mapping.

This module is the Persona_Engine subsystem for HAKI.  It applies a consistent
HAKI personality identity to every response, integrates mood and memory context
into the response tone, and provides an intensity control with at least three
ordered levels.

The mood-to-tone mapping is a pure function: angry (confidence ≥ threshold) →
CALMING; sad (confidence ≥ threshold) → ENCOURAGING; everything else → NEUTRAL.

Design: Persona_Engine.
Requirements: 4.3, 4.4, 4.5, 4.6, 6.1, 6.2, 6.3, 6.4, 6.5.

Public types
------------
IntensityLevel
    Ordered enum with at least three levels: MIN, MID, MAX.

Tone
    Response tone directive: CALMING, ENCOURAGING, NEUTRAL.

PersonaContext
    Carries optional MoodResult and optional memory context string; both may
    be None — the engine proceeds with whatever is available (Req 6.5).

mood_to_tone(mood_result, threshold) -> Tone
    Pure function mapping a MoodResult (or None) and confidence threshold to
    a Tone.  No side effects.

PersonaEngine
    Main class.  Entry points:
      shape(response_draft, context)  → str  (shaped response string)
      build_system_prompt(tone, memory_context)  → str
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from core.mood.mood_detector import MoodResult


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class IntensityLevel(IntEnum):
    """
    Ordered personality intensity levels (Req 6.3).

    At least three discrete ordered levels spanning from a defined minimum
    (MIN) to a defined maximum (MAX).  The integer values encode ordering
    so comparisons like ``intensity == IntensityLevel.MIN`` work naturally.
    """

    MIN = 0   # Minimum: conciseness over personality (Req 6.4)
    MID = 1   # Standard: balanced personality + helpfulness
    MAX = 2   # Maximum: full personality expression with wit and emotion


class Tone(str):
    """
    Response tone directives produced by mood-to-tone mapping (Reqs 4.3–4.6).

    String subclass so tone values are transparent in prompts.
    """

    CALMING = "calming"
    ENCOURAGING = "encouraging"
    NEUTRAL = "neutral"


# Tone sentinel constants for ergonomic use throughout the module.
_TONE_CALMING = Tone.CALMING
_TONE_ENCOURAGING = Tone.ENCOURAGING
_TONE_NEUTRAL = Tone.NEUTRAL


# ---------------------------------------------------------------------------
# PersonaContext
# ---------------------------------------------------------------------------


@dataclass
class PersonaContext:
    """
    Aggregated inputs available when the Persona_Engine shapes a response.

    Both fields are optional; the engine proceeds with whatever is provided
    without delaying the response for a missing input (Req 6.5).

    Attributes
    ----------
    mood_result : MoodResult | None
        Classified mood from the Mood_Detector, or None when unavailable.
    memory_context : str | None
        Relevant notes retrieved from Memory_Brain, or None when unavailable.
    """

    mood_result: MoodResult | None = None
    memory_context: str | None = None


# ---------------------------------------------------------------------------
# mood_to_tone — pure function (Task 10.2)
# ---------------------------------------------------------------------------


def mood_to_tone(mood_result: MoodResult | None, threshold: float) -> str:
    """
    Map a MoodResult (and a confidence threshold) to a tone directive.

    Pure function — no side effects, no I/O.

    Rules (Requirements 4.3–4.6):

    +--------------------------+-----------------------------------+-----------+
    | mood_result              | condition                         | → Tone    |
    +--------------------------+-----------------------------------+-----------+
    | None                     | —                                 | NEUTRAL   |
    | unclassifiable           | —                                 | NEUTRAL   |
    | classified angry         | confidence >= threshold           | CALMING   |
    | classified sad           | confidence >= threshold           | ENCOURAGING|
    | classified angry/sad     | confidence < threshold            | NEUTRAL   |
    | classified happy/neutral | any confidence                    | NEUTRAL   |
    +--------------------------+-----------------------------------+-----------+

    Parameters
    ----------
    mood_result : MoodResult | None
        Result from the Mood_Detector.  None is treated the same as
        unclassifiable (Req 4.6).
    threshold : float
        Confidence threshold in [0.0, 1.0].  When confidence < threshold
        the result is treated as below-threshold → NEUTRAL (Req 4.6).

    Returns
    -------
    str
        One of ``Tone.CALMING``, ``Tone.ENCOURAGING``, or ``Tone.NEUTRAL``.
    """
    # None or unclassifiable → NEUTRAL (Req 4.6)
    if mood_result is None or mood_result.unclassifiable:
        return _TONE_NEUTRAL

    # Below-threshold confidence → NEUTRAL (Req 4.6)
    confidence = mood_result.confidence
    if confidence is None or confidence < threshold:
        return _TONE_NEUTRAL

    primary = mood_result.primary_mood

    # angry ≥ threshold → CALMING (Req 4.3)
    if primary == "angry":
        return _TONE_CALMING

    # sad ≥ threshold → ENCOURAGING (Req 4.4)
    if primary == "sad":
        return _TONE_ENCOURAGING

    # happy / neutral / other ≥ threshold → NEUTRAL (Req 4.5)
    return _TONE_NEUTRAL


# ---------------------------------------------------------------------------
# Prompt templates (internal)
# ---------------------------------------------------------------------------

# HAKI identity snippet — always included regardless of intensity (Req 6.1).
_HAKI_IDENTITY = (
    "You are HAKI (Heuristic Augmented Knowledge Interface), a personal AI "
    "assistant with a warm, witty, and deeply contextual personality.  "
    "You are the user's constant companion: knowledgeable, candid, and "
    "occasionally playful.  Always respond as HAKI."
)

_TONE_DIRECTIVES: dict[str, str] = {
    _TONE_CALMING: (
        "The user appears frustrated or upset.  Respond with a calm, composed, "
        "and reassuring tone.  Avoid amplifying tension; instead, de-escalate "
        "and offer patient support."
    ),
    _TONE_ENCOURAGING: (
        "The user appears sad or discouraged.  Respond with a warm, uplifting, "
        "and encouraging tone.  Offer genuine support and motivate the user."
    ),
    _TONE_NEUTRAL: (
        "Respond in a balanced, natural tone appropriate to the conversation."
    ),
}

_INTENSITY_DIRECTIVES: dict[IntensityLevel, str] = {
    IntensityLevel.MIN: (
        "CONCISENESS MODE (minimum intensity): "
        "Prioritize brevity and directness over personality expression.  "
        "Keep your response as short as possible while remaining helpful.  "
        "Omit embellishments, wit, and personality flavour when they conflict "
        "with conciseness (Req 6.4)."
    ),
    IntensityLevel.MID: (
        "STANDARD MODE: Balance helpfulness and personality.  "
        "Be informative and warm; allow light personality expression, "
        "but keep the response focused."
    ),
    IntensityLevel.MAX: (
        "EXPRESSIVE MODE (maximum intensity): "
        "Express the full HAKI personality with wit, emotion, and flair.  "
        "Feel free to be playful, empathetic, and memorable while remaining helpful."
    ),
}


# ---------------------------------------------------------------------------
# PersonaEngine (Task 10.1)
# ---------------------------------------------------------------------------


class PersonaEngine:
    """
    Shape HAKI's responses with a consistent personality identity, intensity
    level, mood-driven tone, and memory context.

    Design: Persona_Engine.
    Requirements: 4.3, 4.4, 4.5, 4.6, 6.1, 6.2, 6.3, 6.4, 6.5.

    Parameters
    ----------
    intensity : IntensityLevel
        Initial personality intensity level (default MID).
    threshold : float
        Confidence threshold forwarded to ``mood_to_tone`` (default 0.6).
    model_provider : object | None
        Optional Model Provider for LLM-backed shaping.  When None, a
        template-based fallback is used so the engine works standalone.
    """

    def __init__(
        self,
        intensity: IntensityLevel = IntensityLevel.MID,
        threshold: float = 0.6,
        model_provider: object | None = None,
    ) -> None:
        self._intensity: IntensityLevel = intensity
        self._threshold: float = threshold
        self._model_provider = model_provider

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def intensity(self) -> IntensityLevel:
        """Current personality intensity level (Req 6.3)."""
        return self._intensity

    @intensity.setter
    def intensity(self, value: IntensityLevel) -> None:
        """Set a new intensity level."""
        if not isinstance(value, IntensityLevel):
            raise TypeError(
                f"intensity must be an IntensityLevel, got {type(value).__name__!r}."
            )
        self._intensity = value

    @property
    def threshold(self) -> float:
        """Confidence threshold used for mood-to-tone mapping."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set a new confidence threshold; must be in [0.0, 1.0]."""
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError(
                f"threshold must be in [0.0, 1.0], got {value!r}."
            )
        self._threshold = float(value)

    # ------------------------------------------------------------------
    # Primary entry point (Task 10.1)
    # ------------------------------------------------------------------

    def shape(self, response_draft: str, context: PersonaContext) -> str:
        """
        Shape *response_draft* with HAKI personality, intensity, mood tone,
        and memory context.

        Behaviour summary
        -----------------
        - Always applies HAKI identity (Req 6.1) — never skipped.
        - Derives the tone from ``context.mood_result`` via ``mood_to_tone``
          when a mood is available; defaults to NEUTRAL otherwise (Reqs 4.3–4.6).
        - Incorporates ``context.memory_context`` when available (Req 6.2).
        - At MIN intensity prioritizes conciseness over personality (Req 6.4).
        - Proceeds with whatever inputs are available; never delays for a
          missing mood or memory context (Req 6.5).

        If a ``model_provider`` was supplied at construction, the shaped
        response is produced by querying the LLM with the constructed system
        prompt.  Otherwise a lightweight template-based shaping is applied as
        a fallback (useful for testing and offline scenarios).

        Parameters
        ----------
        response_draft : str
            Raw response content to shape.
        context : PersonaContext
            Available mood and memory context (either or both may be None).

        Returns
        -------
        str
            Shaped response string with HAKI identity and tone applied.
        """
        tone = mood_to_tone(context.mood_result, self._threshold)
        system_prompt = self.build_system_prompt(tone, context.memory_context)

        if self._model_provider is not None:
            return self._llm_shape(response_draft, system_prompt)
        else:
            return self._template_shape(response_draft, system_prompt, tone)

    # ------------------------------------------------------------------
    # build_system_prompt (Task 10.1)
    # ------------------------------------------------------------------

    def build_system_prompt(
        self,
        tone: str,
        memory_context: str | None,
    ) -> str:
        """
        Build the full system prompt encoding HAKI identity, intensity,
        tone directive, and memory context.

        The prompt is always structured as:

        1. HAKI identity (always present, Req 6.1).
        2. Intensity directive (Req 6.3, 6.4).
        3. Tone directive (Req 6.2, 4.3–4.6).
        4. Memory context, when available (Req 6.2).

        Parameters
        ----------
        tone : str
            One of ``Tone.CALMING``, ``Tone.ENCOURAGING``, ``Tone.NEUTRAL``.
        memory_context : str | None
            Relevant notes from Memory_Brain, or None.

        Returns
        -------
        str
            Complete system prompt string.
        """
        parts: list[str] = [
            "=== HAKI System Prompt ===",
            "",
            "## Identity",
            _HAKI_IDENTITY,
            "",
            "## Intensity",
            _INTENSITY_DIRECTIVES[self._intensity],
            "",
            "## Tone",
            _TONE_DIRECTIVES.get(tone, _TONE_DIRECTIVES[_TONE_NEUTRAL]),
        ]

        if memory_context is not None and memory_context.strip():
            parts += [
                "",
                "## Memory Context",
                (
                    "The following notes from HAKI's memory are relevant to the "
                    "current conversation.  Incorporate them naturally into your "
                    "response tone and content where appropriate:"
                ),
                memory_context.strip(),
            ]

        parts += [
            "",
            "=== End of System Prompt ===",
        ]

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Private shaping helpers
    # ------------------------------------------------------------------

    def _llm_shape(self, response_draft: str, system_prompt: str) -> str:
        """
        Shape the response via the injected model_provider (LLM path).

        Calls ``model_provider.invoke(...)`` with the system prompt and
        response draft.  The LLM is expected to return the shaped text.
        If the provider returns a dict with a ``"text"`` or ``"content"``
        key, that value is used; otherwise the raw return value is
        coerced to str.
        """
        from core.model_provider import Capability  # local import to avoid circulars

        raw = self._model_provider.invoke(  # type: ignore[union-attr]
            response_draft,
            system_prompt=system_prompt,
            capability_hint=Capability.LLM.value,
        )
        if isinstance(raw, dict):
            return str(raw.get("text") or raw.get("content") or response_draft)
        return str(raw) if raw else response_draft

    def _template_shape(
        self,
        response_draft: str,
        system_prompt: str,
        tone: str,
    ) -> str:
        """
        Lightweight template-based shaping used when no model_provider is set.

        At MIN intensity the draft is returned verbatim (conciseness first,
        Req 6.4).  At MID/MAX intensity a brief tone-appropriate preamble
        is prepended so the HAKI identity is always visible (Req 6.1).
        """
        if self._intensity == IntensityLevel.MIN:
            # Minimum intensity: return the draft as-is — conciseness wins
            # over any personality flavour (Req 6.4).
            return response_draft.strip()

        # MID / MAX — prepend a minimal tone-appropriate framing so the
        # HAKI identity is observable even without an LLM.
        preamble = self._tone_preamble(tone)
        draft = response_draft.strip()
        if preamble:
            return f"{preamble} {draft}"
        return draft

    @staticmethod
    def _tone_preamble(tone: str) -> str:
        """Return a short sentence reflecting the current tone for template shaping."""
        if tone == _TONE_CALMING:
            return "Hey, take a breath —"
        if tone == _TONE_ENCOURAGING:
            return "You've got this!"
        return ""  # NEUTRAL: no preamble, let the content speak for itself
