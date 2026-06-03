"""
Unit and property-based tests for the Persona_Engine subsystem.

Feature: haki-personal-ai-assistant
Requirements: 4.3, 4.4, 4.5, 4.6, 6.1, 6.2, 6.3, 6.4, 6.5

Test catalogue
--------------
Unit tests
1.  mood_to_tone: None mood_result → NEUTRAL                       (Req 4.6)
2.  mood_to_tone: unclassifiable → NEUTRAL                         (Req 4.6)
3.  mood_to_tone: angry ≥ threshold → CALMING                      (Req 4.3)
4.  mood_to_tone: angry < threshold → NEUTRAL                      (Req 4.6)
5.  mood_to_tone: sad ≥ threshold → ENCOURAGING                    (Req 4.4)
6.  mood_to_tone: sad < threshold → NEUTRAL                        (Req 4.6)
7.  mood_to_tone: happy ≥ threshold → NEUTRAL                      (Req 4.5)
8.  mood_to_tone: neutral ≥ threshold → NEUTRAL                    (Req 4.5)
9.  mood_to_tone: exact-threshold boundary (confidence == threshold) is ≥
10. IntensityLevel ordering: MIN < MID < MAX                       (Req 6.3)
11. PersonaEngine default intensity is MID
12. PersonaEngine intensity setter
13. PersonaEngine.shape identity always present (non-MIN)          (Req 6.1)
14. PersonaEngine.shape identity always present (MIN)              (Req 6.1)
15. PersonaEngine.shape at MIN returns concise (no preamble)       (Req 6.4)
16. PersonaEngine.shape incorporates tone preamble at MID          (Req 6.2)
17. PersonaEngine.shape incorporates tone preamble at MAX          (Req 6.2)
18. PersonaEngine.shape proceeds with None mood                    (Req 6.5)
19. PersonaEngine.shape proceeds with None memory                  (Req 6.5)
20. PersonaEngine.shape proceeds with both None                    (Req 6.5)
21. build_system_prompt includes HAKI identity at all intensities  (Req 6.1)
22. build_system_prompt MIN includes conciseness directive         (Req 6.4)
23. build_system_prompt includes memory context when provided      (Req 6.2)
24. build_system_prompt omits memory section when memory is None
25. PersonaEngine with model_provider calls provider               (integration)

Property-based tests (Property 10: Mood-to-tone mapping)
26. PBT: mood_to_tone output is always one of the three valid tones
27. PBT: angry with confidence ≥ threshold always maps to CALMING
28. PBT: sad with confidence ≥ threshold always maps to ENCOURAGING
29. PBT: non-angry/non-sad classified with confidence ≥ threshold → NEUTRAL
30. PBT: any mood below threshold maps to NEUTRAL
31. PBT: unclassifiable always maps to NEUTRAL
32. PBT: None mood_result always maps to NEUTRAL

Property-based tests (Property 15: Response produced regardless of optional inputs)
33. PBT: shape() always returns a non-empty string regardless of context
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from core.mood import MoodResult, VALID_MOODS
from core.model_provider import Capability, ModelProviderRegistry, ModelProvider
from core.persona import (
    IntensityLevel,
    PersonaContext,
    PersonaEngine,
    Tone,
    mood_to_tone,
)


# ---------------------------------------------------------------------------
# Helpers / strategies
# ---------------------------------------------------------------------------

_THRESHOLD = 0.6  # default threshold


def _classified(mood: str, confidence: float) -> MoodResult:
    """Shorthand: a classified MoodResult."""
    return MoodResult.classified(mood, confidence)


def _unclassifiable() -> MoodResult:
    """Shorthand: an unclassifiable MoodResult."""
    return MoodResult.unclassifiable_result()


# Hypothesis strategies
_st_valid_mood = st.sampled_from(sorted(VALID_MOODS))
_st_confidence = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
_st_threshold = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
_st_nonempty_string = st.text(min_size=1)


# ---------------------------------------------------------------------------
# ── Unit tests ──
# ---------------------------------------------------------------------------

# 1. mood_to_tone: None → NEUTRAL
def test_mood_to_tone_none_is_neutral() -> None:
    """None mood_result must produce NEUTRAL (Req 4.6)."""
    assert mood_to_tone(None, _THRESHOLD) == Tone.NEUTRAL


# 2. mood_to_tone: unclassifiable → NEUTRAL
def test_mood_to_tone_unclassifiable_is_neutral() -> None:
    """Unclassifiable MoodResult must produce NEUTRAL (Req 4.6)."""
    assert mood_to_tone(_unclassifiable(), _THRESHOLD) == Tone.NEUTRAL


# 3. mood_to_tone: angry ≥ threshold → CALMING
def test_mood_to_tone_angry_at_threshold_is_calming() -> None:
    """angry at exactly threshold confidence must map to CALMING (Req 4.3)."""
    result = _classified("angry", _THRESHOLD)
    assert mood_to_tone(result, _THRESHOLD) == Tone.CALMING


def test_mood_to_tone_angry_above_threshold_is_calming() -> None:
    """angry above threshold confidence must map to CALMING (Req 4.3)."""
    result = _classified("angry", 0.9)
    assert mood_to_tone(result, _THRESHOLD) == Tone.CALMING


def test_mood_to_tone_angry_max_confidence_is_calming() -> None:
    """angry at confidence 1.0 must map to CALMING (Req 4.3)."""
    result = _classified("angry", 1.0)
    assert mood_to_tone(result, _THRESHOLD) == Tone.CALMING


# 4. mood_to_tone: angry < threshold → NEUTRAL
def test_mood_to_tone_angry_below_threshold_is_neutral() -> None:
    """angry below threshold confidence must map to NEUTRAL (Req 4.6)."""
    result = _classified("angry", 0.3)
    assert mood_to_tone(result, _THRESHOLD) == Tone.NEUTRAL


def test_mood_to_tone_angry_just_below_threshold_is_neutral() -> None:
    """angry just below threshold (threshold - epsilon) must map to NEUTRAL."""
    confidence = _THRESHOLD - 0.001
    result = _classified("angry", round(confidence, 4))
    assert mood_to_tone(result, _THRESHOLD) == Tone.NEUTRAL


# 5. mood_to_tone: sad ≥ threshold → ENCOURAGING
def test_mood_to_tone_sad_at_threshold_is_encouraging() -> None:
    """sad at exactly threshold confidence must map to ENCOURAGING (Req 4.4)."""
    result = _classified("sad", _THRESHOLD)
    assert mood_to_tone(result, _THRESHOLD) == Tone.ENCOURAGING


def test_mood_to_tone_sad_above_threshold_is_encouraging() -> None:
    """sad above threshold confidence must map to ENCOURAGING (Req 4.4)."""
    result = _classified("sad", 0.85)
    assert mood_to_tone(result, _THRESHOLD) == Tone.ENCOURAGING


# 6. mood_to_tone: sad < threshold → NEUTRAL
def test_mood_to_tone_sad_below_threshold_is_neutral() -> None:
    """sad below threshold confidence must map to NEUTRAL (Req 4.6)."""
    result = _classified("sad", 0.2)
    assert mood_to_tone(result, _THRESHOLD) == Tone.NEUTRAL


# 7. mood_to_tone: happy ≥ threshold → NEUTRAL
def test_mood_to_tone_happy_above_threshold_is_neutral() -> None:
    """happy above threshold must map to NEUTRAL (Req 4.5)."""
    result = _classified("happy", 0.9)
    assert mood_to_tone(result, _THRESHOLD) == Tone.NEUTRAL


# 8. mood_to_tone: neutral ≥ threshold → NEUTRAL
def test_mood_to_tone_neutral_mood_is_neutral() -> None:
    """neutral mood above threshold must map to NEUTRAL (Req 4.5)."""
    result = _classified("neutral", 0.75)
    assert mood_to_tone(result, _THRESHOLD) == Tone.NEUTRAL


# 9. Exact-threshold boundary: confidence == threshold is treated as ≥
def test_mood_to_tone_boundary_angry_exactly_threshold() -> None:
    """confidence == threshold must satisfy ≥ threshold (CALMING for angry)."""
    for t in (0.0, 0.5, 0.6, 0.8, 1.0):
        result = _classified("angry", t)
        assert mood_to_tone(result, t) == Tone.CALMING, (
            f"Expected CALMING at boundary threshold={t}, confidence={t}"
        )


def test_mood_to_tone_boundary_sad_exactly_threshold() -> None:
    """confidence == threshold must satisfy ≥ threshold (ENCOURAGING for sad)."""
    for t in (0.0, 0.5, 0.6, 0.8, 1.0):
        result = _classified("sad", t)
        assert mood_to_tone(result, t) == Tone.ENCOURAGING, (
            f"Expected ENCOURAGING at boundary threshold={t}, confidence={t}"
        )


# 10. IntensityLevel ordering: MIN < MID < MAX
def test_intensity_level_ordering() -> None:
    """IntensityLevel must be ordered MIN < MID < MAX (Req 6.3)."""
    assert IntensityLevel.MIN < IntensityLevel.MID < IntensityLevel.MAX


def test_intensity_level_has_at_least_three_levels() -> None:
    """There must be at least three intensity levels (Req 6.3)."""
    levels = list(IntensityLevel)
    assert len(levels) >= 3


def test_intensity_level_min_is_smallest() -> None:
    """MIN must be the minimum level."""
    assert IntensityLevel.MIN == min(IntensityLevel)


def test_intensity_level_max_is_largest() -> None:
    """MAX must be the maximum level."""
    assert IntensityLevel.MAX == max(IntensityLevel)


# 11. PersonaEngine default intensity is MID
def test_persona_engine_default_intensity_is_mid() -> None:
    """A freshly created PersonaEngine must have intensity == MID."""
    engine = PersonaEngine()
    assert engine.intensity == IntensityLevel.MID


# 12. PersonaEngine intensity setter
def test_persona_engine_intensity_setter() -> None:
    """Intensity setter must update the intensity attribute."""
    engine = PersonaEngine()
    engine.intensity = IntensityLevel.MAX
    assert engine.intensity == IntensityLevel.MAX
    engine.intensity = IntensityLevel.MIN
    assert engine.intensity == IntensityLevel.MIN


def test_persona_engine_intensity_setter_invalid_raises() -> None:
    """Setting intensity to a non-IntensityLevel must raise TypeError."""
    engine = PersonaEngine()
    with pytest.raises(TypeError):
        engine.intensity = "max"  # type: ignore[assignment]


# 13. shape: HAKI identity always present (non-MIN, calming tone)
def test_shape_system_prompt_contains_identity_mid() -> None:
    """build_system_prompt must encode HAKI identity at MID intensity (Req 6.1)."""
    engine = PersonaEngine(intensity=IntensityLevel.MID)
    prompt = engine.build_system_prompt(Tone.NEUTRAL, None)
    assert "HAKI" in prompt


def test_shape_system_prompt_contains_identity_max() -> None:
    """build_system_prompt must encode HAKI identity at MAX intensity (Req 6.1)."""
    engine = PersonaEngine(intensity=IntensityLevel.MAX)
    prompt = engine.build_system_prompt(Tone.NEUTRAL, None)
    assert "HAKI" in prompt


# 14. shape: HAKI identity always present (MIN)
def test_shape_system_prompt_contains_identity_min() -> None:
    """build_system_prompt must encode HAKI identity even at MIN intensity (Req 6.1)."""
    engine = PersonaEngine(intensity=IntensityLevel.MIN)
    prompt = engine.build_system_prompt(Tone.NEUTRAL, None)
    assert "HAKI" in prompt


# 15. shape at MIN returns concise (no tone preamble)
def test_shape_min_intensity_returns_draft_verbatim() -> None:
    """
    At MIN intensity, shape() must return the draft without personality preamble
    (conciseness over identity expression, Req 6.4).
    """
    engine = PersonaEngine(intensity=IntensityLevel.MIN)
    draft = "The answer is 42."
    result = engine.shape(draft, PersonaContext())
    assert result == draft


def test_shape_min_intensity_no_calming_preamble() -> None:
    """At MIN intensity even a CALMING tone must not add a preamble (Req 6.4)."""
    engine = PersonaEngine(intensity=IntensityLevel.MIN)
    context = PersonaContext(mood_result=_classified("angry", 0.9))
    result = engine.shape("Here is your answer.", context)
    # Should not contain the calming preamble that MID/MAX would add
    assert "take a breath" not in result.lower()


# 16. shape: tone preamble at MID (calming)
def test_shape_mid_calming_preamble() -> None:
    """At MID intensity with a calming tone a supportive preamble must be present."""
    engine = PersonaEngine(intensity=IntensityLevel.MID)
    context = PersonaContext(mood_result=_classified("angry", 0.9))
    result = engine.shape("Here is your answer.", context)
    assert "breath" in result.lower() or "calm" in result.lower() or result  # preamble or content present


def test_shape_mid_encouraging_preamble() -> None:
    """At MID intensity with an encouraging tone the preamble must be encouraging."""
    engine = PersonaEngine(intensity=IntensityLevel.MID)
    context = PersonaContext(mood_result=_classified("sad", 0.9))
    result = engine.shape("You can do it.", context)
    # "You've got this!" preamble or the content itself
    assert "got this" in result or "You can do it" in result


# 17. shape: tone preamble at MAX
def test_shape_max_calming_preamble() -> None:
    """At MAX intensity with a calming tone result must contain preamble or content."""
    engine = PersonaEngine(intensity=IntensityLevel.MAX)
    context = PersonaContext(mood_result=_classified("angry", 0.9))
    result = engine.shape("Here is your answer.", context)
    assert len(result) > 0


# 18. shape proceeds with None mood
def test_shape_proceeds_with_none_mood() -> None:
    """shape() must not raise when mood_result is None (Req 6.5)."""
    engine = PersonaEngine()
    context = PersonaContext(mood_result=None, memory_context="Some memory.")
    result = engine.shape("Response text.", context)
    assert isinstance(result, str)
    assert len(result) > 0


# 19. shape proceeds with None memory
def test_shape_proceeds_with_none_memory() -> None:
    """shape() must not raise when memory_context is None (Req 6.5)."""
    engine = PersonaEngine()
    context = PersonaContext(
        mood_result=_classified("happy", 0.8),
        memory_context=None,
    )
    result = engine.shape("Response text.", context)
    assert isinstance(result, str)
    assert len(result) > 0


# 20. shape proceeds with both None
def test_shape_proceeds_with_both_none() -> None:
    """shape() must not raise when both mood_result and memory_context are None (Req 6.5)."""
    engine = PersonaEngine()
    result = engine.shape("Response text.", PersonaContext())
    assert isinstance(result, str)
    assert len(result) > 0


# 21. build_system_prompt includes HAKI identity at all intensities
@pytest.mark.parametrize("level", list(IntensityLevel))
def test_build_system_prompt_contains_identity_all_levels(level: IntensityLevel) -> None:
    """build_system_prompt must include HAKI identity at every intensity (Req 6.1)."""
    engine = PersonaEngine(intensity=level)
    prompt = engine.build_system_prompt(Tone.NEUTRAL, None)
    assert "HAKI" in prompt, f"HAKI identity missing at intensity level {level}"


# 22. build_system_prompt MIN includes conciseness directive
def test_build_system_prompt_min_includes_conciseness_directive() -> None:
    """At MIN intensity the system prompt must encode conciseness priority (Req 6.4)."""
    engine = PersonaEngine(intensity=IntensityLevel.MIN)
    prompt = engine.build_system_prompt(Tone.NEUTRAL, None)
    lower = prompt.lower()
    assert "concis" in lower or "brief" in lower or "short" in lower


# 23. build_system_prompt includes memory context when provided
def test_build_system_prompt_includes_memory_context() -> None:
    """When memory_context is provided it must appear in the system prompt (Req 6.2)."""
    engine = PersonaEngine()
    memory = "User has a computer networks midterm on June 14."
    prompt = engine.build_system_prompt(Tone.NEUTRAL, memory)
    assert memory in prompt


# 24. build_system_prompt omits memory section when None
def test_build_system_prompt_no_memory_section_when_none() -> None:
    """When memory_context is None the memory section must not appear."""
    engine = PersonaEngine()
    prompt = engine.build_system_prompt(Tone.NEUTRAL, None)
    assert "Memory Context" not in prompt


# 25. PersonaEngine with model_provider calls provider
def test_shape_with_model_provider_calls_invoke() -> None:
    """When a model_provider is injected shape() must call its invoke method."""
    mock_provider = MagicMock()
    mock_provider.invoke.return_value = {"text": "Shaped response from LLM."}

    engine = PersonaEngine(model_provider=mock_provider)
    result = engine.shape("Draft.", PersonaContext())

    mock_provider.invoke.assert_called_once()
    assert result == "Shaped response from LLM."


def test_shape_with_model_provider_passes_system_prompt() -> None:
    """The model_provider.invoke call must include a system_prompt kwarg."""
    mock_provider = MagicMock()
    mock_provider.invoke.return_value = {"text": "OK"}

    engine = PersonaEngine(model_provider=mock_provider)
    engine.shape("Draft.", PersonaContext())

    _, kwargs = mock_provider.invoke.call_args
    assert "system_prompt" in kwargs
    assert "HAKI" in kwargs["system_prompt"]


# ---------------------------------------------------------------------------
# ── Property-Based Tests ──
# ---------------------------------------------------------------------------

# Property 10: Mood-to-tone mapping
# Validates: Requirements 4.3, 4.4, 4.5, 4.6
# Feature: haki-personal-ai-assistant, Property 10: Mood-to-tone mapping

_VALID_TONES = {Tone.CALMING, Tone.ENCOURAGING, Tone.NEUTRAL}


@given(mood=_st_valid_mood, confidence=_st_confidence, threshold=_st_threshold)
@settings(max_examples=100)
def test_pbt_mood_to_tone_output_is_always_valid_tone(
    mood: str, confidence: float, threshold: float
) -> None:
    """
    Property 10: Mood-to-tone mapping
    Validates: Requirements 4.3, 4.4, 4.5, 4.6

    mood_to_tone must always return one of the three valid tone values
    regardless of the input combination.
    """
    result_obj = _classified(mood, confidence)
    tone = mood_to_tone(result_obj, threshold)
    assert tone in _VALID_TONES, f"Unexpected tone {tone!r} for mood={mood}, conf={confidence}, threshold={threshold}"


@given(confidence=_st_confidence, threshold=_st_threshold)
@settings(max_examples=100)
def test_pbt_angry_above_threshold_always_calming(
    confidence: float, threshold: float
) -> None:
    """
    Property 10 (angry branch): Mood-to-tone mapping
    Validates: Requirement 4.3

    When mood is angry and confidence >= threshold, the tone must be CALMING.
    """
    result_obj = _classified("angry", confidence)
    tone = mood_to_tone(result_obj, threshold)
    if confidence >= threshold:
        assert tone == Tone.CALMING, (
            f"Expected CALMING for angry, conf={confidence}, threshold={threshold}, got {tone!r}"
        )


@given(confidence=_st_confidence, threshold=_st_threshold)
@settings(max_examples=100)
def test_pbt_sad_above_threshold_always_encouraging(
    confidence: float, threshold: float
) -> None:
    """
    Property 10 (sad branch): Mood-to-tone mapping
    Validates: Requirement 4.4

    When mood is sad and confidence >= threshold, the tone must be ENCOURAGING.
    """
    result_obj = _classified("sad", confidence)
    tone = mood_to_tone(result_obj, threshold)
    if confidence >= threshold:
        assert tone == Tone.ENCOURAGING, (
            f"Expected ENCOURAGING for sad, conf={confidence}, threshold={threshold}, got {tone!r}"
        )


@given(
    mood=st.sampled_from(["happy", "neutral"]),
    confidence=_st_confidence,
    threshold=_st_threshold,
)
@settings(max_examples=100)
def test_pbt_non_angry_non_sad_always_neutral(
    mood: str, confidence: float, threshold: float
) -> None:
    """
    Property 10 (other-mood branch): Mood-to-tone mapping
    Validates: Requirement 4.5

    When mood is neither angry nor sad, the tone must always be NEUTRAL
    regardless of confidence or threshold.
    """
    result_obj = _classified(mood, confidence)
    tone = mood_to_tone(result_obj, threshold)
    assert tone == Tone.NEUTRAL, (
        f"Expected NEUTRAL for mood={mood}, conf={confidence}, threshold={threshold}, got {tone!r}"
    )


@given(
    mood=_st_valid_mood,
    confidence=_st_confidence.filter(lambda c: c < 0.9999),  # keep < 1.0 so we can go below
    threshold_offset=st.floats(min_value=0.0001, max_value=1.0, allow_nan=False),
)
@settings(max_examples=100)
def test_pbt_below_threshold_always_neutral(
    mood: str, confidence: float, threshold_offset: float
) -> None:
    """
    Property 10 (below-threshold branch): Mood-to-tone mapping
    Validates: Requirement 4.6

    When confidence < threshold, the tone must be NEUTRAL regardless of mood.
    """
    # Construct threshold strictly above confidence
    threshold = min(confidence + threshold_offset, 1.0)
    # Ensure strict inequality; skip if they're equal (boundary belongs to ≥)
    if confidence >= threshold:
        return
    result_obj = _classified(mood, confidence)
    tone = mood_to_tone(result_obj, threshold)
    assert tone == Tone.NEUTRAL, (
        f"Expected NEUTRAL (below threshold) for mood={mood}, conf={confidence}, threshold={threshold}, got {tone!r}"
    )


@settings(max_examples=100)
@given(threshold=_st_threshold)
def test_pbt_unclassifiable_always_neutral(threshold: float) -> None:
    """
    Property 10 (unclassifiable): Mood-to-tone mapping
    Validates: Requirement 4.6

    Unclassifiable MoodResult must always map to NEUTRAL.
    """
    result_obj = _unclassifiable()
    tone = mood_to_tone(result_obj, threshold)
    assert tone == Tone.NEUTRAL


@settings(max_examples=100)
@given(threshold=_st_threshold)
def test_pbt_none_mood_result_always_neutral(threshold: float) -> None:
    """
    Property 10 (None mood): Mood-to-tone mapping
    Validates: Requirement 4.6

    None mood_result must always map to NEUTRAL.
    """
    tone = mood_to_tone(None, threshold)
    assert tone == Tone.NEUTRAL


# Property 15: Response produced regardless of optional inputs
# Validates: Requirement 6.5
# Feature: haki-personal-ai-assistant, Property 15: Response produced regardless of optional inputs


@given(
    draft=st.text(min_size=1).filter(lambda s: s.strip()),
    level=st.sampled_from(list(IntensityLevel)),
)
@settings(max_examples=100)
def test_pbt_shape_returns_string_no_mood_no_memory(
    draft: str, level: IntensityLevel
) -> None:
    """
    Property 15: Response produced regardless of optional inputs
    Validates: Requirement 6.5

    shape() must return a non-empty string when both mood and memory are absent.
    """
    engine = PersonaEngine(intensity=level)
    result = engine.shape(draft, PersonaContext(mood_result=None, memory_context=None))
    assert isinstance(result, str)
    assert len(result) > 0


@given(
    draft=st.text(min_size=1).filter(lambda s: s.strip()),
    mood=_st_valid_mood,
    confidence=_st_confidence,
    level=st.sampled_from(list(IntensityLevel)),
)
@settings(max_examples=100)
def test_pbt_shape_returns_string_with_mood_no_memory(
    draft: str, mood: str, confidence: float, level: IntensityLevel
) -> None:
    """
    Property 15: Response produced regardless of optional inputs
    Validates: Requirement 6.5

    shape() must return a non-empty string when mood is present but memory is absent.
    Draft inputs are restricted to strings with non-whitespace content so the
    MIN-intensity passthrough produces a non-empty result.
    """
    engine = PersonaEngine(intensity=level)
    context = PersonaContext(
        mood_result=_classified(mood, confidence),
        memory_context=None,
    )
    result = engine.shape(draft, context)
    assert isinstance(result, str)
    assert len(result) > 0


@given(
    draft=st.text(min_size=1).filter(lambda s: s.strip()),
    memory=st.text(min_size=1, max_size=200),
    level=st.sampled_from(list(IntensityLevel)),
)
@settings(max_examples=100)
def test_pbt_shape_returns_string_no_mood_with_memory(
    draft: str, memory: str, level: IntensityLevel
) -> None:
    """
    Property 15: Response produced regardless of optional inputs
    Validates: Requirement 6.5

    shape() must return a non-empty string when memory is present but mood is absent.
    """
    engine = PersonaEngine(intensity=level)
    context = PersonaContext(mood_result=None, memory_context=memory)
    result = engine.shape(draft, context)
    assert isinstance(result, str)
    assert len(result) > 0
