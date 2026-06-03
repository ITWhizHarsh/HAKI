"""
Unit tests for the Mood_Detector subsystem.

Feature: haki-personal-ai-assistant
Requirements: 4.1, 4.2, 4.7, 4.8

Test catalogue
--------------
1.  Duration gate — clips < 1 s return unclassifiable (Req 4.7)
2.  Duration gate — clips >= 1 s return a classified result (Req 4.1)
3.  Exactly one result per request (Req 4.8)
4.  Threshold default is 0.6 (Req 4.2)
5.  Threshold is configurable in [0.0, 1.0] (Req 4.2)
6.  Invalid threshold raises ValueError
7.  Output contract — primary_mood always in {angry, sad, happy, neutral} (Req 4.1)
8.  Output contract — confidence always in [0.0, 1.0] when classified (Req 4.1)
9.  Unclassifiable result has no primary_mood or confidence (Req 4.7)
10. Classified result is not unclassifiable (Req 4.8)
11. MoodResult dataclass invariants
12. Custom ModelProvider response is respected when well-formed
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from core.mood import DEFAULT_THRESHOLD, MIN_DURATION_MS, MoodDetector, MoodResult, VALID_MOODS
from core.model_provider import Capability, ModelProviderRegistry, ModelProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Minimal valid audio_features dict for tests that exercise the classifier.
_FEATURES: dict[str, float] = {
    "pitch_mean": 180.0,
    "pitch_std": 20.0,
    "volume_mean": -25.0,
    "volume_std": 5.0,
}


def _features(**overrides: float) -> dict[str, float]:
    """Return a copy of _FEATURES with the given overrides applied."""
    f = dict(_FEATURES)
    f.update(overrides)
    return f


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def detector() -> MoodDetector:
    """A MoodDetector with default settings and the built-in stub provider."""
    return MoodDetector()


# ---------------------------------------------------------------------------
# Test 1 — Duration gate: clips < 1 s return unclassifiable
# ---------------------------------------------------------------------------


def test_duration_gate_zero_ms_is_unclassifiable(detector: MoodDetector) -> None:
    """
    A clip with 0 ms duration must return MoodResult(unclassifiable=True).

    Requirement: 4.7
    """
    result = detector.classify(_FEATURES, duration_ms=0.0)
    assert result.unclassifiable is True


def test_duration_gate_below_1000ms_is_unclassifiable(detector: MoodDetector) -> None:
    """
    Any clip shorter than 1000 ms must return unclassifiable (Req 4.7).
    """
    for ms in (1.0, 100.0, 500.0, 999.0, 999.9):
        result = detector.classify(_FEATURES, duration_ms=ms)
        assert result.unclassifiable is True, (
            f"Expected unclassifiable for {ms} ms, got classified result."
        )


def test_duration_gate_exactly_999ms_is_unclassifiable(detector: MoodDetector) -> None:
    """999 ms (< 1000) must be unclassifiable."""
    result = detector.classify(_FEATURES, duration_ms=999.0)
    assert result.unclassifiable is True


# ---------------------------------------------------------------------------
# Test 2 — Duration gate: clips >= 1 s return a classified result
# ---------------------------------------------------------------------------


def test_duration_gate_1000ms_is_classified(detector: MoodDetector) -> None:
    """
    A clip of exactly 1000 ms must return a classified result (Req 4.1).
    """
    result = detector.classify(_FEATURES, duration_ms=1000.0)
    assert result.unclassifiable is False
    assert result.primary_mood in VALID_MOODS
    assert result.confidence is not None


def test_duration_gate_above_1000ms_is_classified(detector: MoodDetector) -> None:
    """
    Clips longer than 1000 ms must return classified results (Req 4.1).
    """
    for ms in (1001.0, 2000.0, 5000.0, 10000.0):
        result = detector.classify(_FEATURES, duration_ms=ms)
        assert result.unclassifiable is False, (
            f"Expected classified result for {ms} ms."
        )
        assert result.primary_mood in VALID_MOODS
        assert result.confidence is not None


# ---------------------------------------------------------------------------
# Test 3 — Exactly one result per request
# ---------------------------------------------------------------------------


def test_classify_returns_exactly_one_result_short_clip(detector: MoodDetector) -> None:
    """
    classify() must return exactly one MoodResult for a short clip (Req 4.8).
    """
    result = detector.classify(_FEATURES, duration_ms=500.0)
    assert isinstance(result, MoodResult)


def test_classify_returns_exactly_one_result_long_clip(detector: MoodDetector) -> None:
    """
    classify() must return exactly one MoodResult for a sufficiently long clip (Req 4.8).
    """
    result = detector.classify(_FEATURES, duration_ms=2000.0)
    assert isinstance(result, MoodResult)


def test_classify_always_returns_mood_result(detector: MoodDetector) -> None:
    """
    classify() must return a MoodResult instance for any duration (Req 4.8).
    """
    for ms in (0.0, 500.0, 1000.0, 3000.0):
        result = detector.classify(_FEATURES, duration_ms=ms)
        assert isinstance(result, MoodResult), (
            f"Expected MoodResult for {ms} ms, got {type(result).__name__!r}."
        )


# ---------------------------------------------------------------------------
# Test 4 — Threshold default is 0.6
# ---------------------------------------------------------------------------


def test_threshold_default_is_0_6() -> None:
    """
    A freshly created MoodDetector must have threshold == 0.6 (Req 4.2).
    """
    detector = MoodDetector()
    assert detector.threshold == DEFAULT_THRESHOLD
    assert detector.threshold == 0.6


# ---------------------------------------------------------------------------
# Test 5 — Threshold is configurable in [0.0, 1.0]
# ---------------------------------------------------------------------------


def test_threshold_can_be_set_to_zero() -> None:
    """Threshold can be set to 0.0 (inclusive lower bound, Req 4.2)."""
    detector = MoodDetector(threshold=0.0)
    assert detector.threshold == 0.0


def test_threshold_can_be_set_to_one() -> None:
    """Threshold can be set to 1.0 (inclusive upper bound, Req 4.2)."""
    detector = MoodDetector(threshold=1.0)
    assert detector.threshold == 1.0


def test_threshold_can_be_set_to_mid_value() -> None:
    """Threshold can be set to any value in (0.0, 1.0) (Req 4.2)."""
    for val in (0.1, 0.3, 0.5, 0.6, 0.75, 0.9):
        d = MoodDetector(threshold=val)
        assert d.threshold == pytest.approx(val)


def test_threshold_can_be_updated_via_setter() -> None:
    """Threshold can be updated after construction via the setter (Req 4.2)."""
    detector = MoodDetector()
    assert detector.threshold == 0.6
    detector.threshold = 0.8
    assert detector.threshold == pytest.approx(0.8)
    detector.threshold = 0.3
    assert detector.threshold == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Test 6 — Invalid threshold raises ValueError
# ---------------------------------------------------------------------------


def test_threshold_below_zero_raises() -> None:
    """Threshold < 0.0 must raise ValueError."""
    with pytest.raises(ValueError):
        MoodDetector(threshold=-0.01)


def test_threshold_above_one_raises() -> None:
    """Threshold > 1.0 must raise ValueError."""
    with pytest.raises(ValueError):
        MoodDetector(threshold=1.01)


def test_threshold_setter_below_zero_raises() -> None:
    """Setting threshold < 0.0 via property setter must raise ValueError."""
    d = MoodDetector()
    with pytest.raises(ValueError):
        d.threshold = -0.5


def test_threshold_setter_above_one_raises() -> None:
    """Setting threshold > 1.0 via property setter must raise ValueError."""
    d = MoodDetector()
    with pytest.raises(ValueError):
        d.threshold = 1.5


# ---------------------------------------------------------------------------
# Test 7 — Output contract: primary_mood in {angry, sad, happy, neutral}
# ---------------------------------------------------------------------------


def test_primary_mood_is_valid_label(detector: MoodDetector) -> None:
    """
    primary_mood must be one of {angry, sad, happy, neutral} (Req 4.1).
    """
    result = detector.classify(_FEATURES, duration_ms=1000.0)
    assert result.primary_mood in VALID_MOODS


def test_primary_mood_is_valid_for_various_features(detector: MoodDetector) -> None:
    """
    primary_mood must always be a valid label across different feature combos (Req 4.1).
    """
    feature_variants = [
        _features(pitch_mean=100.0, volume_mean=-40.0),   # low pitch, low vol → sad/neutral
        _features(pitch_mean=250.0, volume_mean=-10.0),   # high pitch, high vol → angry
        _features(pitch_mean=230.0, volume_mean=-35.0),   # high pitch, low vol → happy
        _features(pitch_mean=160.0, volume_mean=-28.0),   # mid pitch, mid vol → neutral
    ]
    for f in feature_variants:
        result = detector.classify(f, duration_ms=2000.0)
        assert result.primary_mood in VALID_MOODS, (
            f"primary_mood {result.primary_mood!r} not in VALID_MOODS for features {f}."
        )


# ---------------------------------------------------------------------------
# Test 8 — Output contract: confidence in [0.0, 1.0] when classified
# ---------------------------------------------------------------------------


def test_confidence_in_range_when_classified(detector: MoodDetector) -> None:
    """
    Confidence must be in [0.0, 1.0] for a classified result (Req 4.1).
    """
    result = detector.classify(_FEATURES, duration_ms=1000.0)
    assert result.confidence is not None
    assert 0.0 <= result.confidence <= 1.0


def test_confidence_in_range_for_various_features(detector: MoodDetector) -> None:
    """
    Confidence must be in [0.0, 1.0] for all feature combinations (Req 4.1).
    """
    feature_variants = [
        _features(pitch_mean=100.0, volume_mean=-40.0),
        _features(pitch_mean=250.0, volume_mean=-10.0),
        _features(pitch_mean=230.0, volume_mean=-35.0),
        _features(pitch_mean=160.0, volume_mean=-28.0),
    ]
    for f in feature_variants:
        result = detector.classify(f, duration_ms=2000.0)
        assert result.confidence is not None
        assert 0.0 <= result.confidence <= 1.0, (
            f"confidence {result.confidence} out of range for features {f}."
        )


# ---------------------------------------------------------------------------
# Test 9 — Unclassifiable result has no primary_mood or confidence
# ---------------------------------------------------------------------------


def test_unclassifiable_has_no_primary_mood(detector: MoodDetector) -> None:
    """
    An unclassifiable MoodResult must have primary_mood == None (Req 4.7).
    """
    result = detector.classify(_FEATURES, duration_ms=500.0)
    assert result.unclassifiable is True
    assert result.primary_mood is None


def test_unclassifiable_has_no_confidence(detector: MoodDetector) -> None:
    """
    An unclassifiable MoodResult must have confidence == None (Req 4.7).
    """
    result = detector.classify(_FEATURES, duration_ms=500.0)
    assert result.unclassifiable is True
    assert result.confidence is None


# ---------------------------------------------------------------------------
# Test 10 — Classified result is not unclassifiable
# ---------------------------------------------------------------------------


def test_classified_result_not_unclassifiable(detector: MoodDetector) -> None:
    """
    A classified MoodResult must have unclassifiable == False (Req 4.8).
    """
    result = detector.classify(_FEATURES, duration_ms=1000.0)
    assert result.unclassifiable is False
    assert result.is_classified() is True


def test_unclassifiable_result_not_classified(detector: MoodDetector) -> None:
    """
    An unclassifiable result must have is_classified() == False (Req 4.8).
    """
    result = detector.classify(_FEATURES, duration_ms=0.0)
    assert result.is_classified() is False


# ---------------------------------------------------------------------------
# Test 11 — MoodResult dataclass invariants
# ---------------------------------------------------------------------------


def test_mood_result_classified_factory() -> None:
    """MoodResult.classified() creates a valid classified result."""
    r = MoodResult.classified("happy", 0.85)
    assert r.primary_mood == "happy"
    assert r.confidence == pytest.approx(0.85)
    assert r.unclassifiable is False
    assert r.is_classified() is True


def test_mood_result_unclassifiable_factory() -> None:
    """MoodResult.unclassifiable_result() creates a valid unclassifiable result."""
    r = MoodResult.unclassifiable_result()
    assert r.unclassifiable is True
    assert r.primary_mood is None
    assert r.confidence is None
    assert r.is_classified() is False


def test_mood_result_invalid_mood_raises() -> None:
    """MoodResult with an invalid primary_mood must raise ValueError."""
    with pytest.raises(ValueError):
        MoodResult(primary_mood="confused", confidence=0.7, unclassifiable=False)


def test_mood_result_confidence_out_of_range_raises() -> None:
    """MoodResult with confidence > 1.0 must raise ValueError."""
    with pytest.raises(ValueError):
        MoodResult(primary_mood="happy", confidence=1.1, unclassifiable=False)


def test_mood_result_confidence_negative_raises() -> None:
    """MoodResult with confidence < 0.0 must raise ValueError."""
    with pytest.raises(ValueError):
        MoodResult(primary_mood="happy", confidence=-0.1, unclassifiable=False)


def test_mood_result_unclassifiable_with_mood_raises() -> None:
    """MoodResult with unclassifiable=True and a primary_mood must raise ValueError."""
    with pytest.raises(ValueError):
        MoodResult(primary_mood="happy", confidence=None, unclassifiable=True)


def test_mood_result_all_valid_moods_accepted() -> None:
    """MoodResult accepts all four valid primary moods."""
    for mood in VALID_MOODS:
        r = MoodResult.classified(mood, 0.75)
        assert r.primary_mood == mood


def test_mood_result_boundary_confidence_zero() -> None:
    """MoodResult accepts confidence == 0.0."""
    r = MoodResult.classified("neutral", 0.0)
    assert r.confidence == 0.0


def test_mood_result_boundary_confidence_one() -> None:
    """MoodResult accepts confidence == 1.0."""
    r = MoodResult.classified("neutral", 1.0)
    assert r.confidence == 1.0


# ---------------------------------------------------------------------------
# Test 12 — Custom ModelProvider response is respected when well-formed
# ---------------------------------------------------------------------------


class _FixedMoodProvider(ModelProvider):
    """
    A minimal ModelProvider stub that returns a pre-configured mood response.
    Used to verify that MoodDetector honours a well-formed provider response.
    """

    def __init__(
        self,
        primary_mood: str,
        confidence: float,
        registry: ModelProviderRegistry | None = None,
    ) -> None:
        if registry is None:
            registry = ModelProviderRegistry()
        super().__init__(registry)
        self._mood = primary_mood
        self._conf = confidence

    @property
    def capability(self) -> Capability:
        return Capability.MOOD

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        return {"primary_mood": self._mood, "confidence": self._conf}

    async def invoke_stream(self, input: Any, **kwargs: Any):
        yield {"primary_mood": self._mood, "confidence": self._conf}


def test_custom_provider_classified_response_is_used() -> None:
    """
    When the Model Provider returns a valid {primary_mood, confidence} response,
    MoodDetector must use it as-is (Req 4.1).
    """
    provider = _FixedMoodProvider(primary_mood="sad", confidence=0.92)
    detector = MoodDetector(model_provider=provider)
    result = detector.classify(_FEATURES, duration_ms=2000.0)

    assert result.primary_mood == "sad"
    assert result.confidence == pytest.approx(0.92)
    assert result.unclassifiable is False


def test_custom_provider_angry_response() -> None:
    """Provider returning 'angry' with confidence 0.8 produces correct MoodResult."""
    provider = _FixedMoodProvider(primary_mood="angry", confidence=0.8)
    detector = MoodDetector(model_provider=provider)
    result = detector.classify(_FEATURES, duration_ms=1500.0)

    assert result.primary_mood == "angry"
    assert result.confidence == pytest.approx(0.8)


def test_custom_provider_happy_response() -> None:
    """Provider returning 'happy' with confidence 0.95 produces correct MoodResult."""
    provider = _FixedMoodProvider(primary_mood="happy", confidence=0.95)
    detector = MoodDetector(model_provider=provider)
    result = detector.classify(_FEATURES, duration_ms=1000.0)

    assert result.primary_mood == "happy"
    assert result.confidence == pytest.approx(0.95)


def test_duration_gate_takes_priority_over_provider() -> None:
    """
    Duration gate must return unclassifiable even when a valid provider is injected
    and duration_ms < 1000 (Req 4.7 takes precedence over Req 4.1 provider call).
    """
    provider = _FixedMoodProvider(primary_mood="happy", confidence=0.99)
    detector = MoodDetector(model_provider=provider)
    result = detector.classify(_FEATURES, duration_ms=500.0)

    # Provider should NOT be consulted for a short clip.
    assert result.unclassifiable is True
    assert result.primary_mood is None
