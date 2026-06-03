"""
Mood Detector — prosodic mood classification with confidence and duration gating.

This module is the Mood_Detector subsystem for HAKI. It classifies a single
primary mood from vocal/prosodic features passed as an already-extracted
feature dict, gates on minimum speech duration, routes through the Model
Provider's ``mood`` capability for classification, and emits exactly one
``MoodResult`` per request.

Audio feature extraction from raw audio bytes is the Voice_Engine's
responsibility; this module only consumes the pre-extracted feature dict.

Design: Mood_Detector.
Requirements: 4.1, 4.7, 4.8.

Public types
------------
MoodResult
    Either ``{ primary_mood: str, confidence: float }`` or
    ``{ unclassifiable: True }``.

MoodDetector
    Main class. Entry point:
      classify(audio_features, duration_ms) -> MoodResult
    Configurable:
      threshold: float  — confidence threshold, default 0.6, range [0.0, 1.0]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.model_provider import (
    Capability,
    ModelProvider,
    ModelProviderRegistry,
    StubModelProvider,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The set of valid primary mood labels.
VALID_MOODS: frozenset[str] = frozenset({"angry", "sad", "happy", "neutral"})

#: Minimum speech duration in milliseconds required for classification (Req 4.1, 4.7).
MIN_DURATION_MS: float = 1000.0

#: Default confidence threshold (Req 4.2).
DEFAULT_THRESHOLD: float = 0.6


# ---------------------------------------------------------------------------
# MoodResult
# ---------------------------------------------------------------------------


@dataclass
class MoodResult:
    """
    Result of a mood classification request.

    Two mutually exclusive states (Req 4.8):

    Classified
        ``primary_mood`` is one of {angry, sad, happy, neutral} and
        ``confidence`` is in [0.0, 1.0].  ``unclassifiable`` is False.

    Unclassifiable
        ``unclassifiable`` is True.  ``primary_mood`` is None and
        ``confidence`` is None.  Returned when the clip is shorter than
        1 second (Req 4.7).

    Parameters
    ----------
    primary_mood : str | None
        Classified mood label, or None for unclassifiable results.
    confidence : float | None
        Classification confidence in [0.0, 1.0], or None for unclassifiable.
    unclassifiable : bool
        True iff the request could not be classified (e.g. too short).
    """

    primary_mood: str | None = None
    confidence: float | None = None
    unclassifiable: bool = False

    def __post_init__(self) -> None:
        if self.unclassifiable:
            # Unclassifiable state: no mood or confidence should be set.
            if self.primary_mood is not None or self.confidence is not None:
                raise ValueError(
                    "MoodResult: primary_mood and confidence must be None "
                    "when unclassifiable=True."
                )
        else:
            # Classified state: both must be present and valid.
            if self.primary_mood not in VALID_MOODS:
                raise ValueError(
                    f"MoodResult: primary_mood must be one of {sorted(VALID_MOODS)}, "
                    f"got {self.primary_mood!r}."
                )
            if self.confidence is None:
                raise ValueError("MoodResult: confidence must not be None when classified.")
            if not (0.0 <= self.confidence <= 1.0):
                raise ValueError(
                    f"MoodResult: confidence must be in [0.0, 1.0], got {self.confidence!r}."
                )

    @classmethod
    def classified(cls, primary_mood: str, confidence: float) -> "MoodResult":
        """
        Convenience constructor for a classified result.

        Parameters
        ----------
        primary_mood : str
            One of {angry, sad, happy, neutral}.
        confidence : float
            Confidence value in [0.0, 1.0].
        """
        return cls(primary_mood=primary_mood, confidence=confidence, unclassifiable=False)

    @classmethod
    def unclassifiable_result(cls) -> "MoodResult":
        """
        Convenience constructor for an unclassifiable result (Req 4.7).
        """
        return cls(primary_mood=None, confidence=None, unclassifiable=True)

    def is_classified(self) -> bool:
        """Return True iff this result carries a primary mood."""
        return not self.unclassifiable


# ---------------------------------------------------------------------------
# MoodDetector
# ---------------------------------------------------------------------------


class MoodDetector:
    """
    Classify one primary mood from prosodic audio features via the Model Provider.

    The detector gates on minimum speech duration (Req 4.1, 4.7), routes the
    feature dict through the Model Provider's ``mood`` capability for acoustic
    classification, and emits exactly one ``MoodResult`` per call (Req 4.8).

    Parameters
    ----------
    model_provider : ModelProvider | None
        The mood ModelProvider backend to use.  When None (default), a
        ``StubModelProvider`` backed by a fresh ``ModelProviderRegistry``
        is created, so the detector works out of the box without a real
        model backend.
    threshold : float
        Configurable confidence threshold in [0.0, 1.0], default 0.6 (Req 4.2).
        Values outside this range raise ``ValueError`` at construction time.

    Design: Mood_Detector.
    Requirements: 4.1, 4.2, 4.7, 4.8.
    """

    def __init__(
        self,
        model_provider: ModelProvider | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self._threshold = self._validated_threshold(threshold)
        if model_provider is None:
            registry = ModelProviderRegistry()
            self._model_provider: ModelProvider = StubModelProvider(
                Capability.MOOD, registry
            )
        else:
            self._model_provider = model_provider

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        """Confidence threshold for classification (Req 4.2)."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set a new confidence threshold (Req 4.2); must be in [0.0, 1.0]."""
        self._threshold = self._validated_threshold(value)

    # ------------------------------------------------------------------
    # Primary entry point (Req 4.1, 4.7, 4.8)
    # ------------------------------------------------------------------

    def classify(
        self,
        audio_features: dict[str, Any],
        duration_ms: float,
    ) -> MoodResult:
        """
        Classify the primary mood from prosodic audio features.

        Emits exactly one ``MoodResult`` per call (Req 4.8):
        - Returns ``MoodResult(unclassifiable=True)`` when ``duration_ms < 1000``
          (Req 4.7).
        - Otherwise, routes the feature dict through the Model Provider and
          returns a classified ``MoodResult(primary_mood, confidence)`` (Req 4.1).

        Parameters
        ----------
        audio_features : dict[str, Any]
            Pre-extracted prosodic features. Expected keys include at minimum:
            ``pitch_mean``, ``pitch_std``, ``volume_mean``, ``volume_std``.
            Additional derived features (zero-crossing rate, spectral centroid,
            etc.) may also be present.
        duration_ms : float
            Duration of the captured speech clip in milliseconds.

        Returns
        -------
        MoodResult
            Exactly one result per call (Req 4.8).
        """
        # Duration gate (Req 4.1, 4.7)
        if duration_ms < MIN_DURATION_MS:
            return MoodResult.unclassifiable_result()

        # Route through Model Provider (Req 4.1)
        raw = self._model_provider.invoke(
            audio_features,
            duration_ms=duration_ms,
        )

        # Interpret the provider's response into a MoodResult.
        return self._interpret_response(raw)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _interpret_response(self, raw: Any) -> MoodResult:
        """
        Convert the raw Model Provider response into a ``MoodResult``.

        The stub backend echoes the input dict back with metadata.  A real
        backend is expected to return a dict with ``"primary_mood"`` and
        ``"confidence"`` keys.

        When the provider response includes a valid ``primary_mood`` and a
        numeric ``confidence``, those are used directly.  Otherwise, a
        deterministic fallback derives the mood from pitch/volume heuristics
        so the module is testable without a live model.

        Returns
        -------
        MoodResult
            Always a valid classified result (never unclassifiable here,
            since the duration gate already handled the short-clip case).
        """
        if isinstance(raw, dict):
            primary_mood = raw.get("primary_mood")
            confidence_raw = raw.get("confidence")

            # If the provider returned a valid mood and confidence, use them.
            if (
                primary_mood in VALID_MOODS
                and confidence_raw is not None
                and isinstance(confidence_raw, (int, float))
                and 0.0 <= float(confidence_raw) <= 1.0
            ):
                return MoodResult.classified(
                    primary_mood=primary_mood,
                    confidence=float(confidence_raw),
                )

            # Stub / fallback: derive mood from prosodic heuristics embedded
            # in the echoed audio_features dict.
            features = raw.get("input", raw)
            if isinstance(features, dict):
                return self._heuristic_classify(features)

        # Ultimate fallback — neutral with minimum confidence.
        return MoodResult.classified(primary_mood="neutral", confidence=0.0)

    @staticmethod
    def _heuristic_classify(features: dict[str, Any]) -> MoodResult:
        """
        Rule-based prosodic classifier used when the Model Provider stub does
        not return a structured ``primary_mood`` / ``confidence`` response.

        This is an intentional, transparent fallback so unit tests can run
        without a live model.  The rules are based on well-known prosodic
        indicators (Req 4.1):

        - High pitch mean AND high volume mean → angry (high arousal, negative)
        - High pitch mean AND low volume mean → happy (high arousal, positive)
        - Low pitch mean AND low volume mean → sad (low arousal, negative)
        - Otherwise → neutral

        Thresholds are chosen for normalized prosodic features (Hz / dB):
        - pitch_mean > 200 Hz  → high pitch
        - volume_mean > -20 dB → high volume  (RMS energy closer to 0 dB = louder)

        Confidence is a simple normalized product of the feature strengths.
        """
        try:
            pitch_mean = float(features.get("pitch_mean", 150.0))
            volume_mean = float(features.get("volume_mean", -30.0))
        except (TypeError, ValueError):
            return MoodResult.classified(primary_mood="neutral", confidence=0.5)

        high_pitch = pitch_mean > 200.0
        high_volume = volume_mean > -20.0

        # Compute a simple confidence as the normalized distance from the
        # neutral zone.
        pitch_strength = min(abs(pitch_mean - 175.0) / 175.0, 1.0)
        volume_strength = min(abs(volume_mean + 25.0) / 25.0, 1.0)
        confidence = round(min((pitch_strength + volume_strength) / 2.0, 1.0), 4)
        # Ensure confidence is always at least 0.5 for a classified result
        # so it is genuinely usable.
        confidence = max(confidence, 0.5)

        if high_pitch and high_volume:
            mood = "angry"
        elif high_pitch and not high_volume:
            mood = "happy"
        elif not high_pitch and high_volume:
            mood = "angry"  # low pitch + loud = aggressive angry variant
        else:
            mood = "sad" if pitch_mean < 150.0 else "neutral"

        return MoodResult.classified(primary_mood=mood, confidence=confidence)

    @staticmethod
    def _validated_threshold(value: float) -> float:
        """Validate and return a threshold value; raise ValueError if out of range."""
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"threshold must be a float, got {type(value).__name__!r}."
            )
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError(
                f"threshold must be in [0.0, 1.0], got {value!r}."
            )
        return float(value)
