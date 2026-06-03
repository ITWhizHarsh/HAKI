"""
Mood Detector sub-package.

Owns acoustic mood classification from prosodic features, confidence gating,
and duration-based unclassifiable results.

Design reference: Mood_Detector.
Requirements: 4.1, 4.7, 4.8.
"""

from .mood_detector import (
    MoodDetector,
    MoodResult,
    VALID_MOODS,
    MIN_DURATION_MS,
    DEFAULT_THRESHOLD,
)

__all__ = [
    "MoodDetector",
    "MoodResult",
    "VALID_MOODS",
    "MIN_DURATION_MS",
    "DEFAULT_THRESHOLD",
]
