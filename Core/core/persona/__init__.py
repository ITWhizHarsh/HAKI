"""
Persona Engine sub-package.

Shapes HAKI responses with a consistent personality identity, mood-driven
tone, configurable intensity, and memory context integration.

Design reference: Persona_Engine.
Requirements: 4.3, 4.4, 4.5, 4.6, 6.1, 6.2, 6.3, 6.4, 6.5.
"""

from .persona_engine import (
    IntensityLevel,
    PersonaContext,
    PersonaEngine,
    Tone,
    mood_to_tone,
)

__all__ = [
    "IntensityLevel",
    "PersonaContext",
    "PersonaEngine",
    "Tone",
    "mood_to_tone",
]
