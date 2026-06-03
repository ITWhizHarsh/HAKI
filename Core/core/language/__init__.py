"""
Language Engine — public interface for HAKI's language subsystem.

Exports
-------
LanguageEngine
    Main class: analyze(), generate_constraints(), build_language_prompt_segment(),
    get_tts_routing().

LanguageComposition
    Type alias for the four composition labels:
    ``"hindi"`` | ``"english"`` | ``"hinglish"`` | ``"unknown"``.

TokenOrigin
    Dataclass carrying a token's word text and detected origin.

AnalysisResult
    Dataclass carrying the composition label and list of TokenOrigin objects.

UninterpretableInputError
    Exception raised when composition is ``"unknown"`` (Req 5.5).
"""

from core.language.language_engine import (
    AnalysisResult,
    LanguageComposition,
    LanguageEngine,
    TokenOrigin,
    UninterpretableInputError,
)

__all__ = [
    "LanguageEngine",
    "LanguageComposition",
    "TokenOrigin",
    "AnalysisResult",
    "UninterpretableInputError",
]
