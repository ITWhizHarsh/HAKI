"""
Language Engine — language composition analysis, generation constraints,
and per-token TTS pronunciation routing.

This module is the Language_Engine subsystem for HAKI. It tokenizes input
text, tags each token with its language origin (hindi / english / unknown),
classifies the overall composition, produces LLM prompt constraints that
enforce the correct response language, and emits a per-token TTS routing map
that the Voice_Engine passes to the TTS subsystem for per-word pronunciation
selection.

Design: Language_Engine, Voice Pipeline (Hinglish).
Requirements: 5.1, 5.2, 5.3, 5.4, 5.5.

Public types
------------
TokenOrigin
    A single token and its detected language origin.

AnalysisResult
    Full analysis: overall composition label + list of TokenOrigin objects.

LanguageComposition
    String-enum alias for the four composition values used in AnalysisResult.

UninterpretableInputError
    Raised by LanguageEngine.analyze() when the composition is ``unknown``
    (i.e. HAKI genuinely cannot determine what language the user used).
    Callers should catch this and respond with "not understood + rephrase".

LanguageEngine
    Main class. Entry points:
      analyze(text)              — tokenize + tag + classify
      generate_constraints(res)  — produce LLM prompt constraint dict
      build_language_prompt_segment(constraints) — format constraint for prompt
      get_tts_routing(res)       — per-token voice routing list

TTS routing note
----------------
``get_tts_routing`` returns a list of dicts that is intended to be passed
*alongside* the response text to the Voice_Engine / TTS subsystem so it can
select the correct pronunciation model on a per-word basis (Req 5.4).
Hindi-origin tokens use the ``"hindi"`` voice; English-origin and unknown
tokens use the ``"english"`` voice (safe default for unknown).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Constants — romanized Hindi lexicon
# ---------------------------------------------------------------------------

# A curated set of common romanized (transliterated) Hindi words.
# Any token whose lowercase form is in this set is tagged as "hindi" origin
# even though it is written in Latin script.
_ROMANIZED_HINDI_LEXICON: frozenset[str] = frozenset(
    [
        # Pronouns / common short words
        "aap", "main", "hum", "tum", "woh", "yeh", "ye", "wo",
        # Verbs / verb roots
        "hai", "hain", "tha", "thi", "ho", "hoga", "hogi", "honge",
        "hoon", "hote", "hoti", "hota",
        "karo", "karna", "kar", "kiya", "kiye", "ki", "karte",
        "bolo", "batao", "samjho", "dekho", "suno", "jao", "aao",
        "jata", "jati", "jate", "aata", "aati", "aate",
        "chahiye", "chahta", "chahti", "chahte",
        # Common question words
        "kya", "kyun", "kaun", "kab", "kahan", "kaise", "kitna", "kitne",
        # Negation / affirmation
        "nahi", "nahin", "nahi", "haan", "ji", "bilkul",
        # Adjectives / adverbs
        "theek", "accha", "acha", "bura", "bada", "chota", "zyada", "kam",
        "bahut", "thoda", "ekdum", "bilkul", "seedha",
        # Time / sequence
        "abhi", "phir", "kal", "aaj", "parso", "pehle", "baad", "jab", "tab",
        # Conjunctions / discourse markers
        "lekin", "aur", "ya", "toh", "kyunki", "isliye",
        "matlab", "matlb", "yani",
        # Common nouns / address terms
        "bhai", "yaar", "dost", "sir", "beta", "beti", "maa", "baap",
        "naam", "kaam", "baat", "cheez", "waqt", "din", "raat",
        # Misc / filler
        "bas", "samajh", "arre", "oye", "acche", "chalo", "achha",
        # Postpositions / particles
        "mein", "ko", "ne",
        # Additional common transliterations
        "rahe", "raha", "rahi", "karenge", "karega", "karegi",
        "bolte", "sochna", "socha",
    ]
)

# Devanagari Unicode block: U+0900 – U+097F
_DEVANAGARI_START = 0x0900
_DEVANAGARI_END = 0x097F


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

# Re-export the string literals as a module-level type alias for clarity.
LanguageComposition = Literal["hindi", "english", "hinglish", "unknown"]


@dataclass
class TokenOrigin:
    """A single token and its detected language origin.

    Attributes
    ----------
    word : str
        The original token text as it appeared in the input.
    origin : Literal["hindi", "english", "unknown"]
        Detected language origin for this token.
    """

    word: str
    origin: Literal["hindi", "english", "unknown"]


@dataclass
class AnalysisResult:
    """Full result of a language composition analysis.

    Attributes
    ----------
    composition : LanguageComposition
        Overall composition of the input text.
    tokens : list[TokenOrigin]
        Per-token origin tags produced during analysis.
    """

    composition: LanguageComposition
    tokens: list[TokenOrigin]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UninterpretableInputError(Exception):
    """
    Raised by :py:meth:`LanguageEngine.analyze` when the composition of the
    input is ``"unknown"`` — i.e. HAKI genuinely cannot determine the language.

    Callers (e.g. the Orchestrator) should catch this and respond with the
    "not understood + prompt rephrase" path (Requirement 5.5).
    """


# ---------------------------------------------------------------------------
# LanguageEngine
# ---------------------------------------------------------------------------


class LanguageEngine:
    """
    Language composition analyser, constraint generator, and TTS router.

    All methods are synchronous and cheap (no model calls in this
    implementation — only script detection and lexicon heuristics).  A
    model-backed layer can be added later as an additional tier in
    :py:meth:`_detect_token_origin` without changing the public interface.

    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5.
    """

    # ------------------------------------------------------------------
    # Task 8.1 — language composition analysis and per-token origin tagging
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> AnalysisResult:
        """
        Tokenize *text* and classify its language composition.

        Steps
        -----
        1. Tokenize into word tokens (punctuation and whitespace stripped).
        2. Tag each token's origin via a layered heuristic:
           a. If any character in the token is in the Devanagari Unicode
              block → ``"hindi"``.
           b. Else if the lowercase token is in the romanized-Hindi lexicon
              → ``"hindi"``.
           c. Else if all non-numeric characters are Latin-script → ``"english"``.
           d. Otherwise → ``"unknown"``.
        3. Derive the composition from the multiset of origins:
           - All hindi tokens → ``"hindi"``
           - All english tokens → ``"english"``
           - Mix of hindi + english → ``"hinglish"``
           - Anything else (all unknown, or empty after tokenization)
             → ``"unknown"``  →  raises :class:`UninterpretableInputError`.

        Parameters
        ----------
        text : str
            Input text from the user (spoken transcript or typed query).

        Returns
        -------
        AnalysisResult
            Composition label + per-token origin tags.

        Raises
        ------
        UninterpretableInputError
            When composition is ``"unknown"`` (Requirement 5.5).
        """
        tokens = self._tokenize(text)

        if not tokens:
            raise UninterpretableInputError(
                "Input is empty or contains no recognizable tokens."
            )

        tagged: list[TokenOrigin] = [
            TokenOrigin(word=tok, origin=self._detect_token_origin(tok))
            for tok in tokens
        ]

        composition = self._classify_composition(tagged)

        if composition == "unknown":
            raise UninterpretableInputError(
                f"Cannot determine language composition for input: {text!r}"
            )

        return AnalysisResult(composition=composition, tokens=tagged)

    # ------------------------------------------------------------------
    # Task 8.2 — generation language constraints
    # ------------------------------------------------------------------

    def generate_constraints(self, analysis_result: AnalysisResult) -> dict:
        """
        Return a prompt constraint dictionary for the LLM based on composition.

        Mapping
        -------
        hinglish → instruct the LLM to respond in Hinglish (≥1 Hindi-origin
                   word + ≥1 English-origin word; never fully Hindi).
        hindi    → instruct the LLM to respond entirely in Hindi.
        english  → instruct the LLM to respond entirely in English.
        unknown  → safe fallback: respond in English.

        Parameters
        ----------
        analysis_result : AnalysisResult
            Result of a prior :py:meth:`analyze` call.

        Returns
        -------
        dict
            ``{"language_instruction": <str>, "composition": <str>}``

        Requirements: 5.2, 5.3.
        """
        composition = analysis_result.composition

        if composition == "hinglish":
            return {
                "language_instruction": (
                    "Respond in Hinglish: your response MUST contain at least one "
                    "Hindi-origin word and at least one English-origin word. "
                    "Do NOT respond entirely in Hindi."
                ),
                "composition": "hinglish",
            }
        elif composition == "hindi":
            return {
                "language_instruction": "Respond entirely in Hindi.",
                "composition": "hindi",
            }
        elif composition == "english":
            return {
                "language_instruction": "Respond entirely in English.",
                "composition": "english",
            }
        else:
            # "unknown" — safe English fallback (should not occur in practice
            # because analyze() raises on unknown, but generate_constraints
            # accepts any AnalysisResult so callers can pass custom objects).
            return {
                "language_instruction": "Respond in English.",
                "composition": "english",
            }

    def build_language_prompt_segment(self, constraints: dict) -> str:
        """
        Format a constraints dict into a plain-text string for insertion into
        an LLM system prompt.

        Parameters
        ----------
        constraints : dict
            Dictionary produced by :py:meth:`generate_constraints`.

        Returns
        -------
        str
            A single-line prompt instruction string, e.g.:
            ``"[Language Instruction] Respond entirely in English."``

        Requirements: 5.2, 5.3.
        """
        instruction = constraints.get("language_instruction", "Respond in English.")
        return f"[Language Instruction] {instruction}"

    # ------------------------------------------------------------------
    # Task 8.3 — per-token TTS pronunciation routing
    # ------------------------------------------------------------------

    def get_tts_routing(self, analysis_result: AnalysisResult) -> list[dict]:
        """
        Return a per-token TTS routing list for the Voice_Engine / TTS subsystem.

        Each entry is a dict::

            {"word": <str>, "origin": <str>, "voice": <"hindi"|"english">}

        Routing rules
        -------------
        - Hindi-origin tokens  → ``"voice": "hindi"``
        - English-origin tokens → ``"voice": "english"``
        - Unknown-origin tokens → ``"voice": "english"``  (safe default)

        This map is passed **alongside the response text** to the Voice_Engine
        so that the TTS subsystem can select the correct pronunciation model
        on a per-word basis (Requirement 5.4).

        Parameters
        ----------
        analysis_result : AnalysisResult
            Result of a prior :py:meth:`analyze` call.

        Returns
        -------
        list[dict]
            Ordered list of per-token routing instructions, one entry per
            token in ``analysis_result.tokens``.

        Requirements: 5.4.
        """
        routing: list[dict] = []
        for token in analysis_result.tokens:
            voice = "hindi" if token.origin == "hindi" else "english"
            routing.append(
                {
                    "word": token.word,
                    "origin": token.origin,
                    "voice": voice,
                }
            )
        return routing

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Split *text* into word tokens, discarding punctuation and whitespace.

        Returns an empty list for blank/whitespace-only input.
        """
        # Use regex to find sequences of word characters (letters, digits,
        # combining marks).  This naturally handles Unicode Devanagari.
        return re.findall(r"[\w\u0900-\u097F]+", text, re.UNICODE)

    @staticmethod
    def _is_devanagari_token(token: str) -> bool:
        """Return True if *any* character in the token is in the Devanagari block."""
        return any(
            _DEVANAGARI_START <= ord(ch) <= _DEVANAGARI_END for ch in token
        )

    @staticmethod
    def _is_latin_token(token: str) -> bool:
        """
        Return True if the token consists entirely of Latin-script letters
        (and optional digits/underscores — common in mixed Hinglish tokens).

        We check the Unicode category of each letter character; Latin letters
        have category starting with "L" and their script is "Latin" per
        unicodedata.  For simplicity we accept any token whose *letter*
        characters are all in the Latin script ranges (Basic Latin, Latin-1
        Supplement, etc., U+0000–U+024F broadly).
        """
        for ch in token:
            if ch.isalpha():
                cp = ord(ch)
                # Latin Extended blocks: U+0000–U+024F
                if cp > 0x024F:
                    return False
        return True

    @classmethod
    def _detect_token_origin(cls, token: str) -> Literal["hindi", "english", "unknown"]:
        """
        Detect the language origin of a single token via layered heuristics.

        Layer 1 — Script detection:
            Any Devanagari character → ``"hindi"``.

        Layer 2 — Romanized-Hindi lexicon:
            Lowercase token in :data:`_ROMANIZED_HINDI_LEXICON` → ``"hindi"``.

        Layer 3 — Latin script:
            All letter characters are Latin → ``"english"``.

        Fallback:
            ``"unknown"`` for anything else (numerics-only, symbols, etc.).
        """
        if cls._is_devanagari_token(token):
            return "hindi"

        lower = token.lower()
        if lower in _ROMANIZED_HINDI_LEXICON:
            return "hindi"

        if cls._is_latin_token(token):
            return "english"

        return "unknown"

    @staticmethod
    def _classify_composition(
        tagged: list[TokenOrigin],
    ) -> LanguageComposition:
        """
        Derive the overall composition from a list of tagged tokens.

        Rules
        -----
        - No tokens (empty list)         → ``"unknown"``
        - All origins are ``"hindi"``    → ``"hindi"``
        - All origins are ``"english"``  → ``"english"``
        - Mix of hindi + english present → ``"hinglish"``
          (unknown tokens are ignored for this determination when
          at least one definitive language is present)
        - All origins are ``"unknown"``  → ``"unknown"``
        """
        if not tagged:
            return "unknown"

        origins = {t.origin for t in tagged}
        has_hindi = "hindi" in origins
        has_english = "english" in origins
        has_only_unknown = origins == {"unknown"}

        if has_only_unknown:
            return "unknown"
        if has_hindi and has_english:
            return "hinglish"
        if has_hindi:
            return "hindi"
        if has_english:
            return "english"
        # Mixed unknown + one language: treat as that language
        return "unknown"
