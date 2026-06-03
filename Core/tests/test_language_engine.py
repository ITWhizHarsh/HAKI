"""
Unit and property-based tests for the Language_Engine subsystem.

Feature: haki-personal-ai-assistant
Requirements: 5.1, 5.2, 5.3, 5.4, 5.5

Test catalogue
--------------
1.  Devanagari script detection — all tokens hindi, composition hindi
2.  Romanized Hindi lexicon — recognised as hindi tokens, composition hindi
3.  Pure English — all english, composition english
4.  Hinglish mix — hindi + english tokens, composition hinglish
5.  Uninterpretable input — empty string / gibberish raises UninterpretableInputError
6.  generate_constraints for hinglish — requires mixed output
7.  generate_constraints for hindi — Hindi-only instruction
8.  generate_constraints for english — English-only instruction
9.  get_tts_routing — each token gets the correct voice assignment
10. No language picker — hindi/english/hinglish accepted without rejection
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from core.language import (
    AnalysisResult,
    LanguageEngine,
    TokenOrigin,
    UninterpretableInputError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> LanguageEngine:
    """A fresh LanguageEngine instance for each test."""
    return LanguageEngine()


# ---------------------------------------------------------------------------
# Test 1: Devanagari script detection
# ---------------------------------------------------------------------------


def test_devanagari_all_hindi(engine: LanguageEngine) -> None:
    """
    'मैं ठीक हूँ' → all tokens are hindi-origin, composition is hindi.

    Requirement: 5.1 — Language_Engine accepts Hindi input.
    """
    result = engine.analyze("मैं ठीक हूँ")
    assert result.composition == "hindi"
    for token in result.tokens:
        assert token.origin == "hindi", (
            f"Token '{token.word}' expected origin 'hindi', got '{token.origin}'"
        )


def test_devanagari_single_word(engine: LanguageEngine) -> None:
    """A single Devanagari word is classified as hindi composition."""
    result = engine.analyze("नमस्ते")
    assert result.composition == "hindi"
    assert result.tokens[0].origin == "hindi"


# ---------------------------------------------------------------------------
# Test 2: Romanized Hindi lexicon
# ---------------------------------------------------------------------------


def test_romanized_hindi_composition(engine: LanguageEngine) -> None:
    """
    'kya kar rahe ho' → all tokens recognised as hindi-origin.

    'kya' is in the lexicon; 'kar', 'rahe', 'ho' are also in the lexicon.
    Composition must be hindi.

    Requirement: 5.1.
    """
    result = engine.analyze("kya kar rahe ho")
    assert result.composition == "hindi"
    for token in result.tokens:
        assert token.origin == "hindi", (
            f"Token '{token.word}' expected hindi origin, got '{token.origin}'"
        )


def test_romanized_hindi_individual_words(engine: LanguageEngine) -> None:
    """
    Common romanized Hindi words are each individually tagged as hindi-origin.
    """
    hindi_words = ["aap", "kya", "hai", "main", "nahi", "haan", "theek",
                   "bahut", "abhi", "lekin", "toh", "yeh", "woh", "aur",
                   "bas", "bhai", "yaar"]
    for word in hindi_words:
        result = engine.analyze(word)
        assert result.tokens[0].origin == "hindi", (
            f"Expected '{word}' to be tagged as hindi, got '{result.tokens[0].origin}'"
        )


# ---------------------------------------------------------------------------
# Test 3: Pure English
# ---------------------------------------------------------------------------


def test_pure_english_composition(engine: LanguageEngine) -> None:
    """
    'what are you doing' → all english-origin tokens, composition english.

    Requirement: 5.3 — monolingual English in → English out.
    """
    result = engine.analyze("what are you doing")
    assert result.composition == "english"
    for token in result.tokens:
        assert token.origin == "english", (
            f"Token '{token.word}' expected english origin, got '{token.origin}'"
        )


def test_pure_english_longer_sentence(engine: LanguageEngine) -> None:
    """Longer English sentence classified correctly."""
    result = engine.analyze("I am going to the library to study for my exam")
    assert result.composition == "english"


# ---------------------------------------------------------------------------
# Test 4: Hinglish mix
# ---------------------------------------------------------------------------


def test_hinglish_mix_composition(engine: LanguageEngine) -> None:
    """
    'kya time hai' → 'kya' and 'hai' are hindi-origin, 'time' is english-origin.
    Composition must be hinglish.

    Requirement: 5.2 — Language_Engine detects hinglish composition.
    """
    result = engine.analyze("kya time hai")
    assert result.composition == "hinglish"

    origins = {t.word: t.origin for t in result.tokens}
    assert origins.get("kya") == "hindi", f"'kya' should be hindi, got {origins.get('kya')}"
    assert origins.get("time") == "english", f"'time' should be english, got {origins.get('time')}"
    assert origins.get("hai") == "hindi", f"'hai' should be hindi, got {origins.get('hai')}"


def test_hinglish_mix_devanagari_and_latin(engine: LanguageEngine) -> None:
    """
    Mix of Devanagari and English words → hinglish composition.
    """
    # 'schedule' is English, 'मेरा' is Devanagari Hindi
    result = engine.analyze("मेरा schedule kya hai")
    assert result.composition == "hinglish"


# ---------------------------------------------------------------------------
# Test 5: Uninterpretable input
# ---------------------------------------------------------------------------


def test_empty_string_raises(engine: LanguageEngine) -> None:
    """
    Empty string raises UninterpretableInputError (Req 5.5).
    """
    with pytest.raises(UninterpretableInputError):
        engine.analyze("")


def test_whitespace_only_raises(engine: LanguageEngine) -> None:
    """
    Whitespace-only input raises UninterpretableInputError.
    """
    with pytest.raises(UninterpretableInputError):
        engine.analyze("   \t\n  ")


def test_gibberish_raises(engine: LanguageEngine) -> None:
    """
    A string that contains only non-Latin, non-Devanagari, non-lexicon characters
    raises UninterpretableInputError (Req 5.5).

    Pure Latin gibberish like 'xzqwp' is still tagged as English-origin
    (Latin script → English), so it does NOT raise.  Only tokens that
    are genuinely unclassifiable produce an 'unknown' composition.

    A string of only punctuation/symbols yields no tokens after tokenization
    → empty token list → raises UninterpretableInputError.
    """
    with pytest.raises(UninterpretableInputError):
        engine.analyze("!@#$%^&*()")  # symbols only, no word tokens produced


def test_multiple_gibberish_tokens_raise(engine: LanguageEngine) -> None:
    """Strings that tokenize to zero word tokens raise UninterpretableInputError."""
    with pytest.raises(UninterpretableInputError):
        engine.analyze("--- *** >>>")  # punctuation only, no word tokens


# ---------------------------------------------------------------------------
# Test 6: generate_constraints for hinglish
# ---------------------------------------------------------------------------


def test_generate_constraints_hinglish(engine: LanguageEngine) -> None:
    """
    Hinglish composition → constraint requires ≥1 Hindi-origin word
    AND ≥1 English-origin word, and must NOT be entirely Hindi.

    Requirement: 5.2.
    """
    result = engine.analyze("kya time hai")
    constraints = engine.generate_constraints(result)

    assert constraints["composition"] == "hinglish"
    instr = constraints["language_instruction"]
    # Must mention both languages and the "not entirely Hindi" rule
    assert "Hindi" in instr or "hindi" in instr.lower()
    assert "English" in instr or "english" in instr.lower()
    assert "NOT" in instr or "not" in instr.lower()


def test_generate_constraints_hinglish_exact_keys(engine: LanguageEngine) -> None:
    """generate_constraints returns exactly the expected dict keys."""
    result = engine.analyze("kya time hai")
    constraints = engine.generate_constraints(result)

    assert set(constraints.keys()) == {"language_instruction", "composition"}


# ---------------------------------------------------------------------------
# Test 7: generate_constraints for hindi
# ---------------------------------------------------------------------------


def test_generate_constraints_hindi(engine: LanguageEngine) -> None:
    """
    Hindi composition → constraint instructs LLM to respond entirely in Hindi.

    Requirement: 5.3.
    """
    result = engine.analyze("मैं ठीक हूँ")
    constraints = engine.generate_constraints(result)

    assert constraints["composition"] == "hindi"
    instr = constraints["language_instruction"]
    assert "Hindi" in instr or "hindi" in instr.lower()


def test_generate_constraints_hindi_romanized(engine: LanguageEngine) -> None:
    """Romanized Hindi input also yields hindi constraint."""
    result = engine.analyze("aap kya kar rahe ho")
    constraints = engine.generate_constraints(result)
    assert constraints["composition"] == "hindi"


# ---------------------------------------------------------------------------
# Test 8: generate_constraints for english
# ---------------------------------------------------------------------------


def test_generate_constraints_english(engine: LanguageEngine) -> None:
    """
    English composition → constraint instructs LLM to respond entirely in English.

    Requirement: 5.3.
    """
    result = engine.analyze("what are you doing today")
    constraints = engine.generate_constraints(result)

    assert constraints["composition"] == "english"
    instr = constraints["language_instruction"]
    assert "English" in instr or "english" in instr.lower()


# ---------------------------------------------------------------------------
# Test 9: get_tts_routing
# ---------------------------------------------------------------------------


def test_get_tts_routing_correct_voices(engine: LanguageEngine) -> None:
    """
    'kya time hai' → routing assigns 'hindi' voice to kya/hai, 'english' to time.

    Requirement: 5.4.
    """
    result = engine.analyze("kya time hai")
    routing = engine.get_tts_routing(result)

    routing_map = {entry["word"]: entry for entry in routing}

    assert routing_map["kya"]["voice"] == "hindi"
    assert routing_map["hai"]["voice"] == "hindi"
    assert routing_map["time"]["voice"] == "english"


def test_get_tts_routing_structure(engine: LanguageEngine) -> None:
    """Each routing entry has 'word', 'origin', and 'voice' keys."""
    result = engine.analyze("what are you doing")
    routing = engine.get_tts_routing(result)

    for entry in routing:
        assert "word" in entry
        assert "origin" in entry
        assert "voice" in entry
        assert entry["voice"] in ("hindi", "english")


def test_get_tts_routing_devanagari(engine: LanguageEngine) -> None:
    """Devanagari tokens get 'hindi' voice in TTS routing."""
    result = engine.analyze("मैं ठीक हूँ")
    routing = engine.get_tts_routing(result)
    for entry in routing:
        assert entry["voice"] == "hindi", (
            f"Expected 'hindi' voice for Devanagari token '{entry['word']}', "
            f"got '{entry['voice']}'"
        )


def test_get_tts_routing_unknown_defaults_to_english(engine: LanguageEngine) -> None:
    """
    An unknown-origin token in an otherwise English or Hindi context
    should get 'english' voice as the safe default.
    """
    # Manually build a result with an unknown token embedded in English context
    tokens = [
        TokenOrigin(word="hello", origin="english"),
        TokenOrigin(word="xzqwp", origin="unknown"),
        TokenOrigin(word="world", origin="english"),
    ]
    # composition is "english" because there are English tokens (unknown ignored)
    result = AnalysisResult(composition="english", tokens=tokens)
    routing = engine.get_tts_routing(result)

    routing_map = {entry["word"]: entry for entry in routing}
    assert routing_map["hello"]["voice"] == "english"
    assert routing_map["xzqwp"]["voice"] == "english"  # safe default for unknown
    assert routing_map["world"]["voice"] == "english"


def test_get_tts_routing_count_matches_tokens(engine: LanguageEngine) -> None:
    """get_tts_routing returns one entry per token."""
    result = engine.analyze("kya time hai abhi")
    routing = engine.get_tts_routing(result)
    assert len(routing) == len(result.tokens)


# ---------------------------------------------------------------------------
# Test 10: No language picker — all valid compositions are accepted
# ---------------------------------------------------------------------------


def test_hindi_input_accepted_no_rejection(engine: LanguageEngine) -> None:
    """
    Hindi input is accepted without rejection (Requirement 5.1).
    analyze() must not raise for pure Hindi.
    """
    result = engine.analyze("kya aap theek hain")
    assert result.composition in ("hindi", "hinglish")  # all known Hindi words


def test_english_input_accepted_no_rejection(engine: LanguageEngine) -> None:
    """
    English input is accepted without rejection (Requirement 5.1).
    """
    result = engine.analyze("can you help me please")
    assert result.composition == "english"


def test_hinglish_input_accepted_no_rejection(engine: LanguageEngine) -> None:
    """
    Hinglish input is accepted without rejection (Requirement 5.1).
    """
    result = engine.analyze("bhai what is the time")
    assert result.composition in ("hindi", "hinglish")


def test_devanagari_hindi_accepted_no_rejection(engine: LanguageEngine) -> None:
    """
    Devanagari Hindi is accepted without prompting the user to pick a language.
    """
    result = engine.analyze("नमस्ते कैसे हो")
    assert result.composition == "hindi"


# ---------------------------------------------------------------------------
# build_language_prompt_segment helper
# ---------------------------------------------------------------------------


def test_build_language_prompt_segment_contains_instruction(engine: LanguageEngine) -> None:
    """build_language_prompt_segment wraps the instruction in a labelled segment."""
    constraints = {
        "language_instruction": "Respond entirely in English.",
        "composition": "english",
    }
    segment = engine.build_language_prompt_segment(constraints)
    assert "Respond entirely in English." in segment
    assert "[Language Instruction]" in segment


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Build a small strategy of clearly-English sentences
_ENGLISH_SENTENCES = st.sampled_from([
    "what time is it",
    "hello how are you",
    "can you help me",
    "tell me a story",
    "open the calendar",
    "search for the latest news",
    "play some music please",
    "remind me tomorrow morning",
])

# Build a strategy of clearly-romanized Hindi sentences
_HINDI_SENTENCES = st.sampled_from([
    "kya kar rahe ho",
    "aap kahan hain",
    "main theek hoon",
    "bahut acha hai",
    "haan bilkul",
    "abhi nahi lekin baad mein",
    "bhai kya baat hai",
    "toh kya karo",
])

# Hinglish: sentences that mix known Hindi words with English words
_HINGLISH_SENTENCES = st.sampled_from([
    "kya time hai",
    "bhai what is happening",
    "yeh calendar open karo",
    "main study kar raha hoon",
    "abhi meeting hai",
    "kya you can help me",
    "aur what else",
    "bahut good idea",
])


@given(sentence=_ENGLISH_SENTENCES)
@settings(max_examples=50)
def test_property_english_sentences_not_rejected(sentence: str) -> None:
    """
    Feature: haki-personal-ai-assistant, Property 11: Language acceptance

    English sentences must be accepted (not raise UninterpretableInputError).
    Validates: Requirements 5.1
    """
    eng = LanguageEngine()
    result = eng.analyze(sentence)
    assert result.composition == "english"


@given(sentence=_HINDI_SENTENCES)
@settings(max_examples=50)
def test_property_hindi_sentences_not_rejected(sentence: str) -> None:
    """
    Feature: haki-personal-ai-assistant, Property 11: Language acceptance

    Romanized Hindi sentences must be accepted and classified as hindi.
    Validates: Requirements 5.1
    """
    eng = LanguageEngine()
    result = eng.analyze(sentence)
    assert result.composition == "hindi"


@given(sentence=_HINGLISH_SENTENCES)
@settings(max_examples=50)
def test_property_hinglish_composition(sentence: str) -> None:
    """
    Feature: haki-personal-ai-assistant, Property 12: Hinglish response composition

    Hinglish sentences must be classified as hinglish, and generate_constraints
    must return a hinglish constraint requiring both languages.
    Validates: Requirements 5.2
    """
    eng = LanguageEngine()
    result = eng.analyze(sentence)
    assert result.composition == "hinglish"
    constraints = eng.generate_constraints(result)
    assert constraints["composition"] == "hinglish"
    instr = constraints["language_instruction"]
    assert "Hindi" in instr or "hindi" in instr.lower()
    assert "English" in instr or "english" in instr.lower()


@given(sentence=_ENGLISH_SENTENCES | _HINDI_SENTENCES)
@settings(max_examples=50)
def test_property_monolingual_response_composition(sentence: str) -> None:
    """
    Feature: haki-personal-ai-assistant, Property 13: Monolingual response composition

    Monolingual (hindi or english) sentences must generate a single-language
    constraint matching their composition.
    Validates: Requirements 5.3
    """
    eng = LanguageEngine()
    result = eng.analyze(sentence)
    assert result.composition in ("hindi", "english")
    constraints = eng.generate_constraints(result)
    assert constraints["composition"] == result.composition


@given(sentence=_ENGLISH_SENTENCES | _HINDI_SENTENCES | _HINGLISH_SENTENCES)
@settings(max_examples=50)
def test_property_tts_routing_voices_are_valid(sentence: str) -> None:
    """
    Feature: haki-personal-ai-assistant, Property 14: Per-word pronunciation routing

    Every token in a TTS routing result must have a valid voice value
    ('hindi' or 'english').
    Validates: Requirements 5.4
    """
    eng = LanguageEngine()
    result = eng.analyze(sentence)
    routing = eng.get_tts_routing(result)
    for entry in routing:
        assert entry["voice"] in ("hindi", "english"), (
            f"Invalid voice '{entry['voice']}' for word '{entry['word']}'"
        )
        assert entry["origin"] in ("hindi", "english", "unknown"), (
            f"Invalid origin '{entry['origin']}' for word '{entry['word']}'"
        )
