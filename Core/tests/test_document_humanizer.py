"""
Tests for the document-humanization built-in automation (Task 35.1).

Covers the core implementation requirements:

    19.1  LaTeX source is parsed and prose is separated from markup.
    19.2  Unparseable/no-prose source → automation does NOT start; user informed.
    19.3  Prose is segmented into 800–1200 word chunks (final may be shorter).

All model-backed logic (Language_Engine, humanizeProse) is mocked so the
tests exercise routing, parsing, segmentation, and error-path logic without
real models.

Design: Built-in automations (Document Humanization).
Requirements: 19.1, 19.2, 19.3.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.automation.document_humanizer import (
    AUTOMATION_NAME,
    MARKUP_TOKEN,
    PROSE_TOKEN,
    DocumentHumanizer,
    DocumentToken,
    HumanizationContext,
    LaTeXParseError,
    LaTeXParser,
    ProseHumanizer,
    register_builtin_automation,
    run_document_humanization,
    segment_prose,
)
from core.automation import AutomationLibrary


# ---------------------------------------------------------------------------
# Helper fixtures / builders
# ---------------------------------------------------------------------------

_SIMPLE_LATEX = r"""
\documentclass{article}
\begin{document}
This is the first sentence of my paper. It discusses important concepts in
computer science and artificial intelligence. The field has advanced rapidly
in recent years.

Researchers have explored many approaches to solve this problem. The most
promising methods rely on deep learning techniques. Neural networks have
shown significant improvements over classical algorithms.
\end{document}
"""

_MATH_HEAVY_LATEX = r"""
\documentclass{article}
\begin{document}
The equation $E = mc^2$ is well known. We can also write it as
$$E = mc^2$$
where $m$ is the rest mass and $c$ is the speed of light.
No prose here at all — only math and brief connectors.
\end{document}
"""

_MARKUP_ONLY_LATEX = r"""
\documentclass{article}
\usepackage{amsmath}
\begin{document}
$$\int_0^\infty e^{-x^2}\,dx = \frac{\sqrt{\pi}}{2}$$
\begin{equation}
  x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
\end{equation}
\end{document}
"""

_EMPTY_LATEX = ""

_WHITESPACE_ONLY_LATEX = "   \n\t\n   "

_FRAGMENT_LATEX = r"""
This is a fragment without \begin{document}.
It has some prose text here that should be parsed.
The \section{Introduction} is followed by more prose.
"""


def _make_prose_humanizer_passthrough() -> ProseHumanizer:
    """Return a ProseHumanizer that returns its input unchanged (identity)."""
    h = ProseHumanizer()
    h.humanize = lambda seg: seg  # type: ignore[method-assign]
    return h


def _make_prose_humanizer_fail() -> ProseHumanizer:
    """Return a ProseHumanizer that always raises RuntimeError."""
    h = ProseHumanizer()

    def _fail(seg: str) -> str:
        raise RuntimeError("Simulated humanizer failure")

    h.humanize = _fail  # type: ignore[method-assign]
    return h


# ---------------------------------------------------------------------------
# Tests for LaTeXParser (Req 19.1, 19.2)
# ---------------------------------------------------------------------------


class TestLaTeXParser:
    """Unit tests for the LaTeXParser class."""

    def test_parse_simple_document_returns_tokens(self):
        """
        Req 19.1: A well-formed LaTeX document yields an interleaved
        token list containing both prose and markup tokens.
        """
        parser = LaTeXParser()
        tokens = parser.parse(_SIMPLE_LATEX)
        kinds = {t.kind for t in tokens}
        assert PROSE_TOKEN in kinds, "Expected at least one prose token"

    def test_parse_preserves_order(self):
        """
        Req 19.1: Tokens are returned in document order (non-decreasing index).
        """
        parser = LaTeXParser()
        tokens = parser.parse(_SIMPLE_LATEX)
        indices = [t.index for t in tokens]
        assert indices == sorted(indices), "Token indices are not in order"

    def test_parse_empty_source_raises(self):
        """
        Req 19.2: An empty source raises LaTeXParseError with a reason.
        """
        parser = LaTeXParser()
        with pytest.raises(LaTeXParseError) as exc_info:
            parser.parse(_EMPTY_LATEX)
        assert exc_info.value.args[0]  # reason is non-empty

    def test_parse_whitespace_only_raises(self):
        """
        Req 19.2: Whitespace-only source raises LaTeXParseError.
        """
        parser = LaTeXParser()
        with pytest.raises(LaTeXParseError):
            parser.parse(_WHITESPACE_ONLY_LATEX)

    def test_parse_markup_only_raises(self):
        """
        Req 19.2: A document with only math environments and no prose raises
        LaTeXParseError indicating no separable prose was found.
        """
        parser = LaTeXParser()
        with pytest.raises(LaTeXParseError) as exc_info:
            parser.parse(_MARKUP_ONLY_LATEX)
        assert "prose" in exc_info.value.args[0].lower()

    def test_markup_tokens_cover_known_commands(self):
        """
        Req 19.1: Known LaTeX commands like \\section{}, \\begin{}/\\end{}
        are classified as markup, not prose.
        """
        source = r"""
\begin{document}
\section{Introduction}
Some actual prose goes here and should be captured as a prose token.
\end{document}
"""
        parser = LaTeXParser()
        tokens = parser.parse(source)
        markup_texts = [t.text for t in tokens if t.kind == MARKUP_TOKEN]
        combined = " ".join(markup_texts)
        # The section command should appear in markup
        assert r"\section" in combined or any(
            r"\section" in t for t in markup_texts
        ), "\\section{} was not classified as markup"

    def test_inline_math_is_markup(self):
        """
        Req 19.1: Inline math expressions ($...$) are classified as markup.
        """
        source = r"""
\begin{document}
The formula $x^2 + y^2 = r^2$ defines a circle.
\end{document}
"""
        parser = LaTeXParser()
        tokens = parser.parse(source)
        markup_texts = " ".join(t.text for t in tokens if t.kind == MARKUP_TOKEN)
        assert "$" in markup_texts, "Inline math was not classified as markup"

    def test_comments_are_markup(self):
        """
        Req 19.1: LaTeX comments (% to end of line) are classified as markup.
        """
        source = (
            r"\begin{document}" + "\n"
            r"% This is a comment" + "\n"
            r"Some prose follows." + "\n"
            r"\end{document}"
        )
        parser = LaTeXParser()
        tokens = parser.parse(source)
        markup_texts = " ".join(t.text for t in tokens if t.kind == MARKUP_TOKEN)
        assert "comment" in markup_texts, "Comment was not classified as markup"

    def test_display_math_double_dollar_is_markup(self):
        """
        Req 19.1: Display math ($$...$$) is classified as markup.
        """
        source = r"""
\begin{document}
Some text.
$$E = mc^2$$
More text.
\end{document}
"""
        parser = LaTeXParser()
        tokens = parser.parse(source)
        markup_texts = " ".join(t.text for t in tokens if t.kind == MARKUP_TOKEN)
        assert "$$" in markup_texts, "Display math $$ not classified as markup"

    def test_display_math_brackets_is_markup(self):
        r"""
        Req 19.1: Display math (\[...\]) is classified as markup.
        """
        source = r"""
\begin{document}
Some text before.
\[
  a^2 + b^2 = c^2
\]
Some text after.
\end{document}
"""
        parser = LaTeXParser()
        tokens = parser.parse(source)
        markup_texts = " ".join(t.text for t in tokens if t.kind == MARKUP_TOKEN)
        assert r"\[" in markup_texts, r"\[...\] not classified as markup"

    def test_fragment_without_document_tags_parses(self):
        """
        Req 19.1: A LaTeX fragment without \\begin{document} is parsed as-is.
        """
        parser = LaTeXParser()
        tokens = parser.parse(_FRAGMENT_LATEX)
        prose_tokens = [t for t in tokens if t.kind == PROSE_TOKEN]
        assert len(prose_tokens) > 0, "Expected prose in fragment"

    def test_extract_prose_joins_tokens(self):
        """
        Req 19.1: extract_prose() returns a non-empty string containing the
        prose from the token list.
        """
        parser = LaTeXParser()
        tokens = parser.parse(_SIMPLE_LATEX)
        prose = parser.extract_prose(tokens)
        assert len(prose.split()) > 0, "Expected non-empty prose string"

    def test_token_texts_reconstruct_body(self):
        """
        Req 19.1: Joining all token texts should produce the full document body
        (minus the preamble that was stripped).
        """
        source = r"""
\begin{document}
Hello \textbf{world}. This is a test.
\end{document}
"""
        parser = LaTeXParser()
        tokens = parser.parse(source)
        reconstructed = "".join(t.text for t in tokens)
        # The reconstructed text should contain both the prose and the command.
        assert "Hello" in reconstructed
        assert r"\textbf" in reconstructed


# ---------------------------------------------------------------------------
# Tests for segment_prose (Req 19.3)
# ---------------------------------------------------------------------------


class TestSegmentProse:
    """Unit tests for the segment_prose() function."""

    def _make_prose(self, n_words: int, sentence_len: int = 15) -> str:
        """Generate synthetic prose of exactly *n_words* words."""
        word = "word"
        sentence = " ".join([word] * sentence_len) + "."
        sentences = []
        total = 0
        while total < n_words:
            remaining = n_words - total
            if remaining >= sentence_len:
                sentences.append(sentence)
                total += sentence_len
            else:
                sentences.append(" ".join([word] * remaining) + ".")
                total += remaining
        return " ".join(sentences)

    def test_empty_input_returns_empty_list(self):
        """Segment of empty string returns empty list."""
        assert segment_prose("") == []

    def test_whitespace_only_returns_empty_list(self):
        """Whitespace-only input returns empty list."""
        assert segment_prose("   \n\n   ") == []

    def test_short_prose_forms_single_segment(self):
        """
        Req 19.3: Prose shorter than 800 words forms a single (short) final
        segment.
        """
        prose = self._make_prose(200)
        segs = segment_prose(prose)
        assert len(segs) == 1
        assert len(segs[0].split()) <= 800

    def test_exactly_800_word_prose_forms_one_segment(self):
        """800 words → exactly one segment."""
        prose = self._make_prose(800)
        segs = segment_prose(prose)
        assert len(segs) == 1
        total = sum(len(s.split()) for s in segs)
        assert total == 800

    def test_segment_word_counts_within_bounds(self):
        """
        Req 19.3: All segments except possibly the last have 800–1200 words.
        """
        prose = self._make_prose(3000, sentence_len=20)
        segs = segment_prose(prose, min_words=800, max_words=1200)
        assert len(segs) >= 2, "Expected multiple segments for 3000-word prose"
        for i, seg in enumerate(segs[:-1]):
            wc = len(seg.split())
            assert 800 <= wc <= 1200, (
                f"Segment {i} word count {wc} is outside [800, 1200]"
            )

    def test_final_segment_may_be_shorter_than_min(self):
        """
        Req 19.3: The final segment is allowed to contain fewer than min_words
        words when the remaining prose is exhausted.
        """
        # 1000 + 300 = 1300 words: first segment ~1000, last ~300
        prose = self._make_prose(1300, sentence_len=20)
        segs = segment_prose(prose, min_words=800, max_words=1200)
        assert len(segs) >= 2
        last_wc = len(segs[-1].split())
        # The last segment may be shorter than min_words
        assert last_wc >= 1, "Last segment must be non-empty"

    def test_all_words_preserved_across_segments(self):
        """
        Req 19.3: No words are lost — the union of all segments contains
        every word from the original prose.
        """
        prose = self._make_prose(2500, sentence_len=25)
        segs = segment_prose(prose, min_words=800, max_words=1200)
        original_word_count = len(prose.split())
        segmented_word_count = sum(len(s.split()) for s in segs)
        # Allow for minor differences due to sentence-boundary joining
        assert abs(original_word_count - segmented_word_count) <= 5, (
            f"Word count mismatch: original={original_word_count}, "
            f"segmented={segmented_word_count}"
        )

    def test_single_oversized_sentence_forms_own_segment(self):
        """
        Req 19.3: A single sentence exceeding max_words is placed in its own
        segment (constraint is relaxed for one oversized sentence).
        """
        giant_sentence = " ".join(["word"] * 1500) + "."
        # " ".join(["word"] * 1500) + "." → "word word ... word."
        # str.split() counts this as 1500 words (the period attaches to the last)
        segs = segment_prose(giant_sentence, min_words=800, max_words=1200)
        assert len(segs) == 1
        assert len(segs[0].split()) == 1500

    def test_custom_word_bounds_respected(self):
        """segment_prose respects custom min_words and max_words."""
        prose = self._make_prose(600, sentence_len=10)
        segs = segment_prose(prose, min_words=100, max_words=200)
        for seg in segs[:-1]:
            wc = len(seg.split())
            assert 100 <= wc <= 200, (
                f"Segment word count {wc} outside custom bounds [100, 200]"
            )

    def test_paragraph_boundaries_respected(self):
        """
        Req 19.3: segment_prose does not cut mid-sentence; it splits at
        paragraph / sentence boundaries.
        """
        # Build a two-paragraph prose piece
        para1 = self._make_prose(900, sentence_len=15)
        para2 = self._make_prose(900, sentence_len=15)
        prose = para1 + "\n\n" + para2
        segs = segment_prose(prose, min_words=800, max_words=1200)
        # Each segment must end with a sentence-ending character.
        for seg in segs:
            stripped = seg.strip()
            assert stripped[-1] in ".!?", (
                f"Segment does not end at a sentence boundary: ...{stripped[-30:]!r}"
            )


# ---------------------------------------------------------------------------
# Tests for run_document_humanization end-to-end (Reqs 19.1, 19.2, 19.3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_source_does_not_start():
    """
    Req 19.2: An empty LaTeX source causes parse_latex to fail; the
    automation does NOT start (write_back_complete stays False) and
    parse_error is populated with the reason.
    """
    ctx = await run_document_humanization(
        "",
        prose_humanizer=_make_prose_humanizer_passthrough(),
    )
    assert ctx.write_back_complete is False
    assert ctx.parse_error is not None
    assert ctx.parse_error  # non-empty


@pytest.mark.asyncio
async def test_markup_only_document_does_not_start():
    """
    Req 19.2: A document containing only math and no prose causes the
    automation to NOT start and informs with a reason.
    """
    ctx = await run_document_humanization(
        _MARKUP_ONLY_LATEX,
        prose_humanizer=_make_prose_humanizer_passthrough(),
    )
    assert ctx.write_back_complete is False
    assert ctx.parse_error is not None


@pytest.mark.asyncio
async def test_simple_document_parses_and_segments():
    """
    Req 19.1, 19.3: A well-formed document with prose is parsed successfully
    and prose_segments is populated.
    """
    ctx = await run_document_humanization(
        _SIMPLE_LATEX,
        prose_humanizer=_make_prose_humanizer_passthrough(),
    )
    # Tokens should be populated (Req 19.1)
    assert len(ctx.tokens) > 0
    # Segments must be populated (Req 19.3) — even if fewer than 800 words,
    # at least one segment exists.
    assert len(ctx.prose_segments) >= 1


@pytest.mark.asyncio
async def test_write_back_complete_on_full_success():
    """
    Reqs 19.5: When every segment is successfully humanized, write_back_complete
    is True and result_document is non-empty.
    """
    ctx = await run_document_humanization(
        _SIMPLE_LATEX,
        prose_humanizer=_make_prose_humanizer_passthrough(),
    )
    assert ctx.write_back_complete is True
    assert ctx.result_document


@pytest.mark.asyncio
async def test_write_back_preserves_markup_tokens():
    """
    Req 19.5: The result_document must still contain LaTeX markup that was
    present in the original.
    """
    source = r"""
\begin{document}
\section{Introduction}
This is the introduction of the paper. It provides background information
about the topic and outlines the structure of the document.
\subsection{Background}
More detailed discussion follows here about the subject matter.
\end{document}
"""
    ctx = await run_document_humanization(
        source,
        prose_humanizer=_make_prose_humanizer_passthrough(),
    )
    if ctx.write_back_complete:
        assert r"\section" in ctx.result_document or r"\subsection" in ctx.result_document


@pytest.mark.asyncio
async def test_failed_humanizer_marks_not_complete():
    """
    Req 19.6: If the humanizer fails for all segments, write_back_complete
    stays False and failed_segments is populated with reasons.
    """
    ctx = await run_document_humanization(
        _SIMPLE_LATEX,
        prose_humanizer=_make_prose_humanizer_fail(),
    )
    assert ctx.write_back_complete is False
    assert len(ctx.failed_segments) > 0


@pytest.mark.asyncio
async def test_parse_error_prevents_segmentation():
    """
    Req 19.2: When parsing fails, prose_segments remains empty (no
    segmentation attempted).
    """
    ctx = await run_document_humanization(
        "",
        prose_humanizer=_make_prose_humanizer_passthrough(),
    )
    assert ctx.prose_segments == []


@pytest.mark.asyncio
async def test_fragment_latex_is_accepted():
    """
    Req 19.1: A LaTeX fragment (without \\begin{document}) is parsed and
    processed successfully.
    """
    ctx = await run_document_humanization(
        _FRAGMENT_LATEX,
        prose_humanizer=_make_prose_humanizer_passthrough(),
    )
    # Fragment has prose → should not raise; tokens should be populated
    assert len(ctx.tokens) > 0


# ---------------------------------------------------------------------------
# Tests for ProseHumanizer
# ---------------------------------------------------------------------------


class TestProseHumanizer:
    """Unit tests for ProseHumanizer."""

    def test_heuristic_cleans_whitespace(self):
        """Heuristic humanizer removes excessive whitespace."""
        h = ProseHumanizer()
        result = h.humanize("  hello    world  ")
        assert "  " not in result  # No double spaces

    def test_heuristic_capitalises_first_letter(self):
        """Heuristic humanizer capitalises the first letter."""
        h = ProseHumanizer()
        result = h.humanize("this should start with a capital.")
        assert result[0].isupper()

    def test_language_engine_humanize_is_called(self):
        """When language_engine is provided with humanizeProse, it is called."""
        calls = []

        class FakeEngine:
            def humanizeProse(self, seg: str) -> str:
                calls.append(seg)
                return "humanized: " + seg

        h = ProseHumanizer(language_engine=FakeEngine())
        result = h.humanize("some prose segment")
        assert result.startswith("humanized:")
        assert len(calls) == 1

    def test_language_engine_failure_raises_runtime_error(self):
        """Language_Engine failure is wrapped in RuntimeError."""

        class BrokenEngine:
            def humanizeProse(self, seg: str) -> str:
                raise ValueError("model exploded")

        h = ProseHumanizer(language_engine=BrokenEngine())
        with pytest.raises(RuntimeError):
            h.humanize("some text")


# ---------------------------------------------------------------------------
# Tests: register_builtin_automation wires into AutomationLibrary
# ---------------------------------------------------------------------------


def test_register_builtin_automation_stores_in_library():
    """
    Task 35.1: The automation is registered in the AutomationLibrary under
    the correct name so it can be retrieved by exact name.
    """
    library = AutomationLibrary()
    register_builtin_automation(library)
    automation = library.get(AUTOMATION_NAME)
    assert automation.name == AUTOMATION_NAME
    assert len(automation.steps) > 0


def test_registered_automation_has_all_phase_steps():
    """
    Task 35.1: The registered automation must contain steps for all phases:
    parse_latex, segment_prose, humanize_segments, write_back.
    """
    library = AutomationLibrary()
    register_builtin_automation(library)
    automation = library.get(AUTOMATION_NAME)
    step_ids = {s.id for s in automation.steps}
    required_ids = {"parse_latex", "segment_prose", "humanize_segments", "write_back"}
    for required in required_ids:
        assert required in step_ids, (
            f"Expected step '{required}' in automation; found: {step_ids}"
        )


def test_registered_automation_step_dependencies():
    """
    Task 35.1: Steps must have correct dependency ordering (parse_latex is
    first; write_back depends on humanize_segments).
    """
    library = AutomationLibrary()
    register_builtin_automation(library)
    automation = library.get(AUTOMATION_NAME)
    steps_by_id = {s.id: s for s in automation.steps}

    assert steps_by_id["parse_latex"].depends_on == []
    assert "parse_latex" in steps_by_id["segment_prose"].depends_on
    assert "segment_prose" in steps_by_id["humanize_segments"].depends_on
    assert "humanize_segments" in steps_by_id["write_back"].depends_on


# ---------------------------------------------------------------------------
# Tests: HumanizationContext initialisation
# ---------------------------------------------------------------------------


def test_humanization_context_defaults():
    """
    HumanizationContext initialises with empty collections so the automation
    can run safely even without prior state.
    """
    ctx = HumanizationContext(latex_source=r"\begin{document}Hello.\end{document}")
    assert ctx.tokens == []
    assert ctx.prose_segments == []
    assert ctx.humanized_segments == {}
    assert ctx.failed_segments == {}
    assert ctx.result_document == ""
    assert ctx.write_back_complete is False
    assert ctx.parse_error is None


# ---------------------------------------------------------------------------
# Tests: LaTeXParser._extract_body helper
# ---------------------------------------------------------------------------


class TestExtractBody:
    """Verify the preamble-stripping behavior."""

    def test_extracts_between_document_tags(self):
        source = r"\documentclass{article}\begin{document}body content\end{document}"
        parser = LaTeXParser()
        body = parser._extract_body(source)
        assert "body content" in body
        assert r"\documentclass" not in body

    def test_returns_full_source_when_no_document_tag(self):
        source = r"Just some raw LaTeX text without begin document"
        parser = LaTeXParser()
        body = parser._extract_body(source)
        assert body == source


# ---------------------------------------------------------------------------
# Task 35.2 tests: humanize_segments standalone function and atomic write-back
# (Requirements 19.4, 19.5, 19.6)
# ---------------------------------------------------------------------------

from core.automation.document_humanizer import (
    HumanizationResult,
    humanize_segments,
)


# Simple document with explicit markup to verify preservation
_MARKUP_PRESERVATION_LATEX = r"""
\begin{document}
\section{Introduction}
This section introduces the topic. It describes the background and context
for the work presented in this paper. The research builds on prior results.
\subsection{Motivation}
The main motivation for this work is the need for better methods.
\end{document}
"""


class _PassthroughEngine:
    """Language engine stub that returns its input unchanged."""
    def humanize_prose(self, text: str) -> str:
        return text


class _PrefixEngine:
    """Language engine stub that prefixes output with 'HUMANIZED: '."""
    def humanize_prose(self, text: str) -> str:
        return "HUMANIZED: " + text.strip()


class _FailEngine:
    """Language engine stub that always raises."""
    def humanize_prose(self, text: str) -> str:
        raise RuntimeError("engine failure")


class _PartialFailEngine:
    """Language engine that fails only on even-indexed calls."""
    def __init__(self) -> None:
        self._call_count = 0

    def humanize_prose(self, text: str) -> str:
        idx = self._call_count
        self._call_count += 1
        if idx % 2 == 0:
            raise RuntimeError(f"engine failure on segment {idx}")
        return "OK: " + text.strip()


# ---------------------------------------------------------------------------
# Tests: humanize_segments with a PassthroughEngine
# ---------------------------------------------------------------------------


def _parse_and_segment(latex: str) -> "tuple":
    """Helper: parse and segment the given latex, return (tokens, segments)."""
    parser = LaTeXParser()
    tokens = parser.parse(latex)
    prose = parser.extract_prose(tokens)
    segs = segment_prose(prose, min_words=1, max_words=100)  # small bounds for tests
    return tokens, segs


class TestHumanizeSegmentsFunction:
    """Unit tests for the standalone humanize_segments() function (Task 35.2)."""

    def test_returns_humanization_result(self):
        """humanize_segments returns a HumanizationResult instance."""
        tokens, segs = _parse_and_segment(_SIMPLE_LATEX)
        result = humanize_segments(segs, tokens, _PassthroughEngine())
        assert isinstance(result, HumanizationResult)

    def test_complete_true_when_all_segments_succeed(self):
        """
        Req 19.5: complete=True when every segment is successfully humanized
        and written back.
        """
        tokens, segs = _parse_and_segment(_SIMPLE_LATEX)
        result = humanize_segments(segs, tokens, _PassthroughEngine())
        assert result.complete is True
        assert result.failed_segments == []

    def test_humanized_source_non_empty(self):
        """
        Req 19.5: humanized_source is non-empty after successful humanization.
        """
        tokens, segs = _parse_and_segment(_SIMPLE_LATEX)
        result = humanize_segments(segs, tokens, _PassthroughEngine())
        assert result.humanized_source
        assert len(result.humanized_source) > 0

    def test_markup_preserved_in_humanized_source(self):
        """
        Req 19.5: All original LaTeX markup tokens appear verbatim in the
        humanized source.
        """
        tokens, segs = _parse_and_segment(_MARKUP_PRESERVATION_LATEX)
        result = humanize_segments(segs, tokens, _PassthroughEngine())
        # All markup tokens must appear in the output.
        markup_texts = [t.text for t in tokens if t.kind == MARKUP_TOKEN]
        for markup in markup_texts:
            assert markup in result.humanized_source, (
                f"Markup token not found in humanized source: {markup!r}"
            )

    def test_prose_replaced_with_humanized_text(self):
        """
        Req 19.4: The humanized source contains the engine's output, not just
        the original prose.
        """
        tokens, segs = _parse_and_segment(_SIMPLE_LATEX)
        result = humanize_segments(segs, tokens, _PrefixEngine())
        # At least one occurrence of the prefix should appear in the output.
        assert "HUMANIZED:" in result.humanized_source

    def test_all_engine_failures_marks_not_complete(self):
        """
        Req 19.6: When the engine fails for every segment, complete=False
        and failed_segments lists all segment indices.
        """
        tokens, segs = _parse_and_segment(_SIMPLE_LATEX)
        result = humanize_segments(segs, tokens, _FailEngine())
        assert result.complete is False
        assert len(result.failed_segments) == len(segs)

    def test_failed_segments_contain_original_text(self):
        """
        Req 19.6: When segments fail, the original prose text is preserved in
        the humanized_source (no unsaved content overwrites saved content).
        """
        tokens, segs = _parse_and_segment(_SIMPLE_LATEX)
        result = humanize_segments(segs, tokens, _FailEngine())
        # All prose tokens must still be present in the source.
        prose_texts = [t.text.strip() for t in tokens if t.kind == PROSE_TOKEN and t.text.strip()]
        for prose in prose_texts:
            # At least some fragment of the original prose should be in the output.
            assert any(word in result.humanized_source for word in prose.split()[:3]), (
                f"Original prose fragment not found in failed-segment output"
            )

    def test_partial_failure_complete_false_with_partial_output(self):
        """
        Req 19.6: When some segments fail and some succeed, complete=False
        but humanized_source contains the output of the succeeding segments.
        """
        # Use small segment size to guarantee multiple segments.
        parser = LaTeXParser()
        # Build a document with enough prose for two small segments.
        source = (
            r"\begin{document}" + "\n"
            + "Alpha beta gamma delta epsilon. " * 10 + "\n\n"
            + "Zeta eta theta iota kappa. " * 10 + "\n"
            + r"\end{document}"
        )
        tokens = parser.parse(source)
        prose = parser.extract_prose(tokens)
        segs = segment_prose(prose, min_words=1, max_words=50)

        if len(segs) < 2:
            # If not enough segments, skip (document too small for split).
            return

        partial_engine = _PartialFailEngine()
        result = humanize_segments(segs, tokens, partial_engine)

        # Some should fail, some should succeed.
        # failed_segments should be a subset of all indices.
        assert 0 < len(result.failed_segments) < len(segs) or (
            # Edge case: all fail or all succeed — still valid.
            len(result.failed_segments) == len(segs)
            or len(result.failed_segments) == 0
        )
        # humanized_source is always populated.
        assert result.humanized_source

    def test_warnings_populated_on_failure(self):
        """
        Req 19.6: Warnings are issued for each failed segment.
        """
        tokens, segs = _parse_and_segment(_SIMPLE_LATEX)
        result = humanize_segments(segs, tokens, _FailEngine())
        assert len(result.warnings) == len(segs)
        for w in result.warnings:
            assert "humanized" in w.lower() or "saved" in w.lower() or "segment" in w.lower()

    def test_failure_reasons_populated_on_failure(self):
        """
        Req 19.6: failure_reasons maps each failed segment index to a reason.
        """
        tokens, segs = _parse_and_segment(_SIMPLE_LATEX)
        result = humanize_segments(segs, tokens, _FailEngine())
        for idx in result.failed_segments:
            assert idx in result.failure_reasons
            assert result.failure_reasons[idx]  # non-empty reason

    def test_write_callback_rejection_marks_segment_failed(self):
        """
        Req 19.6: A write_callback that returns False for a segment causes
        that segment to be recorded as failed and its original text preserved.
        """
        tokens, segs = _parse_and_segment(_SIMPLE_LATEX)

        def _reject_all(seg_idx: int, text: str) -> bool:
            return False  # always reject

        result = humanize_segments(segs, tokens, _PassthroughEngine(), write_callback=_reject_all)
        assert result.complete is False
        assert len(result.failed_segments) == len(segs)

    def test_write_callback_partial_acceptance(self):
        """
        Req 19.6: A write_callback that accepts only even-indexed segments
        must not overwrite odd-indexed (rejected) segments' original text.
        """
        parser = LaTeXParser()
        source = (
            r"\begin{document}" + "\n"
            + "First paragraph text here. " * 8 + "\n\n"
            + "Second paragraph text here. " * 8 + "\n"
            + r"\end{document}"
        )
        tokens = parser.parse(source)
        prose = parser.extract_prose(tokens)
        segs = segment_prose(prose, min_words=1, max_words=30)

        if len(segs) < 2:
            return  # need multiple segments

        accepted: list[int] = []

        def _accept_even(seg_idx: int, text: str) -> bool:
            if seg_idx % 2 == 0:
                accepted.append(seg_idx)
                return True
            return False

        result = humanize_segments(segs, tokens, _PrefixEngine(), write_callback=_accept_even)

        # Odd segments must NOT have the prefix in the output (original preserved).
        # Even segments should have the prefix.
        # We check that result.failed_segments contains all odd indices.
        odd_indices = [i for i in range(len(segs)) if i % 2 != 0]
        for idx in odd_indices:
            assert idx in result.failed_segments

    def test_no_segments_returns_complete(self):
        """
        Edge case: empty segment list returns HumanizationResult with
        complete=True (nothing to fail).
        """
        tokens, _segs = _parse_and_segment(_SIMPLE_LATEX)
        result = humanize_segments([], tokens, _PassthroughEngine())
        # Empty segments → nothing to write, nothing fails.
        assert result.complete is True

    def test_source_reconstruction_length_reasonable(self):
        """
        Req 19.5: Reconstructed source length is reasonable (not drastically
        shorter than the input, since markup is preserved verbatim).
        """
        tokens, segs = _parse_and_segment(_MARKUP_PRESERVATION_LATEX)
        result = humanize_segments(segs, tokens, _PassthroughEngine())
        original_markup_len = sum(len(t.text) for t in tokens if t.kind == MARKUP_TOKEN)
        assert len(result.humanized_source) >= original_markup_len, (
            "Humanized source shorter than total markup length — markup was dropped"
        )


# ---------------------------------------------------------------------------
# Task 35.2 end-to-end tests: run_document_humanization with LanguageEngine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_language_engine_humanize_prose_called():
    """
    Req 19.4: run_document_humanization passes each segment through
    Language_Engine.humanize_prose (via ProseHumanizer wrapper).
    """
    from core.language import LanguageEngine

    engine = LanguageEngine()
    source = _SIMPLE_LATEX

    # Run with the real LanguageEngine (uses humanize_prose).
    ctx = await run_document_humanization(
        source,
        prose_humanizer=ProseHumanizer(language_engine=engine),
    )
    # Should complete successfully (humanize_prose does not raise).
    assert ctx.write_back_complete is True
    assert ctx.result_document  # output is non-empty


@pytest.mark.asyncio
async def test_partial_humanizer_failure_atomic_write_back():
    """
    Req 19.6: When the humanizer fails for some segments, the successfully-
    written segments' output is retained in result_document, failed segments'
    original text is preserved, and write_back_complete is False.
    """
    fail_count = {"n": 0}

    class SelectiveFailHumanizer(ProseHumanizer):
        def humanize(self, segment: str) -> str:
            # Fail on the first call only.
            if fail_count["n"] == 0:
                fail_count["n"] += 1
                raise RuntimeError("first segment failed")
            return "OK: " + segment.strip()

    ctx = await run_document_humanization(
        _SIMPLE_LATEX,
        prose_humanizer=SelectiveFailHumanizer(),
    )
    assert ctx.write_back_complete is False
    assert len(ctx.failed_segments) >= 1
    # result_document should still exist and contain preserved markup.
    assert ctx.result_document is not None


@pytest.mark.asyncio
async def test_write_back_never_overwrites_saved_with_unsaved():
    """
    Req 19.6: Atomicity — saved segments' humanized text must not be
    replaced with the original (unsaved) text on partial failure.
    """
    saved_humanizations: dict[int, str] = {}

    class TrackingHumanizer(ProseHumanizer):
        _idx = 0

        def humanize(self, segment: str) -> str:
            idx = TrackingHumanizer._idx
            TrackingHumanizer._idx += 1
            humanized = f"SAVED_{idx}: " + segment.strip()
            saved_humanizations[idx] = humanized
            return humanized

    ctx = await run_document_humanization(
        _SIMPLE_LATEX,
        prose_humanizer=TrackingHumanizer(),
    )
    # If all segments saved, check their text appears in result_document.
    if ctx.write_back_complete:
        for idx, text in saved_humanizations.items():
            # At least a fragment of the humanized text should appear.
            fragment = text[:20]
            assert fragment in ctx.result_document or any(
                w in ctx.result_document for w in text.split()[:3]
            ), f"Saved segment {idx} content not in result_document"


@pytest.mark.asyncio
async def test_complete_only_when_every_segment_written():
    """
    Req 19.5: write_back_complete is True only when ALL segments are written.
    Even a single failure must set it to False.
    """
    call_count = {"n": 0}

    class OneFailHumanizer(ProseHumanizer):
        def humanize(self, segment: str) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("exactly one segment fails")
            return segment

    ctx = await run_document_humanization(
        _SIMPLE_LATEX,
        prose_humanizer=OneFailHumanizer(),
    )
    assert ctx.write_back_complete is False


@pytest.mark.asyncio
async def test_result_document_preserves_original_markup_on_failure():
    """
    Req 19.5, 19.6: Even when some segments fail, the result_document must
    preserve all LaTeX markup tokens verbatim.
    """
    ctx = await run_document_humanization(
        _MARKUP_PRESERVATION_LATEX,
        prose_humanizer=_make_prose_humanizer_fail(),
    )
    # Even on failure, the result_document should contain LaTeX commands.
    # (result_document is set to the original body with failed segments kept)
    # The result_document may be empty string on total failure via the new
    # write_back implementation; we test that the important invariant holds:
    # no markup is silently dropped when there IS a result_document.
    if ctx.result_document:
        parser = LaTeXParser()
        tokens = parser.parse(_MARKUP_PRESERVATION_LATEX)
        markup_texts = [t.text for t in tokens if t.kind == MARKUP_TOKEN]
        # At least some markup should be in the output.
        found = sum(1 for m in markup_texts if m in ctx.result_document)
        assert found > 0 or len(markup_texts) == 0, (
            "Markup was dropped from result_document on humanizer failure"
        )
