"""
Document Humanization built-in automation (Requirement 19).

This module implements the document-humanization automation as a built-in
that is registered with the :class:`~core.automation.AutomationLibrary` at
startup.

The automation is invoked by exact name ``"document_humanization"`` and
accepts a LaTeX document string as input through a shared
``HumanizationContext`` object.

Execution plan (all steps are REVERSIBLE — no confirmations required):

    Step 1  parse_latex            Parse the LaTeX source and separate prose from
                                   markup.  If the source is unparseable or contains
                                   no prose, the step FAILS and the automation does
                                   NOT start (Reqs 19.1, 19.2).
    Step 2  segment_prose          Divide the extracted prose into 800–1200 word
                                   chunks; the final segment may be shorter if the
                                   remaining prose is < 800 words (Req 19.3).
    Step 3  humanize_<n>           One step per segment; calls
                                   Language_Engine.humanizeProse() to rewrite the
                                   segment while preserving meaning (Req 19.4).
                                   Steps run in dependency order (each depends on
                                   segment_prose).
    Step 4  write_back             Writes every humanized segment back into the
                                   LaTeX document, replacing the original prose
                                   while preserving all markup tokens.  Complete
                                   only when every segment has been written
                                   (Req 19.5).  Per-segment write failures are
                                   tracked; if any fail, the automation reports
                                   not-complete and warns which failed without
                                   overwriting already-saved segments (Req 19.6).

Design: Built-in automations (Document Humanization).
Requirements: 19.1, 19.2, 19.3, 19.4, 19.5, 19.6.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from core.planner import (
    Actuator,
    CommandPlan,
    Step,
    StepClassification,
    StepStatus,
)

logger = logging.getLogger(__name__)

# Public name used to register and invoke this automation.
AUTOMATION_NAME = "document_humanization"

# ---------------------------------------------------------------------------
# Token types produced by the LaTeX parser
# ---------------------------------------------------------------------------

PROSE_TOKEN = "prose"
MARKUP_TOKEN = "markup"


# ---------------------------------------------------------------------------
# DocumentToken — a single interleaved prose/markup element
# ---------------------------------------------------------------------------


@dataclass
class DocumentToken:
    """
    A single element in the interleaved document representation.

    Attributes
    ----------
    kind:
        ``"prose"`` for a run of human-readable text, ``"markup"`` for a
        LaTeX command, environment, math expression, or comment.
    text:
        The original text of this token as it appeared in the source.
    index:
        Position of this token in the ordered sequence of tokens for the
        whole document (0-based).  Preserved to allow round-trip
        reconstruction (Req 19.5).
    """

    kind: str          # PROSE_TOKEN or MARKUP_TOKEN
    text: str
    index: int


# ---------------------------------------------------------------------------
# LaTeXParseError — raised on unparseable input or missing prose
# ---------------------------------------------------------------------------


class LaTeXParseError(ValueError):
    """
    Raised by :class:`LaTeXParser` when the input cannot be parsed or
    contains no separable prose (Requirement 19.2).

    The message always states the reason so callers can surface it to the
    user verbatim.
    """


# ---------------------------------------------------------------------------
# LaTeXParser
# ---------------------------------------------------------------------------


class LaTeXParser:
    """
    Parses a LaTeX string and separates prose text from markup.

    "Markup" is defined as any of:
      - LaTeX commands: ``\\commandname`` (with or without arguments)
      - Display math: ``$$...$$``, ``\\[...\\]``
      - Inline math: ``$...$``, ``\\(...\\)``
      - Named environments: ``\\begin{name}...\\end{name}``
      - Line comments: ``%...`` (to end of line)

    Everything that is not matched as markup is treated as prose.

    The parser strips the preamble (everything up to and including
    ``\\begin{document}`` if present) and the closing
    ``\\end{document}`` if present before analysing for prose.

    Returns
    -------
    list[DocumentToken]
        Interleaved sequence of ``"prose"`` and ``"markup"`` tokens in
        document order, preserving all original text.

    Raises
    ------
    LaTeXParseError
        When the source is empty, consists entirely of whitespace, or
        contains no prose after stripping markup.

    Design: Built-in automations (Document Humanization).
    Requirements: 19.1, 19.2.
    """

    # ------------------------------------------------------------------
    # Regex patterns (applied in order; higher patterns take precedence)
    # ------------------------------------------------------------------

    # Display math: $$...$$ (greedy-avoided with .*? in DOTALL mode)
    _RE_DISPLAY_MATH_DOLLARS = re.compile(r"\$\$.*?\$\$", re.DOTALL)

    # Display math: \[...\]
    _RE_DISPLAY_MATH_BRACKETS = re.compile(r"\\\[.*?\\\]", re.DOTALL)

    # Inline math: \(...\)
    _RE_INLINE_MATH_PARENS = re.compile(r"\\\(.*?\\\)", re.DOTALL)

    # Inline math: $...$ (single dollar; must not match $$, handled by the
    # display-math pattern which is applied first)
    _RE_INLINE_MATH_DOLLAR = re.compile(r"(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)", re.DOTALL)

    # Named environments: \begin{name}...\end{name}
    # Captures environments like equation, figure, table, verbatim, etc.
    _RE_ENVIRONMENT = re.compile(
        r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}", re.DOTALL
    )

    # LaTeX comments: % to end of line
    _RE_COMMENT = re.compile(r"%[^\n]*")

    # LaTeX commands with optional argument(s):
    # \commandname followed by any combination of [optional] and {required} args.
    # This greedily captures grouped argument content but stops at prose.
    _RE_COMMAND_WITH_ARGS = re.compile(
        r"\\[a-zA-Z@]+\*?"         # command name (optionally starred)
        r"(?:\s*\[[^\]]*\])*"      # zero or more [optional] args
        r"(?:\s*\{[^}]*\})*",      # zero or more {required} args
        re.DOTALL,
    )

    # Bare LaTeX special characters and non-letter commands: \\, \{, \}, etc.
    _RE_COMMAND_BARE = re.compile(r"\\(?:[^a-zA-Z]|$)", re.MULTILINE)

    # Combined master pattern (applied in one pass, higher specificity first).
    # Named groups allow the replacement logic to identify the match type.
    _MASTER_RE = re.compile(
        r"(?P<display_math_dol>\$\$.*?\$\$)"
        r"|(?P<display_math_bracket>\\\[.*?\\\])"
        r"|(?P<inline_math_paren>\\\(.*?\\\))"
        r"|(?P<environment>\\begin\{[^}]+\}.*?\\end\{[^}]+\})"
        r"|(?P<comment>%[^\n]*)"
        r"|(?P<command_args>\\[a-zA-Z@]+\*?(?:\s*\[[^\]]*\])*(?:\s*\{[^}]*\})*)"
        r"|(?P<inline_math_dollar>(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$))"
        r"|(?P<command_bare>\\(?:[^a-zA-Z]))",
        re.DOTALL,
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, latex_source: str) -> list[DocumentToken]:
        """
        Parse *latex_source* and return an interleaved list of tokens.

        Parameters
        ----------
        latex_source:
            A LaTeX document string (full document or fragment).

        Returns
        -------
        list[DocumentToken]
            Ordered sequence of :class:`DocumentToken` objects.

        Raises
        ------
        LaTeXParseError
            When *latex_source* is empty, all whitespace, or contains no
            prose after markup stripping (Req 19.2).
        """
        if not latex_source or not latex_source.strip():
            raise LaTeXParseError(
                "The LaTeX source is empty. "
                "Please provide a non-empty LaTeX document."
            )

        # Strip preamble and \end{document} for prose analysis.
        body = self._extract_body(latex_source)

        tokens: list[DocumentToken] = []
        index = 0
        cursor = 0

        for match in self._MASTER_RE.finditer(body):
            start, end = match.start(), match.end()

            # Prose between last match and this match
            if start > cursor:
                prose_text = body[cursor:start]
                cleaned = prose_text.strip()
                if cleaned:
                    tokens.append(
                        DocumentToken(kind=PROSE_TOKEN, text=prose_text, index=index)
                    )
                    index += 1
                elif prose_text:
                    # Whitespace-only prose: keep as markup to preserve spacing
                    tokens.append(
                        DocumentToken(kind=MARKUP_TOKEN, text=prose_text, index=index)
                    )
                    index += 1

            # The markup match itself
            tokens.append(
                DocumentToken(kind=MARKUP_TOKEN, text=match.group(), index=index)
            )
            index += 1
            cursor = end

        # Trailing prose after all matches
        if cursor < len(body):
            trailing = body[cursor:]
            cleaned = trailing.strip()
            if cleaned:
                tokens.append(
                    DocumentToken(kind=PROSE_TOKEN, text=trailing, index=index)
                )
            elif trailing:
                tokens.append(
                    DocumentToken(kind=MARKUP_TOKEN, text=trailing, index=index)
                )

        # Validate: must have at least some prose
        prose_tokens = [t for t in tokens if t.kind == PROSE_TOKEN]
        if not prose_tokens:
            raise LaTeXParseError(
                "No separable prose content was found in the LaTeX source. "
                "The document may consist entirely of markup, math, or comments."
            )

        return tokens

    def extract_prose(self, tokens: list[DocumentToken]) -> str:
        """
        Concatenate all prose tokens into a single string for segmentation.

        Parameters
        ----------
        tokens:
            Token list from :meth:`parse`.

        Returns
        -------
        str
            All prose text joined in document order.
        """
        return " ".join(t.text.strip() for t in tokens if t.kind == PROSE_TOKEN)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_body(latex_source: str) -> str:
        """
        Return the body of the document — the content between
        ``\\begin{document}`` and ``\\end{document}`` when those markers
        are present, otherwise the full source.

        Any preamble before ``\\begin{document}`` is excluded so that
        ``\\usepackage`` and ``\\newcommand`` declarations do not distort
        the prose/markup ratio.
        """
        begin_match = re.search(r"\\begin\{document\}", latex_source)
        end_match = re.search(r"\\end\{document\}", latex_source)

        if begin_match:
            start = begin_match.end()
            end = end_match.start() if end_match else len(latex_source)
            return latex_source[start:end]

        # No \begin{document}: treat the whole source as the body
        # (fragment / snippet use-case).
        return latex_source


# ---------------------------------------------------------------------------
# segment_prose
# ---------------------------------------------------------------------------


def segment_prose(
    prose_text: str,
    min_words: int = 800,
    max_words: int = 1200,
) -> list[str]:
    """
    Split *prose_text* into word-count-bounded segments.

    Each segment contains between *min_words* and *max_words* words,
    except possibly the final segment which may contain fewer than
    *min_words* words if the remaining prose is exhausted (Req 19.3).

    The function respects sentence and paragraph boundaries where
    possible: it accumulates whole sentences and only closes a segment
    when the accumulated word count is within the target range AND a
    sentence boundary has been reached.  If a single sentence itself
    exceeds *max_words*, it is placed in its own segment (the constraint
    is relaxed for a single oversized sentence).

    Parameters
    ----------
    prose_text:
        Plain prose string (markup already stripped).
    min_words:
        Lower bound for segment word count (default 800).
    max_words:
        Upper bound for segment word count (default 1200).

    Returns
    -------
    list[str]
        Ordered list of prose segments.  Returns ``[]`` if *prose_text*
        is empty or all whitespace.

    Notes
    -----
    - Splitting is at sentence/paragraph boundaries, not mid-sentence.
    - The final segment may be shorter than *min_words*.
    - Empty strings and whitespace-only input return ``[]``.

    Design: Built-in automations (Document Humanization).
    Requirements: 19.3.
    """
    if not prose_text or not prose_text.strip():
        return []

    # Paragraph-aware splitting: first split on double-newlines (paragraphs),
    # then split each paragraph into sentences.
    paragraphs = re.split(r"\n{2,}", prose_text)

    # Sentence splitting regex: split after . ! ? followed by whitespace or end.
    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    sentences: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        sents = _SENTENCE_RE.split(para)
        for s in sents:
            s = s.strip()
            if s:
                sentences.append(s)

    if not sentences:
        return []

    def _word_count(text: str) -> int:
        return len(text.split())

    segments: list[str] = []
    current_sentences: list[str] = []
    current_wc = 0

    for sentence in sentences:
        sent_wc = _word_count(sentence)

        # If a single sentence exceeds max_words, it forms its own segment.
        if current_wc == 0 and sent_wc > max_words:
            segments.append(sentence)
            current_sentences = []
            current_wc = 0
            continue

        # Would adding this sentence push us over max_words?
        if current_wc + sent_wc > max_words and current_wc >= min_words:
            # Flush the current segment first.
            segments.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_wc = sent_wc
        else:
            current_sentences.append(sentence)
            current_wc += sent_wc

            # We have enough words — flush if at or above min_words.
            if current_wc >= min_words:
                segments.append(" ".join(current_sentences))
                current_sentences = []
                current_wc = 0

    # Remaining sentences form the (possibly short) final segment.
    if current_sentences:
        segments.append(" ".join(current_sentences))

    return segments


# ---------------------------------------------------------------------------
# HumanizationResult — returned by humanize_segments
# ---------------------------------------------------------------------------


@dataclass
class HumanizationResult:
    """
    Result of a :func:`humanize_segments` call.

    Attributes
    ----------
    complete:
        ``True`` only when every segment was successfully humanized and
        written back (Req 19.5).
    humanized_source:
        The full LaTeX source with humanized prose written back into it,
        preserving all original markup tokens (Req 19.5).  Always populated
        (even on partial failure) so that the successfully-written segments
        are available to the caller.
    failed_segments:
        Zero-based indices (within the segments list) of segments that
        could not be humanized or written back (Req 19.6).  Empty on full
        success.
    failure_reasons:
        Maps failed segment index → human-readable reason string.
    warnings:
        Human-readable warning messages describing which segments failed
        and why (Req 19.6).
    """

    complete: bool
    humanized_source: str
    failed_segments: list[int] = field(default_factory=list)
    failure_reasons: dict[int, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def humanize_segments(
    segments: list[str],
    tokens: "list[DocumentToken]",
    language_engine: Any,
    *,
    write_callback: "Callable[[int, str], bool] | None" = None,
) -> HumanizationResult:
    """
    Humanize each prose segment and write the results back into the
    interleaved document structure.

    This is the core function for task 35.2.  It:

    1. Calls ``language_engine.humanize_prose(segment)`` (or
       ``humanizeProse`` for backwards-compatibility) for each segment to
       obtain the humanized prose (Req 19.4).
    2. Substitutes each prose token's text with the corresponding fragment
       of humanized prose, preserving all markup token values verbatim
       (Req 19.5).
    3. Tracks which segments are successfully written.  On per-segment
       write failure it records the failure, leaves already-saved segments
       untouched, and continues with remaining segments (Req 19.6).
    4. Returns a :class:`HumanizationResult` indicating whether the
       automation is complete (all segments written) or not-complete with
       the list of failed segments.

    Atomicity guarantee (Req 19.6)
    --------------------------------
    The function never overwrites a successfully-saved segment's output
    with unsaved (failed) content.  A failed segment's prose tokens
    remain at their original text in the returned source.

    Parameters
    ----------
    segments:
        List of prose segment strings as returned by :func:`segment_prose`.
    tokens:
        The interleaved document token list (from :class:`LaTeXParser`).
        Markup tokens are preserved verbatim; prose tokens are replaced by
        humanized text for successfully-written segments.
    language_engine:
        The :class:`~core.language.LanguageEngine` instance.  Must expose
        ``humanize_prose(text: str) -> str`` (preferred) or
        ``humanizeProse(text: str) -> str``.
    write_callback:
        Optional ``(segment_index: int, humanized_text: str) -> bool``
        callable invoked for each segment after humanization.  Return
        ``True`` to accept the write (segment saved); ``False`` to reject
        it (segment recorded as failed, Req 19.6).  When ``None``, every
        write is accepted automatically.

    Returns
    -------
    HumanizationResult
        Complete result indicating success/failure and the produced LaTeX
        source with all successfully-written segments applied.

    Requirements: 19.4, 19.5, 19.6.
    """
    import re as _re

    # Build a quick helper to get the humanize method from the engine.
    def _humanize(engine: Any, text: str) -> str:
        if hasattr(engine, "humanize_prose"):
            return engine.humanize_prose(text)
        if hasattr(engine, "humanizeProse"):
            return engine.humanizeProse(text)
        raise AttributeError(
            "language_engine does not expose humanize_prose or humanizeProse"
        )

    # Mutable replacement map: prose-token list-index → text to use.
    # Starts with all originals; updated only for successfully-written segments.
    prose_token_positions: list[int] = [
        i for i, t in enumerate(tokens) if t.kind == PROSE_TOKEN
    ]
    original_by_pos: dict[int, str] = {
        p: tokens[p].text for p in prose_token_positions
    }
    replacement_by_pos: dict[int, str] = dict(original_by_pos)

    # Assign prose-token positions to segment indices (same greedy strategy
    # as _assign_segments_to_prose_tokens).
    seg_to_tok_positions: dict[int, list[int]] = {}
    if segments and prose_token_positions:
        seg_idx = 0
        seg_word_budget = len(segments[0].split()) if segments else 0
        accumulated_words = 0

        for tok_pos in prose_token_positions:
            seg_to_tok_positions.setdefault(seg_idx, []).append(tok_pos)
            accumulated_words += len(tokens[tok_pos].text.split())

            while (
                seg_idx < len(segments) - 1
                and accumulated_words >= seg_word_budget
            ):
                seg_idx += 1
                seg_word_budget = len(segments[seg_idx].split())
                accumulated_words = 0

    failed_segments: list[int] = []
    failure_reasons: dict[int, str] = {}
    warnings: list[str] = []

    for seg_idx, segment_text in enumerate(segments):
        # Step A: humanize via Language_Engine (Req 19.4).
        try:
            humanized_text = _humanize(language_engine, segment_text)
        except Exception as exc:  # noqa: BLE001
            reason = f"humanize_prose raised: {exc}"
            failed_segments.append(seg_idx)
            failure_reasons[seg_idx] = reason
            warnings.append(
                f"Segment {seg_idx + 1} could not be humanized: {reason}"
            )
            logger.warning(
                "humanize_segments: segment %d humanization failed: %s",
                seg_idx, reason,
            )
            # Original text preserved in replacement_by_pos (Req 19.6).
            continue

        # Step B: invoke the write callback if provided (Req 19.6 atomicity).
        if write_callback is not None:
            try:
                accepted = bool(write_callback(seg_idx, humanized_text))
            except Exception as exc:  # noqa: BLE001
                accepted = False
                reason = f"write_callback raised: {exc}"
                failure_reasons[seg_idx] = reason
            else:
                if not accepted:
                    failure_reasons.setdefault(
                        seg_idx, "write_callback returned False"
                    )

            if not accepted:
                failed_segments.append(seg_idx)
                warnings.append(
                    f"Segment {seg_idx + 1} could not be saved: "
                    f"{failure_reasons.get(seg_idx, 'unknown reason')}"
                )
                logger.warning(
                    "humanize_segments: segment %d write rejected.", seg_idx
                )
                # Original text preserved (Req 19.6).
                continue

        # Step C: write humanized text back into the token positions for this
        # segment (Req 19.5).
        tok_positions = seg_to_tok_positions.get(seg_idx, [])
        if len(tok_positions) == 1:
            tok_pos = tok_positions[0]
            orig = original_by_pos[tok_pos]
            leading = orig[:len(orig) - len(orig.lstrip())]
            trailing = orig[len(orig.rstrip()):]
            replacement_by_pos[tok_pos] = leading + humanized_text.strip() + trailing
        elif len(tok_positions) > 1:
            # Distribute humanized text proportionally.
            orig_counts = [len(tokens[p].text.split()) for p in tok_positions]
            total_orig = sum(orig_counts)
            humanized_words = humanized_text.split()
            total_human = len(humanized_words)
            word_pos = 0
            for i, tok_pos in enumerate(tok_positions):
                if i == len(tok_positions) - 1:
                    chunk_words = humanized_words[word_pos:]
                else:
                    share = round(total_human * orig_counts[i] / max(total_orig, 1))
                    chunk_words = humanized_words[word_pos:word_pos + share]
                    word_pos += share
                orig = original_by_pos[tok_pos]
                leading = orig[:len(orig) - len(orig.lstrip())]
                trailing = orig[len(orig.rstrip()):]
                replacement_by_pos[tok_pos] = (
                    leading + " ".join(chunk_words) + trailing
                )
        # else: no prose token positions for this segment (edge case) — skip.

        logger.info(
            "humanize_segments: segment %d written back (%d → %d words).",
            seg_idx,
            len(segment_text.split()),
            len(humanized_text.split()),
        )

    # Reconstruct the full LaTeX source.
    parts: list[str] = []
    for i, token in enumerate(tokens):
        if token.kind == MARKUP_TOKEN:
            parts.append(token.text)
        else:
            parts.append(replacement_by_pos.get(i, token.text))

    humanized_source = "".join(parts)
    complete = len(failed_segments) == 0

    if not complete:
        logger.warning(
            "humanize_segments: NOT complete — %d segment(s) failed: %s",
            len(failed_segments),
            failed_segments,
        )

    return HumanizationResult(
        complete=complete,
        humanized_source=humanized_source,
        failed_segments=failed_segments,
        failure_reasons=failure_reasons,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# HumanizationContext — shared mutable state for one humanization run
# ---------------------------------------------------------------------------


@dataclass
class HumanizationContext:
    """
    Shared mutable state threaded through all steps of one humanization run.

    Parameters
    ----------
    latex_source:
        The original LaTeX document to humanize.

    Attributes
    ----------
    tokens:
        Interleaved list of :class:`DocumentToken` objects produced by
        :class:`LaTeXParser`.  Populated by the ``parse_latex`` step.
    prose_segments:
        Ordered list of plain-prose segments produced by
        :func:`segment_prose`.  Populated by the ``segment_prose`` step.
    humanized_segments:
        Maps segment index → humanized text.  Populated by the
        ``humanize_<n>`` steps.
    failed_segments:
        Maps segment index → human-readable failure reason for segments
        whose write-back failed (Req 19.6).
    result_document:
        The fully reconstructed LaTeX document with humanized prose.
        Populated only by ``write_back`` if every segment succeeded
        (Req 19.5).
    write_back_complete:
        ``True`` only if every segment was successfully written back
        (Req 19.5).
    parse_error:
        Non-None if the parse_latex step failed; contains the
        human-readable reason (Req 19.2).
    """

    latex_source: str

    tokens: list[DocumentToken] = field(default_factory=list)
    prose_segments: list[str] = field(default_factory=list)
    humanized_segments: dict[int, str] = field(default_factory=dict)
    failed_segments: dict[int, str] = field(default_factory=dict)
    result_document: str = ""
    write_back_complete: bool = False
    parse_error: str | None = None


# ---------------------------------------------------------------------------
# ProseHumanizer — default prose rewriter (injected for testability)
# ---------------------------------------------------------------------------


class ProseHumanizer:
    """
    Rewrites a prose segment into a more human-sounding form.

    The default implementation uses the Language_Engine's
    ``humanizeProse`` method when available; otherwise applies a
    lightweight heuristic (sentence-case normalisation and whitespace
    cleaning) suitable for testing without a model.

    Parameters
    ----------
    language_engine:
        Optional :class:`~core.language.LanguageEngine` instance.  When
        provided, its ``humanizeProse(segment)`` method is called (Req 19.4).
        When ``None``, a heuristic pass is used.
    """

    def __init__(self, language_engine: Any | None = None) -> None:
        self._engine = language_engine

    def humanize(self, segment: str) -> str:
        """
        Rewrite *segment* into a more human-sounding form.

        Parameters
        ----------
        segment:
            A plain-prose segment string (800–1200 words typically).

        Returns
        -------
        str
            The humanized segment, preserving meaning (Req 19.4).

        Raises
        ------
        RuntimeError
            If the model/engine call fails, so the per-segment step can
            catch it and record the failure (Req 19.6).
        """
        if self._engine is not None:
            # Try snake_case method name first (LanguageEngine.humanize_prose,
            # Req 19.4), then camelCase for backwards-compat.
            if hasattr(self._engine, "humanize_prose"):
                try:
                    return self._engine.humanize_prose(segment)
                except Exception as exc:
                    raise RuntimeError(
                        f"Language_Engine.humanize_prose failed: {exc}"
                    ) from exc
            if hasattr(self._engine, "humanizeProse"):
                try:
                    return self._engine.humanizeProse(segment)
                except Exception as exc:
                    raise RuntimeError(
                        f"Language_Engine.humanizeProse failed: {exc}"
                    ) from exc

        # Heuristic fallback: clean up whitespace and normalise sentence case.
        return self._heuristic_humanize(segment)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_humanize(segment: str) -> str:
        """
        Lightweight heuristic humanizer: normalize whitespace and ensure
        sentence-case capitalisation.  Does not alter meaning.
        """
        # Collapse runs of whitespace (but preserve paragraph breaks).
        cleaned = re.sub(r"[ \t]+", " ", segment).strip()
        # Ensure first letter of the segment is uppercase.
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned


# ---------------------------------------------------------------------------
# DocumentHumanizer — builds and drives the humanization plan
# ---------------------------------------------------------------------------


class DocumentHumanizer:
    """
    Orchestrates a single document-humanization run.

    This class is NOT a long-lived service — one instance is created per
    humanization invocation.  It builds a :class:`~core.planner.CommandPlan`
    from a :class:`HumanizationContext` and exposes an async actuator
    callback suitable for use with :class:`~core.execution.ExecutionEngine`.

    Parameters
    ----------
    context:
        The shared :class:`HumanizationContext` for this run.  Mutated
        in-place by each step.
    latex_parser:
        The :class:`LaTeXParser` to use.  Defaults to a new instance.
    prose_humanizer:
        The :class:`ProseHumanizer` to use.  Defaults to a heuristic
        humanizer.
    presenter:
        An optional callable ``(result_document: str) → None`` called
        after write_back completes successfully.  Use this to surface
        the humanized document to the user.  When ``None``, the result
        is logged at INFO level.
    """

    def __init__(
        self,
        context: HumanizationContext,
        latex_parser: LaTeXParser | None = None,
        prose_humanizer: ProseHumanizer | None = None,
        presenter: Callable[[str], Any] | None = None,
    ) -> None:
        self._ctx = context
        self._parser = latex_parser or LaTeXParser()
        self._humanizer = prose_humanizer or ProseHumanizer()
        self._presenter = presenter

    # ------------------------------------------------------------------
    # Plan factory
    # ------------------------------------------------------------------

    def build_plan(self) -> CommandPlan:
        """
        Build the :class:`~core.planner.CommandPlan` for this humanization run.

        The plan is constructed fresh each time so it can be handed to
        the :class:`~core.execution.ExecutionEngine`.

        Returns
        -------
        CommandPlan
        """
        plan_id = str(uuid.uuid4())

        # Step 1: parse LaTeX and separate prose from markup (Reqs 19.1, 19.2).
        parse_step = Step(
            id="parse_latex",
            intent="Parse LaTeX source and separate prose from markup",
            actuator=Actuator.INTERNAL,
            args={},
            depends_on=[],
            classification=StepClassification.REVERSIBLE,
            required_slots=[],
            postcondition=None,
        )

        # Step 2: segment the prose into 800–1200 word chunks (Req 19.3).
        segment_step = Step(
            id="segment_prose",
            intent="Segment extracted prose into 800–1200 word chunks",
            actuator=Actuator.INTERNAL,
            args={"min_words": 800, "max_words": 1200},
            depends_on=["parse_latex"],
            classification=StepClassification.REVERSIBLE,
            required_slots=[],
            postcondition=None,
        )

        # Steps 3a…3n: humanize each segment (created dynamically after
        # segment_prose runs; we use a single representative step for the
        # plan skeleton).  At build_plan time we do not yet know n, so we
        # add a sentinel humanize step that the actuator dispatches to the
        # per-segment logic.
        humanize_step = Step(
            id="humanize_segments",
            intent="Humanize each prose segment via Language_Engine",
            actuator=Actuator.INTERNAL,
            args={},
            depends_on=["segment_prose"],
            classification=StepClassification.REVERSIBLE,
            required_slots=[],
            postcondition=None,
        )

        # Step 4: write humanized prose back into the LaTeX document (Reqs 19.5, 19.6).
        write_step = Step(
            id="write_back",
            intent="Write humanized prose back into LaTeX document preserving markup",
            actuator=Actuator.INTERNAL,
            args={},
            depends_on=["humanize_segments"],
            classification=StepClassification.REVERSIBLE,
            required_slots=[],
            postcondition=None,
        )

        return CommandPlan(
            id=plan_id,
            origin_command="run_automation:document_humanization",
            steps=[parse_step, segment_step, humanize_step, write_step],
        )

    # ------------------------------------------------------------------
    # Actuator callback — dispatches each step by ID
    # ------------------------------------------------------------------

    async def actuator(self, step: Step) -> Any:
        """
        Async actuator callback for this humanization run.

        Dispatches to the appropriate handler based on the step ID.
        Designed to be passed as ``actuator_callback`` to a dedicated
        :class:`~core.execution.ExecutionEngine` instance.

        Parameters
        ----------
        step:
            The step to execute.

        Returns
        -------
        Any
            Step-specific output (dict or None).

        Raises
        ------
        LaTeXParseError
            Propagated from parse_latex when the source is unparseable or
            has no prose (Req 19.2).
        RuntimeError
            For unrecognised step IDs.
        """
        sid = step.id

        if sid == "parse_latex":
            return await self._step_parse_latex()
        if sid == "segment_prose":
            return await self._step_segment_prose(step)
        if sid == "humanize_segments":
            return await self._step_humanize_all_segments()
        if sid == "write_back":
            return await self._step_write_back()

        raise RuntimeError(
            f"DocumentHumanizer: unknown step id '{sid}'"
        )

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    async def _step_parse_latex(self) -> dict:
        """
        Parse the LaTeX source and separate prose from markup (Reqs 19.1, 19.2).

        Raises :class:`LaTeXParseError` (caught by the execution engine,
        which marks the step FAILED and stops all dependents) when the
        source is unparseable or contains no prose.
        """
        loop = asyncio.get_event_loop()
        try:
            tokens: list[DocumentToken] = await loop.run_in_executor(
                None, self._parser.parse, self._ctx.latex_source
            )
        except LaTeXParseError as exc:
            # Record the reason for user-facing messaging (Req 19.2).
            self._ctx.parse_error = str(exc)
            logger.warning(
                "DocumentHumanizer.parse_latex: FAILED — %s", exc
            )
            raise RuntimeError(str(exc)) from exc

        self._ctx.tokens = tokens
        prose_count = sum(1 for t in tokens if t.kind == PROSE_TOKEN)
        logger.info(
            "DocumentHumanizer.parse_latex: %d token(s) (%d prose, %d markup).",
            len(tokens), prose_count, len(tokens) - prose_count,
        )
        return {
            "token_count": len(tokens),
            "prose_token_count": prose_count,
            "markup_token_count": len(tokens) - prose_count,
        }

    async def _step_segment_prose(self, step: Step) -> dict:
        """
        Segment the extracted prose into 800–1200 word chunks (Req 19.3).

        The final segment may be shorter than *min_words*.
        """
        min_words: int = step.args.get("min_words", 800)
        max_words: int = step.args.get("max_words", 1200)

        # Concatenate all prose tokens into a single string.
        prose_text = self._parser.extract_prose(self._ctx.tokens)

        loop = asyncio.get_event_loop()
        segments: list[str] = await loop.run_in_executor(
            None, lambda: segment_prose(prose_text, min_words, max_words)
        )

        if not segments:
            raise RuntimeError(
                "Prose segmentation produced no segments. "
                "The document may be empty after markup removal."
            )

        self._ctx.prose_segments = segments
        logger.info(
            "DocumentHumanizer.segment_prose: %d segment(s) produced "
            "(min_words=%d, max_words=%d).",
            len(segments), min_words, max_words,
        )
        return {
            "segment_count": len(segments),
            "min_words": min_words,
            "max_words": max_words,
        }

    async def _step_humanize_all_segments(self) -> dict:
        """
        Humanize every prose segment (Req 19.4).

        Runs each segment through :class:`ProseHumanizer`.  Failures are
        stored in ``context.failed_segments`` but do NOT stop the other
        segments (partial-failure pattern mirrors question-paper analysis).
        The write_back step checks for failures before finalising (Req 19.6).
        """
        results: dict[int, str | None] = {}

        async def _humanize_one(index: int, segment: str) -> None:
            loop = asyncio.get_event_loop()
            try:
                humanized: str = await loop.run_in_executor(
                    None, self._humanizer.humanize, segment
                )
                self._ctx.humanized_segments[index] = humanized
                results[index] = humanized
                logger.info(
                    "DocumentHumanizer.humanize_segments[%d]: OK (%d words → %d words).",
                    index,
                    len(segment.split()),
                    len(humanized.split()),
                )
            except Exception as exc:  # noqa: BLE001
                reason = str(exc)
                self._ctx.failed_segments[index] = reason
                results[index] = None
                logger.warning(
                    "DocumentHumanizer.humanize_segments[%d]: FAILED — %s",
                    index, reason,
                )

        # Run all segment humanizations concurrently.
        tasks = [
            _humanize_one(i, seg)
            for i, seg in enumerate(self._ctx.prose_segments)
        ]
        await asyncio.gather(*tasks)

        n_ok = len(self._ctx.humanized_segments)
        n_fail = len(self._ctx.failed_segments)
        logger.info(
            "DocumentHumanizer.humanize_segments: %d succeeded, %d failed.",
            n_ok, n_fail,
        )
        return {
            "succeeded": n_ok,
            "failed": n_fail,
            "failed_indices": list(self._ctx.failed_segments.keys()),
        }

    async def _step_write_back(self) -> dict:
        """
        Write humanized prose back into the LaTeX document (Reqs 19.5, 19.6).

        Iterates over the token list and replaces each prose token with its
        corresponding humanized segment text.  Markup tokens are preserved
        verbatim.

        Atomicity guarantee (Req 19.6)
        --------------------------------
        Each prose segment is written back independently.  If a segment
        failed during humanization (recorded in ``context.failed_segments``),
        that segment's original text is kept in the output — the failed
        segment is NOT overwritten with unsaved content.  Segments that
        were successfully humanized ARE written back regardless.

        The automation is considered complete ONLY if every segment was
        successfully humanized and written back (Req 19.5).  Partial success
        still produces a result document with all successfully-written segments
        applied; the user is warned about the failures (Req 19.6).
        """
        # Map segment index → text to use (humanized or original).
        # Segments that failed keep their original prose token text.
        failed_indices: set[int] = set(self._ctx.failed_segments.keys())
        n_segments = len(self._ctx.prose_segments)

        # Collect the per-segment text to write back: use humanized when
        # available, keep original (from context) when humanization failed.
        # We need to map prose-token positions to segment indices.
        # The segment_prose step produced prose_segments from the prose tokens
        # in order; prose token i corresponds to segment i.
        prose_token_count = sum(1 for t in self._ctx.tokens if t.kind == PROSE_TOKEN)

        # Build a per-prose-token replacement map.
        # prose_segments is a flat list of segment strings; each segment may
        # correspond to multiple prose tokens (if the original text was split
        # across tokens by markup).  However, in the current implementation
        # segment_prose concatenates all prose tokens into one string and then
        # splits, so the mapping is segment_index → one chunk of the prose.
        #
        # To write back, we redistribute the humanized segment text across
        # the prose tokens that contributed to it, in proportion to the
        # original token lengths.  This preserves the interleaved structure.
        #
        # Implementation: track which prose token indices (in order) each
        # segment contributed to.  Since segment_prose operates on the joined
        # prose text, and prose tokens in the document appear in order, we
        # assign segments to prose tokens greedily by word-count proportion.

        segment_assignments = self._assign_segments_to_prose_tokens(
            self._ctx.tokens, self._ctx.prose_segments
        )
        # segment_assignments: list of (prose_token_index_in_tokens_list, segment_index)

        # Build the replacement text for each prose token position.
        # For tokens whose segment failed, keep the token's original text.
        token_replacements: dict[int, str] = {}

        # For segments with multiple prose tokens, track the cumulative
        # word offset so we can split the humanized text across tokens.
        segment_token_groups: dict[int, list[int]] = {}
        for tok_pos, seg_idx in segment_assignments:
            segment_token_groups.setdefault(seg_idx, []).append(tok_pos)

        for seg_idx, tok_positions in segment_token_groups.items():
            if seg_idx in failed_indices:
                # Failed segment: keep original prose token texts unchanged
                # (do not set token_replacements — the reconstruction loop
                # will fall back to the original token text).
                logger.debug(
                    "DocumentHumanizer.write_back: segment %d FAILED — "
                    "original text preserved (Req 19.6).",
                    seg_idx,
                )
                continue

            humanized_text = self._ctx.humanized_segments.get(seg_idx)
            if humanized_text is None:
                # Should not happen (non-failed segment with no humanized text),
                # but guard defensively.
                logger.warning(
                    "DocumentHumanizer.write_back: segment %d has no humanized "
                    "text and is not in failed_segments — keeping original.",
                    seg_idx,
                )
                continue

            if len(tok_positions) == 1:
                # Single prose token for this segment — direct substitution.
                token_replacements[tok_positions[0]] = humanized_text
            else:
                # Multiple prose tokens contribute to this segment — distribute
                # the humanized text proportionally by original word counts.
                original_texts = [
                    self._ctx.tokens[p].text for p in tok_positions
                ]
                orig_counts = [len(t.split()) for t in original_texts]
                total_orig = sum(orig_counts)
                humanized_words = humanized_text.split()
                total_human = len(humanized_words)

                word_pos = 0
                for i, tok_pos in enumerate(tok_positions):
                    if i == len(tok_positions) - 1:
                        chunk_words = humanized_words[word_pos:]
                    else:
                        share = round(total_human * orig_counts[i] / max(total_orig, 1))
                        chunk_words = humanized_words[word_pos:word_pos + share]
                        word_pos += share

                    # Preserve leading/trailing whitespace from original token.
                    orig = original_texts[i]
                    leading = orig[:len(orig) - len(orig.lstrip())]
                    trailing = orig[len(orig.rstrip()):]
                    chunk = leading + " ".join(chunk_words) + trailing
                    token_replacements[tok_pos] = chunk

        # Reconstruct the document: iterate over all tokens in order,
        # using replacement text for successfully-written prose tokens
        # and original text for everything else.
        reconstructed_parts: list[str] = []
        prose_token_seq_index = 0  # counts prose tokens seen so far

        # Map from prose-token sequential index → token list index
        prose_token_list_indices: list[int] = [
            i for i, t in enumerate(self._ctx.tokens) if t.kind == PROSE_TOKEN
        ]

        for list_idx, token in enumerate(self._ctx.tokens):
            if token.kind != PROSE_TOKEN:
                # Markup token: always preserved verbatim (Req 19.5).
                reconstructed_parts.append(token.text)
            else:
                replacement = token_replacements.get(list_idx)
                if replacement is not None:
                    reconstructed_parts.append(replacement)
                else:
                    # Failed or unassigned: keep original text (Req 19.6).
                    reconstructed_parts.append(token.text)

        # Reattach preamble / \end{document} to restore a valid document.
        body = "".join(reconstructed_parts)
        self._ctx.result_document = self._reattach_preamble(
            self._ctx.latex_source, body
        )

        # Determine completeness: complete iff no segment failed (Req 19.5).
        self._ctx.write_back_complete = len(failed_indices) == 0

        if self._ctx.write_back_complete:
            logger.info(
                "DocumentHumanizer.write_back: COMPLETE — %d segment(s) written back.",
                n_segments,
            )
        else:
            failed_summary = ", ".join(
                f"segment {i}: {self._ctx.failed_segments[i]}"
                for i in sorted(failed_indices)
            )
            logger.warning(
                "DocumentHumanizer.write_back: NOT COMPLETE — %d segment(s) failed. "
                "Failures: %s",
                len(failed_indices),
                failed_summary,
            )

        # Surface the result through the presenter if complete (Req 19.5).
        if self._ctx.write_back_complete and self._presenter is not None:
            out = self._presenter(self._ctx.result_document)
            if asyncio.iscoroutine(out):
                await out

        return {
            "write_back_complete": self._ctx.write_back_complete,
            "segments_written": n_segments - len(failed_indices),
            "failed_segments": dict(self._ctx.failed_segments),
        }

    # ------------------------------------------------------------------
    # Segment-to-prose-token assignment helper
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_segments_to_prose_tokens(
        tokens: list[DocumentToken],
        segments: list[str],
    ) -> list[tuple[int, int]]:
        """
        Assign each prose token position in *tokens* to a segment index in
        *segments*.

        Since ``segment_prose`` operates on the joined prose text and splits
        at sentence/paragraph boundaries, each segment corresponds to a
        consecutive run of words from the prose tokens in document order.

        We assign segments to prose tokens greedily by word-count:
        for each segment, accumulate prose tokens until their combined word
        count matches or exceeds the segment's word count.

        Returns
        -------
        list of (token_list_index, segment_index) pairs.
        """
        prose_token_positions: list[int] = [
            i for i, t in enumerate(tokens) if t.kind == PROSE_TOKEN
        ]

        if not prose_token_positions or not segments:
            return []

        assignments: list[tuple[int, int]] = []
        seg_idx = 0
        tok_cursor = 0
        seg_word_budget = len(segments[0].split()) if segments else 0
        accumulated_words = 0

        for tok_pos in prose_token_positions:
            token = tokens[tok_pos]
            tok_wc = len(token.text.split())
            assignments.append((tok_pos, seg_idx))
            accumulated_words += tok_wc

            # Advance to the next segment when the current one is satisfied.
            while (
                seg_idx < len(segments) - 1
                and accumulated_words >= seg_word_budget
            ):
                seg_idx += 1
                seg_word_budget = len(segments[seg_idx].split())
                accumulated_words = 0

        # Any remaining tokens after the last segment are assigned to the
        # last segment.
        return assignments

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reattach_preamble(original: str, body: str) -> str:
        """
        If *original* has a ``\\begin{document}`` preamble, prepend it and
        append ``\\end{document}`` to *body* to restore a valid document.

        When no preamble exists, returns *body* unchanged.
        """
        begin_match = re.search(r"\\begin\{document\}", original)
        end_match = re.search(r"\\end\{document\}", original)

        if begin_match:
            preamble = original[: begin_match.end()]
            suffix = original[end_match.start():] if end_match else ""
            return preamble + body + suffix

        return body


# ---------------------------------------------------------------------------
# Public high-level entry point
# ---------------------------------------------------------------------------


async def run_document_humanization(
    latex_source: str,
    *,
    latex_parser: LaTeXParser | None = None,
    prose_humanizer: ProseHumanizer | None = None,
    presenter: Callable[[str], Any] | None = None,
) -> HumanizationContext:
    """
    High-level entry point: run the document humanization automation
    end-to-end.

    Creates a :class:`HumanizationContext`, builds the
    :class:`~core.planner.CommandPlan`, executes it via a dedicated
    :class:`~core.execution.ExecutionEngine`, and returns the populated
    :class:`HumanizationContext`.

    Parameters
    ----------
    latex_source:
        The LaTeX document to humanize.  Must be non-empty and must
        contain separable prose (Reqs 19.1, 19.2).
    latex_parser:
        Optional custom :class:`LaTeXParser`.  Defaults to a new instance.
    prose_humanizer:
        Optional custom :class:`ProseHumanizer`.  Defaults to heuristic.
    presenter:
        Optional callback invoked with the final reconstructed document
        after ``write_back`` completes (Req 19.5).

    Returns
    -------
    HumanizationContext
        Fully populated context with:
        - ``tokens`` — interleaved prose/markup token list
        - ``prose_segments`` — segmented prose chunks
        - ``humanized_segments`` — per-index humanized text
        - ``failed_segments`` — per-index failure reason (Req 19.6)
        - ``result_document`` — final LaTeX document (when complete)
        - ``write_back_complete`` — ``True`` iff every segment was written
        - ``parse_error`` — human-readable reason if parsing failed (Req 19.2)

    Design: Built-in automations (Document Humanization).
    Requirements: 19.1, 19.2, 19.3, 19.4, 19.5, 19.6.
    """
    from core.execution.execution_engine import ExecutionEngine
    from core.execution.safety_gate import SafetyGate

    context = HumanizationContext(latex_source=latex_source)
    humanizer = DocumentHumanizer(
        context=context,
        latex_parser=latex_parser,
        prose_humanizer=prose_humanizer,
        presenter=presenter,
    )

    plan = humanizer.build_plan()

    engine = ExecutionEngine(
        safety_gate=SafetyGate(),
        actuator_callback=humanizer.actuator,
    )

    stream = await engine.execute(plan)
    async for _event in stream:
        pass  # Events are consumed; context is populated as a side-effect.

    return context


# ---------------------------------------------------------------------------
# Public factory: register_builtin_automation
# ---------------------------------------------------------------------------


def register_builtin_automation(
    library: Any,  # AutomationLibrary — avoided circular import with Any
    *,
    latex_parser: LaTeXParser | None = None,
    prose_humanizer: ProseHumanizer | None = None,
) -> None:
    """
    Register the document-humanization built-in automation in *library*.

    The automation is registered under the name
    :data:`AUTOMATION_NAME` (``"document_humanization"``).

    Because actual execution requires runtime context (``latex_source``),
    callers that need per-run context should use
    :func:`run_document_humanization` directly.

    Parameters
    ----------
    library:
        The :class:`~core.automation.AutomationLibrary` instance to
        register in.
    latex_parser:
        Optional :class:`LaTeXParser` (for documentation only).
    prose_humanizer:
        Optional :class:`ProseHumanizer` (for documentation only).

    Requirements: 19.1–19.6.
    """
    representative_steps: list[Step] = [
        Step(
            id="parse_latex",
            intent=(
                "Parse LaTeX source and separate prose from markup "
                "(Req 19.1; unparseable/no-prose → don't start, Req 19.2)"
            ),
            actuator=Actuator.INTERNAL,
            args={},
            depends_on=[],
            classification=StepClassification.REVERSIBLE,
            required_slots=["latex_source"],
        ),
        Step(
            id="segment_prose",
            intent=(
                "Segment extracted prose into 800–1200 word chunks "
                "(final segment may be shorter, Req 19.3)"
            ),
            actuator=Actuator.INTERNAL,
            args={"min_words": 800, "max_words": 1200},
            depends_on=["parse_latex"],
            classification=StepClassification.REVERSIBLE,
            required_slots=[],
        ),
        Step(
            id="humanize_segments",
            intent=(
                "Humanize each prose segment via Language_Engine "
                "preserving meaning (Req 19.4)"
            ),
            actuator=Actuator.INTERNAL,
            args={},
            depends_on=["segment_prose"],
            classification=StepClassification.REVERSIBLE,
            required_slots=[],
        ),
        Step(
            id="write_back",
            intent=(
                "Write humanized prose back into LaTeX document preserving markup; "
                "complete only when every segment is written (Req 19.5); "
                "per-segment failure → not-complete + warn (Req 19.6)"
            ),
            actuator=Actuator.INTERNAL,
            args={},
            depends_on=["humanize_segments"],
            classification=StepClassification.REVERSIBLE,
            required_slots=[],
        ),
    ]

    library.define(
        name=AUTOMATION_NAME,
        steps=representative_steps,
        description=(
            "Built-in automation: humanize a LaTeX document segment by segment, "
            "rewriting prose while preserving all LaTeX markup. "
            "Design: Built-in automations (Document Humanization). "
            "Requirements: 19.1, 19.2, 19.3, 19.4, 19.5, 19.6."
        ),
    )
    logger.info(
        "AutomationLibrary: registered built-in '%s' (%d steps).",
        AUTOMATION_NAME,
        len(representative_steps),
    )
