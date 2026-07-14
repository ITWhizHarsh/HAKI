"""
Question-paper analysis built-in automation (Requirement 18).

This module implements the question-paper analysis automation as a built-in
that is registered with the :class:`~core.automation.AutomationLibrary` at
startup.

The automation is invoked by exact name ``"question_paper_analysis"`` and
accepts a list of question-paper texts (strings) as input through a shared
``AnalysisContext`` object.

Execution plan (all steps are REVERSIBLE — no confirmations required):

    Step 1  validate_input          Guard: require ≥1 paper; else don't start (Req 18.1).
    Step 2  extract_topics_<n>      One step per paper; extracts topics (parallel).
                                    Papers that fail are tracked but do not stop others
                                    (partial processing, Req 18.6).
    Step 3  identify_recurring      Aggregates per-paper topics; identifies topics that
                                    appear in ≥2 papers (Req 18.2).
    Step 4  cross_reference         Queries Memory_Brain/RAG to annotate each recurring
                                    topic with matching course content (Req 18.3).
    Step 5  present_results         Builds the prioritised chapter list ordered by
                                    recurrence count and marks the automation complete
                                    (Reqs 18.4, 18.5).

The automation is considered COMPLETE only after Step 5 presents the list
(Req 18.5).  If one or more per-paper extraction steps fail, the remaining
steps execute over the successfully processed papers and the final report
includes the failures and their reasons (Req 18.6).

Design: Built-in automations (Question-Paper Analysis).
Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 18.6.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from core.planner import (
    Actuator,
    CommandPlan,
    Step,
    StepClassification,
    StepStatus,
)

logger = logging.getLogger(__name__)

# Public name used to register and invoke this automation.
AUTOMATION_NAME = "question_paper_analysis"


# ---------------------------------------------------------------------------
# AnalysisContext — shared mutable state for one analysis run
# ---------------------------------------------------------------------------


@dataclass
class AnalysisContext:
    """
    Shared mutable state threaded through all steps of one analysis run.

    Parameters
    ----------
    papers:
        List of question-paper texts to analyse.  Each element is the
        full text of a single paper (string).

    Attributes
    ----------
    per_paper_topics:
        Maps paper index → list of extracted topic strings.  Populated
        by the ``extract_topics`` steps.
    failed_papers:
        Maps paper index → human-readable failure reason for papers whose
        extraction failed (Req 18.6).
    recurring_topics:
        Counter mapping topic string → recurrence count (number of
        papers in which the topic appeared).  Populated by the
        ``identify_recurring`` step.
    annotated_topics:
        Maps topic string → list of matching course-content excerpts
        retrieved from Memory_Brain (Req 18.3).  Populated by the
        ``cross_reference`` step.
    result_presented:
        Set to ``True`` only after the ``present_results`` step emits the
        prioritised list (Req 18.5).
    prioritised_list:
        Final output: ordered list of ``(topic, recurrence_count,
        [annotations])`` triples ready for display (Req 18.4).
    """

    papers: list[str]

    per_paper_topics: dict[int, list[str]] = field(default_factory=dict)
    failed_papers: dict[int, str] = field(default_factory=dict)
    recurring_topics: Counter = field(default_factory=Counter)
    annotated_topics: dict[str, list[str]] = field(default_factory=dict)
    result_presented: bool = False
    prioritised_list: list[tuple[str, int, list[str]]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Topic extractor interface (injected for testability)
# ---------------------------------------------------------------------------


class TopicExtractor:
    """
    Extracts topic keywords from a question-paper text.

    The default implementation uses a simple heuristic (unique
    meaningful words).  In production this is replaced by an LLM call
    via the :class:`~core.model_provider.ModelProvider`.

    Parameters
    ----------
    model_provider:
        Optional :class:`~core.model_provider.ModelProvider` for the
        ``Capability.LLM`` capability.  When ``None``, the heuristic
        extractor is used (suitable for testing without models).
    """

    # A basic stop-word set used by the heuristic extractor.
    _STOP: frozenset[str] = frozenset({
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "it", "this", "that", "which", "who",
        "what", "how", "when", "where", "not", "no", "if", "as", "so",
        "than", "then", "q", "question", "marks", "section", "part",
        "answer", "write", "explain", "describe", "define", "state",
        "list", "discuss", "give", "find", "show", "calculate", "derive",
        "prove", "compare", "contrast", "comment", "briefly",
    })

    def __init__(self, model_provider: Any | None = None) -> None:
        self._model_provider = model_provider

    def extract(self, paper_text: str) -> list[str]:
        """
        Extract a list of topic keywords from *paper_text*.

        When a model provider is available and configured for LLM
        capability, delegates to it; otherwise uses a heuristic.

        Parameters
        ----------
        paper_text:
            Full text of one question paper.

        Returns
        -------
        list[str]
            Deduplicated list of topic strings (lowercased).

        Raises
        ------
        RuntimeError
            If the model call fails, so the per-paper step can catch it,
            record the failure, and continue with remaining papers
            (Req 18.6).
        """
        if self._model_provider is not None:
            try:
                return self._extract_via_llm(paper_text)
            except Exception as exc:
                raise RuntimeError(
                    f"LLM topic extraction failed: {exc}"
                ) from exc

        return self._extract_heuristic(paper_text)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_via_llm(self, paper_text: str) -> list[str]:
        """
        Call the LLM (via ModelProvider) to extract topics.

        The prompt instructs the model to return a comma-separated list
        of topic keywords relevant to the examination questions in the
        paper.  The result is parsed and deduplicated.
        """
        prompt = (
            "You are an academic assistant.  Read the following examination "
            "question paper and extract the main academic topics or concepts "
            "that are tested.  Return ONLY a comma-separated list of topic "
            "keywords (no explanations, no numbering).  Be specific and "
            "concise.\n\nQuestion paper:\n"
            + paper_text[:8000]  # Guard against very long papers
        )
        raw: str = self._model_provider.invoke(prompt)
        topics = [t.strip().lower() for t in raw.split(",") if t.strip()]
        return list(dict.fromkeys(topics))  # deduplicate, preserve order

    def _extract_heuristic(self, paper_text: str) -> list[str]:
        """
        Simple heuristic: extract unique meaningful words of length ≥4
        that are not stop words.  Groups of 2–3 adjacent meaningful words
        are also included as bigrams/trigrams via a sliding window.
        """
        import re

        # Tokenise on non-alphanumeric runs
        tokens = [
            t.lower()
            for t in re.split(r"[^a-zA-Z]+", paper_text)
            if len(t) >= 4 and t.lower() not in self._STOP
        ]
        seen: dict[str, None] = {}

        # Unigrams
        for t in tokens:
            seen.setdefault(t, None)

        # Bigrams
        for i in range(len(tokens) - 1):
            phrase = f"{tokens[i]} {tokens[i+1]}"
            seen.setdefault(phrase, None)

        return list(seen.keys())[:50]  # cap to 50 topics per paper


# ---------------------------------------------------------------------------
# QuestionPaperAnalyzer — builds and drives the analysis plan
# ---------------------------------------------------------------------------


class QuestionPaperAnalyzer:
    """
    Orchestrates a single question-paper analysis run.

    This class is NOT a long-lived service — one instance is created per
    analysis invocation.  It builds an :class:`~core.planner.CommandPlan`
    from an :class:`AnalysisContext` and exposes an async actuator callback
    suitable for use with :class:`~core.execution.ExecutionEngine`.

    Parameters
    ----------
    context:
        The shared :class:`AnalysisContext` for this run.  Mutated in-place
        by each step.
    memory_brain:
        Optional :class:`~core.memory.MemoryBrain` for RAG cross-reference
        (Req 18.3).  When ``None``, cross-reference is skipped and topics
        are annotated as having no course content.
    topic_extractor:
        The :class:`TopicExtractor` to use.  Defaults to a heuristic
        extractor when ``None``.
    presenter:
        An optional async callable ``(prioritised_list) → None`` called
        after the list is built.  Use this to surface the results to the
        user through the voice or UI layer.  When ``None``, the result is
        logged at INFO level.
    """

    def __init__(
        self,
        context: AnalysisContext,
        memory_brain: Any | None = None,
        topic_extractor: TopicExtractor | None = None,
        presenter: Callable[[list[tuple[str, int, list[str]]]], Any] | None = None,
    ) -> None:
        self._ctx = context
        self._memory = memory_brain
        self._extractor = topic_extractor or TopicExtractor()
        self._presenter = presenter

    # ------------------------------------------------------------------
    # Plan factory
    # ------------------------------------------------------------------

    def build_plan(self) -> CommandPlan:
        """
        Build the :class:`~core.planner.CommandPlan` for this analysis run.

        Returns
        -------
        CommandPlan
            A plan whose steps map 1-to-1 to the five analysis phases
            described in the module docstring.

        The plan is constructed fresh each time so it can be handed to
        the :class:`~core.execution.ExecutionEngine` via the
        :class:`~core.automation.AutomationLibrary`.
        """
        plan_id = str(uuid.uuid4())
        n_papers = len(self._ctx.papers)

        # Step 1: validate ≥1 paper (Req 18.1)
        validate_step = Step(
            id="validate_input",
            intent="Validate that at least one question paper was provided",
            actuator=Actuator.INTERNAL,
            args={"min_papers": 1},
            depends_on=[],
            classification=StepClassification.REVERSIBLE,
            required_slots=[],
            postcondition=None,
        )

        # Steps 2a…2n: one extract_topics step per paper (parallel) (Req 18.6)
        extract_steps: list[Step] = []
        for i in range(n_papers):
            extract_steps.append(
                Step(
                    id=f"extract_topics_{i}",
                    intent=f"Extract topics from question paper {i + 1}",
                    actuator=Actuator.INTERNAL,
                    args={"paper_index": i},
                    depends_on=["validate_input"],
                    classification=StepClassification.REVERSIBLE,
                    required_slots=[],
                    postcondition=None,
                )
            )

        extract_ids = [s.id for s in extract_steps]

        # Step 3: identify recurring topics (Req 18.2).
        # Always depends on validate_input (plus all extract steps) so that
        # a validate failure (e.g. 0 papers) stops the whole plan even when
        # there are no per-paper extraction steps to block on.
        identify_depends = ["validate_input"] + extract_ids
        identify_step = Step(
            id="identify_recurring",
            intent="Identify topics recurring in 2 or more question papers",
            actuator=Actuator.INTERNAL,
            args={},
            depends_on=identify_depends,
            classification=StepClassification.REVERSIBLE,
            required_slots=[],
            postcondition=None,
        )

        # Step 4: cross-reference with Memory_Brain / RAG (Req 18.3)
        cross_ref_step = Step(
            id="cross_reference",
            intent="Cross-reference recurring topics against course content in Memory_Brain",
            actuator=Actuator.INTERNAL,
            args={},
            depends_on=["identify_recurring"],
            classification=StepClassification.REVERSIBLE,
            required_slots=[],
            postcondition=None,
        )

        # Step 5: present prioritised list (Reqs 18.4, 18.5)
        present_step = Step(
            id="present_results",
            intent="Present the prioritised chapter list ordered by recurrence",
            actuator=Actuator.INTERNAL,
            args={},
            depends_on=["cross_reference"],
            classification=StepClassification.REVERSIBLE,
            required_slots=[],
            postcondition=None,
        )

        all_steps = [validate_step] + extract_steps + [
            identify_step,
            cross_ref_step,
            present_step,
        ]

        return CommandPlan(
            id=plan_id,
            origin_command="run_automation:question_paper_analysis",
            steps=all_steps,
        )

    # ------------------------------------------------------------------
    # Actuator callback — dispatches each step by ID
    # ------------------------------------------------------------------

    async def actuator(self, step: Step) -> Any:
        """
        Async actuator callback for this analysis run.

        Dispatches to the appropriate handler based on the step ID
        prefix.  Designed to be passed as ``actuator_callback`` to a
        dedicated :class:`~core.execution.ExecutionEngine` instance.

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
        RuntimeError
            For validate failures (Req 18.1) or unrecognised step IDs.
        """
        sid = step.id

        if sid == "validate_input":
            return await self._step_validate_input(step)

        if sid.startswith("extract_topics_"):
            paper_index = int(sid.split("_")[-1])
            return await self._step_extract_topics(paper_index)

        if sid == "identify_recurring":
            return await self._step_identify_recurring()

        if sid == "cross_reference":
            return await self._step_cross_reference()

        if sid == "present_results":
            return await self._step_present_results()

        raise RuntimeError(
            f"QuestionPaperAnalyzer: unknown step id '{sid}'"
        )

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    async def _step_validate_input(self, step: Step) -> dict:
        """
        Guard step: require ≥1 paper (Req 18.1).

        Raises ``RuntimeError`` (caught by the execution engine, which
        marks the step FAILED and stops all dependents) when no papers
        are provided.
        """
        n = len(self._ctx.papers)
        min_papers: int = step.args.get("min_papers", 1)
        if n < min_papers:
            raise RuntimeError(
                "Question-paper analysis requires at least one question paper. "
                "Please provide one or more paper texts and try again."
            )
        logger.info(
            "QuestionPaperAnalyzer.validate_input: %d paper(s) provided — OK.",
            n,
        )
        return {"papers_count": n}

    async def _step_extract_topics(self, paper_index: int) -> dict:
        """
        Extract topics from the paper at *paper_index* (Req 18.2).

        This step DOES NOT raise on extraction failure — it instead
        records the failure in ``context.failed_papers`` so that the
        identify_recurring step can compute over the remaining papers
        (Req 18.6).  The step still returns a dict so the execution
        engine marks it COMPLETED and does not stop dependents.

        Returns
        -------
        dict
            ``{"paper_index": int, "topics": [str], "error": str|None}``
        """
        paper_text = self._ctx.papers[paper_index]

        loop = asyncio.get_event_loop()
        try:
            topics: list[str] = await loop.run_in_executor(
                None, self._extractor.extract, paper_text
            )
            self._ctx.per_paper_topics[paper_index] = topics
            logger.info(
                "QuestionPaperAnalyzer.extract_topics[%d]: extracted %d topic(s).",
                paper_index,
                len(topics),
            )
            return {"paper_index": paper_index, "topics": topics, "error": None}

        except Exception as exc:  # noqa: BLE001
            reason = str(exc)
            self._ctx.failed_papers[paper_index] = reason
            logger.warning(
                "QuestionPaperAnalyzer.extract_topics[%d]: FAILED — %s",
                paper_index,
                reason,
            )
            # Return (not raise) so dependents continue over processed papers.
            return {"paper_index": paper_index, "topics": [], "error": reason}

    async def _step_identify_recurring(self) -> dict:
        """
        Identify topics appearing in ≥2 successfully processed papers
        (Req 18.2).

        Returns
        -------
        dict
            ``{"recurring_topics": {topic: count}, "total_papers_processed": int}``
        """
        topic_counter: Counter[str] = Counter()
        n_processed = len(self._ctx.per_paper_topics)

        for topics in self._ctx.per_paper_topics.values():
            # Count each topic once per paper (not per occurrence within paper)
            topic_counter.update(set(topics))

        # Retain only those appearing in ≥2 papers (Req 18.2)
        recurring = Counter({t: c for t, c in topic_counter.items() if c >= 2})
        self._ctx.recurring_topics = recurring

        logger.info(
            "QuestionPaperAnalyzer.identify_recurring: %d recurring topic(s) "
            "from %d processed paper(s).",
            len(recurring),
            n_processed,
        )
        return {
            "recurring_topics": dict(recurring),
            "total_papers_processed": n_processed,
        }

    async def _step_cross_reference(self) -> dict:
        """
        Query Memory_Brain for each recurring topic and annotate it with
        matching course content (Req 18.3).

        When no Memory_Brain is configured or a topic has no matching
        notes, the topic is annotated with an empty list.

        Returns
        -------
        dict
            ``{"annotations": {topic: [excerpt_str]}}``
        """
        annotations: dict[str, list[str]] = {}

        if not self._ctx.recurring_topics:
            self._ctx.annotated_topics = annotations
            return {"annotations": annotations}

        for topic in self._ctx.recurring_topics:
            if self._memory is not None:
                try:
                    # Retrieve up to 3 notes per topic (lightweight) (Req 18.3)
                    notes = await self._memory.aretrieve(topic, k=3)
                    # Use the first 200 chars of each note body as the excerpt
                    excerpts = [
                        n.body[:200].strip() for n in notes if n.body.strip()
                    ]
                    annotations[topic] = excerpts
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "QuestionPaperAnalyzer.cross_reference: "
                        "Memory_Brain query failed for topic '%s': %s",
                        topic,
                        exc,
                    )
                    annotations[topic] = []
            else:
                annotations[topic] = []

        self._ctx.annotated_topics = annotations
        logger.info(
            "QuestionPaperAnalyzer.cross_reference: annotated %d topic(s).",
            len(annotations),
        )
        return {"annotations": annotations}

    async def _step_present_results(self) -> dict:
        """
        Build and present the prioritised chapter list (Reqs 18.4, 18.5).

        The list is ordered by descending recurrence count.  Topics with
        the same count are sorted alphabetically for stable output.

        The automation is considered complete ONLY when this step
        executes successfully (Req 18.5).

        Returns
        -------
        dict
            ``{"prioritised_list": [...], "failed_papers": {...}}``

        The returned dict is also stored in ``context.prioritised_list``
        and ``context.result_presented`` is set to ``True``.
        """
        # Sort by recurrence (desc) then topic name (asc) for stability
        prioritised: list[tuple[str, int, list[str]]] = sorted(
            [
                (
                    topic,
                    count,
                    self._ctx.annotated_topics.get(topic, []),
                )
                for topic, count in self._ctx.recurring_topics.items()
            ],
            key=lambda x: (-x[1], x[0]),
        )

        self._ctx.prioritised_list = prioritised
        self._ctx.result_presented = True

        # Log a human-readable summary
        if prioritised:
            summary_lines = [
                f"  {i+1}. '{t}' (in {c} papers)"
                + (f" — {len(a)} course note(s)" if a else "")
                for i, (t, c, a) in enumerate(prioritised)
            ]
            logger.info(
                "QuestionPaperAnalyzer.present_results: prioritised list "
                "(%d entries):\n%s",
                len(prioritised),
                "\n".join(summary_lines),
            )
        else:
            logger.info(
                "QuestionPaperAnalyzer.present_results: no recurring topics found "
                "(each topic appeared in only one paper or no papers were processed)."
            )

        # Include failed-paper report in the output (Req 18.6)
        if self._ctx.failed_papers:
            failures_summary = ", ".join(
                f"paper {i+1}: {reason}"
                for i, reason in sorted(self._ctx.failed_papers.items())
            )
            logger.warning(
                "QuestionPaperAnalyzer.present_results: %d paper(s) failed "
                "extraction — results are computed over processed papers only. "
                "Failures: %s",
                len(self._ctx.failed_papers),
                failures_summary,
            )

        # Surface results through the presenter (voice/UI layer) (Req 18.5)
        if self._presenter is not None:
            result = self._presenter(prioritised)
            if asyncio.iscoroutine(result):
                await result

        return {
            "prioritised_list": prioritised,
            "failed_papers": {
                i: reason for i, reason in self._ctx.failed_papers.items()
            },
        }


# ---------------------------------------------------------------------------
# Public factory: register_builtin_automation
# ---------------------------------------------------------------------------


async def run_question_paper_analysis(
    papers: list[str],
    *,
    memory_brain: Any | None = None,
    topic_extractor: TopicExtractor | None = None,
    presenter: Callable[[list[tuple[str, int, list[str]]]], Any] | None = None,
) -> AnalysisContext:
    """
    High-level entry point: run the question-paper analysis end-to-end.

    Creates an :class:`AnalysisContext`, builds the
    :class:`~core.planner.CommandPlan`, executes it via a dedicated
    :class:`~core.execution.ExecutionEngine`, and returns the populated
    :class:`AnalysisContext`.

    This function is used when the automation is invoked directly (e.g.
    from the orchestrator intent router) rather than through the
    :class:`~core.automation.AutomationLibrary`'s ``run()`` method.

    Parameters
    ----------
    papers:
        List of question-paper full texts.  Must contain ≥1 element;
        an empty list causes the validate_input step to fail (Req 18.1).
    memory_brain:
        Optional :class:`~core.memory.MemoryBrain` for RAG annotations.
    topic_extractor:
        Optional custom :class:`TopicExtractor` (default: heuristic).
    presenter:
        Optional async/sync callable invoked with the prioritised list
        after ``present_results`` runs (Req 18.5).

    Returns
    -------
    AnalysisContext
        Fully populated context with:
        - ``per_paper_topics`` — extracted topics per processed paper
        - ``failed_papers`` — papers that failed and why (Req 18.6)
        - ``recurring_topics`` — Counter of topics in ≥2 papers
        - ``annotated_topics`` — topics annotated with course notes
        - ``prioritised_list`` — final ordered output (Req 18.4)
        - ``result_presented`` — ``True`` when the list was presented
          (Req 18.5); ``False`` if execution was interrupted before
          ``present_results`` ran.

    Examples
    --------
    >>> import asyncio
    >>> from core.automation.question_paper_analysis import run_question_paper_analysis
    >>> papers = ["Q1. Explain TCP/IP layers. Q2. Describe OSI model.", ...]
    >>> ctx = asyncio.run(run_question_paper_analysis(papers))
    >>> print(ctx.prioritised_list)
    """
    from core.execution.execution_engine import ExecutionEngine
    from core.execution.safety_gate import SafetyGate

    context = AnalysisContext(papers=papers)
    analyzer = QuestionPaperAnalyzer(
        context=context,
        memory_brain=memory_brain,
        topic_extractor=topic_extractor,
        presenter=presenter,
    )

    plan = analyzer.build_plan()

    # Use a fresh ExecutionEngine with the analyzer's actuator callback.
    # No confirmation gate is needed (all steps are REVERSIBLE).
    engine = ExecutionEngine(
        safety_gate=SafetyGate(),
        actuator_callback=analyzer.actuator,
    )

    stream = await engine.execute(plan)
    async for _event in stream:
        pass  # Events are consumed; context is populated as a side-effect.

    return context


def register_builtin_automation(
    library: Any,  # AutomationLibrary — avoided circular import with Any
    *,
    memory_brain: Any | None = None,
    topic_extractor: TopicExtractor | None = None,
) -> None:
    """
    Register the question-paper analysis built-in automation in *library*.

    The automation is registered under the name
    :data:`AUTOMATION_NAME` (``"question_paper_analysis"``).

    Because the :class:`~core.automation.AutomationLibrary` stores static
    :class:`~core.planner.Step` objects (without per-run context), the
    question-paper analysis automation requires runtime context
    (``papers``).  The registration here stores a sentinel plan with a
    single INTERNAL step whose ``args`` describe the automation's
    interface.  Callers that need per-run context should use
    :func:`run_question_paper_analysis` directly.

    To run with real paper data, bypass ``library.run()`` and call
    :func:`run_question_paper_analysis` instead.

    Parameters
    ----------
    library:
        The :class:`~core.automation.AutomationLibrary` instance to
        register in.
    memory_brain:
        Optional :class:`~core.memory.MemoryBrain` stored on the
        registration record (for documentation purposes only; actual
        execution wires the brain at run time via
        :func:`run_question_paper_analysis`).
    topic_extractor:
        Optional :class:`TopicExtractor` (for documentation only).

    Requirements: 18.1–18.6.
    """
    # Build a representative set of steps for the registration record.
    # These document the automation's shape; real execution uses
    # run_question_paper_analysis() which builds a context-specific plan.
    representative_steps: list[Step] = [
        Step(
            id="validate_input",
            intent=(
                "Validate that at least one question paper is provided "
                "(Req 18.1: ≥1 paper required)"
            ),
            actuator=Actuator.INTERNAL,
            args={"min_papers": 1},
            depends_on=[],
            classification=StepClassification.REVERSIBLE,
            required_slots=["papers"],
        ),
        Step(
            id="extract_topics",
            intent=(
                "Extract topics from each question paper "
                "(one step per paper, run in parallel; Req 18.2)"
            ),
            actuator=Actuator.INTERNAL,
            args={"paper_index": "<varies>"},
            depends_on=["validate_input"],
            classification=StepClassification.REVERSIBLE,
            required_slots=[],
        ),
        Step(
            id="identify_recurring",
            intent=(
                "Identify topics recurring in 2 or more papers "
                "(Req 18.2)"
            ),
            actuator=Actuator.INTERNAL,
            args={},
            depends_on=["extract_topics"],
            classification=StepClassification.REVERSIBLE,
            required_slots=[],
        ),
        Step(
            id="cross_reference",
            intent=(
                "Cross-reference recurring topics against Memory_Brain/RAG "
                "course content and annotate (Req 18.3)"
            ),
            actuator=Actuator.INTERNAL,
            args={},
            depends_on=["identify_recurring"],
            classification=StepClassification.REVERSIBLE,
            required_slots=[],
        ),
        Step(
            id="present_results",
            intent=(
                "Present the prioritised chapter list ordered by recurrence "
                "(Reqs 18.4, 18.5); includes failed-paper report (Req 18.6)"
            ),
            actuator=Actuator.INTERNAL,
            args={},
            depends_on=["cross_reference"],
            classification=StepClassification.REVERSIBLE,
            required_slots=[],
        ),
    ]

    library.define(
        name=AUTOMATION_NAME,
        steps=representative_steps,
        description=(
            "Analyses one or more examination question papers: extracts topics "
            "per paper, identifies topics recurring in ≥2 papers, cross-references "
            "them against course notes in Memory_Brain, and presents a prioritised "
            "chapter list ordered by recurrence. "
            "Requires ≥1 paper (Req 18.1). "
            "Partial processing completes over processed papers and reports "
            "failures (Req 18.6). "
            "Design: Built-in automations (Question-Paper Analysis). "
            "Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 18.6."
        ),
    )
    logger.info(
        "QuestionPaperAnalysis: registered built-in automation '%s'.",
        AUTOMATION_NAME,
    )
