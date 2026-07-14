"""
Tests for the question-paper analysis built-in automation (Task 34.1).

Covers the core implementation requirements:

    18.1  ≥1 paper required; analysis does not start with 0 papers.
    18.2  Topics recurring in ≥2 papers are identified.
    18.3  Recurring topics are annotated with matching Memory_Brain notes.
    18.4  Prioritised list is ordered by descending recurrence count.
    18.5  Automation is complete only when the list is presented.
    18.6  Partial processing: continues over processed papers and reports failures.

All model-backed logic (topic extraction, Memory_Brain) is mocked so the
tests exercise routing, gating, ordering, and atomicity without real models.

Design: Built-in automations (Question-Paper Analysis).
Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 18.6.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.automation.question_paper_analysis import (
    AUTOMATION_NAME,
    AnalysisContext,
    QuestionPaperAnalyzer,
    TopicExtractor,
    register_builtin_automation,
    run_question_paper_analysis,
)
from core.automation import AutomationLibrary


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_extractor(topics_by_paper: dict[int, list[str]]) -> TopicExtractor:
    """
    Return a TopicExtractor whose ``extract`` method returns a pre-set list
    of topics for each call (in order of call).

    Parameters
    ----------
    topics_by_paper:
        Maps paper index → topic list.  If the index is not in the map,
        the extractor raises RuntimeError to simulate a failure.
    """
    extractor = TopicExtractor(model_provider=None)

    call_count = {"n": 0}

    def fake_extract(paper_text: str) -> list[str]:
        idx = call_count["n"]
        call_count["n"] += 1
        if idx not in topics_by_paper:
            raise RuntimeError(f"Simulated extraction failure for paper {idx}")
        return topics_by_paper[idx]

    extractor.extract = fake_extract  # type: ignore[method-assign]
    return extractor


def _make_memory(retrieval_map: dict[str, list[str]]) -> Any:
    """
    Return a mock MemoryBrain whose ``aretrieve(query, k)`` returns a list
    of mock notes whose ``body`` is the string from *retrieval_map[query]*.

    For queries not in the map, returns an empty list.
    """
    mock_brain = MagicMock()

    async def fake_aretrieve(query: str, k: int = 5):
        notes = []
        for body_text in retrieval_map.get(query, []):
            note = MagicMock()
            note.body = body_text
            notes.append(note)
        return notes

    mock_brain.aretrieve = fake_aretrieve
    return mock_brain


# ---------------------------------------------------------------------------
# Test 18.1 — require ≥1 paper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_papers_list_does_not_complete():
    """
    Req 18.1: If no papers are provided the validate_input step fails and
    the automation does NOT complete (result_presented stays False).
    """
    ctx = await run_question_paper_analysis(papers=[])
    # validate_input should have failed → present_results never ran
    assert ctx.result_presented is False


@pytest.mark.asyncio
async def test_single_paper_is_accepted():
    """
    Req 18.1: A single paper is sufficient to start the analysis.
    """
    extractor = _make_extractor({0: ["calculus", "integration"]})
    ctx = await run_question_paper_analysis(
        papers=["Q1. Differentiate sin(x)."],
        topic_extractor=extractor,
    )
    # With only one paper, no topic can recur in ≥2 papers,
    # but the automation must still present an (empty) list and complete.
    assert ctx.result_presented is True


# ---------------------------------------------------------------------------
# Test 18.2 — recurring topics (≥2 papers)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_topics_recurring_in_two_or_more_papers_are_identified():
    """
    Req 18.2: Topics appearing in ≥2 papers are classified as recurring;
    topics unique to one paper are excluded.
    """
    extractor = _make_extractor(
        {
            0: ["tcp/ip", "osi model", "routing"],
            1: ["tcp/ip", "switching", "osi model"],
            2: ["routing", "switching"],
        }
    )
    ctx = await run_question_paper_analysis(
        papers=["paper1", "paper2", "paper3"],
        topic_extractor=extractor,
    )
    # "tcp/ip" in papers 0 and 1 → recurrence 2
    assert ctx.recurring_topics["tcp/ip"] == 2
    # "osi model" in papers 0 and 1 → recurrence 2
    assert ctx.recurring_topics["osi model"] == 2
    # "routing" in papers 0 and 2 → recurrence 2
    assert ctx.recurring_topics["routing"] == 2
    # "switching" in papers 1 and 2 → recurrence 2
    assert ctx.recurring_topics["switching"] == 2
    # No topic with recurrence < 2 should be in recurring_topics
    for topic, count in ctx.recurring_topics.items():
        assert count >= 2, f"Non-recurring topic '{topic}' ({count}) in recurring set"


@pytest.mark.asyncio
async def test_topic_appearing_in_only_one_paper_is_not_recurring():
    """
    Req 18.2: A topic that appears in only a single paper must NOT be in
    the recurring set.
    """
    extractor = _make_extractor(
        {
            0: ["algorithms", "sorting", "unique_only_paper_0"],
            1: ["algorithms", "sorting"],
        }
    )
    ctx = await run_question_paper_analysis(
        papers=["paper1", "paper2"],
        topic_extractor=extractor,
    )
    assert "unique_only_paper_0" not in ctx.recurring_topics


# ---------------------------------------------------------------------------
# Test 18.3 — cross-reference annotations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recurring_topics_are_annotated_from_memory_brain():
    """
    Req 18.3: Each recurring topic is annotated with content retrieved from
    Memory_Brain.
    """
    extractor = _make_extractor(
        {0: ["matrices", "eigenvalues"], 1: ["matrices", "determinants"]}
    )
    memory = _make_memory(
        {
            "matrices": ["Matrices are rectangular arrays of numbers."],
            "eigenvalues": [],           # No course content available
            "determinants": [],          # No course content
        }
    )
    ctx = await run_question_paper_analysis(
        papers=["paper1", "paper2"],
        topic_extractor=extractor,
        memory_brain=memory,
    )
    # "matrices" recurs in both papers
    assert "matrices" in ctx.annotated_topics
    assert len(ctx.annotated_topics["matrices"]) == 1
    assert "Matrices are rectangular arrays" in ctx.annotated_topics["matrices"][0]


@pytest.mark.asyncio
async def test_topic_with_no_course_content_has_empty_annotation():
    """
    Req 18.3: A recurring topic with no matching course content in
    Memory_Brain is annotated with an empty list (not an error).
    """
    extractor = _make_extractor(
        {0: ["thermodynamics"], 1: ["thermodynamics"]}
    )
    memory = _make_memory({})   # No notes at all
    ctx = await run_question_paper_analysis(
        papers=["paper1", "paper2"],
        topic_extractor=extractor,
        memory_brain=memory,
    )
    assert ctx.annotated_topics.get("thermodynamics") == []


# ---------------------------------------------------------------------------
# Test 18.4 — prioritised list ordered by recurrence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prioritised_list_ordered_by_descending_recurrence():
    """
    Req 18.4: The prioritised chapter list must be ordered by descending
    recurrence count.
    """
    extractor = _make_extractor(
        {
            0: ["topic_a", "topic_b", "topic_c"],
            1: ["topic_a", "topic_b"],
            2: ["topic_a"],
        }
    )
    ctx = await run_question_paper_analysis(
        papers=["paper1", "paper2", "paper3"],
        topic_extractor=extractor,
    )
    recurrence_counts = [count for _, count, _ in ctx.prioritised_list]
    assert recurrence_counts == sorted(recurrence_counts, reverse=True), (
        f"Prioritised list is not sorted by descending recurrence: {ctx.prioritised_list}"
    )


@pytest.mark.asyncio
async def test_prioritised_list_includes_all_recurring_topics():
    """
    Req 18.4: Every topic identified as recurring must appear in the list.
    """
    extractor = _make_extractor(
        {
            0: ["alpha", "beta", "gamma"],
            1: ["alpha", "gamma"],
        }
    )
    ctx = await run_question_paper_analysis(
        papers=["paper1", "paper2"],
        topic_extractor=extractor,
    )
    list_topics = {t for t, _, _ in ctx.prioritised_list}
    for topic in ctx.recurring_topics:
        assert topic in list_topics, f"Recurring topic '{topic}' missing from list"


# ---------------------------------------------------------------------------
# Test 18.5 — complete only when list is presented
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_result_presented_flag_is_true_on_full_success():
    """
    Req 18.5: ``result_presented`` must be ``True`` after successful execution.
    """
    extractor = _make_extractor({0: ["physics"], 1: ["physics"]})
    ctx = await run_question_paper_analysis(
        papers=["paper1", "paper2"],
        topic_extractor=extractor,
    )
    assert ctx.result_presented is True


@pytest.mark.asyncio
async def test_presenter_callback_is_invoked_with_prioritised_list():
    """
    Req 18.5: The optional presenter callback must be called with the final
    prioritised list so the user can receive the results.
    """
    presented: list = []

    def presenter(prioritised_list):
        presented.extend(prioritised_list)

    extractor = _make_extractor({0: ["circuits"], 1: ["circuits"]})
    await run_question_paper_analysis(
        papers=["paper1", "paper2"],
        topic_extractor=extractor,
        presenter=presenter,
    )
    # The presenter must have been called and received the list
    assert len(presented) >= 1
    topics_presented = [t for t, _, _ in presented]
    assert "circuits" in topics_presented


# ---------------------------------------------------------------------------
# Test 18.6 — partial processing: continue over processed papers, report failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partial_processing_continues_over_successful_papers():
    """
    Req 18.6: When one paper fails extraction, the analysis completes over
    the remaining papers that succeeded.
    """
    # Paper 0 succeeds; paper 1 fails; paper 2 succeeds.
    # topics_by_paper does NOT include index 1 → extractor raises RuntimeError
    extractor = _make_extractor(
        {
            0: ["organic_chemistry", "reactions"],
            # index 1 is absent → will raise RuntimeError
            2: ["organic_chemistry", "thermodynamics"],
        }
    )
    ctx = await run_question_paper_analysis(
        papers=["paper1", "paper2_will_fail", "paper3"],
        topic_extractor=extractor,
    )
    # Paper 1 should be recorded as failed
    assert 1 in ctx.failed_papers

    # Papers 0 and 2 should still have topics
    assert 0 in ctx.per_paper_topics
    assert 2 in ctx.per_paper_topics

    # "organic_chemistry" appears in papers 0 and 2 → recurring
    assert ctx.recurring_topics.get("organic_chemistry", 0) >= 2

    # The automation must still complete (present_results ran)
    assert ctx.result_presented is True


@pytest.mark.asyncio
async def test_failed_papers_are_reported_in_results():
    """
    Req 18.6: The final output includes which papers failed and why.
    """
    extractor = _make_extractor({0: ["math"]})   # paper 1 will fail
    ctx = await run_question_paper_analysis(
        papers=["paper1", "paper2_fail"],
        topic_extractor=extractor,
    )
    assert 1 in ctx.failed_papers
    assert ctx.failed_papers[1]  # non-empty reason string


@pytest.mark.asyncio
async def test_all_papers_fail_still_presents_empty_list():
    """
    Req 18.6: Even when all per-paper extractions fail, present_results
    should still run (with an empty recurring set) and mark the automation
    complete.
    """
    # No paper indices in the map → all will fail
    extractor = _make_extractor({})
    ctx = await run_question_paper_analysis(
        papers=["paper1", "paper2"],
        topic_extractor=extractor,
    )
    assert ctx.result_presented is True
    assert ctx.prioritised_list == []
    assert len(ctx.failed_papers) == 2


# ---------------------------------------------------------------------------
# Test: register_builtin_automation wires into AutomationLibrary
# ---------------------------------------------------------------------------


def test_register_builtin_automation_stores_in_library():
    """
    Task 34.1: The automation is registered in the AutomationLibrary under
    the correct name so it can be retrieved by exact name.
    """
    library = AutomationLibrary()
    register_builtin_automation(library)
    automation = library.get(AUTOMATION_NAME)
    assert automation.name == AUTOMATION_NAME
    assert len(automation.steps) > 0


def test_registered_automation_has_all_phase_steps():
    """
    Task 34.1: The registered automation must contain representative steps
    for all analysis phases.
    """
    library = AutomationLibrary()
    register_builtin_automation(library)
    automation = library.get(AUTOMATION_NAME)
    step_ids = {s.id for s in automation.steps}
    required_ids = {
        "validate_input",
        "extract_topics",
        "identify_recurring",
        "cross_reference",
        "present_results",
    }
    for required in required_ids:
        assert required in step_ids, (
            f"Expected step '{required}' to be in automation steps; "
            f"found: {step_ids}"
        )


# ---------------------------------------------------------------------------
# Test: AnalysisContext initialisation
# ---------------------------------------------------------------------------


def test_analysis_context_defaults():
    """
    AnalysisContext initialises with empty collections so the automation
    can run safely even without prior state.
    """
    ctx = AnalysisContext(papers=["paper1"])
    assert ctx.per_paper_topics == {}
    assert ctx.failed_papers == {}
    assert ctx.recurring_topics == Counter()
    assert ctx.annotated_topics == {}
    assert ctx.result_presented is False
    assert ctx.prioritised_list == []


# ---------------------------------------------------------------------------
# Test: TopicExtractor heuristic path
# ---------------------------------------------------------------------------


def test_heuristic_extractor_returns_non_empty_for_non_trivial_text():
    """
    The default heuristic TopicExtractor returns at least one topic for
    a non-trivial examination paper text.
    """
    extractor = TopicExtractor()
    paper = (
        "Q1. Describe the OSI model and explain each layer. [10 marks]\n"
        "Q2. What is the difference between TCP and UDP? [5 marks]\n"
        "Q3. Explain network routing algorithms. [15 marks]"
    )
    topics = extractor.extract(paper)
    assert isinstance(topics, list)
    assert len(topics) > 0


def test_heuristic_extractor_deduplicates_topics():
    """
    The heuristic extractor must not return duplicate topic strings.
    """
    extractor = TopicExtractor()
    paper = "algorithms algorithms algorithms sorting sorting"
    topics = extractor.extract(paper)
    assert len(topics) == len(set(topics)), "Duplicate topics returned"
