"""
Unit and property-based tests for hybrid retrieval (Task 14.2).

Covers:
  - Req 7.3: only notes sharing ≥1 term/topic with the query are returned
  - Req 7.7: "what do you know about X" returns matching notes and excludes
             non-matching ones
  - Req 8.2 / 8.3: superseded notes are never returned

**Validates: Requirements 7.3, 7.7**

Testing conventions:
  - _DeterministicEmbedProvider mirrors the stub in test_memory_indexer.py.
  - Embeddings are constructed so that the term-match filter is the decisive
    gate (not vector similarity), letting us test the filtering logic cleanly.
  - Property tests use Hypothesis and run ≥100 iterations each.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from core.memory import MemoryBrain, Note
from core.memory.memory_brain import _extract_query_terms, _note_matches_terms


# ---------------------------------------------------------------------------
# Deterministic stub embedding provider (same pattern as test_memory_indexer)
# ---------------------------------------------------------------------------


class _DeterministicEmbedProvider:
    """
    Stub embeddings provider that maps text to a reproducible 8-D float vector.

    Uses character-code arithmetic so similar text produces similar vectors.
    Always returns ``list[float]`` for Indexer compatibility.
    """

    _DIM = 8

    def invoke(self, text: str, **_kwargs: Any) -> list[float]:  # noqa: D401
        vec = [0.0] * self._DIM
        for i, ch in enumerate(text):
            vec[i % self._DIM] += ord(ch)
        mag = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / mag for x in vec]


_PROVIDER = _DeterministicEmbedProvider()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_brain(tmp_path: Path) -> MemoryBrain:
    """Return an initialised MemoryBrain backed by *tmp_path*."""
    brain = MemoryBrain(vault_path=tmp_path / "vault", embeddings_provider=_PROVIDER)
    brain.init()
    return brain


def _brain_in_tempdir() -> tuple[MemoryBrain, tempfile.TemporaryDirectory]:
    """Return (brain, tmp_dir) for use inside property tests (avoids fixtures)."""
    td = tempfile.TemporaryDirectory()
    brain = MemoryBrain(vault_path=Path(td.name) / "vault", embeddings_provider=_PROVIDER)
    brain.init()
    return brain, td


# ---------------------------------------------------------------------------
# Unit tests for _extract_query_terms
# ---------------------------------------------------------------------------


class TestExtractQueryTerms:
    def test_strips_stop_words(self):
        terms = _extract_query_terms("what do you know about networks")
        assert "what" not in terms
        assert "do" not in terms
        assert "you" not in terms
        assert "networks" in terms

    def test_strips_short_tokens(self):
        terms = _extract_query_terms("is it a big deal")
        assert "is" not in terms
        assert "it" not in terms
        assert "a" not in terms
        assert "big" in terms
        assert "deal" in terms

    def test_lowercase_output(self):
        terms = _extract_query_terms("Computer Networks Midterm")
        assert "computer" in terms
        assert "networks" in terms
        assert "midterm" in terms

    def test_all_stop_words_returns_empty(self):
        terms = _extract_query_terms("what do you know about")
        assert terms == set()

    def test_empty_string_returns_empty(self):
        assert _extract_query_terms("") == set()

    def test_punctuation_stripped(self):
        terms = _extract_query_terms("exam, midterm! networks?")
        assert "exam" in terms
        assert "midterm" in terms
        assert "networks" in terms

    def test_numbers_kept_if_long_enough(self):
        terms = _extract_query_terms("deadline 2024")
        assert "2024" in terms

    def test_what_do_you_know_about_pattern(self):
        """The canonical 'what do you know about X' intent extracts just X."""
        terms = _extract_query_terms("what do you know about computer networks")
        assert "computer" in terms
        assert "networks" in terms
        # Stop words removed
        assert "what" not in terms
        assert "know" not in terms


# ---------------------------------------------------------------------------
# Unit tests for _note_matches_terms
# ---------------------------------------------------------------------------


class TestNoteMatchesTerms:
    def test_body_match(self):
        note = Note(body="Midterm for computer networks is on June 14")
        assert _note_matches_terms(note, {"networks"}) is True

    def test_topics_match(self):
        note = Note(body="Some unrelated body text", topics=["computer-networks"])
        assert _note_matches_terms(note, {"networks"}) is True

    def test_tags_match(self):
        note = Note(body="Some unrelated body text", tags=["exam"])
        assert _note_matches_terms(note, {"exam"}) is True

    def test_no_match(self):
        note = Note(body="Birthday party invitation for Alice", topics=["birthday"])
        assert _note_matches_terms(note, {"networks", "midterm", "exam"}) is False

    def test_case_insensitive_body_match(self):
        note = Note(body="NETWORKS exam MIDTERM upcoming")
        assert _note_matches_terms(note, {"networks"}) is True
        assert _note_matches_terms(note, {"midterm"}) is True

    def test_partial_substring_match(self):
        note = Note(body="computer-networking is hard")
        assert _note_matches_terms(note, {"network"}) is True

    def test_empty_terms_returns_false(self):
        note = Note(body="some content here")
        assert _note_matches_terms(note, set()) is False


# ---------------------------------------------------------------------------
# Unit tests: hybrid retrieval end-to-end
# ---------------------------------------------------------------------------


class TestHybridRetrieval:
    def test_returns_empty_for_empty_vault(self, tmp_path):
        """Empty vault must return []."""
        brain = _make_brain(tmp_path)
        results = brain.retrieve("networks midterm")
        assert results == []

    def test_matching_note_is_returned(self, tmp_path):
        """A note whose body matches query terms must appear in results."""
        brain = _make_brain(tmp_path)
        brain.remember("Midterm for computer networks is on June 14", topics=["networks"])
        brain.remember("Birthday party for Alice on Saturday", topics=["birthday"])

        results = brain.retrieve("networks midterm")
        assert len(results) >= 1
        bodies = [n.body for n in results]
        assert any("networks" in b.lower() or "midterm" in b.lower() for b in bodies)

    def test_non_matching_note_excluded_despite_vector_similarity(self, tmp_path):
        """
        A note with high vector similarity but NO shared terms must be excluded.

        We achieve this by storing two notes — one that shares the query term
        'networks' and one that does not — then checking the non-matching one
        is absent from results.
        """
        brain = _make_brain(tmp_path)
        r1 = brain.remember("The exam for computer networks is June 14", topics=["networks"])
        r2 = brain.remember("Birthday celebration party invitation happy day", topics=["birthday"])

        results = brain.retrieve("networks exam")
        result_ids = {n.id for n in results}

        assert r1.note_id in result_ids, "Matching note must be returned"
        assert r2.note_id not in result_ids, "Non-matching note must be excluded"

    def test_superseded_note_never_returned(self, tmp_path):
        """Superseded notes must always be excluded (Req 8.2, 8.3)."""
        brain = _make_brain(tmp_path)
        old_result = brain.remember("Old network fact: exam on June 1", topics=["networks"])
        new_result = brain.remember("Updated network fact: exam on June 14", topics=["networks"])

        # Manually mark the old note as superseded
        old_note = brain._vault.load(old_result.note_id)
        assert old_note is not None
        old_note.superseded_by = new_result.note_id
        brain._vault.store(old_note)
        brain._indexer.index_note(old_note)

        results = brain.retrieve("networks exam")
        result_ids = {n.id for n in results}
        assert old_result.note_id not in result_ids, "Superseded note must never be returned"

    def test_what_do_you_know_about_returns_matching_notes(self, tmp_path):
        """
        'What do you know about X' intent returns notes about X (Req 7.7).
        """
        brain = _make_brain(tmp_path)
        brain.remember("Computer networks midterm is June 14", topics=["networks"])
        brain.remember("Assignment deadline for algorithms is July 3", topics=["algorithms"])

        results = brain.retrieve("what do you know about networks")
        assert len(results) >= 1
        bodies = " ".join(n.body for n in results).lower()
        assert "networks" in bodies

    def test_what_do_you_know_about_excludes_non_matching(self, tmp_path):
        """
        'What do you know about X' must NOT return notes unrelated to X (Req 7.7).
        """
        brain = _make_brain(tmp_path)
        brain.remember("Computer networks exam coming up", topics=["networks"])
        r_unrelated = brain.remember("Birthday gift ideas for friend party", topics=["birthday"])

        results = brain.retrieve("what do you know about networks")
        result_ids = {n.id for n in results}
        assert r_unrelated.note_id not in result_ids

    def test_topics_field_used_for_term_matching(self, tmp_path):
        """Topics list on a note counts for term matching even if not in body."""
        brain = _make_brain(tmp_path)
        # Body doesn't mention 'networks' explicitly; it's only in topics
        r = brain.remember("Upcoming assessment on the 14th", topics=["networks", "midterm"])
        brain.remember("Totally unrelated content about cooking recipes", topics=["food"])

        results = brain.retrieve("networks midterm")
        result_ids = {n.id for n in results}
        assert r.note_id in result_ids

    def test_tags_field_used_for_term_matching(self, tmp_path):
        """Tags on a note count for term matching."""
        brain = _make_brain(tmp_path)
        r = brain.remember("Assessment coming soon", tags=["networks", "exam"])
        brain.remember("Grocery shopping list for the week", tags=["food"])

        results = brain.retrieve("networks exam")
        result_ids = {n.id for n in results}
        assert r.note_id in result_ids

    def test_result_count_bounded_by_k(self, tmp_path):
        """retrieve() returns at most k results."""
        brain = _make_brain(tmp_path)
        for i in range(10):
            brain.remember(f"Note {i} about networks and exam preparation", topics=["networks"])

        results = brain.retrieve("networks exam", k=3)
        assert len(results) <= 3

    def test_all_returned_notes_are_not_superseded(self, tmp_path):
        """Every returned note must have superseded_by == None."""
        brain = _make_brain(tmp_path)
        r1 = brain.remember("Old networks fact from last year", topics=["networks"])
        r2 = brain.remember("New networks fact this year", topics=["networks"])

        # Supersede r1 by r2
        old_note = brain._vault.load(r1.note_id)
        old_note.superseded_by = r2.note_id
        brain._vault.store(old_note)
        brain._indexer.index_note(old_note)

        results = brain.retrieve("networks")
        assert all(n.superseded_by is None for n in results)

    def test_retrieve_returns_empty_without_provider(self, tmp_path):
        """Pre-Task-14 behaviour: no provider → always returns []."""
        brain = MemoryBrain(vault_path=tmp_path / "vault")
        brain.init()
        brain.remember("Some content about networks")
        assert brain.retrieve("networks") == []


# ---------------------------------------------------------------------------
# Property-based tests
# **Validates: Requirements 7.3, 7.7**
# ---------------------------------------------------------------------------

# Feature: haki-personal-ai-assistant, Property 20: hybrid_retrieval_term_filter_exclusion
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    matching_body=st.text(
        alphabet=st.characters(whitelist_categories=("L", "Zs")),
        min_size=5,
        max_size=100,
    ),
    unrelated_body=st.text(
        alphabet=st.characters(whitelist_categories=("L", "Zs")),
        min_size=5,
        max_size=100,
    ),
)
def test_property_non_matching_notes_excluded(
    matching_body: str,
    unrelated_body: str,
) -> None:
    """
    **Validates: Requirements 7.3**

    # Feature: haki-personal-ai-assistant, Property 20: hybrid_retrieval_term_filter_exclusion

    Property: for any pair of notes where only one contains the query term,
    the non-matching note is excluded even if it is in the vector index.
    """
    # Use a fixed, unique search term that won't appear in random text
    search_term = "zzzqqqxxx"
    # Force the matching note to include the search term
    matching = matching_body.strip() or "placeholder"
    full_matching_body = f"{matching} {search_term}"
    unrelated = unrelated_body.strip() or "other"
    # Ensure unrelated body does not accidentally contain the search term
    if search_term in unrelated.lower():
        unrelated = "totally different content here about cooking"

    brain, td = _brain_in_tempdir()
    try:
        r_match = brain.remember(full_matching_body, topics=[search_term])
        r_unrelated = brain.remember(unrelated, topics=["birthday", "cooking"])

        results = brain.retrieve(search_term)
        result_ids = {n.id for n in results}

        # The matching note must be returned
        assert r_match.note_id in result_ids, (
            f"Expected matching note in results. Query: '{search_term}', "
            f"body: '{full_matching_body}'"
        )
        # The unrelated note must NOT be returned
        assert r_unrelated.note_id not in result_ids, (
            f"Non-matching note appeared in results. Query: '{search_term}', "
            f"unrelated body: '{unrelated}'"
        )
    finally:
        td.cleanup()


# Feature: haki-personal-ai-assistant, Property 21: superseded_notes_always_excluded
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    note_body=st.text(
        alphabet=st.characters(whitelist_categories=("L", "Zs")),
        min_size=5,
        max_size=100,
    ),
)
def test_property_superseded_notes_never_returned(note_body: str) -> None:
    """
    **Validates: Requirements 7.3, 7.7**

    # Feature: haki-personal-ai-assistant, Property 21: superseded_notes_always_excluded

    Property: a note marked with superseded_by is never returned by retrieve(),
    regardless of how well it matches the query terms.
    """
    search_term = "uniqueterm999"
    body = (note_body.strip() or "content") + f" {search_term}"

    brain, td = _brain_in_tempdir()
    try:
        r_old = brain.remember(body, topics=[search_term])
        r_new = brain.remember(f"Replacement note {search_term}", topics=[search_term])

        # Supersede the old note
        old_note = brain._vault.load(r_old.note_id)
        assert old_note is not None
        old_note.superseded_by = r_new.note_id
        brain._vault.store(old_note)
        brain._indexer.index_note(old_note)

        results = brain.retrieve(search_term)
        result_ids = {n.id for n in results}

        assert r_old.note_id not in result_ids, (
            "Superseded note must never appear in retrieve() results"
        )
        # The superseding note must be present (it still matches)
        assert r_new.note_id in result_ids, (
            "The replacement (non-superseded) note should be returned"
        )
    finally:
        td.cleanup()


# Feature: haki-personal-ai-assistant, Property 22: what_do_you_know_about_returns_matching
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    topic=st.from_regex(r"[a-z]{4,12}", fullmatch=True),
)
def test_property_what_do_you_know_about_returns_matching(topic: str) -> None:
    """
    **Validates: Requirements 7.7**

    # Feature: haki-personal-ai-assistant, Property 22: what_do_you_know_about_returns_matching

    Property: retrieve("what do you know about <topic>") returns notes whose
    content contains <topic> and does NOT return notes that have no match.
    """
    # Ensure topic is not itself a stop word or too short
    from core.memory.memory_brain import _STOP_WORDS
    if topic in _STOP_WORDS or len(topic) < 3:
        return

    brain, td = _brain_in_tempdir()
    try:
        r_match = brain.remember(f"Important facts about {topic} for the exam", topics=[topic])
        r_no_match = brain.remember("Grocery list: apples, bananas, milk, bread today", topics=["shopping"])

        query = f"what do you know about {topic}"
        results = brain.retrieve(query)
        result_ids = {n.id for n in results}

        assert r_match.note_id in result_ids, (
            f"Note about '{topic}' must be returned for query '{query}'"
        )
        assert r_no_match.note_id not in result_ids, (
            f"Unrelated grocery note must not appear for query '{query}'"
        )
    finally:
        td.cleanup()


# Feature: haki-personal-ai-assistant, Property 23: retrieve_result_count_bounded_by_k
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    k=st.integers(min_value=1, max_value=10),
    num_notes=st.integers(min_value=0, max_value=15),
)
def test_property_retrieve_count_bounded_by_k(k: int, num_notes: int) -> None:
    """
    **Validates: Requirements 7.3**

    # Feature: haki-personal-ai-assistant, Property 23: retrieve_result_count_bounded_by_k

    Property: retrieve(query, k=k) never returns more than k notes.
    """
    search_term = "boundedterm"
    brain, td = _brain_in_tempdir()
    try:
        for i in range(num_notes):
            brain.remember(f"Note {i} about {search_term} and exam prep", topics=[search_term])

        results = brain.retrieve(search_term, k=k)
        assert len(results) <= k, (
            f"retrieve(k={k}) returned {len(results)} results, expected ≤ {k}"
        )
    finally:
        td.cleanup()
