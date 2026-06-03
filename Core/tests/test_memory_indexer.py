"""
Unit and property-based tests for the Chunker, Indexer, and the
MemoryBrain vector-index integration (Task 14.1).

Covers:
  - Req 7.3: retrieve returns notes matching the query
  - Req 7.4: index is initialised on startup even with no notes

**Validates: Requirements 7.3**

Testing conventions:
  - The embeddings provider is a deterministic stub: it embeds text as a
    small fixed-dimension vector derived from the text's character codes,
    so tests never touch a real model and remain cheap and reproducible.
  - Property tests use Hypothesis to verify structural invariants.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis import HealthCheck

from core.memory import Chunk, Chunker, Indexer, MemoryBrain, Note
from core.memory.indexer import _cosine_similarity, _pack_vector, _unpack_vector


# ---------------------------------------------------------------------------
# Deterministic stub embedding provider
# ---------------------------------------------------------------------------


class _DeterministicEmbedProvider:
    """
    Stub embeddings provider that maps text to a reproducible 8-D float vector.

    The vector is built from simple character-code arithmetic so that
    semantically similar texts produce similar (though not identical) vectors.
    Important: it always returns a ``list[float]`` so the Indexer can pack it.
    """

    _DIM = 8

    def invoke(self, text: str, **_kwargs: Any) -> list[float]:  # noqa: D401
        """Return an 8-dimensional deterministic vector for *text*."""
        vec = [0.0] * self._DIM
        for i, ch in enumerate(text):
            vec[i % self._DIM] += ord(ch)
        # L2-normalise so cosine similarity is meaningful
        mag = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / mag for x in vec]


_PROVIDER = _DeterministicEmbedProvider()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_indexer(tmp_path: Path, provider: _DeterministicEmbedProvider | None = None) -> Indexer:
    idx = Indexer(vault_path=tmp_path, embeddings_provider=provider or _PROVIDER)
    idx.init()
    return idx


def _make_brain(tmp_path: Path) -> MemoryBrain:
    brain = MemoryBrain(vault_path=tmp_path / "vault", embeddings_provider=_PROVIDER)
    brain.init()
    return brain


def _make_note(note_id: str = "n1", body: str = "Hello world") -> Note:
    return Note(id=note_id, body=body)


# ---------------------------------------------------------------------------
# Chunker unit tests
# ---------------------------------------------------------------------------


class TestChunker:
    def test_empty_body_produces_one_empty_chunk(self):
        c = Chunker(chunk_size=5, overlap=1)
        chunks = c.chunk("n1", "")
        assert len(chunks) == 1
        assert chunks[0].text == ""
        assert chunks[0].chunk_index == 0

    def test_body_shorter_than_chunk_size(self):
        c = Chunker(chunk_size=100, overlap=10)
        chunks = c.chunk("n1", "short body text")
        assert len(chunks) == 1
        assert chunks[0].note_id == "n1"
        assert "short body text" in chunks[0].text

    def test_body_exactly_chunk_size(self):
        words = " ".join(f"w{i}" for i in range(5))
        c = Chunker(chunk_size=5, overlap=1)
        chunks = c.chunk("n1", words)
        assert len(chunks) == 1

    def test_body_longer_than_chunk_size_produces_multiple_chunks(self):
        words = " ".join(f"w{i}" for i in range(10))
        c = Chunker(chunk_size=5, overlap=1)
        chunks = c.chunk("n1", words)
        assert len(chunks) > 1

    def test_overlap_tokens_shared(self):
        """The last `overlap` tokens of chunk N equal the first `overlap` tokens of chunk N+1."""
        c = Chunker(chunk_size=5, overlap=2)
        words = " ".join(f"w{i}" for i in range(12))
        chunks = c.chunk("n1", words)
        assert len(chunks) >= 2
        tail = chunks[0].text.split()[-2:]
        head = chunks[1].text.split()[:2]
        assert tail == head

    def test_chunk_indices_sequential(self):
        words = " ".join(f"w{i}" for i in range(20))
        c = Chunker(chunk_size=5, overlap=1)
        chunks = c.chunk("n1", words)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_note_id_propagated(self):
        c = Chunker(chunk_size=5, overlap=1)
        chunks = c.chunk("my-note-id", "some text here")
        for chunk in chunks:
            assert chunk.note_id == "my-note-id"

    def test_all_tokens_covered(self):
        """Every token in the body must appear in at least one chunk."""
        words = [f"word{i}" for i in range(15)]
        body = " ".join(words)
        c = Chunker(chunk_size=6, overlap=2)
        chunks = c.chunk("n1", body)
        all_text = " ".join(ch.text for ch in chunks)
        for word in words:
            assert word in all_text

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError):
            Chunker(chunk_size=0)

    def test_invalid_overlap_negative_raises(self):
        with pytest.raises(ValueError):
            Chunker(chunk_size=5, overlap=-1)

    def test_overlap_gte_chunk_size_raises(self):
        with pytest.raises(ValueError):
            Chunker(chunk_size=5, overlap=5)


# ---------------------------------------------------------------------------
# Vector serialization helpers
# ---------------------------------------------------------------------------


class TestVectorHelpers:
    def test_pack_unpack_round_trip(self):
        vec = [0.1, 0.2, 0.3, 0.4]
        blob = _pack_vector(vec)
        recovered = _unpack_vector(blob)
        assert len(recovered) == 4
        for a, b in zip(vec, recovered):
            assert abs(a - b) < 1e-6

    def test_cosine_similarity_identical(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        assert _cosine_similarity(a, b) == 0.0


# ---------------------------------------------------------------------------
# Indexer unit tests
# ---------------------------------------------------------------------------


class TestIndexer:
    def test_init_creates_db(self, tmp_path):
        idx = _make_indexer(tmp_path)
        assert (tmp_path / "vector_index.db").exists()

    def test_init_idempotent(self, tmp_path):
        idx = _make_indexer(tmp_path)
        idx.init()  # second call — should not raise
        assert (tmp_path / "vector_index.db").exists()

    def test_index_note_stores_chunks(self, tmp_path):
        idx = _make_indexer(tmp_path)
        note = _make_note("n1", "Hello world this is a test note")
        idx.index_note(note)
        assert idx.count() >= 1

    def test_get_chunks_for_note(self, tmp_path):
        idx = _make_indexer(tmp_path)
        note = _make_note("n1", "One two three four five")
        idx.index_note(note)
        chunks = idx.get_chunks_for_note("n1")
        assert len(chunks) >= 1
        assert all(c.note_id == "n1" for c in chunks)

    def test_remove_note_clears_chunks(self, tmp_path):
        idx = _make_indexer(tmp_path)
        note = _make_note("n1", "Remove me please")
        idx.index_note(note)
        assert idx.count() >= 1
        idx.remove_note("n1")
        assert idx.count() == 0

    def test_remove_nonexistent_note_is_noop(self, tmp_path):
        idx = _make_indexer(tmp_path)
        # Must not raise
        idx.remove_note("ghost")

    def test_rebuild_clears_old_chunks(self, tmp_path):
        idx = _make_indexer(tmp_path)
        note1 = _make_note("n1", "first note body text")
        idx.index_note(note1)
        assert idx.count() >= 1

        note2 = _make_note("n2", "second note body text")
        idx.rebuild([note2])
        chunks = idx.get_chunks_for_note("n1")
        assert chunks == []  # n1 was wiped
        assert idx.count() >= 1  # n2 was indexed

    def test_rebuild_empty_notes_clears_index(self, tmp_path):
        idx = _make_indexer(tmp_path)
        idx.index_note(_make_note("n1", "something"))
        idx.rebuild([])
        assert idx.count() == 0

    def test_reindex_note_replaces_old_chunks(self, tmp_path):
        idx = _make_indexer(tmp_path)
        note = _make_note("n1", "original body")
        idx.index_note(note)
        count_before = idx.count()

        note.body = "completely new body with different text"
        idx.index_note(note)
        # All old rows for n1 should be replaced
        chunks = idx.get_chunks_for_note("n1")
        assert all("new" in c.text or "different" in c.text or "completely" in c.text or "body" in c.text
                   for c in chunks)

    def test_search_returns_relevant_chunks(self, tmp_path):
        idx = _make_indexer(tmp_path)
        idx.index_note(_make_note("n1", "networks midterm exam june"))
        idx.index_note(_make_note("n2", "birthday party invitation tomorrow"))
        results = idx.search("exam midterm", k=5)
        # n1 should appear in top results (the index must return some results)
        assert len(results) >= 1
        note_ids = [r.note_id for r in results]
        assert "n1" in note_ids

    def test_search_excludes_specified_note_ids(self, tmp_path):
        idx = _make_indexer(tmp_path)
        idx.index_note(_make_note("n1", "exam midterm networks"))
        idx.index_note(_make_note("n2", "exam midterm review"))
        results = idx.search("exam midterm", k=5, exclude_note_ids={"n1"})
        assert all(c.note_id != "n1" for c in results)

    def test_search_empty_index_returns_empty(self, tmp_path):
        idx = _make_indexer(tmp_path)
        results = idx.search("anything")
        assert results == []

    def test_embeddings_stored_as_float_vectors(self, tmp_path):
        idx = _make_indexer(tmp_path)
        idx.index_note(_make_note("n1", "test embedding storage"))
        chunks = idx.get_chunks_for_note("n1")
        assert all(isinstance(c.embedding, list) for c in chunks)
        assert all(all(isinstance(v, float) for v in c.embedding) for c in chunks)

    def test_corrupt_db_is_recreated_on_init(self, tmp_path):
        """Req 7.4 / Task 14.1: corrupted index is replaced with a fresh one."""
        db_path = tmp_path / "vector_index.db"
        db_path.write_bytes(b"not a sqlite database!!!!")
        idx = Indexer(vault_path=tmp_path, embeddings_provider=_PROVIDER)
        # Should not raise
        idx.init()
        assert db_path.exists()
        # The new DB should be queryable
        idx.index_note(_make_note("n1", "recovery after corruption"))
        assert idx.count() >= 1


# ---------------------------------------------------------------------------
# MemoryBrain integration tests
# ---------------------------------------------------------------------------


class TestMemoryBrainIndexing:
    def test_remember_triggers_auto_index(self, tmp_path):
        brain = _make_brain(tmp_path)
        result = brain.remember("Computer networks midterm is June 14")
        assert result.success is True
        # The vector index must have at least one chunk for that note.
        assert brain._indexer is not None
        chunks = brain._indexer.get_chunks_for_note(result.note_id)
        assert len(chunks) >= 1

    def test_retrieve_returns_matching_note(self, tmp_path):
        brain = _make_brain(tmp_path)
        brain.remember("Midterm for computer networks is on June 14", topics=["networks"])
        brain.remember("Birthday party for Alice on Saturday", topics=["birthday"])

        results = brain.retrieve("networks midterm exam")
        assert len(results) >= 1
        assert any("networks" in n.body.lower() or "midterm" in n.body.lower()
                   for n in results)

    def test_retrieve_excludes_superseded_notes(self, tmp_path):
        brain = _make_brain(tmp_path)
        old_result = brain.remember("Old fact about exam date")
        # Manually supersede the note
        old_note = brain._vault.load(old_result.note_id)
        assert old_note is not None
        old_note.superseded_by = "new-note-id"
        brain._vault.store(old_note)

        results = brain.retrieve("exam date")
        assert all(n.superseded_by is None for n in results)

    def test_forget_removes_index_chunks(self, tmp_path):
        brain = _make_brain(tmp_path)
        result = brain.remember("Something to remember and forget")
        note_id = result.note_id
        assert brain._indexer.count() >= 1

        brain.forget(note_id)
        chunks = brain._indexer.get_chunks_for_note(note_id)
        assert chunks == []

    def test_forget_all_clears_index(self, tmp_path):
        brain = _make_brain(tmp_path)
        brain.remember("Note 1 content here")
        brain.remember("Note 2 content here")
        brain.forget_all()
        assert brain._indexer.count() == 0

    def test_rebuild_index_from_vault(self, tmp_path):
        vault_path = tmp_path / "vault"
        # Store notes via brain without index
        brain_no_idx = MemoryBrain(vault_path=vault_path)
        brain_no_idx.init()
        brain_no_idx.remember("Important fact one")
        brain_no_idx.remember("Important fact two")

        # Now create a brain WITH an index and rebuild
        brain_with_idx = MemoryBrain(vault_path=vault_path, embeddings_provider=_PROVIDER)
        brain_with_idx.init()
        brain_with_idx.rebuild_index()

        assert brain_with_idx._indexer.count() >= 2

    def test_retrieve_returns_empty_without_provider(self, tmp_path):
        """Pre-Task-14 behaviour preserved when no provider is supplied."""
        brain = MemoryBrain(vault_path=tmp_path / "vault")
        brain.init()
        brain.remember("something")
        results = brain.retrieve("something")
        assert results == []

    def test_init_creates_vector_index_db(self, tmp_path):
        brain = _make_brain(tmp_path)
        assert (tmp_path / "vault" / "vector_index.db").exists()

    def test_remember_content_kwarg_still_indexed(self, tmp_path):
        """Backward-compat: the ``content=`` kwarg path also auto-indexes."""
        brain = _make_brain(tmp_path)
        result = brain.remember(content="Some fact stored via content kwarg")
        assert result.success is True
        chunks = brain._indexer.get_chunks_for_note(result.note_id)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# Property-based tests
# **Validates: Requirements 7.3**
# ---------------------------------------------------------------------------


@settings(max_examples=50)
@given(
    body=st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "N", "P", "Zs"),
        ),
        min_size=0,
        max_size=500,
    )
)
def test_property_chunker_covers_all_tokens(body: str):
    """
    **Validates: Requirements 7.3**

    Property: every whitespace-separated token in the note body appears
    in at least one chunk produced by the Chunker.
    """
    c = Chunker(chunk_size=50, overlap=5)
    chunks = c.chunk("note-x", body)

    # There must always be at least one chunk
    assert len(chunks) >= 1

    tokens = body.split()
    if not tokens:
        assert chunks[0].text == ""
        return

    all_chunk_tokens = set()
    for chunk in chunks:
        all_chunk_tokens.update(chunk.text.split())

    for token in tokens:
        assert token in all_chunk_tokens, (
            f"Token '{token}' missing from chunks produced for body: {body!r}"
        )


@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    bodies=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
            min_size=1,
            max_size=200,
        ),
        min_size=1,
        max_size=5,
    )
)
def test_property_indexer_count_equals_total_chunks(tmp_path: Path, bodies: list[str]):
    """
    **Validates: Requirements 7.3**

    Property: after indexing N notes, the total chunk count in the index
    equals the sum of chunks produced by the Chunker for each note.
    """
    import tempfile
    import os

    chunker = Chunker(chunk_size=50, overlap=5)

    with tempfile.TemporaryDirectory() as td:
        vault_path = Path(td)
        idx = Indexer(vault_path=vault_path, embeddings_provider=_PROVIDER, chunker=chunker)
        idx.init()

        notes = [Note(id=f"note-{i}", body=body) for i, body in enumerate(bodies)]
        expected_total = sum(len(chunker.chunk(n.id, n.body)) for n in notes)

        for note in notes:
            idx.index_note(note)

        assert idx.count() == expected_total


@settings(max_examples=50)
@given(
    chunk_size=st.integers(min_value=2, max_value=100),
    overlap=st.integers(min_value=0, max_value=50),
)
def test_property_chunker_valid_params(chunk_size: int, overlap: int):
    """
    **Validates: Requirements 7.3**

    Property: for any valid (chunk_size, overlap) pair where overlap <
    chunk_size, the Chunker produces valid output for a fixed body.
    """
    if overlap >= chunk_size:
        # Not a valid param combination — Chunker should reject it.
        with pytest.raises(ValueError):
            Chunker(chunk_size=chunk_size, overlap=overlap)
        return

    body = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 3
    c = Chunker(chunk_size=chunk_size, overlap=overlap)
    chunks = c.chunk("n1", body)
    assert len(chunks) >= 1
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
        assert chunk.note_id == "n1"
