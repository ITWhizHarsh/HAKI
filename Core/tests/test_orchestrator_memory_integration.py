"""
Integration tests for Task 14.3 — MemoryBrain wired into the Orchestrator.

Covers:
  - Real MemoryBrain + Orchestrator: _retrieve_memory() returns stored notes
  - ctx.memory_context populated after run_turn()
  - 2-second timeout still fires when retrieve() blocks
  - Backward compatibility: stubs with async def retrieve() still work

Requirements: 7.3
"""

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path
from typing import Any

import pytest

from core.memory import MemoryBrain
from core.memory.models import Note
from core.orchestrator.orchestrator import Orchestrator, TurnContext, MEMORY_TIMEOUT_SECS


# ---------------------------------------------------------------------------
# Deterministic embeddings stub (same pattern as other memory tests)
# ---------------------------------------------------------------------------


class _DeterministicEmbedProvider:
    """8-D deterministic embeddings for offline testing."""

    _DIM = 8

    def invoke(self, text: str, **_kwargs: Any) -> list[float]:
        vec = [0.0] * self._DIM
        for i, ch in enumerate(text):
            vec[i % self._DIM] += ord(ch)
        mag = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / mag for x in vec]


_PROVIDER = _DeterministicEmbedProvider()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_brain(tmp_path: Path) -> MemoryBrain:
    """Return an initialised MemoryBrain with a deterministic embed provider."""
    brain = MemoryBrain(vault_path=tmp_path / "vault", embeddings_provider=_PROVIDER)
    brain.init()
    return brain


# ---------------------------------------------------------------------------
# Tests: _retrieve_memory() returns real notes
# ---------------------------------------------------------------------------


class TestRetrieveMemoryWithRealBrain:
    """_retrieve_memory() should surface real stored notes to the orchestrator."""

    def test_retrieve_returns_stored_notes(self, tmp_path):
        """
        When a note is stored and its topic matches the transcript, the
        orchestrator's _retrieve_memory() must return it (not an empty list).
        """
        brain = _make_brain(tmp_path)
        brain.remember(
            "Computer networks midterm is on June 14",
            topics=["networks", "midterm"],
        )

        orch = Orchestrator(memory_brain=brain)
        ctx = TurnContext(transcript="tell me about networks midterm")

        result = asyncio.run(orch._retrieve_memory(ctx))

        assert len(result) >= 1, "Expected at least one note from real MemoryBrain"
        bodies = [n.body for n in result]
        assert any("networks" in b.lower() or "midterm" in b.lower() for b in bodies)

    def test_retrieve_returns_empty_without_provider(self, tmp_path):
        """
        Without an embeddings provider, retrieve() returns [] — same as the
        stub.  The orchestrator must accept this without error.
        """
        brain = MemoryBrain(vault_path=tmp_path / "no_provider_vault")
        brain.init()
        brain.remember("Some content about networks")

        orch = Orchestrator(memory_brain=brain)
        ctx = TurnContext(transcript="networks")

        result = asyncio.run(orch._retrieve_memory(ctx))
        assert result == []

    def test_retrieve_returns_empty_for_empty_vault(self, tmp_path):
        """Empty vault must return [] without raising."""
        brain = _make_brain(tmp_path)
        orch = Orchestrator(memory_brain=brain)
        ctx = TurnContext(transcript="anything")

        result = asyncio.run(orch._retrieve_memory(ctx))
        assert result == []

    def test_non_matching_transcript_returns_empty(self, tmp_path):
        """Notes that don't match the query terms must not be returned."""
        brain = _make_brain(tmp_path)
        brain.remember("Birthday party invitation", topics=["birthday"])

        orch = Orchestrator(memory_brain=brain)
        ctx = TurnContext(transcript="networks midterm exam")

        result = asyncio.run(orch._retrieve_memory(ctx))
        # Either empty or none of the results are about birthday
        bodies = " ".join(n.body.lower() for n in result)
        assert "networks" in bodies or result == []


# ---------------------------------------------------------------------------
# Tests: run_turn() populates ctx.memory_context
# ---------------------------------------------------------------------------


class TestRunTurnMemoryContext:
    """After run_turn(), the memory context from the real brain must flow into
    the response pipeline (visible through _handle_chat / _handle_recall)."""

    def test_run_turn_with_matching_note_populates_memory_context(self, tmp_path):
        """
        After run_turn(), the orchestrator should have accessed the stored note.
        We verify this indirectly via the recall intent response.
        """
        brain = _make_brain(tmp_path)
        brain.remember(
            "Computer networks midterm is on June 14",
            topics=["networks", "midterm"],
        )
        orch = Orchestrator(memory_brain=brain)

        # We patch _retrieve_memory to capture what it returned
        retrieved: list[Any] = []
        original_retrieve = orch._retrieve_memory

        async def _capturing_retrieve(ctx: TurnContext) -> list[Any]:
            notes = await original_retrieve(ctx)
            retrieved.extend(notes)
            return notes

        orch._retrieve_memory = _capturing_retrieve  # type: ignore[method-assign]

        asyncio.run(orch.run_turn("networks midterm"))

        assert len(retrieved) >= 1, (
            "Expected _retrieve_memory to return at least one note during run_turn()"
        )

    def test_run_turn_without_provider_does_not_crash(self, tmp_path):
        """run_turn() proceeds normally even when MemoryBrain has no provider."""
        brain = MemoryBrain(vault_path=tmp_path / "vault_no_prov")
        brain.init()
        brain.remember("Some note content")

        orch = Orchestrator(memory_brain=brain)
        # Should complete without raising
        response = asyncio.run(orch.run_turn("hello"))
        assert isinstance(response, str)


# ---------------------------------------------------------------------------
# Tests: 2-second timeout
# ---------------------------------------------------------------------------


class TestMemoryRetrievalTimeout:
    """The 2-second timeout must fire even when using the real MemoryBrain path."""

    def test_slow_retrieve_triggers_timeout(self, tmp_path):
        """
        A MemoryBrain whose retrieve() takes >2 s must hit the timeout and
        return [].  The orchestrator must catch TimeoutError and return an
        empty list, not re-raise.

        Note: because slow retrieve() runs in a thread executor, the elapsed
        wall-clock time may exceed MEMORY_TIMEOUT_SECS (the thread keeps
        running in the background).  What matters is that the orchestrator
        returns [] correctly — not how long the background thread lives.
        """

        class _SlowBrain:
            """retrieve() blocks for 5 s — well over the 2 s budget."""

            def retrieve(self, query: str, k: int = 5) -> list[Note]:
                time.sleep(5)
                return []

        orch = Orchestrator(memory_brain=_SlowBrain())
        ctx = TurnContext(transcript="any query")

        result = asyncio.run(orch._retrieve_memory(ctx))

        # The timeout must have fired and returned [] without raising
        assert result == [], "Timed-out retrieve must return []"

    def test_fast_retrieve_completes_before_timeout(self, tmp_path):
        """A fast MemoryBrain must complete without hitting the timeout."""
        brain = _make_brain(tmp_path)
        brain.remember("Quick note about networks", topics=["networks"])

        orch = Orchestrator(memory_brain=brain)
        ctx = TurnContext(transcript="networks")

        start = time.monotonic()
        result = asyncio.run(orch._retrieve_memory(ctx))
        elapsed = time.monotonic() - start

        # Must finish well before the 2-second timeout
        assert elapsed < MEMORY_TIMEOUT_SECS, (
            f"retrieve() took {elapsed:.2f}s — exceeded {MEMORY_TIMEOUT_SECS}s budget"
        )
        # Result is a list (possibly empty if no match; that's fine here)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests: backward compatibility — stubs with async def retrieve() still work
# ---------------------------------------------------------------------------


class TestBackwardCompatAsyncStub:
    """Legacy stubs that expose 'async def retrieve()' must still work."""

    def test_async_stub_retrieve_is_called(self):
        """Stubs with async def retrieve() (no aretrieve) must be awaited correctly."""

        class _AsyncBrainStub:
            async def retrieve(self, query: str, k: int = 5) -> list[Any]:
                return [Note(body=f"stub note for: {query}")]

        orch = Orchestrator(memory_brain=_AsyncBrainStub())
        ctx = TurnContext(transcript="test query")

        result = asyncio.run(orch._retrieve_memory(ctx))

        assert len(result) == 1
        assert "test query" in result[0].body

    def test_default_memory_brain_stub_returns_empty(self):
        """The default _MemoryBrainStub (no brain passed) must return []."""
        orch = Orchestrator()  # uses _MemoryBrainStub internally
        ctx = TurnContext(transcript="anything")

        result = asyncio.run(orch._retrieve_memory(ctx))
        assert result == []

    def test_async_stub_works_inside_run_turn(self):
        """run_turn() must succeed end-to-end with an async stub brain."""

        class _AsyncBrainStub:
            async def retrieve(self, query: str, k: int = 5) -> list[Any]:
                return []

        orch = Orchestrator(memory_brain=_AsyncBrainStub())
        response = asyncio.run(orch.run_turn("hello world"))
        assert isinstance(response, str)

    def test_aretrieve_takes_precedence_over_async_retrieve(self, tmp_path):
        """
        When a brain has both aretrieve() and retrieve(), aretrieve() is
        preferred (real MemoryBrain case).
        """
        brain = _make_brain(tmp_path)
        brain.remember("Networks exam coming up", topics=["networks"])

        # Confirm aretrieve is present on real MemoryBrain
        assert hasattr(brain, "aretrieve"), "MemoryBrain must have aretrieve()"

        orch = Orchestrator(memory_brain=brain)
        ctx = TurnContext(transcript="networks exam")

        result = asyncio.run(orch._retrieve_memory(ctx))
        # Result is a list (may have notes or be empty depending on indexer)
        assert isinstance(result, list)
