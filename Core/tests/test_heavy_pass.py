"""
Unit tests for HeavyPassExtractor (Heavy Pass fallback extraction).

Feature: haki-brain-memory-processing-pipeline
Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 5.5
"""

from __future__ import annotations

import asyncio

import pytest

from core.memory.heavy_pass import (
    HeavyPassExtractor,
    HeavyPassResult,
    HeavyPassStatus,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeChromaCollection:
    """Stub ChromaDB collection with a scripted count()/query() response."""

    def __init__(self, count: int = 0, query_result: dict | None = None, raise_on_query: bool = False):
        self._count = count
        self._query_result = query_result or {}
        self._raise_on_query = raise_on_query

    def count(self) -> int:
        return self._count

    def query(self, **kwargs):
        if self._raise_on_query:
            raise RuntimeError("ChromaDB backend unavailable")
        return self._query_result


class FakeLLMRouter:
    """Stub LLMRouter — returns a scripted response, raises, or hangs."""

    def __init__(self, response: str = "", raise_exc: Exception | None = None, hang: bool = False):
        self._response = response
        self._raise_exc = raise_exc
        self._hang = hang
        self.last_kwargs: dict | None = None

    async def chat(self, user_message: str, system_prompt: str = "", *, prefer_local: bool = False, **kwargs) -> str:
        self.last_kwargs = {
            "user_message": user_message,
            "system_prompt": system_prompt,
            "prefer_local": prefer_local,
            **kwargs,
        }
        if self._hang:
            await asyncio.sleep(999)
        if self._raise_exc:
            raise self._raise_exc
        return self._response


def _query_result_for(title: str, content: str) -> dict:
    return {
        "ids": [["note-1"]],
        "metadatas": [[{"title": title}]],
        "documents": [[content]],
    }


# ---------------------------------------------------------------------------
# Fresh synthesis (no old note found)
# ---------------------------------------------------------------------------


async def test_fresh_synthesis_success_no_old_note():
    """Req 2.6 — empty ChromaDB collection produces a fresh Memory_Note with no EVOLVED_FROM."""
    chroma = FakeChromaCollection(count=0)
    llm = FakeLLMRouter(response="A concise fresh memory note about the source file.")
    extractor = HeavyPassExtractor(chroma_collection=chroma, llm_router=llm)

    result = await extractor.extract("some raw source content", "notes.txt")

    assert result.status == HeavyPassStatus.SUCCESS
    assert result.memory_content == "A concise fresh memory note about the source file."
    assert result.evolutionary_link is None
    assert result.old_memory_note_name is None
    # prefer_local=True must be used for the Heavy Pass (Req 2.7 / design).
    assert llm.last_kwargs["prefer_local"] is True


# ---------------------------------------------------------------------------
# Evolutionary synthesis (old note found)
# ---------------------------------------------------------------------------


async def test_evolutionary_success_parses_evolved_from():
    """Req 2.3, 2.4, 2.5 — old note found triggers evolutionary prompt and EVOLVED_FROM parsing."""
    chroma = FakeChromaCollection(
        count=1,
        query_result=_query_result_for("Old Note Title", "Old note body content."),
    )
    llm_response = "Merged memory note body.\nEVOLVED_FROM: Old Note Title\n"
    llm = FakeLLMRouter(response=llm_response)
    extractor = HeavyPassExtractor(chroma_collection=chroma, llm_router=llm)

    result = await extractor.extract("new source content", "update.txt")

    assert result.status == HeavyPassStatus.SUCCESS
    assert result.evolutionary_link == "Old Note Title"
    assert result.old_memory_note_name == "Old Note Title"
    assert "EVOLVED_FROM" not in result.memory_content
    assert "Merged memory note body." in result.memory_content


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


async def test_extract_times_out():
    """Req 2.7 — LLM call exceeding the configured timeout returns HeavyPassStatus.TIMEOUT."""
    chroma = FakeChromaCollection(count=0)
    llm = FakeLLMRouter(hang=True)
    extractor = HeavyPassExtractor(chroma_collection=chroma, llm_router=llm, timeout_secs=0.05)

    result = await extractor.extract("content", "slow.txt")

    assert result.status == HeavyPassStatus.TIMEOUT
    assert result.error_msg is not None


# ---------------------------------------------------------------------------
# Empty LLM response
# ---------------------------------------------------------------------------


async def test_extract_empty_llm_response_is_llm_error():
    """Req 2.8 — an empty/whitespace-only LLM response yields HeavyPassStatus.LLM_ERROR."""
    chroma = FakeChromaCollection(count=0)
    llm = FakeLLMRouter(response="   \n  ")
    extractor = HeavyPassExtractor(chroma_collection=chroma, llm_router=llm)

    result = await extractor.extract("content", "empty.txt")

    assert result.status == HeavyPassStatus.LLM_ERROR
    assert result.error_msg is not None


# ---------------------------------------------------------------------------
# ChromaDB query exception
# ---------------------------------------------------------------------------


async def test_extract_chroma_query_exception_falls_back_to_fresh_synthesis():
    """Req 5.5 — a ChromaDB query exception is swallowed and treated as no old note found."""
    chroma = FakeChromaCollection(count=5, raise_on_query=True)
    llm = FakeLLMRouter(response="Fresh note despite Chroma failure.")
    extractor = HeavyPassExtractor(chroma_collection=chroma, llm_router=llm)

    result = await extractor.extract("content", "broken_chroma.txt")

    assert result.status == HeavyPassStatus.SUCCESS
    assert result.evolutionary_link is None
    assert result.old_memory_note_name is None


async def test_extract_unexpected_exception_returns_error_status():
    """A non-timeout, non-empty-response exception surfaces as HeavyPassStatus.ERROR."""
    chroma = FakeChromaCollection(count=0)
    llm = FakeLLMRouter(raise_exc=RuntimeError("LLM backend crashed"))
    extractor = HeavyPassExtractor(chroma_collection=chroma, llm_router=llm)

    result = await extractor.extract("content", "crash.txt")

    assert result.status == HeavyPassStatus.ERROR
    assert "LLM backend crashed" in result.error_msg


# ---------------------------------------------------------------------------
# No chroma_collection injected
# ---------------------------------------------------------------------------


async def test_extract_without_chroma_collection_uses_fresh_synthesis():
    """No chroma_collection injected (e.g. before the Vault has embeddings) → fresh synthesis path."""
    llm = FakeLLMRouter(response="Fresh note, no chroma at all.")
    extractor = HeavyPassExtractor(chroma_collection=None, llm_router=llm)

    result = await extractor.extract("content", "no_chroma.txt")

    assert result.status == HeavyPassStatus.SUCCESS
    assert result.old_memory_note_name is None
