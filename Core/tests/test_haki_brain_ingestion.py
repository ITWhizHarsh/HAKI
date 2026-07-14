"""
Sanity tests for HAKIBrain's ingestion pipeline (_ingest_file / ingest_pending).

Feature: haki-brain-memory-processing-pipeline
Task: 11 (checkpoint)

Covers:
  - Fast Pass success: extractable entity -> note in wiki/, file moved to processed/.
  - Heavy Pass fallback: no Fast Pass entities, stub LLM succeeds -> note + move.
  - Both-pass failure: Heavy Pass fails -> file stays in raw/.
  - Low-memory guard: Heavy Pass skipped when memory guard trips -> file stays in raw/.
  - Collision-safe processed/ destination naming.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.memory.haki_brain import HAKIBrain
from core.memory.heavy_pass import HeavyPassResult, HeavyPassStatus


class FakeLLMRouter:
    """Stub LLMRouter returning a scripted response or raising an error status."""

    def __init__(self, response: str | None = "Some synthesized memory content."):
        self._response = response

    async def chat(self, user_message: str, system_prompt: str = "", *, prefer_local: bool = False, **kwargs) -> str:
        if self._response is None:
            return ""
        return self._response


@pytest.fixture
def vault(tmp_path: Path, monkeypatch) -> Path:
    v = tmp_path / "HAKI_Brain"
    for folder in ("raw", "processed", "wiki"):
        (v / folder).mkdir(parents=True, exist_ok=True)
    # HAKIBrain.init() validates using the HAKI_OBSIDIAN_VAULT env var
    monkeypatch.setenv("HAKI_OBSIDIAN_VAULT", str(v))
    return v


def make_brain(vault: Path, llm_router=None) -> HAKIBrain:
    brain = HAKIBrain(obsidian_vault_path=vault, llm_router=llm_router)
    brain.init()
    return brain


@pytest.mark.asyncio
async def test_fast_pass_success_writes_note_and_moves_file(vault: Path):
    raw_file = vault / "raw" / "note.txt"
    raw_file.write_text("Contact me at harsh@example.com for details.", encoding="utf-8")

    brain = make_brain(vault, llm_router=None)
    results = await brain.ingest_pending()

    assert len(results) == 1
    result = results[0]
    assert result.success is True
    assert result.pass_used == "fast"
    assert result.wiki_page_path is not None

    # Source file moved out of raw/, into processed/.
    assert not raw_file.exists()
    processed_files = list((vault / "processed").glob("*"))
    assert len(processed_files) == 1

    # A Memory_Note was written to wiki/.
    wiki_files = list((vault / "wiki").glob("*.md"))
    assert len(wiki_files) == 1


@pytest.mark.asyncio
async def test_heavy_pass_fallback_writes_note_and_moves_file(vault: Path):
    # Content with no spaCy-detectable entities, emails, urls, phones, or markers.
    raw_file = vault / "raw" / "plain.txt"
    raw_file.write_text("just some plain lowercase words with nothing special", encoding="utf-8")

    llm_router = FakeLLMRouter(response="A concise summarized memory note.")
    brain = make_brain(vault, llm_router=llm_router)
    results = await brain.ingest_pending()

    assert len(results) == 1
    result = results[0]
    assert result.success is True
    assert result.pass_used == "heavy"
    assert result.wiki_page_path is not None

    assert not raw_file.exists()
    processed_files = list((vault / "processed").glob("*"))
    assert len(processed_files) == 1

    wiki_files = list((vault / "wiki").glob("*.md"))
    assert len(wiki_files) == 1


@pytest.mark.asyncio
async def test_both_pass_failure_leaves_file_in_raw(vault: Path):
    raw_file = vault / "raw" / "plain2.txt"
    raw_file.write_text("just some plain lowercase words with nothing special", encoding="utf-8")

    # llm_router=None -> HeavyPassExtractor.extract() will error out (no router).
    brain = make_brain(vault, llm_router=None)
    results = await brain.ingest_pending()

    assert len(results) == 1
    result = results[0]
    assert result.success is False

    # File remains untouched in raw/.
    assert raw_file.exists()
    assert list((vault / "processed").glob("*")) == []
    assert list((vault / "wiki").glob("*.md")) == []


@pytest.mark.asyncio
async def test_low_memory_guard_skips_heavy_pass(vault: Path, monkeypatch):
    raw_file = vault / "raw" / "plain3.txt"
    raw_file.write_text("just some plain lowercase words with nothing special", encoding="utf-8")

    llm_router = FakeLLMRouter(response="Should never be reached.")
    brain = make_brain(vault, llm_router=llm_router)
    monkeypatch.setattr(brain, "_check_low_memory", lambda: True)

    results = await brain.ingest_pending()

    assert len(results) == 1
    result = results[0]
    assert result.success is False
    assert "low memory" in (result.error or "").lower()

    # File remains untouched in raw/, Heavy Pass never invoked.
    assert raw_file.exists()
    assert list((vault / "processed").glob("*")) == []
    assert list((vault / "wiki").glob("*.md")) == []


def test_get_processed_dest_is_collision_safe(vault: Path):
    brain = make_brain(vault, llm_router=None)

    filename = "duplicate.txt"
    existing = vault / "processed" / filename
    existing.write_text("already here", encoding="utf-8")

    dest = brain._get_processed_dest(filename)

    assert dest != existing
    assert dest.parent == vault / "processed"
    assert dest.name != filename
