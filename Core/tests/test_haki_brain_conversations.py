"""
Sanity tests for HAKIBrain's conversation processing pipeline
(process_pending_conversations / _process_conversation_log).

Feature: haki-brain-memory-processing-pipeline
Task: 13 (checkpoint)

Covers:
  - Unprocessed logs are picked up and processed in chronological order.
  - Today's log is excluded by the cutoff (Req 6.4).
  - Already-processed logs are skipped on a second run (Req 6.5, 6.8).
  - Conversation log files remain unchanged/unmoved after processing
    (Req 6.3, 6.7) — unlike raw/ ingestion.
  - A log stays unmarked (retried on next run) when both passes fail.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from core.memory.haki_brain import HAKIBrain


class FakeLLMRouter:
    """Stub LLMRouter returning a scripted response."""

    def __init__(self, response: str | None = "Some synthesized memory content."):
        self._response = response

    async def chat(self, user_message: str, system_prompt: str = "", *, prefer_local: bool = False, **kwargs) -> str:
        if self._response is None:
            return ""
        return self._response


@pytest.fixture
def vault(tmp_path: Path, monkeypatch) -> Path:
    v = tmp_path / "HAKI_Brain"
    for folder in ("raw", "processed", "wiki", "conversations"):
        (v / folder).mkdir(parents=True, exist_ok=True)
    # HAKIBrain.init() validates using the HAKI_OBSIDIAN_VAULT env var
    monkeypatch.setenv("HAKI_OBSIDIAN_VAULT", str(v))
    return v


def make_brain(vault: Path, llm_router=None) -> HAKIBrain:
    brain = HAKIBrain(obsidian_vault_path=vault, llm_router=llm_router)
    brain.init()
    return brain


def _write_log(vault: Path, day: date, content: str) -> Path:
    log_path = vault / "conversations" / f"{day.isoformat()}.md"
    log_path.write_text(content, encoding="utf-8")
    return log_path


@pytest.mark.asyncio
async def test_unprocessed_logs_picked_up_in_order(vault: Path):
    yesterday = date.today() - timedelta(days=1)
    two_days_ago = date.today() - timedelta(days=2)

    # Fast-Pass-extractable content (contains an email address).
    log_old = _write_log(vault, two_days_ago, "Contact me at harsh@example.com please.")
    log_recent = _write_log(vault, yesterday, "Reach out at someone@example.com too.")

    brain = make_brain(vault, llm_router=None)
    results = await brain.process_pending_conversations()

    assert len(results) == 2
    # Chronological oldest-first.
    assert results[0].source_file == log_old.name
    assert results[1].source_file == log_recent.name
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_todays_log_excluded_by_cutoff(vault: Path):
    today = date.today()
    _write_log(vault, today, "Contact me at harsh@example.com for details.")

    brain = make_brain(vault, llm_router=None)
    results = await brain.process_pending_conversations()

    assert results == []


@pytest.mark.asyncio
async def test_already_processed_logs_skipped_on_second_run(vault: Path):
    yesterday = date.today() - timedelta(days=1)
    _write_log(vault, yesterday, "Contact me at harsh@example.com for details.")

    brain = make_brain(vault, llm_router=None)

    first_results = await brain.process_pending_conversations()
    assert len(first_results) == 1
    assert first_results[0].success is True

    second_results = await brain.process_pending_conversations()
    assert second_results == []


@pytest.mark.asyncio
async def test_conversation_files_remain_unmoved_after_processing(vault: Path):
    yesterday = date.today() - timedelta(days=1)
    log_path = _write_log(vault, yesterday, "Contact me at harsh@example.com for details.")
    original_content = log_path.read_text(encoding="utf-8")

    brain = make_brain(vault, llm_router=None)
    results = await brain.process_pending_conversations()

    assert len(results) == 1
    assert results[0].success is True

    # File still exists in conversations/, unmoved and unmodified.
    assert log_path.exists()
    assert log_path.read_text(encoding="utf-8") == original_content
    # Never moved to processed/.
    assert list((vault / "processed").glob("*")) == []


@pytest.mark.asyncio
async def test_log_stays_unmarked_when_both_passes_fail(vault: Path):
    yesterday = date.today() - timedelta(days=1)
    # Plain content with no Fast-Pass-detectable entities.
    log_path = _write_log(
        vault, yesterday, "just some plain lowercase words with nothing special"
    )

    # llm_router=None -> HeavyPassExtractor.extract() will error out (no router).
    brain = make_brain(vault, llm_router=None)
    results = await brain.process_pending_conversations()

    assert len(results) == 1
    assert results[0].success is False

    # File remains untouched.
    assert log_path.exists()

    # Not marked processed -> retried on next run.
    assert brain._process_tracker.is_processed(log_path.name) is False
    second_results = await brain.process_pending_conversations()
    assert len(second_results) == 1
    assert second_results[0].source_file == log_path.name
