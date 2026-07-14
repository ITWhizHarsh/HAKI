"""
Unit tests for MemoryNoteWriter (core/memory/memory_note_writer.py).

Sanity-verifies write_fast_pass_note() and write_heavy_pass_note() against a
temp vault directory: successful writes land in wiki/, no .tmp file is left
behind, a missing provenance target skips the write and returns None, and the
evolutionary link is included only when the target note resolves.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.memory.fast_pass import Entity
from core.memory.memory_note_writer import MemoryNoteWriter, validate_wiki_link


@pytest.fixture
def vault(tmp_path):
    """A vault with a source file already present so provenance resolves."""
    v = tmp_path / "vault"
    (v / "wiki").mkdir(parents=True)
    (v / "raw").mkdir()
    source = v / "raw" / "note.md"
    source.write_text("Harsh Kumar works at Acme Corp.")
    return v


def test_write_fast_pass_note_success(vault):
    writer = MemoryNoteWriter(vault_root=vault)
    entity = Entity(text="Harsh Kumar", label="PERSON", start=0, end=11)

    result = writer.write_fast_pass_note(
        entity=entity,
        source_filename="note.md",
        source_vault_rel="raw/note.md",
        run_date="2025-01-01",
    )

    assert result is not None
    assert result.path.exists()
    assert result.path.parent == vault / "wiki"
    assert result.wiki_link == f"[[{result.title}]]"

    # No leftover .tmp files in wiki/
    tmp_files = list((vault / "wiki").glob("*.tmp*"))
    assert tmp_files == []

    content = result.path.read_text()
    assert "[[raw/note.md]]" in content
    assert "Harsh Kumar" not in content or "PERSON" in content  # entity type rendered


def test_write_fast_pass_note_missing_provenance_returns_none(vault):
    writer = MemoryNoteWriter(vault_root=vault)
    entity = Entity(text="Harsh Kumar", label="PERSON", start=0, end=11)

    result = writer.write_fast_pass_note(
        entity=entity,
        source_filename="missing.md",
        source_vault_rel="raw/missing.md",
        run_date="2025-01-01",
    )

    assert result is None
    # Nothing should have been written to wiki/
    assert list((vault / "wiki").glob("*.md")) == []


def test_write_heavy_pass_note_success_no_evolution(vault):
    writer = MemoryNoteWriter(vault_root=vault)
    heavy_result = SimpleNamespace(
        memory_content="Harsh works at Acme Corp as an engineer.",
        old_memory_note_name=None,
    )

    result = writer.write_heavy_pass_note(
        result=heavy_result,
        source_filename="note.md",
        source_vault_rel="raw/note.md",
        run_date="2025-01-01",
    )

    assert result is not None
    assert result.path.exists()
    content = result.path.read_text()
    assert "Evolved from" not in content
    assert list((vault / "wiki").glob("*.tmp*")) == []


def test_write_heavy_pass_note_missing_provenance_returns_none(vault):
    writer = MemoryNoteWriter(vault_root=vault)
    heavy_result = SimpleNamespace(
        memory_content="Some synthesized content.",
        old_memory_note_name=None,
    )

    result = writer.write_heavy_pass_note(
        result=heavy_result,
        source_filename="missing.md",
        source_vault_rel="raw/missing.md",
        run_date="2025-01-01",
    )

    assert result is None
    assert list((vault / "wiki").glob("*.md")) == []


def test_write_heavy_pass_note_includes_evolutionary_link_when_target_resolves(vault):
    # Pre-create the "old" note that the evolutionary link should point to.
    old_note = vault / "wiki" / "old-concept.md"
    old_note.write_text("---\ntitle: old-concept\n---\nOld content.")

    writer = MemoryNoteWriter(vault_root=vault)
    heavy_result = SimpleNamespace(
        memory_content="Updated synthesized content.",
        old_memory_note_name="old-concept",
    )

    result = writer.write_heavy_pass_note(
        result=heavy_result,
        source_filename="note.md",
        source_vault_rel="raw/note.md",
        run_date="2025-01-01",
    )

    assert result is not None
    content = result.path.read_text()
    assert "[[old-concept]]" in content
    assert "Evolved from" in content


def test_write_heavy_pass_note_omits_evolutionary_link_when_target_missing(vault):
    writer = MemoryNoteWriter(vault_root=vault)
    heavy_result = SimpleNamespace(
        memory_content="Updated synthesized content.",
        old_memory_note_name="does-not-exist",
    )

    result = writer.write_heavy_pass_note(
        result=heavy_result,
        source_filename="note.md",
        source_vault_rel="raw/note.md",
        run_date="2025-01-01",
    )

    assert result is not None
    content = result.path.read_text()
    assert "does-not-exist" not in content
    assert "Evolved from" not in content


def test_validate_wiki_link_true_and_false_cases(vault):
    (vault / "wiki" / "existing-note.md").write_text("content")

    assert validate_wiki_link("existing-note", vault) is True
    assert validate_wiki_link("nonexistent-note", vault) is False
