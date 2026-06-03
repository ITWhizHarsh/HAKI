"""
Unit tests for vault init, durable writes, atomicity, and MemoryBrain.

Covers:
  - Req 7.1: StoreResult confirmed only after durable write
  - Req 7.2: no partial note left on write failure
  - Req 7.4: vault + empty index exist after init, even with no notes
  - Req 7.5: notes persist (roundtrip via file)
  - Vault.note_path, delete, list_all
  - Note.to_markdown / Note.from_markdown round trip
  - MemoryBrain.remember, forget, forget_all
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from core.memory import MemoryBrain, Note, NoteSource, StoreResult, Vault
from core.memory.models import Chunk
from core.memory.serializer import NoteSerializer, NoteSerializationError

_ser = NoteSerializer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vault(tmp_path: Path) -> Vault:
    v = Vault(tmp_path / "vault")
    v.init()
    return v


def _minimal_note(**kwargs) -> Note:
    defaults = dict(
        id="test-note-id",
        body="Hello, HAKI!",
        tags=["tag1"],
        topics=["topic-a"],
    )
    defaults.update(kwargs)
    return Note(**defaults)


# ---------------------------------------------------------------------------
# Note serialisation round-trip (via NoteSerializer from task 13.1)
# ---------------------------------------------------------------------------

class TestNoteRoundTrip:
    def test_basic_round_trip(self):
        note = Note(
            id="2024-06-01T12-03-22-a1b2",
            body="Midterm is June 14.",
            tags=["exam", "networks"],
            topics=["computer-networks"],
            source=NoteSource.USER_STATED,
            private=False,
            superseded_by=None,
            learned_session=None,
        )
        md = _ser.serialize(note)
        restored = _ser.deserialize(md)

        assert restored.id == note.id
        assert restored.body.strip() == note.body.strip()
        assert restored.tags == note.tags
        assert restored.topics == note.topics
        assert restored.source == note.source
        assert restored.private == note.private
        assert restored.superseded_by is None
        assert restored.learned_session is None

    def test_private_note_round_trip(self):
        note = Note(id="priv-1", body="Secret info", private=True)
        restored = _ser.deserialize(_ser.serialize(note))
        assert restored.private is True

    def test_superseded_by_round_trip(self):
        note = Note(id="old-1", body="old info", superseded_by="new-1")
        restored = _ser.deserialize(_ser.serialize(note))
        assert restored.superseded_by == "new-1"

    def test_learned_session_round_trip(self):
        note = Note(id="l-1", body="learned fact", learned_session="2024-06-01T12-00")
        restored = _ser.deserialize(_ser.serialize(note))
        assert restored.learned_session == "2024-06-01T12-00"

    def test_empty_tags_and_topics(self):
        note = Note(id="e-1", body="plain note", tags=[], topics=[])
        restored = _ser.deserialize(_ser.serialize(note))
        assert restored.tags == []
        assert restored.topics == []

    def test_multiline_body(self):
        body = "Line one.\nLine two.\nLine three."
        note = Note(id="multi-1", body=body)
        restored = _ser.deserialize(_ser.serialize(note))
        assert restored.body.strip() == body

    def test_from_markdown_missing_front_matter_raises(self):
        with pytest.raises(NoteSerializationError):
            _ser.deserialize("No front matter here at all.")


# ---------------------------------------------------------------------------
# Vault.init
# ---------------------------------------------------------------------------

class TestVaultInit:
    def test_creates_directory(self, tmp_path):
        vault_dir = tmp_path / "new_vault"
        assert not vault_dir.exists()
        v = Vault(vault_dir)
        v.init()
        assert vault_dir.is_dir()

    def test_creates_empty_index(self, tmp_path):
        v = _make_vault(tmp_path)
        index_path = v.path / "index.json"
        assert index_path.exists()
        data = json.loads(index_path.read_text())
        assert data == {"notes": []}

    def test_init_idempotent(self, tmp_path):
        v = _make_vault(tmp_path)
        v.init()   # second call — should not raise or corrupt
        index_path = v.path / "index.json"
        data = json.loads(index_path.read_text())
        assert data == {"notes": []}

    def test_init_with_existing_notes_preserves_index(self, tmp_path):
        v = _make_vault(tmp_path)
        note = _minimal_note()
        v.store(note)
        v.init()   # re-init should not wipe the index
        index_path = v.path / "index.json"
        data = json.loads(index_path.read_text())
        assert note.id in data["notes"]


# ---------------------------------------------------------------------------
# Vault.store — durable write and atomicity
# ---------------------------------------------------------------------------

class TestVaultStore:
    def test_store_returns_success(self, tmp_path):
        v = _make_vault(tmp_path)
        note = _minimal_note()
        result = v.store(note)
        assert result.success is True
        assert result.note_id == note.id
        assert result.error is None

    def test_store_creates_md_file(self, tmp_path):
        v = _make_vault(tmp_path)
        note = _minimal_note(id="abc-123", body="Test content")
        v.store(note)
        assert (v.path / "abc-123.md").exists()

    def test_store_updates_index(self, tmp_path):
        v = _make_vault(tmp_path)
        note = _minimal_note(id="idx-1")
        v.store(note)
        data = json.loads((v.path / "index.json").read_text())
        assert "idx-1" in data["notes"]

    def test_store_no_tmp_file_after_success(self, tmp_path):
        v = _make_vault(tmp_path)
        note = _minimal_note(id="notmp-1")
        v.store(note)
        # No .tmp artefact should be left behind
        tmp_files = list(v.path.glob("*.tmp"))
        assert tmp_files == []

    def test_store_persists_across_load(self, tmp_path):
        """Req 7.5: notes survive a fresh Vault instance (simulated restart)."""
        v1 = _make_vault(tmp_path)
        note = _minimal_note(id="persist-1", body="Persist me")
        v1.store(note)

        # Simulate restart: new Vault instance at same path
        v2 = Vault(tmp_path / "vault")
        loaded = v2.load("persist-1")
        assert loaded is not None
        assert loaded.body.strip() == "Persist me"

    def test_store_failure_returns_fail_result(self, tmp_path):
        """Req 7.1, 7.2: on failure return StoreResult.fail, no partial file."""
        v = _make_vault(tmp_path)
        note = _minimal_note(id="fail-note")

        # Make write fail by making the vault dir read-only
        v.path.chmod(0o555)  # r-xr-xr-x
        try:
            result = v.store(note)
        finally:
            v.path.chmod(0o755)  # restore

        assert result.success is False
        assert result.error is not None
        # The final .md file must NOT exist (no partial write)
        assert not (v.path / "fail-note.md").exists()

    def test_store_no_partial_tmp_after_failure(self, tmp_path):
        """Req 7.2: no .tmp artefact after a failed write."""
        v = _make_vault(tmp_path)
        note = _minimal_note(id="clean-tmp")

        v.path.chmod(0o555)
        try:
            v.store(note)
        finally:
            v.path.chmod(0o755)

        tmp_files = list(v.path.glob("*.tmp"))
        assert tmp_files == []

    def test_store_serialisation_error_returns_fail(self, tmp_path):
        """Req 7.2: serialisation error before write also returns failure."""
        v = _make_vault(tmp_path)
        note = _minimal_note(id="serial-fail")

        with patch.object(NoteSerializer, "serialize", side_effect=RuntimeError("boom")):
            result = v.store(note)

        assert result.success is False
        assert "boom" in result.error


# ---------------------------------------------------------------------------
# Vault.load / list_all / delete
# ---------------------------------------------------------------------------

class TestVaultLoadAndDelete:
    def test_load_returns_note(self, tmp_path):
        v = _make_vault(tmp_path)
        note = _minimal_note(id="load-1", body="I am loaded")
        v.store(note)
        loaded = v.load("load-1")
        assert loaded is not None
        assert loaded.id == "load-1"
        assert loaded.body.strip() == "I am loaded"

    def test_load_missing_returns_none(self, tmp_path):
        v = _make_vault(tmp_path)
        assert v.load("nonexistent") is None

    def test_list_all_empty_vault(self, tmp_path):
        v = _make_vault(tmp_path)
        assert v.list_all() == []

    def test_list_all_returns_stored_notes(self, tmp_path):
        v = _make_vault(tmp_path)
        n1 = _minimal_note(id="la-1", body="first")
        n2 = _minimal_note(id="la-2", body="second")
        v.store(n1)
        v.store(n2)
        ids = {n.id for n in v.list_all()}
        assert ids == {"la-1", "la-2"}

    def test_delete_removes_file_and_returns_true(self, tmp_path):
        v = _make_vault(tmp_path)
        note = _minimal_note(id="del-1")
        v.store(note)
        result = v.delete("del-1")
        assert result is True
        assert not (v.path / "del-1.md").exists()

    def test_delete_nonexistent_returns_false(self, tmp_path):
        v = _make_vault(tmp_path)
        assert v.delete("ghost") is False

    def test_delete_updates_index(self, tmp_path):
        v = _make_vault(tmp_path)
        note = _minimal_note(id="del-idx")
        v.store(note)
        v.delete("del-idx")
        data = json.loads((v.path / "index.json").read_text())
        assert "del-idx" not in data["notes"]

    def test_note_path_helper(self, tmp_path):
        v = Vault(tmp_path / "vault")
        expected = tmp_path / "vault" / "my-id.md"
        assert v.note_path("my-id") == expected


# ---------------------------------------------------------------------------
# StoreResult
# ---------------------------------------------------------------------------

class TestStoreResult:
    def test_ok(self):
        r = StoreResult.ok("note-123")
        assert r.success is True
        assert r.note_id == "note-123"
        assert r.error is None

    def test_fail(self):
        r = StoreResult.fail("disk full")
        assert r.success is False
        assert r.note_id is None
        assert r.error == "disk full"


# ---------------------------------------------------------------------------
# MemoryBrain
# ---------------------------------------------------------------------------

class TestMemoryBrain:
    def test_init_creates_vault(self, tmp_path):
        brain = MemoryBrain(vault_path=tmp_path / "vault")
        brain.init()
        assert (tmp_path / "vault").is_dir()
        assert (tmp_path / "vault" / "index.json").exists()

    def test_remember_returns_success(self, tmp_path):
        brain = MemoryBrain(vault_path=tmp_path / "vault")
        brain.init()
        result = brain.remember("My first memory")
        assert result.success is True
        assert result.note_id is not None

    def test_remember_persists_to_disk(self, tmp_path):
        brain = MemoryBrain(vault_path=tmp_path / "vault")
        brain.init()
        result = brain.remember("Persist this", tags=["t1"])
        # Verify the file actually exists on disk
        note_file = tmp_path / "vault" / f"{result.note_id}.md"
        assert note_file.exists()
        text = note_file.read_text(encoding="utf-8")
        assert "Persist this" in text

    def test_remember_with_tags_and_topics(self, tmp_path):
        brain = MemoryBrain(vault_path=tmp_path / "vault")
        brain.init()
        result = brain.remember(
            "Networks exam",
            tags=["exam"],
            topics=["networks"],
        )
        assert result.success is True
        loaded = brain.all_notes()
        assert len(loaded) == 1
        assert loaded[0].tags == ["exam"]
        assert loaded[0].topics == ["networks"]

    def test_forget_removes_note(self, tmp_path):
        brain = MemoryBrain(vault_path=tmp_path / "vault", skip_local_guard=True)
        brain.init()
        result = brain.remember("Remove me")
        forget_result = brain.forget(result.note_id)
        assert forget_result.success is True
        assert brain.all_notes() == []

    def test_forget_nonexistent_returns_false(self, tmp_path):
        brain = MemoryBrain(vault_path=tmp_path / "vault", skip_local_guard=True)
        brain.init()
        forget_result = brain.forget("does-not-exist")
        assert forget_result.success is False
        assert forget_result.error is not None

    def test_forget_all_empties_vault(self, tmp_path):
        brain = MemoryBrain(vault_path=tmp_path / "vault", skip_local_guard=True)
        brain.init()
        brain.remember("note 1")
        brain.remember("note 2")
        ok = brain.forget_all()
        assert ok.success is True
        assert brain.all_notes() == []

    def test_forget_all_empty_vault_returns_true(self, tmp_path):
        brain = MemoryBrain(vault_path=tmp_path / "vault", skip_local_guard=True)
        brain.init()
        result = brain.forget_all()
        assert result.success is True

    def test_retrieve_stub_returns_empty(self, tmp_path):
        """Retrieve is a stub until Task 14.1 — must return empty list."""
        brain = MemoryBrain(vault_path=tmp_path / "vault")
        brain.init()
        brain.remember("some content")
        # Stub always returns []
        results = brain.retrieve("some content")
        assert results == []

    def test_notes_persist_across_brain_restart(self, tmp_path):
        """Req 7.5: notes survive creating a new MemoryBrain at the same path."""
        vault_path = tmp_path / "vault"
        brain1 = MemoryBrain(vault_path=vault_path)
        brain1.init()
        result = brain1.remember("Persistent fact", tags=["p"])

        # Simulate restart
        brain2 = MemoryBrain(vault_path=vault_path)
        notes = brain2.all_notes()
        ids = [n.id for n in notes]
        assert result.note_id in ids
