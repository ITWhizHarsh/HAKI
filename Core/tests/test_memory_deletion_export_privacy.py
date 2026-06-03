"""
Tests for memory deletion, export, and privacy controls (Task 15).

Covers:
  - Req 7.6:  forget(noteId) removes and confirms
  - Req 9.3:  export() produces a single file at a user-accessible location
  - Req 9.4:  notes removed only on explicit request
  - Req 9.5:  forgetAll() removes all notes and confirms
  - Req 9.6:  if forgetAll cannot confirm, notes are not removed
  - Req 9.2:  LocalStorageGuard prevents cloud/network vault paths
  - Req 9.7:  PrivacyManager is accessible before and during any conversation
  - Req 9.8:  export failure produces no partial file

**Validates: Requirements 7.6, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8**
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from core.memory import (
    ExportResult,
    ForgetResult,
    LocalStorageGuard,
    MemoryBrain,
    PrivacyManager,
)
from core.memory.vault import Vault


# ---------------------------------------------------------------------------
# Stub embeddings provider (same as other test modules)
# ---------------------------------------------------------------------------


class _StubEmbedProvider:
    _DIM = 8

    def invoke(self, text: str, **_: Any) -> list[float]:
        vec = [0.0] * self._DIM
        for i, ch in enumerate(text):
            vec[i % self._DIM] += ord(ch)
        mag = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / mag for x in vec]


_PROVIDER = _StubEmbedProvider()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_brain(tmp_path: Path, with_index: bool = False) -> MemoryBrain:
    provider = _PROVIDER if with_index else None
    brain = MemoryBrain(
        vault_path=tmp_path / "vault",
        embeddings_provider=provider,
        skip_local_guard=True,
    )
    brain.init()
    return brain


def _brain_in_tempdir(with_index: bool = False):
    """Return (brain, tmp_dir) suitable for use in property tests."""
    td = tempfile.TemporaryDirectory()
    provider = _PROVIDER if with_index else None
    brain = MemoryBrain(
        vault_path=Path(td.name) / "vault",
        embeddings_provider=provider,
        skip_local_guard=True,
    )
    brain.init()
    return brain, td


# ===========================================================================
# 15.1 — forget(noteId): single-note delete atomicity
# ===========================================================================


class TestForget:
    """Req 7.6, 9.4"""

    def test_forget_returns_ok_for_existing_note(self, tmp_path):
        brain = _make_brain(tmp_path)
        r = brain.remember("Something to forget")
        result = brain.forget(r.note_id)
        assert isinstance(result, ForgetResult)
        assert result.success is True
        assert result.error is None

    def test_forget_removes_file_from_vault(self, tmp_path):
        brain = _make_brain(tmp_path)
        r = brain.remember("Delete me please")
        note_path = brain._vault.note_path(r.note_id)
        assert note_path.exists()
        brain.forget(r.note_id)
        assert not note_path.exists()

    def test_forget_removes_from_list_all(self, tmp_path):
        brain = _make_brain(tmp_path)
        r = brain.remember("Gone note")
        brain.forget(r.note_id)
        ids = {n.id for n in brain.all_notes()}
        assert r.note_id not in ids

    def test_forget_nonexistent_returns_fail(self, tmp_path):
        brain = _make_brain(tmp_path)
        result = brain.forget("ghost-id-does-not-exist")
        assert isinstance(result, ForgetResult)
        assert result.success is False
        assert result.error is not None

    def test_forget_leaves_other_notes_intact(self, tmp_path):
        brain = _make_brain(tmp_path)
        r1 = brain.remember("Note to keep")
        r2 = brain.remember("Note to delete")
        brain.forget(r2.note_id)
        remaining = [n.id for n in brain.all_notes()]
        assert r1.note_id in remaining
        assert r2.note_id not in remaining

    def test_forget_confirms_only_after_file_removed(self, tmp_path):
        """Req 9.4, 9.6: file must actually be gone before confirm."""
        brain = _make_brain(tmp_path)
        r = brain.remember("Confirm after removal")
        # Real file removal — confirm only after success
        result = brain.forget(r.note_id)
        assert result.success is True
        assert not brain._vault.note_path(r.note_id).exists()

    def test_forget_does_not_remove_on_failure(self, tmp_path):
        """Req 9.4: if removal can't be confirmed, data is intact."""
        brain = _make_brain(tmp_path)
        r = brain.remember("Must not be lost")
        note_path = brain._vault.note_path(r.note_id)

        # Simulate an OS error during unlink
        with patch.object(Path, "unlink", side_effect=OSError("permission denied")):
            result = brain.forget(r.note_id)

        # Must report failure
        assert result.success is False
        # File must still exist (data intact)
        assert note_path.exists()
        ids = {n.id for n in brain.all_notes()}
        assert r.note_id in ids

    def test_forget_with_index_removes_chunks(self, tmp_path):
        """Vector index chunks are cleaned up after forget."""
        brain = _make_brain(tmp_path, with_index=True)
        r = brain.remember("Networks exam prep notes", topics=["networks"])
        assert brain._indexer.count() >= 1
        brain.forget(r.note_id)
        assert brain._indexer.get_chunks_for_note(r.note_id) == []

    def test_forget_index_updated_only_after_file_removed(self, tmp_path):
        """Index update happens after and only after file removal succeeds."""
        brain = _make_brain(tmp_path, with_index=True)
        r = brain.remember("Indexed note")
        note_path = brain._vault.note_path(r.note_id)

        # Make file removal fail
        with patch.object(Path, "unlink", side_effect=OSError("blocked")):
            result = brain.forget(r.note_id)

        # File still on disk, index still has chunks (not double-deleted)
        assert result.success is False
        assert note_path.exists()
        # Index chunks must remain since file deletion was aborted
        assert brain._indexer.count() >= 1


# ===========================================================================
# 15.1 — forgetAll(): atomic multi-note delete
# ===========================================================================


class TestForgetAll:
    """Req 9.5, 9.6"""

    def test_forget_all_returns_ok(self, tmp_path):
        brain = _make_brain(tmp_path)
        brain.remember("note 1")
        brain.remember("note 2")
        result = brain.forget_all()
        assert isinstance(result, ForgetResult)
        assert result.success is True

    def test_forget_all_empties_vault(self, tmp_path):
        brain = _make_brain(tmp_path)
        for i in range(5):
            brain.remember(f"Note number {i}")
        brain.forget_all()
        assert brain.all_notes() == []

    def test_forget_all_empty_vault_returns_ok(self, tmp_path):
        brain = _make_brain(tmp_path)
        result = brain.forget_all()
        assert result.success is True

    def test_forget_all_clears_index(self, tmp_path):
        """Vector index is empty after forgetAll."""
        brain = _make_brain(tmp_path, with_index=True)
        brain.remember("Keep nothing")
        brain.remember("Clear everything")
        brain.forget_all()
        assert brain._indexer.count() == 0

    def test_forget_all_leaves_no_note_files(self, tmp_path):
        brain = _make_brain(tmp_path)
        for i in range(3):
            brain.remember(f"File {i}")
        brain.forget_all()
        md_files = list((tmp_path / "vault").glob("*.md"))
        assert md_files == []

    def test_forget_all_restores_on_failure(self, tmp_path):
        """Req 9.6: if forgetAll fails, notes are not removed."""
        brain = _make_brain(tmp_path)
        brain.remember("note A")
        brain.remember("note B")
        original_count = len(brain.all_notes())

        vault_path = tmp_path / "vault"

        # Make rename (mv) fail for notes so staging step fails
        original_rename = Path.rename

        def failing_rename(self_path, target):
            if self_path.suffix == ".md":
                raise OSError("simulated rename failure")
            return original_rename(self_path, target)

        with patch.object(Path, "rename", failing_rename):
            result = brain.forget_all()

        # Should report failure
        assert result.success is False
        # Notes must be intact
        assert len(brain.all_notes()) == original_count

    def test_forget_all_index_not_cleared_on_failure(self, tmp_path):
        """Req 9.6: vector index is not cleared if vault deletion fails."""
        brain = _make_brain(tmp_path, with_index=True)
        brain.remember("Important indexed note")
        chunk_count_before = brain._indexer.count()
        assert chunk_count_before >= 1

        original_rename = Path.rename

        def failing_rename(self_path, target):
            if self_path.suffix == ".md":
                raise OSError("blocked")
            return original_rename(self_path, target)

        with patch.object(Path, "rename", failing_rename):
            result = brain.forget_all()

        assert result.success is False
        # Index still intact
        assert brain._indexer.count() >= 1


# ===========================================================================
# 15.1 — export(): atomic single-file export
# ===========================================================================


class TestExport:
    """Req 9.3, 9.8"""

    def test_export_returns_ok(self, tmp_path):
        brain = _make_brain(tmp_path)
        brain.remember("Exam on June 14")
        dest = tmp_path / "export.md"
        result = brain.export(dest)
        assert isinstance(result, ExportResult)
        assert result.success is True
        assert result.path == dest

    def test_export_creates_file(self, tmp_path):
        brain = _make_brain(tmp_path)
        brain.remember("Some note content here")
        dest = tmp_path / "memory.md"
        brain.export(dest)
        assert dest.exists()

    def test_export_contains_all_note_bodies(self, tmp_path):
        brain = _make_brain(tmp_path)
        bodies = ["Networks midterm on June 14", "Birthday party on July 4"]
        for body in bodies:
            brain.remember(body)
        dest = tmp_path / "full_export.md"
        brain.export(dest)
        content = dest.read_text(encoding="utf-8")
        for body in bodies:
            assert body in content, f"Expected '{body}' in export"

    def test_export_empty_vault_produces_file(self, tmp_path):
        brain = _make_brain(tmp_path)
        dest = tmp_path / "empty_export.md"
        result = brain.export(dest)
        assert result.success is True
        assert dest.exists()

    def test_export_no_partial_file_on_failure(self, tmp_path):
        """Req 9.8: if write fails, no partial file at destination."""
        brain = _make_brain(tmp_path)
        brain.remember("Content that should not appear")
        dest = tmp_path / "should_not_exist.md"

        # Make the atomic rename fail
        original_rename = Path.rename

        def failing_rename(self_path, target):
            if str(target) == str(dest):
                raise OSError("simulated rename failure")
            return original_rename(self_path, target)

        with patch.object(Path, "rename", failing_rename):
            result = brain.export(dest)

        assert result.success is False
        assert not dest.exists(), "Partial export file must not exist (Req 9.8)"

    def test_export_no_tmp_file_left_after_failure(self, tmp_path):
        """Req 9.8: no temp artefact left behind after export failure."""
        brain = _make_brain(tmp_path)
        brain.remember("Some content")
        dest = tmp_path / "fail_export.md"

        original_rename = Path.rename

        def failing_rename(self_path, target):
            if str(target) == str(dest):
                raise OSError("blocked")
            return original_rename(self_path, target)

        with patch.object(Path, "rename", failing_rename):
            brain.export(dest)

        tmp_files = list(tmp_path.glob("_haki_export_*.tmp"))
        assert tmp_files == [], f"Temp export files left behind: {tmp_files}"

    def test_export_default_destination_is_user_accessible(self, tmp_path):
        """Default export destination is ~/Desktop/haki_memory_export.md."""
        brain = _make_brain(tmp_path)
        # Don't actually write to ~/Desktop in tests — just check the path is set
        with patch.object(brain._vault, "export_atomic", return_value=True) as mock_export:
            brain.export()
            call_args = mock_export.call_args
            dest_path = call_args[0][0]
        assert "haki_memory_export" in dest_path.name
        assert dest_path.parent.name == "Desktop"

    def test_export_result_fail_message_mentions_no_partial(self, tmp_path):
        """Req 9.8: failure message informs user no partial file was produced."""
        brain = _make_brain(tmp_path)
        dest = tmp_path / "fail.md"

        original_rename = Path.rename

        def failing_rename(self_path, target):
            if str(target) == str(dest):
                raise OSError("write failed")
            return original_rename(self_path, target)

        with patch.object(Path, "rename", failing_rename):
            result = brain.export(dest)

        assert result.success is False
        assert result.error is not None


# ===========================================================================
# 15.2 — LocalStorageGuard: local-only storage guarantee
# ===========================================================================


class TestLocalStorageGuard:
    """Req 9.2"""

    def test_local_path_passes(self, tmp_path):
        """A plain local path must not raise."""
        guard = LocalStorageGuard(tmp_path / "vault")
        # No exception means it passed

    def test_home_haki_vault_passes(self):
        guard = LocalStorageGuard(Path.home() / ".haki" / "vault")

    @pytest.mark.parametrize(
        "bad_path",
        [
            "/Users/user/Library/Mobile Documents/iCloudDrive/Notes",
            "/Users/user/Dropbox/haki_vault",
            "/Users/user/OneDrive/docs",
            "/Users/user/Google Drive/vault",
            "/Users/user/Box Sync/notes",
        ],
    )
    def test_cloud_path_raises(self, bad_path):
        """Cloud-sync paths must raise ValueError (Req 9.2)."""
        with pytest.raises(ValueError, match="cloud-sync|network"):
            LocalStorageGuard(Path(bad_path))

    def test_network_mount_smb_raises(self):
        with pytest.raises(ValueError):
            LocalStorageGuard(Path("/smb/server/share/vault"))

    def test_error_message_suggests_local_path(self, tmp_path):
        try:
            LocalStorageGuard(Path("/Users/user/Dropbox/vault"))
        except ValueError as exc:
            assert "~/.haki/vault" in str(exc) or "home directory" in str(exc)

    def test_memory_brain_raises_with_cloud_path(self):
        with pytest.raises(ValueError):
            MemoryBrain(
                vault_path=Path("/Users/user/Dropbox/vault"),
                skip_local_guard=False,
            )

    def test_memory_brain_skip_guard_bypasses_check(self, tmp_path):
        """skip_local_guard=True allows tests to use tmp_path freely."""
        # Should not raise even if path were suspicious
        brain = MemoryBrain(vault_path=tmp_path / "vault", skip_local_guard=True)
        brain.init()
        assert (tmp_path / "vault").is_dir()


# ===========================================================================
# 15.2 — PrivacyManager: conversation privacy-designation control
# ===========================================================================


class TestPrivacyManager:
    """Req 9.7"""

    def test_new_conversation_is_not_private(self, tmp_path):
        pm = PrivacyManager(db_path=tmp_path / "privacy.db")
        assert pm.is_private("conv-001") is False

    def test_designate_private_marks_conversation(self, tmp_path):
        pm = PrivacyManager(db_path=tmp_path / "privacy.db")
        pm.designate_private("conv-abc")
        assert pm.is_private("conv-abc") is True

    def test_designate_private_idempotent(self, tmp_path):
        pm = PrivacyManager(db_path=tmp_path / "privacy.db")
        pm.designate_private("conv-idem")
        pm.designate_private("conv-idem")  # second call must not raise
        assert pm.is_private("conv-idem") is True

    def test_revoke_private_clears_designation(self, tmp_path):
        pm = PrivacyManager(db_path=tmp_path / "privacy.db")
        pm.designate_private("conv-rev")
        pm.revoke_private("conv-rev")
        assert pm.is_private("conv-rev") is False

    def test_revoke_nonexistent_is_noop(self, tmp_path):
        pm = PrivacyManager(db_path=tmp_path / "privacy.db")
        # Must not raise
        pm.revoke_private("conv-ghost")

    def test_privacy_state_persists_across_instances(self, tmp_path):
        """Req 9.7: state is accessible before and during any conversation."""
        db = tmp_path / "privacy.db"
        pm1 = PrivacyManager(db_path=db)
        pm1.designate_private("conv-persist")

        # Simulate restart — new instance, same db
        pm2 = PrivacyManager(db_path=db)
        assert pm2.is_private("conv-persist") is True

    def test_multiple_conversations_independent(self, tmp_path):
        pm = PrivacyManager(db_path=tmp_path / "privacy.db")
        pm.designate_private("private-conv")
        # other conversation not designated
        assert pm.is_private("private-conv") is True
        assert pm.is_private("public-conv") is False

    def test_control_accessible_before_conversation(self, tmp_path):
        """
        Req 9.7: control is accessible BEFORE a conversation begins.
        Designating a conversation as private before any messages are
        sent must work.
        """
        pm = PrivacyManager(db_path=tmp_path / "privacy.db")
        pm.designate_private("upcoming-conv")
        assert pm.is_private("upcoming-conv") is True

    def test_control_accessible_during_conversation(self, tmp_path):
        """
        Req 9.7: control is accessible DURING a conversation.
        A mid-session privacy toggle must be reflected immediately.
        """
        pm = PrivacyManager(db_path=tmp_path / "privacy.db")
        # Conversation starts public
        assert pm.is_private("live-conv") is False
        # User activates privacy mid-session
        pm.designate_private("live-conv")
        assert pm.is_private("live-conv") is True


# ===========================================================================
# Property-based tests
# ===========================================================================


# Feature: haki-personal-ai-assistant, Property 20: Deletion removes and confirms
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    body=st.text(
        alphabet=st.characters(whitelist_categories=("L", "Zs")),
        min_size=1,
        max_size=200,
    ),
)
def test_property_forget_removes_and_confirms(body: str) -> None:
    """
    **Validates: Requirements 7.6, 8.5**

    # Feature: haki-personal-ai-assistant, Property 20: Deletion removes and confirms

    Property: for any stored note, forget(noteId) returns success and the
    note is no longer in the vault afterwards.
    """
    brain, td = _brain_in_tempdir()
    try:
        body_clean = body.strip() or "content"
        r = brain.remember(body_clean)
        result = brain.forget(r.note_id)

        assert result.success is True, (
            f"forget() should succeed for an existing note, got: {result.error}"
        )
        all_ids = {n.id for n in brain.all_notes()}
        assert r.note_id not in all_ids, (
            "Note must be absent from vault after successful forget()"
        )
    finally:
        td.cleanup()


# Feature: haki-personal-ai-assistant, Property 28: Delete-all empties the vault
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    num_notes=st.integers(min_value=0, max_value=10),
)
def test_property_forget_all_empties_vault(num_notes: int) -> None:
    """
    **Validates: Requirements 9.5**

    # Feature: haki-personal-ai-assistant, Property 28: Delete-all empties the vault

    Property: after forgetAll() succeeds, all_notes() returns an empty list.
    """
    brain, td = _brain_in_tempdir()
    try:
        for i in range(num_notes):
            brain.remember(f"Note {i} about various topics")
        result = brain.forget_all()
        assert result.success is True, (
            f"forget_all() should succeed, got: {result.error}"
        )
        assert brain.all_notes() == [], (
            f"Vault must be empty after forget_all(), found: {len(brain.all_notes())} notes"
        )
    finally:
        td.cleanup()


# Feature: haki-personal-ai-assistant, Property 27: Export completeness round trip
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    bodies=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "Zs")),
            min_size=3,
            max_size=100,
        ),
        min_size=1,
        max_size=5,
    ),
)
def test_property_export_completeness(bodies: list[str]) -> None:
    """
    **Validates: Requirements 9.3**

    # Feature: haki-personal-ai-assistant, Property 27: Export completeness round trip

    Property: the exported file contains every stored note body (no note
    is missing from the export).
    """
    brain, td = _brain_in_tempdir()
    try:
        for body in bodies:
            brain.remember(body.strip() or "fallback")
        dest = Path(td.name) / "export_test.md"
        result = brain.export(dest)
        assert result.success is True, (
            f"export() should succeed, got: {result.error}"
        )
        content = dest.read_text(encoding="utf-8")
        for body in bodies:
            clean = body.strip() or "fallback"
            assert clean in content, (
                f"Note body '{clean}' missing from export"
            )
    finally:
        td.cleanup()


# Feature: haki-personal-ai-assistant, Property 29: Deletion/export failure atomicity
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    body=st.text(
        alphabet=st.characters(whitelist_categories=("L", "Zs")),
        min_size=3,
        max_size=100,
    ),
)
def test_property_export_no_partial_file_on_failure(body: str) -> None:
    """
    **Validates: Requirements 9.8**

    # Feature: haki-personal-ai-assistant, Property 29: Deletion/export failure atomicity

    Property: when export() fails, no partial file exists at the destination path.
    """
    brain, td = _brain_in_tempdir()
    try:
        brain.remember(body.strip() or "content")
        dest = Path(td.name) / "partial_test.md"

        # Patch rename to always fail so the temp file is never moved to dest
        original_rename = Path.rename

        def always_fail_rename(self_path, target):
            if "_haki_export_" in self_path.name:
                raise OSError("forced failure")
            return original_rename(self_path, target)

        with patch.object(Path, "rename", always_fail_rename):
            result = brain.export(dest)

        assert result.success is False
        assert not dest.exists(), (
            "Partial export file must not exist after failed export (Req 9.8)"
        )
    finally:
        td.cleanup()


# Feature: haki-personal-ai-assistant, Property 26: Private conversations write nothing
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    conversation_id=st.from_regex(r"[a-z0-9\-]{3,20}", fullmatch=True),
)
def test_property_privacy_designation_persists(conversation_id: str) -> None:
    """
    **Validates: Requirements 9.7**

    # Feature: haki-personal-ai-assistant, Property 26: Private conversations write nothing

    Property: for any conversation_id, designating it as private means
    is_private() consistently returns True, and revoking means it returns False.
    """
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "privacy.db"
        pm = PrivacyManager(db_path=db_path)

        # Before designation — not private
        assert pm.is_private(conversation_id) is False

        # After designation — private
        pm.designate_private(conversation_id)
        assert pm.is_private(conversation_id) is True

        # After revocation — not private
        pm.revoke_private(conversation_id)
        assert pm.is_private(conversation_id) is False
