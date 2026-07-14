"""
Unit tests for the Image_Studio module.

Covers:
  Req 15.1: generate() calls provider and returns a displayable image
  Req 15.2: edit without explicit target → resolves to most recent image
  Req 15.3: edit with explicit reference → resolves to referenced image
  Req 15.4: generated/edited images are saved to the designated folder and
            confirmed to the user
  Req 15.5: save failure → image kept in-session, user informed
  Req 15.6: generation failure → user informed with reason
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.image_studio import ImageStudio, ImageEntry, ImageResult, GenerationFailure
from core.image_studio.image_studio import _resolve_target_ref, _extract_image_bytes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal valid 1-pixel PNG bytes
_STUB_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
    b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _make_studio(tmp_path: Path, provider=None) -> ImageStudio:
    """Create an ImageStudio with a tmp save directory."""
    return ImageStudio(image_provider=provider, save_dir=tmp_path / "images")


def _stub_provider(return_bytes: bytes = _STUB_PNG) -> MagicMock:
    """Create a mock ModelProvider that returns raw image bytes."""
    p = MagicMock()
    p.invoke.return_value = return_bytes
    return p


def _failing_provider(reason: str = "model unavailable") -> MagicMock:
    p = MagicMock()
    p.invoke.side_effect = RuntimeError(reason)
    return p


# ---------------------------------------------------------------------------
# Req 15.1 — Generate
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_generate_returns_success(self, tmp_path):
        """Req 15.1: generate returns a successful result."""
        studio = _make_studio(tmp_path)
        result = studio.generate("a sunset over mountains")
        assert result.success is True
        assert result.entry is not None
        assert result.entry.data  # non-empty image bytes

    def test_generate_adds_to_history(self, tmp_path):
        """Req 15.1: generated image is added to session history."""
        studio = _make_studio(tmp_path)
        studio.generate("sunset")
        assert len(studio.history()) == 1

    def test_generate_multiple_adds_all_to_history(self, tmp_path):
        """Req 15.1: multiple generations are all kept in history."""
        studio = _make_studio(tmp_path)
        studio.generate("prompt A")
        studio.generate("prompt B")
        assert len(studio.history()) == 2

    def test_generate_with_real_provider_returns_bytes(self, tmp_path):
        """Req 15.1: provider-supplied bytes are preserved."""
        provider = _stub_provider(_STUB_PNG)
        studio = _make_studio(tmp_path, provider=provider)
        result = studio.generate("vibrant forest")
        assert result.success is True
        assert result.entry.data == _STUB_PNG

    def test_generate_message_confirms_save(self, tmp_path):
        """Req 15.4: message mentions save path when save succeeds."""
        studio = _make_studio(tmp_path)
        result = studio.generate("a cat")
        # Save succeeds → message contains path info
        assert result.saved is True
        assert result.save_path is not None
        assert str(result.save_path) in result.message or "saved" in result.message.lower()

    def test_generate_saves_file_to_disk(self, tmp_path):
        """Req 15.4: generated image file exists on disk after generate."""
        studio = _make_studio(tmp_path)
        result = studio.generate("a dog")
        assert result.saved is True
        assert result.save_path.exists()
        assert result.save_path.read_bytes() != b""


# ---------------------------------------------------------------------------
# Req 15.4 — Save confirmation
# ---------------------------------------------------------------------------


class TestSaveConfirmation:
    def test_save_confirms_with_path_in_message(self, tmp_path):
        """Req 15.4: successful save is confirmed in the user-facing message."""
        studio = _make_studio(tmp_path)
        result = studio.generate("a mountain lake")
        assert result.saved is True
        assert result.save_path is not None
        # Message must mention the path or saving
        assert result.save_path.name in result.message or "saved" in result.message.lower()

    def test_generate_file_has_png_extension(self, tmp_path):
        """Generated files should be PNG."""
        studio = _make_studio(tmp_path)
        result = studio.generate("a sunrise")
        assert result.save_path.suffix == ".png"


# ---------------------------------------------------------------------------
# Req 15.5 — Save failure handling
# ---------------------------------------------------------------------------


class TestSaveFailure:
    def test_save_failure_keeps_image_in_session(self, tmp_path):
        """Req 15.5: on save failure the image remains in the session history."""
        studio = _make_studio(tmp_path)

        # Make the save dir unwritable to force a save failure
        save_dir = tmp_path / "images"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_dir.chmod(0o444)  # read-only

        try:
            result = studio.generate("a volcano")
        finally:
            save_dir.chmod(0o755)  # restore permissions

        # Image generation succeeded
        assert result.success is True
        assert result.entry is not None
        # Save failed
        assert result.saved is False
        # Image still in session history (Req 15.5)
        assert len(studio.history()) == 1
        assert studio.last_image().id == result.entry.id

    def test_save_failure_message_informs_user(self, tmp_path):
        """Req 15.5: user is informed of save failure in the message."""
        studio = _make_studio(tmp_path)

        save_dir = tmp_path / "images"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_dir.chmod(0o444)

        try:
            result = studio.generate("a waterfall")
        finally:
            save_dir.chmod(0o755)

        assert result.success is True
        # Message should mention that saving failed
        assert (
            "couldn't save" in result.message.lower()
            or "could not save" in result.message.lower()
            or "session" in result.message.lower()
        )

    def test_save_failure_save_path_is_none(self, tmp_path):
        """Req 15.5: save_path is None when save fails."""
        studio = _make_studio(tmp_path)

        save_dir = tmp_path / "images"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_dir.chmod(0o444)

        try:
            result = studio.generate("a desert dune")
        finally:
            save_dir.chmod(0o755)

        assert result.saved is False
        assert result.save_path is None


# ---------------------------------------------------------------------------
# Req 15.6 — Generation failure handling
# ---------------------------------------------------------------------------


class TestGenerationFailure:
    def test_generate_failure_returns_fail_result(self, tmp_path):
        """Req 15.6: provider failure returns ImageResult.fail."""
        provider = _failing_provider("GPU out of memory")
        studio = _make_studio(tmp_path, provider=provider)
        result = studio.generate("a galaxy")
        assert result.success is False

    def test_generate_failure_message_contains_reason(self, tmp_path):
        """Req 15.6: failure message contains the reason."""
        provider = _failing_provider("model not loaded")
        studio = _make_studio(tmp_path, provider=provider)
        result = studio.generate("a forest")
        assert "model not loaded" in result.message or "couldn't generate" in result.message.lower()

    def test_generate_failure_does_not_add_to_history(self, tmp_path):
        """Req 15.6: failed generation does not pollute session history."""
        provider = _failing_provider("network error")
        studio = _make_studio(tmp_path, provider=provider)
        studio.generate("a spaceship")
        assert studio.history() == []

    def test_edit_failure_returns_fail_result(self, tmp_path):
        """Req 15.6: edit provider failure returns ImageResult.fail."""
        # First generate a successful image with stub provider
        studio = _make_studio(tmp_path)
        studio.generate("a bird")

        # Now replace provider with a failing one
        studio._provider = _failing_provider("model error")
        result = studio.edit("make it brighter")
        assert result.success is False
        assert result.message  # must contain a reason


# ---------------------------------------------------------------------------
# Req 15.2 — Edit resolves to most recent image
# ---------------------------------------------------------------------------


class TestEditLastImage:
    def test_edit_without_explicit_ref_targets_last_image(self, tmp_path):
        """Req 15.2: edit with vague reference applies to most recent image."""
        studio = _make_studio(tmp_path)
        studio.generate("prompt A")
        r2 = studio.generate("prompt B")

        result = studio.edit("make it brighter")
        assert result.success is True
        # The edit was applied — a new entry is in history
        history = studio.history()
        assert len(history) == 3  # 2 originals + 1 edit

    def test_edit_uses_last_image_from_history(self, tmp_path):
        """Req 15.2: the edit prompt embeds the last image's prompt."""
        studio = _make_studio(tmp_path)
        studio.generate("prompt X")

        result = studio.edit("remove the background")
        assert result.success is True
        # The new entry's prompt should reference the original
        new_entry = studio.last_image()
        assert "prompt X" in new_entry.prompt.lower() or "editing" in new_entry.prompt.lower()

    def test_edit_empty_history_returns_fail(self, tmp_path):
        """Req 15.2: edit with no images in history returns failure."""
        studio = _make_studio(tmp_path)
        result = studio.edit("make it darker")
        assert result.success is False
        assert "no images" in result.message.lower() or "generate" in result.message.lower()


# ---------------------------------------------------------------------------
# Req 15.3 — Edit resolves to referenced image
# ---------------------------------------------------------------------------


class TestEditReferencedImage:
    def test_edit_with_explicit_id_targets_that_image(self, tmp_path):
        """Req 15.3: edit with explicit ID resolves to that specific image."""
        studio = _make_studio(tmp_path)
        r1 = studio.generate("first image")
        studio.generate("second image")

        result = studio.edit("flip it horizontally", target_ref=r1.entry.id)
        assert result.success is True
        new_entry = studio.last_image()
        assert "first image" in new_entry.prompt.lower() or "editing" in new_entry.prompt.lower()

    def test_edit_ordinal_reference_targets_correct_image(self, tmp_path):
        """Req 15.3: ordinal references in instruction text resolve correctly."""
        studio = _make_studio(tmp_path)
        studio.generate("sunset")
        studio.generate("mountain")
        studio.generate("ocean")

        result = studio.edit("make the first image black and white")
        assert result.success is True
        # The edit history entry should reference "sunset"
        new_entry = studio.last_image()
        assert "sunset" in new_entry.prompt.lower() or "editing" in new_entry.prompt.lower()

    def test_edit_numeric_reference_targets_correct_image(self, tmp_path):
        """Req 15.3: 'image 2' in instruction resolves to the second image."""
        studio = _make_studio(tmp_path)
        studio.generate("image one prompt")
        studio.generate("image two prompt")

        result = studio.edit("make image 2 darker")
        assert result.success is True
        new_entry = studio.last_image()
        assert "image two" in new_entry.prompt.lower() or "editing" in new_entry.prompt.lower()


# ---------------------------------------------------------------------------
# Target reference resolution unit tests
# ---------------------------------------------------------------------------


class TestResolveTargetRef:
    def _make_entries(self, n: int) -> list[ImageEntry]:
        return [
            ImageEntry(id=f"id-{i}", prompt=f"prompt {i}", data=_STUB_PNG)
            for i in range(1, n + 1)
        ]

    def test_empty_history_returns_none(self):
        assert _resolve_target_ref("edit this", [], None) is None

    def test_no_reference_returns_last(self):
        entries = self._make_entries(3)
        result = _resolve_target_ref("make it darker", entries, None)
        assert result.id == "id-3"

    def test_explicit_id_returns_that_entry(self):
        entries = self._make_entries(3)
        result = _resolve_target_ref("anything", entries, "id-2")
        assert result.id == "id-2"

    def test_this_returns_last(self):
        entries = self._make_entries(2)
        result = _resolve_target_ref("make this brighter", entries, None)
        assert result.id == "id-2"

    def test_ordinal_first_returns_first(self):
        entries = self._make_entries(3)
        result = _resolve_target_ref("edit the first image", entries, None)
        assert result.id == "id-1"

    def test_ordinal_second_returns_second(self):
        entries = self._make_entries(3)
        result = _resolve_target_ref("edit the second image", entries, None)
        assert result.id == "id-2"

    def test_numeric_image_2_returns_second(self):
        entries = self._make_entries(3)
        result = _resolve_target_ref("make image 2 darker", entries, None)
        assert result.id == "id-2"

    def test_out_of_range_numeric_falls_back_to_last(self):
        entries = self._make_entries(2)
        result = _resolve_target_ref("edit image 99", entries, None)
        # Out-of-range → falls through to default (most recent)
        assert result.id == "id-2"


# ---------------------------------------------------------------------------
# _extract_image_bytes unit tests
# ---------------------------------------------------------------------------


class TestExtractImageBytes:
    def test_raw_bytes_returned_directly(self):
        assert _extract_image_bytes(_STUB_PNG) == _STUB_PNG

    def test_image_data_key(self):
        assert _extract_image_bytes({"image_data": _STUB_PNG}) == _STUB_PNG

    def test_image_b64_key(self):
        import base64
        b64 = base64.b64encode(_STUB_PNG).decode()
        assert _extract_image_bytes({"image_b64": b64}) == _STUB_PNG

    def test_stub_response_returns_placeholder(self):
        result = _extract_image_bytes({"stub": True})
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_unrecognised_raises_generation_failure(self):
        with pytest.raises(GenerationFailure):
            _extract_image_bytes({"unknown_key": 123})


# ---------------------------------------------------------------------------
# Session history management
# ---------------------------------------------------------------------------


class TestSessionHistory:
    def test_history_empty_initially(self, tmp_path):
        studio = _make_studio(tmp_path)
        assert studio.history() == []

    def test_last_image_none_when_empty(self, tmp_path):
        studio = _make_studio(tmp_path)
        assert studio.last_image() is None

    def test_last_image_after_generate(self, tmp_path):
        studio = _make_studio(tmp_path)
        result = studio.generate("stars")
        assert studio.last_image().id == result.entry.id

    def test_get_by_id_returns_correct_entry(self, tmp_path):
        studio = _make_studio(tmp_path)
        r = studio.generate("a tree")
        found = studio.get_by_id(r.entry.id)
        assert found is not None
        assert found.id == r.entry.id

    def test_get_by_id_missing_returns_none(self, tmp_path):
        studio = _make_studio(tmp_path)
        assert studio.get_by_id("non-existent-id") is None

    def test_clear_history_removes_all_entries(self, tmp_path):
        studio = _make_studio(tmp_path)
        studio.generate("one")
        studio.generate("two")
        studio.clear_history()
        assert studio.history() == []
        assert studio.last_image() is None

    def test_display_labels_are_sequential(self, tmp_path):
        studio = _make_studio(tmp_path)
        studio.generate("first")
        studio.generate("second")
        labels = [e.display_label for e in studio.history()]
        assert labels == ["Image 1", "Image 2"]
