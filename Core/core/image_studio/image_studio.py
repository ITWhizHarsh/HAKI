"""
Image_Studio — image generation, editing, session history, and persistence.

Provides:
- ``ImageStudio.generate(prompt)`` — generate a new image and save it (15.1, 15.4)
- ``ImageStudio.edit(instruction, target_ref)`` — edit a session image (15.2, 15.3)
- Session image history tracking (15.2, 15.3)
- Save-with-confirm to a designated user-accessible folder (15.4)
- Retain in-session + inform on save failure (15.5)
- Inform with reason on generation/edit failure (15.6)

Image model runs through the Model Provider IMAGE capability (15.1, 20).

Design: Image_Studio.
Requirements: 15.1–15.6.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.model_provider.model_provider import ModelProvider

logger = logging.getLogger(__name__)

# Default folder where generated images are saved (user-accessible on macOS).
_DEFAULT_SAVE_DIR: Path = Path.home() / "Pictures" / "HAKI"

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ImageEntry:
    """
    A single image in the session history.

    Attributes
    ----------
    id : str
        Stable unique identifier for this image within the session.
    prompt : str
        The original prompt or description used to generate the image.
    data : bytes
        Raw image bytes (PNG/JPEG/etc. as returned by the model provider).
    created_at : datetime
        UTC timestamp of when the image was generated.
    saved_path : Path | None
        Path on disk where the image was saved, or ``None`` when saving
        failed or has not been attempted.
    display_label : str
        Short label suitable for UI display (e.g. "Image 1").
    """

    id: str
    prompt: str
    data: bytes
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    saved_path: Path | None = None
    display_label: str = ""

    def __post_init__(self) -> None:
        if not self.display_label:
            # Will be set by ImageStudio when appending to history
            self.display_label = f"Image {self.id[:8]}"


@dataclass
class ImageResult:
    """
    The outcome of a generate or edit operation.

    Attributes
    ----------
    success : bool
        ``True`` when the image was generated/edited successfully.
    entry : ImageEntry | None
        The resulting image entry (populated on success).
    saved : bool
        Whether the image was durably saved to disk.
    save_path : Path | None
        The path where the image was saved (populated when ``saved`` is
        ``True``).
    message : str
        Human-readable confirmation or error message for the user.
    """

    success: bool
    entry: ImageEntry | None = None
    saved: bool = False
    save_path: Path | None = None
    message: str = ""

    @classmethod
    def ok(
        cls,
        entry: ImageEntry,
        *,
        saved: bool,
        save_path: Path | None,
        message: str,
    ) -> "ImageResult":
        return cls(
            success=True,
            entry=entry,
            saved=saved,
            save_path=save_path,
            message=message,
        )

    @classmethod
    def fail(cls, message: str) -> "ImageResult":
        return cls(success=False, message=message)


@dataclass
class GenerationFailure(Exception):
    """
    Raised internally when the model provider cannot generate/edit an image.
    The ``reason`` is surfaced to the user (Req 15.6).
    """

    reason: str

    def __str__(self) -> str:
        return self.reason


# ---------------------------------------------------------------------------
# Reference resolution helpers (15.2, 15.3)
# ---------------------------------------------------------------------------

# Patterns that indicate the user wants to edit the most recent image.
_LAST_IMAGE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(this|that|it|the image|the picture|the last|the latest|the current)\b", re.I),
    re.compile(r"\b(above|previous|recent)\b", re.I),
]

# Patterns that hint at an ordinal reference like "the second image",
# "image number 2", or "the first one".
_ORDINAL_WORDS: dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
}


def _resolve_target_ref(
    instruction: str,
    history: list[ImageEntry],
    explicit_id: str | None,
) -> ImageEntry | None:
    """
    Resolve which session image the user is referring to.

    Resolution order (Req 15.2, 15.3):
    1. ``explicit_id`` — caller supplies an exact image ID (highest priority).
    2. Numeric index in instruction text, e.g. "edit image 2".
    3. Ordinal word, e.g. "edit the first image".
    4. Any reference to "the last image" / "that image" / etc. → most recent.
    5. No history → None.

    Parameters
    ----------
    instruction : str
        The user's edit instruction text.
    history : list[ImageEntry]
        Ordered list of session images (oldest first).
    explicit_id : str | None
        A specific image ID to use if supplied by the caller.

    Returns
    -------
    ImageEntry | None
        The resolved target image, or ``None`` when the history is empty
        or the reference cannot be resolved.
    """
    if not history:
        return None

    # 1. Explicit ID takes priority (Req 15.3)
    if explicit_id is not None:
        for entry in history:
            if entry.id == explicit_id:
                return entry
        # ID not found — fall through to instruction-based resolution

    lower = instruction.lower()

    # 2. Numeric index: "image 3", "picture 2", etc.
    num_match = re.search(r'\b(?:image|picture|photo)\s+#?(\d+)\b', lower)
    if num_match:
        idx = int(num_match.group(1)) - 1  # 1-based → 0-based
        if 0 <= idx < len(history):
            return history[idx]

    # 3. Ordinal words: "first image", "the second one", etc.
    for word, ordinal in _ORDINAL_WORDS.items():
        if word in lower:
            idx = ordinal - 1
            if 0 <= idx < len(history):
                return history[idx]

    # 4. Any vague reference → most recent image (Req 15.2)
    for pattern in _LAST_IMAGE_PATTERNS:
        if pattern.search(lower):
            return history[-1]

    # 5. Default to most recent when instruction gives no clue (Req 15.2)
    return history[-1]


# ---------------------------------------------------------------------------
# Image data helpers
# ---------------------------------------------------------------------------


def _extract_image_bytes(provider_response: Any) -> bytes:
    """
    Extract raw image bytes from a Model Provider response.

    Handles three common response shapes returned by stubs and real APIs:
    - ``{"image_data": bytes}``
    - ``{"image_b64": "<base64-string>"}``
    - Bare ``bytes`` object.

    Raises :class:`GenerationFailure` when the response contains no
    recognisable image payload.
    """
    if isinstance(provider_response, bytes):
        return provider_response

    if isinstance(provider_response, dict):
        # Raw bytes field
        if isinstance(provider_response.get("image_data"), bytes):
            return provider_response["image_data"]

        # Base-64 encoded field
        b64 = provider_response.get("image_b64") or provider_response.get("image_base64")
        if isinstance(b64, str):
            try:
                return base64.b64decode(b64)
            except Exception as exc:
                raise GenerationFailure(
                    f"Model provider returned malformed base64 image data: {exc}"
                ) from exc

        # Stub response — build a minimal placeholder so tests work without
        # a real diffusion model.
        if provider_response.get("stub"):
            # Return 1-pixel PNG (smallest valid PNG)
            _ONE_PIXEL_PNG = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
                b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
                b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            return _ONE_PIXEL_PNG

    raise GenerationFailure(
        "Model provider returned an unrecognised image response. "
        "Expected bytes or a dict with 'image_data' or 'image_b64'."
    )


# ---------------------------------------------------------------------------
# ImageStudio
# ---------------------------------------------------------------------------


class ImageStudio:
    """
    Image generation and editing subsystem.

    Thread-safety
    -------------
    The session history is protected by a ``threading.Lock`` so that
    concurrent IPC callbacks (rare in practice) do not race.

    Parameters
    ----------
    image_provider : ModelProvider | None
        A Model Provider for the ``Capability.IMAGE`` capability.  When
        ``None``, a stub that returns a placeholder response is used —
        useful for unit tests without a real diffusion model.
    save_dir : Path | None
        Directory where generated images are saved.  Defaults to
        ``~/Pictures/HAKI``.  Created on first save if it does not exist.

    Design: Image_Studio.
    Requirements: 15.1–15.6.
    """

    def __init__(
        self,
        image_provider: "ModelProvider | None" = None,
        save_dir: Path | None = None,
    ) -> None:
        self._provider = image_provider
        self._save_dir: Path = save_dir or _DEFAULT_SAVE_DIR
        self._history: list[ImageEntry] = []
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> ImageResult:
        """
        Generate a new image from *prompt* via the image Model Provider,
        add it to the session history, display it, and save it to the
        designated folder (Req 15.1, 15.4).

        On generation failure: inform the user with a reason (Req 15.6).
        On save failure: keep the image in-session and inform the user
        (Req 15.5).

        Parameters
        ----------
        prompt : str
            Natural-language description of the image to generate.

        Returns
        -------
        ImageResult
            ``ImageResult.ok(...)`` with the image entry and save status.
            ``ImageResult.fail(reason)`` when generation fails (Req 15.6).
        """
        logger.debug("ImageStudio.generate: prompt=%r", prompt[:80])

        # --- Step 1: invoke the image model (Req 15.1) ---
        try:
            raw_response = self._invoke_provider(prompt, operation="generate")
            image_bytes = _extract_image_bytes(raw_response)
        except GenerationFailure as exc:
            # Req 15.6: inform user with reason; do not silently fail
            reason = (
                f"I couldn't generate the image. Reason: {exc}"
            )
            logger.warning("ImageStudio.generate failed: %s", exc)
            return ImageResult.fail(reason)
        except Exception as exc:
            reason = (
                f"Image generation encountered an unexpected error: {exc}"
            )
            logger.error("ImageStudio.generate unexpected error: %r", exc)
            return ImageResult.fail(reason)

        # --- Step 2: build the image entry and add to session history ---
        entry = self._make_entry(prompt, image_bytes)

        # --- Step 3: attempt to save to disk (Req 15.4) ---
        saved, save_path, save_msg = self._save_image(entry)

        if saved:
            entry.saved_path = save_path
            message = (
                f"Here's your image! I've saved it to {save_path} "
                f"({entry.display_label})."
            )
        else:
            # Req 15.5: keep in-session, inform user of save failure
            message = (
                f"Here's your image! However, I couldn't save it to the folder: "
                f"{save_msg}. The image is available in this session."
            )

        return ImageResult.ok(
            entry,
            saved=saved,
            save_path=save_path,
            message=message,
        )

    def edit(
        self,
        instruction: str,
        target_ref: str | None = None,
    ) -> ImageResult:
        """
        Edit a session image using *instruction* and the image Model Provider
        (Req 15.2, 15.3).

        Edit-target resolution (Req 15.2, 15.3):
        - If *target_ref* is a known image ID → use that image (15.3).
        - Otherwise resolve from *instruction* text:
          - Ordinal / numeric reference → that specific image (15.3).
          - "this" / "that" / "the last" / unspecified → most recent (15.2).
        - If the history is empty → fail with a clear message.

        On generation failure: inform the user with a reason (Req 15.6).
        On save failure: keep the image in-session and inform (Req 15.5).

        Parameters
        ----------
        instruction : str
            Natural-language edit instruction, e.g. "make it brighter".
        target_ref : str | None
            Optional explicit image ID to edit (used by UI when the user
            selects a specific image from the history panel).

        Returns
        -------
        ImageResult
            ``ImageResult.ok(...)`` with the edited image entry.
            ``ImageResult.fail(reason)`` when target resolution or
            generation fails.
        """
        logger.debug(
            "ImageStudio.edit: instruction=%r, target_ref=%r",
            instruction[:80],
            target_ref,
        )

        # --- Step 1: resolve edit target (Req 15.2, 15.3) ---
        with self._lock:
            history_snapshot = list(self._history)

        target = _resolve_target_ref(instruction, history_snapshot, target_ref)
        if target is None:
            return ImageResult.fail(
                "There are no images in the current session to edit. "
                "Please generate an image first."
            )

        # --- Step 2: invoke the image model for editing (Req 15.2) ---
        edit_prompt = f"{instruction} [editing: {target.prompt}]"
        try:
            raw_response = self._invoke_provider(edit_prompt, operation="edit")
            image_bytes = _extract_image_bytes(raw_response)
        except GenerationFailure as exc:
            reason = f"I couldn't edit the image. Reason: {exc}"
            logger.warning("ImageStudio.edit failed: %s", exc)
            return ImageResult.fail(reason)
        except Exception as exc:
            reason = f"Image editing encountered an unexpected error: {exc}"
            logger.error("ImageStudio.edit unexpected error: %r", exc)
            return ImageResult.fail(reason)

        # --- Step 3: build entry for the edited image ---
        entry = self._make_entry(edit_prompt, image_bytes)

        # --- Step 4: attempt to save (Req 15.4) ---
        saved, save_path, save_msg = self._save_image(entry)

        if saved:
            entry.saved_path = save_path
            message = (
                f"Done! I've edited the image and saved it to {save_path} "
                f"({entry.display_label})."
            )
        else:
            # Req 15.5: keep in-session, inform user of save failure
            message = (
                f"Done! I've edited the image, but couldn't save it to the folder: "
                f"{save_msg}. The edited image is available in this session."
            )

        return ImageResult.ok(
            entry,
            saved=saved,
            save_path=save_path,
            message=message,
        )

    # ------------------------------------------------------------------
    # Session history access
    # ------------------------------------------------------------------

    def history(self) -> list[ImageEntry]:
        """Return a snapshot of the session image history (oldest first)."""
        with self._lock:
            return list(self._history)

    def last_image(self) -> ImageEntry | None:
        """Return the most recently generated/edited image, or ``None``."""
        with self._lock:
            return self._history[-1] if self._history else None

    def get_by_id(self, image_id: str) -> ImageEntry | None:
        """Return the session image with *image_id*, or ``None``."""
        with self._lock:
            for entry in self._history:
                if entry.id == image_id:
                    return entry
        return None

    def clear_history(self) -> None:
        """Clear the in-session image history (does not affect saved files)."""
        with self._lock:
            self._history.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _invoke_provider(self, prompt: str, operation: str) -> Any:
        """
        Invoke the image Model Provider.

        When no real provider is configured a lightweight stub response
        is returned so that unit tests can exercise the full
        generate/edit/save/history logic without a live model.

        Raises
        ------
        GenerationFailure
            Re-raised from the provider with a human-readable reason.
        """
        if self._provider is None:
            # Stub path — return a response that _extract_image_bytes handles
            logger.debug(
                "ImageStudio._invoke_provider: no provider configured, using stub"
            )
            return {"stub": True, "operation": operation, "prompt": prompt}

        try:
            return self._provider.invoke(prompt)
        except Exception as exc:
            raise GenerationFailure(str(exc)) from exc

    def _make_entry(self, prompt: str, image_bytes: bytes) -> ImageEntry:
        """Build a new :class:`ImageEntry` and append it to the session history."""
        with self._lock:
            position = len(self._history) + 1
            entry = ImageEntry(
                id=str(uuid.uuid4()),
                prompt=prompt,
                data=image_bytes,
                display_label=f"Image {position}",
            )
            self._history.append(entry)
        return entry

    def _save_image(
        self, entry: ImageEntry
    ) -> tuple[bool, Path | None, str]:
        """
        Save *entry.data* to the designated save directory (Req 15.4).

        Returns
        -------
        tuple[bool, Path | None, str]
            (saved, save_path, message)
            - ``saved`` — True when the file was written successfully.
            - ``save_path`` — absolute path on disk, or None on failure.
            - ``message`` — confirmation or error description.
        """
        try:
            self._save_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return False, None, f"could not create save directory: {exc}"

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        # Sanitize ID to first 8 chars for a readable filename
        filename = f"haki_{timestamp}_{entry.id[:8]}.png"
        dest = self._save_dir / filename

        try:
            dest.write_bytes(entry.data)
        except OSError as exc:
            return False, None, f"disk write failed: {exc}"

        logger.info("ImageStudio: saved image to %s", dest)
        return True, dest, f"saved to {dest}"
