"""Internal development replacement gate.

Set ``HAKI_VOICE_DEV_REPLACEMENT=1`` in the process environment to activate the
new local voice path (``VoiceUnixServer`` + ``VoiceSessionPipeline``).

Gate rules:
- When enabled: only the new local path is active; no legacy voice route is
  selected, wrapped, or fallen back to.
- When disabled: the existing non-voice IPC handlers and the legacy voice path
  are left completely untouched.
- Non-voice IPC handlers are always preserved regardless of gate state.

This module contains no archive imports and no references to legacy voice
components.  It is safe to import from anywhere in ``core.voice`` or
``core.ipc``.

Design: §11 (safe migration sequence, step 3).
Requirements: 1.5–1.6.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Public gate constant — evaluated once at import time so the gate state is
# stable for the lifetime of the process (no hot-reload for security).
# ---------------------------------------------------------------------------

VOICE_REPLACEMENT_GATE_ENABLED: bool = os.getenv("HAKI_VOICE_DEV_REPLACEMENT", "0") == "1"
"""True only when ``HAKI_VOICE_DEV_REPLACEMENT=1`` is set in the environment.

The internal development replacement gate activates the new local voice path.
It must not be used to select any legacy voice component.  The gate is
intentionally a *replacement* switch: when it is on, *only* the new path runs.
"""


def gate_enabled() -> bool:
    """Return the stable gate state for callers that need a callable interface."""
    return VOICE_REPLACEMENT_GATE_ENABLED


__all__ = [
    "VOICE_REPLACEMENT_GATE_ENABLED",
    "gate_enabled",
]
