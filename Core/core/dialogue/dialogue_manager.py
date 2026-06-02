"""
Dialogue_Manager — interactive slot filling and ambiguity resolution.

Pauses plan execution when required slots are missing or ambiguous,
asks the user clarifying questions, resumes on answer, and handles
decline/default/abandon paths.

Design: Dialogue_Manager.
Requirements: 23.1–23.9.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SlotFillResult:
    """
    Outcome of a slot-fill assessment.

    If *sufficient* is True, execution can proceed.  Otherwise *missing*
    lists the slot names that still need values.
    """

    sufficient: bool
    missing: list[str] = field(default_factory=list)
    resolved: dict[str, Any] = field(default_factory=dict)  # slots filled from Memory_Brain


class DialogueManager:
    """
    Conducts interactive clarification before and during task execution.

    This is a stub implementation.  The full implementation (Tasks 11, 23)
    adds LLM-based ambiguity detection, Memory_Brain-backed auto-fill,
    pause/resume integration with the Execution_Engine, and the
    options-presentation path for ambiguous contacts.
    """

    def __init__(self, memory_brain: Any | None = None) -> None:
        self._memory_brain = memory_brain

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(self, request: str, needed_slots: list[str]) -> SlotFillResult:
        """
        Determine whether *needed_slots* are satisfied for *request*.

        Attempts to fill slots from Memory_Brain first (Req 23.2).  Any
        slots that cannot be auto-filled are returned in *missing*.

        Stub: no real memory lookup; all slots returned as missing.
        """
        resolved: dict[str, Any] = {}
        still_missing: list[str] = []

        for slot in needed_slots:
            value = self._fill_from_memory(slot, request)
            if value is not None:
                resolved[slot] = value
            else:
                still_missing.append(slot)

        return SlotFillResult(
            sufficient=len(still_missing) == 0,
            missing=still_missing,
            resolved=resolved,
        )

    def ask(self, questions: list[str]) -> dict[str, str]:
        """
        Present clarifying questions to the user and collect answers (Req 23.1, 23.3).

        Stub: returns empty dict; real IPC/Voice interaction added in Task 11.
        """
        return {}

    def on_decline(self, slot: str, has_default: bool) -> str:
        """
        Handle a user decline for a required slot (Req 23.6, 23.7).

        Returns "use_default" if a default exists, "abandon_step" otherwise.
        """
        return "use_default" if has_default else "abandon_step"

    def present_options(self, candidates: list[Any]) -> Any | None:
        """
        Show multiple candidates to the user and return their choice (Req 23.8).

        Never auto-picks — only the user may select.
        Stub: returns None (no selection without real UI/voice).
        """
        return None

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _fill_from_memory(self, slot: str, context: str) -> Any | None:
        """Stub: query Memory_Brain for a slot value. Returns None until Task 23."""
        return None
