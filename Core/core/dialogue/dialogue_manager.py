"""
Dialogue_Manager — interactive slot filling and ambiguity resolution.

Pauses plan execution when required slots are missing or ambiguous,
asks the user clarifying questions, resumes on answer, and handles
decline/default/abandon paths.

Implements memory-first slot assessment (Task 25.1):
  - assess() identifies missing slots for a request (Req 23.1, 23.4)
  - fill_from_memory() resolves missing slots from Memory_Brain first,
    before any user question is asked (Req 23.2)
  - Resolved slots are tracked per-session so they are never re-asked
    (Req 23.4)
  - Memory-resolved values are confirmed with the user before use (Req 23.5)

Mid-task pause/resume (Task 25.2):
  - on_decision_point() pauses execution at a mid-task decision point,
    retaining all task state, and resumes after the user answers (Req 23.3, 23.9)
  - on_decline() handles user decline with or without a default value
    (Req 23.6, 23.7)
  - present_options() presents gathered candidates without auto-selecting
    (Req 23.8)

Design: Dialogue_Manager.
Requirements: 23.1–23.9.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


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


@dataclass
class MemoryFillResult:
    """
    Outcome of :meth:`DialogueManager.fill_from_memory`.

    Attributes
    ----------
    resolved:
        Slot names mapped to the values found in memory.
    still_missing:
        Slot names that could not be resolved from memory and must be
        asked of the user.
    """

    resolved: dict[str, Any] = field(default_factory=dict)
    still_missing: list[str] = field(default_factory=list)


@dataclass
class PausePoint:
    """
    Captures the full state needed to resume after a mid-task pause.

    Created by :meth:`DialogueManager.on_decision_point` and stored as
    :attr:`DialogueManager._current_pause` until execution resumes
    (Req 23.3, 23.9).

    Attributes
    ----------
    step_id:
        The step that triggered the pause.
    question:
        The clarifying question or decision being presented to the user.
    slot_name:
        The slot awaiting a value, or ``None`` when the pause is not
        slot-linked.
    completed_steps:
        Steps that were completed before this pause.  The caller MUST
        NOT re-execute any of these steps when resuming (Req 23.9).
    task_state:
        Arbitrary task state to retain across the pause so execution can
        continue exactly where it left off (Req 23.3).
    """

    step_id: str
    question: str
    slot_name: str | None
    completed_steps: list[str]
    task_state: dict


@dataclass
class DeclineResult:
    """
    Outcome of :meth:`DialogueManager.on_decline`.

    Attributes
    ----------
    action:
        ``"use_default"`` when a default value is available (Req 23.6),
        ``"abandon_step"`` when no default exists (Req 23.7).
    value:
        The default value to use when *action* is ``"use_default"``,
        ``None`` otherwise.
    message:
        Human-readable description of the outcome — what default was
        applied, or which step is being abandoned.
    """

    action: str  # "use_default" | "abandon_step"
    value: Any | None
    message: str


@dataclass
class CandidatePresentation:
    """
    Result of :meth:`DialogueManager.present_options` when no callback
    is supplied.

    ``auto_selected`` is always ``False`` — the Dialogue_Manager never
    picks a candidate automatically (Req 23.8).

    Attributes
    ----------
    candidates:
        All candidates passed to :meth:`present_options`.
    selection:
        The user's chosen candidate, or ``None`` when no callback was
        provided and the system is awaiting a selection.
    auto_selected:
        Always ``False``.  Included so callers can assert that no
        auto-selection occurred.
    """

    candidates: list
    selection: Any | None = None
    auto_selected: bool = False


#: Async callable used to present a question to the user and receive their
#: answer.  Receives a single question string and returns the answer string.
AskCallbackAsync = Callable[[str], Awaitable[str]]

#: Sync callable alternative.
AskCallbackSync = Callable[[str], str]


class DialogueManager:
    """
    Conducts interactive clarification before and during task execution.

    Implements memory-first slot assessment (Task 25.1):

    1. :meth:`assess` checks which required slots are already resolved
       in the current session or available in *request* context
       (Req 23.1, 23.4).
    2. :meth:`fill_from_memory` attempts to resolve remaining missing
       slots from the Memory_Brain before any user question is asked
       (Req 23.2).  Memory-resolved values are **not** automatically
       marked resolved — callers should confirm with the user (Req 23.5)
       and then call :meth:`mark_resolved`.
    3. :meth:`ask` queues questions only for slots that are still missing
       after memory-fill (Req 23.1, 23.3).  Already-resolved slots are
       never re-asked (Req 23.4).
    4. :meth:`mark_resolved` / :meth:`is_resolved` / :meth:`get_resolved`
       maintain a session-scoped resolved-slot registry that gates all
       question logic.
    5. :meth:`reset_session` clears session state for a new interaction.

    Parameters
    ----------
    memory_brain:
        Optional :class:`~core.memory.MemoryBrain` instance.  When
        provided, :meth:`fill_from_memory` queries it to auto-fill slots.
    ask_callback:
        Async callable ``(question: str) → answer: str`` used by
        :meth:`ask` to present a question to the user and collect their
        response.  When ``None``, :meth:`ask` records questions as pending
        but returns empty answers (suitable for testing the gating logic
        without real I/O).
    sync_ask_callback:
        Sync alternative to *ask_callback*.  Cannot be supplied together
        with *ask_callback*.
    """

    def __init__(
        self,
        memory_brain: Any | None = None,
        ask_callback: AskCallbackAsync | None = None,
        sync_ask_callback: AskCallbackSync | None = None,
    ) -> None:
        if ask_callback is not None and sync_ask_callback is not None:
            raise ValueError(
                "Provide either ask_callback or sync_ask_callback, not both."
            )

        self._memory_brain = memory_brain
        self._async_ask = ask_callback
        self._sync_ask = sync_ask_callback

        # Session-scoped resolved-slot registry (Req 23.4)
        # Maps slot_name -> resolved value.
        self._resolved_slots: dict[str, Any] = {}

        # Pending questions registered by the latest ask() call.
        # Maps question_text -> slot_name (None when not slot-linked).
        self._pending_questions: list[str] = []

        # Mid-task pause state (Req 23.3, 23.9)
        self._current_pause: PausePoint | None = None
        self._pause_event: asyncio.Event = asyncio.Event()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        """
        Clear all session state for a new interaction.

        After this call :meth:`is_resolved` returns ``False`` for all
        slots, and :meth:`ask` will present questions as if this is a
        fresh conversation.
        """
        self._resolved_slots.clear()
        self._pending_questions.clear()
        logger.debug("DialogueManager: session reset.")

    # ------------------------------------------------------------------
    # Core assessment (Req 23.1, 23.4)
    # ------------------------------------------------------------------

    def assess(self, request: str, needed_slots: list[str]) -> SlotFillResult:
        """
        Determine whether *needed_slots* are satisfied for *request*.

        Slots that are already resolved in this session are counted as
        satisfied without any memory or user-question lookup (Req 23.4).

        Parameters
        ----------
        request:
            The natural-language request string.  Passed through for
            context but not parsed in this implementation — slot
            resolution is handled by :meth:`fill_from_memory`.
        needed_slots:
            Names of slots required before execution can proceed.

        Returns
        -------
        SlotFillResult
            ``sufficient=True`` when all slots are resolved.  Otherwise
            ``sufficient=False`` with ``missing`` listing unresolved slot
            names and ``resolved`` holding the session-resolved values.
        """
        resolved: dict[str, Any] = {}
        still_missing: list[str] = []

        for slot in needed_slots:
            if self.is_resolved(slot):
                # Slot already resolved in current session — never re-ask (Req 23.4)
                resolved[slot] = self.get_resolved(slot)
            else:
                still_missing.append(slot)

        return SlotFillResult(
            sufficient=len(still_missing) == 0,
            missing=still_missing,
            resolved=resolved,
        )

    # ------------------------------------------------------------------
    # Memory-first slot resolution (Req 23.2)
    # ------------------------------------------------------------------

    def fill_from_memory(
        self,
        missing_slots: list[str],
        memory_brain: Any | None = None,
    ) -> MemoryFillResult:
        """
        Attempt to resolve *missing_slots* from the Memory_Brain (Req 23.2).

        For each slot that is not yet resolved in this session, the method
        queries the Memory_Brain using the slot name as a retrieval query.
        If at least one matching note is found, the body of the best-match
        note is used as the slot value (pending user confirmation per
        Req 23.5).

        **Important:** memory-resolved slots are returned in
        ``MemoryFillResult.resolved`` but are NOT automatically marked as
        resolved in the session registry.  The caller is responsible for
        presenting the resolved value to the user for confirmation
        (Req 23.5) and then calling :meth:`mark_resolved` for each
        confirmed slot.

        Parameters
        ----------
        missing_slots:
            Slot names that :meth:`assess` determined are not yet resolved.
        memory_brain:
            Optional override for the Memory_Brain instance.  When
            ``None``, falls back to the one supplied at construction time.

        Returns
        -------
        MemoryFillResult
            ``resolved`` maps slot names to values found in memory.
            ``still_missing`` lists slots that could not be resolved.
        """
        brain = memory_brain if memory_brain is not None else self._memory_brain

        resolved: dict[str, Any] = {}
        still_missing: list[str] = []

        for slot in missing_slots:
            # Skip slots already resolved in this session (Req 23.4)
            if self.is_resolved(slot):
                resolved[slot] = self.get_resolved(slot)
                continue

            if brain is not None:
                value = self._query_memory_for_slot(brain, slot)
                if value is not None:
                    resolved[slot] = value
                    logger.debug(
                        "DialogueManager: slot '%s' resolved from memory: %r",
                        slot,
                        value,
                    )
                else:
                    still_missing.append(slot)
            else:
                still_missing.append(slot)

        return MemoryFillResult(resolved=resolved, still_missing=still_missing)

    # ------------------------------------------------------------------
    # Slot registry (Req 23.4)
    # ------------------------------------------------------------------

    def mark_resolved(self, slot: str, value: Any) -> None:
        """
        Record that *slot* has been resolved to *value* in this session.

        After this call :meth:`is_resolved` returns ``True`` for *slot*
        and :meth:`assess` will not include it in the ``missing`` list.
        This is the write side of the no-re-ask guarantee (Req 23.4).

        Parameters
        ----------
        slot:
            The slot name to mark resolved.
        value:
            The confirmed resolved value (Req 23.5 — caller confirms
            before calling this method).
        """
        self._resolved_slots[slot] = value
        logger.debug("DialogueManager: slot '%s' marked resolved: %r", slot, value)

    def is_resolved(self, slot: str) -> bool:
        """
        Return ``True`` if *slot* has been resolved in this session (Req 23.4).

        Parameters
        ----------
        slot:
            The slot name to check.
        """
        return slot in self._resolved_slots

    def get_resolved(self, slot: str) -> Any:
        """
        Return the resolved value for *slot* (Req 23.4).

        Parameters
        ----------
        slot:
            A slot name that :meth:`is_resolved` returns ``True`` for.

        Returns
        -------
        Any
            The resolved value, or ``None`` if the slot is not resolved.
        """
        return self._resolved_slots.get(slot)

    # ------------------------------------------------------------------
    # User interaction (Req 23.1, 23.3)
    # ------------------------------------------------------------------

    async def ask(self, questions: list[str]) -> dict[str, str]:
        """
        Present clarifying questions to the user and collect answers
        (Req 23.1, 23.3).

        Pauses execution until answers are received.  Slots that are
        already resolved are skipped — no question is generated for them
        (Req 23.4).

        Parameters
        ----------
        questions:
            List of question strings to pose to the user.

        Returns
        -------
        dict[str, str]
            Maps each question to the user's answer.  When no
            *ask_callback* is configured (test/headless mode), returns an
            empty dict and records questions as pending.
        """
        if not questions:
            return {}

        # Record pending questions (gating: execution must not proceed until
        # answers are returned, Req 23.1).
        self._pending_questions = list(questions)

        answers: dict[str, str] = {}

        if self._async_ask is None and self._sync_ask is None:
            # No I/O backend — headless mode (testing gating logic).
            logger.debug(
                "DialogueManager.ask: no callback configured; "
                "recording %d pending question(s).",
                len(questions),
            )
            return answers

        for question in questions:
            try:
                answer = await self._invoke_ask(question)
                answers[question] = answer
            except Exception as exc:
                logger.error(
                    "DialogueManager.ask: error collecting answer for '%s': %s",
                    question,
                    exc,
                )
                answers[question] = ""

        # Clear pending questions once answered
        self._pending_questions = []
        return answers

    def on_decline(self, slot: str, has_default: bool, default_value: Any = None) -> DeclineResult:
        """
        Handle a user decline for a required slot (Req 23.6, 23.7).

        Parameters
        ----------
        slot:
            The slot name the user declined to fill.
        has_default:
            Whether a default value is available for this slot.
        default_value:
            The default value to use when *has_default* is ``True``.
            Ignored when *has_default* is ``False``.

        Returns
        -------
        DeclineResult
            ``action="use_default"`` with the applied *default_value* and
            a descriptive message when *has_default* is ``True`` (Req 23.6).
            ``action="abandon_step"`` with ``value=None`` and a message
            explaining the dependent step is abandoned when *has_default*
            is ``False`` (Req 23.7).
        """
        if has_default:
            logger.debug(
                "DialogueManager.on_decline: slot '%s' declined; "
                "applying default value %r (Req 23.6).",
                slot,
                default_value,
            )
            return DeclineResult(
                action="use_default",
                value=default_value,
                message=(
                    f"No value provided for '{slot}'; "
                    f"proceeding with the default value: {default_value!r}."
                ),
            )
        else:
            logger.debug(
                "DialogueManager.on_decline: slot '%s' declined with no default; "
                "abandoning dependent step (Req 23.7).",
                slot,
            )
            return DeclineResult(
                action="abandon_step",
                value=None,
                message=(
                    f"No value provided for '{slot}' and no default is available; "
                    f"only the step that requires '{slot}' (and its dependents) "
                    f"will be abandoned — all other independent steps will proceed."
                ),
            )

    async def on_decision_point(
        self,
        step_id: str,
        question: str,
        slot_name: str | None,
        completed_steps: list[str],
        task_state: dict,
    ) -> tuple[str, list[str]]:
        """
        Pause execution at a mid-task decision point and resume after the
        user answers (Req 23.3, 23.9).

        Creates a :class:`PausePoint`, signals the pause, asks the
        *question* via the configured callback, then clears the pause
        state before returning.

        Parameters
        ----------
        step_id:
            The step that triggered the decision point.
        question:
            The clarifying question to present to the user.
        slot_name:
            The slot awaiting a value, or ``None``.
        completed_steps:
            Steps already completed before this pause.  Returned to the
            caller so execution can skip them on resume (Req 23.9).
        task_state:
            Arbitrary task state retained across the pause (Req 23.3).

        Returns
        -------
        tuple[str, list[str]]
            ``(answer, completed_steps)`` — the user's answer string and
            the list of step IDs that must NOT be re-executed (Req 23.9).
        """
        # Build and store the pause point (Req 23.3)
        self._current_pause = PausePoint(
            step_id=step_id,
            question=question,
            slot_name=slot_name,
            completed_steps=list(completed_steps),
            task_state=dict(task_state),
        )
        # Signal that execution is paused
        self._pause_event.set()
        logger.debug(
            "DialogueManager: paused at step '%s' — question: %r (Req 23.3).",
            step_id,
            question,
        )

        # Ask the user (pauses execution here until answer is received)
        answers = await self.ask([question])
        answer = answers.get(question, "")

        # Resume: clear the pause state (Req 23.9)
        retained_completed = list(self._current_pause.completed_steps)
        self._current_pause = None
        self._pause_event.clear()
        logger.debug(
            "DialogueManager: resuming from step '%s'; "
            "skipping %d already-completed step(s) (Req 23.9).",
            step_id,
            len(retained_completed),
        )

        return answer, retained_completed

    def present_options(
        self,
        candidates: list[Any],
        present_callback: Callable[[list[Any]], Any] | None = None,
    ) -> CandidatePresentation | Any:
        """
        Present multiple gathered candidates to the user for selection
        (Req 23.8).

        Never auto-selects a candidate — only the user may choose.

        Parameters
        ----------
        candidates:
            Ordered list of candidate values to present.
        present_callback:
            Optional ``(candidates) → selection`` callable.  When
            provided, it is called to obtain the user's choice and that
            choice is returned directly.  When ``None``, a
            :class:`CandidatePresentation` with ``selection=None`` and
            ``auto_selected=False`` is returned, indicating that the
            system is waiting for a user selection (Req 23.8).

        Returns
        -------
        CandidatePresentation | Any
            A :class:`CandidatePresentation` (``selection=None``,
            ``auto_selected=False``) when no callback is given, or the
            user's selection returned by *present_callback*.
        """
        if present_callback is not None:
            logger.debug(
                "DialogueManager.present_options: presenting %d candidate(s) "
                "via callback (Req 23.8).",
                len(candidates),
            )
            selection = present_callback(candidates)
            return selection

        # No callback: return a CandidatePresentation awaiting user input.
        # auto_selected is always False — never auto-pick (Req 23.8).
        logger.debug(
            "DialogueManager.present_options: no callback; "
            "returning CandidatePresentation with selection=None (Req 23.8).",
        )
        return CandidatePresentation(
            candidates=list(candidates),
            selection=None,
            auto_selected=False,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pending_questions(self) -> list[str]:
        """
        Questions registered by the last :meth:`ask` call that have not
        yet been answered.  Non-empty while execution is gated waiting
        for user input (Req 23.1).
        """
        return list(self._pending_questions)

    @property
    def resolved_slots(self) -> dict[str, Any]:
        """Read-only view of all slots resolved in this session."""
        return dict(self._resolved_slots)

    @property
    def is_paused(self) -> bool:
        """
        ``True`` when a :class:`PausePoint` is active (i.e. execution is
        paused at a mid-task decision point, Req 23.3).
        """
        return self._current_pause is not None

    @property
    def current_pause(self) -> PausePoint | None:
        """
        The active :class:`PausePoint`, or ``None`` when execution is
        not paused (Req 23.3, 23.9).
        """
        return self._current_pause

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _query_memory_for_slot(self, brain: Any, slot: str) -> Any | None:
        """
        Query *brain* for a value matching *slot* (Req 23.2).

        Uses the slot name as the retrieval query.  Returns the body of
        the best-match note when found, ``None`` otherwise.

        Parameters
        ----------
        brain:
            A :class:`~core.memory.MemoryBrain` instance (typed as ``Any``
            to avoid a hard import cycle).
        slot:
            The slot name to use as the query term.
        """
        try:
            notes = brain.retrieve(slot, k=1)
            if notes:
                return notes[0].body
        except Exception as exc:
            logger.warning(
                "DialogueManager: memory query for slot '%s' raised: %s",
                slot,
                exc,
            )
        return None

    async def _invoke_ask(self, question: str) -> str:
        """
        Dispatch a single question to the configured ask callback.

        Handles both async and sync callbacks (the sync variant is run in
        a thread executor to avoid blocking the event loop).
        """
        if self._async_ask is not None:
            return await self._async_ask(question)

        # Sync callback — run in executor to avoid blocking
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_ask, question)
