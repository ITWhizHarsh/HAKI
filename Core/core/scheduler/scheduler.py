"""
Scheduler — severity-based reminder scheduling and issuance.

Design: Scheduler.
Requirements: 12.1 – 12.11.

Responsibilities
----------------
* Task creation with mandatory Severity assignment (12.1, 12.11).
* Reminder-offset computation relative to task due date using Clock (12.2–12.4,
  12.7, 12.8, 12.10).
* Dual-channel issuance (VOICE + NOTIFICATION) with per-reminder failure
  isolation (12.6, 12.9).
* Birthday day-of confirmation prompt (12.5).

Architecture notes
------------------
The Scheduler is a pure-Python, synchronous class with injected adapters for
the Clock, Voice_Engine, and Notification system so the scheduling logic can
be tested without side effects.

Voice and Notification adapters are simple callables:
    voice_adapter(task: Task, reminder: Reminder) -> None  (raises on failure)
    notification_adapter(task: Task, reminder: Reminder) -> None  (raises on failure)

The ``on_failure`` callback is called whenever a reminder channel fails or
a reminder cannot be issued:
    on_failure(task: Task, reminder: Reminder, reason: str) -> None

The ``on_notification`` callback is called whenever the user should be
notified of a non-critical event (e.g., default severity was applied):
    on_notification(message: str) -> None

Testing seams
-------------
* ``clock`` — inject a Clock instance with ``_override_now`` for deterministic
  time.
* ``voice_adapter`` / ``notification_adapter`` — inject fakes/mocks.
* ``on_failure`` — inject a collector callable.
* ``on_notification`` — inject a collector callable.
* ``_custom_policies`` — dict[Severity, ReminderPolicy]; mutate directly in
  tests to simulate user-configured custom policies.
"""

from __future__ import annotations

import datetime
import logging
from typing import Callable, Dict, List, Optional

from ..clock.clock import Clock, ClockResult, ClockUnavailable
from .models import (
    DEFAULT_POLICIES,
    Prereq,
    Reminder,
    ReminderChannel,
    ReminderPolicy,
    ReminderState,
    Severity,
    Task,
    TaskSource,
    TaskStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

VoiceAdapter = Callable[["Task", "Reminder"], None]
NotificationAdapter = Callable[["Task", "Reminder"], None]
FailureCallback = Callable[["Task", "Reminder", str], None]
NotificationCallback = Callable[[str], None]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class Scheduler:
    """Creates tasks, computes reminder schedules, and issues reminders.

    Parameters
    ----------
    clock:
        The Clock instance used for all current-time queries (14.2).
        Defaults to a plain ``Clock()`` if not supplied.
    voice_adapter:
        Callable that issues a reminder via the Voice_Engine (12.6).
        Signature: ``(task, reminder) -> None``.  May raise to simulate failure.
    notification_adapter:
        Callable that issues a reminder via the on-screen notification system
        (12.6).  Signature: ``(task, reminder) -> None``.  May raise to
        simulate failure.
    on_failure:
        Called when a reminder cannot be issued (12.9).
        Signature: ``(task, reminder, reason) -> None``.
    on_notification:
        Called to surface informational messages to the user (e.g., default
        severity applied) (12.11).
        Signature: ``(message: str) -> None``.
    """

    def __init__(
        self,
        clock: Optional[Clock] = None,
        voice_adapter: Optional[VoiceAdapter] = None,
        notification_adapter: Optional[NotificationAdapter] = None,
        on_failure: Optional[FailureCallback] = None,
        on_notification: Optional[NotificationCallback] = None,
    ) -> None:
        self._clock: Clock = clock or Clock()
        self._voice_adapter: VoiceAdapter = voice_adapter or _noop_adapter
        self._notification_adapter: NotificationAdapter = (
            notification_adapter or _noop_adapter
        )
        self._on_failure: FailureCallback = on_failure or _noop_failure
        self._on_notification: NotificationCallback = (
            on_notification or _noop_notification
        )

        # User-configured custom policies (override DEFAULT_POLICIES for a
        # given Severity when valid).  Tests can inject entries here.
        self._custom_policies: Dict[Severity, ReminderPolicy] = {}

    # ------------------------------------------------------------------
    # Public API — Task creation (Requirements 12.1, 12.11)
    # ------------------------------------------------------------------

    def create_task(
        self,
        title: str,
        due_date: Optional[datetime.datetime] = None,
        severity: Optional[Severity] = None,
        description: str = "",
        prerequisites: Optional[List[Prereq]] = None,
        source: TaskSource = TaskSource.MANUAL,
    ) -> Task:
        """Create and return a new Task, assigning a Severity.

        Requirements: 12.1, 12.11.

        When ``severity`` is ``None`` (indeterminate), ``Severity.DEFAULT``
        is assigned and the user is notified via ``on_notification`` (12.11).
        """
        severity_was_defaulted = False

        if severity is None:
            # 12.11 — indeterminate: apply default + notify user
            severity = Severity.DEFAULT
            severity_was_defaulted = True
            self._on_notification(
                "Severity could not be determined; the default severity has been "
                "applied. You can update it at any time."
            )

        task = Task(
            title=title,
            description=description,
            due_date=due_date,
            severity=severity,
            status=TaskStatus.UPCOMING,
            prerequisites=prerequisites or [],
            source=source,
            severity_was_defaulted=severity_was_defaulted,
        )
        logger.debug("Created task %s (severity=%s)", task.id, task.severity)
        return task

    # ------------------------------------------------------------------
    # Public API — Reminder computation (Requirements 12.2–12.4, 12.7, 12.8, 12.10)
    # ------------------------------------------------------------------

    def compute_reminders(self, task: Task) -> List[Reminder]:
        """Return the full set of Reminders for *task*.

        Algorithm
        ---------
        1. Resolve the effective ReminderPolicy (custom-when-valid else
           default) (12.7, 12.8).
        2. For each offset, compute ``fire_at = due_date + offset``.
        3. Compare ``fire_at`` to ``now()`` from the Clock:
           - ``fire_at < now``  → elapsed; set ``fire_at = now`` so the
             reminder fires immediately (12.10).
           - ``fire_at >= now`` → schedule normally.
        4. For BIRTHDAY severity, add a dedicated day-of prompt Reminder
           (12.5).

        Requirements: 12.2, 12.3, 12.4, 12.5, 12.7, 12.8, 12.10.
        """
        if task.due_date is None:
            return []

        now_dt = self._now()
        policy = self._effective_policy(task.severity)
        reminders: List[Reminder] = []

        for offset in policy.offsets:
            fire_at = task.due_date + offset
            if fire_at < now_dt:
                # 12.10 — elapsed window: fire immediately
                fire_at = now_dt
            reminders.append(
                Reminder(
                    task_id=task.id,
                    fire_at=fire_at,
                    channels=[ReminderChannel.VOICE, ReminderChannel.NOTIFICATION],
                    state=ReminderState.SCHEDULED,
                    is_birthday_day_of=False,
                )
            )

        # 12.5 — Birthday day-of prompt
        if task.severity == Severity.BIRTHDAY:
            birthday_day_of = task.due_date.replace(
                hour=9, minute=0, second=0, microsecond=0
            )
            if birthday_day_of < now_dt:
                birthday_day_of = now_dt
            reminders.append(
                Reminder(
                    task_id=task.id,
                    fire_at=birthday_day_of,
                    channels=[ReminderChannel.VOICE, ReminderChannel.NOTIFICATION],
                    state=ReminderState.SCHEDULED,
                    is_birthday_day_of=True,
                )
            )

        return reminders

    # ------------------------------------------------------------------
    # Public API — Reminder issuance (Requirements 12.6, 12.9)
    # ------------------------------------------------------------------

    def issue_reminders(self, task: Task, reminders: List[Reminder]) -> None:
        """Issue all *reminders* for *task* via both channels.

        Per-reminder and per-channel failures are isolated: a failure on
        one channel (or one reminder) does NOT prevent the remaining
        reminders from being issued (12.9).

        For each reminder this method attempts VOICE, then NOTIFICATION,
        recording ``FAILED`` state on any exception and calling
        ``on_failure`` with a reason string.

        Requirements: 12.6, 12.9.
        """
        for reminder in reminders:
            self._issue_one_reminder(task, reminder)

    def issue_reminder(self, task: Task, reminder: Reminder) -> None:
        """Issue a single *reminder* for *task* via both channels.

        Convenience wrapper around ``issue_reminders`` for callers that
        process one reminder at a time.
        """
        self._issue_one_reminder(task, reminder)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _issue_one_reminder(self, task: Task, reminder: Reminder) -> None:
        """Attempt issuance on both channels, isolating each failure."""
        any_channel_failed = False
        failure_reasons: list[str] = []

        for channel in reminder.channels:
            try:
                if channel == ReminderChannel.VOICE:
                    self._voice_adapter(task, reminder)
                elif channel == ReminderChannel.NOTIFICATION:
                    self._notification_adapter(task, reminder)
            except Exception as exc:  # noqa: BLE001
                any_channel_failed = True
                reason = f"Channel {channel.value} failed: {exc}"
                failure_reasons.append(reason)
                logger.warning(
                    "Reminder %s for task %s failed on %s: %s",
                    reminder.id,
                    task.id,
                    channel.value,
                    exc,
                )

        if any_channel_failed:
            # Only mark the reminder FAILED if *all* channels failed; if at
            # least one channel succeeded the user received the reminder.
            full_failure = len(failure_reasons) == len(reminder.channels)
            if full_failure:
                reminder.state = ReminderState.FAILED
            self._on_failure(task, reminder, "; ".join(failure_reasons))
        else:
            reminder.state = ReminderState.FIRED

    def _effective_policy(self, severity: Severity) -> ReminderPolicy:
        """Return the custom policy for *severity* if valid, else the default.

        Requirements: 12.7, 12.8.
        """
        custom = self._custom_policies.get(severity)
        if custom is not None and custom.is_valid():
            return custom  # 12.7 — valid custom overrides default
        # 12.8 — invalid/incomplete custom falls back to default
        return DEFAULT_POLICIES[severity]

    def _now(self) -> datetime.datetime:
        """Return the current timezone-aware datetime from the Clock.

        If the Clock is unavailable, fall back to ``datetime.now()`` in UTC.
        This is a graceful-degradation path; callers in production should
        ensure the Clock is healthy.

        Requirement: 14.2 (Scheduler uses current time from Clock).
        """
        result = self._clock.now()
        if isinstance(result, ClockResult):
            # Re-construct a timezone-aware datetime from the Clock reading.
            # We use the raw datetime from the clock's internal read so that
            # the timezone is preserved exactly as the Clock saw it (e.g., UTC
            # when an _override_now returning a UTC datetime was injected).
            # Fallback: combine date + time with UTC when tz cannot be resolved.
            try:
                import zoneinfo as _zi
                tz = _zi.ZoneInfo(result.timezone)
                return datetime.datetime.combine(result.date, result.time, tzinfo=tz)
            except Exception:
                # POSIX abbreviation or unresolvable name — use UTC
                return datetime.datetime.combine(
                    result.date, result.time, tzinfo=datetime.timezone.utc
                )
        # ClockUnavailable — graceful fallback (14.5)
        logger.warning("Clock unavailable (%s); using system datetime", result.reason)
        return datetime.datetime.now(tz=datetime.timezone.utc)


# ---------------------------------------------------------------------------
# Default no-op adapters (used when no real adapter is injected)
# ---------------------------------------------------------------------------


def _noop_adapter(task: "Task", reminder: "Reminder") -> None:  # noqa: ARG001
    """No-op voice/notification adapter."""


def _noop_failure(task: "Task", reminder: "Reminder", reason: str) -> None:  # noqa: ARG001
    """No-op failure callback."""


def _noop_notification(message: str) -> None:  # noqa: ARG001
    """No-op user-notification callback."""
