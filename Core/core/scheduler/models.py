"""
Data models for the Scheduler subsystem.

Design: Data Models (Task & Reminder).
Requirements: 12.1, 12.2, 12.3, 12.4, 12.7, 12.8, 12.11.

Public types
------------
Severity
    Enumeration of task/event severity levels.  Every Task must carry one;
    when severity cannot be determined, DEFAULT is assigned (12.1, 12.11).

TaskSource, TaskStatus
    Provenance and lifecycle state of a Task.

ReminderChannel
    The two channels through which a Reminder is issued: VOICE and
    NOTIFICATION (12.6).

ReminderState
    Lifecycle state of an individual Reminder instance.

Prereq
    A prerequisite sub-item tracked inside a Task (13.6).

Task
    Central data record — id, title, description, due date, severity, status,
    prerequisites, source.  ``source`` is ``manual | comms | command``.

Reminder
    A single scheduled fire-point for a Task, with per-channel delivery
    state encoded in ``state``.

ReminderPolicy
    Defines the offset schedule for a given Severity.  ``offsets`` is a
    list of ``datetime.timedelta`` objects (negative = before due date).
    ``custom`` marks user-overridden policies.

DEFAULT_POLICIES
    Pre-built ReminderPolicy instances for every Severity value, matching
    the design requirements:
      ASSIGNMENT / EXAM  → [-7d, -3d]           (12.2)
      BIRTHDAY           → [-14d, -1d]           (12.4)
      DEFAULT            → [-1d]                 (12.3)
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    """Severity classification for tasks and events (Requirement 12.1)."""

    ASSIGNMENT = "ASSIGNMENT"
    EXAM = "EXAM"
    BIRTHDAY = "BIRTHDAY"
    DEFAULT = "DEFAULT"


class TaskSource(str, Enum):
    """How the task was created."""

    MANUAL = "manual"
    COMMS = "comms"
    COMMAND = "command"


class TaskStatus(str, Enum):
    """Lifecycle status of a Task."""

    UPCOMING = "UPCOMING"
    COMPLETE = "COMPLETE"


class ReminderChannel(str, Enum):
    """Delivery channel for a Reminder (Requirement 12.6)."""

    VOICE = "VOICE"
    NOTIFICATION = "NOTIFICATION"


class ReminderState(str, Enum):
    """Current state of a Reminder."""

    SCHEDULED = "SCHEDULED"
    FIRED = "FIRED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Prereq
# ---------------------------------------------------------------------------


@dataclass
class Prereq:
    """A prerequisite sub-item inside a Task (Requirement 13.6).

    Attributes
    ----------
    id:
        Stable unique identifier.
    title:
        Human-readable description of the prerequisite.
    status:
        Completion status — either ``UPCOMING`` or ``COMPLETE``.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    status: TaskStatus = TaskStatus.UPCOMING


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """A schedulable work item with a severity classification.

    Requirements: 12.1, 12.11, 13.6.

    Attributes
    ----------
    id:
        Stable unique identifier.
    title:
        Short human-readable name.
    description:
        Optional longer description.
    due_date:
        When the task is due (timezone-aware datetime recommended).
    severity:
        Always set; defaults to ``Severity.DEFAULT`` when indeterminate
        (12.11).
    status:
        ``UPCOMING`` or ``COMPLETE``.
    prerequisites:
        Ordered list of prerequisite items (13.6).
    source:
        ``manual | comms | command``.
    severity_was_defaulted:
        ``True`` when the caller could not determine a severity and the
        scheduler fell back to ``Severity.DEFAULT`` (12.11).  This flag
        is used to trigger user notification.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    due_date: Optional[datetime.datetime] = None
    severity: Severity = Severity.DEFAULT
    status: TaskStatus = TaskStatus.UPCOMING
    prerequisites: List[Prereq] = field(default_factory=list)
    source: TaskSource = TaskSource.MANUAL
    severity_was_defaulted: bool = False

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for persistence / logging)."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "severity": self.severity.value,
            "status": self.status.value,
            "source": self.source.value,
            "severity_was_defaulted": self.severity_was_defaulted,
            "prerequisites": [
                {"id": p.id, "title": p.title, "status": p.status.value}
                for p in self.prerequisites
            ],
        }


# ---------------------------------------------------------------------------
# Reminder
# ---------------------------------------------------------------------------


@dataclass
class Reminder:
    """A single scheduled fire-point for a Task (Requirement 12.6, 12.9).

    Attributes
    ----------
    id:
        Stable unique identifier.
    task_id:
        The Task this Reminder belongs to.
    fire_at:
        When this Reminder should be issued.
    channels:
        Delivery channels — must contain both VOICE and NOTIFICATION (12.6).
    state:
        ``SCHEDULED | FIRED | FAILED``.
    is_birthday_day_of:
        ``True`` for the special birthday day-of confirmation prompt (12.5).
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    fire_at: Optional[datetime.datetime] = None
    channels: List[ReminderChannel] = field(
        default_factory=lambda: [ReminderChannel.VOICE, ReminderChannel.NOTIFICATION]
    )
    state: ReminderState = ReminderState.SCHEDULED
    is_birthday_day_of: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "fire_at": self.fire_at.isoformat() if self.fire_at else None,
            "channels": [c.value for c in self.channels],
            "state": self.state.value,
            "is_birthday_day_of": self.is_birthday_day_of,
        }


# ---------------------------------------------------------------------------
# ReminderPolicy
# ---------------------------------------------------------------------------


@dataclass
class ReminderPolicy:
    """Defines the reminder-offset schedule for a given Severity.

    Requirements: 12.2, 12.3, 12.4, 12.7, 12.8.

    Attributes
    ----------
    severity:
        The Severity this policy applies to.
    offsets:
        List of ``timedelta`` values relative to the due date (negative =
        before the due date).  E.g., ``timedelta(days=-7)`` = 7 days before.
    custom:
        ``True`` when the user has overridden the default offsets (12.7).
    """

    severity: Severity
    offsets: List[datetime.timedelta]
    custom: bool = False

    def is_valid(self) -> bool:
        """Return True when the policy has at least one offset (12.8).

        A policy with an empty offset list is considered invalid/incomplete
        and will cause callers to fall back to the default policy.
        """
        return len(self.offsets) > 0


# ---------------------------------------------------------------------------
# Default policies (Requirements 12.2, 12.3, 12.4)
# ---------------------------------------------------------------------------

#: Pre-built default ReminderPolicy instances keyed by Severity.
DEFAULT_POLICIES: dict[Severity, ReminderPolicy] = {
    Severity.ASSIGNMENT: ReminderPolicy(
        severity=Severity.ASSIGNMENT,
        offsets=[datetime.timedelta(days=-7), datetime.timedelta(days=-3)],
    ),
    Severity.EXAM: ReminderPolicy(
        severity=Severity.EXAM,
        offsets=[datetime.timedelta(days=-7), datetime.timedelta(days=-3)],
    ),
    Severity.BIRTHDAY: ReminderPolicy(
        severity=Severity.BIRTHDAY,
        offsets=[datetime.timedelta(days=-14), datetime.timedelta(days=-1)],
    ),
    Severity.DEFAULT: ReminderPolicy(
        severity=Severity.DEFAULT,
        offsets=[datetime.timedelta(days=-1)],
    ),
}
