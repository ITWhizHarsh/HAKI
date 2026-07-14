"""
Scheduler sub-package — severity-based reminders and task tracking.

Implements task creation with severity assignment, reminder-offset computation
using the Clock, dual-channel reminder issuance with per-reminder failure
isolation, and the persistent Task_Tracker.

Design: Scheduler, Task_Tracker, Data Models (Task & Reminder).
Requirements: 12.1 – 12.11, 13.1, 13.2, 13.7.
"""

from .models import (
    Severity,
    TaskSource,
    TaskStatus,
    ReminderChannel,
    ReminderState,
    Prereq,
    Task,
    Reminder,
    ReminderPolicy,
    DEFAULT_POLICIES,
)
from .scheduler import Scheduler
from .task_tracker import TaskAddError, TaskTracker, SQLiteTaskStore, create_sqlite_task_tracker

__all__ = [
    # Enums / constants
    "Severity",
    "TaskSource",
    "TaskStatus",
    "ReminderChannel",
    "ReminderState",
    # Data models
    "Prereq",
    "Task",
    "Reminder",
    "ReminderPolicy",
    "DEFAULT_POLICIES",
    # Scheduler
    "Scheduler",
    # Task_Tracker
    "TaskTracker",
    "TaskAddError",
    "SQLiteTaskStore",
    "create_sqlite_task_tracker",
]
