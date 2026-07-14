"""
Tests for the Task_Tracker subsystem (task 30.1).

Covers:
- Persistent task list with due dates and severity (Req 13.1)
- Incomplete tasks listed sorted by due date within 2 s (Req 13.2)
- Add failure atomicity: no partial entry, inform, retain details (Req 13.7)

Design: Task_Tracker.
Requirements: 13.1, 13.2, 13.7.
"""

from __future__ import annotations

import datetime
import time
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from core.scheduler.models import (
    Prereq,
    Severity,
    Task,
    TaskSource,
    TaskStatus,
)
from core.scheduler.task_tracker import (
    TaskAddError,
    TaskTracker,
    SQLiteTaskStore,
    create_sqlite_task_tracker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TZ = datetime.timezone.utc


def _dt(year=2024, month=6, day=1, hour=0, minute=0) -> datetime.datetime:
    """Return a UTC-aware datetime for use in tests."""
    return datetime.datetime(year, month, day, hour, minute, tzinfo=_TZ)


def _make_task(
    title: str = "Test Task",
    due_date: datetime.datetime | None = None,
    severity: Severity = Severity.DEFAULT,
    status: TaskStatus = TaskStatus.UPCOMING,
    prerequisites: List[Prereq] | None = None,
    source: TaskSource = TaskSource.MANUAL,
) -> Task:
    return Task(
        title=title,
        description="",
        due_date=due_date,
        severity=severity,
        status=status,
        prerequisites=prerequisites or [],
        source=source,
    )


def _make_tracker(tmp_path: Path, on_add_failure=None) -> TaskTracker:
    """Create a TaskTracker backed by a SQLite database in tmp_path."""
    db = tmp_path / "tasks.db"
    return create_sqlite_task_tracker(
        db_path=db,
        on_add_failure=on_add_failure,
    )


# ===========================================================================
# 30.1.1 — Persistent task list (Req 13.1)
# ===========================================================================


class TestPersistence:
    """Requirements: 13.1."""

    def test_added_task_is_retrievable(self, tmp_path):
        """Tasks stored with add() are returned by list()."""
        tracker = _make_tracker(tmp_path)
        task = _make_task("Read chapter", due_date=_dt(2024, 9, 1))
        tracker.add(task)
        tasks = tracker.list(incomplete_only=False)
        ids = [t.id for t in tasks]
        assert task.id in ids

    def test_task_fields_are_preserved(self, tmp_path):
        """All Task fields survive a round-trip through SQLite."""
        prereq = Prereq(title="Step 1")
        task = Task(
            title="My Task",
            description="Some details",
            due_date=_dt(2024, 10, 15),
            severity=Severity.EXAM,
            status=TaskStatus.UPCOMING,
            prerequisites=[prereq],
            source=TaskSource.COMMS,
            severity_was_defaulted=False,
        )
        # Create a fresh tracker instance that will load from the same DB
        db = tmp_path / "tasks.db"
        store = SQLiteTaskStore(db_path=db)
        store.persist(task)
        loaded_tasks = store.load()
        assert len(loaded_tasks) == 1
        loaded = loaded_tasks[0]
        assert loaded.id == task.id
        assert loaded.title == "My Task"
        assert loaded.description == "Some details"
        assert loaded.due_date == _dt(2024, 10, 15)
        assert loaded.severity == Severity.EXAM
        assert loaded.status == TaskStatus.UPCOMING
        assert loaded.source == TaskSource.COMMS

    def test_prerequisites_are_preserved(self, tmp_path):
        """Prerequisites survive a round-trip through SQLite."""
        prereqs = [
            Prereq(title="Step A", status=TaskStatus.UPCOMING),
            Prereq(title="Step B", status=TaskStatus.COMPLETE),
        ]
        task = _make_task("Parent Task", prerequisites=prereqs)
        db = tmp_path / "tasks.db"
        store = SQLiteTaskStore(db_path=db)
        store.persist(task)
        loaded_tasks = store.load()
        assert len(loaded_tasks[0].prerequisites) == 2
        titles = [p.title for p in loaded_tasks[0].prerequisites]
        assert "Step A" in titles
        assert "Step B" in titles

    def test_multiple_tasks_are_all_stored(self, tmp_path):
        """All added tasks are returned by list()."""
        tracker = _make_tracker(tmp_path)
        for i in range(5):
            tracker.add(_make_task(f"Task {i}", due_date=_dt(2024, 6, i + 1)))
        tasks = tracker.list(incomplete_only=False)
        assert len(tasks) == 5

    def test_persistence_across_new_tracker_instance(self, tmp_path):
        """Tasks survive reopening the database (persist across restart)."""
        db = tmp_path / "tasks.db"
        t1 = create_sqlite_task_tracker(db_path=db)
        task = _make_task("Persistent Task", due_date=_dt(2024, 12, 1))
        t1.add(task)

        # Re-open with a new instance pointing at the same DB file
        t2 = create_sqlite_task_tracker(db_path=db)
        tasks = t2.list(incomplete_only=False)
        assert any(t.id == task.id for t in tasks)

    def test_severity_is_stored_and_retrieved(self, tmp_path):
        """Severity values round-trip correctly."""
        tracker = _make_tracker(tmp_path)
        for sev in Severity:
            task = _make_task(f"Task {sev}", severity=sev)
            tracker.add(task)
        tasks = tracker.list(incomplete_only=False)
        stored_severities = {t.severity for t in tasks}
        assert stored_severities == set(Severity)

    def test_task_without_due_date_is_stored(self, tmp_path):
        """Tasks without a due_date are stored with due_date=None."""
        tracker = _make_tracker(tmp_path)
        task = _make_task("No due date", due_date=None)
        tracker.add(task)
        tasks = tracker.list(incomplete_only=False)
        assert len(tasks) == 1
        assert tasks[0].due_date is None


# ===========================================================================
# 30.1.2 — Sorted incomplete listing within 2 s (Req 13.2)
# ===========================================================================


class TestSortedIncompleteListing:
    """Requirements: 13.2."""

    def test_incomplete_only_excludes_complete_tasks(self, tmp_path):
        """list(incomplete_only=True) returns only UPCOMING tasks."""
        tracker = _make_tracker(tmp_path)
        upcoming = _make_task("Upcoming", due_date=_dt(2024, 9, 5), status=TaskStatus.UPCOMING)
        complete = _make_task("Done", due_date=_dt(2024, 9, 1), status=TaskStatus.COMPLETE)
        tracker.add(upcoming)
        tracker.add(complete)
        tasks = tracker.list(incomplete_only=True)
        ids = [t.id for t in tasks]
        assert upcoming.id in ids
        assert complete.id not in ids

    def test_incomplete_only_false_returns_all(self, tmp_path):
        """list(incomplete_only=False) returns all tasks regardless of status."""
        tracker = _make_tracker(tmp_path)
        tracker.add(_make_task("T1", status=TaskStatus.UPCOMING))
        tracker.add(_make_task("T2", status=TaskStatus.COMPLETE))
        tasks = tracker.list(incomplete_only=False)
        assert len(tasks) == 2

    def test_sorted_by_due_date_ascending(self, tmp_path):
        """Incomplete tasks are returned ordered by due_date ascending."""
        tracker = _make_tracker(tmp_path)
        # Add in reverse order to ensure sorting is applied
        dates = [
            _dt(2024, 12, 1),
            _dt(2024, 8, 1),
            _dt(2024, 10, 15),
        ]
        for i, d in enumerate(dates):
            tracker.add(_make_task(f"Task {i}", due_date=d))

        tasks = tracker.list(incomplete_only=True)
        actual_dates = [t.due_date for t in tasks]
        assert actual_dates == sorted(actual_dates)

    def test_sorted_ascending_with_many_tasks(self, tmp_path):
        """Due-date ordering holds for a larger number of tasks."""
        tracker = _make_tracker(tmp_path)
        import random
        rng = random.Random(42)
        for i in range(20):
            d = _dt(2024, rng.randint(1, 12), rng.randint(1, 28))
            tracker.add(_make_task(f"Task {i}", due_date=d))
        tasks = tracker.list(incomplete_only=True)
        dates = [t.due_date for t in tasks]
        assert dates == sorted(dates)

    def test_tasks_without_due_date_appear_at_end(self, tmp_path):
        """Tasks with no due_date are listed after tasks that have one."""
        tracker = _make_tracker(tmp_path)
        with_due = _make_task("With due", due_date=_dt(2024, 6, 1))
        without_due = _make_task("No due", due_date=None)
        tracker.add(without_due)
        tracker.add(with_due)
        tasks = tracker.list(incomplete_only=True)
        # All None entries must be at the tail
        due_dates = [t.due_date for t in tasks]
        first_none = next(
            (i for i, d in enumerate(due_dates) if d is None), len(due_dates)
        )
        assert all(due_dates[j] is None for j in range(first_none, len(due_dates)))

    def test_empty_list_when_no_tasks(self, tmp_path):
        """list() returns [] when no tasks have been added."""
        tracker = _make_tracker(tmp_path)
        assert tracker.list() == []

    def test_empty_list_when_all_complete(self, tmp_path):
        """list(incomplete_only=True) returns [] when all tasks are COMPLETE."""
        tracker = _make_tracker(tmp_path)
        tracker.add(_make_task("Done", status=TaskStatus.COMPLETE))
        assert tracker.list(incomplete_only=True) == []

    def test_list_returns_within_2_seconds(self, tmp_path):
        """list() completes within the 2 s budget (Req 13.2)."""
        tracker = _make_tracker(tmp_path)
        for i in range(100):
            tracker.add(_make_task(f"Task {i}", due_date=_dt(2024, 1, 1)))
        start = time.monotonic()
        tracker.list(incomplete_only=True)
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, f"list() took {elapsed:.2f} s (budget: 2 s)"

    def test_list_default_returns_incomplete_only(self, tmp_path):
        """list() with no argument defaults to incomplete_only=True."""
        tracker = _make_tracker(tmp_path)
        tracker.add(_make_task("Upcoming", status=TaskStatus.UPCOMING))
        tracker.add(_make_task("Done", status=TaskStatus.COMPLETE))
        tasks = tracker.list()
        assert len(tasks) == 1
        assert tasks[0].status == TaskStatus.UPCOMING


# ===========================================================================
# 30.1.3 — Add failure atomicity (Req 13.7)
# ===========================================================================


class TestAddFailureAtomicity:
    """Requirements: 13.7."""

    def test_add_failure_raises_task_add_error(self, tmp_path):
        """add() raises TaskAddError on persistence failure."""
        failures: list[tuple] = []

        def fail_persist(task: Task) -> None:
            raise OSError("disk full")

        tracker = TaskTracker(
            persist=fail_persist,
            on_add_failure=lambda t, exc: failures.append((t, str(exc))),
        )
        task = _make_task("Failing task")

        with pytest.raises(TaskAddError) as exc_info:
            tracker.add(task)

        assert "disk full" in str(exc_info.value)

    def test_add_failure_retains_task_details_on_exception(self, tmp_path):
        """TaskAddError carries the original task details for retry (Req 13.7)."""
        def fail_persist(task: Task) -> None:
            raise RuntimeError("db error")

        tracker = TaskTracker(persist=fail_persist)
        original_task = _make_task("Important Task", due_date=_dt(2024, 11, 1))

        with pytest.raises(TaskAddError) as exc_info:
            tracker.add(original_task)

        error = exc_info.value
        # The task details are retained on the exception for retry
        assert error.task.id == original_task.id
        assert error.task.title == "Important Task"
        assert error.task.due_date == _dt(2024, 11, 1)

    def test_add_failure_leaves_no_in_memory_entry(self, tmp_path):
        """On failure no partial entry is in the in-process store (Req 13.7)."""
        good_tasks: list[Task] = []

        def selective_persist(task: Task) -> None:
            if task.title == "Bad task":
                raise IOError("write error")

        tracker = TaskTracker(persist=selective_persist)
        # Add one good task first
        good = _make_task("Good task", due_date=_dt(2024, 6, 1))
        tracker.add(good)

        # Attempt to add a task that will fail mid-write
        bad_task = _make_task("Bad task")
        with pytest.raises(TaskAddError):
            tracker.add(bad_task)

        # Only the good task should be in the tracker
        tasks = tracker.list(incomplete_only=False)
        assert len(tasks) == 1
        assert tasks[0].id == good.id

    def test_add_failure_calls_on_add_failure_callback(self, tmp_path):
        """on_add_failure is called with the task and exception on add failure."""
        failures: list[tuple] = []

        def fail_persist(task: Task) -> None:
            raise RuntimeError("failure reason")

        tracker = TaskTracker(
            persist=fail_persist,
            on_add_failure=lambda t, exc: failures.append((t, exc)),
        )
        task = _make_task("Task")

        with pytest.raises(TaskAddError):
            tracker.add(task)

        assert len(failures) == 1
        failed_task, exc = failures[0]
        assert failed_task.id == task.id
        assert "failure reason" in str(exc)

    def test_add_failure_does_not_call_callback_on_success(self, tmp_path):
        """on_add_failure is NOT called when add succeeds."""
        failures: list[tuple] = []
        tracker = TaskTracker(
            on_add_failure=lambda t, exc: failures.append((t, exc)),
        )
        tracker.add(_make_task("Success"))
        assert failures == []

    def test_sqlite_add_db_constraint_violation_is_atomic(self, tmp_path):
        """Inserting a duplicate id does not corrupt the database."""
        tracker = _make_tracker(tmp_path)
        task = _make_task("Original")
        tracker.add(task)

        # Try adding a task with the same id — should fail cleanly
        duplicate = Task(
            id=task.id,  # same id → UNIQUE constraint violation
            title="Duplicate",
            description="",
            due_date=None,
        )
        with pytest.raises(TaskAddError):
            tracker.add(duplicate)

        # Tracker should still have exactly one task
        tasks = tracker.list(incomplete_only=False)
        assert len(tasks) == 1
        assert tasks[0].title == "Original"


# ===========================================================================
# 30.1.4 — SQLiteTaskStore atomicity (Req 13.7)
# ===========================================================================


class TestSQLiteTaskStoreAtomicity:
    """Verify the SQLiteTaskStore's atomic write guarantee (Req 13.7)."""

    def test_successful_persist_and_load_round_trip(self, tmp_path):
        """persist + load round-trips a Task correctly."""
        db = tmp_path / "tasks.db"
        store = SQLiteTaskStore(db_path=db)
        task = _make_task("My Task", due_date=_dt(2024, 9, 1), severity=Severity.ASSIGNMENT)
        store.persist(task)
        tasks = store.load()
        assert len(tasks) == 1
        assert tasks[0].id == task.id
        assert tasks[0].severity == Severity.ASSIGNMENT

    def test_failed_persist_leaves_db_unchanged(self, tmp_path):
        """A failed persist call leaves the database completely unchanged."""
        db = tmp_path / "tasks.db"
        store = SQLiteTaskStore(db_path=db)
        # Persist one good task
        good = _make_task("Good", due_date=_dt(2024, 6, 1))
        store.persist(good)

        # Construct a task with an ID identical to the good task —
        # the INSERT will fail on the UNIQUE constraint
        duplicate = Task(
            id=good.id,
            title="Should Not Appear",
            description="",
            due_date=None,
        )
        import sqlite3 as _sqlite3
        with pytest.raises(_sqlite3.IntegrityError):
            store.persist(duplicate)

        # Database should still have exactly one task (the original)
        tasks = store.load()
        assert len(tasks) == 1
        assert tasks[0].title == "Good"

    def test_prerequisites_not_written_on_task_row_failure(self, tmp_path):
        """If the task row insert fails, no orphaned prerequisite rows are written."""
        db = tmp_path / "tasks.db"
        store = SQLiteTaskStore(db_path=db)
        # First task succeeds
        first = _make_task("First")
        store.persist(first)

        # Second task has same id — will fail; has prereqs we must NOT see
        prereqs = [Prereq(title="Pre A"), Prereq(title="Pre B")]
        duplicate = Task(id=first.id, title="Dup", prerequisites=prereqs)

        import sqlite3 as _sqlite3
        with pytest.raises(_sqlite3.IntegrityError):
            store.persist(duplicate)

        # Reload — should still have exactly the original task with no extra prereqs
        tasks = store.load()
        assert len(tasks) == 1
        assert tasks[0].title == "First"
        # First task had no prerequisites
        assert tasks[0].prerequisites == []

    def test_load_is_sorted_by_due_date(self, tmp_path):
        """load() returns tasks sorted by due_date ascending."""
        db = tmp_path / "tasks.db"
        store = SQLiteTaskStore(db_path=db)
        # Persist in reverse order
        for month in [12, 3, 8]:
            store.persist(_make_task(f"Task {month}", due_date=_dt(2024, month, 1)))
        tasks = store.load()
        dates = [t.due_date for t in tasks]
        assert dates == sorted(dates)


# ===========================================================================
# 30.1.5 — TaskAddError model
# ===========================================================================


class TestTaskAddError:
    def test_error_message_contains_task_id(self):
        task = _make_task("My Important Task")
        cause = RuntimeError("disk full")
        err = TaskAddError(task, cause)
        assert task.id in str(err)

    def test_error_has_task_attribute(self):
        task = _make_task("Task")
        cause = RuntimeError("x")
        err = TaskAddError(task, cause)
        assert err.task is task

    def test_error_has_cause_attribute(self):
        task = _make_task("Task")
        cause = RuntimeError("custom error")
        err = TaskAddError(task, cause)
        assert err.cause is cause
