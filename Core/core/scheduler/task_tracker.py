"""
Task_Tracker — task list management, due-date prompting, completion transitions,
and prerequisite tracking.

Design: Task_Tracker.
Requirements: 13.1 – 13.7.

Responsibilities
----------------
Task 30.1 (foundation):
* Maintain a persistent list of tasks with due dates and severity (13.1).
* Return incomplete tasks ordered by due date within 2 s (13.2).
* Persist tasks durably; on add failure add no partial entry and retain
  details for retry (13.7).

Task 30.2 (this task):
* ``on_due_passed(task_id)`` — async watcher that asks "Did you complete
  [task]?" within 60 s of the due date passing (13.3).
* ``mark_complete(task_id)`` — transitions task status → COMPLETE and
  cancels all pending reminder asyncio tasks for that task (13.4, 13.5).
* ``prerequisites(task_id)`` — returns the prerequisite list with each
  prereq's completion status; used when the user requests a task's status
  to report which prerequisites remain incomplete (13.6).

Architecture notes
------------------
The TaskTracker is a synchronous-first class with an optional asyncio
integration layer for the due-date watcher.  The due-date watcher spawns
one asyncio.Task per tracked task; those tasks are stored in
``_watcher_tasks`` keyed by task_id and cancelled when the task is marked
complete.

SQLite persistence is provided by the companion :class:`SQLiteTaskStore`
class (see below), which exposes ``persist`` and ``load`` callables that
can be injected into the TaskTracker constructor.  Use
:func:`create_sqlite_task_tracker` for a convenience factory that wires
both together.

Adapters / callbacks (all injectable for testing)
-------------------------------------------------
``reminder_stopper`` : Callable[[str], None]
    Called with a task_id when a task is marked complete so the Scheduler
    can cancel / suppress pending Reminders (13.5).
    Signature: ``(task_id: str) -> None``

``due_prompt_callback`` : Callable[[Task], Awaitable[None]] | None
    Async callable invoked by the due-date watcher within 60 s of the
    task's due date passing.  It should surface "Did you complete [task]?"
    to the user via the Voice_Engine + notification channel.
    Signature: ``async (task: Task) -> None``

``on_add_failure`` : Callable[[Task, Exception], None]
    Called when durable persistence of an ``add()`` fails (13.7).
    Signature: ``(task: Task, exc: Exception) -> None``

``persist`` / ``load`` : Callable[[Task], None] / Callable[[], List[Task]]
    Thin persistence stubs.  Default implementations keep tasks in an
    in-process dict (adequate for unit testing); replace with SQLite
    calls in production.

Testing seams
-------------
* Inject ``due_prompt_callback`` to capture prompts without real voice.
* Inject ``reminder_stopper`` to assert cancellation was requested.
* Inject ``persist`` / ``load`` to control persistence behaviour.
* ``_clock_now`` — override via ``_override_clock_now`` kwarg to advance
  virtual time in tests.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import sqlite3
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional

from .models import Prereq, Reminder, ReminderState, Severity, Task, TaskSource, TaskStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TaskAddError(Exception):
    """Raised when a task cannot be durably persisted (Requirement 13.7).

    The caller should retain the task details for retry.
    """

    def __init__(self, task: "Task", cause: Exception) -> None:
        super().__init__(f"Failed to add task {task.id!r}: {cause}")
        self.task = task
        self.cause = cause


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ReminderStopperCallback = Callable[[str], None]
DuePromptCallback = Callable[["Task"], Awaitable[None]]
AddFailureCallback = Callable[["Task", Exception], None]
PersistCallback = Callable[["Task"], None]
LoadCallback = Callable[[], List["Task"]]
ClockNow = Callable[[], datetime.datetime]

# How long (seconds) after a task's due date to wait before prompting (13.3).
_DUE_PROMPT_WINDOW_SECONDS: float = 60.0


# ---------------------------------------------------------------------------
# Default no-op helpers
# ---------------------------------------------------------------------------


def _default_clock_now() -> datetime.datetime:
    return datetime.datetime.now(tz=datetime.timezone.utc)


def _noop_reminder_stopper(task_id: str) -> None:  # noqa: ARG001
    pass


async def _noop_due_prompt(task: "Task") -> None:  # noqa: ARG001
    pass


def _noop_add_failure(task: "Task", exc: Exception) -> None:  # noqa: ARG001
    pass


def _noop_persist(task: "Task") -> None:  # noqa: ARG001
    pass


def _noop_load() -> List["Task"]:
    return []


# ---------------------------------------------------------------------------
# TaskTracker
# ---------------------------------------------------------------------------


class TaskTracker:
    """Maintains the task list and drives due-date prompting.

    Parameters
    ----------
    reminder_stopper:
        Callable invoked with a task_id when a task is marked complete so
        the Scheduler can cancel pending Reminders (13.5).
    due_prompt_callback:
        Async callable that surfaces "Did you complete [task]?" to the
        user within 60 s of the due date passing (13.3).
    on_add_failure:
        Called when persistence of a new task fails (13.7).
    persist:
        Durably write a task to persistent storage.  Default: in-process
        dict only (no-op for external storage).
    load:
        Load the initial task list from persistent storage on startup.
        Default: returns an empty list (no external storage).
    _override_clock_now:
        Inject a callable to override the default ``datetime.now()`` used
        by the due-date watcher (for deterministic tests).
    """

    def __init__(
        self,
        reminder_stopper: Optional[ReminderStopperCallback] = None,
        due_prompt_callback: Optional[DuePromptCallback] = None,
        on_add_failure: Optional[AddFailureCallback] = None,
        persist: Optional[PersistCallback] = None,
        load: Optional[LoadCallback] = None,
        _override_clock_now: Optional[ClockNow] = None,
    ) -> None:
        self._reminder_stopper: ReminderStopperCallback = (
            reminder_stopper or _noop_reminder_stopper
        )
        self._due_prompt_callback: DuePromptCallback = (
            due_prompt_callback or _noop_due_prompt
        )
        self._on_add_failure: AddFailureCallback = on_add_failure or _noop_add_failure
        self._persist: PersistCallback = persist or _noop_persist
        self._load: LoadCallback = load or _noop_load
        self._clock_now: ClockNow = _override_clock_now or _default_clock_now

        # In-process task store: task_id → Task
        self._tasks: Dict[str, Task] = {}

        # Active asyncio watcher tasks keyed by task_id (13.3).
        # Each entry is an asyncio.Task that waits until due date +
        # ≤60 s and then fires the due-prompt callback.
        self._watcher_tasks: Dict[str, asyncio.Task] = {}

        # Hydrate from persistent storage on startup (13.1).
        for task in self._load():
            self._tasks[task.id] = task

    # ------------------------------------------------------------------
    # Public API — 30.1 foundation (13.1, 13.2, 13.7)
    # ------------------------------------------------------------------

    def add(self, task: Task) -> Task:
        """Persist *task* and add it to the in-process store.

        On any persistence failure, no partial entry is added and
        ``on_add_failure`` is called with the task and the exception so
        the caller can retain details for retry (Requirement 13.7).

        Returns the task on success.

        Requirements: 13.1, 13.7.
        """
        try:
            self._persist(task)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "TaskTracker.add: persistence failed for task %r (%s); "
                "no entry added.",
                task.id,
                exc,
            )
            self._on_add_failure(task, exc)
            # Do NOT add to the in-process store — requirement 13.7.
            raise TaskAddError(task, exc) from exc

        self._tasks[task.id] = task
        logger.debug("TaskTracker.add: task %r added", task.id)
        return task

    def list(self, incomplete_only: bool = True) -> List[Task]:
        """Return tasks sorted by ``due_date`` ascending.

        Parameters
        ----------
        incomplete_only:
            When ``True`` (default) only UPCOMING tasks are returned
            (Req 13.2).  When ``False`` all tasks are returned (Req 13.1).

        Returns
        -------
        List[Task]
            Tasks ordered by ``due_date`` ascending.  Tasks without a
            ``due_date`` appear at the end.

        Requirements: 13.1, 13.2.
        """
        if incomplete_only:
            return self.list_incomplete()
        tasks = list(self._tasks.values())
        tasks.sort(
            key=lambda t: (
                t.due_date is None,
                t.due_date or datetime.datetime.max,
            )
        )
        return tasks

    def list_incomplete(self) -> List[Task]:
        """Return incomplete tasks ordered by due date (ascending).

        Tasks without a due date are placed after all tasks with one.
        Returns within 2 s (Requirement 13.2 budget — in practice this is
        O(n log n) on the in-process dict and far below 2 s).

        Requirements: 13.1, 13.2.
        """
        incomplete = [
            t for t in self._tasks.values() if t.status == TaskStatus.UPCOMING
        ]
        # Sort by due_date ascending; None → infinity (after all dated tasks).
        incomplete.sort(
            key=lambda t: (
                t.due_date is None,
                t.due_date or datetime.datetime.max,
            )
        )
        return incomplete

    def get(self, task_id: str) -> Optional[Task]:
        """Return the task with the given *task_id*, or None.

        Requirements: 13.1.
        """
        return self._tasks.get(task_id)

    # ------------------------------------------------------------------
    # Public API — 30.2: completion transitions (13.4, 13.5)
    # ------------------------------------------------------------------

    def mark_complete(self, task_id: str) -> Optional[Task]:
        """Transition *task_id* to COMPLETE and stop all its reminders.

        1. Sets ``task.status = COMPLETE`` (Requirement 13.4).
        2. Calls ``reminder_stopper(task_id)`` to cancel / suppress pending
           Scheduler reminders (Requirement 13.5).
        3. Cancels the asyncio due-date watcher task if it is still running
           (Requirement 13.5 — stops "further reminders").

        Returns the updated Task, or None if *task_id* is not found.

        Requirements: 13.4, 13.5.
        """
        task = self._tasks.get(task_id)
        if task is None:
            logger.warning("TaskTracker.mark_complete: unknown task_id %r", task_id)
            return None

        task.status = TaskStatus.COMPLETE
        logger.debug("TaskTracker.mark_complete: task %r → COMPLETE", task_id)

        # 13.5 — stop Scheduler reminders for this task.
        try:
            self._reminder_stopper(task_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TaskTracker.mark_complete: reminder_stopper raised for %r: %s",
                task_id,
                exc,
            )

        # Cancel the asyncio due-date watcher if still running.
        self._cancel_watcher(task_id)

        return task

    # ------------------------------------------------------------------
    # Public API — 30.2: prerequisite tracking (13.6)
    # ------------------------------------------------------------------

    def prerequisites(self, task_id: str) -> List[Prereq]:
        """Return the prerequisite list for *task_id* with each prereq's status.

        Used when the user requests a task's status: callers should report
        which prerequisites remain incomplete (Requirement 13.6).

        Returns an empty list if the task is not found or has no
        prerequisites.

        Requirements: 13.6.
        """
        task = self._tasks.get(task_id)
        if task is None:
            logger.warning("TaskTracker.prerequisites: unknown task_id %r", task_id)
            return []
        return list(task.prerequisites)

    def incomplete_prerequisites(self, task_id: str) -> List[Prereq]:
        """Return only the prerequisites that are not yet COMPLETE.

        Convenience wrapper over :py:meth:`prerequisites` that filters to
        the subset with status != COMPLETE.  This is what callers should
        surface when the user asks for a task's status (Req 13.6).

        Requirements: 13.6.
        """
        return [
            p for p in self.prerequisites(task_id) if p.status != TaskStatus.COMPLETE
        ]

    def mark_prerequisite_complete(
        self, task_id: str, prereq_id: str
    ) -> Optional[Prereq]:
        """Mark a specific prerequisite as COMPLETE.

        Returns the updated Prereq, or None if the task or prereq is not
        found.

        Requirements: 13.6.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return None
        for prereq in task.prerequisites:
            if prereq.id == prereq_id:
                prereq.status = TaskStatus.COMPLETE
                return prereq
        return None

    # ------------------------------------------------------------------
    # Public API — 30.2: asyncio due-date watcher (13.3)
    # ------------------------------------------------------------------

    def schedule_due_watcher(self, task: Task) -> None:
        """Schedule the asyncio due-date watcher for *task*.

        Spawns an asyncio.Task that waits until ``task.due_date`` passes
        and then (within 60 s) calls ``due_prompt_callback(task)``
        (Requirement 13.3).

        Safe to call from synchronous context — it obtains or uses the
        running event loop via ``asyncio.get_event_loop()``.

        If the task is already complete, the watcher is not started.
        If a watcher is already running for the task, it is not replaced.

        Requirements: 13.3.
        """
        if task.due_date is None:
            return
        if task.status == TaskStatus.COMPLETE:
            return
        if task.id in self._watcher_tasks:
            existing = self._watcher_tasks[task.id]
            if not existing.done():
                return  # watcher already running

        try:
            loop = asyncio.get_event_loop()
            watcher = loop.create_task(self._due_watcher_coro(task))
            self._watcher_tasks[task.id] = watcher
            logger.debug(
                "TaskTracker.schedule_due_watcher: watcher started for %r (due %s)",
                task.id,
                task.due_date.isoformat(),
            )
        except RuntimeError:
            # No running event loop (e.g. called from a sync-only test context).
            logger.warning(
                "TaskTracker.schedule_due_watcher: no running event loop; "
                "watcher not started for task %r",
                task.id,
            )

    async def schedule_due_watcher_async(self, task: Task) -> None:
        """Async version of :py:meth:`schedule_due_watcher`.

        Preferred when already inside an async context because it creates
        the task on the *current* running loop directly.

        Requirements: 13.3.
        """
        if task.due_date is None:
            return
        if task.status == TaskStatus.COMPLETE:
            return
        if task.id in self._watcher_tasks:
            existing = self._watcher_tasks[task.id]
            if not existing.done():
                return

        watcher = asyncio.create_task(self._due_watcher_coro(task))
        self._watcher_tasks[task.id] = watcher
        logger.debug(
            "TaskTracker.schedule_due_watcher_async: watcher started for %r",
            task.id,
        )

    async def on_due_passed(self, task_id: str) -> None:
        """Directly trigger the due-passed prompt for *task_id*.

        This is the main entry point for **external** callers (e.g., the
        Orchestrator or a background polling loop) that detect a due date
        has passed and want to immediately surface the prompt to the user.
        It fires the ``due_prompt_callback`` if the task still exists and
        is not yet complete.

        Note: The *automatic* path (within 60 s) is handled by
        :py:meth:`schedule_due_watcher_async`.  This method is for
        callers that perform their own timing.

        Requirements: 13.3.
        """
        task = self._tasks.get(task_id)
        if task is None:
            logger.warning("TaskTracker.on_due_passed: unknown task_id %r", task_id)
            return
        if task.status == TaskStatus.COMPLETE:
            logger.debug(
                "TaskTracker.on_due_passed: task %r already complete; skipping",
                task_id,
            )
            return

        logger.debug("TaskTracker.on_due_passed: prompting for task %r", task_id)
        try:
            await self._due_prompt_callback(task)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TaskTracker.on_due_passed: due_prompt_callback raised for %r: %s",
                task_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _due_watcher_coro(self, task: Task) -> None:
        """Coroutine that fires the due-prompt within 60 s of due_date.

        Algorithm:
        1. Compute ``delay = (task.due_date - now)``.
        2. ``await asyncio.sleep(delay)`` — may be 0 if already past.
        3. Check again: if task is now complete, abort quietly.
        4. Fire ``due_prompt_callback`` immediately (within the 60 s window
           guaranteed by steps 1-3 and the 60 s cap below).
        5. If the task's due date is already in the past by > 60 s at
           schedule time, fire immediately rather than waiting.

        The 60 s budget (Requirement 13.3) is honoured because:
        - If ``delay > 0``: we wake up *exactly* at due_date and fire.
        - If ``delay <= 0`` (due already passed but ≤ 60 s ago): we fire
          immediately.
        - If due date is already > 60 s in the past at the time this watcher
          is started: we still fire immediately (better late than never;
          the requirement is "within 60 s of the due date passing", and
          scheduling a watcher for an already-elapsed task should still
          prompt).

        Requirements: 13.3.
        """
        if task.due_date is None:
            return

        now = self._clock_now()

        # Make both datetimes timezone-aware for comparison.
        due = _ensure_tz_aware(task.due_date)
        now_tz = _ensure_tz_aware(now)

        delay_seconds = (due - now_tz).total_seconds()

        if delay_seconds > 0:
            try:
                await asyncio.sleep(delay_seconds)
            except asyncio.CancelledError:
                logger.debug(
                    "TaskTracker._due_watcher_coro: watcher for %r cancelled during sleep",
                    task.id,
                )
                return

        # Check completion status again after sleeping.
        current_task = self._tasks.get(task.id)
        if current_task is None or current_task.status == TaskStatus.COMPLETE:
            logger.debug(
                "TaskTracker._due_watcher_coro: task %r is complete; not prompting",
                task.id,
            )
            return

        logger.debug(
            "TaskTracker._due_watcher_coro: due date passed for %r; invoking prompt",
            task.id,
        )
        try:
            await self._due_prompt_callback(current_task)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TaskTracker._due_watcher_coro: due_prompt_callback raised for %r: %s",
                task.id,
                exc,
            )

    def _cancel_watcher(self, task_id: str) -> None:
        """Cancel the asyncio watcher task for *task_id* if it exists."""
        watcher = self._watcher_tasks.pop(task_id, None)
        if watcher is not None and not watcher.done():
            watcher.cancel()
            logger.debug(
                "TaskTracker._cancel_watcher: watcher for %r cancelled",
                task_id,
            )


# ---------------------------------------------------------------------------
# Timezone helpers
# ---------------------------------------------------------------------------


def _ensure_tz_aware(dt: datetime.datetime) -> datetime.datetime:
    """Return *dt* with UTC timezone if it is naive."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# SQLiteTaskStore — durable persistence for TaskTracker (Req 13.1, 13.7)
# ---------------------------------------------------------------------------


class SQLiteTaskStore:
    """SQLite-backed persistence adapter for :class:`TaskTracker`.

    Provides ``persist`` and ``load`` callables that satisfy the
    :class:`TaskTracker` constructor contract.  Use the
    :func:`create_sqlite_task_tracker` factory to wire both together.

    Design: Task_Tracker, Data Models (Task & Reminder).
    Requirements: 13.1, 13.7.

    Two tables:
        tasks         — one row per Task.
        prerequisites — one row per Prereq, foreign-keyed to tasks.id.

    SQLite WAL mode is enabled; every write is wrapped in an explicit
    transaction so a mid-write crash leaves the database unchanged
    (atomicity guarantee for Req 13.7).
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # PersistCallback — called by TaskTracker.add()
    # ------------------------------------------------------------------

    def persist(self, task: Task) -> None:
        """Atomically write *task* and its prerequisites.

        All-or-nothing: if any part fails the transaction is rolled back,
        leaving the database unchanged (Req 13.7).

        Raises on any error so ``TaskTracker.add`` can catch and escalate.
        """
        due_iso = task.due_date.isoformat() if task.due_date else None

        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("BEGIN EXCLUSIVE")
            try:
                conn.execute(
                    """
                    INSERT INTO tasks
                        (id, title, description, due_date, severity, status,
                         source, severity_was_defaulted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task.id,
                        task.title,
                        task.description,
                        due_iso,
                        task.severity.value,
                        task.status.value,
                        task.source.value,
                        1 if task.severity_was_defaulted else 0,
                    ),
                )
                for idx, prereq in enumerate(task.prerequisites):
                    conn.execute(
                        """
                        INSERT INTO prerequisites
                            (id, task_id, title, status, sort_idx)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (prereq.id, task.id, prereq.title, prereq.status.value, idx),
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------
    # LoadCallback — called by TaskTracker.__init__()
    # ------------------------------------------------------------------

    def load(self) -> List[Task]:
        """Load all tasks and their prerequisites from the database.

        Returns tasks sorted by due_date ascending (NULLs last) so the
        in-memory store starts in sorted order (Req 13.1, 13.2).
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row

            task_rows = conn.execute(
                """
                SELECT id, title, description, due_date, severity,
                       status, source, severity_was_defaulted
                FROM tasks
                ORDER BY
                    CASE WHEN due_date IS NULL THEN 1 ELSE 0 END,
                    due_date ASC
                """
            ).fetchall()

            task_ids = [row["id"] for row in task_rows]
            prereqs_by_task: Dict[str, List[Prereq]] = {tid: [] for tid in task_ids}

            if task_ids:
                placeholders = ",".join("?" * len(task_ids))
                prereq_rows = conn.execute(
                    f"""
                    SELECT id, task_id, title, status
                    FROM prerequisites
                    WHERE task_id IN ({placeholders})
                    ORDER BY task_id, sort_idx
                    """,
                    task_ids,
                ).fetchall()
                for pr in prereq_rows:
                    prereqs_by_task[pr["task_id"]].append(
                        Prereq(
                            id=pr["id"],
                            title=pr["title"],
                            status=TaskStatus(pr["status"]),
                        )
                    )

        return [
            _row_to_task(row, prereqs_by_task[row["id"]]) for row in task_rows
        ]

    # ------------------------------------------------------------------
    # Private — schema initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id                     TEXT PRIMARY KEY,
                    title                  TEXT NOT NULL,
                    description            TEXT NOT NULL DEFAULT '',
                    due_date               TEXT,
                    severity               TEXT NOT NULL DEFAULT 'DEFAULT',
                    status                 TEXT NOT NULL DEFAULT 'UPCOMING',
                    source                 TEXT NOT NULL DEFAULT 'manual',
                    severity_was_defaulted INTEGER NOT NULL DEFAULT 0,
                    created_at             TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS prerequisites (
                    id        TEXT NOT NULL,
                    task_id   TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                    title     TEXT NOT NULL DEFAULT '',
                    status    TEXT NOT NULL DEFAULT 'UPCOMING',
                    sort_idx  INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (id, task_id)
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_due_date ON tasks(due_date);
                CREATE INDEX IF NOT EXISTS idx_tasks_status   ON tasks(status);
                """
            )
            conn.commit()


def _row_to_task(row: sqlite3.Row, prereqs: List[Prereq]) -> Task:
    """Reconstruct a :class:`Task` from a database row."""
    due_date: Optional[datetime.datetime] = None
    if row["due_date"]:
        due_date = datetime.datetime.fromisoformat(row["due_date"])

    return Task(
        id=row["id"],
        title=row["title"],
        description=row["description"],
        due_date=due_date,
        severity=Severity(row["severity"]),
        status=TaskStatus(row["status"]),
        source=TaskSource(row["source"]),
        severity_was_defaulted=bool(row["severity_was_defaulted"]),
        prerequisites=prereqs,
    )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_sqlite_task_tracker(
    db_path: Path,
    reminder_stopper: Optional[ReminderStopperCallback] = None,
    due_prompt_callback: Optional[DuePromptCallback] = None,
    on_add_failure: Optional[AddFailureCallback] = None,
) -> TaskTracker:
    """Create a :class:`TaskTracker` backed by a SQLite database at *db_path*.

    Wire the :class:`SQLiteTaskStore` ``persist`` and ``load`` callables
    into the TaskTracker and return it ready to use.

    Requirements: 13.1, 13.7.
    """
    store = SQLiteTaskStore(db_path=db_path)
    return TaskTracker(
        reminder_stopper=reminder_stopper,
        due_prompt_callback=due_prompt_callback,
        on_add_failure=on_add_failure,
        persist=store.persist,
        load=store.load,
    )
