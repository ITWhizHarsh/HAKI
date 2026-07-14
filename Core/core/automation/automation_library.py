"""
AutomationLibrary — stores and executes named automations via the shared
Execution_Engine.

The library is the single registry of all user-defined (and built-in)
named automations.  It reuses the same :class:`~core.execution.ExecutionEngine`
that runs ad-hoc :class:`~core.planner.CommandPlan` objects so that step
execution, cancellation, safety gating, and progress reporting are all
shared (design: Automation_Library + Execution_Engine).

Public contract (from design)
-------------------------------
  AutomationLibrary:
    define(name, steps) / get(name) / nearest(name)    (17.1, 17.3, 17.4)
    run(name) -> AsyncIterator[StepEvent]               (17.2, 17.4)
    cancel()                                            (17.5, 17.6)

Key behaviours
--------------
* ``define`` — stores the automation; duplicate names replace the old entry
  so that automations can be updated.
* ``get`` — exact-name lookup; raises :class:`AutomationNotFoundError` when
  the name is not registered.
* ``nearest`` — returns the stored automation name with the smallest
  Levenshtein edit distance to the query.  Used by ``run()`` to suggest
  the closest match when no exact match exists (Req 17.4).
* ``run`` — wraps the automation's steps into a fresh :class:`CommandPlan`
  and passes it to the shared :class:`ExecutionEngine`.  Streams
  :class:`StepEvent` objects back to the caller so the caller always knows
  which step is currently executing (Req 17.2).  On no exact match, yields
  a single :class:`StepEvent` of type ``FAILED`` carrying the suggestion
  message and **does not start any execution** (Req 17.4).
* ``cancel`` — delegates to the shared :class:`ExecutionEngine`'s cancel()
  method, stopping unstarted steps and interrupting the in-progress step
  within its cancellation bound (Req 17.5, 17.6).

Design: Automation_Library + Execution_Engine.
Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import AsyncGenerator, AsyncIterator, Optional

from core.execution.execution_engine import ExecutionEngine, StepEvent, StepEventType
from core.planner import Actuator, CommandPlan, Step, StepClassification, StepStatus

from .models import NamedAutomation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class AutomationNotFoundError(KeyError):
    """
    Raised by :meth:`AutomationLibrary.get` when no automation is
    registered under the exact requested name.
    """


# ---------------------------------------------------------------------------
# Edit-distance helper (Levenshtein)
# ---------------------------------------------------------------------------


def _levenshtein(a: str, b: str) -> int:
    """
    Compute the Levenshtein edit distance between strings *a* and *b*.

    Uses a standard DP approach with O(min(len(a), len(b))) space.
    Case-insensitive comparison is performed by callers (the library
    normalises names when storing and querying).

    Parameters
    ----------
    a, b:
        The strings to compare.

    Returns
    -------
    int
        The number of single-character edits (insertions, deletions,
        substitutions) required to transform *a* into *b*.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Ensure a is the shorter string for memory efficiency.
    if len(a) > len(b):
        a, b = b, a

    prev = list(range(len(a) + 1))
    for j, ch_b in enumerate(b, 1):
        curr = [j] + [0] * len(a)
        for i, ch_a in enumerate(a, 1):
            if ch_a == ch_b:
                curr[i] = prev[i - 1]
            else:
                curr[i] = 1 + min(prev[i], curr[i - 1], prev[i - 1])
        prev = curr

    return prev[len(a)]


# ---------------------------------------------------------------------------
# AutomationLibrary
# ---------------------------------------------------------------------------


class AutomationLibrary:
    """
    Registry of named automations and their execution dispatcher.

    Named automations are persisted to an optional SQLite database so
    they survive HAKI restarts (Req 17.1).  When ``db_path`` is ``None``
    (default), the library operates in-memory only.

    Parameters
    ----------
    execution_engine:
        The shared :class:`~core.execution.ExecutionEngine` instance.
        Must be the same engine used by the Mac_Controller for ad-hoc
        plans so that cancellation, safety gating, and step-event
        streaming are all consistent (design: Automation_Library +
        Execution_Engine).  When ``None``, a default engine (no
        actuator, no safety gate) is created — useful for testing.
    db_path:
        Optional path to the SQLite database file where automations are
        persisted.  The parent directory is created automatically.  Pass
        ``None`` (default) to keep automations in-memory only.

    Design: Automation_Library + Execution_Engine.
    Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7.
    """

    def __init__(
        self,
        execution_engine: ExecutionEngine | None = None,
        db_path: Optional[Path] = None,
    ) -> None:
        self._engine: ExecutionEngine = execution_engine or ExecutionEngine()
        # Internal store: exact name → NamedAutomation
        self._store: dict[str, NamedAutomation] = {}

        # SQLite persistence (optional — only when db_path is provided)
        self._db_path: Optional[str] = None
        if db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db_path = str(db_path)
            with self._open_conn() as conn:
                self._init_schema(conn)
            self._hydrate_store()

    # ------------------------------------------------------------------
    # Storage interface (Req 17.1, 17.3)
    # ------------------------------------------------------------------

    def define(self, name: str, steps: list[Step], description: str = "") -> NamedAutomation:
        """
        Register a new named automation (or replace an existing one).

        Duplicate names are allowed — the new definition replaces the
        old one so that automations can be updated without deleting and
        re-creating them.

        Parameters
        ----------
        name:
            Unique automation name.  Must be a non-empty string.
        steps:
            Ordered list of :class:`~core.planner.Step` objects.
        description:
            Optional human-readable description.

        Returns
        -------
        NamedAutomation
            The stored :class:`NamedAutomation` instance.

        Requirements: 17.1.
        """
        if not name:
            raise ValueError("Automation name must be a non-empty string.")
        automation = NamedAutomation(name=name, steps=steps, description=description)
        # Persist to SQLite if configured (before updating in-process store).
        if self._db_path is not None:
            self._persist(automation)
        self._store[name] = automation
        logger.debug("AutomationLibrary.define: stored '%s' (%d steps)", name, len(steps))
        return automation

    def get(self, name: str) -> NamedAutomation:
        """
        Return the automation registered under *name*.

        Parameters
        ----------
        name:
            Exact automation name (case-sensitive).

        Returns
        -------
        NamedAutomation

        Raises
        ------
        AutomationNotFoundError
            When no automation is registered under *name*.

        Requirements: 17.3.
        """
        if name in self._store:
            return self._store[name]
        raise AutomationNotFoundError(
            f"No automation named '{name}' is registered."
        )

    def list_names(self) -> list[str]:
        """
        Return all registered automation names in insertion order.

        Returns
        -------
        list[str]
            Names of all stored automations.
        """
        return list(self._store.keys())

    # ------------------------------------------------------------------
    # Nearest-name suggestion (Req 17.4)
    # ------------------------------------------------------------------

    def nearest(self, name: str) -> str | None:
        """
        Return the registered automation name that is closest to *name*
        by Levenshtein edit distance.

        When the library is empty, returns ``None``.

        This method is used by :meth:`run` to produce a helpful suggestion
        when the user requests an automation that does not exactly match
        any stored name (Req 17.4).

        Parameters
        ----------
        name:
            The (potentially misspelled or approximate) automation name
            requested by the user.

        Returns
        -------
        str | None
            The stored name with the smallest edit distance, or ``None``
            when the library is empty.

        Requirements: 17.4.
        """
        if not self._store:
            return None

        best_name: str | None = None
        best_dist: int = 2**31 - 1

        for stored_name in self._store:
            dist = _levenshtein(name.lower(), stored_name.lower())
            if dist < best_dist:
                best_dist = dist
                best_name = stored_name

        return best_name

    # ------------------------------------------------------------------
    # Run interface (Req 17.2, 17.4, 17.5, 17.6, 17.7)
    # ------------------------------------------------------------------

    async def run(self, name: str) -> AsyncGenerator[StepEvent, None]:  # type: ignore[misc]
        """
        Execute the automation registered under *name* via the shared
        :class:`~core.execution.ExecutionEngine`.

        If *name* exactly matches a stored automation, its steps are
        wrapped into a fresh :class:`~core.planner.CommandPlan` and
        passed to the engine.  :class:`StepEvent` objects are streamed
        back to the caller, allowing it to report the currently executing
        step (Req 17.2).

        If *name* does NOT exactly match any stored automation:
        - No execution takes place (Req 17.4).
        - A single ``FAILED`` :class:`StepEvent` is yielded whose
          ``message`` contains the nearest suggestion so the user knows
          what they might have meant.
        - ``nearest(name)`` is called internally to compute the suggestion.

        Parameters
        ----------
        name:
            Exact automation name to run.

        Yields
        ------
        StepEvent
            Lifecycle events for each step, ending with a
            ``PLAN_COMPLETE`` event that carries the final
            :class:`~core.execution.ExecutionReport` in ``event.data``.

        Requirements: 17.2, 17.4, 17.5, 17.6, 17.7.
        """
        # ------------------------------------------------------------------
        # Exact-name check (Req 17.4: no fuzzy execution)
        # ------------------------------------------------------------------
        if name not in self._store:
            suggestion = self.nearest(name)
            if suggestion is not None:
                suggestion_msg = (
                    f"No automation named '{name}' found. "
                    f"Did you mean '{suggestion}'?"
                )
            else:
                suggestion_msg = (
                    f"No automation named '{name}' found and the library "
                    f"is empty — no suggestions available."
                )
            logger.info(
                "AutomationLibrary.run: no exact match for '%s'. %s",
                name, suggestion_msg,
            )
            # Yield one FAILED event and stop — do not execute anything.
            yield StepEvent(
                event_type=StepEventType.FAILED,
                step_id=None,
                step=None,
                message=suggestion_msg,
                data={"requested_name": name, "nearest_name": suggestion},
            )
            return

        # ------------------------------------------------------------------
        # Build a fresh CommandPlan from the automation's steps (Req 17.2)
        # ------------------------------------------------------------------
        automation = self._store[name]
        plan = CommandPlan(
            id=str(uuid.uuid4()),
            origin_command=f"run_automation:{name}",
            steps=list(automation.steps),  # shallow copy to preserve originals
        )

        logger.info(
            "AutomationLibrary.run: starting '%s' (%d steps) via ExecutionEngine",
            name, len(plan.steps),
        )

        # ------------------------------------------------------------------
        # Stream execution events (Req 17.2, 17.5, 17.6, 17.7)
        # ------------------------------------------------------------------
        stream = await self._engine.execute(plan)
        async for event in stream:
            yield event

    # ------------------------------------------------------------------
    # Cancellation (Req 17.5, 17.6)
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """
        Cancel the currently executing automation.

        Delegates to the shared :class:`~core.execution.ExecutionEngine`'s
        ``cancel()`` method, which:
        - Stops all unstarted steps immediately (Req 17.6).
        - Interrupts the in-progress step within its cancellation bound
          (Req 17.6).
        - Partitions the final report into completed vs. not-performed
          steps (Req 17.7 / 21.14).

        Requirements: 17.5, 17.6.
        """
        logger.info("AutomationLibrary.cancel: delegating to ExecutionEngine.")
        self._engine.cancel()

    # ------------------------------------------------------------------
    # SQLite persistence helpers (Req 17.1)
    # ------------------------------------------------------------------

    def _open_conn(self) -> sqlite3.Connection:
        """Open and return a new file-backed SQLite connection."""
        conn = sqlite3.connect(self._db_path)  # type: ignore[arg-type]
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        """Create the automations table if it does not exist."""
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS automations (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL DEFAULT '',
                steps_json  TEXT NOT NULL DEFAULT '[]',
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_automations_name ON automations(name);
            """
        )
        conn.commit()

    def _persist(self, automation: NamedAutomation) -> None:
        """
        Atomically upsert *automation* into the SQLite database.

        Uses INSERT ... ON CONFLICT(name) DO UPDATE so an existing entry
        with the same name is replaced atomically (Req 17.1).
        """
        steps_json = json.dumps(
            [_step_to_dict(s) for s in automation.steps],
            ensure_ascii=False,
        )
        sql = """
            INSERT INTO automations (id, name, description, steps_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                id          = excluded.id,
                description = excluded.description,
                steps_json  = excluded.steps_json,
                updated_at  = datetime('now')
        """
        with self._open_conn() as conn:
            conn.execute("BEGIN EXCLUSIVE")
            try:
                conn.execute(
                    sql,
                    (automation.id, automation.name, automation.description, steps_json),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def _hydrate_store(self) -> None:
        """Load all automations from the SQLite database into the in-process store."""
        with self._open_conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, name, description, steps_json FROM automations"
            ).fetchall()

        for row in rows:
            try:
                raw_steps: list = json.loads(row["steps_json"])
                steps = [_step_from_dict(d) for d in raw_steps]
                automation = NamedAutomation(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    steps=steps,
                )
                self._store[automation.name] = automation
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "AutomationLibrary._hydrate_store: failed to load '%s': %s",
                    row["name"],
                    exc,
                )


# ---------------------------------------------------------------------------
# Step serialisation helpers (used by SQLite persistence)
# ---------------------------------------------------------------------------


def _step_to_dict(step: Step) -> dict:
    """
    Serialise a :class:`~core.planner.Step` to a plain dict for JSON storage.

    The ``postcondition`` callable is excluded because it cannot be
    serialised to JSON.  It will be ``None`` when the step is deserialised.
    """
    return {
        "id": step.id,
        "intent": step.intent,
        "actuator": step.actuator.value,
        "args": step.args,
        "depends_on": step.depends_on,
        "classification": step.classification.value,
        "required_slots": step.required_slots,
        # status is always PENDING on reload (execution state is transient)
        "status": StepStatus.PENDING.value,
    }


def _step_from_dict(d: dict) -> Step:
    """
    Deserialise a Step from a plain dict produced by :func:`_step_to_dict`.

    Unknown enum values are coerced to safe defaults.
    """
    actuator_str = str(d.get("actuator", "internal")).lower()
    try:
        actuator = Actuator(actuator_str)
    except ValueError:
        actuator = Actuator.INTERNAL

    classification_str = str(d.get("classification", "unknown")).lower()
    try:
        classification = StepClassification(classification_str)
    except ValueError:
        classification = StepClassification.UNKNOWN

    return Step(
        id=str(d.get("id", "")),
        intent=str(d.get("intent", "")),
        actuator=actuator,
        args=dict(d.get("args", {})),
        depends_on=list(d.get("depends_on", [])),
        classification=classification,
        required_slots=list(d.get("required_slots", [])),
        status=StepStatus.PENDING,
        postcondition=None,
    )
