"""
Planner — CommandPlan / Step data model and LLM-based plan generation.

Converts a natural-language command into an ordered, dependency-aware
CommandPlan.  Each Step is annotated with its actuator backend and safety
classification (CONSEQUENTIAL | REVERSIBLE | UNKNOWN).  Memory-backed
slot filling avoids prompting the user for facts already in Memory_Brain.

Design: Data Models (CommandPlan & Step), Planning.
Requirements: 17.2, 21.1, 21.7, 22.1, 22.4, 22.7.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class StepStatus(str, Enum):
    PENDING = "pending"
    AWAITING_CONFIRM = "awaiting_confirm"
    AWAITING_CLARIFY = "awaiting_clarify"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepClassification(str, Enum):
    CONSEQUENTIAL = "consequential"  # Req 22.1: requires confirmation
    REVERSIBLE = "reversible"        # Req 22.1: runs without confirmation
    UNKNOWN = "unknown"              # Req 22.4: treated as CONSEQUENTIAL


class Actuator(str, Enum):
    APPLESCRIPT = "applescript"
    AX = "ax"
    APPLE_EVENTS = "apple_events"
    CDP = "cdp"
    VISION = "vision"
    CALENDAR = "calendar"
    NOTIFICATIONS = "notifications"
    INTERNAL = "internal"  # Pure-Python Core steps (no OS actuation)


class SlotStatus(str, Enum):
    """
    Lifecycle state of a dialogue slot (Req 23.1).

    PENDING         Slot is declared but not yet resolved.
    FILLED          Slot value was provided (from memory or user).
    DECLINED        User explicitly declined to provide the slot value.
    DEFAULT_APPLIED The slot was given a default value because the user
                    declined and a default was available (Req 23.6).
    """

    PENDING = "pending"
    FILLED = "filled"
    DECLINED = "declined"
    DEFAULT_APPLIED = "default_applied"


# ---------------------------------------------------------------------------
# Slot data model (Task 21.1)
# ---------------------------------------------------------------------------


@dataclass
class Slot:
    """
    A named dialogue slot that must be resolved before a Step can run.

    Slots are used by the Dialogue_Manager to track what information is
    needed from the user (or from Memory_Brain) before a step executes.
    The Planner declares slot names in ``Step.required_slots``; the
    Dialogue_Manager instantiates full ``Slot`` objects as it resolves
    them (Req 23.1, 23.2, 23.6).

    Attributes
    ----------
    name    The slot identifier, e.g. "recipient", "professor_email".
    value   The resolved value, or ``None`` when still PENDING.
    status  Current resolution state (see :class:`SlotStatus`).
    default Default value to apply when the user declines (Req 23.6).

    Design: Data Models (CommandPlan & Step).
    Requirements: 23.1, 23.2, 23.6.
    """

    name: str
    value: Any | None = None
    status: SlotStatus = SlotStatus.PENDING
    default: Any | None = None

    def fill(self, value: Any) -> None:
        """Mark this slot as FILLED with *value*."""
        self.value = value
        self.status = SlotStatus.FILLED

    def decline(self) -> None:
        """
        Handle a user declining to provide this slot.

        If a default is configured, apply it (Req 23.6); otherwise mark
        the slot as DECLINED so the Execution_Engine can decide whether
        to abandon the step.
        """
        if self.default is not None:
            self.value = self.default
            self.status = SlotStatus.DEFAULT_APPLIED
        else:
            self.status = SlotStatus.DECLINED

    @property
    def is_resolved(self) -> bool:
        """Return ``True`` when the slot has a usable value."""
        return self.status in (SlotStatus.FILLED, SlotStatus.DEFAULT_APPLIED)


# ---------------------------------------------------------------------------
# Step data model (Task 21.1)
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """
    A single executable unit within a CommandPlan.

    Attributes
    ----------
    id              Stable unique identifier for this step.
    intent          Human-readable description of the step's goal,
                    e.g. "open app", "send message", "click element".
    actuator        Backend that will execute this step (Req 21.1).
    args            Named arguments for the actuator (e.g. {"app": "Safari"}).
    depends_on      IDs of steps that must COMPLETE before this step may
                    start.  Defines the dependency graph that drives
                    ordering and parallelism (Req 17.2).
    classification  Safety class; UNKNOWN is treated as CONSEQUENTIAL at
                    execution time (Reqs 22.1, 22.4, 22.7).
    required_slots  Slot names that must be resolved (from memory or user)
                    before this step runs.  The Dialogue_Manager fills
                    these from Memory_Brain first (Req 23.1, 23.2).
    status          Lifecycle state of this step (Req 21.1).

    Design: Data Models (CommandPlan & Step).
    Requirements: 17.2, 21.1, 22.1, 22.4, 22.7, 23.1.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    intent: str = ""
    actuator: Actuator = Actuator.INTERNAL
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)  # IDs of prerequisite steps
    classification: StepClassification = StepClassification.UNKNOWN
    required_slots: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING


# ---------------------------------------------------------------------------
# CommandPlan data model + dependency graph (Task 21.1)
# ---------------------------------------------------------------------------


@dataclass
class CommandPlan:
    """
    Ordered sequence of Steps generated from a single NL command.

    The ``steps`` list is the canonical ordered sequence, but execution
    is governed by the dependency graph expressed via ``Step.depends_on``
    rather than positional order alone.  Independent steps may run in
    parallel; steps must not start until all ``depends_on`` predecessors
    have COMPLETED (Req 17.2).

    Design: Data Models (CommandPlan & Step).
    Requirements: 21.1.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    origin_command: str = ""
    steps: list[Step] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Dependency graph helpers (Req 17.2)
    # ------------------------------------------------------------------

    def ready_steps(self) -> list[Step]:
        """
        Return steps that are PENDING and whose dependencies have all
        COMPLETED.

        Independent steps (empty depends_on) are always ready as long as
        their status is PENDING.  This enables parallel execution of
        steps that do not depend on each other (Req 17.2).

        Returns
        -------
        list[Step]
            Steps in the original list order that are ready to run.
        """
        completed_ids = {
            s.id for s in self.steps if s.status == StepStatus.COMPLETED
        }
        return [
            s
            for s in self.steps
            if s.status == StepStatus.PENDING
            and set(s.depends_on).issubset(completed_ids)
        ]

    def dependent_steps(self, step_id: str) -> list[Step]:
        """
        Return all steps that directly depend on *step_id*.

        Used when a step fails or is rejected: its direct dependents
        must be SKIPPED (and their dependents transitively) so
        independent steps can still continue (Reqs 17.7, 21.14).

        Parameters
        ----------
        step_id:
            The ID of the step whose direct dependents are requested.

        Returns
        -------
        list[Step]
            Direct dependents (not transitive) still in PENDING state.
        """
        return [
            s
            for s in self.steps
            if step_id in s.depends_on and s.status == StepStatus.PENDING
        ]

    def transitive_dependents(self, step_id: str) -> list[Step]:
        """
        Return all steps that transitively depend on *step_id*.

        Performs a BFS from *step_id* through the ``depends_on`` graph
        and returns every downstream step that is still PENDING.  These
        should be SKIPPEd when *step_id* fails or is rejected (Reqs 17.7,
        21.9, 21.12, 21.13, 21.14).

        Parameters
        ----------
        step_id:
            The ID of the failed / rejected step.

        Returns
        -------
        list[Step]
            All transitive PENDING dependents in BFS order.
        """
        # Build forward adjacency: predecessor → set of direct dependents
        adjacency: dict[str, set[str]] = defaultdict(set)
        step_map: dict[str, Step] = {s.id: s for s in self.steps}
        for s in self.steps:
            for dep in s.depends_on:
                adjacency[dep].add(s.id)

        visited: set[str] = set()
        queue: deque[str] = deque([step_id])
        result: list[Step] = []

        while queue:
            current = queue.popleft()
            for child_id in adjacency.get(current, set()):
                if child_id in visited:
                    continue
                visited.add(child_id)
                child = step_map.get(child_id)
                if child and child.status == StepStatus.PENDING:
                    result.append(child)
                    queue.append(child_id)

        return result

    def topological_order(self) -> list[Step]:
        """
        Return the steps in a valid topological execution order.

        Uses Kahn's algorithm (BFS-based) so that when multiple orderings
        are valid, steps appear as early as possible (enabling maximum
        parallelism).  Raises :class:`CyclicDependencyError` if a cycle
        is detected in the dependency graph.

        Returns
        -------
        list[Step]
            All steps ordered such that each step appears after all its
            ``depends_on`` predecessors.

        Raises
        ------
        CyclicDependencyError
            When the dependency graph contains a cycle, making execution
            impossible.
        """
        step_map: dict[str, Step] = {s.id: s for s in self.steps}
        in_degree: dict[str, int] = {s.id: 0 for s in self.steps}
        adjacency: dict[str, list[str]] = {s.id: [] for s in self.steps}

        for s in self.steps:
            for dep_id in s.depends_on:
                if dep_id in adjacency:
                    adjacency[dep_id].append(s.id)
                    in_degree[s.id] += 1

        queue: deque[str] = deque(
            s_id for s_id, deg in in_degree.items() if deg == 0
        )
        ordered: list[Step] = []

        while queue:
            current_id = queue.popleft()
            ordered.append(step_map[current_id])
            for child_id in adjacency.get(current_id, []):
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)

        if len(ordered) != len(self.steps):
            raise CyclicDependencyError(
                f"CommandPlan '{self.id}' has a cyclic dependency. "
                "Execution is impossible."
            )

        return ordered

    def has_cycle(self) -> bool:
        """
        Return ``True`` if the dependency graph contains a cycle.

        Cycles make a plan un-executable (topological sort fails), so
        the Planner should detect them during plan generation and raise
        an error before returning the plan to the caller.
        """
        try:
            self.topological_order()
            return False
        except CyclicDependencyError:
            return True

    def is_complete(self) -> bool:
        """
        Return ``True`` when all steps have reached a terminal state
        (COMPLETED, FAILED, or SKIPPED).
        """
        terminal = {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED}
        return all(s.status in terminal for s in self.steps)

    def completed_steps(self) -> list[Step]:
        """Return the subset of steps whose status is COMPLETED."""
        return [s for s in self.steps if s.status == StepStatus.COMPLETED]

    def failed_steps(self) -> list[Step]:
        """Return the subset of steps whose status is FAILED."""
        return [s for s in self.steps if s.status == StepStatus.FAILED]

    def skipped_steps(self) -> list[Step]:
        """Return the subset of steps whose status is SKIPPED."""
        return [s for s in self.steps if s.status == StepStatus.SKIPPED]


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class CyclicDependencyError(ValueError):
    """
    Raised when the CommandPlan dependency graph contains a cycle.

    A cyclic plan cannot be executed because no valid topological
    ordering exists.
    """


class PlanGenerationError(RuntimeError):
    """
    Raised by :class:`Planner` when the LLM fails to produce a valid plan
    or the response cannot be parsed into a CommandPlan.
    """


# ---------------------------------------------------------------------------
# Planner system prompt (Task 21.2)
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = """\
You are HAKI's command planner. Given a natural-language command and a
memory context (facts already known), produce a structured JSON execution
plan.

## Your task
Convert the command into a CommandPlan: an ordered list of Steps, each
with:
- id: a short unique identifier (e.g. "step_1", "step_2")
- intent: what this step does in plain English
- actuator: one of "applescript" | "ax" | "apple_events" | "cdp" | "vision" | "calendar" | "notifications" | "internal"
- args: a dict of named arguments the actuator needs (e.g. {"app": "Safari", "url": "https://..."})
- depends_on: list of step ids this step must wait for ([] for independent steps)
- classification: "reversible" | "consequential" | "unknown"
  * reversible: opening apps, reading content, opening tabs (no side effects)
  * consequential: sending messages, deleting files, creating calendar events, placing calls, making purchases
  * unknown: when you cannot determine the impact
- required_slots: list of slot names (strings) that need to be filled before this step runs ([] if none)

## Actuator guidance
- applescript: launch apps, open files, system-level scripting
- ax: accessibility actions — click UI elements, read window text
- apple_events: scriptable app control (Mail, Messages, Calendar via AppleScript)
- cdp: Arc/Chromium browser automation — open tabs, navigate, fill forms, click web elements
- vision: screen capture + OCR + element detection for inaccessible UI
- calendar: EventKit calendar read/write
- notifications: issue system notifications
- internal: Python-level logic with no OS actuation

## Memory context
The memory_context field contains facts already known from Memory_Brain.
Use these facts to fill slot values directly instead of marking them as
required_slots. Only list a slot as required if it cannot be resolved
from memory.

## Slot filling from memory (Req 21.7)
If the command references a stored fact (e.g. "email my professor",
"open my notes app"), fill the corresponding arg from the memory context
rather than leaving it as a required slot. Prefer memory-backed values.

## Output format
Respond with ONLY a JSON object, no prose:
{
  "steps": [
    {
      "id": "step_1",
      "intent": "...",
      "actuator": "...",
      "args": {},
      "depends_on": [],
      "classification": "...",
      "required_slots": []
    }
  ]
}

Ensure depends_on references are valid step ids in the same plan.
Steps with no dependencies may run in parallel; use depends_on to enforce
ordering only when strictly necessary.
"""


# ---------------------------------------------------------------------------
# Planner (Task 21.2)
# ---------------------------------------------------------------------------


class Planner:
    """
    LLM-based command planner.

    Converts a natural-language command into a :class:`CommandPlan`.
    Slot values referenced in the command that are already stored in
    Memory_Brain are filled automatically instead of asking the user
    (Req 21.7).

    The planner delegates to the LLM capability of the ``model_provider``
    (via ``invoke``).  When no provider is given it falls back to a
    minimal single-step stub so the rest of the system can function
    without a live model.

    Parameters
    ----------
    model_provider:
        A :class:`~core.model_provider.ModelProvider` configured for the
        ``Capability.LLM`` capability.  When ``None``, planning falls back
        to the stub implementation.
    memory_brain:
        A :class:`~core.memory.MemoryBrain` instance used to retrieve
        relevant facts for slot filling (Req 21.7).  When ``None``,
        memory-backed slot filling is skipped.

    Design: Planning.
    Requirements: 21.1, 21.7.
    """

    def __init__(
        self,
        model_provider: Any | None = None,
        memory_brain: Any | None = None,
    ) -> None:
        self._model_provider = model_provider
        self._memory_brain = memory_brain

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, command: str, memory_context: list[Any] | None = None) -> CommandPlan:
        """
        Generate a :class:`CommandPlan` for *command*.

        Steps:
        1. Retrieve relevant facts from Memory_Brain using *command* as the
           query, unless *memory_context* is explicitly provided (Req 21.7).
        2. Build the LLM prompt with memory context for slot filling.
        3. Call the LLM provider and parse the JSON response into a
           ``CommandPlan``.
        4. Validate the dependency graph; raise :class:`PlanGenerationError`
           if a cycle is detected.

        Falls back to a single-step stub plan when no model provider is
        configured (useful during bootstrapping / unit tests).

        Parameters
        ----------
        command:
            The natural-language command to plan, e.g.
            "open Safari and search for the HAKI GitHub repo".
        memory_context:
            Optional list of :class:`~core.memory.models.Note` objects to
            inject as slot-filling context.  When ``None`` and a
            ``memory_brain`` is configured, facts are retrieved
            automatically.

        Returns
        -------
        CommandPlan
            The generated plan with at least one step.

        Raises
        ------
        PlanGenerationError
            When the LLM response cannot be parsed or the resulting plan
            contains a cyclic dependency.
        """
        # ------------------------------------------------------------------
        # Step 1: Retrieve memory context for slot filling (Req 21.7)
        # ------------------------------------------------------------------
        if memory_context is None:
            memory_context = self._retrieve_memory(command)

        # ------------------------------------------------------------------
        # Step 2: Fall back to stub when no LLM provider is available
        # ------------------------------------------------------------------
        if self._model_provider is None:
            return self._stub_plan(command)

        # ------------------------------------------------------------------
        # Step 3: Build LLM prompt with memory context
        # ------------------------------------------------------------------
        prompt = self._build_prompt(command, memory_context)

        # ------------------------------------------------------------------
        # Step 4: Invoke LLM and parse response
        # ------------------------------------------------------------------
        try:
            raw = self._model_provider.invoke(prompt, system=_PLANNER_SYSTEM_PROMPT)
        except Exception as exc:
            logger.warning("Planner: LLM invocation failed (%s); falling back to stub", exc)
            return self._stub_plan(command)

        # Extract text from various response shapes
        response_text = self._extract_response_text(raw)

        # ------------------------------------------------------------------
        # Step 5: Parse JSON into CommandPlan
        # ------------------------------------------------------------------
        try:
            plan = self._parse_plan(command, response_text)
        except PlanGenerationError as exc:
            logger.warning("Planner: failed to parse LLM plan (%s); falling back to stub", exc)
            return self._stub_plan(command)

        # ------------------------------------------------------------------
        # Step 6: Fill slots from memory context (Req 21.7)
        # ------------------------------------------------------------------
        if memory_context:
            plan = self._fill_slots_from_memory(plan, memory_context)

        return plan

    # ------------------------------------------------------------------
    # Memory retrieval (Req 21.7)
    # ------------------------------------------------------------------

    def _retrieve_memory(self, command: str) -> list[Any]:
        """
        Retrieve relevant notes from Memory_Brain for *command*.

        Returns an empty list if no brain is configured or retrieval
        fails.
        """
        if self._memory_brain is None:
            return []
        try:
            return self._memory_brain.retrieve(command, k=5)
        except Exception as exc:
            logger.warning("Planner: memory retrieval failed (%s); proceeding without context", exc)
            return []

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, command: str, memory_context: list[Any]) -> str:
        """
        Build the user-turn prompt for the LLM planner.

        Includes any memory context as a structured fact list so the LLM
        can fill slots without needing to ask the user (Req 21.7).
        """
        parts: list[str] = [f"Command: {command}"]

        if memory_context:
            parts.append("\nMemory context (known facts — use these to fill slots):")
            for note in memory_context:
                body = getattr(note, "body", str(note)).strip()
                if body:
                    parts.append(f"- {body}")

        parts.append("\nProduce the JSON plan:")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _extract_response_text(self, raw: Any) -> str:
        """
        Extract the response text from various provider return shapes.

        The stub providers return a dict; real LLM APIs may return a
        string, a dict with a ``content`` or ``text`` key, or an object.
        """
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict):
            # Common shapes: {input: ..., ...stub response...} or {content: "..."}
            for key in ("content", "text", "response", "output", "input"):
                if key in raw and isinstance(raw[key], str):
                    return raw[key]
            # Stub returns the prompt as "input" — return as-is for fallback
            return json.dumps(raw)
        return str(raw)

    def _parse_plan(self, command: str, response_text: str) -> CommandPlan:
        """
        Parse *response_text* (expected JSON) into a :class:`CommandPlan`.

        If parsing fails (malformed JSON, missing fields) a
        :class:`PlanGenerationError` is raised, which is caught by
        :meth:`plan` to fall back to the stub implementation.

        Raises
        ------
        PlanGenerationError
            When the response is not valid JSON or is structurally invalid.
        CyclicDependencyError
            When the parsed plan has a cyclic dependency graph.
        """
        # Extract JSON from the response (the model might wrap it in prose)
        json_text = self._extract_json_block(response_text)

        try:
            data = json.loads(json_text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise PlanGenerationError(
                f"Planner: LLM response is not valid JSON: {exc}\n"
                f"Response was: {response_text[:200]!r}"
            ) from exc

        if not isinstance(data, dict) or "steps" not in data:
            raise PlanGenerationError(
                f"Planner: LLM response missing 'steps' key. Got: {data!r}"
            )

        steps: list[Step] = []
        for raw_step in data.get("steps", []):
            if not isinstance(raw_step, dict):
                continue
            step = self._parse_step(raw_step)
            steps.append(step)

        if not steps:
            raise PlanGenerationError(
                "Planner: LLM returned an empty step list."
            )

        plan = CommandPlan(origin_command=command, steps=steps)

        # Validate for cycles (Req 17.2 — a cyclic plan cannot execute)
        if plan.has_cycle():
            raise CyclicDependencyError(
                f"Planner: generated plan for '{command}' contains a "
                "cyclic dependency and cannot be executed."
            )

        return plan

    def _extract_json_block(self, text: str) -> str:
        """
        Extract the first JSON object ``{...}`` from *text*.

        Some LLMs wrap JSON in prose or markdown code fences.  This
        extractor finds the first ``{`` and its matching ``}`` using a
        simple brace-count approach.
        """
        start = text.find("{")
        if start == -1:
            return text  # will fail in json.loads with a clear error

        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        return text[start:]  # unbalanced — pass to json.loads for error

    def _parse_step(self, raw: dict[str, Any]) -> Step:
        """
        Build a :class:`Step` from a raw dict parsed from the LLM JSON.

        Unknown / invalid enum values are coerced to safe defaults:
        - Unknown actuator → ``Actuator.INTERNAL``
        - Unknown classification → ``StepClassification.UNKNOWN`` (safe)
        """
        # Parse actuator — default to INTERNAL on unknown
        actuator_str = str(raw.get("actuator", "internal")).lower()
        try:
            actuator = Actuator(actuator_str)
        except ValueError:
            logger.debug("Planner: unknown actuator '%s'; defaulting to INTERNAL", actuator_str)
            actuator = Actuator.INTERNAL

        # Parse classification — default to UNKNOWN on unknown (fail-safe)
        classification_str = str(raw.get("classification", "unknown")).lower()
        try:
            classification = StepClassification(classification_str)
        except ValueError:
            logger.debug(
                "Planner: unknown classification '%s'; defaulting to UNKNOWN",
                classification_str,
            )
            classification = StepClassification.UNKNOWN

        # Use the LLM-provided step id if present, else generate one
        step_id = str(raw.get("id", "")).strip() or str(uuid.uuid4())

        depends_on = [str(d) for d in raw.get("depends_on", []) if d]
        required_slots = [str(s) for s in raw.get("required_slots", []) if s]
        args = dict(raw.get("args", {})) if isinstance(raw.get("args"), dict) else {}

        return Step(
            id=step_id,
            intent=str(raw.get("intent", "")),
            actuator=actuator,
            args=args,
            depends_on=depends_on,
            classification=classification,
            required_slots=required_slots,
            status=StepStatus.PENDING,
        )

    # ------------------------------------------------------------------
    # Memory-backed slot filling (Req 21.7)
    # ------------------------------------------------------------------

    def _fill_slots_from_memory(
        self,
        plan: CommandPlan,
        memory_context: list[Any],
    ) -> CommandPlan:
        """
        Fill required slots in each step from *memory_context*.

        For every step with non-empty ``required_slots``, this method
        attempts to resolve each slot by searching the memory context for
        a note whose body or topics contain a value for that slot.  Slots
        that are successfully resolved are removed from ``required_slots``
        and their resolved value is added to the step's ``args``.

        Slots that cannot be resolved from memory remain in
        ``required_slots`` so the Dialogue_Manager can ask the user
        (Req 23.1, 23.2).

        Parameters
        ----------
        plan:
            The plan whose steps may have unresolved required slots.
        memory_context:
            Notes retrieved from Memory_Brain for this command.

        Returns
        -------
        CommandPlan
            The same plan with as many slots filled as possible from
            memory.
        """
        if not memory_context:
            return plan

        # Build a flat text corpus from memory for slot resolution
        memory_texts = [
            getattr(note, "body", str(note)).strip()
            for note in memory_context
            if getattr(note, "body", "").strip()
        ]

        for step in plan.steps:
            if not step.required_slots:
                continue

            resolved: list[str] = []
            still_missing: list[str] = []

            for slot in step.required_slots:
                value = self._resolve_slot_from_memory(slot, memory_texts)
                if value is not None:
                    # Inject into args (do not overwrite an existing value)
                    if slot not in step.args:
                        step.args[slot] = value
                    resolved.append(slot)
                else:
                    still_missing.append(slot)

            step.required_slots = still_missing

        return plan

    def _resolve_slot_from_memory(
        self,
        slot_name: str,
        memory_texts: list[str],
    ) -> str | None:
        """
        Attempt to resolve *slot_name* by scanning *memory_texts*.

        This is a heuristic resolver that looks for patterns like
        "slot_name is/are/= value" or "my slot_name is value" in the
        memory notes and returns the first candidate value found.

        The slot name is normalised for matching: underscores are replaced
        with a flexible whitespace pattern so both "professor_email" and
        "professor email" match the same note text.

        Returns ``None`` when no candidate is found so the slot stays in
        ``required_slots`` for the Dialogue_Manager to handle.
        """
        import re

        # Build a flexible regex key: each underscore becomes \s+ so
        # "professor_email" matches "professor email" in the note text.
        slot_parts = [re.escape(p) for p in slot_name.split("_") if p]
        if not slot_parts:
            return None
        slot_pattern = r"\s+".join(slot_parts)

        for text in memory_texts:
            text_lower = text.lower()

            # Quick pre-filter: at least the first word of the slot must appear
            first_part = slot_parts[0].lower()
            if first_part not in text_lower:
                continue

            # Patterns: "slot_key is|are|:|= VALUE" or "my slot_key is VALUE"
            value_pattern = r"([^\n,;]{2,60})"
            patterns = [
                rf"(?:my\s+)?{slot_pattern}\s+(?:is|are|=|:)\s+{value_pattern}",
                rf"{slot_pattern}\s+{value_pattern}\s*(?:$|[,;.\n])",
            ]

            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    value = match.group(1).strip().rstrip(".,;")
                    # Reject values that are only stop words or too short
                    if value and len(value) >= 2:
                        return value

        return None

    # ------------------------------------------------------------------
    # Stub fallback
    # ------------------------------------------------------------------

    def _stub_plan(self, command: str) -> CommandPlan:
        """
        Fallback stub plan used when no LLM provider is configured.

        Returns a single-step plan with classification UNKNOWN.
        This keeps the rest of the system functional during bootstrapping.
        """
        step = Step(
            intent=command,
            actuator=Actuator.INTERNAL,
            classification=StepClassification.UNKNOWN,
        )
        return CommandPlan(origin_command=command, steps=[step])
