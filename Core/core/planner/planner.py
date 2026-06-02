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

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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


@dataclass
class Step:
    """
    A single executable unit within a CommandPlan.

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


@dataclass
class CommandPlan:
    """
    Ordered sequence of Steps generated from a single NL command.

    Design: Data Models (CommandPlan & Step).
    Requirements: 21.1.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    origin_command: str = ""
    steps: list[Step] = field(default_factory=list)

    def ready_steps(self) -> list[Step]:
        """
        Return steps that are PENDING and whose dependencies have all
        COMPLETED (enables parallelism, Req 17.2).
        """
        completed_ids = {s.id for s in self.steps if s.status == StepStatus.COMPLETED}
        return [
            s
            for s in self.steps
            if s.status == StepStatus.PENDING and set(s.depends_on).issubset(completed_ids)
        ]


class Planner:
    """
    LLM-based command planner.

    Converts a natural-language command into a CommandPlan.  Slot values
    referenced in the command that are already stored in Memory_Brain are
    filled automatically instead of asking the user (Req 21.7).

    This is a stub implementation; full LLM dispatch added in Task 21.2.
    """

    def __init__(self, memory_brain: Any | None = None) -> None:
        self._memory_brain = memory_brain

    def plan(self, command: str) -> CommandPlan:
        """
        Generate a CommandPlan for *command*.

        Stub: returns a single-step plan with classification UNKNOWN.
        Full LLM-based planning added in Task 21.2.
        """
        step = Step(
            intent=command,
            actuator=Actuator.INTERNAL,
            classification=StepClassification.UNKNOWN,
        )
        return CommandPlan(origin_command=command, steps=[step])
