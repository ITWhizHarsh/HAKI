"""
NamedAutomation data model.

A NamedAutomation pairs a human-readable name with an ordered sequence
of Step objects that define the automation's behaviour.  The name is the
stable key used to retrieve and run the automation via the
Automation_Library.

Design reference: Automation_Library, Data Models (NamedAutomation).
Requirements: 17.1, 17.3.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.planner import Step, CommandPlan


@dataclass
class NamedAutomation:
    """
    A named, reusable automation stored in the Automation_Library.

    Attributes
    ----------
    name:
        The unique human-readable name for this automation (e.g.
        "morning_brief", "send_weekly_report").  Looked up by exact
        name at run time (Req 17.2, 17.4).
    steps:
        Ordered list of :class:`~core.planner.Step` objects that define
        what the automation does.  The order is preserved and passed to
        the Execution_Engine as a :class:`~core.planner.CommandPlan`
        when the automation runs (Req 17.2, 17.3).
    description:
        Optional human-readable description of what the automation does.
        Presented to the user in listings and confirmation messages.
    id:
        Stable internal unique identifier (UUID).  Not used for lookup —
        that is done by ``name`` (exact match, Req 17.3).

    Design: Automation_Library, Data Models.
    Requirements: 17.1, 17.3.
    """

    name: str
    steps: list["Step"] = field(default_factory=list)
    description: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("NamedAutomation.name must be a non-empty string.")
        if not isinstance(self.steps, list):
            raise TypeError("NamedAutomation.steps must be a list of Step objects.")

    def to_command_plan(self) -> "CommandPlan":
        """
        Wrap this automation's steps in a :class:`~core.planner.CommandPlan`.

        The resulting plan can be handed directly to the
        :class:`~core.execution.ExecutionEngine` for execution.

        Design: Automation_Library.
        Requirements: 17.2.
        """
        from core.planner import CommandPlan

        return CommandPlan(
            id=str(uuid.uuid4()),
            origin_command=f"run automation: {self.name}",
            steps=list(self.steps),
        )
