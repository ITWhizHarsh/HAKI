"""
ActionableItem data model for the Comms_Reader subsystem.

Represents a calendar event, task, or reminder extracted from an incoming
WhatsApp message or email.

Design: Comms_Reader, Data Models (ActionableItem).
Requirements: 10.1, 10.2, 10.3, 10.4.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ActionableType(str, Enum):
    """Classification of an actionable item (Req 10.3)."""

    EVENT = "EVENT"
    TASK = "TASK"
    REMINDER = "REMINDER"


@dataclass
class ActionableItem:
    """
    A single actionable item extracted from a message or email.

    Attributes
    ----------
    id:
        Stable unique identifier for this item.
    source_account:
        The account identifier (e.g. email address or WhatsApp account name)
        from which the message was received.
    source_message_id:
        Identifier for the originating message (e.g. message UID or hash).
    type:
        Classification of the actionable — EVENT, TASK, or REMINDER (Req 10.3).
    description:
        Human-readable description extracted from the message (Req 10.3).
    date:
        Optional extracted date string (e.g. "2024-06-15") (Req 10.3).
    time:
        Optional extracted time string (e.g. "14:30") (Req 10.3).
    location:
        Optional extracted location string (Req 10.3).
    needs_clarification:
        ``True`` when the item is missing an explicit date or explicit time
        (Req 10.4).  Set automatically via :meth:`compute_needs_clarification`.
    """

    id: str
    source_account: str
    source_message_id: str
    type: ActionableType
    description: str
    date: Optional[str] = None
    time: Optional[str] = None
    location: Optional[str] = None
    needs_clarification: bool = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        source_account: str,
        source_message_id: str,
        type: ActionableType,
        description: str,
        date: Optional[str] = None,
        time: Optional[str] = None,
        location: Optional[str] = None,
    ) -> "ActionableItem":
        """
        Create a new ActionableItem with a generated UUID, computing
        ``needs_clarification`` automatically.

        ``needs_clarification`` is ``True`` when either *date* or *time* is
        absent — or both — because the item cannot be scheduled precisely
        without that information (Req 10.4).
        """
        needs_clarification = (date is None) or (time is None)
        return cls(
            id=str(uuid.uuid4()),
            source_account=source_account,
            source_message_id=source_message_id,
            type=type,
            description=description,
            date=date,
            time=time,
            location=location,
            needs_clarification=needs_clarification,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def compute_needs_clarification(self) -> bool:
        """
        (Re)compute and update ``needs_clarification`` based on the current
        ``date`` and ``time`` fields.

        Returns the computed value.  Per Req 10.4 an item is flagged when
        it lacks an explicit date OR an explicit time.
        """
        self.needs_clarification = (self.date is None) or (self.time is None)
        return self.needs_clarification

    def to_dict(self) -> dict:
        """Serialise to a plain dict (useful for IPC / notifications)."""
        return {
            "id": self.id,
            "source_account": self.source_account,
            "source_message_id": self.source_message_id,
            "type": self.type.value,
            "description": self.description,
            "date": self.date,
            "time": self.time,
            "location": self.location,
            "needs_clarification": self.needs_clarification,
        }
