"""
Orchestrator sub-package.

Owns the conversational turn loop, intent routing, and cancellation.
The Orchestrator sequences subsystems and manages turn lifecycle; it does not
contain capability logic itself.

Design reference: Architecture → The Orchestrator, Intent Routing.
Requirements: 3, 4, 5, 6, 7, 23.
"""

from .orchestrator import Orchestrator

__all__ = ["Orchestrator"]
