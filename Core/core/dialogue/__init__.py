"""
Dialogue sub-package.

Owns the Dialogue_Manager: ambiguity detection, slot filling, clarifying
questions, mid-plan pause/resume, and disambiguation of multiple
candidate contacts / options.

Design reference: Dialogue_Manager, Agentic Execution Engine.
Requirements: 23.
"""

from .dialogue_manager import SlotFillResult, DialogueManager

__all__ = ["SlotFillResult", "DialogueManager"]
