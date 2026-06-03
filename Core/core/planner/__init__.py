"""
Planner sub-package.

Owns the CommandPlan and Step data models, the LLM-based planner, the
dependency graph, and the Safety_Gate that classifies steps as
CONSEQUENTIAL, REVERSIBLE, or UNKNOWN before execution.

Design reference: Planning, Agentic Execution Engine, Safety_Gate.
Requirements: 21, 22, 23.
"""

from .planner import (
    # Enumerations
    StepStatus,
    StepClassification,
    Actuator,
    SlotStatus,
    # Data models
    Slot,
    Step,
    CommandPlan,
    # Planner
    Planner,
    # Exceptions
    CyclicDependencyError,
    PlanGenerationError,
)

__all__ = [
    "StepStatus",
    "StepClassification",
    "Actuator",
    "SlotStatus",
    "Slot",
    "Step",
    "CommandPlan",
    "Planner",
    "CyclicDependencyError",
    "PlanGenerationError",
]

