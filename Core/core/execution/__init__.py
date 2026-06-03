"""
Execution sub-package.

Owns the Safety_Gate (action classification and confirmation gating)
and the ExecutionEngine (dependency-aware plan execution with mid-plan
pause/resume and rejection/no-response handling).

Design reference: Safety_Gate, Execution loop.
Requirements: 22.1 – 22.8.
"""

from .safety_gate import (
    ConfirmationRequest,
    ConfirmationResult,
    SafetyGate,
    SafetyGateTimeout,
)
from .execution_engine import (
    ExecutionEngine,
    StepEvent,
    StepEventType,
    ExecutionReport,
)

__all__ = [
    # Safety_Gate
    "ConfirmationRequest",
    "ConfirmationResult",
    "SafetyGate",
    "SafetyGateTimeout",
    # ExecutionEngine
    "ExecutionEngine",
    "StepEvent",
    "StepEventType",
    "ExecutionReport",
]
