"""
Execution sub-package.

Owns the Safety_Gate (action classification and confirmation gating),
the ExecutionEngine (dependency-aware plan execution with mid-plan
pause/resume and rejection/no-response handling), and the
CDPWebActuator (Arc/Chromium web actuator via Chrome DevTools Protocol).

Design reference: Safety_Gate, Execution loop, Mac_Controller.
Requirements: 17.2, 21.5, 21.6, 21.8, 21.12, 21.13, 22.1 – 22.8.
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
from .models import (
    PlanCompletionEvent,
    StepEvent as ModelStepEvent,
)
from .errors import (
    AppNotInstalledError,
    ElementNotFoundError,
    WebsiteUnreachableError,
)
from .cdp_actuator import (
    ActuatorResult,
    CDPWebActuator,
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
    # Models (Req 21.8)
    "PlanCompletionEvent",
    "ModelStepEvent",
    # Named failure exceptions (Reqs 21.9, 21.12, 21.13)
    "AppNotInstalledError",
    "ElementNotFoundError",
    "WebsiteUnreachableError",
    # CDP web actuator (Reqs 21.5, 21.6, 21.12, 21.13)
    "ActuatorResult",
    "CDPWebActuator",
]
