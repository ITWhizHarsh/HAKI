"""
gui_agent package — Gemini-Sidecar GUI Screen Control Agent.

Public exports:
    GeminiVisionClient, GeminiAPIError, BoundingBox, NativePixelBox, GeminiAction
    MacQuartzExecutor, ExecutorUnavailableError
    SidecarAgentLoop
    HITLBridge, HITLTimeoutError

Design: Gemini-Sidecar Architecture design document.
Requirements: 3.1, 4.1, 5.1, 8.1
"""

__all__ = [
    "GeminiVisionClient",
    "GeminiAPIError",
    "BoundingBox",
    "NativePixelBox",
    "GeminiAction",
    "MacQuartzExecutor",
    "ExecutorUnavailableError",
    "SidecarAgentLoop",
    "HITLBridge",
    "HITLTimeoutError",
]

try:
    from .gemini_vision_client import (
        GeminiVisionClient,
        GeminiAPIError,
        BoundingBox,
        NativePixelBox,
        GeminiAction,
    )
except ImportError:
    pass

try:
    from .mac_quartz_executor import MacQuartzExecutor, ExecutorUnavailableError
except ImportError:
    pass

try:
    from .sidecar_agent_loop import SidecarAgentLoop
except ImportError:
    pass

try:
    from .hitl_bridge import HITLBridge, HITLTimeoutError
except ImportError:
    pass
