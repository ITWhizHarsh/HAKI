"""
HAKI Core — local orchestration service (the Mind).

This package exposes the orchestrator, model_provider, memory, learning,
planner, dialogue, language, persona, ipc, and automation sub-packages.
Import the top-level package to confirm the service is importable before
spinning up the gRPC server.
"""

__version__ = "0.1.0"

# Make the automation sub-package importable as ``core.automation``.
from . import automation  # noqa: F401  (side-effect import for discoverability)
