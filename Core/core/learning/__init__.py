"""
Learning sub-package.

Owns the Learning_Engine: conversation-end detection, durable-item
extraction via LLM, conflict-supersede logic, and per-item write
atomicity.

Design reference: Autonomous Learning loop.
Requirements: 8, 9.1.
"""

from .learning_engine import (
    FailedItem,
    IdleWatcher,
    LearnedItem,
    LearningEngine,
    LearningReport,
)

__all__ = [
    "FailedItem",
    "IdleWatcher",
    "LearnedItem",
    "LearningEngine",
    "LearningReport",
]
