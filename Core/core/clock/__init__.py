"""
Clock sub-package.

Provides the single source of truth for the current date, time, and
timezone across all HAKI subsystems.  Unavailability is surfaced as a
typed result rather than an exception so callers never have to guard
against a raised error.

Design reference: Clock component.
Requirements: 14.1, 14.2, 14.3, 14.4, 14.5.
"""

from .clock import Clock, ClockResult, ClockUnavailable

__all__ = ["Clock", "ClockResult", "ClockUnavailable"]
