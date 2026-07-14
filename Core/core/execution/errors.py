"""
Named failure exception types for the ExecutionEngine.

These exceptions are raised by actuator callbacks to signal specific, named
failure modes that the engine handles distinctly — stopping transitive
dependents and recording a structured reason in the execution report.

Each exception carries enough information for the ExecutionEngine to produce
a human-readable failure reason that appears in the PlanCompletionEvent
(Req 21.8).

Design: ExecutionEngine, Execution loop.
Requirements: 21.9, 21.12, 21.13.
"""

from __future__ import annotations


class AppNotInstalledError(Exception):
    """
    Raised when an actuator detects that a required application is not
    installed on the device (Req 21.9).

    When this exception propagates out of an actuator callback the
    ExecutionEngine will:
    - Mark the step as FAILED with ``reason="App not installed: <app_name>"``.
    - Stop all transitive dependents (mark them SKIPPED).
    - Allow independent (non-dependent) steps to continue running.

    Parameters
    ----------
    app_name:
        The name of the application that was not found.

    Example
    -------
    ::

        raise AppNotInstalledError("WhatsApp")
        # reason: "App not installed: WhatsApp"
    """

    def __init__(self, app_name: str) -> None:
        self.app_name = app_name
        super().__init__(f"App not installed: {app_name}")


class ElementNotFoundError(Exception):
    """
    Raised when an actuator cannot locate a required UI element on screen
    (Req 21.12).

    When this exception propagates out of an actuator callback the
    ExecutionEngine will:
    - Mark the step as FAILED with ``reason="Element not found: <description>"``.
    - Stop all transitive dependents (mark them SKIPPED).
    - Allow independent (non-dependent) steps to continue running.

    Parameters
    ----------
    description:
        A human-readable description of the element that could not be found
        (e.g. "Submit button in Safari checkout form").

    Example
    -------
    ::

        raise ElementNotFoundError("Submit button in checkout form")
        # reason: "Element not found: Submit button in checkout form"
    """

    def __init__(self, description: str) -> None:
        self.description = description
        super().__init__(f"Element not found: {description}")


class WebsiteUnreachableError(Exception):
    """
    Raised when an actuator cannot reach a target website or web resource
    (Req 21.13).

    When this exception propagates out of an actuator callback the
    ExecutionEngine will:
    - Mark the step as FAILED with ``reason="Website unreachable: <url>"``.
    - Stop all transitive dependents (mark them SKIPPED).
    - Allow independent (non-dependent) steps to continue running.

    Parameters
    ----------
    url:
        The URL that could not be reached.

    Example
    -------
    ::

        raise WebsiteUnreachableError("https://example.com")
        # reason: "Website unreachable: https://example.com"
    """

    def __init__(self, url: str) -> None:
        self.url = url
        super().__init__(f"Website unreachable: {url}")
