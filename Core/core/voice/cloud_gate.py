"""Explicit, session-scoped Gemini Live eligibility decisions.

Gemini Live is never an availability fallback.  A gate instance tracks active
voice sessions, starts every one disabled, and can only return ``gemini_live``
when the active session was explicitly enabled and a defined qualifying
condition is present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from .diagnostics import GateCondition, GateDiagnostic, VoiceDiagnosticEvent, VoiceRoute


ThermalState = Literal["nominal", "fair", "serious", "critical"]

_LOW_BATTERY: GateCondition = "low_battery"
_THERMAL_THROTTLING: GateCondition = "thermal_throttling"
_ULTRA_COMPLEX_REASONING: GateCondition = "ultra_complex_reasoning"
_ALL_CONDITIONS: tuple[GateCondition, ...] = (
    _LOW_BATTERY,
    _THERMAL_THROTTLING,
    _ULTRA_COMPLEX_REASONING,
)


class CloudEscalationSessionInactiveError(RuntimeError):
    """An enablement action targeted a session that is not currently active."""


@dataclass(frozen=True, slots=True)
class CloudEscalationState:
    """UI-safe state returned after an explicit active-session action."""

    session_id: UUID
    active: bool
    gemini_enabled: bool


@dataclass(frozen=True, slots=True)
class GateInput:
    """Non-content-bearing measurements required to evaluate one voice turn.

    ``gemini_enabled_for_session`` is an input-side snapshot retained for the
    protocol contract.  The gate always derives the effective value from its
    own active-session state, preventing a caller from manufacturing cloud
    eligibility merely by setting this field.
    """

    session_id: UUID
    gemini_enabled_for_session: bool
    battery_percent: int | None
    external_power_connected: bool | None
    thermal_state: ThermalState
    assembled_prompt_tokens: int
    validated_tool_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, UUID):
            raise ValueError("session_id must be a UUID")
        if not isinstance(self.gemini_enabled_for_session, bool):
            raise ValueError("gemini_enabled_for_session must be a boolean")
        if self.battery_percent is not None and (
            not isinstance(self.battery_percent, int)
            or isinstance(self.battery_percent, bool)
            or not 0 <= self.battery_percent <= 100
        ):
            raise ValueError("battery_percent must be an integer from 0 through 100 or None")
        if self.external_power_connected is not None and not isinstance(
            self.external_power_connected, bool
        ):
            raise ValueError("external_power_connected must be a boolean or None")
        if self.thermal_state not in {"nominal", "fair", "serious", "critical"}:
            raise ValueError("thermal_state must be nominal, fair, serious, or critical")
        for name, value in (
            ("assembled_prompt_tokens", self.assembled_prompt_tokens),
            ("validated_tool_count", self.validated_tool_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class GateDecision:
    """The single explicit route selection made for one voice turn."""

    route: VoiceRoute
    qualifying_conditions: tuple[GateCondition, ...]
    diagnostic: GateDiagnostic
    session_id: UUID | None = None

    @property
    def eligible(self) -> bool:
        """Whether the router may invoke Gemini Live exactly once for this turn."""
        return self.route == "gemini_live"

    def diagnostic_event(self, *, turn_id: UUID) -> VoiceDiagnosticEvent:
        """Create the required gate evaluation diagnostic for this decision."""
        if self.session_id is None:
            raise RuntimeError("gate decision was not produced by CloudEscalationGate")
        return VoiceDiagnosticEvent(
            session_id=self.session_id,
            turn_id=turn_id,
            stage="cloud_gate",
            outcome="completed",
            selected_route=self.route,
            gate=self.diagnostic,
        )


@dataclass(frozen=True, slots=True)
class CloudInvocationFailure:
    """Terminal report for a failed eligible Gemini invocation.

    This value intentionally has no retry target, fallback route, or provider
    field.  It makes the required no-retry/no-fallback outcome explicit for the
    router and diagnostics boundary.
    """

    decision: GateDecision
    error_class: str
    retry_attempted: bool = False
    fallback_attempted: bool = False
    recovery_outcome: Literal["reported_no_fallback"] = "reported_no_fallback"

    def __post_init__(self) -> None:
        if not self.decision.eligible:
            raise ValueError("only an eligible Gemini decision can report a cloud invocation failure")
        if not self.error_class:
            raise ValueError("cloud invocation failure requires an error class")
        if self.retry_attempted or self.fallback_attempted:
            raise ValueError("eligible cloud failures must not retry or select a fallback")

    def diagnostic_event(self, *, turn_id: UUID) -> VoiceDiagnosticEvent:
        """Report the failed selected route without changing its selection."""
        return VoiceDiagnosticEvent(
            session_id=_decision_session_id(self.decision),
            turn_id=turn_id,
            stage="cloud_gate",
            outcome="failed",
            selected_route="gemini_live",
            gate=self.decision.diagnostic,
            error_class=self.error_class,
            recovery_outcome=self.recovery_outcome,
        )


class CloudEscalationGate:
    """Own explicit Gemini enablement for one or more active voice sessions."""

    def __init__(self) -> None:
        self._active_sessions: set[UUID] = set()
        self._enabled_sessions: set[UUID] = set()

    def register_session(self, session_id: UUID) -> CloudEscalationState:
        """Start a new session with Gemini disabled, regardless of prior state."""
        _require_session_id(session_id)
        self._active_sessions.add(session_id)
        self._enabled_sessions.discard(session_id)
        return self.ui_state(session_id)

    def set_enabled(self, session_id: UUID, *, enabled: bool) -> CloudEscalationState:
        """Apply an explicit UI action to the currently active session only."""
        _require_session_id(session_id)
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        if session_id not in self._active_sessions:
            raise CloudEscalationSessionInactiveError("Gemini Live can only be changed for an active session")
        if enabled:
            self._enabled_sessions.add(session_id)
        else:
            self._enabled_sessions.discard(session_id)
        return self.ui_state(session_id)

    def enable(self, session_id: UUID) -> CloudEscalationState:
        """Convenience alias for the explicit enable UI action."""
        return self.set_enabled(session_id, enabled=True)

    def disable(self, session_id: UUID) -> CloudEscalationState:
        """Convenience alias for the explicit disable UI action."""
        return self.set_enabled(session_id, enabled=False)

    def end_session(self, session_id: UUID) -> CloudEscalationState:
        """Remove enablement when the active voice session ends."""
        _require_session_id(session_id)
        self._enabled_sessions.discard(session_id)
        self._active_sessions.discard(session_id)
        return self.ui_state(session_id)

    def ui_state(self, session_id: UUID) -> CloudEscalationState:
        """Return state for display without exposing any turn content."""
        _require_session_id(session_id)
        active = session_id in self._active_sessions
        return CloudEscalationState(
            session_id=session_id,
            active=active,
            gemini_enabled=active and session_id in self._enabled_sessions,
        )

    def evaluate(self, gate_input: GateInput) -> GateDecision:
        """Select Gemini only for explicitly enabled active sessions that qualify."""
        enabled = self.ui_state(gate_input.session_id).gemini_enabled
        qualifying = _qualifying_conditions(gate_input)
        route: VoiceRoute = "gemini_live" if enabled and qualifying else "local_qwen"
        diagnostic = GateDiagnostic(
            enabled=enabled,
            evaluated=_ALL_CONDITIONS,
            battery_percent=gate_input.battery_percent,
            external_power_connected=gate_input.external_power_connected,
            thermal_state=gate_input.thermal_state,
            assembled_prompt_tokens=gate_input.assembled_prompt_tokens,
            validated_tool_count=gate_input.validated_tool_count,
            qualifying=qualifying,
            selected_route=route,
        )
        return GateDecision(
            route=route,
            qualifying_conditions=qualifying,
            diagnostic=diagnostic,
            session_id=gate_input.session_id,
        )

    def report_eligible_invocation_failure(
        self,
        decision: GateDecision,
        error: BaseException | str,
    ) -> CloudInvocationFailure:
        """Convert a Gemini failure to a terminal no-fallback report.

        The provider invocation is deliberately outside this gate.  This method
        is the only failure outcome it supplies, so it cannot retry Qwen,
        another cloud provider, legacy voice code, or an archive route.
        """
        error_class = error if isinstance(error, str) else type(error).__name__
        return CloudInvocationFailure(decision=decision, error_class=error_class)


def _qualifying_conditions(gate_input: GateInput) -> tuple[GateCondition, ...]:
    conditions: list[GateCondition] = []
    if gate_input.battery_percent is not None and gate_input.battery_percent <= 20 and gate_input.external_power_connected is False:
        conditions.append(_LOW_BATTERY)
    if gate_input.thermal_state in {"serious", "critical"}:
        conditions.append(_THERMAL_THROTTLING)
    if gate_input.assembled_prompt_tokens > 16_000 or gate_input.validated_tool_count > 6:
        conditions.append(_ULTRA_COMPLEX_REASONING)
    return tuple(conditions)


def _decision_session_id(decision: GateDecision) -> UUID:
    if decision.session_id is None:
        raise RuntimeError("gate decision was not produced by CloudEscalationGate")
    return decision.session_id


def _require_session_id(session_id: UUID) -> None:
    if not isinstance(session_id, UUID):
        raise ValueError("session_id must be a UUID")


__all__ = [
    "CloudEscalationGate",
    "CloudEscalationSessionInactiveError",
    "CloudEscalationState",
    "CloudInvocationFailure",
    "GateDecision",
    "GateInput",
    "ThermalState",
]
