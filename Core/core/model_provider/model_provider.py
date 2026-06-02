"""
Model Provider abstraction.

Provides a uniform invoke / invokeStream interface over six capability
backends (STT, LLM, TTS, mood, image, embeddings).  Each capability has
an independently configurable mode (local | api).  Mode changes are
applied at the start of the *next* invocation — never mid-invocation.
API use requires prior disclosure acknowledgement; unavailability is
always surfaced — no silent fallback between modes.

Design: Model Provider Abstraction.
Requirements: 20.2, 20.3, 20.4, 20.5, 20.6, 20.7.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Capability(str, Enum):
    """The six independently-configurable AI capability backends (Req 20.2)."""

    STT = "stt"
    LLM = "llm"
    TTS = "tts"
    MOOD = "mood"
    IMAGE = "image"
    EMBEDDINGS = "embeddings"


class ProcessingMode(str, Enum):
    """
    Processing mode for a capability backend (Req 20.2).

    LOCAL — use an on-device model; data never leaves the device.
    API   — use an external API; requires prior disclosure (Req 20.5).
    """

    LOCAL = "local"
    API = "api"


# Backward-compatible alias kept so existing imports continue to work.
ModelMode = ProcessingMode


# ---------------------------------------------------------------------------
# CapabilityConfig
# ---------------------------------------------------------------------------


@dataclass
class CapabilityConfig:
    """
    Runtime configuration snapshot for a single capability backend.

    Read at the *start* of each invocation so that a mode change set via
    ``ModelProviderRegistry.set_mode()`` takes effect on the very next
    call without interrupting an in-progress invocation (Req 20.3).
    """

    capability: Capability
    mode: ProcessingMode = ProcessingMode.LOCAL
    # Keychain reference for the API key — never the secret value (Req 20.2).
    api_key_ref: str | None = None


# ---------------------------------------------------------------------------
# ModelProviderRegistry
# ---------------------------------------------------------------------------


class ModelProviderRegistry:
    """
    Thread-safe registry that stores a ``CapabilityConfig`` per capability.

    All config reads and writes are serialised through a ``threading.Lock``
    so that ``set_mode()`` from one thread and ``get_config()`` from an
    invocation thread never race.

    Default mode for every capability is ``ProcessingMode.LOCAL``.
    """

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._configs: dict[Capability, CapabilityConfig] = {
            cap: CapabilityConfig(capability=cap) for cap in Capability
        }

    # ------------------------------------------------------------------
    # Configuration reads / writes
    # ------------------------------------------------------------------

    def set_mode(self, capability: Capability, mode: ProcessingMode) -> None:
        """
        Update the mode for *capability*.

        The new mode is stored immediately; any ``ModelProvider`` that
        calls ``get_config()`` after this point will see the new mode.
        In-progress invocations that already called ``get_config()`` at
        their start are unaffected (Req 20.3).
        """
        with self._lock:
            self._configs[capability].mode = mode

    def get_config(self, capability: Capability) -> CapabilityConfig:
        """
        Return a *snapshot* of the current ``CapabilityConfig`` for
        *capability*.

        ModelProvider implementations must call this at the very start
        of ``invoke`` / ``invoke_stream`` to capture the mode for that
        invocation (Req 20.3).
        """
        with self._lock:
            cfg = self._configs[capability]
            # Return a shallow copy so the caller's snapshot is immutable.
            return CapabilityConfig(
                capability=cfg.capability,
                mode=cfg.mode,
                api_key_ref=cfg.api_key_ref,
            )

    def set_api_key_ref(self, capability: Capability, key_ref: str) -> None:
        """Store a Keychain reference (not the secret value) for *capability*."""
        with self._lock:
            self._configs[capability].api_key_ref = key_ref


# ---------------------------------------------------------------------------
# ModelProvider — abstract base class
# ---------------------------------------------------------------------------


class ModelProvider(ABC):
    """
    Abstract base class for all capability-specific model backends.

    Concrete subclasses must:
    - Declare their ``capability`` via the abstract property.
    - Implement ``invoke`` and ``invoke_stream``.
    - Call ``self._registry.get_config(self.capability)`` **at the very
      start** of both methods to resolve the mode for that invocation
      (Req 20.3, 20.4).

    A shared ``ModelProviderRegistry`` is injected at construction so
    tests can provide their own registry, and so multiple providers can
    share a single source of truth.
    """

    def __init__(self, registry: ModelProviderRegistry) -> None:
        self._registry: ModelProviderRegistry = registry

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def capability(self) -> Capability:
        """The capability this provider handles (e.g. ``Capability.LLM``)."""

    @abstractmethod
    def invoke(self, input: Any, **kwargs: Any) -> Any:
        """
        Synchronous invocation.

        Implementations **must** call
        ``self._registry.get_config(self.capability)`` as their first
        statement to capture the resolved mode for this invocation
        (Req 20.3, 20.4).
        """

    @abstractmethod
    async def invoke_stream(self, input: Any, **kwargs: Any) -> AsyncIterator[Any]:
        """
        Async streaming invocation.

        Implementations **must** call
        ``self._registry.get_config(self.capability)`` as their first
        statement to capture the resolved mode for this invocation
        (Req 20.3, 20.4).
        """
        # Allow subclasses to use ``yield`` directly or return an async
        # generator; this default body satisfies the type checker.
        return
        yield  # pragma: no cover — makes this an async generator stub


# ---------------------------------------------------------------------------
# Custom exceptions (Req 20.5, 20.6, 20.7)
# ---------------------------------------------------------------------------


class DisclosureRequiredError(Exception):
    """
    Raised when an API backend is invoked without prior disclosure
    acknowledgement (Req 20.5).
    """


class ApiUnavailableError(Exception):
    """
    Raised when the external API backend is unreachable or returns an
    error — no silent fallback to local (Req 20.6).
    """


class LocalModelLoadError(Exception):
    """
    Raised when the local model backend fails to load — the user must
    reconfigure before further invocations are accepted (Req 20.7).
    """


# ---------------------------------------------------------------------------
# DisclosureTracker (thread-safe; Req 20.5)
# ---------------------------------------------------------------------------


class DisclosureTracker:
    """
    Thread-safe registry that records which capabilities have received an
    explicit disclosure acknowledgement from the user.

    Before any API backend is first invoked, ``disclose_api_usage()``
    must call ``acknowledge()``.  If it has not been called,
    ``ApiBackend`` raises ``DisclosureRequiredError`` (Req 20.5).
    """

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._acknowledged: set[Capability] = set()

    def acknowledge(self, capability: Capability) -> None:
        """Record that the user has acknowledged API data-leaving-device for *capability*."""
        with self._lock:
            self._acknowledged.add(capability)

    def is_acknowledged(self, capability: Capability) -> bool:
        """Return True iff disclosure has been acknowledged for *capability*."""
        with self._lock:
            return capability in self._acknowledged

    def reset(self, capability: Capability) -> None:
        """
        Remove the acknowledgement for *capability*.

        Used in tests and when the user changes the API configuration so
        that disclosure must be presented again.
        """
        with self._lock:
            self._acknowledged.discard(capability)


# ---------------------------------------------------------------------------
# Default / stub provider (used until real backends are wired in Task 5.2)
# ---------------------------------------------------------------------------


class StubModelProvider(ModelProvider):
    """
    Stub ``ModelProvider`` that echoes inputs back with metadata.

    Useful for unit-testing the registry and mode-resolution logic
    without wiring real models (per testing conventions in conftest.py).
    """

    def __init__(self, capability: Capability, registry: ModelProviderRegistry) -> None:
        super().__init__(registry)
        self._capability = capability

    @property
    def capability(self) -> Capability:
        return self._capability

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        # Read config FIRST — satisfies Req 20.3 / 20.4.
        config = self._registry.get_config(self.capability)
        return {
            "capability": config.capability.value,
            "mode": config.mode.value,
            "input": input,
            "stub": True,
            **kwargs,
        }

    async def invoke_stream(self, input: Any, **kwargs: Any) -> AsyncIterator[Any]:
        # Read config FIRST — satisfies Req 20.3 / 20.4.
        config = self._registry.get_config(self.capability)
        yield {
            "capability": config.capability.value,
            "mode": config.mode.value,
            "input": input,
            "stub": True,
            "chunk": 0,
            **kwargs,
        }


# ---------------------------------------------------------------------------
# LocalBackend — on-device stub backend (Req 20.7)
# ---------------------------------------------------------------------------


class LocalBackend(ModelProvider):
    """
    Concrete local backend stub.

    When ``_loaded`` is ``True`` (the default), invocations succeed and
    return a dict echoing the input together with backend metadata.

    When ``_loaded`` is ``False`` (simulated via ``simulate_load_failure()``),
    every invocation raises ``LocalModelLoadError``.  The flag can be
    reset with ``restore()`` for test teardown.

    No silent fallback to the API backend is ever performed (Req 20.7).
    """

    def __init__(self, capability: Capability, registry: ModelProviderRegistry) -> None:
        super().__init__(registry)
        self._capability = capability
        self._loaded: bool = True

    @property
    def capability(self) -> Capability:
        return self._capability

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def simulate_load_failure(self) -> None:
        """Mark this backend as failed (simulates a load error)."""
        self._loaded = False

    def restore(self) -> None:
        """Restore the backend to the loaded state."""
        self._loaded = True

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        # Resolve mode at invocation start (Req 20.3).
        config = self._registry.get_config(self.capability)
        if not self._loaded:
            raise LocalModelLoadError(
                f"Local model for {self.capability.value} failed to load. "
                "Reconfigure to proceed."
            )
        return {
            "capability": config.capability.value,
            "mode": "local",
            "input": input,
            "backend": "local",
            "stub": True,
            **kwargs,
        }

    async def invoke_stream(self, input: Any, **kwargs: Any) -> AsyncIterator[Any]:
        # Resolve mode at invocation start (Req 20.3).
        config = self._registry.get_config(self.capability)
        if not self._loaded:
            raise LocalModelLoadError(
                f"Local model for {self.capability.value} failed to load. "
                "Reconfigure to proceed."
            )
        yield {
            "capability": config.capability.value,
            "mode": "local",
            "input": input,
            "backend": "local",
            "stub": True,
            "chunk": 0,
            **kwargs,
        }


# ---------------------------------------------------------------------------
# ApiBackend — external API stub backend (Req 20.5, 20.6)
# ---------------------------------------------------------------------------


class ApiBackend(ModelProvider):
    """
    Concrete API backend stub.

    Invocation gates:
    1. Checks ``disclosure_tracker.is_acknowledged(capability)``; if not
       acknowledged raises ``DisclosureRequiredError`` (Req 20.5).
    2. When ``_available`` is ``False`` (simulated via
       ``simulate_api_unavailable()``), raises ``ApiUnavailableError``
       (Req 20.6).

    No silent fallback to the local backend is ever performed.
    """

    def __init__(
        self,
        capability: Capability,
        registry: ModelProviderRegistry,
        disclosure_tracker: DisclosureTracker,
    ) -> None:
        super().__init__(registry)
        self._capability = capability
        self._disclosure_tracker = disclosure_tracker
        self._available: bool = True

    @property
    def capability(self) -> Capability:
        return self._capability

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def simulate_api_unavailable(self) -> None:
        """Mark this backend as unavailable (simulates a network / auth error)."""
        self._available = False

    def restore(self) -> None:
        """Restore the backend to the available state."""
        self._available = True

    # ------------------------------------------------------------------
    # Shared pre-invocation checks
    # ------------------------------------------------------------------

    def _check_preconditions(self) -> None:
        """
        Run disclosure and availability checks.

        Called at the very start of both ``invoke`` and ``invoke_stream``
        after the mode snapshot has been captured (Req 20.3).
        """
        if not self._disclosure_tracker.is_acknowledged(self.capability):
            raise DisclosureRequiredError(
                f"API disclosure required for {self.capability.value} before first use."
            )
        if not self._available:
            raise ApiUnavailableError(
                f"API backend for {self.capability.value} is unavailable."
            )

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        # Resolve mode at invocation start (Req 20.3).
        config = self._registry.get_config(self.capability)
        self._check_preconditions()
        return {
            "capability": config.capability.value,
            "mode": "api",
            "input": input,
            "backend": "api",
            "stub": True,
            **kwargs,
        }

    async def invoke_stream(self, input: Any, **kwargs: Any) -> AsyncIterator[Any]:
        # Resolve mode at invocation start (Req 20.3).
        config = self._registry.get_config(self.capability)
        self._check_preconditions()
        yield {
            "capability": config.capability.value,
            "mode": "api",
            "input": input,
            "backend": "api",
            "stub": True,
            "chunk": 0,
            **kwargs,
        }


# ---------------------------------------------------------------------------
# BackendRouter — routes to local or API based on current mode (Req 20.3)
# ---------------------------------------------------------------------------


class BackendRouter(ModelProvider):
    """
    Routes each invocation to the local or API backend based on the mode
    resolved from the registry at invocation start (Req 20.3).

    No silent fallback between modes (Req 20.6, 20.7): errors from either
    backend propagate unmodified.
    """

    def __init__(
        self,
        capability: Capability,
        registry: ModelProviderRegistry,
        local_backend: LocalBackend,
        api_backend: ApiBackend,
    ) -> None:
        super().__init__(registry)
        self._capability = capability
        self._local_backend = local_backend
        self._api_backend = api_backend

    @property
    def capability(self) -> Capability:
        return self._capability

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        # Resolve mode at invocation start (Req 20.3).
        config = self._registry.get_config(self.capability)
        if config.mode == ProcessingMode.LOCAL:
            return self._local_backend.invoke(input, **kwargs)
        else:
            return self._api_backend.invoke(input, **kwargs)

    async def invoke_stream(self, input: Any, **kwargs: Any) -> AsyncIterator[Any]:
        # Resolve mode at invocation start (Req 20.3).
        config = self._registry.get_config(self.capability)
        if config.mode == ProcessingMode.LOCAL:
            async for chunk in self._local_backend.invoke_stream(input, **kwargs):
                yield chunk
        else:
            async for chunk in self._api_backend.invoke_stream(input, **kwargs):
                yield chunk


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_default_registry_and_routers() -> tuple[
    ModelProviderRegistry,
    DisclosureTracker,
    dict[Capability, BackendRouter],
]:
    """
    Create a fresh ``ModelProviderRegistry``, a ``DisclosureTracker``, and
    one ``BackendRouter`` per capability, all wired with new ``LocalBackend``
    and ``ApiBackend`` instances.

    Returns
    -------
    registry : ModelProviderRegistry
    disclosure_tracker : DisclosureTracker
    routers : dict[Capability, BackendRouter]
        One entry per ``Capability``.
    """
    registry = ModelProviderRegistry()
    disclosure_tracker = DisclosureTracker()
    routers: dict[Capability, BackendRouter] = {}
    for cap in Capability:
        local = LocalBackend(cap, registry)
        api = ApiBackend(cap, registry, disclosure_tracker)
        routers[cap] = BackendRouter(cap, registry, local, api)
    return registry, disclosure_tracker, routers
