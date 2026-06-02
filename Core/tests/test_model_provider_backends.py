"""
Functional tests for local + API backend stubs, disclosure gating,
failure propagation, BackendRouter routing, and the factory function.

Feature: haki-personal-ai-assistant
Requirements: 20.5, 20.6, 20.7
"""

from __future__ import annotations

import pytest

from core.model_provider import (
    ApiBackend,
    ApiUnavailableError,
    BackendRouter,
    Capability,
    DisclosureRequiredError,
    DisclosureTracker,
    LocalBackend,
    LocalModelLoadError,
    ModelProviderRegistry,
    ProcessingMode,
    create_default_registry_and_routers,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_registry_and_tracker(
    cap: Capability = Capability.LLM,
    mode: ProcessingMode = ProcessingMode.LOCAL,
) -> tuple[ModelProviderRegistry, DisclosureTracker]:
    reg = ModelProviderRegistry()
    reg.set_mode(cap, mode)
    tracker = DisclosureTracker()
    return reg, tracker


def make_router(
    cap: Capability = Capability.LLM,
    mode: ProcessingMode = ProcessingMode.LOCAL,
) -> tuple[BackendRouter, LocalBackend, ApiBackend, ModelProviderRegistry, DisclosureTracker]:
    reg, tracker = make_registry_and_tracker(cap, mode)
    local = LocalBackend(cap, reg)
    api = ApiBackend(cap, reg, tracker)
    router = BackendRouter(cap, reg, local, api)
    return router, local, api, reg, tracker


# ---------------------------------------------------------------------------
# LocalBackend
# ---------------------------------------------------------------------------


def test_local_backend_returns_local_mode_output_when_loaded():
    """LocalBackend returns a 'local' mode dict when loaded."""
    reg = ModelProviderRegistry()
    backend = LocalBackend(Capability.STT, reg)
    result = backend.invoke("audio", extra="x")
    assert result["mode"] == "local"
    assert result["backend"] == "local"
    assert result["input"] == "audio"
    assert result["capability"] == "stt"
    assert result["stub"] is True
    assert result["extra"] == "x"


def test_local_backend_raises_local_model_load_error_when_not_loaded():
    """LocalBackend raises LocalModelLoadError after simulate_load_failure (Req 20.7)."""
    reg = ModelProviderRegistry()
    backend = LocalBackend(Capability.TTS, reg)
    backend.simulate_load_failure()
    with pytest.raises(LocalModelLoadError, match="tts"):
        backend.invoke("speak")


def test_local_backend_restore_re_enables_invocation():
    """restore() clears the load failure and allows invocations again."""
    reg = ModelProviderRegistry()
    backend = LocalBackend(Capability.LLM, reg)
    backend.simulate_load_failure()
    backend.restore()
    result = backend.invoke("hi")
    assert result["mode"] == "local"


# ---------------------------------------------------------------------------
# ApiBackend
# ---------------------------------------------------------------------------


def test_api_backend_raises_disclosure_required_when_not_acknowledged():
    """ApiBackend raises DisclosureRequiredError when disclosure not yet given (Req 20.5)."""
    reg = ModelProviderRegistry()
    tracker = DisclosureTracker()
    backend = ApiBackend(Capability.LLM, reg, tracker)
    with pytest.raises(DisclosureRequiredError, match="llm"):
        backend.invoke("prompt")


def test_api_backend_succeeds_after_disclosure_acknowledgement():
    """ApiBackend succeeds once the user has acknowledged the disclosure (Req 20.5)."""
    reg = ModelProviderRegistry()
    tracker = DisclosureTracker()
    tracker.acknowledge(Capability.LLM)
    backend = ApiBackend(Capability.LLM, reg, tracker)
    result = backend.invoke("prompt")
    assert result["mode"] == "api"
    assert result["backend"] == "api"
    assert result["input"] == "prompt"
    assert result["stub"] is True


def test_api_backend_raises_api_unavailable_even_after_disclosure():
    """ApiBackend raises ApiUnavailableError when unavailable, even if disclosed (Req 20.6)."""
    reg = ModelProviderRegistry()
    tracker = DisclosureTracker()
    tracker.acknowledge(Capability.LLM)
    backend = ApiBackend(Capability.LLM, reg, tracker)
    backend.simulate_api_unavailable()
    with pytest.raises(ApiUnavailableError, match="llm"):
        backend.invoke("prompt")


def test_api_backend_restore_re_enables_invocation():
    """restore() clears the unavailable flag and allows invocations again."""
    reg = ModelProviderRegistry()
    tracker = DisclosureTracker()
    tracker.acknowledge(Capability.LLM)
    backend = ApiBackend(Capability.LLM, reg, tracker)
    backend.simulate_api_unavailable()
    backend.restore()
    result = backend.invoke("prompt")
    assert result["mode"] == "api"


# ---------------------------------------------------------------------------
# BackendRouter — routing
# ---------------------------------------------------------------------------


def test_backend_router_routes_to_local_when_mode_local():
    """BackendRouter delegates to LocalBackend when mode=LOCAL."""
    router, _, _, _, _ = make_router(Capability.LLM, ProcessingMode.LOCAL)
    result = router.invoke("hello")
    assert result["backend"] == "local"
    assert result["mode"] == "local"


def test_backend_router_routes_to_api_when_mode_api_and_disclosed():
    """BackendRouter delegates to ApiBackend when mode=API and disclosure given."""
    router, _, _, reg, tracker = make_router(Capability.LLM, ProcessingMode.API)
    tracker.acknowledge(Capability.LLM)
    result = router.invoke("hello")
    assert result["backend"] == "api"
    assert result["mode"] == "api"


# ---------------------------------------------------------------------------
# BackendRouter — no silent fallback (Req 20.5, 20.6, 20.7)
# ---------------------------------------------------------------------------


def test_backend_router_propagates_disclosure_required_no_silent_fallback():
    """
    BackendRouter propagates DisclosureRequiredError without swallowing it
    — no silent fallback to local (Req 20.5).
    """
    router, _, _, reg, _ = make_router(Capability.LLM, ProcessingMode.API)
    # Disclosure NOT acknowledged → must raise, not silently use local.
    with pytest.raises(DisclosureRequiredError):
        router.invoke("hello")


def test_backend_router_propagates_api_unavailable_no_silent_fallback():
    """
    BackendRouter propagates ApiUnavailableError without falling back to
    local (Req 20.6).
    """
    router, _, api, reg, tracker = make_router(Capability.LLM, ProcessingMode.API)
    tracker.acknowledge(Capability.LLM)
    api.simulate_api_unavailable()
    with pytest.raises(ApiUnavailableError):
        router.invoke("hello")


def test_backend_router_propagates_local_load_error_no_silent_fallback():
    """
    BackendRouter propagates LocalModelLoadError without falling back to
    API (Req 20.7).
    """
    router, local, _, reg, _ = make_router(Capability.LLM, ProcessingMode.LOCAL)
    local.simulate_load_failure()
    with pytest.raises(LocalModelLoadError):
        router.invoke("hello")


# ---------------------------------------------------------------------------
# BackendRouter — invoke_stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backend_router_invoke_stream_local_route():
    """BackendRouter.invoke_stream routes to LocalBackend and yields chunks."""
    router, _, _, _, _ = make_router(Capability.STT, ProcessingMode.LOCAL)
    chunks = []
    async for chunk in router.invoke_stream("audio"):
        chunks.append(chunk)
    assert len(chunks) >= 1
    assert chunks[0]["backend"] == "local"
    assert chunks[0]["mode"] == "local"


@pytest.mark.asyncio
async def test_backend_router_invoke_stream_api_route():
    """BackendRouter.invoke_stream routes to ApiBackend when mode=API and disclosed."""
    router, _, _, reg, tracker = make_router(Capability.TTS, ProcessingMode.API)
    tracker.acknowledge(Capability.TTS)
    chunks = []
    async for chunk in router.invoke_stream("speak"):
        chunks.append(chunk)
    assert len(chunks) >= 1
    assert chunks[0]["backend"] == "api"
    assert chunks[0]["mode"] == "api"


@pytest.mark.asyncio
async def test_backend_router_invoke_stream_propagates_local_load_error():
    """invoke_stream propagates LocalModelLoadError — no silent fallback (Req 20.7)."""
    router, local, _, _, _ = make_router(Capability.LLM, ProcessingMode.LOCAL)
    local.simulate_load_failure()
    with pytest.raises(LocalModelLoadError):
        async for _ in router.invoke_stream("hello"):
            pass


@pytest.mark.asyncio
async def test_backend_router_invoke_stream_propagates_disclosure_required():
    """invoke_stream propagates DisclosureRequiredError — no silent fallback (Req 20.5)."""
    router, _, _, reg, _ = make_router(Capability.LLM, ProcessingMode.API)
    with pytest.raises(DisclosureRequiredError):
        async for _ in router.invoke_stream("hello"):
            pass


# ---------------------------------------------------------------------------
# create_default_registry_and_routers factory
# ---------------------------------------------------------------------------


def test_create_default_registry_and_routers_one_router_per_capability():
    """Factory returns exactly one BackendRouter per Capability."""
    registry, tracker, routers = create_default_registry_and_routers()
    assert set(routers.keys()) == set(Capability)
    assert len(routers) == len(list(Capability))


def test_create_default_registry_and_routers_types():
    """Factory returns correct types."""
    registry, tracker, routers = create_default_registry_and_routers()
    assert isinstance(registry, ModelProviderRegistry)
    assert isinstance(tracker, DisclosureTracker)
    for cap, router in routers.items():
        assert isinstance(router, BackendRouter)
        assert router.capability == cap


def test_create_default_registry_and_routers_all_local_by_default():
    """All capabilities default to LOCAL mode in the factory output."""
    registry, tracker, routers = create_default_registry_and_routers()
    for cap in Capability:
        result = routers[cap].invoke("test_input")
        assert result["mode"] == "local", f"{cap} should default to LOCAL"
        assert result["backend"] == "local"


def test_create_default_registry_and_routers_api_works_after_disclosure():
    """Switching a router to API mode works after acknowledgement."""
    registry, tracker, routers = create_default_registry_and_routers()
    registry.set_mode(Capability.LLM, ProcessingMode.API)
    tracker.acknowledge(Capability.LLM)
    result = routers[Capability.LLM].invoke("prompt")
    assert result["mode"] == "api"
    assert result["backend"] == "api"
