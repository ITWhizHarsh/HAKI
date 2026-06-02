"""
Unit tests for the ModelProvider registry and per-invocation mode resolution.

Feature: haki-personal-ai-assistant
Requirements: 20.2, 20.3, 20.4
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from core.model_provider import (
    Capability,
    CapabilityConfig,
    ModelMode,
    ModelProvider,
    ModelProviderRegistry,
    ProcessingMode,
    StubModelProvider,
)


# ---------------------------------------------------------------------------
# Enum / alias smoke tests
# ---------------------------------------------------------------------------


def test_capability_enum_has_all_six_values():
    """Req 20.2 — six independently-configurable capabilities."""
    assert set(Capability) == {
        Capability.STT,
        Capability.LLM,
        Capability.TTS,
        Capability.MOOD,
        Capability.IMAGE,
        Capability.EMBEDDINGS,
    }


def test_processing_mode_values():
    """ProcessingMode has LOCAL and API variants (Req 20.2)."""
    assert ProcessingMode.LOCAL.value == "local"
    assert ProcessingMode.API.value == "api"


def test_model_mode_is_alias_for_processing_mode():
    """ModelMode is the backward-compatible alias for ProcessingMode."""
    assert ModelMode is ProcessingMode


# ---------------------------------------------------------------------------
# ModelProviderRegistry — config reads and writes
# ---------------------------------------------------------------------------


def test_registry_defaults_to_local_for_all_capabilities():
    """All capabilities default to LOCAL mode (Req 20.2)."""
    registry = ModelProviderRegistry()
    for cap in Capability:
        cfg = registry.get_config(cap)
        assert cfg.mode == ProcessingMode.LOCAL, f"{cap} should default to LOCAL"
        assert cfg.capability == cap


def test_registry_set_mode_updates_config():
    """set_mode() persists the new mode immediately (Req 20.2)."""
    registry = ModelProviderRegistry()
    registry.set_mode(Capability.LLM, ProcessingMode.API)
    assert registry.get_config(Capability.LLM).mode == ProcessingMode.API


def test_registry_set_mode_is_per_capability():
    """Mode change for one capability does not affect others (Req 20.2)."""
    registry = ModelProviderRegistry()
    registry.set_mode(Capability.LLM, ProcessingMode.API)
    for cap in Capability:
        if cap != Capability.LLM:
            assert registry.get_config(cap).mode == ProcessingMode.LOCAL


def test_registry_get_config_returns_snapshot():
    """
    get_config() returns a shallow copy; mutating it does not affect the
    stored config (ensures invocation snapshots are independent).
    """
    registry = ModelProviderRegistry()
    registry.set_mode(Capability.STT, ProcessingMode.API)

    snapshot = registry.get_config(Capability.STT)
    snapshot.mode = ProcessingMode.LOCAL  # mutate the snapshot

    # The registry's stored value must remain API.
    assert registry.get_config(Capability.STT).mode == ProcessingMode.API


def test_registry_set_api_key_ref():
    """api_key_ref is stored and returned via get_config()."""
    registry = ModelProviderRegistry()
    registry.set_api_key_ref(Capability.LLM, "keychain://haki/llm")
    cfg = registry.get_config(Capability.LLM)
    assert cfg.api_key_ref == "keychain://haki/llm"


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------


def test_registry_is_thread_safe():
    """
    Concurrent set_mode + get_config calls from multiple threads must not
    raise and must leave the registry in a consistent state.
    """
    registry = ModelProviderRegistry()
    errors: list[Exception] = []

    def writer():
        for _ in range(200):
            try:
                registry.set_mode(Capability.LLM, ProcessingMode.API)
                registry.set_mode(Capability.LLM, ProcessingMode.LOCAL)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    def reader():
        for _ in range(200):
            try:
                cfg = registry.get_config(Capability.LLM)
                assert cfg.mode in (ProcessingMode.LOCAL, ProcessingMode.API)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Thread-safety errors: {errors}"


# ---------------------------------------------------------------------------
# ModelProvider.invoke — mode resolved at invocation start (Req 20.3, 20.4)
# ---------------------------------------------------------------------------


def test_invoke_resolves_mode_at_start():
    """
    The mode captured by invoke() is the mode at the moment of the call,
    not at provider construction time (Req 20.3, 20.4).
    """
    registry = ModelProviderRegistry()
    stub = StubModelProvider(Capability.LLM, registry)

    registry.set_mode(Capability.LLM, ProcessingMode.LOCAL)
    result = stub.invoke("hello")
    assert result["mode"] == "local"

    # Change mode BEFORE the next call.
    registry.set_mode(Capability.LLM, ProcessingMode.API)
    result2 = stub.invoke("world")
    assert result2["mode"] == "api"


def test_invoke_carries_input_and_kwargs():
    """invoke() returns the input and any extra kwargs in stub output."""
    registry = ModelProviderRegistry()
    stub = StubModelProvider(Capability.TTS, registry)
    result = stub.invoke("speak this", language="en")
    assert result["input"] == "speak this"
    assert result["language"] == "en"
    assert result["capability"] == "tts"


# ---------------------------------------------------------------------------
# ModelProvider.invoke_stream — async streaming (Req 20.4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_stream_yields_at_least_one_chunk():
    """invoke_stream() must yield at least one chunk (Req 20.4)."""
    registry = ModelProviderRegistry()
    stub = StubModelProvider(Capability.STT, registry)

    chunks = []
    async for chunk in stub.invoke_stream("audio frames"):
        chunks.append(chunk)

    assert len(chunks) >= 1


@pytest.mark.asyncio
async def test_invoke_stream_resolves_mode_at_start():
    """
    The mode in the first yielded chunk reflects the mode at invocation
    start, not construction time (Req 20.3, 20.4).
    """
    registry = ModelProviderRegistry()
    stub = StubModelProvider(Capability.LLM, registry)

    registry.set_mode(Capability.LLM, ProcessingMode.API)
    chunks = []
    async for chunk in stub.invoke_stream("prompt"):
        chunks.append(chunk)

    assert chunks[0]["mode"] == "api"


# ---------------------------------------------------------------------------
# CapabilityConfig dataclass
# ---------------------------------------------------------------------------


def test_capability_config_defaults():
    """CapabilityConfig defaults to LOCAL mode and no api_key_ref."""
    cfg = CapabilityConfig(capability=Capability.IMAGE)
    assert cfg.mode == ProcessingMode.LOCAL
    assert cfg.api_key_ref is None


def test_capability_config_explicit_values():
    """CapabilityConfig stores explicitly provided values."""
    cfg = CapabilityConfig(
        capability=Capability.EMBEDDINGS,
        mode=ProcessingMode.API,
        api_key_ref="keychain://haki/embeddings",
    )
    assert cfg.mode == ProcessingMode.API
    assert cfg.api_key_ref == "keychain://haki/embeddings"
