"""
Unit tests for LLMRouter's prefer_local routing behavior, used by the Heavy
Pass memory pipeline to keep memory synthesis fully offline.

Feature: haki-brain-memory-processing-pipeline
Requirements: 2.7 (Heavy Pass calls Bonsai-8B via LLMRouter with prefer_local=True)
"""

from __future__ import annotations

import pytest

from core.model_provider.llm_router import LLMRouter, LLMRouterConfig, LLMTier


# ---------------------------------------------------------------------------
# _routing_order
# ---------------------------------------------------------------------------


def test_prefer_local_returns_only_local_mlx_tier():
    """prefer_local=True bypasses all cloud tiers, returning LOCAL_MLX only."""
    router = LLMRouter(LLMRouterConfig(
        groq_api_key="k", cerebras_api_key="k", gemini_api_key="k",
    ))
    order = router._routing_order("short message", prefer_large=False, prefer_local=True)
    assert order == [LLMTier.LOCAL_MLX]


def test_prefer_local_overrides_prefer_large():
    """prefer_local=True takes precedence over prefer_large_context."""
    router = LLMRouter()
    order = router._routing_order("x" * 20_000, prefer_large=True, prefer_local=True)
    assert order == [LLMTier.LOCAL_MLX]


def test_prefer_local_false_uses_normal_cloud_first_order():
    """Without prefer_local, the normal fast/large routing order is used."""
    router = LLMRouter()
    order = router._routing_order("short message", prefer_large=False, prefer_local=False)
    assert order[0] == LLMTier.GROQ
    assert LLMTier.LOCAL_MLX in order


# ---------------------------------------------------------------------------
# _stream_local_mlx — standby-first ordering when prefer_standby=True
# ---------------------------------------------------------------------------


async def test_stream_local_mlx_prefers_standby_model_first(monkeypatch):
    """
    prefer_standby=True (set via prefer_local at the chat()/stream_chat() layer)
    tries the standby model (Bonsai-8B) before the primary model (xLAM).
    """
    router = LLMRouter(LLMRouterConfig(
        mlx_primary_model="primary-model",
        mlx_standby_model="standby-model",
    ))

    attempted_models: list[str] = []

    async def fake_run_mlx_model(self, model_id, message, system, history=None):
        attempted_models.append(model_id)
        yield f"token-from-{model_id}"

    monkeypatch.setattr(LLMRouter, "_run_mlx_model", fake_run_mlx_model)

    tokens = [
        t async for t in router._stream_local_mlx(
            "hello", "sys", prefer_standby=True
        )
    ]

    assert attempted_models == ["standby-model"]
    assert tokens == ["token-from-standby-model"]


async def test_stream_local_mlx_default_order_tries_primary_first(monkeypatch):
    """Without prefer_standby, the primary model (xLAM) is tried first."""
    router = LLMRouter(LLMRouterConfig(
        mlx_primary_model="primary-model",
        mlx_standby_model="standby-model",
    ))

    attempted_models: list[str] = []

    async def fake_run_mlx_model(self, model_id, message, system, history=None):
        attempted_models.append(model_id)
        yield f"token-from-{model_id}"

    monkeypatch.setattr(LLMRouter, "_run_mlx_model", fake_run_mlx_model)

    tokens = [
        t async for t in router._stream_local_mlx(
            "hello", "sys", prefer_standby=False
        )
    ]

    assert attempted_models == ["primary-model"]
    assert tokens == ["token-from-primary-model"]


async def test_stream_local_mlx_falls_back_to_primary_when_standby_fails(monkeypatch):
    """If the standby model fails to load, the primary model is tried next."""
    router = LLMRouter(LLMRouterConfig(
        mlx_primary_model="primary-model",
        mlx_standby_model="standby-model",
    ))

    attempted_models: list[str] = []

    async def fake_run_mlx_model(self, model_id, message, system, history=None):
        attempted_models.append(model_id)
        if model_id == "standby-model":
            raise RuntimeError("standby model failed to load")
        yield f"token-from-{model_id}"

    monkeypatch.setattr(LLMRouter, "_run_mlx_model", fake_run_mlx_model)

    tokens = [
        t async for t in router._stream_local_mlx(
            "hello", "sys", prefer_standby=True
        )
    ]

    assert attempted_models == ["standby-model", "primary-model"]
    assert tokens == ["token-from-primary-model"]


# ---------------------------------------------------------------------------
# chat() end-to-end with prefer_local=True
# ---------------------------------------------------------------------------


async def test_chat_prefer_local_dispatches_only_to_local_mlx(monkeypatch):
    """
    chat(prefer_local=True) never attempts a cloud tier, even when cloud API
    keys are configured.
    """
    router = LLMRouter(LLMRouterConfig(
        groq_api_key="would-fail-if-called",
        cerebras_api_key="would-fail-if-called",
        gemini_api_key="would-fail-if-called",
    ))

    dispatched_tiers: list[LLMTier] = []

    async def fake_dispatch(self, tier, message, system, history, *, prefer_standby=False):
        dispatched_tiers.append(tier)
        yield "offline-response"

    monkeypatch.setattr(LLMRouter, "_dispatch", fake_dispatch)

    response = await router.chat("synthesize this", "system prompt", prefer_local=True)

    assert dispatched_tiers == [LLMTier.LOCAL_MLX]
    assert response == "offline-response"
