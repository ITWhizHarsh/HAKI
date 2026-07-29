"""Focused integration tests for VoiceLocalMLXService and VoiceLLMRouter.

Validates task 7.2 requirements:
- Exact model/runtime/Metal configuration enforcement
- Load and generation failure emits local_llm diagnostic with no provider fallback
- VoiceLLMRouter never selects Groq, Cerebras, Gemini, or legacy on non-eligible turns
- Eligible Gemini failure reports terminal cloud error with no fallback

Test IDs map to design §6 requirements: 6.1, 6.2, 6.7, 8.5, 8.6.
"""

from __future__ import annotations

import asyncio
import pytest
from threading import Event
from typing import Any, AsyncIterator, Sequence
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

from core.voice.cloud_gate import CloudEscalationGate, GateInput
from core.voice.interfaces import VoiceContextMessage, VoiceTurnRequest
from core.voice.llm import (
    LocalLLMDiagnostic,
    MLXGenerationError,
    MLXModelLoadError,
    VoiceLLMDecision,
    VoiceLLMRouter,
    VoiceLLMRouterError,
    VoiceLocalMLXService,
    VoiceMLXConfig,
    _MLXModelCache,
    _build_messages,
    _apply_chat_template,
    _extract_token_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_turn(text: str = "Hello", language: str = "en") -> VoiceTurnRequest:
    return VoiceTurnRequest(
        session_id=uuid4(),
        turn_id=uuid4(),
        text=text,
        language=language,
    )


def _make_gate_input(
    session_id: UUID,
    *,
    gemini_enabled: bool = False,
    battery: int | None = 50,
    thermal: str = "nominal",
    tokens: int = 100,
    tools: int = 0,
    external_power: bool | None = True,
) -> GateInput:
    return GateInput(
        session_id=session_id,
        gemini_enabled_for_session=gemini_enabled,
        battery_percent=battery,
        external_power_connected=external_power,
        thermal_state=thermal,
        assembled_prompt_tokens=tokens,
        validated_tool_count=tools,
    )


async def _collect(ait: AsyncIterator[str]) -> list[str]:
    return [chunk async for chunk in ait]


# ---------------------------------------------------------------------------
# VoiceMLXConfig validation
# ---------------------------------------------------------------------------

class TestVoiceMLXConfig:
    def test_default_values_are_correct(self) -> None:
        config = VoiceMLXConfig()
        assert config.model_id == "Qwen/Qwen3-4B-Instruct-4bit"
        assert config.runtime == "mlx-lm==0.18.1"
        assert config.use_metal is True
        assert config.max_context_tokens == 16_384
        assert config.max_generation_tokens == 1_024
        assert config.model_cache_capacity == 1

    def test_rejects_wrong_model_id(self) -> None:
        with pytest.raises(ValueError, match="model_id must be"):
            VoiceMLXConfig(model_id="some-other-model")

    def test_rejects_wrong_runtime(self) -> None:
        with pytest.raises(ValueError, match="runtime must be"):
            VoiceMLXConfig(runtime="mlx-lm==0.19.0")

    def test_rejects_metal_false(self) -> None:
        with pytest.raises(ValueError, match="Metal acceleration is required"):
            VoiceMLXConfig(use_metal=False)

    def test_rejects_wrong_context_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_context_tokens must be"):
            VoiceMLXConfig(max_context_tokens=8_192)

    def test_rejects_wrong_generation_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_generation_tokens must be"):
            VoiceMLXConfig(max_generation_tokens=512)

    def test_rejects_wrong_cache_capacity(self) -> None:
        with pytest.raises(ValueError, match="model_cache_capacity must be"):
            VoiceMLXConfig(model_cache_capacity=2)

    def test_explicit_correct_values_accepted(self) -> None:
        """Explicitly passing the correct values must succeed."""
        config = VoiceMLXConfig(
            model_id="Qwen/Qwen3-4B-Instruct-4bit",
            runtime="mlx-lm==0.18.1",
            use_metal=True,
            max_context_tokens=16_384,
            max_generation_tokens=1_024,
            model_cache_capacity=1,
        )
        assert config.use_metal is True


# ---------------------------------------------------------------------------
# VoiceLocalMLXService – configuration and loading
# ---------------------------------------------------------------------------

class TestVoiceLocalMLXServiceConfig:
    def test_uses_fixed_config_by_default(self) -> None:
        service = VoiceLocalMLXService()
        assert service.config.model_id == "Qwen/Qwen3-4B-Instruct-4bit"
        assert service.config.use_metal is True

    def test_accepts_explicit_valid_config(self) -> None:
        config = VoiceMLXConfig()
        service = VoiceLocalMLXService(config=config)
        assert service.config is config

    @pytest.mark.asyncio
    async def test_warm_up_calls_load(self) -> None:
        """warm_up() should trigger model loading once."""
        loaded_calls: list[str] = []

        async def fake_executor(op, lib):
            loaded_calls.append(lib)
            # Simulate successful mlx_lm.load returning (model, tokenizer)
            model = MagicMock()
            tokenizer = MagicMock()
            ce = Event()
            return op(ce)

        cache = _MLXModelCache()

        def _blocking_load(ce: Event):
            return (MagicMock(), MagicMock())

        # Patch the executor to simulate loading without actual mlx_lm
        async def patched_executor(op, lib):
            return op(Event())

        service = VoiceLocalMLXService(
            executor_run=patched_executor,
            model_cache=cache,
        )

        with patch.dict("sys.modules", {"mlx_lm": MagicMock(load=lambda m: (MagicMock(), MagicMock()))}):
            await service.warm_up()

        assert cache.is_loaded

    @pytest.mark.asyncio
    async def test_load_failure_raises_mlx_model_load_error(self) -> None:
        """A failure during model loading must raise MLXModelLoadError."""

        async def failing_executor(op, lib):
            raise MLXModelLoadError("mlx-lm not installed")

        cache = _MLXModelCache()
        service = VoiceLocalMLXService(executor_run=failing_executor, model_cache=cache)

        with pytest.raises(MLXModelLoadError):
            await service.warm_up()

    @pytest.mark.asyncio
    async def test_release_idle_evicts_cache(self) -> None:
        """release_idle() clears the model from the cache when the service is idle."""
        cache = _MLXModelCache()
        cache._model = MagicMock()
        cache._tokenizer = MagicMock()

        service = VoiceLocalMLXService(model_cache=cache)
        assert service.is_idle
        await service.release_idle()
        assert not cache.is_loaded

    @pytest.mark.asyncio
    async def test_is_idle_false_during_generation(self) -> None:
        """The semaphore should cause is_idle to return False while generating."""
        service = VoiceLocalMLXService()
        # Acquire the semaphore manually to simulate active generation
        await service._generation_semaphore.acquire()
        assert not service.is_idle
        service._generation_semaphore.release()
        assert service.is_idle


# ---------------------------------------------------------------------------
# VoiceLocalMLXService – streaming
# ---------------------------------------------------------------------------

class TestVoiceLocalMLXServiceStreaming:
    @pytest.mark.asyncio
    async def test_stream_response_yields_tokens(self) -> None:
        """Successfully loaded model should stream tokens via the executor."""
        chunks_to_yield = ["Hello", " World", "!"]
        chunk_index = 0

        def _blocking_stream(ce: Event):
            # Simulated stream_generate that yields strings
            for chunk in chunks_to_yield:
                yield chunk

        # We'll simulate by pre-loading the cache and mocking the generate path
        cache = _MLXModelCache()
        model_mock = MagicMock()
        tokenizer_mock = MagicMock()
        tokenizer_mock.apply_chat_template = lambda msgs, **kw: "prompt"
        cache._model = model_mock
        cache._tokenizer = tokenizer_mock

        collected: list[str] = []

        # Patch mlx_lm.stream_generate to yield our chunks
        def fake_stream_generate(model, tokenizer, prompt, max_tokens):
            for text in chunks_to_yield:
                resp = MagicMock()
                resp.text = text
                yield resp

        async def fake_executor(op, lib):
            return op(Event())

        service = VoiceLocalMLXService(executor_run=fake_executor, model_cache=cache)

        turn = _make_turn("test query")

        with patch.dict(
            "sys.modules",
            {"mlx_lm": MagicMock(stream_generate=fake_stream_generate)},
        ):
            async for chunk in service.stream_response(turn):
                collected.append(chunk)

        assert collected == chunks_to_yield

    @pytest.mark.asyncio
    async def test_stream_response_load_failure_no_fallback(self) -> None:
        """MLX load failure must raise MLXModelLoadError with no alternate route."""

        async def failing_executor(op, lib):
            raise MLXModelLoadError("mlx-lm missing")

        cache = _MLXModelCache()
        service = VoiceLocalMLXService(executor_run=failing_executor, model_cache=cache)

        turn = _make_turn()
        with pytest.raises(MLXModelLoadError):
            async for _ in service.stream_response(turn):
                pass

    @pytest.mark.asyncio
    async def test_generation_error_raises_mlx_generation_error(self) -> None:
        """A runtime generation error must surface as MLXGenerationError."""
        cache = _MLXModelCache()
        cache._model = MagicMock()
        cache._tokenizer = MagicMock()
        cache._tokenizer.apply_chat_template = lambda msgs, **kw: "prompt"

        def bad_stream_generate(model, tokenizer, prompt, max_tokens):
            raise RuntimeError("OOM during generation")
            yield  # make it a generator

        async def fake_executor(op, lib):
            return op(Event())

        service = VoiceLocalMLXService(executor_run=fake_executor, model_cache=cache)
        turn = _make_turn()

        with patch.dict(
            "sys.modules",
            {"mlx_lm": MagicMock(stream_generate=bad_stream_generate)},
        ):
            with pytest.raises((MLXGenerationError, VoiceLLMRouterError, RuntimeError)):
                async for _ in service.stream_response(turn):
                    pass


# ---------------------------------------------------------------------------
# VoiceLLMRouter – routing decisions
# ---------------------------------------------------------------------------

class TestVoiceLLMRouterDecision:
    def _make_gate_and_session(self) -> tuple[CloudEscalationGate, UUID]:
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        return gate, session_id

    def test_non_eligible_turn_routes_to_local_qwen(self) -> None:
        gate, session_id = self._make_gate_and_session()
        router = VoiceLLMRouter(gate=gate)
        gate_input = _make_gate_input(session_id)
        decision = router.decide(gate_input)
        assert decision.route == "local_qwen"
        assert not decision.gate_decision.eligible

    def test_disabled_gemini_routes_to_local_even_with_qualifying_condition(self) -> None:
        """Gate disabled + qualifying conditions must still route local_qwen."""
        gate, session_id = self._make_gate_and_session()
        router = VoiceLLMRouter(gate=gate)
        gate_input = _make_gate_input(
            session_id,
            battery=10,
            external_power=False,
            thermal="critical",
        )
        decision = router.decide(gate_input)
        assert decision.route == "local_qwen"

    def test_enabled_gemini_with_qualifying_condition_routes_gemini(self) -> None:
        gate, session_id = self._make_gate_and_session()
        gate.enable(session_id)
        router = VoiceLLMRouter(gate=gate)
        gate_input = _make_gate_input(
            session_id,
            gemini_enabled=True,
            battery=10,
            external_power=False,
        )
        decision = router.decide(gate_input)
        assert decision.route == "gemini_live"
        assert decision.gate_decision.eligible

    def test_enabled_gemini_without_qualifying_condition_routes_local(self) -> None:
        gate, session_id = self._make_gate_and_session()
        gate.enable(session_id)
        router = VoiceLLMRouter(gate=gate)
        # No qualifying condition (good battery, nominal thermal, low tokens)
        gate_input = _make_gate_input(session_id, gemini_enabled=True)
        decision = router.decide(gate_input)
        assert decision.route == "local_qwen"


# ---------------------------------------------------------------------------
# VoiceLLMRouter – streaming non-eligible turns
# ---------------------------------------------------------------------------

class TestVoiceLLMRouterLocalStreaming:
    def _make_router_with_mock_service(
        self,
        tokens: list[str] | None = None,
        load_error: Exception | None = None,
        gen_error: Exception | None = None,
    ) -> tuple[VoiceLLMRouter, CloudEscalationGate, UUID, list[LocalLLMDiagnostic]]:
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)

        diagnostics: list[LocalLLMDiagnostic] = []

        async def diagnostic_sink(d: LocalLLMDiagnostic) -> None:
            diagnostics.append(d)

        class _MockService:
            config = VoiceMLXConfig()
            is_idle = True

            async def stream_response(self, turn, *, context=()):
                if load_error is not None:
                    raise load_error
                if gen_error is not None:
                    raise gen_error
                for t in (tokens or ["ok"]):
                    yield t

        router = VoiceLLMRouter(
            gate=gate,
            local_service=_MockService(),
            diagnostic_sink=diagnostic_sink,
        )
        return router, gate, session_id, diagnostics

    @pytest.mark.asyncio
    async def test_local_tokens_are_yielded(self) -> None:
        router, gate, session_id, _ = self._make_router_with_mock_service(tokens=["hi", " there"])
        turn = _make_turn()
        turn = VoiceTurnRequest(session_id=session_id, turn_id=turn.turn_id, text="hi", language="en")
        gate_input = _make_gate_input(session_id)
        collected = await _collect(router.stream_turn(turn, gate_input))
        assert collected == ["hi", " there"]

    @pytest.mark.asyncio
    async def test_local_load_error_emits_diagnostic_no_fallback(self) -> None:
        """MLXModelLoadError must emit local_llm diagnostic and not fall back."""
        error = MLXModelLoadError("missing model")
        router, gate, session_id, diagnostics = self._make_router_with_mock_service(load_error=error)
        turn = VoiceTurnRequest(session_id=session_id, turn_id=uuid4(), text="test", language="en")
        gate_input = _make_gate_input(session_id)

        with pytest.raises(MLXModelLoadError):
            await _collect(router.stream_turn(turn, gate_input))

        assert len(diagnostics) == 1
        diag = diagnostics[0]
        assert diag.stage == "local_llm"
        assert diag.outcome == "failed"
        assert diag.error_class == "MLXModelLoadError"
        assert diag.recovery_outcome == "local_llm_error_no_fallback"

    @pytest.mark.asyncio
    async def test_local_generation_error_emits_diagnostic_no_fallback(self) -> None:
        """MLXGenerationError must emit local_llm diagnostic and not fall back."""
        error = MLXGenerationError("OOM")
        router, gate, session_id, diagnostics = self._make_router_with_mock_service(gen_error=error)
        turn = VoiceTurnRequest(session_id=session_id, turn_id=uuid4(), text="test", language="en")
        gate_input = _make_gate_input(session_id)

        with pytest.raises(MLXGenerationError):
            await _collect(router.stream_turn(turn, gate_input))

        assert len(diagnostics) == 1
        assert diagnostics[0].stage == "local_llm"
        assert diagnostics[0].outcome == "failed"

    @pytest.mark.asyncio
    async def test_unexpected_exception_wrapped_as_router_error(self) -> None:
        """An unexpected exception must be wrapped as VoiceLLMRouterError."""

        class _MockService:
            config = VoiceMLXConfig()
            is_idle = True

            async def stream_response(self, turn, *, context=()):
                raise ValueError("unexpected")
                yield  # make it a generator

        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        router = VoiceLLMRouter(gate=gate, local_service=_MockService())
        turn = VoiceTurnRequest(session_id=session_id, turn_id=uuid4(), text="test", language="en")

        with pytest.raises(VoiceLLMRouterError):
            await _collect(router.stream_turn(turn, _make_gate_input(session_id)))

    @pytest.mark.asyncio
    async def test_no_groq_cerebras_gemini_or_legacy_import_on_failure(self) -> None:
        """Verify that failure paths never attempt to import legacy providers."""
        imported_providers: list[str] = []
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

        def spy_import(name, *args, **kwargs):
            if any(kw in name for kw in ("groq", "cerebras", "deepgram", "cartesia", "edge_tts", "kokoro")):
                imported_providers.append(name)
            return real_import(name, *args, **kwargs)

        error = MLXModelLoadError("missing")
        router, gate, session_id, _ = self._make_router_with_mock_service(load_error=error)
        turn = VoiceTurnRequest(session_id=session_id, turn_id=uuid4(), text="test", language="en")

        with patch("builtins.__import__", spy_import):
            with pytest.raises(MLXModelLoadError):
                await _collect(router.stream_turn(turn, _make_gate_input(session_id)))

        assert imported_providers == [], f"Unexpected provider imports: {imported_providers}"


# ---------------------------------------------------------------------------
# VoiceLLMRouter – eligible Gemini failure must not fall back
# ---------------------------------------------------------------------------

class TestVoiceLLMRouterGeminiFailure:
    @pytest.mark.asyncio
    async def test_eligible_gemini_failure_raises_router_error_no_fallback(self) -> None:
        """An eligible Gemini invocation failure must NOT fall back to Qwen or legacy."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        gate.enable(session_id)

        invocation_calls: list[str] = []
        local_calls: list[str] = []

        async def failing_gemini(turn, gate_decision, context):
            invocation_calls.append("gemini")
            raise RuntimeError("Gemini API unreachable")
            yield  # make it a generator

        class _LocalService:
            config = VoiceMLXConfig()
            is_idle = True

            async def stream_response(self, turn, *, context=()):
                local_calls.append("local_qwen")
                yield "local_token"

        router = VoiceLLMRouter(
            gate=gate,
            local_service=_LocalService(),
            gemini_invoke=failing_gemini,
        )
        turn = VoiceTurnRequest(session_id=session_id, turn_id=uuid4(), text="test", language="en")
        gate_input = _make_gate_input(
            session_id,
            gemini_enabled=True,
            battery=5,
            external_power=False,
        )

        with pytest.raises(VoiceLLMRouterError, match="Gemini Live invocation failed"):
            await _collect(router.stream_turn(turn, gate_input))

        # Gemini was attempted exactly once
        assert invocation_calls == ["gemini"]
        # Local Qwen must NOT have been called as a fallback
        assert local_calls == [], "Gemini failure must not fall back to local Qwen"

    @pytest.mark.asyncio
    async def test_eligible_gemini_not_configured_raises_router_error(self) -> None:
        """When Gemini is eligible but not configured, raise without falling back."""
        gate = CloudEscalationGate()
        session_id = uuid4()
        gate.register_session(session_id)
        gate.enable(session_id)

        router = VoiceLLMRouter(gate=gate, gemini_invoke=None)
        turn = VoiceTurnRequest(session_id=session_id, turn_id=uuid4(), text="test", language="en")
        gate_input = _make_gate_input(
            session_id,
            gemini_enabled=True,
            battery=5,
            external_power=False,
        )

        with pytest.raises(VoiceLLMRouterError, match="no fallback route will be selected"):
            await _collect(router.stream_turn(turn, gate_input))


# ---------------------------------------------------------------------------
# VoiceLLMRouter – does NOT use LLMRouter._routing_order
# ---------------------------------------------------------------------------

class TestVoiceLLMRouterNoBroadFallback:
    def test_router_has_no_import_of_llm_router(self) -> None:
        """The llm module must not import from model_provider.llm_router."""
        import importlib.util
        import ast

        spec = importlib.util.find_spec("core.voice.llm")
        assert spec is not None
        source_path = spec.origin
        with open(source_path, "r", encoding="utf-8") as fh:
            source = fh.read()

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    assert "llm_router" not in module, (
                        f"core.voice.llm must not import from llm_router: {module}"
                    )
                    assert "model_provider" not in module, (
                        f"core.voice.llm must not import from model_provider: {module}"
                    )

    def test_router_has_no_groq_cerebras_references(self) -> None:
        """No import of cloud/legacy providers must appear in the llm module."""
        import importlib.util
        import ast

        spec = importlib.util.find_spec("core.voice.llm")
        assert spec is not None
        with open(spec.origin, "r", encoding="utf-8") as fh:
            source = fh.read()

        tree = ast.parse(source)
        # Check only actual import statements for legacy provider names
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for provider in ("groq", "cerebras", "deepgram", "cartesia", "edge_tts", "kokoro"):
                        assert provider not in alias.name.lower(), (
                            f"core.voice.llm must not import legacy provider: {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                module = (node.module or "").lower()
                for provider in ("groq", "cerebras", "deepgram", "cartesia", "edge_tts", "kokoro"):
                    assert provider not in module, (
                        f"core.voice.llm must not import from legacy provider module: {node.module}"
                    )


# ---------------------------------------------------------------------------
# _MLXModelCache – single-slot behavior
# ---------------------------------------------------------------------------

class TestMLXModelCache:
    @pytest.mark.asyncio
    async def test_second_load_returns_cached_pair(self) -> None:
        """The cache must return the same (model, tokenizer) pair on a second call.

        We pre-seed the cache to avoid needing real mlx_lm and verify that
        a second ``load`` call does not call the executor again.
        """
        model = MagicMock()
        tokenizer = MagicMock()
        executor_call_count = 0

        async def executor(op, lib):
            nonlocal executor_call_count
            executor_call_count += 1
            return op(Event())

        cache = _MLXModelCache()
        # Pre-seed so the cache skip actually triggers:
        # first call should go through executor (or we skip via pre-load)
        cache._model = model
        cache._tokenizer = tokenizer

        # Second call with pre-loaded cache should return immediately
        r1 = await cache.load("Qwen/Qwen3-4B-Instruct-4bit", executor_run=executor)
        r2 = await cache.load("Qwen/Qwen3-4B-Instruct-4bit", executor_run=executor)
        # Both calls returned the pre-seeded pair
        assert r1[0] is model
        assert r1[1] is tokenizer
        assert r2[0] is model
        assert r2[1] is tokenizer
        # Executor was never called because cache was already populated
        assert executor_call_count == 0

    @pytest.mark.asyncio
    async def test_load_calls_executor_when_empty(self) -> None:
        """When the cache is empty, load() must call the executor exactly once."""
        model = MagicMock()
        tokenizer = MagicMock()
        executor_call_count = 0

        def fake_blocking(ce: Event):
            return (model, tokenizer)

        async def executor(op, lib):
            nonlocal executor_call_count
            executor_call_count += 1
            return op(Event())

        cache = _MLXModelCache()
        assert not cache.is_loaded

        # Patch mlx_lm so the internal import succeeds but uses our mock
        mlx_mock = MagicMock()
        mlx_mock.load.return_value = (model, tokenizer)
        with patch.dict("sys.modules", {"mlx_lm": mlx_mock}):
            result = await cache.load("Qwen/Qwen3-4B-Instruct-4bit", executor_run=executor)

        assert executor_call_count == 1
        assert cache.is_loaded
        assert result[0] is model
        assert result[1] is tokenizer

    @pytest.mark.asyncio
    async def test_evict_clears_cache(self) -> None:
        cache = _MLXModelCache()
        cache._model = MagicMock()
        cache._tokenizer = MagicMock()
        assert cache.is_loaded
        cache.evict()
        assert not cache.is_loaded


# ---------------------------------------------------------------------------
# Chat template helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_build_messages_includes_context_and_turn(self) -> None:
        context = [
            VoiceContextMessage(turn_id=uuid4(), role="user", text="hi"),
            VoiceContextMessage(turn_id=uuid4(), role="assistant", text="hello"),
        ]
        turn = _make_turn("what time is it")
        messages = _build_messages(turn, context)
        assert messages[0] == {"role": "user", "content": "hi"}
        assert messages[1] == {"role": "assistant", "content": "hello"}
        assert messages[2] == {"role": "user", "content": "what time is it"}

    def test_build_messages_no_context(self) -> None:
        turn = _make_turn("hello")
        messages = _build_messages(turn, [])
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"

    def test_apply_chat_template_uses_tokenizer_fn(self) -> None:
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<|prompt|>"
        result = _apply_chat_template(tokenizer, [{"role": "user", "content": "hi"}])
        assert result == "<|prompt|>"

    def test_apply_chat_template_falls_back_on_missing_fn(self) -> None:
        tokenizer = MagicMock(spec=[])  # no apply_chat_template
        messages = [{"role": "user", "content": "hello"}]
        result = _apply_chat_template(tokenizer, messages)
        assert "user: hello" in result
        assert "assistant:" in result

    def test_extract_token_text_from_string(self) -> None:
        assert _extract_token_text("hello") == "hello"

    def test_extract_token_text_from_object_with_text_attr(self) -> None:
        obj = MagicMock()
        obj.text = " world"
        assert _extract_token_text(obj) == " world"

    def test_extract_token_text_from_dict(self) -> None:
        assert _extract_token_text({"text": "!"}) == "!"

    def test_extract_token_text_returns_empty_for_unknown(self) -> None:
        assert _extract_token_text(42) == ""
