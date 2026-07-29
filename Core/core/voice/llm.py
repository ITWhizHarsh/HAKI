"""Fixed local Qwen MLX service and voice-specific LLM router.

Only ``Qwen/Qwen3-4B-Instruct-4bit`` through ``mlx-lm==0.18.1`` with Metal
acceleration is used as the normal voice LLM route.  This module intentionally
does not import or delegate to the broad ``LLMRouter``, ``Groq``, ``Cerebras``,
``Gemini``, or any other cloud/legacy provider.

``VoiceLLMRouter`` calls ``CloudEscalationGate`` first; turns not marked
eligible by the gate are sent to ``VoiceLocalMLXService``.  An eligible Gemini
failure is reported as a ``CloudInvocationFailure`` with no fallback.  A Qwen
load or terminal generation error ends the affected turn with a ``local_llm``
diagnostic and a user-facing error.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from threading import Event
from typing import Awaitable, Literal
from uuid import UUID

from .cloud_gate import (
    CloudEscalationGate,
    CloudInvocationFailure,
    GateDecision,
    GateInput,
)
from .interfaces import (
    LocalVoiceLLM,
    VoiceContextMessage,
    VoiceLanguage,
    VoiceTurnRequest,
)
from .tools import ToolCallResult, VoiceToolAdapter, _try_parse_tool_call

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed MLX configuration
# ---------------------------------------------------------------------------

_REQUIRED_MODEL_ID = "Qwen/Qwen3-4B-Instruct-4bit"
_REQUIRED_RUNTIME = "mlx-lm==0.18.1"
_MAX_CONTEXT_TOKENS = 16_384
_MAX_GENERATION_TOKENS = 1_024
_MODEL_CACHE_CAPACITY = 1


@dataclass(frozen=True, slots=True)
class VoiceMLXConfig:
    """Immutable, design-mandated configuration for the local Qwen service.

    Validation ensures the caller cannot accidentally select a different model
    ID, runtime, or context limit.  Metal is always enabled; no code path may
    set ``use_metal=False``.
    """

    model_id: str = _REQUIRED_MODEL_ID
    runtime: str = _REQUIRED_RUNTIME
    use_metal: bool = True
    max_context_tokens: int = _MAX_CONTEXT_TOKENS
    max_generation_tokens: int = _MAX_GENERATION_TOKENS
    model_cache_capacity: int = _MODEL_CACHE_CAPACITY

    def __post_init__(self) -> None:
        if self.model_id != _REQUIRED_MODEL_ID:
            raise ValueError(
                f"voice LLM model_id must be {_REQUIRED_MODEL_ID!r}; "
                f"other models are not permitted in the voice runtime"
            )
        if self.runtime != _REQUIRED_RUNTIME:
            raise ValueError(
                f"voice LLM runtime must be {_REQUIRED_RUNTIME!r}; "
                f"other runtimes are not permitted in the voice runtime"
            )
        if not self.use_metal:
            raise ValueError(
                "Metal acceleration is required in the voice runtime; "
                "use_metal must not be False"
            )
        if (
            not isinstance(self.max_context_tokens, int)
            or isinstance(self.max_context_tokens, bool)
            or self.max_context_tokens != _MAX_CONTEXT_TOKENS
        ):
            raise ValueError(
                f"max_context_tokens must be {_MAX_CONTEXT_TOKENS}"
            )
        if (
            not isinstance(self.max_generation_tokens, int)
            or isinstance(self.max_generation_tokens, bool)
            or self.max_generation_tokens != _MAX_GENERATION_TOKENS
        ):
            raise ValueError(
                f"max_generation_tokens must be {_MAX_GENERATION_TOKENS}"
            )
        if (
            not isinstance(self.model_cache_capacity, int)
            or isinstance(self.model_cache_capacity, bool)
            or self.model_cache_capacity != _MODEL_CACHE_CAPACITY
        ):
            raise ValueError(
                f"model_cache_capacity must be {_MODEL_CACHE_CAPACITY}"
            )


# ---------------------------------------------------------------------------
# Diagnostic sink type alias (shared with the voice diagnostic store)
# ---------------------------------------------------------------------------

class LocalLLMDiagnosticStage(str, Enum):
    LOCAL_LLM = "local_llm"


@dataclass(frozen=True, slots=True)
class LocalLLMDiagnostic:
    """Content-free diagnostic emitted on load or generation failure."""

    stage: Literal["local_llm"] = "local_llm"
    outcome: Literal["failed", "started", "completed", "cancelled"] = "failed"
    session_id: UUID | None = None
    turn_id: UUID | None = None
    error_class: str | None = None
    recovery_outcome: str | None = None

    def __post_init__(self) -> None:
        if self.stage != "local_llm":
            raise ValueError("LocalLLMDiagnostic stage must be local_llm")


DiagnosticSink = Callable[[LocalLLMDiagnostic], Awaitable[None] | None]


# ---------------------------------------------------------------------------
# MLX model cache
# ---------------------------------------------------------------------------

class MLXModelLoadError(RuntimeError):
    """The local Qwen model could not be loaded from the pre-provisioned path."""


class MLXGenerationError(RuntimeError):
    """MLX terminated a token stream with an unrecoverable error."""


@dataclass
class _MLXModelCache:
    """One-slot cache for the warmed MLX model and tokenizer.

    The capacity is fixed at one to match the design requirement.  Evicting
    and reloading the cached model is intentionally done only by
    ``VoiceResourceManager`` (``release_idle``), not by the LLM service itself.
    """

    _model: object = field(default=None, init=False, repr=False)
    _tokenizer: object = field(default=None, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    async def load(
        self,
        model_id: str,
        *,
        executor_run: Callable[[Callable[[Event], object], str], Awaitable[object]],
    ) -> tuple[object, object]:
        """Load the model if the cache is empty; return the cached pair otherwise."""
        async with self._lock:
            if self.is_loaded:
                return self._model, self._tokenizer

            def _blocking_load(cancel_event: Event) -> tuple[object, object]:
                try:
                    from mlx_lm import load as mlx_load  # type: ignore[import]
                except ImportError as exc:
                    raise MLXModelLoadError(
                        f"mlx-lm is not installed; cannot load {model_id}"
                    ) from exc
                # Metal is always used; never pass metal=False
                model, tokenizer = mlx_load(model_id)
                return model, tokenizer

            try:
                result = await executor_run(_blocking_load, "mlx-lm")
            except MLXModelLoadError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise MLXModelLoadError(
                    f"failed to load local MLX model {model_id}: {exc}"
                ) from exc

            self._model, self._tokenizer = result
            return self._model, self._tokenizer

    def evict(self) -> None:
        """Remove the cached model and tokenizer; called only by resource manager."""
        self._model = None
        self._tokenizer = None


# A single module-level cache shared between all service instances so
# different turns in the same warm session reuse the loaded weights.
_MODULE_MODEL_CACHE = _MLXModelCache()


# ---------------------------------------------------------------------------
# Local Qwen streaming service
# ---------------------------------------------------------------------------

class VoiceLocalMLXService:
    """Stream ordered LLM tokens from the local Qwen3-4B-Instruct-4bit model.

    This service implements the ``LocalVoiceLLM`` protocol.  It is the *only*
    LLM backend used for non-eligible voice turns; it does not delegate to
    ``LLMRouter`` or select any cloud provider.

    One ``asyncio.Semaphore`` serializes concurrent generation requests so a
    single loaded model is never called re-entrantly.  Cancellation between
    decode steps is supported through a cooperative threading event.
    """

    def __init__(
        self,
        *,
        config: VoiceMLXConfig | None = None,
        executor_run: Callable[[Callable[[Event], object], str], Awaitable[object]] | None = None,
        model_cache: _MLXModelCache | None = None,
        diagnostic_sink: DiagnosticSink | None = None,
    ) -> None:
        self._config = config or VoiceMLXConfig()
        self._executor_run = executor_run or self._default_executor_run
        self._cache = model_cache if model_cache is not None else _MODULE_MODEL_CACHE
        self._diagnostic_sink = diagnostic_sink
        self._generation_semaphore = asyncio.Semaphore(1)

    @property
    def config(self) -> VoiceMLXConfig:
        return self._config

    @property
    def is_idle(self) -> bool:
        """True when no generation is in progress; used by resource manager."""
        return self._generation_semaphore._value == 1  # noqa: SLF001

    async def warm_up(self) -> None:
        """Pre-load the model so the first voice turn has no cold-start penalty."""
        await self._load_model()

    async def release_idle(self) -> None:
        """Evict the cached model; called only when the service is idle."""
        if not self.is_idle:
            raise RuntimeError("VoiceLocalMLXService: cannot release while generation is active")
        self._cache.evict()

    async def _load_model(self) -> tuple[object, object]:
        """Ensure the model is loaded, raising MLXModelLoadError on failure."""
        return await self._cache.load(
            self._config.model_id,
            executor_run=self._executor_run,
        )

    async def stream_response(
        self,
        turn: VoiceTurnRequest,
        *,
        context: Sequence[VoiceContextMessage] = (),
    ) -> AsyncIterator[str]:
        """Yield incremental text chunks from the local Qwen model.

        Loads the model if necessary (a Qwen load failure raises
        ``MLXModelLoadError``).  The semaphore serializes concurrent
        generation; cancellation is propagated cooperatively between decode
        steps via a threading ``Event``.
        """
        model, tokenizer = await self._load_model()

        async with self._generation_semaphore:
            async for chunk in self._generate(turn, context, model, tokenizer):
                yield chunk

    async def _generate(
        self,
        turn: VoiceTurnRequest,
        context: Sequence[VoiceContextMessage],
        model: object,
        tokenizer: object,
    ) -> AsyncIterator[str]:
        """Run the MLX token stream through the blocking executor, yielding chunks."""
        messages = _build_messages(turn, context)
        cancel_event = Event()
        chunks: asyncio.Queue[str | BaseException] = asyncio.Queue()

        def _blocking_generate(ce: Event) -> None:
            """Run mlx_lm.stream_generate and push each token into the queue."""
            try:
                from mlx_lm import stream_generate as mlx_stream  # type: ignore[import]
            except ImportError as exc:
                chunks.put_nowait(MLXGenerationError(f"mlx-lm is not installed: {exc}"))
                return

            try:
                prompt = _apply_chat_template(tokenizer, messages)
                generator = mlx_stream(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=self._config.max_generation_tokens,
                )
                for response in generator:
                    if ce.is_set():
                        break
                    token_text = _extract_token_text(response)
                    if token_text:
                        chunks.put_nowait(token_text)
                # Sentinel: None signals normal completion
                chunks.put_nowait(None)
            except Exception as exc:  # noqa: BLE001
                chunks.put_nowait(MLXGenerationError(f"MLX generation error: {exc}"))

        # Run the blocking generation on the single-worker executor
        gen_task = asyncio.ensure_future(
            self._executor_run(_blocking_generate, "mlx-lm")
        )

        try:
            while True:
                # Poll the chunk queue; also handle executor task completion
                try:
                    item = chunks.get_nowait()
                except asyncio.QueueEmpty:
                    if gen_task.done():
                        # Drain any remaining items then exit
                        while not chunks.empty():
                            remaining = chunks.get_nowait()
                            if remaining is None:
                                return
                            if isinstance(remaining, BaseException):
                                raise remaining
                            yield remaining
                        return
                    # Yield control briefly to avoid busy-wait
                    await asyncio.sleep(0)
                    continue

                if item is None:
                    # Normal stream end
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        except asyncio.CancelledError:
            cancel_event.set()
            gen_task.cancel()
            try:
                await gen_task
            except (asyncio.CancelledError, Exception):
                pass
            raise

    @staticmethod
    async def _default_executor_run(
        operation: Callable[[Event], object],
        library: str,
    ) -> object:
        """Fallback executor used when no pipeline executor is injected."""
        loop = asyncio.get_running_loop()
        cancel_event = Event()
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="haki-voice-local-model") as pool:
            future = loop.run_in_executor(pool, operation, cancel_event)
            try:
                return await future
            except asyncio.CancelledError:
                cancel_event.set()
                future.cancel()
                raise


# ---------------------------------------------------------------------------
# VoiceLLMRouter
# ---------------------------------------------------------------------------

class VoiceLLMRouterError(RuntimeError):
    """A voice LLM routing error with no legacy or cloud fallback."""


@dataclass(frozen=True, slots=True)
class VoiceLLMDecision:
    """Route decision for one voice turn with its gate diagnostic."""

    route: Literal["local_qwen", "gemini_live"]
    gate_decision: GateDecision
    session_id: UUID


class VoiceLLMRouter:
    """Route voice turns to local Qwen or (when explicitly eligible) Gemini Live.

    This router intentionally:
    - does NOT delegate to ``LLMRouter._routing_order()``
    - does NOT fall back to Groq, Cerebras, Gemini, or legacy voice routes
    - calls ``CloudEscalationGate.evaluate`` for every turn
    - invokes Qwen for all non-eligible turns
    - treats a Qwen load or terminal generation error as a ``local_llm``
      stage failure with no retry or alternate provider

    The router does not hold a reference to ``VoiceSession``; all gate state is
    owned by the gate itself and identified by session ID.
    """

    def __init__(
        self,
        *,
        gate: CloudEscalationGate,
        local_service: VoiceLocalMLXService | None = None,
        gemini_invoke: Callable[[VoiceTurnRequest, GateDecision, Sequence[VoiceContextMessage]], AsyncIterator[str]] | None = None,
        diagnostic_sink: DiagnosticSink | None = None,
        tool_adapter: VoiceToolAdapter | None = None,
    ) -> None:
        self._gate = gate
        self._local_service = local_service or VoiceLocalMLXService(
            diagnostic_sink=diagnostic_sink
        )
        self._gemini_invoke = gemini_invoke
        self._diagnostic_sink = diagnostic_sink
        self._tool_adapter = tool_adapter

    @property
    def local_service(self) -> VoiceLocalMLXService:
        return self._local_service

    def decide(self, gate_input: GateInput) -> VoiceLLMDecision:
        """Evaluate the gate and return the routing decision for one turn.

        Only the gate's own active-session enablement state is authoritative.
        The ``gemini_enabled_for_session`` field in ``gate_input`` is treated
        as input metadata only; the gate derives the effective decision from its
        own registered session state.
        """
        gate_decision = self._gate.evaluate(gate_input)
        return VoiceLLMDecision(
            route=gate_decision.route,
            gate_decision=gate_decision,
            session_id=gate_input.session_id,
        )

    async def stream_turn(
        self,
        turn: VoiceTurnRequest,
        gate_input: GateInput,
        *,
        context: Sequence[VoiceContextMessage] = (),
    ) -> AsyncIterator[str]:
        """Yield text tokens for one voice turn.

        For non-eligible turns: invokes ``VoiceLocalMLXService``.
        For eligible turns: invokes the Gemini callable if provided, otherwise
        falls back to a local_llm failure (not to Qwen, another cloud, or legacy).

        Any MLX load or generation failure emits a ``local_llm`` diagnostic and
        re-raises without selecting any other route.
        """
        decision = self.decide(gate_input)
        if not decision.gate_decision.eligible:
            async for chunk in self._stream_local(turn, context=context):
                yield chunk
        else:
            async for chunk in self._stream_gemini_eligible(turn, decision, context=context):
                yield chunk

    async def _stream_local(
        self,
        turn: VoiceTurnRequest,
        *,
        context: Sequence[VoiceContextMessage],
    ) -> AsyncIterator[str]:
        """Stream from local Qwen; intercept tool calls when tool_adapter is set."""
        try:
            if self._tool_adapter is None:
                # Fast path: no tool interception
                async for chunk in self._local_service.stream_response(turn, context=context):
                    yield chunk
            else:
                async for chunk in self._stream_local_with_tools(turn, context=context):
                    yield chunk
        except asyncio.CancelledError:
            raise
        except (MLXModelLoadError, MLXGenerationError) as exc:
            await self._emit_diagnostic(
                LocalLLMDiagnostic(
                    stage="local_llm",
                    outcome="failed",
                    session_id=turn.session_id,
                    turn_id=turn.turn_id,
                    error_class=type(exc).__name__,
                    recovery_outcome="local_llm_error_no_fallback",
                )
            )
            raise
        except Exception as exc:  # noqa: BLE001
            await self._emit_diagnostic(
                LocalLLMDiagnostic(
                    stage="local_llm",
                    outcome="failed",
                    session_id=turn.session_id,
                    turn_id=turn.turn_id,
                    error_class=type(exc).__name__,
                    recovery_outcome="local_llm_error_no_fallback",
                )
            )
            raise VoiceLLMRouterError(
                f"local Qwen generation failed for turn {turn.turn_id}: {exc}"
            ) from exc

    async def _stream_local_with_tools(
        self,
        turn: VoiceTurnRequest,
        *,
        context: Sequence[VoiceContextMessage],
    ) -> AsyncIterator[str]:
        """Stream from local Qwen with mid-generation tool call interception.

        Accumulates tokens.  When a complete JSON object is detected, generation
        stops and the tool call is executed.  The result is appended to context
        and generation resumes from a follow-up call.  When no tool call is
        detected the accumulated tokens are yielded normally.
        """
        accumulated = ""
        tokens_before_tool: list[str] = []
        tool_detected = False

        async for chunk in self._local_service.stream_response(turn, context=context):
            accumulated += chunk
            parsed = _try_parse_tool_call(accumulated)
            if parsed is not None and "tool" in parsed:
                # Tool call detected; stop yielding raw tokens
                tool_detected = True
                break
            # No complete tool call yet; yield the token
            tokens_before_tool.append(chunk)
            yield chunk

        if not tool_detected:
            # Normal completion — nothing more to do
            return

        # Execute the tool call
        assert self._tool_adapter is not None
        result: ToolCallResult | None = await self._tool_adapter.execute_tool_call(
            accumulated,
            turn_id=turn.turn_id,
            session_id=turn.session_id,
        )

        if result is None:
            # Schema rejection — emit a safe diagnostic text and stop
            yield "[Tool call could not be processed. Please try rephrasing your request.]"
            return

        # Build tool result context message to append for the follow-up generation
        if result.success:
            tool_result_text = f"[Tool result for {result.tool_name}]: {result.data}"
        else:
            tool_result_text = f"[Tool call failed for {result.tool_name}]: {result.error_message}"

        # Append tool result to context for the follow-up generation
        tool_result_message = VoiceContextMessage(
            turn_id=turn.turn_id,
            role="assistant",
            text=tool_result_text,
        )
        extended_context = list(context) + [tool_result_message]

        # Re-invoke generation with the extended context (non-recursive — one tool call per turn)
        async for chunk in self._local_service.stream_response(turn, context=extended_context):
            yield chunk

    async def _stream_gemini_eligible(
        self,
        turn: VoiceTurnRequest,
        decision: VoiceLLMDecision,
        *,
        context: Sequence[VoiceContextMessage],
    ) -> AsyncIterator[str]:
        """Invoke Gemini Live; report a terminal cloud failure with no alternate route."""
        if self._gemini_invoke is None:
            # Gemini is eligible but no invocation callable is registered.
            # This is a configuration error; report it as a cloud failure with
            # no fallback (not Qwen, not legacy).
            failure = self._gate.report_eligible_invocation_failure(
                decision.gate_decision,
                "GeminiInvocationNotConfigured",
            )
            await self._emit_cloud_failure_diagnostic(failure, turn.turn_id)
            raise VoiceLLMRouterError(
                "Gemini Live is eligible for this turn but no invocation callable was provided; "
                "the turn cannot proceed and no fallback route will be selected"
            )
        try:
            async for chunk in self._gemini_invoke(turn, decision.gate_decision, context):
                yield chunk
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            failure = self._gate.report_eligible_invocation_failure(decision.gate_decision, exc)
            await self._emit_cloud_failure_diagnostic(failure, turn.turn_id)
            raise VoiceLLMRouterError(
                f"Gemini Live invocation failed for turn {turn.turn_id} "
                f"with no fallback: {exc}"
            ) from exc

    async def _emit_diagnostic(self, diagnostic: LocalLLMDiagnostic) -> None:
        if self._diagnostic_sink is None:
            _logger.warning(
                "local_llm diagnostic: stage=%s outcome=%s error_class=%s",
                diagnostic.stage,
                diagnostic.outcome,
                diagnostic.error_class,
            )
            return
        import inspect

        result = self._diagnostic_sink(diagnostic)
        if inspect.isawaitable(result):
            await result

    async def _emit_cloud_failure_diagnostic(
        self,
        failure: CloudInvocationFailure,
        turn_id: UUID,
    ) -> None:
        event = failure.diagnostic_event(turn_id=turn_id)
        _logger.warning(
            "cloud_gate diagnostic: stage=%s outcome=%s error_class=%s",
            event.stage,
            event.outcome,
            event.error_class,
        )


# ---------------------------------------------------------------------------
# Chat template helpers
# ---------------------------------------------------------------------------

def _build_messages(
    turn: VoiceTurnRequest,
    context: Sequence[VoiceContextMessage],
) -> list[dict[str, str]]:
    """Assemble an OpenAI-style message list from session-owned context."""
    messages: list[dict[str, str]] = []
    for msg in context:
        role = "user" if msg.role == "user" else "assistant"
        messages.append({"role": role, "content": msg.text})
    messages.append({"role": "user", "content": turn.text})
    return messages


def _apply_chat_template(tokenizer: object, messages: list[dict[str, str]]) -> str:
    """Apply the model's chat template if available, else fall back to raw text."""
    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    if apply_fn is not None:
        try:
            prompt = apply_fn(messages, tokenize=False, add_generation_prompt=True)
            if isinstance(prompt, str):
                return prompt
        except Exception:  # noqa: BLE001
            pass
    # Minimal fallback: concatenate role-prefixed turns
    parts: list[str] = []
    for msg in messages:
        parts.append(f"{msg['role']}: {msg['content']}")
    parts.append("assistant:")
    return "\n".join(parts)


def _extract_token_text(response: object) -> str:
    """Extract the text field from a stream_generate response object."""
    # mlx_lm stream_generate yields objects with a ``text`` attribute or
    # plain strings depending on the version.
    if isinstance(response, str):
        return response
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    # Some versions yield a dict
    if isinstance(response, dict):
        text = response.get("text", "")
        return text if isinstance(text, str) else ""
    return ""


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "DiagnosticSink",
    "LocalLLMDiagnostic",
    "LocalLLMDiagnosticStage",
    "MLXGenerationError",
    "MLXModelLoadError",
    "VoiceLLMDecision",
    "VoiceLLMRouter",
    "VoiceLLMRouterError",
    "VoiceLocalMLXService",
    "VoiceMLXConfig",
]
