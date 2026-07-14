"""
LLM Router — Heterogeneous compute scheduling for HAKI (2026 architecture).

Cloud tier routing order:
    FAST (≤8k tokens):   Groq (Llama-3.3-70B) → Cerebras (Llama-4-Scout) → Gemini
    LARGE (>8k tokens):  Gemini 2.5 Flash (1M ctx) → Groq → Cerebras

Local tier (ANE / GPU-free, total ≤3.2GB active):
    Primary:   mlx-community/xLAM-2-3b-fc-r-4bit  (~1.85 GB) — tool-calling specialist
    Standby:   PrismML/Bonsai-8B-mlx-1bit          (~1.28 GB) — cold offline fallback

Mood metadata is stripped from user input and re-injected as a system-prompt
persona instruction before any tier is called.

Env vars read at construction time:
    HAKI_GROQ_API_KEY
    HAKI_CEREBRAS_API_KEY
    HAKI_GEMINI_API_KEY
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mood metadata
# ---------------------------------------------------------------------------

_MOOD_RE = re.compile(r"^\[METADATA:\s*MOOD=([A-Z_]+)\]\s*", re.IGNORECASE)


def parse_mood_metadata(text: str) -> tuple[str, str | None]:
    """Strip [METADATA: MOOD=TAG] prefix. Returns (clean_text, tag | None)."""
    m = _MOOD_RE.match(text)
    if m:
        return text[m.end():], m.group(1).upper()
    return text, None


def mood_to_persona_instruction(mood: str | None) -> str:
    """Return a Hinglish persona tone instruction for the given mood tag."""
    if mood is None:
        return ""
    return {
        "ANGRY_SHOUT": (
            "The user sounds frustrated/angry. Respond with calm, witty Hinglish. "
            'Style: "Ok chill out boss, handle kar rahe hain — thoda relax kar." '
            "De-escalate while keeping it light."
        ),
        "SAD_LOW_ENERGY": (
            "The user sounds sad or low-energy. Respond with warm, encouraging "
            "Hinglish. Uplift them gently."
        ),
    }.get(mood, "")


# ---------------------------------------------------------------------------
# Tier enum
# ---------------------------------------------------------------------------


class LLMTier(str, Enum):
    GROQ      = "groq"
    CEREBRAS  = "cerebras"
    GEMINI    = "gemini"
    LOCAL_MLX = "local_mlx"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LLMRouterConfig:
    # Cloud keys
    groq_api_key:     str | None = None
    cerebras_api_key: str | None = None
    gemini_api_key:   str | None = None

    # Model IDs
    groq_model:     str = "llama-3.3-70b-versatile"
    cerebras_model: str = "llama-3.3-70b"
    gemini_model:   str = "gemini-2.5-flash"

    # Local MLX models — GPU-free, ANE/CPU only
    mlx_primary_model:  str = "mlx-community/xLAM-2-3b-fc-r-4bit"   # 1.85 GB
    mlx_standby_model:  str = "mlx-community/Bonsai-8B-1bit"         # 1.28 GB

    # Routing threshold: chars above this → prefer Gemini large context
    large_context_threshold: int = 8_000   # ≈ 2k tokens

    request_timeout: float = 30.0


# ---------------------------------------------------------------------------
# LLM Router
# ---------------------------------------------------------------------------


class LLMRouter:
    """
    Cascading streaming LLM router.

    Never surfaces partial responses on failure — silently advances to next
    tier.  All state is read-only after construction (thread-safe per call).
    """

    def __init__(self, config: LLMRouterConfig | None = None) -> None:
        self._cfg = config or LLMRouterConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def stream_chat(
        self,
        user_message: str,
        system_prompt: str = "",
        *,
        prefer_large_context: bool = False,
        prefer_local: bool = False,
        history: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream tokens for *user_message*.

        Strips mood metadata → builds full system prompt → tries tiers in order.

        Parameters
        ----------
        prefer_local : bool
            When True, bypass all cloud tiers and route exclusively to the
            local MLX tier, trying the standby model (Bonsai-8B) before the
            primary model (xLAM). Used by the Heavy Pass memory pipeline to
            keep memory synthesis fully offline.
        history : list[dict] | None
            Prior conversation turns as ``{"role": "user"|"assistant",
            "content": str}`` items.  Spliced between the system prompt and
            the current user message so the model has running context.
            Mood metadata is stripped from the *current* user_message only —
            history items are passed through untouched.
        """
        clean_msg, mood = parse_mood_metadata(user_message)
        mood_instr = mood_to_persona_instruction(mood)
        full_system = _join_prompts(system_prompt, mood_instr)
        history = history or []

        tiers = self._routing_order(clean_msg, prefer_large_context, prefer_local)

        for tier in tiers:
            try:
                async for token in self._dispatch(
                    tier, clean_msg, full_system, history, prefer_standby=prefer_local
                ):
                    yield token
                return
            except Exception as exc:
                logger.warning("[LLMRouter] %s failed: %s — next tier", tier, exc)

        logger.error("[LLMRouter] All tiers exhausted")
        yield (
            "Bhai, abhi koi bhi AI service respond nahi kar rahi. "
            "Check your connection and API keys."
        )

    async def chat(
        self,
        user_message: str,
        system_prompt: str = "",
        *,
        prefer_large_context: bool = False,
        prefer_local: bool = False,
        history: list[dict] | None = None,
    ) -> str:
        """Blocking convenience wrapper."""
        parts: list[str] = []
        async for t in self.stream_chat(
            user_message,
            system_prompt,
            prefer_large_context=prefer_large_context,
            prefer_local=prefer_local,
            history=history,
        ):
            parts.append(t)
        return "".join(parts)

    # ------------------------------------------------------------------
    # Routing order
    # ------------------------------------------------------------------

    def _routing_order(
        self, message: str, prefer_large: bool, prefer_local: bool = False
    ) -> list[LLMTier]:
        if prefer_local:
            # Skip cloud tiers entirely — local MLX only.
            return [LLMTier.LOCAL_MLX]
        large = prefer_large or len(message) > self._cfg.large_context_threshold
        if large:
            # Gemini first — 1M token context window, free tier
            return [LLMTier.GEMINI, LLMTier.GROQ, LLMTier.CEREBRAS, LLMTier.LOCAL_MLX]
        # Fast conversational: Groq LPU first, Cerebras as rate-limit fallback
        return [LLMTier.GROQ, LLMTier.CEREBRAS, LLMTier.GEMINI, LLMTier.LOCAL_MLX]

    async def _dispatch(
        self,
        tier: LLMTier,
        message: str,
        system: str,
        history: list[dict],
        *,
        prefer_standby: bool = False,
    ) -> AsyncIterator[str]:
        if tier == LLMTier.GROQ:
            async for t in self._stream_groq(message, system, history):
                yield t
        elif tier == LLMTier.CEREBRAS:
            async for t in self._stream_cerebras(message, system, history):
                yield t
        elif tier == LLMTier.GEMINI:
            async for t in self._stream_gemini(message, system, history):
                yield t
        elif tier == LLMTier.LOCAL_MLX:
            async for t in self._stream_local_mlx(
                message, system, history, prefer_standby=prefer_standby
            ):
                yield t

    # ------------------------------------------------------------------
    # Groq — Llama-3.3-70B, LPU ~800 tok/s
    # ------------------------------------------------------------------

    async def _stream_groq(
        self, message: str, system: str, history: list[dict] | None = None
    ) -> AsyncIterator[str]:
        if not self._cfg.groq_api_key:
            raise RuntimeError("HAKI_GROQ_API_KEY not set")
        try:
            from groq import AsyncGroq  # type: ignore[import]
        except ImportError:
            raise RuntimeError("pip install groq")

        client = AsyncGroq(api_key=self._cfg.groq_api_key)
        tokens: list[str] = []
        stream = await client.chat.completions.create(
            model=self._cfg.groq_model,
            messages=_messages(system, message, history),
            stream=True,
            max_tokens=2048,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                tokens.append(delta.content)

        for token in tokens:
            yield token

    # ------------------------------------------------------------------
    # Cerebras — Llama-4-Scout, 2,000+ TPS, rate-limit fallback for Groq
    # ------------------------------------------------------------------

    async def _stream_cerebras(
        self, message: str, system: str, history: list[dict] | None = None
    ) -> AsyncIterator[str]:
        """
        Stream via Cerebras Cloud SDK (OpenAI-compatible).
        Requires: pip install cerebras-cloud-sdk
        """
        if not self._cfg.cerebras_api_key:
            raise RuntimeError("HAKI_CEREBRAS_API_KEY not set")
        try:
            from cerebras.cloud.sdk import AsyncCerebras  # type: ignore[import]
        except ImportError:
            raise RuntimeError("pip install cerebras-cloud-sdk")

        client = AsyncCerebras(api_key=self._cfg.cerebras_api_key)
        tokens: list[str] = []
        stream = await client.chat.completions.create(
            model=self._cfg.cerebras_model,
            messages=_messages(system, message, history),
            stream=True,
            max_tokens=2048,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                tokens.append(delta.content)
        for token in tokens:
            yield token

    # ------------------------------------------------------------------
    # Gemini 2.5 Flash — 1M token context, free tier via AI Studio
    # New google-genai SDK (required for AQ.* authentication keys)
    # ------------------------------------------------------------------

    async def _stream_gemini(
        self, message: str, system: str, history: list[dict] | None = None
    ) -> AsyncIterator[str]:
        if not self._cfg.gemini_api_key:
            raise RuntimeError("HAKI_GEMINI_API_KEY not set")
        try:
            from google import genai  # type: ignore[import]
            from google.genai import types  # type: ignore[import]
        except ImportError:
            raise RuntimeError("pip install google-genai")
        import asyncio

        client = genai.Client(api_key=self._cfg.gemini_api_key)
        config = (
            types.GenerateContentConfig(system_instruction=system)
            if system
            else None
        )

        # Gemini's generate_content takes a single message string. Prepend a
        # compact text rendering of the prior turns so the model has running
        # conversation context.
        full_message = _render_history_text(history) + message

        def _sync() -> list[str]:
            chunks: list[str] = []
            for chunk in client.models.generate_content_stream(
                model=self._cfg.gemini_model,
                contents=full_message,
                config=config,
            ):
                if getattr(chunk, "text", None):
                    chunks.append(chunk.text)
            return chunks

        tokens = await asyncio.get_event_loop().run_in_executor(None, _sync)
        for t in tokens:
            yield t

    # ------------------------------------------------------------------
    # Local MLX — GPU-free, runs entirely on ANE + CPU
    # Primary:  xLAM-2-3b-fc-r-4bit  (1.85 GB, tool-calling specialist)
    # Standby:  Bonsai-8B-mlx-1bit    (1.28 GB, cold offline)
    # ------------------------------------------------------------------

    async def _stream_local_mlx(
        self,
        message: str,
        system: str,
        history: list[dict] | None = None,
        *,
        prefer_standby: bool = False,
    ) -> AsyncIterator[str]:
        """
        Try the local MLX models in order, falling back to the next model on
        failure.

        Parameters
        ----------
        prefer_standby : bool
            When True, try ``mlx_standby_model`` (Bonsai-8B) first, falling
            back to ``mlx_primary_model`` (xLAM) if the standby model fails
            to load. Defaults to the normal primary-first order.
        """
        order = (
            [self._cfg.mlx_standby_model, self._cfg.mlx_primary_model]
            if prefer_standby
            else [self._cfg.mlx_primary_model, self._cfg.mlx_standby_model]
        )
        for model_id in order:
            try:
                async for token in self._run_mlx_model(model_id, message, system, history):
                    yield token
                return
            except Exception as exc:
                logger.warning("[LLMRouter] MLX model %s failed: %s", model_id, exc)
        raise RuntimeError("Both local MLX models failed")

    async def _run_mlx_model(
        self, model_id: str, message: str, system: str, history: list[dict] | None = None
    ) -> AsyncIterator[str]:
        """
        Run an MLX model via mlx_lm.  Loads on first call; subsequent calls
        reuse the cached model in memory.

        Requires: pip install mlx-lm
        The model runs on ANE + CPU (metal=False) — zero GPU/Metal usage.
        """
        try:
            import mlx_lm  # type: ignore[import]
        except ImportError:
            raise RuntimeError("pip install mlx-lm")
        import asyncio

        history_text = _render_history_text(history)
        prompt = (
            f"<|system|>\n{system}\n{history_text}<|user|>\n{message}\n<|assistant|>\n"
            if system
            else f"{history_text}<|user|>\n{message}\n<|assistant|>\n"
        )

        def _generate() -> str:
            model, tokenizer = mlx_lm.load(model_id)
            return mlx_lm.generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=1024,
                verbose=False,
            )

        result = await asyncio.get_event_loop().run_in_executor(None, _generate)
        # mlx_lm.generate returns the full string; yield word-by-word for
        # streaming semantics so the TTS clause chunker can start early.
        for word in result.split(" "):
            yield word + " "


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _messages(
    system: str, user: str, history: list[dict] | None = None
) -> list[dict[str, str]]:
    """
    Build the OpenAI-compatible message list:
        system (if any) + all history items + current user message.

    Each history item is ``{"role": "user"|"assistant", "content": str}``.
    Only items with a valid role and non-empty content are spliced in.
    """
    msgs: list[dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    for item in history or []:
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": str(content)})
    msgs.append({"role": "user", "content": user})
    return msgs


def _render_history_text(history: list[dict] | None) -> str:
    """
    Render conversation history as compact plain text for tiers that take a
    single prompt string (Gemini, local MLX).

    Produces lines like::

        User: ...
        HAKI: ...

    Returns an empty string when there is no history (so callers can simply
    prepend the result to the current user message).
    """
    if not history:
        return ""
    lines: list[str] = []
    for item in history:
        role = item.get("role")
        content = item.get("content")
        if not content:
            continue
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"HAKI: {content}")
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _join_prompts(*parts: str) -> str:
    return "\n\n".join(p for p in parts if p)
