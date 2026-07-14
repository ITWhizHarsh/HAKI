"""
Model Provider sub-package.

Owns the ModelProvider registry, CapabilityConfig, per-capability
local / API backend resolution, disclosure gating, and failure handling.
Also owns the real LLM router, STT engine, TTS engine, and embeddings engine.

Design reference: Model Provider Abstraction.
Requirements: 20.
"""

from .model_provider import (
    # Enumerations & config
    Capability,
    ProcessingMode,
    ModelMode,          # backward-compatible alias for ProcessingMode
    CapabilityConfig,
    # Registry
    ModelProviderRegistry,
    # ABC
    ModelProvider,
    # Stub (testing helper)
    StubModelProvider,
    # Custom exceptions (Req 20.5, 20.6, 20.7)
    DisclosureRequiredError,
    ApiUnavailableError,
    LocalModelLoadError,
    # Disclosure tracking (Req 20.5)
    DisclosureTracker,
    # Concrete backends (Task 5.2)
    LocalBackend,
    ApiBackend,
    BackendRouter,
    # Factory
    create_default_registry_and_routers,
)
from .llm_router import LLMRouter, LLMRouterConfig, LLMTier, parse_mood_metadata
from .stt_engine import STTEngine, TranscriptResult, STTEngineStatus
from .tts_engine import TTSEngine, async_chunk_stream, chunk_stream
from .embeddings_engine import (
    EmbeddingsEngine,
    MultilingualEmbeddingFunction,
    get_chroma_client,
)

__all__ = [
    # Model provider abstraction
    "Capability",
    "ProcessingMode",
    "ModelMode",
    "CapabilityConfig",
    "ModelProviderRegistry",
    "ModelProvider",
    "StubModelProvider",
    "DisclosureRequiredError",
    "ApiUnavailableError",
    "LocalModelLoadError",
    "DisclosureTracker",
    "LocalBackend",
    "ApiBackend",
    "BackendRouter",
    "create_default_registry_and_routers",
    # LLM routing
    "LLMRouter",
    "LLMRouterConfig",
    "LLMTier",
    "parse_mood_metadata",
    # STT
    "STTEngine",
    "TranscriptResult",
    "STTEngineStatus",
    # TTS
    "TTSEngine",
    "async_chunk_stream",
    "chunk_stream",
    # Embeddings
    "EmbeddingsEngine",
    "MultilingualEmbeddingFunction",
    "get_chroma_client",
]
