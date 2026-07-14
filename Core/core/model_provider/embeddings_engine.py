"""
Embeddings Engine — IBM Granite ModernBERT 32k-token context window.

Upgrade from paraphrase-multilingual-MiniLM-L12-v2 (512-token cap) to
ibm-granite/granite-embedding-97m-multilingual-r2:

    Architecture : ModernBERT + Rotary Position Embeddings (RoPE)
    Context window: 32,768 tokens  (entire Markdown files in one shot)
    Size          : ~97 MB on disk
    Languages     : multilingual incl. Hindi / Hinglish code-mix
    Hardware      : CPU inference, no GPU required

ChromaDB runs in local persistent mode (SQLite/Parquet on SSD).

API fallbacks for massive batches:
    Gemini Embedding API  (free tier, text-embedding-004)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_CHROMA_PATH = Path.home() / ".haki" / "chroma_db"
_DEFAULT_MODEL       = "ibm-granite/granite-embedding-97m-multilingual-r2"

# Switch to API when batch exceeds this many documents
_LARGE_BATCH_THRESHOLD = 32


# ---------------------------------------------------------------------------
# ChromaDB client
# ---------------------------------------------------------------------------


def get_chroma_client(persist_directory: "Path | str | None" = None):
    """Return a persistent ChromaDB client backed by local SSD."""
    try:
        import chromadb  # type: ignore[import]
    except ImportError:
        raise RuntimeError("pip install chromadb")

    path = Path(persist_directory or _DEFAULT_CHROMA_PATH)
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    logger.info("[Embeddings] ChromaDB at %s", path)
    return client


# ---------------------------------------------------------------------------
# Embedding function (ChromaDB-compatible)
# ---------------------------------------------------------------------------


class MultilingualEmbeddingFunction:
    """
    Embedding function backed by IBM Granite ModernBERT (32k ctx, ~97 MB).

    Falls back to the Gemini Embedding API for batches that exceed
    _LARGE_BATCH_THRESHOLD documents (e.g. full codebase or massive PDFs).

    Parameters
    ----------
    model_name:
        HuggingFace model ID.  Default: ibm-granite/granite-embedding-97m-multilingual-r2
    gemini_api_key:
        Optional Gemini API key for large-batch fallback.
    large_batch_threshold:
        Number of documents above which the API fallback is preferred.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        gemini_api_key: str | None = None,
        large_batch_threshold: int = _LARGE_BATCH_THRESHOLD,
    ) -> None:
        self._model_name  = model_name
        self._gemini_key  = gemini_api_key
        self._threshold   = large_batch_threshold
        self._model       = None   # lazy-loaded
        self._model_ready = False  # True once the first embed completes

    # ChromaDB EmbeddingFunction interface
    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return self.embed(input)

    def name(self) -> str:
        """Return the model name (required by ChromaDB 1.5+)."""
        return self._model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        if len(texts) > self._threshold and self._gemini_key:
            try:
                return self._gemini_embed(texts)
            except Exception as exc:
                logger.warning("[Embeddings] Gemini API failed: %s — local", exc)
        return self._local_embed(texts)

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]

    # Langchain/ChromaDB explicit method support
    #
    # IMPORTANT: ChromaDB 1.5+ calls ``embed_query(input=<documents>)`` during a
    # ``query(query_texts=...)`` call, where ``input`` is the *list* of query
    # texts.  It expects a *list of embeddings* back (one vector per input),
    # exactly like ``__call__`` / ``embed_documents`` — NOT a single flattened
    # vector.  Returning a single vector makes Chroma misread the 384 floats as
    # 384 scalar query vectors and crash with "list index out of range", which
    # is why inserts worked but every semantic search failed.  Both methods now
    # normalise the input to a list of strings and return a list of vectors.
    def embed_query(  # noqa: A002
        self, input: "str | list[str] | None" = None, *, text: str = "", **_
    ) -> list[list[float]]:
        return self.embed(self._normalize_input(input, text))

    def embed_documents(  # noqa: A002
        self, input: "str | list[str] | None" = None, *, texts: "list[str] | None" = None, **_
    ) -> list[list[float]]:
        return self.embed(self._normalize_input(input, texts))

    @staticmethod
    def _normalize_input(primary, fallback) -> list[str]:
        """Coerce whatever Chroma/Langchain passes into a flat list of strings."""
        value = primary if primary is not None else fallback
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(v) for v in value]

    # ------------------------------------------------------------------
    # Local: IBM Granite ModernBERT (CPU, 32k context, ~97 MB)
    # ------------------------------------------------------------------

    def _local_embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed using sentence-transformers with the Granite model.

        Requires: pip install sentence-transformers
        The model is downloaded once and cached to ~/.cache/huggingface.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import]
            except ImportError:
                raise RuntimeError("pip install sentence-transformers")

            logger.info("[Embeddings] Loading %s …", self._model_name)
            self._model = SentenceTransformer(
                self._model_name,
                trust_remote_code=True,   # required for ModernBERT
            )
            logger.info("[Embeddings] Model loaded (ctx=%d)", self._context_length())

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            # Granite supports up to 32768 tokens — no truncation for normal docs
            batch_size=8,
        )
        return embeddings.tolist()

    def _context_length(self) -> int:
        try:
            return self._model.max_seq_length  # type: ignore[union-attr]
        except Exception:
            return 32_768

    # ------------------------------------------------------------------
    # API fallback: Gemini text-embedding-004
    # ------------------------------------------------------------------

    def _gemini_embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed via Google Gemini text-embedding-004 (free tier).

        Processes in parallel batches of 100 (Gemini batch limit).
        Requires: pip install google-generativeai
        """
        try:
            import google.generativeai as genai  # type: ignore[import]
        except ImportError:
            raise RuntimeError("pip install google-generativeai")

        genai.configure(api_key=self._gemini_key)
        results: list[list[float]] = []
        # Gemini batch limit: 100
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=batch,
                task_type="retrieval_document",
            )
            # Response is list of dicts with "values" key
            for item in response["embedding"]:
                results.append(item)
        return results


# ---------------------------------------------------------------------------
# EmbeddingsEngine — ModelProvider-compatible wrapper
# ---------------------------------------------------------------------------


class EmbeddingsEngine:
    """
    High-level embeddings engine.

    Implements the ModelProvider.invoke() contract used by the Indexer and
    HAKIBrain, and exposes ChromaDB collection management.

    Parameters
    ----------
    model_name:
        Embedding model ID.  Defaults to the Granite ModernBERT model.
    gemini_api_key:
        Optional Gemini API key for large-document batches.
    chroma_persist_dir:
        Where ChromaDB stores its data (defaults to ~/.haki/chroma_db).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        gemini_api_key: str | None = None,
        chroma_persist_dir: "Path | str | None" = None,
        # Legacy params kept for backward-compat (ignored)
        cohere_api_key: str | None = None,
    ) -> None:
        self._embed_fn = MultilingualEmbeddingFunction(
            model_name=model_name,
            gemini_api_key=gemini_api_key,
        )
        self._chroma_dir    = chroma_persist_dir
        self._chroma_client = None

    # ModelProvider interface
    def invoke(self, input: str, **_) -> list[float]:  # noqa: A002
        return self._embed_fn.embed_single(input)

    async def invoke_stream(self, input: str, **_):  # noqa: A002
        yield self.invoke(input)

    # ChromaDB helpers
    def get_or_create_collection(self, name: str, metadata: dict | None = None):
        if self._chroma_client is None:
            self._chroma_client = get_chroma_client(self._chroma_dir)
        return self._chroma_client.get_or_create_collection(
            name=name,
            embedding_function=self._embed_fn,
            metadata=metadata or {"hnsw:space": "cosine"},
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._embed_fn.embed(texts)
