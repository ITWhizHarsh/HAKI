"""
Chunker — splits Note bodies into overlapping text chunks for embedding.

Chunks are produced by a simple word-token-based sliding window:
  - Target size:   ~200 tokens  (words/whitespace-separated tokens)
  - Overlap:        ~20 tokens
  - Empty / very short bodies that fit within a single chunk produce
    exactly one Chunk.

Only the body text is chunked (front matter is not part of a Chunk).

Design: Vault + RAG design (indexing).
Requirements: 7.3.
"""

from __future__ import annotations

from .models import Chunk

# Default chunking parameters
_DEFAULT_CHUNK_SIZE = 200    # approximate token count per chunk
_DEFAULT_OVERLAP = 20        # approximate token overlap between adjacent chunks


class Chunker:
    """
    Splits note body text into overlapping text chunks.

    Tokenisation is whitespace-based (split on whitespace), which is a
    cheap and language-agnostic approximation suitable for sentence-
    embedding models.

    Parameters
    ----------
    chunk_size:
        Target number of tokens per chunk (default 200).
    overlap:
        Number of tokens shared between consecutive chunks (default 20).
    """

    def __init__(
        self,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        overlap: int = _DEFAULT_OVERLAP,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, note_id: str, body: str) -> list[Chunk]:
        """
        Split *body* into overlapping chunks and return them as a list
        of :class:`~core.memory.models.Chunk` objects.

        Parameters
        ----------
        note_id:
            The ID of the parent Note (stored on each Chunk).
        body:
            The plain Markdown body text to chunk.  YAML front matter
            must already be stripped.

        Returns
        -------
        list[Chunk]
            One or more Chunks.  An empty body returns a single Chunk
            whose ``text`` is an empty string.
        """
        # Tokenise: split on whitespace (preserves content words)
        tokens = body.split()

        # Always produce at least one chunk (even for empty bodies)
        if not tokens:
            return [Chunk(note_id=note_id, chunk_index=0, text="")]

        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            text = " ".join(chunk_tokens)
            chunks.append(
                Chunk(note_id=note_id, chunk_index=chunk_index, text=text)
            )
            chunk_index += 1
            if end == len(tokens):
                break
            start += step

        return chunks
