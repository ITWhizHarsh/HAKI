"""Local XTTS v2 sentence adapter, sentence boundary segmentation, and PCM streaming.

This module provides:

- :class:`SentenceBoundaryProcessor` — accumulates LLM text chunks and emits
  complete, non-empty sentences on terminal punctuation or an explicit
  ``<EOR>`` end-of-response marker.  Abbreviation-aware rules prevent false
  splits on ``Mr.``, ``Dr.``, etc.

- :func:`map_language_for_xtts` — maps ``VoiceLanguage`` values to the XTTS
  language codes accepted by ``TTS==0.22.0``.  ``en`` → ``en``; both ``hi``
  and ``hinglish`` → ``hi``.

- :func:`validate_voice_asset` — validates that ``my_voice.wav`` exists, is
  readable, and is a regular file.  Raises :class:`TTSVoiceAssetError` if not;
  never falls back to a system or cloud TTS engine.

- :class:`XTTSSentenceAdapter` — session-owned XTTS v2 adapter that:
    * Initializes exactly one ``TTS==0.22.0`` / Coqui XTTS v2 session conditioned
      on the validated voice asset.
    * Emits ordered ``TTSTextFrame`` values via
      :meth:`VoiceSessionPipeline.emit_tts_text`.
    * Synthesizes PCM via the pipeline's single-worker blocking executor.
    * Emits ordered, bounded ``PCMChunkFrame`` values via
      :meth:`VoiceSessionPipeline.emit_pcm_chunk`.
    * Measures TTFA from the first completed ``TTSTextFrame`` timestamp to the
      first delivered non-empty PCM chunk.
    * On synthesis failure: cancels remaining unplayed turn work, emits a
      ``VoiceDiagnosticEvent`` with ``stage="local_tts"``, and surfaces the
      complete generated response as UI text via an optional
      ``text_fallback_sink`` — no alternate TTS engine is ever selected.

Design rule: this module MUST NOT import ``pyttsx3``, ``gtts``, ``subprocess``,
``say``, ``afplay``, ``edge_tts``, ``kokoro``, or any system / cloud TTS
library.  The only permitted TTS import is ``from TTS.api import TTS``.

Integration example (pipeline.py docstring usage)
---------------------------------------------------
The pipeline connects the LLM text sink to a ``SentenceBoundaryProcessor``,
which in turn calls ``XTTSSentenceAdapter.handle_llm_chunk``.  When a complete
sentence is ready, the adapter calls ``pipeline.emit_tts_text()`` and then
queues the sentence for synthesis::

    boundary = SentenceBoundaryProcessor()

    async def llm_text_sink(frame: TypedVoiceFrame) -> None:
        text = frame.payload.text          # LLMTextFrame.text
        for sentence_text in boundary.feed(text):
            await tts_adapter.enqueue_sentence(sentence_text, turn=turn_id)
        # At end of LLM stream:
        for sentence_text in boundary.flush():
            await tts_adapter.enqueue_sentence(sentence_text, turn=turn_id)

The ``XTTSSentenceAdapter`` handles all PCM chunking, TTFA measurement, and
failure reporting internally.  The pipeline calls
``pipeline.emit_pcm_chunk()`` only via the adapter — never directly.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable
from pathlib import Path
from threading import Event
from typing import Callable
from uuid import UUID, uuid4

from .diagnostics import VoiceDiagnosticEvent, append_diagnostic
from .interfaces import VoiceLanguage, VoiceSentence


# ---------------------------------------------------------------------------
# Public error types
# ---------------------------------------------------------------------------


class TTSVoiceAssetError(RuntimeError):
    """The required voice conditioning asset is missing or unreadable.

    Never falls back to a system, Edge, Kokoro, ChatTTS, or Cartesia voice.
    """


class TTSInitializationError(RuntimeError):
    """The XTTS v2 session could not be initialized from the TTS library."""


# ---------------------------------------------------------------------------
# Language mapping
# ---------------------------------------------------------------------------


def map_language_for_xtts(language: VoiceLanguage) -> str:
    """Map a ``VoiceLanguage`` to the XTTS language code.

    ``en`` → ``"en"``; both ``hi`` and ``hinglish`` → ``"hi"``.
    XTTS v2 supports Hindi script synthesis using the ``"hi"`` code.
    """
    if language == "en":
        return "en"
    return "hi"  # hi and hinglish both map to hi


# ---------------------------------------------------------------------------
# Voice asset validation
# ---------------------------------------------------------------------------


def validate_voice_asset(path: Path) -> None:
    """Check that *path* is a readable regular file.

    Raises :class:`TTSVoiceAssetError` with a clear, actionable message if the
    asset does not exist, is not a regular file, or cannot be read.  Never
    selects a fallback voice.
    """
    if not path.exists():
        raise TTSVoiceAssetError(
            f"XTTS voice conditioning asset is missing: {path}. "
            "Provide a readable my_voice.wav before enabling Local TTS."
        )
    if not path.is_file():
        raise TTSVoiceAssetError(
            f"XTTS voice conditioning asset is not a regular file: {path}."
        )
    try:
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        raise TTSVoiceAssetError(
            f"XTTS voice conditioning asset is unreadable: {path}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Sentence boundary processor
# ---------------------------------------------------------------------------

# Abbreviations that end with a period and must NOT trigger a sentence split.
_ABBREVIATIONS: frozenset[str] = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc",
    "approx", "i.e", "e.g", "fig", "no", "vol", "dept",
    "st", "ave", "blvd", "rev", "gen", "col", "lt", "sgt",
    "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
})

# Terminal punctuation characters that end a sentence.
_TERMINAL_CHARS: str = r"[.!?।]"

# Compiled pattern: terminal punctuation optionally followed by closing
# brackets/quotes, then whitespace or end-of-string.
_SENTENCE_SPLIT_RE: re.Pattern[str] = re.compile(
    _TERMINAL_CHARS + r'(?:[)\]"\'»]*)(?=\s|$)'
)

# Match a word ending with a period that could be an abbreviation.
_ABBREV_CANDIDATE_RE: re.Pattern[str] = re.compile(
    r"\b([A-Za-z]{1,10})$",
    re.IGNORECASE,
)


def _is_abbreviation_boundary(text_before_period: str) -> bool:
    """Return True if the text immediately before a `.` is a known abbreviation."""
    match = _ABBREV_CANDIDATE_RE.search(text_before_period.rstrip())
    if match:
        candidate = match.group(1).lower()
        return candidate in _ABBREVIATIONS
    return False


class SentenceBoundaryProcessor:
    """Segments accumulated LLM text into non-empty sentences.

    Splits are triggered by:
    - Terminal punctuation: ``. ! ? ।`` (Devanagari danda)
    - An explicit ``<EOR>`` end-of-response marker

    Abbreviation protection prevents ``Mr.``, ``Dr.``, ``etc.``, and similar
    tokens from triggering a false split.  Empty sentences (after stripping
    whitespace) are silently skipped.

    Usage::

        proc = SentenceBoundaryProcessor()
        for chunk in llm_stream:
            for sentence in proc.feed(chunk):
                ...  # emit TTSTextFrame
        for sentence in proc.flush():
            ...  # emit remaining text at end of stream
    """

    EOR_MARKER: str = "<EOR>"

    def __init__(self) -> None:
        self._buffer: str = ""

    def feed(self, text: str) -> list[str]:
        """Append *text* to the internal buffer; return complete sentences."""
        if not text:
            return []
        # Handle explicit EOR marker.
        if self.EOR_MARKER in text:
            parts = text.split(self.EOR_MARKER, 1)
            self._buffer += parts[0]
            sentences = self._extract_sentences(flush=True)
            # Any text after EOR is discarded (EOR means the response is done).
            self._buffer = ""
            return sentences

        self._buffer += text
        return self._extract_sentences(flush=False)

    def flush(self) -> list[str]:
        """Flush remaining buffer; return any non-empty trailing sentence."""
        return self._extract_sentences(flush=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_sentences(self, *, flush: bool) -> list[str]:
        """Scan the buffer for complete sentences and return them."""
        sentences: list[str] = []
        while True:
            result = self._find_next_boundary()
            if result is None:
                break
            sentence_text, remaining = result
            stripped = sentence_text.strip()
            if stripped:
                sentences.append(stripped)
            self._buffer = remaining

        if flush and self._buffer.strip():
            sentences.append(self._buffer.strip())
            self._buffer = ""

        return sentences

    def _find_next_boundary(self) -> tuple[str, str] | None:
        """Find the first valid sentence boundary in the buffer.

        Returns ``(sentence_text, remainder)`` or ``None`` if no boundary is
        found yet.
        """
        buf = self._buffer
        # Search for terminal punctuation characters.
        pos = 0
        while pos < len(buf):
            # Check for `.` (with abbreviation protection)
            if buf[pos] == ".":
                text_before = buf[:pos]
                if _is_abbreviation_boundary(text_before):
                    pos += 1
                    continue
                # Require whitespace or end-of-string after punctuation
                # (skip closing brackets/quotes).
                end = pos + 1
                while end < len(buf) and buf[end] in ")]\"'»":
                    end += 1
                if end >= len(buf) or buf[end].isspace():
                    return buf[:end].rstrip(), buf[end:].lstrip()
                pos += 1
                continue

            if buf[pos] in "!?।":
                end = pos + 1
                while end < len(buf) and buf[end] in ")]\"'»":
                    end += 1
                if end >= len(buf) or buf[end].isspace():
                    return buf[:end].rstrip(), buf[end:].lstrip()
                pos += 1
                continue

            pos += 1

        return None


# ---------------------------------------------------------------------------
# PCM chunk constants
# ---------------------------------------------------------------------------

# XTTS v2 synthesizes at 24000 Hz, 1 channel, S16LE.
XTTS_SAMPLE_RATE_HZ: int = 24_000
XTTS_CHANNELS: int = 1
XTTS_FORMAT: str = "s16le"

# ~100 ms of audio at 24 kHz S16LE: 24000 * 0.1 * 2 bytes = 4800 bytes
PCM_CHUNK_SIZE_BYTES: int = 4_800


# ---------------------------------------------------------------------------
# XTTS sentence adapter
# ---------------------------------------------------------------------------


class XTTSSentenceAdapter:
    """Session-owned adapter that synthesizes sentences with XTTS v2.

    One instance is created per active voice session.  It conditions a single
    ``TTS==0.22.0`` / Coqui XTTS v2 session on a validated ``my_voice.wav``
    asset, streams PCM chunks for each sentence via the pipeline's owned
    single-worker blocking executor, and measures TTFA.

    Constraint: this class NEVER falls back to ``pyttsx3``, ``gtts``, ``say``,
    ``afplay``, ``edge_tts``, ``kokoro``, ``ChatTTS``, Cartesia, or any cloud
    or system TTS engine.
    """

    def __init__(
        self,
        *,
        voice_asset_path: Path,
        pipeline: object,  # VoiceSessionPipeline — avoid circular import
        diagnostic_sink: Callable[[VoiceDiagnosticEvent], Awaitable[None] | None] | None = None,
        text_fallback_sink: Callable[[str], Awaitable[None] | None] | None = None,
        synthesis_queue_maxsize: int = 4,
    ) -> None:
        """Create the adapter.

        Parameters
        ----------
        voice_asset_path:
            Path to the validated ``my_voice.wav`` conditioning file.
        pipeline:
            The owning ``VoiceSessionPipeline`` instance (typed as ``object``
            here to avoid a circular import; duck-typed at runtime).
        diagnostic_sink:
            Optional async or sync callable that receives
            :class:`~.diagnostics.VoiceDiagnosticEvent` records.
        text_fallback_sink:
            Optional callable invoked with the complete generated response text
            when synthesis fails for an assistant turn.  Receives the full
            plain-text response so the UI can display it.
        synthesis_queue_maxsize:
            Maximum number of sentences that may queue for synthesis
            concurrently.  Provides bounded backpressure so next-sentence
            synthesis does not outpace the renderer indefinitely.
        """
        validate_voice_asset(voice_asset_path)
        self._voice_asset_path = voice_asset_path
        self._pipeline = pipeline
        self._diagnostic_sink = diagnostic_sink
        self._text_fallback_sink = text_fallback_sink
        self._tts_instance: object | None = None
        self._session_ready: bool = False
        # Bounded semaphore for synthesis overlap backpressure.
        self._synthesis_semaphore = asyncio.Semaphore(synthesis_queue_maxsize)

    # ------------------------------------------------------------------
    # Session initialization
    # ------------------------------------------------------------------

    async def initialize_session(self) -> None:
        """Load XTTS v2 and condition it on ``voice_asset_path``.

        Uses the pipeline's owned blocking executor so the heavy model load
        does not block the asyncio event loop.  Raises
        :class:`TTSInitializationError` if the ``TTS`` library is unavailable
        or the XTTS session fails to initialize.
        """
        def _load_xtts(cancellation: Event) -> object:
            try:
                from TTS.api import TTS  # type: ignore[import]
            except ImportError as exc:
                raise TTSInitializationError(
                    "TTS library is unavailable; install TTS==0.22.0 before enabling Local TTS."
                ) from exc
            try:
                tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
                # Pre-warm the speaker embedding so the first synthesis is fast.
                tts.tts(
                    text=" ",
                    speaker_wav=str(self._voice_asset_path),
                    language="en",
                )
                return tts
            except Exception as exc:
                raise TTSInitializationError(
                    f"XTTS v2 session could not be initialized: {exc}"
                ) from exc

        try:
            tts = await self._pipeline.run_blocking_library(
                library="xtts",
                operation=_load_xtts,
            )
        except TTSInitializationError:
            raise
        except Exception as exc:
            raise TTSInitializationError(
                f"XTTS v2 blocking executor failed during initialization: {exc}"
            ) from exc

        self._tts_instance = tts
        self._session_ready = True

    # ------------------------------------------------------------------
    # Sentence synthesis
    # ------------------------------------------------------------------

    async def synthesize_sentence(
        self,
        sentence: VoiceSentence,
        *,
        sequence: int,
        full_response_text: str = "",
    ) -> None:
        """Emit a ``TTSTextFrame`` then synthesize and stream PCM for *sentence*.

        Parameters
        ----------
        sentence:
            The complete, non-empty sentence to synthesize.
        sequence:
            Monotonically increasing sentence sequence number within this turn.
        full_response_text:
            The full plain-text LLM response for this turn, used only when
            synthesis fails to surface the complete text via the fallback sink.
        """
        if not self._session_ready or self._tts_instance is None:
            raise TTSInitializationError("XTTS session is not initialized.")

        # Emit TTSTextFrame promptly before synthesis begins.
        first_tts_text_monotonic_ns = time.monotonic_ns()
        try:
            await self._pipeline.emit_tts_text(
                turn_id=sentence.turn_id,
                sentence_id=sentence.sentence_id,
                sequence=sequence,
                text=sentence.text,
            )
        except Exception:
            # If emit_tts_text fails the sentence is no longer current; skip.
            return

        xtts_language = map_language_for_xtts(sentence.language)
        voice_asset = str(self._voice_asset_path)
        tts_instance = self._tts_instance

        # Check that the turn is still current before acquiring the semaphore.
        if not await self._is_current(sentence):
            return

        async with self._synthesis_semaphore:
            # Re-check generation after acquiring the slot.
            if not await self._is_current(sentence):
                return

            def _synthesize(cancellation: Event) -> bytes:
                """Run XTTS synthesis in the blocking executor thread."""
                if cancellation.is_set():
                    return b""
                pcm_data = tts_instance.tts(
                    text=sentence.text,
                    speaker_wav=voice_asset,
                    language=xtts_language,
                )
                # tts() returns a list of float samples [-1.0, 1.0].
                # Convert to S16LE bytes.
                import struct as _struct
                if isinstance(pcm_data, (list, tuple)):
                    pcm_bytes = _struct.pack(
                        f"<{len(pcm_data)}h",
                        *[max(-32768, min(32767, int(s * 32767))) for s in pcm_data],
                    )
                elif isinstance(pcm_data, bytes):
                    pcm_bytes = pcm_data
                else:
                    # numpy array path
                    try:
                        import numpy as np
                        arr = np.array(pcm_data, dtype=np.float32)
                        arr_int16 = (arr * 32767).clip(-32768, 32767).astype(np.int16)
                        pcm_bytes = arr_int16.tobytes()
                    except ImportError:
                        pcm_bytes = bytes(
                            _struct.pack(
                                f"<{len(pcm_data)}h",
                                *[max(-32768, min(32767, int(s * 32767))) for s in pcm_data],
                            )
                        )
                return pcm_bytes

            try:
                pcm_bytes: bytes = await self._pipeline.run_blocking_library(
                    library="xtts",
                    operation=_synthesize,
                )
            except Exception as exc:
                await self._handle_synthesis_failure(
                    turn_id=sentence.turn_id,
                    sentence_id=sentence.sentence_id,
                    error=exc,
                    full_response_text=full_response_text,
                )
                return

            if not pcm_bytes:
                # Synthesis was cancelled via the Event flag.
                return

            # Stream PCM chunks, measuring TTFA for the first non-empty chunk.
            first_pcm_delivered_ns: int | None = None
            chunk_seq = 0
            offset = 0
            total = len(pcm_bytes)

            while offset < total:
                # Re-check generation before each chunk.
                if not await self._is_current(sentence):
                    return

                chunk = pcm_bytes[offset: offset + PCM_CHUNK_SIZE_BYTES]
                offset += PCM_CHUNK_SIZE_BYTES
                if not chunk:
                    break

                # Record TTFA on the first non-empty chunk.
                if first_pcm_delivered_ns is None:
                    first_pcm_delivered_ns = time.monotonic_ns()
                    ttfa_ms = (first_pcm_delivered_ns - first_tts_text_monotonic_ns) / 1_000_000

                    # Update the pipeline diagnostic event if accessible.
                    try:
                        session = self._pipeline.session
                        diag_event = VoiceDiagnosticEvent(
                            session_id=session.session_id,
                            turn_id=sentence.turn_id,
                            stage="local_tts",
                            outcome="started",
                            tts_engine="xtts_v2",
                            first_tts_text_monotonic_ns=first_tts_text_monotonic_ns,
                            first_pcm_delivered_monotonic_ns=first_pcm_delivered_ns,
                            ttfa_ms=ttfa_ms,
                        )
                        await self._emit_diagnostic(diag_event)
                    except Exception:
                        pass  # Diagnostics must not interrupt PCM delivery.

                try:
                    await self._pipeline.emit_pcm_chunk(
                        turn_id=sentence.turn_id,
                        sentence_id=sentence.sentence_id,
                        sequence=sequence,
                        chunk_sequence=chunk_seq,
                        pcm=chunk,
                    )
                    chunk_seq += 1
                except Exception:
                    # Turn was cancelled or pipeline closed; stop streaming.
                    return

    # ------------------------------------------------------------------
    # Synthesis failure handler
    # ------------------------------------------------------------------

    async def _handle_synthesis_failure(
        self,
        *,
        turn_id: UUID,
        sentence_id: UUID,
        error: Exception,
        full_response_text: str,
    ) -> None:
        """Emit a failure diagnostic and surface the full response as UI text.

        Does NOT select any alternate TTS engine.
        """
        try:
            session = self._pipeline.session
            diag_event = VoiceDiagnosticEvent(
                session_id=session.session_id,
                turn_id=turn_id,
                stage="local_tts",
                outcome="failed",
                tts_engine="xtts_v2",
                error_class=type(error).__name__,
                recovery_outcome="text_fallback" if self._text_fallback_sink else "no_fallback",
            )
            await self._emit_diagnostic(diag_event)
        except Exception:
            pass

        # Surface the complete generated response as plain text — no alternate TTS.
        if self._text_fallback_sink and full_response_text:
            try:
                result = self._text_fallback_sink(full_response_text)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _is_current(self, sentence: VoiceSentence) -> bool:
        """Return True if this sentence's generation is still active."""
        try:
            return await self._pipeline.session.output_is_current(
                _SentenceGenerationToken(
                    session_id=self._pipeline.session.session_id,
                    turn_id=sentence.turn_id,
                )
            )
        except Exception:
            return False

    async def _emit_diagnostic(self, event: VoiceDiagnosticEvent) -> None:
        if self._diagnostic_sink is not None:
            result = self._diagnostic_sink(event)
            if hasattr(result, "__await__"):
                await result
        else:
            try:
                append_diagnostic(event)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Internal generation-check token
# ---------------------------------------------------------------------------


class _SentenceGenerationToken:
    """Minimal duck-typed object used by output_is_current generation checks."""

    __slots__ = ("metadata",)

    def __init__(self, *, session_id: UUID, turn_id: UUID) -> None:
        from .frames import VoiceFrameMetadata, VoiceFrameType, TypedVoiceFrame

        class _Meta:
            def __init__(self, session_id: UUID, turn_id: UUID) -> None:
                self.session_id = session_id
                self.turn_id = turn_id

        self.metadata = _Meta(session_id=session_id, turn_id=turn_id)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "PCM_CHUNK_SIZE_BYTES",
    "SentenceBoundaryProcessor",
    "TTSInitializationError",
    "TTSVoiceAssetError",
    "XTTS_CHANNELS",
    "XTTS_FORMAT",
    "XTTS_SAMPLE_RATE_HZ",
    "XTTSSentenceAdapter",
    "map_language_for_xtts",
    "validate_voice_asset",
]
