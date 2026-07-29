"""
STT Engine — Sensory pipeline targeting Apple Neural Engine (ANE).

GPU-liberating strategy: both local STT models run on .cpuAndNeuralEngine,
leaving Metal GPU completely free for the local LLM.

Cloud tier:
    Deepgram Flux  — model-integrated end-of-turn detection, minimal lag
    (hot-swap to local automatically when credits deplete)

Local ANE tier:
    WhisperKit Tiny CoreML  — .cpuAndNeuralEngine, zero GPU touch
    (installed model path supplied; Python calls the compiled CoreML artefact
     via coremltools / subprocess bridge to the WhisperKit Swift binary)

SenseVoice-Small remains as a CPU-only third tier for Hinglish + emotion.

Env vars:
    HAKI_DEEPGRAM_API_KEY
"""

from __future__ import annotations

import asyncio
import io
import logging
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import AsyncIterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class TranscriptResult:
    text: str
    is_final: bool
    emotion: str | None = None    # "happy" | "angry" | "sad" | None
    language: str | None = None
    confidence: float = 1.0


class STTEngineStatus(str, Enum):
    AVAILABLE   = "available"
    UNAVAILABLE = "unavailable"   # transient — retried after 60 s
    DEPLETED    = "depleted"      # permanent (credits gone)


# ---------------------------------------------------------------------------
# STT Engine
# ---------------------------------------------------------------------------


class STTEngine:
    """
    Cascading STT: Groq Whisper → Deepgram Flux → WhisperKit ANE → SenseVoice CPU.

    Parameters
    ----------
    groq_api_key:
        Groq API key (same key used by the LLM).  None skips the Groq tier.
        When present, Groq `whisper-large-v3-turbo` is the PRIMARY engine.
    deepgram_api_key:
        Deepgram API key.  None skips the cloud tier.
    whisperkit_model_dir:
        Path to a downloaded WhisperKit Tiny CoreML artefact directory.
        If None, the Python-native mlx-whisper fallback is used instead.
    sensevoice_model_path:
        Path to a local SenseVoice-Small model.  None → auto-download.
    """

    def __init__(
        self,
        groq_api_key: str | None = None,
        deepgram_api_key: str | None = None,
        whisperkit_model_dir: str | None = None,
        sensevoice_model_path: str | None = None,
    ) -> None:
        self._groq_key           = groq_api_key
        self._deepgram_key       = deepgram_api_key
        self._whisperkit_dir     = whisperkit_model_dir
        self._sensevoice_path    = sensevoice_model_path
        self._deepgram_status    = STTEngineStatus.AVAILABLE
        self._sensevoice_model   = None   # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def transcribe_stream(
        self, audio_bytes: bytes, sample_rate: int = 16_000
    ) -> AsyncIterator[TranscriptResult]:
        """Yield TranscriptResult objects; final one has is_final=True."""

        # ── Tier 1: Groq whisper-large-v3-turbo (cloud, PRIMARY) ──────
        if self._groq_key:
            try:
                result = await self._groq_whisper(audio_bytes, sample_rate)
                yield result
                return
            except Exception as exc:
                logger.warning("[STT] Groq Whisper failed (%s) — falling back", exc)

        # ── Tier 2: Deepgram Flux (cloud, streaming WebSocket) ────────
        if self._deepgram_key and self._deepgram_status == STTEngineStatus.AVAILABLE:
            try:
                async for r in self._deepgram_flux(audio_bytes, sample_rate):
                    yield r
                return
            except _DeepgramDepleted:
                logger.warning("[STT] Deepgram credits depleted — locking to ANE local")
                self._deepgram_status = STTEngineStatus.DEPLETED
            except Exception as exc:
                logger.warning("[STT] Deepgram failed (%s) — falling back", exc)
                self._deepgram_status = STTEngineStatus.UNAVAILABLE
                asyncio.get_event_loop().call_later(60, self._reset_deepgram)

        # ── Tier 3: WhisperKit Tiny CoreML (.cpuAndNeuralEngine) ──────
        try:
            result = await self._whisperkit_ane(audio_bytes, sample_rate)
            yield result
            return
        except Exception as exc:
            logger.warning("[STT] WhisperKit ANE failed (%s) — SenseVoice", exc)

        # ── Tier 4: SenseVoice-Small (CPU, Hinglish + emotion) ────────
        try:
            result = await self._sensevoice(audio_bytes, sample_rate)
            yield result
            return
        except Exception as exc:
            logger.error("[STT] All STT tiers failed: %s", exc)
            yield TranscriptResult(text="", is_final=True, confidence=0.0)

    async def transcribe(
        self, audio_bytes: bytes, sample_rate: int = 16_000
    ) -> TranscriptResult:
        """Non-streaming convenience wrapper."""
        final: TranscriptResult | None = None
        async for r in self.transcribe_stream(audio_bytes, sample_rate):
            final = r
        return final or TranscriptResult(text="", is_final=True)

    def mark_deepgram_depleted(self) -> None:
        self._deepgram_status = STTEngineStatus.DEPLETED

    def restore_deepgram(self) -> None:
        self._deepgram_status = STTEngineStatus.AVAILABLE

    # ------------------------------------------------------------------
    # Tier 1: Groq whisper-large-v3-turbo (cloud, PRIMARY)
    # ------------------------------------------------------------------

    @staticmethod
    def _pcm_to_wav(audio_bytes: bytes, sample_rate: int) -> bytes:
        """
        Wrap raw PCM signed 16-bit little-endian mono audio into an in-memory
        WAV container.  Groq's transcription endpoint needs a real audio file,
        not bare PCM samples.
        """
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)          # mono
            wav.setsampwidth(2)          # 16-bit
            wav.setframerate(sample_rate)
            wav.writeframes(audio_bytes)
        return buf.getvalue()

    async def _groq_whisper(
        self, audio_bytes: bytes, sample_rate: int
    ) -> TranscriptResult:
        """
        Transcribe buffered audio via Groq's `whisper-large-v3-turbo`.

        Understands English, Hindi, and Hinglish.  `language="en"` is passed
        intentionally so Hindi/Hinglish speech is transcribed into Latin
        (romanized) characters rather than Devanagari.

        The Groq SDK call is synchronous, so it runs inside an executor to
        avoid blocking the event loop (same pattern as the Deepgram tier).
        """
        try:
            from groq import Groq  # type: ignore[import]
        except ImportError:
            raise RuntimeError("pip install groq")

        client = Groq(api_key=self._groq_key)
        wav_bytes = self._pcm_to_wav(audio_bytes, sample_rate)

        loop = asyncio.get_event_loop()

        def _transcribe_sync() -> str:
            response = client.audio.transcriptions.create(
                file=("audio.wav", wav_bytes),
                model="whisper-large-v3-turbo",
                language="en",          # romanize Hindi/Hinglish into Latin script
                response_format="text",
            )
            # response_format="text" returns a plain string; be defensive in
            # case the SDK returns an object with a `.text` attribute.
            if isinstance(response, str):
                return response.strip()
            return getattr(response, "text", "").strip()

        text = await loop.run_in_executor(None, _transcribe_sync)

        # Filter out empty/whitespace transcripts
        if not text or not text.strip():
            return TranscriptResult(text="", is_final=True, confidence=0.0)

        return TranscriptResult(
            text=text.strip(),
            is_final=True,
            language="en",
            confidence=1.0,
        )

    # ------------------------------------------------------------------
    # Tier 2: Deepgram (prerecorded batch — most reliable for buffered audio)
    # ------------------------------------------------------------------

    async def _deepgram_flux(
        self, audio_bytes: bytes, sample_rate: int
    ) -> AsyncIterator[TranscriptResult]:
        """
        Transcribe buffered audio via Deepgram's prerecorded API.

        Using prerecorded (batch) instead of live-streaming because we
        already have the full audio buffer before calling Deepgram.
        This avoids the live-stream race condition where CloseStream
        arrives before the final transcript, causing empty results.
        """
        try:
            from deepgram import DeepgramClient, PrerecordedOptions  # type: ignore[import]
        except ImportError:
            raise RuntimeError("pip install deepgram-sdk")

        client = DeepgramClient(self._deepgram_key)

        # Wrap sync SDK call in executor so it doesn't block the event loop
        loop = asyncio.get_event_loop()

        def _transcribe_sync() -> str:
            response = client.listen.prerecorded.v("1").transcribe_file(
                {"buffer": audio_bytes, "mimetype": "audio/raw"},
                PrerecordedOptions(
                    model="nova-2",
                    language="multi",
                    smart_format=True,
                    punctuate=True,
                    channels=1,
                    sample_rate=sample_rate,
                    encoding="linear16",
                ),
            )
            try:
                return (
                    response.results.channels[0]
                    .alternatives[0]
                    .transcript.strip()
                )
            except (IndexError, AttributeError):
                return ""

        text = await loop.run_in_executor(None, _transcribe_sync)

        if text:
            yield TranscriptResult(text=text, is_final=True, confidence=1.0)

    # ------------------------------------------------------------------
    # Tier 2: WhisperKit Tiny CoreML — .cpuAndNeuralEngine, zero GPU
    # ------------------------------------------------------------------

    async def _whisperkit_ane(
        self, audio_bytes: bytes, sample_rate: int
    ) -> TranscriptResult:
        """
        Transcribe via WhisperKit's CoreML artefact, targeting ANE exclusively.

        Two execution paths, tried in order:
        A) whisperkittools Python package (pip install whisperkittools)
           — loads the CoreML model directly via coremltools, routes to ANE.
        B) mlx-whisper fallback (if whisperkittools not installed)
           — runs whisper-tiny on MLX; not strictly ANE-only but still
             leaves the Metal GPU untouched.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._whisperkit_sync, audio_bytes, sample_rate
        )

    def _whisperkit_sync(self, audio_bytes: bytes, sample_rate: int) -> TranscriptResult:
        import numpy as np
        audio_np = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )

        # Path A: whisperkittools (CoreML → ANE)
        try:
            import whisperkittools as wkt  # type: ignore[import]

            model_dir = self._whisperkit_dir or "argmaxinc/whisperkit-coreml"
            pipeline = wkt.WhisperPipeline.from_pretrained(
                model_dir,
                # Force .cpuAndNeuralEngine — leaves Metal GPU completely free
                compute_units="cpuAndNeuralEngine",
            )
            result = pipeline.transcribe(audio_np)
            text = result.get("text", "").strip()
            
            # Filter known Whisper silence hallucinations
            hallucinations = [
                "a few minutes later, i'll go to the next room.",
                "i'll go to the next room.",
                "thank you for watching.",
                "thank you.",
                "you",
            ]
            if text.lower().strip('. ') in hallucinations or text.lower().count("you") > 2:
                text = ""

            return TranscriptResult(
                text=text,
                is_final=True,
                language=result.get("language"),
            )
        except ImportError:
            pass  # fall through to mlx-whisper
        except Exception as exc:
            logger.warning("[STT] whisperkittools failed: %s — mlx-whisper", exc)

        # Path B: mlx-whisper (ANE-adjacent; no Metal GPU)
        try:
            import mlx_whisper  # type: ignore[import]

            model_id = (
                "mlx-community/whisper-tiny-mlx"
                if self._whisperkit_dir is None
                else self._whisperkit_dir
            )
            result = mlx_whisper.transcribe(
                audio_np,
                path_or_hf_repo=model_id,
                language=None,   # auto-detect Hindi/English mix
                task="transcribe",
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
            )
            text = result.get("text", "").strip()
            
            # Filter known Whisper silence hallucinations
            hallucinations = [
                "a few minutes later, i'll go to the next room.",
                "i'll go to the next room.",
                "thank you for watching.",
                "thank you.",
                "you",
            ]
            if text.lower().strip('. ') in hallucinations or text.lower().count("you") > 2:
                text = ""

            return TranscriptResult(
                text=text,
                is_final=True,
                language=result.get("language"),
            )
        except ImportError:
            raise RuntimeError(
                "Neither whisperkittools nor mlx-whisper is installed. "
                "Run: pip install whisperkittools  OR  pip install mlx-whisper"
            )

    # ------------------------------------------------------------------
    # Tier 3: SenseVoice-Small — CPU, Hinglish-native, emotion detection
    # ------------------------------------------------------------------

    async def _sensevoice(
        self, audio_bytes: bytes, sample_rate: int
    ) -> TranscriptResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._sensevoice_sync, audio_bytes, sample_rate
        )

    def _sensevoice_sync(self, audio_bytes: bytes, sample_rate: int) -> TranscriptResult:
        try:
            from funasr import AutoModel  # type: ignore[import]
            import numpy as np
        except ImportError:
            raise RuntimeError("pip install funasr torch torchaudio")

        if self._sensevoice_model is None:
            self._sensevoice_model = AutoModel(
                model=self._sensevoice_path or "iic/SenseVoiceSmall",
                trust_remote_code=True,
                device="cpu",
            )

        audio_np = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        out = self._sensevoice_model.generate(
            input=audio_np, cache={}, language="auto",
            use_itn=True, batch_size_s=60, merge_vad=True,
        )
        if not out:
            return TranscriptResult(text="", is_final=True)

        raw = out[0].get("text", "").strip()
        emotion: str | None = None
        text = raw
        for tag, label in [
            ("<|HAPPY|>", "happy"), ("<|ANGRY|>", "angry"),
            ("<|SAD|>", "sad"), ("<|NEUTRAL|>", None), ("<|SURPRISED|>", None),
        ]:
            if raw.startswith(tag):
                emotion = label
                text = raw[len(tag):].strip()
                break

        return TranscriptResult(text=text, is_final=True, emotion=emotion)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_deepgram(self) -> None:
        if self._deepgram_status == STTEngineStatus.UNAVAILABLE:
            self._deepgram_status = STTEngineStatus.AVAILABLE


class _DeepgramDepleted(Exception):
    pass
