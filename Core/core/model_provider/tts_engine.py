"""
TTS Engine — ANE-native mouth pipeline for HAKI (2026 architecture).

GPU-liberating strategy: primary local TTS runs entirely on the Apple Neural
Engine via a CoreML-compiled Kokoro-82M model, leaving Metal GPU free for the
local LLM.  Hinglish phonemization uses misaki for code-switched text.

Tier order:
    1. Kokoro-82M-CoreML   — local, ANE-native, <2 GB, misaki phonemizer
    2. ChatTTS             — local CPU fallback (conversational prosody)
    3. Cartesia Sonic 3.5  — cloud fallback via Bengaluru Blue Machines hub
                             (sub-15ms regional RTT, streaming PCM)

Env vars:
    HAKI_CARTESIA_API_KEY

Chunked streaming: LLM tokens are buffered at clause boundaries and the first
audio chunk is emitted as soon as the first clause is synthesised — TTS never
waits for the full LLM response.

AVSpeechSynthesizer / macOS `say` are completely removed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import unicodedata
from typing import AsyncIterator, Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Edge-TTS voice tuning (shared by this engine and the live IPC speaking path)
# ---------------------------------------------------------------------------
#
# These are user-tunable via environment variables so the voice can be
# adjusted without touching code:
#
#     HAKI_TTS_VOICE   — edge-tts voice id            (default hi-IN-MadhurNeural)
#     HAKI_TTS_RATE    — Speech Rate Adjustment (%)   (default +32%)
#     HAKI_TTS_PITCH   — Pitch Adjustment (Hz)        (default +17Hz)
#
# edge-tts requires an explicit sign on the rate/pitch deltas (e.g. "+32%",
# "-11%", "+17Hz").  ``_normalize_signed`` adds a leading "+" when the user
# supplies a bare number so values like "32%" / "17Hz" still work.

_DEFAULT_EDGE_VOICE = "hi-IN-MadhurNeural"
_DEFAULT_EDGE_RATE = "+32%"
_DEFAULT_EDGE_PITCH = "+17Hz"


def _normalize_signed(value: str, *, unit: str) -> str:
    """Ensure an edge-tts rate/pitch delta carries an explicit +/- sign.

    Accepts ``"32"``, ``"32%"``, ``"+32%"``, ``"-11%"`` etc. and returns a
    value edge-tts accepts (e.g. ``"+32%"``).  ``unit`` is ``"%"`` or ``"Hz"``.
    """
    v = (value or "").strip()
    if not v:
        return v
    # Strip a trailing unit so we can normalise the numeric core, re-add later.
    core = v
    if core.lower().endswith(unit.lower()):
        core = core[: -len(unit)]
    core = core.strip()
    if core and core[0] not in "+-":
        core = "+" + core
    return f"{core}{unit}"


def get_edge_voice_settings() -> tuple[str, str, str]:
    """Return ``(voice, rate, pitch)`` for edge-tts, honouring env overrides."""
    voice = os.environ.get("HAKI_TTS_VOICE", _DEFAULT_EDGE_VOICE).strip() or _DEFAULT_EDGE_VOICE
    rate = _normalize_signed(
        os.environ.get("HAKI_TTS_RATE", _DEFAULT_EDGE_RATE), unit="%"
    ) or _DEFAULT_EDGE_RATE
    pitch = _normalize_signed(
        os.environ.get("HAKI_TTS_PITCH", _DEFAULT_EDGE_PITCH), unit="Hz"
    ) or _DEFAULT_EDGE_PITCH
    return voice, rate, pitch

# ---------------------------------------------------------------------------
# Clause chunking
# ---------------------------------------------------------------------------

_MIN_WORDS    = 5
_BOUNDARY_RE  = re.compile(r"[.!?,;:।]")   # includes Hindi danda


def _is_devanagari(text: str) -> bool:
    return any(unicodedata.name(ch, "").startswith("DEVANAGARI") for ch in text)


def chunk_stream(tokens: Iterator[str]) -> Iterator[str]:
    buf: list[str] = []
    wc = 0
    for tok in tokens:
        buf.append(tok)
        wc += len(tok.split())
        if _BOUNDARY_RE.search(tok) and wc >= _MIN_WORDS:
            clause = "".join(buf).strip()
            if clause:
                yield clause
            buf.clear(); wc = 0
    if buf:
        clause = "".join(buf).strip()
        if clause:
            yield clause


async def async_chunk_stream(stream: AsyncIterator[str]) -> AsyncIterator[str]:
    buf: list[str] = []
    wc = 0
    async for tok in stream:
        buf.append(tok)
        wc += len(tok.split())
        if _BOUNDARY_RE.search(tok) and wc >= _MIN_WORDS:
            clause = "".join(buf).strip()
            if clause:
                yield clause
            buf.clear(); wc = 0
    if buf:
        clause = "".join(buf).strip()
        if clause:
            yield clause


# ---------------------------------------------------------------------------
# TTS Engine
# ---------------------------------------------------------------------------


class TTSEngine:
    """
    Streaming TTS: Kokoro CoreML (ANE) → ChatTTS (CPU) → Cartesia Sonic (cloud).

    Parameters
    ----------
    kokoro_coreml_dir:
        Path to the Kokoro-82M-CoreML model directory.
        None → falls back to the pure-Python kokoro package path.
    cartesia_api_key:
        Cartesia API key for cloud fallback.
    cartesia_voice_id:
        Cartesia voice ID to use.  Defaults to a neutral English/Hindi voice.
    """

    def __init__(
        self,
        kokoro_coreml_dir: str | None = None,
        cartesia_api_key: str | None = None,
        cartesia_voice_id: str = "694f9389-aac1-45b6-b726-9d9369183238",
    ) -> None:
        self._coreml_dir       = kokoro_coreml_dir
        self._cartesia_key     = cartesia_api_key
        self._cartesia_voice   = cartesia_voice_id

        # Lazy-loaded models
        self._kokoro_pipeline  = None
        self._chattts_model    = None

        # Availability flags (reset per session)
        self._kokoro_ok  = True
        self._chattts_ok = True

    # ------------------------------------------------------------------
    # Public: streaming interface for LLM token streams
    # ------------------------------------------------------------------

    async def speak_stream(
        self, token_stream: AsyncIterator[str]
    ) -> AsyncIterator[tuple[bytes, int]]:
        """
        Consume LLM tokens → clause-chunk → synthesise → yield (pcm, rate).

        The first audio chunk is yielded as soon as the first clause is ready.
        """
        async for clause in async_chunk_stream(token_stream):
            clause = clause.strip()
            if not clause:
                continue
            use_hindi = _is_devanagari(clause)
            try:
                audio, rate = await self._synth(clause, hindi=use_hindi)
                if audio:
                    yield audio, rate
            except Exception as exc:
                logger.warning("[TTS] clause synth failed %r: %s", clause[:30], exc)

    async def synthesise_text(
        self, text: str, use_hindi: bool | None = None
    ) -> tuple[bytes, int]:
        if use_hindi is None:
            use_hindi = _is_devanagari(text)
        return await self._synth(text, hindi=use_hindi)

    # ------------------------------------------------------------------
    # Cascading synthesis
    # ------------------------------------------------------------------

    async def _synth(self, text: str, *, hindi: bool) -> tuple[bytes, int]:
        # Tier 0 (primary): Microsoft Edge-TTS — hi-IN-MadhurNeural
        try:
            return await self._edge_tts(text)
        except Exception as exc:
            logger.warning("[TTS] Edge-TTS failed: %s — falling back to Kokoro", exc)

        if self._kokoro_ok:
            try:
                return await self._kokoro_coreml(text, hindi=hindi)
            except Exception as exc:
                logger.warning("[TTS] Kokoro CoreML failed: %s — ChatTTS", exc)
                self._kokoro_ok = False

        if self._chattts_ok:
            try:
                return await self._chattts(text)
            except Exception as exc:
                logger.warning("[TTS] ChatTTS failed: %s — falling back to Cartesia", exc)
                self._chattts_ok = False

        if self._cartesia_key:
            try:
                return await self._cartesia_sonic(text, hindi=hindi)
            except Exception as exc:
                logger.warning("[TTS] Cartesia Sonic failed: %s — falling back to macOS say", exc)

        # Fallback to macOS system TTS (always works)
        return await self._macos_say(text)

    # ------------------------------------------------------------------
    # Tier 0 (primary): Microsoft Edge-TTS → hi-IN-MadhurNeural
    # ------------------------------------------------------------------

    async def _edge_tts(self, text: str) -> tuple[bytes, int]:
        """
        Synthesise via Microsoft Edge-TTS (online neural voice).

        edge-tts streams MP3 audio.  We collect the audio chunks, write them to
        a temp .mp3 file, then decode MP3 → raw PCM int16 @ 24 kHz using the
        macOS-native `afconvert` tool (no extra Python audio dependency).

        Returns (pcm_bytes, 24000).
        """
        import edge_tts

        voice, rate, pitch = get_edge_voice_settings()
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch,
        )

        mp3_chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                mp3_chunks.append(chunk["data"])

        if not mp3_chunks:
            raise RuntimeError("Edge-TTS returned empty audio")

        mp3_bytes = b"".join(mp3_chunks)

        # Decode MP3 → PCM int16 @ 24 kHz via afconvert + stdlib wave.
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._decode_mp3_to_pcm, mp3_bytes)

    @staticmethod
    def _decode_mp3_to_pcm(mp3_bytes: bytes) -> tuple[bytes, int]:
        import subprocess
        import tempfile
        import wave
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            mp3_path = f.name
            f.write(mp3_bytes)

        wav_path = mp3_path[:-4] + ".wav"
        try:
            proc = subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@24000", mp3_path, wav_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if proc.returncode != 0 or not Path(wav_path).exists():
                raise RuntimeError(
                    f"afconvert MP3→PCM failed: {proc.stderr.decode(errors='replace')}"
                )
            with wave.open(wav_path, "rb") as wf:
                pcm = wf.readframes(wf.getnframes())
            return pcm, 24_000
        finally:
            for path in (mp3_path, wav_path):
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Tier 1: Kokoro-82M-CoreML → Apple Neural Engine
    # ------------------------------------------------------------------

    async def _kokoro_coreml(self, text: str, *, hindi: bool) -> tuple[bytes, int]:
        """
        Synthesise via Kokoro-82M running on CoreML / ANE.

        Execution path (tried in order):
        A) kokoro-onnx with CoreML EP (requires pip install kokoro-onnx onnxruntime-extensions)
           — Loads the ONNX model compiled to CoreML, routes to ANE.
        B) kokoro Python package (pip install kokoro)
           — Pure-Python, CPU execution.  No ANE but keeps GPU free.

        Voice routing:
            Hindi / Devanagari → hf_alpha
            English / Hinglish → af_heart (with misaki Hinglish phonemizer)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._kokoro_sync, text, hindi)

    def _kokoro_sync(self, text: str, hindi: bool) -> tuple[bytes, int]:
        import numpy as np

        voice = "hf_alpha" if hindi else "af_heart"

        # Path A: kokoro-onnx with CoreML execution provider
        try:
            from kokoro_onnx import Kokoro  # type: ignore[import]

            if self._kokoro_pipeline is None:
                model_path = (
                    f"{self._coreml_dir}/kokoro-v0_19.onnx"
                    if self._coreml_dir
                    else "kokoro-v0_19.onnx"          # auto-downloaded
                )
                voices_path = (
                    f"{self._coreml_dir}/voices.bin"
                    if self._coreml_dir
                    else "voices.bin"
                )
                self._kokoro_pipeline = Kokoro(model_path, voices_path)

            pipeline = self._kokoro_pipeline

            # Apply misaki phonemization for Hinglish code-switching
            phonemized = _misaki_phonemize(text, hindi=hindi)

            samples, sample_rate = pipeline.create(
                phonemized,
                voice=voice,
                speed=1.0,
                lang="hi" if hindi else "en-us",
            )
            pcm = (np.array(samples) * 32767).astype(np.int16).tobytes()
            return pcm, sample_rate

        except ImportError:
            pass  # fall through to kokoro Python package

        # Path B: pure-Python kokoro package
        from kokoro import KPipeline  # type: ignore[import]

        if self._kokoro_pipeline is None:
            self._kokoro_pipeline = KPipeline(lang_code="a")

        pipeline = self._kokoro_pipeline
        phonemized = _misaki_phonemize(text, hindi=hindi)

        if hindi:
            try:
                hindi_pipeline = KPipeline(lang_code="h")
                parts = [c.audio for c in hindi_pipeline(phonemized, voice=voice, speed=1.0)]
            except Exception:
                parts = [c.audio for c in pipeline(phonemized, voice=voice, speed=1.0)]
        else:
            parts = [c.audio for c in pipeline(phonemized, voice=voice, speed=1.0)]

        if not parts:
            raise RuntimeError("Kokoro returned empty audio")

        combined = np.concatenate(parts)
        pcm = (combined * 32767).astype(np.int16).tobytes()
        return pcm, 24_000

    # ------------------------------------------------------------------
    # Tier 2: ChatTTS — CPU, conversational prosody
    # ------------------------------------------------------------------

    async def _chattts(self, text: str) -> tuple[bytes, int]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._chattts_sync, text)

    def _chattts_sync(self, text: str) -> tuple[bytes, int]:
        try:
            import ChatTTS, torch, numpy as np  # type: ignore[import]
        except ImportError:
            raise RuntimeError("pip install chattts torch torchaudio")

        if self._chattts_model is None:
            chat = ChatTTS.Chat()
            chat.load(compile=False)
            self._chattts_model = chat

        chat = self._chattts_model
        spk = chat.sample_random_speaker()
        params = ChatTTS.Chat.InferCodeParams(spk_emb=spk, temperature=0.3, top_P=0.7, top_K=20)
        wavs = chat.infer([text], params_infer_code=params)
        if not wavs or wavs[0] is None:
            raise RuntimeError("ChatTTS empty output")

        pcm = (np.array(wavs[0]) * 32767).astype(np.int16).tobytes()
        return pcm, 24_000

    # ------------------------------------------------------------------
    # Tier 3: Cartesia Sonic 3.5 — Bengaluru Blue Machines (<15ms RTT)
    # ------------------------------------------------------------------

    async def _cartesia_sonic(self, text: str, *, hindi: bool) -> tuple[bytes, int]:
        """
        Stream PCM from Cartesia Sonic 3.5 via their WebSocket API.

        Targets the Bengaluru data hub for sub-15ms regional round-trips.
        Requires: pip install cartesia
        """
        if not self._cartesia_key:
            raise RuntimeError("HAKI_CARTESIA_API_KEY not set")

        try:
            from cartesia import AsyncCartesia  # type: ignore[import]
        except ImportError:
            raise RuntimeError("pip install cartesia")

        client = AsyncCartesia(api_key=self._cartesia_key)

        # Cartesia supports streaming PCM output
        pcm_chunks: list[bytes] = []
        ws = await client.tts.websocket()
        try:
            generator = await ws.send(
                model_id="sonic-2",                         # Cartesia model ID (sonic-multilingual was sunsetted)
                transcript=text,
                voice_id=self._cartesia_voice, # voice_id kwarg
                output_format={
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": 22050,
                },
                language="hi" if hindi else "en",
                _experimental_voice_controls={"speed": "normal", "emotion": []},
            )
            async for chunk in generator:
                if "audio" in chunk:
                    pcm_chunks.append(chunk["audio"])
        finally:
            await ws.close()
            await client.close()

        return b"".join(pcm_chunks), 22_050

    async def _macos_say(self, text: str) -> tuple[bytes, int]:
        """
        Fallback using macOS built-in 'say' command + afconvert.
        Always works, no dependencies needed.
        """
        import asyncio
        import tempfile
        from pathlib import Path
        
        # Generate audio files
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
            aiff_path = f.name
        
        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
            raw_path = f.name
        
        try:
            # Use macOS 'say' command to generate AIFF audio
            proc = await asyncio.create_subprocess_exec(
                "say", "-o", aiff_path, text,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0 or not Path(aiff_path).exists():
                logger.error(f"say command failed: {stderr.decode()}")
                return b"", 22050
            
            # Convert AIFF to raw PCM using afconvert (built into macOS)
            proc = await asyncio.create_subprocess_exec(
                "afconvert", "-f", "WAVE", "-d", "LEI16@22050", 
                aiff_path, raw_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            
            # Read the PCM data using wave module
            if Path(raw_path).exists():
                import wave
                with wave.open(raw_path, 'rb') as wf:
                    audio_data = wf.readframes(wf.getnframes())
                logger.info(f"[TTS] macOS say generated {len(audio_data)} bytes")
                return audio_data, 22050
            
            logger.error("[TTS] afconvert failed to create output file")
            return b"", 22050
        except Exception as exc:
            logger.exception(f"[TTS] macOS say fallback failed: {exc}")
            return b"", 22050
        finally:
            # Clean up temp files
            for path in [aiff_path, raw_path]:
                try:
                    Path(path).unlink(missing_ok=True)
                except:
                    pass


# ---------------------------------------------------------------------------
# misaki phonemizer — Hinglish code-switching support
# ---------------------------------------------------------------------------


def _misaki_phonemize(text: str, *, hindi: bool) -> str:
    """
    Apply misaki phonemization for Hinglish code-switched text.

    misaki handles mixed Hindi/English scripts and produces Kokoro-compatible
    phoneme strings.  Falls back to the raw text if misaki is not installed.

    Requires: pip install misaki[en,ja]  (Hindi support via g2p backend)
    """
    try:
        from misaki import en, hi  # type: ignore[import]
        if hindi or _is_devanagari(text):
            phonemes, _ = hi.phonemize(text)
        else:
            phonemes, _ = en.phonemize(text, british=False)
        return phonemes
    except Exception:
        # misaki not installed or phonemization failed — use raw text
        return text
