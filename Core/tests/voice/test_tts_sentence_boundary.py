"""Unit tests for SentenceBoundaryProcessor, map_language_for_xtts,
validate_voice_asset, and XTTSSentenceAdapter.

Coverage targets:
- Sentence segmentation on `.`, `!`, `?`, `।` (Devanagari danda)
- Abbreviation protection (Mr., Dr., etc. must not trigger splits)
- Explicit ``<EOR>`` marker flush
- Empty buffer flush returns empty list
- Language mapping: en→en, hi→hi, hinglish→hi
- Missing / unreadable voice asset raises TTSVoiceAssetError
- Ordered incremental frame emission via a mock pipeline
- Backpressure: synthesis semaphore blocks when full
- Synthesis failure triggers text fallback, no alternate engine
"""

from __future__ import annotations

import asyncio
import stat
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from core.voice.tts import (
    SentenceBoundaryProcessor,
    TTSInitializationError,
    TTSVoiceAssetError,
    XTTSSentenceAdapter,
    map_language_for_xtts,
    validate_voice_asset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wav_file(tmp_path: Path, name: str = "my_voice.wav") -> Path:
    """Write a tiny but readable stub WAV file."""
    p = tmp_path / name
    p.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    return p


def _mock_pipeline(session_id=None, turn_id=None):
    """Return a minimal mock that quacks like VoiceSessionPipeline."""
    sid = session_id or uuid4()
    tid = turn_id or uuid4()
    pipeline = MagicMock()
    pipeline.session = MagicMock()
    pipeline.session.session_id = sid
    pipeline.emit_tts_text = AsyncMock(return_value=MagicMock())
    pipeline.emit_pcm_chunk = AsyncMock(return_value=MagicMock())

    # output_is_current: always True by default
    async def _is_current(token):
        return True

    pipeline.session.output_is_current = _is_current
    return pipeline


# ---------------------------------------------------------------------------
# Language mapping
# ---------------------------------------------------------------------------


class TestMapLanguageForXTTS:
    def test_en_maps_to_en(self):
        assert map_language_for_xtts("en") == "en"

    def test_hi_maps_to_hi(self):
        assert map_language_for_xtts("hi") == "hi"

    def test_hinglish_maps_to_hi(self):
        assert map_language_for_xtts("hinglish") == "hi"


# ---------------------------------------------------------------------------
# validate_voice_asset
# ---------------------------------------------------------------------------


class TestValidateVoiceAsset:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(TTSVoiceAssetError, match="missing"):
            validate_voice_asset(tmp_path / "nonexistent.wav")

    def test_valid_file_passes(self, tmp_path):
        p = _wav_file(tmp_path)
        validate_voice_asset(p)  # Should not raise

    def test_directory_raises(self, tmp_path):
        d = tmp_path / "subdir"
        d.mkdir()
        with pytest.raises(TTSVoiceAssetError, match="not a regular file"):
            validate_voice_asset(d)

    def test_unreadable_file_raises(self, tmp_path):
        p = _wav_file(tmp_path)
        p.chmod(0o000)
        try:
            with pytest.raises(TTSVoiceAssetError, match="unreadable"):
                validate_voice_asset(p)
        finally:
            p.chmod(0o644)  # Restore so tmp_path cleanup works.


# ---------------------------------------------------------------------------
# SentenceBoundaryProcessor — basic splits
# ---------------------------------------------------------------------------


class TestSentenceBoundaryProcessorSplits:
    def test_split_on_period(self):
        proc = SentenceBoundaryProcessor()
        result = proc.feed("Hello world. How are you?")
        assert "Hello world." in result
        assert any("How are you" in s for s in result)

    def test_split_on_exclamation(self):
        proc = SentenceBoundaryProcessor()
        result = proc.feed("Stop! Drop it!")
        assert len(result) == 2
        assert result[0] == "Stop!"
        assert result[1] == "Drop it!"

    def test_split_on_question_mark(self):
        proc = SentenceBoundaryProcessor()
        result = proc.feed("Are you ready? Let me know.")
        assert result[0] == "Are you ready?"

    def test_split_on_devanagari_danda(self):
        proc = SentenceBoundaryProcessor()
        result = proc.feed("नमस्ते। कैसे हो?")
        assert any("नमस्ते।" in s for s in result)

    def test_incremental_feed(self):
        proc = SentenceBoundaryProcessor()
        partial = proc.feed("Hello world")
        assert partial == []
        result = proc.feed(". Done.")
        # "Hello world." should now be complete
        assert any("Hello world." in s for s in result)

    def test_empty_string_returns_empty(self):
        proc = SentenceBoundaryProcessor()
        assert proc.feed("") == []

    def test_multiple_sentences_in_one_chunk(self):
        proc = SentenceBoundaryProcessor()
        result = proc.feed("One. Two. Three.")
        # After "Three." there's no trailing whitespace so it may be held.
        flushed = result + proc.flush()
        texts = " ".join(flushed)
        assert "One." in texts
        assert "Two." in texts
        assert "Three." in texts


# ---------------------------------------------------------------------------
# SentenceBoundaryProcessor — abbreviation protection
# ---------------------------------------------------------------------------


class TestAbbreviationProtection:
    def test_mr_does_not_split(self):
        proc = SentenceBoundaryProcessor()
        result = proc.feed("Mr. Smith went home.")
        # The period after "Mr" must not be treated as a sentence boundary.
        # The sentence "Mr. Smith went home." should come out as one unit.
        flushed = result + proc.flush()
        full_text = " ".join(flushed)
        assert "Mr." in full_text
        # Should NOT split "Mr." off as its own sentence.
        assert not any(s.strip() == "Mr." for s in flushed)

    def test_dr_does_not_split(self):
        proc = SentenceBoundaryProcessor()
        result = proc.feed("Dr. Jones examined the patient.")
        flushed = result + proc.flush()
        full_text = " ".join(flushed)
        assert "Dr." in full_text
        assert not any(s.strip() == "Dr." for s in flushed)

    def test_etc_does_not_split(self):
        proc = SentenceBoundaryProcessor()
        result = proc.feed("Bring apples, etc. and oranges.")
        flushed = result + proc.flush()
        # etc. should not produce a sentence boundary mid-sentence
        full_text = " ".join(flushed)
        assert "etc." in full_text

    def test_prof_does_not_split(self):
        proc = SentenceBoundaryProcessor()
        result = proc.feed("Prof. Rao gave the lecture.")
        flushed = result + proc.flush()
        assert not any(s.strip() == "Prof." for s in flushed)


# ---------------------------------------------------------------------------
# SentenceBoundaryProcessor — EOR marker
# ---------------------------------------------------------------------------


class TestEORMarker:
    def test_eor_flushes_remaining_text(self):
        proc = SentenceBoundaryProcessor()
        # Feed partial text then EOR
        result = proc.feed("This is the last sentence<EOR>")
        assert any("This is the last sentence" in s for s in result)

    def test_eor_with_preceding_complete_sentence(self):
        proc = SentenceBoundaryProcessor()
        result = proc.feed("First sentence. Second sentence<EOR>")
        combined = " ".join(result)
        assert "First sentence." in combined
        assert "Second sentence" in combined

    def test_eor_with_empty_preceding_text(self):
        proc = SentenceBoundaryProcessor()
        result = proc.feed("<EOR>")
        assert result == []

    def test_buffer_after_eor_is_cleared(self):
        proc = SentenceBoundaryProcessor()
        proc.feed("Some text<EOR>")
        # Buffer should be cleared; flush returns nothing.
        assert proc.flush() == []


# ---------------------------------------------------------------------------
# SentenceBoundaryProcessor — flush
# ---------------------------------------------------------------------------


class TestFlush:
    def test_empty_buffer_flush(self):
        proc = SentenceBoundaryProcessor()
        assert proc.flush() == []

    def test_flush_returns_remaining_sentence(self):
        proc = SentenceBoundaryProcessor()
        proc.feed("Incomplete sentence")
        result = proc.flush()
        assert result == ["Incomplete sentence"]

    def test_flush_clears_buffer(self):
        proc = SentenceBoundaryProcessor()
        proc.feed("Some text")
        proc.flush()
        assert proc.flush() == []


# ---------------------------------------------------------------------------
# XTTSSentenceAdapter — voice asset validation
# ---------------------------------------------------------------------------


class TestXTTSSentenceAdapterAssetValidation:
    def test_missing_asset_raises_on_init(self, tmp_path):
        pipeline = _mock_pipeline()
        with pytest.raises(TTSVoiceAssetError):
            XTTSSentenceAdapter(
                voice_asset_path=tmp_path / "missing.wav",
                pipeline=pipeline,
            )

    def test_valid_asset_creates_adapter(self, tmp_path):
        pipeline = _mock_pipeline()
        asset = _wav_file(tmp_path)
        adapter = XTTSSentenceAdapter(voice_asset_path=asset, pipeline=pipeline)
        assert adapter is not None


# ---------------------------------------------------------------------------
# XTTSSentenceAdapter — ordered frame emission (mock pipeline)
# ---------------------------------------------------------------------------


class TestXTTSSentenceAdapterFrameEmission:
    @pytest.mark.asyncio
    async def test_emit_tts_text_called_before_pcm(self, tmp_path):
        """TTSTextFrame must be emitted before PCM chunks are delivered."""
        from core.voice.interfaces import VoiceSentence

        pipeline = _mock_pipeline()
        asset = _wav_file(tmp_path)

        # Fake raw PCM: 9600 bytes = 2 chunks of 4800 bytes each.
        raw_pcm = b"\x00\x01" * 4800

        async def _fake_run_blocking(*, library, operation):
            from threading import Event
            return operation(Event())

        pipeline.run_blocking_library = _fake_run_blocking

        call_order: list[str] = []
        original_emit_tts = pipeline.emit_tts_text
        original_emit_pcm = pipeline.emit_pcm_chunk

        async def _track_tts(**kwargs):
            call_order.append("tts_text")
            return await original_emit_tts(**kwargs)

        async def _track_pcm(**kwargs):
            call_order.append(f"pcm_{kwargs['chunk_sequence']}")
            return await original_emit_pcm(**kwargs)

        pipeline.emit_tts_text = _track_tts
        pipeline.emit_pcm_chunk = _track_pcm

        adapter = XTTSSentenceAdapter(voice_asset_path=asset, pipeline=pipeline)
        adapter._session_ready = True

        # Patch TTS to return pre-built PCM bytes.
        class _FakeTTS:
            def tts(self, text, speaker_wav, language):
                # Return float samples; adapter will convert.
                return [s / 32768.0 for s in raw_pcm[::2]]

        adapter._tts_instance = _FakeTTS()

        sentence = VoiceSentence(
            turn_id=uuid4(),
            sentence_id=uuid4(),
            text="Hello world.",
            language="en",
        )
        await adapter.synthesize_sentence(sentence, sequence=0)

        # TTSTextFrame must be first in call order.
        assert call_order[0] == "tts_text"
        assert any("pcm_" in c for c in call_order)

    @pytest.mark.asyncio
    async def test_pcm_chunks_are_ordered(self, tmp_path):
        """PCM chunk_sequence values must be monotonically increasing."""
        from core.voice.interfaces import VoiceSentence

        pipeline = _mock_pipeline()
        asset = _wav_file(tmp_path)
        # 3 chunks worth of PCM (S16LE float samples).
        num_samples = (4800 * 3) // 2  # 4800 bytes per chunk, 2 bytes per sample

        async def _fake_run_blocking(*, library, operation):
            from threading import Event
            return operation(Event())

        pipeline.run_blocking_library = _fake_run_blocking

        chunk_sequences: list[int] = []

        async def _track_pcm(**kwargs):
            chunk_sequences.append(kwargs["chunk_sequence"])

        pipeline.emit_pcm_chunk = _track_pcm

        adapter = XTTSSentenceAdapter(voice_asset_path=asset, pipeline=pipeline)
        adapter._session_ready = True

        class _FakeTTS:
            def tts(self, text, speaker_wav, language):
                return [0.0] * num_samples

        adapter._tts_instance = _FakeTTS()

        sentence = VoiceSentence(
            turn_id=uuid4(), sentence_id=uuid4(), text="Test.", language="en"
        )
        await adapter.synthesize_sentence(sentence, sequence=1)

        assert chunk_sequences == list(range(len(chunk_sequences)))


# ---------------------------------------------------------------------------
# XTTSSentenceAdapter — synthesis failure triggers text fallback
# ---------------------------------------------------------------------------


class TestXTTSSentenceAdapterFailure:
    @pytest.mark.asyncio
    async def test_synthesis_failure_calls_text_fallback(self, tmp_path):
        """On synthesis failure, text_fallback_sink is called with full response."""
        from core.voice.interfaces import VoiceSentence

        fallback_calls: list[str] = []

        async def _fallback_sink(text: str) -> None:
            fallback_calls.append(text)

        pipeline = _mock_pipeline()
        asset = _wav_file(tmp_path)

        async def _fail_run_blocking(*, library, operation):
            raise RuntimeError("XTTS synthesis failed")

        pipeline.run_blocking_library = _fail_run_blocking

        adapter = XTTSSentenceAdapter(
            voice_asset_path=asset,
            pipeline=pipeline,
            text_fallback_sink=_fallback_sink,
        )
        adapter._session_ready = True
        adapter._tts_instance = object()

        sentence = VoiceSentence(
            turn_id=uuid4(), sentence_id=uuid4(), text="Sentence.", language="en"
        )
        await adapter.synthesize_sentence(
            sentence, sequence=0, full_response_text="The full response text."
        )

        assert len(fallback_calls) == 1
        assert fallback_calls[0] == "The full response text."

    @pytest.mark.asyncio
    async def test_synthesis_failure_no_alternate_engine(self, tmp_path):
        """Synthesis failure must NOT invoke any alternate TTS import."""
        from core.voice.interfaces import VoiceSentence
        import sys

        pipeline = _mock_pipeline()
        asset = _wav_file(tmp_path)

        async def _fail_run_blocking(*, library, operation):
            raise RuntimeError("synthesis error")

        pipeline.run_blocking_library = _fail_run_blocking

        adapter = XTTSSentenceAdapter(voice_asset_path=asset, pipeline=pipeline)
        adapter._session_ready = True
        adapter._tts_instance = object()

        sentence = VoiceSentence(
            turn_id=uuid4(), sentence_id=uuid4(), text="Hi.", language="en"
        )
        # Ensure that no pyttsx3, gtts, edge_tts, kokoro imports are triggered.
        forbidden = {"pyttsx3", "gtts", "edge_tts", "kokoro", "ChatTTS"}
        modules_before = set(sys.modules.keys())

        await adapter.synthesize_sentence(sentence, sequence=0)

        new_modules = set(sys.modules.keys()) - modules_before
        for forbidden_module in forbidden:
            assert not any(
                forbidden_module in mod for mod in new_modules
            ), f"Forbidden module imported: {forbidden_module}"


# ---------------------------------------------------------------------------
# XTTSSentenceAdapter — backpressure (bounded semaphore)
# ---------------------------------------------------------------------------


class TestXTTSSentenceAdapterBackpressure:
    @pytest.mark.asyncio
    async def test_synthesis_queue_bounded(self, tmp_path):
        """The synthesis semaphore limits concurrent synthesis slots."""
        from core.voice.interfaces import VoiceSentence

        pipeline = _mock_pipeline()
        asset = _wav_file(tmp_path)

        synthesis_started = asyncio.Event()
        synthesis_can_proceed = asyncio.Event()

        async def _blocking_run(*, library, operation):
            synthesis_started.set()
            await synthesis_can_proceed.wait()
            from threading import Event
            return operation(Event())

        pipeline.run_blocking_library = _blocking_run

        # Queue size of 1 to make the test deterministic.
        adapter = XTTSSentenceAdapter(
            voice_asset_path=asset,
            pipeline=pipeline,
            synthesis_queue_maxsize=1,
        )
        adapter._session_ready = True

        class _FakeTTS:
            def tts(self, text, speaker_wav, language):
                return []

        adapter._tts_instance = _FakeTTS()

        sentence = VoiceSentence(
            turn_id=uuid4(), sentence_id=uuid4(), text="Test.", language="en"
        )

        # Start one synthesis to saturate the semaphore (queue_maxsize=1).
        t1 = asyncio.create_task(
            adapter.synthesize_sentence(sentence, sequence=0)
        )
        # Wait for first synthesis to start.
        await asyncio.wait_for(synthesis_started.wait(), timeout=2.0)

        # Confirm semaphore is held (value = 0 means fully acquired).
        assert adapter._synthesis_semaphore._value == 0

        # Allow first synthesis to complete.
        synthesis_can_proceed.set()
        await asyncio.wait_for(t1, timeout=2.0)

        # After completion, semaphore should be released.
        assert adapter._synthesis_semaphore._value == 1


# ---------------------------------------------------------------------------
# initialize_session — error paths
# ---------------------------------------------------------------------------


class TestInitializeSession:
    @pytest.mark.asyncio
    async def test_init_raises_when_tts_import_fails(self, tmp_path):
        """TTSInitializationError is raised if TTS library is unavailable."""
        import sys

        pipeline = _mock_pipeline()
        asset = _wav_file(tmp_path)

        async def _fake_run_blocking(*, library, operation):
            # Simulate ImportError from inside the operation.
            from threading import Event
            try:
                return operation(Event())
            except TTSInitializationError:
                raise

        pipeline.run_blocking_library = _fake_run_blocking

        adapter = XTTSSentenceAdapter(voice_asset_path=asset, pipeline=pipeline)

        # Patch TTS import to fail.
        with patch.dict(sys.modules, {"TTS": None, "TTS.api": None}):
            with pytest.raises(TTSInitializationError):
                await adapter.initialize_session()
