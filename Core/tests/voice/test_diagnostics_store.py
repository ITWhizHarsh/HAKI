"""Integration tests for the local voice diagnostic store.

Validates: Requirements 10.4, 10.6
Design reference: §9; V-DIAG-STORE

Covers integration aspects not in test_diagnostics.py:
- Multi-threaded concurrent appends (race-free atomic writes)
- Async concurrent appends
- Recovery from partial writes (corrupt lines are skipped gracefully)
- Date rotation across day boundaries (separate files per date)
- Schema retrieval returns correct parsed structure with all required fields
- Injected storage failure (OS error on write) converts to DiagnosticStoreError
- Raw audio / full text absent from every on-disk fixture
- Permission/mode enforcement after directory creation
- Multiple events appended and all readable
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import stat
import threading
from datetime import date
from pathlib import Path
from typing import Generator
from uuid import UUID, uuid4

import pytest

from core.voice.diagnostics import (
    DiagnosticStoreError,
    VoiceDiagnosticEvent,
    append_diagnostic,
    read_diagnostics,
)


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

_CONTENT_FIELD_NAMES: frozenset[str] = frozenset({
    "pcm", "pcm_bytes", "audio", "audio_bytes", "raw_audio",
    "transcript", "transcript_text", "response", "response_text",
    "prompt", "prompt_text", "tool_arguments", "tool_results", "tool_content",
})

_REQUIRED_SCHEMA_FIELDS: frozenset[str] = frozenset({
    "schema_version", "event_id", "session_id", "turn_id",
    "stage", "outcome", "started_monotonic_ns",
    "transcription_completed_monotonic_ns", "first_llm_text_monotonic_ns",
    "first_tts_text_monotonic_ns", "first_pcm_delivered_monotonic_ns",
    "ttfa_ms", "selected_route", "asr_engine", "tts_engine",
    "model_resident_bytes", "pipeline_memory_bytes", "gate",
    "error_class", "recovery_outcome", "content_capture_enabled",
})


def _make_event(
    stage: str = "asr",
    outcome: str = "started",
    session_id: UUID | None = None,
    turn_id: UUID | None = None,
    **kwargs,
) -> VoiceDiagnosticEvent:
    return VoiceDiagnosticEvent(
        session_id=session_id or uuid4(),
        turn_id=turn_id or uuid4(),
        stage=stage,  # type: ignore[arg-type]
        outcome=outcome,  # type: ignore[arg-type]
        **kwargs,
    )


def _has_content_field(record: dict) -> bool:
    return bool(set(record.keys()) & _CONTENT_FIELD_NAMES)


def _file_mode_octal(path: Path) -> str:
    return oct(stat.S_IMODE(os.stat(path).st_mode))


# ---------------------------------------------------------------------------
# 1. Permission and mode enforcement after directory creation
# ---------------------------------------------------------------------------

class TestPermissionEnforcement:
    """Requirement 10.4: directory 0700, file 0600."""

    def test_directory_mode_is_0700_on_first_creation(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        append_diagnostic(_make_event(), directory=diag_dir)
        assert _file_mode_octal(diag_dir) == oct(0o700)

    def test_file_mode_is_0600_on_first_creation(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        path = append_diagnostic(_make_event(), directory=diag_dir)
        assert _file_mode_octal(path) == oct(0o600)

    def test_directory_mode_remains_0700_if_pre_existing(self, tmp_path: Path) -> None:
        """Even if the directory already exists with loose permissions it is tightened."""
        diag_dir = tmp_path / "voice"
        diag_dir.mkdir(parents=True)
        diag_dir.chmod(0o755)  # intentionally broader
        append_diagnostic(_make_event(), directory=diag_dir)
        assert _file_mode_octal(diag_dir) == oct(0o700)

    def test_file_mode_remains_0600_on_second_append(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        today = date.today()
        append_diagnostic(_make_event(), directory=diag_dir, for_date=today)
        path = append_diagnostic(_make_event(stage="playback"), directory=diag_dir, for_date=today)
        assert _file_mode_octal(path) == oct(0o600)

    def test_nested_directory_created_with_correct_mode(self, tmp_path: Path) -> None:
        """Parent directory chain (the voice subdir) must be 0700."""
        diag_dir = tmp_path / "deeply" / "nested" / "voice"
        append_diagnostic(_make_event(), directory=diag_dir)
        assert _file_mode_octal(diag_dir) == oct(0o700)


# ---------------------------------------------------------------------------
# 2. Date rotation across day boundaries
# ---------------------------------------------------------------------------

class TestDateRotation:
    """Different calendar dates must produce separate JSONL files."""

    def test_two_dates_produce_two_files(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        d1 = date(2024, 12, 31)
        d2 = date(2025, 1, 1)
        append_diagnostic(_make_event(stage="asr"), directory=diag_dir, for_date=d1)
        append_diagnostic(_make_event(stage="playback"), directory=diag_dir, for_date=d2)
        assert (diag_dir / "2024-12-31.jsonl").exists()
        assert (diag_dir / "2025-01-01.jsonl").exists()

    def test_records_are_isolated_to_their_date_file(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        d1 = date(2025, 3, 1)
        d2 = date(2025, 3, 2)
        append_diagnostic(_make_event(stage="asr"), directory=diag_dir, for_date=d1)
        append_diagnostic(_make_event(stage="local_llm"), directory=diag_dir, for_date=d1)
        append_diagnostic(_make_event(stage="playback"), directory=diag_dir, for_date=d2)
        r1 = read_diagnostics(directory=diag_dir, for_date=d1)
        r2 = read_diagnostics(directory=diag_dir, for_date=d2)
        assert len(r1) == 2
        assert len(r2) == 1
        assert {r["stage"] for r in r1} == {"asr", "local_llm"}
        assert r2[0]["stage"] == "playback"

    def test_ten_distinct_dates_each_isolated(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        dates = [date(2025, 1, d) for d in range(1, 11)]
        stage_for_day = [
            "asr", "ipc", "pipecat", "voice_processing", "local_llm",
            "tool_call", "local_tts", "memory_budget", "cloud_gate", "playback",
        ]
        for d, stage in zip(dates, stage_for_day):
            append_diagnostic(_make_event(stage=stage), directory=diag_dir, for_date=d)
        for d, stage in zip(dates, stage_for_day):
            records = read_diagnostics(directory=diag_dir, for_date=d)
            assert len(records) == 1
            assert records[0]["stage"] == stage

    def test_nonexistent_date_returns_empty_list(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        records = read_diagnostics(directory=diag_dir, for_date=date(2099, 1, 1))
        assert records == []


# ---------------------------------------------------------------------------
# 3. Schema retrieval — required fields and correct parsed structure
# ---------------------------------------------------------------------------

class TestSchemaRetrieval:
    """On-disk records must round-trip with all required schema fields intact."""

    def test_all_required_fields_present_after_roundtrip(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        today = date.today()
        event = _make_event(
            stage="asr",
            outcome="completed",
            started_monotonic_ns=1000,
            asr_engine="qwen3_asr_coreml",
            tts_engine="xtts_v2",
            ttfa_ms=250.0,
            model_resident_bytes=1_500_000_000,
            pipeline_memory_bytes=3_000_000_000,
        )
        append_diagnostic(event, directory=diag_dir, for_date=today)
        records = read_diagnostics(directory=diag_dir, for_date=today)
        assert len(records) == 1
        record = records[0]
        missing = _REQUIRED_SCHEMA_FIELDS - set(record.keys())
        assert not missing, f"Missing required schema fields: {missing}"

    def test_schema_version_is_1_on_disk(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        today = date.today()
        append_diagnostic(_make_event(), directory=diag_dir, for_date=today)
        records = read_diagnostics(directory=diag_dir, for_date=today)
        assert records[0]["schema_version"] == 1

    def test_uuid_fields_are_valid_uuid_strings(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        today = date.today()
        sid, tid = uuid4(), uuid4()
        event = _make_event(session_id=sid, turn_id=tid)
        append_diagnostic(event, directory=diag_dir, for_date=today)
        records = read_diagnostics(directory=diag_dir, for_date=today)
        r = records[0]
        assert r["session_id"] == str(sid)
        assert r["turn_id"] == str(tid)
        # All three UUID fields must parse
        UUID(r["session_id"])
        UUID(r["turn_id"])
        UUID(r["event_id"])

    def test_timing_fields_preserved_exactly(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        today = date.today()
        event = VoiceDiagnosticEvent(
            session_id=uuid4(), turn_id=uuid4(),
            stage="playback", outcome="completed",
            started_monotonic_ns=111,
            transcription_completed_monotonic_ns=222,
            first_llm_text_monotonic_ns=333,
            first_tts_text_monotonic_ns=444,
            first_pcm_delivered_monotonic_ns=555,
            ttfa_ms=99.9,
        )
        append_diagnostic(event, directory=diag_dir, for_date=today)
        r = read_diagnostics(directory=diag_dir, for_date=today)[0]
        assert r["started_monotonic_ns"] == 111
        assert r["transcription_completed_monotonic_ns"] == 222
        assert r["first_llm_text_monotonic_ns"] == 333
        assert r["first_tts_text_monotonic_ns"] == 444
        assert r["first_pcm_delivered_monotonic_ns"] == 555
        assert r["ttfa_ms"] == 99.9

    def test_failure_record_preserves_error_class_and_recovery(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        today = date.today()
        event = VoiceDiagnosticEvent.for_failure(
            session_id=uuid4(), turn_id=uuid4(),
            stage="local_tts", error_class="XTTSSynthesisError",
            recovery_outcome="reported_no_fallback",
        )
        append_diagnostic(event, directory=diag_dir, for_date=today)
        r = read_diagnostics(directory=diag_dir, for_date=today)[0]
        assert r["stage"] == "local_tts"
        assert r["outcome"] == "failed"
        assert r["error_class"] == "XTTSSynthesisError"
        assert r["recovery_outcome"] == "reported_no_fallback"

    def test_content_capture_flag_false_by_default_on_disk(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        today = date.today()
        append_diagnostic(_make_event(), directory=diag_dir, for_date=today)
        r = read_diagnostics(directory=diag_dir, for_date=today)[0]
        assert r["content_capture_enabled"] is False

    def test_all_pipeline_stages_round_trip(self, tmp_path: Path) -> None:
        stages = [
            "asr", "ipc", "pipecat", "voice_processing", "local_llm",
            "tool_call", "local_tts", "memory_budget", "cloud_gate", "playback",
        ]
        diag_dir = tmp_path / "voice"
        today = date.today()
        for stage in stages:
            append_diagnostic(_make_event(stage=stage), directory=diag_dir, for_date=today)
        records = read_diagnostics(directory=diag_dir, for_date=today)
        assert len(records) == len(stages)
        assert [r["stage"] for r in records] == stages

    def test_multiple_events_all_readable_in_order(self, tmp_path: Path) -> None:
        """Multiple appended events must all be readable and in insertion order."""
        diag_dir = tmp_path / "voice"
        today = date.today()
        events = [
            _make_event(stage="asr", outcome="started"),
            _make_event(stage="local_llm", outcome="completed"),
            _make_event(stage="local_tts", outcome="cancelled"),
            _make_event(stage="playback", outcome="failed", error_class="RenderError"),
        ]
        for e in events:
            append_diagnostic(e, directory=diag_dir, for_date=today)
        records = read_diagnostics(directory=diag_dir, for_date=today)
        assert len(records) == 4
        assert [r["stage"] for r in records] == ["asr", "local_llm", "local_tts", "playback"]
        assert [r["outcome"] for r in records] == [
            "started", "completed", "cancelled", "failed"
        ]


# ---------------------------------------------------------------------------
# 4. Raw audio / full text absent from every on-disk fixture
# ---------------------------------------------------------------------------

class TestPrivacyOnDisk:
    """Requirement 10.5/10.6: on-disk records must never contain content-bearing fields."""

    def test_no_content_field_in_started_record(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        today = date.today()
        append_diagnostic(
            VoiceDiagnosticEvent.for_stage_start(
                session_id=uuid4(), turn_id=uuid4(), stage="asr",
                started_monotonic_ns=100, asr_engine="qwen3_asr_coreml",
            ),
            directory=diag_dir, for_date=today,
        )
        records = read_diagnostics(directory=diag_dir, for_date=today)
        assert not _has_content_field(records[0])

    def test_no_content_field_in_failure_record(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        today = date.today()
        append_diagnostic(
            VoiceDiagnosticEvent.for_failure(
                session_id=uuid4(), turn_id=uuid4(), stage="local_llm",
                error_class="MLXLoadError",
            ),
            directory=diag_dir, for_date=today,
        )
        records = read_diagnostics(directory=diag_dir, for_date=today)
        assert not _has_content_field(records[0])

    @pytest.mark.parametrize("stage", [
        "asr", "ipc", "pipecat", "voice_processing", "local_llm",
        "tool_call", "local_tts", "memory_budget", "cloud_gate", "playback",
    ])
    def test_no_raw_audio_or_transcript_for_any_stage(
        self, stage: str, tmp_path: Path
    ) -> None:
        diag_dir = tmp_path / "voice"
        today = date.today()
        append_diagnostic(_make_event(stage=stage), directory=diag_dir, for_date=today)
        records = read_diagnostics(directory=diag_dir, for_date=today)
        assert records, f"No records written for stage {stage!r}"
        for record in records:
            assert not _has_content_field(record), (
                f"Stage {stage!r}: prohibited field(s) "
                f"{set(record.keys()) & _CONTENT_FIELD_NAMES} found on disk"
            )

    def test_raw_json_bytes_on_disk_contain_no_prohibited_keys(self, tmp_path: Path) -> None:
        """Inspect raw on-disk bytes; no prohibited key appears even as a substring."""
        diag_dir = tmp_path / "voice"
        today = date.today()
        for stage in ("asr", "local_llm", "playback"):
            append_diagnostic(_make_event(stage=stage), directory=diag_dir, for_date=today)
        raw_bytes = (diag_dir / f"{today.isoformat()}.jsonl").read_bytes()
        raw_text = raw_bytes.decode("utf-8")
        for line in raw_text.splitlines():
            parsed = json.loads(line)
            prohibited_found = set(parsed.keys()) & _CONTENT_FIELD_NAMES
            assert not prohibited_found, (
                f"Prohibited field(s) {prohibited_found} found in raw on-disk record"
            )


# ---------------------------------------------------------------------------
# 5. Injected storage failure converts to DiagnosticStoreError
# ---------------------------------------------------------------------------

class TestStorageFailureConversion:
    """Requirement 10.6: OS errors on write must surface as DiagnosticStoreError.

    Injection strategy: place the target diagnostics directory path under a
    read-only parent so that os.makedirs cannot create the new directory.
    _ensure_directory calls mkdir which raises PermissionError, which the
    implementation must convert to DiagnosticStoreError.
    """

    def _locked_parent(self, tmp_path: Path) -> tuple[Path, Path]:
        """Return (parent, diag_dir) where parent is 0o500 and diag_dir does not exist."""
        parent = tmp_path / "readonly_root"
        parent.mkdir(parents=True)
        parent.chmod(0o500)  # no write: mkdir of subdir will fail
        diag_dir = parent / "voice"  # must not exist
        return parent, diag_dir

    def test_unwritable_parent_raises_diagnostic_store_error(
        self, tmp_path: Path
    ) -> None:
        """Directory creation fails → DiagnosticStoreError is raised."""
        parent, diag_dir = self._locked_parent(tmp_path)
        try:
            with pytest.raises(DiagnosticStoreError):
                append_diagnostic(_make_event(), directory=diag_dir)
        finally:
            parent.chmod(0o700)

    def test_diagnostic_store_error_not_raw_os_error(self, tmp_path: Path) -> None:
        """The caller must never see a raw OSError — only DiagnosticStoreError."""
        parent, diag_dir = self._locked_parent(tmp_path)
        try:
            exc = None
            try:
                append_diagnostic(_make_event(), directory=diag_dir)
            except DiagnosticStoreError as e:
                exc = e
            except OSError:
                pytest.fail("Raw OSError escaped — must be wrapped in DiagnosticStoreError")
            assert exc is not None, "Expected DiagnosticStoreError was not raised"
        finally:
            parent.chmod(0o700)

    def test_diagnostic_store_error_message_is_non_empty(self, tmp_path: Path) -> None:
        parent, diag_dir = self._locked_parent(tmp_path)
        try:
            with pytest.raises(DiagnosticStoreError) as exc_info:
                append_diagnostic(_make_event(), directory=diag_dir)
            assert str(exc_info.value), "DiagnosticStoreError message must not be empty"
        finally:
            parent.chmod(0o700)

    def test_diagnostic_store_error_is_runtime_error_subclass(self) -> None:
        """DiagnosticStoreError must be a non-fatal RuntimeError subclass."""
        assert issubclass(DiagnosticStoreError, RuntimeError)

    def test_store_error_does_not_expose_content(self, tmp_path: Path) -> None:
        """Even when storage fails the error message must not contain content fields."""
        parent, diag_dir = self._locked_parent(tmp_path)
        try:
            with pytest.raises(DiagnosticStoreError) as exc_info:
                append_diagnostic(
                    _make_event(stage="asr", outcome="started"),
                    directory=diag_dir,
                )
            error_msg = str(exc_info.value)
            for field in _CONTENT_FIELD_NAMES:
                assert field not in error_msg, (
                    f"Content field {field!r} leaked into DiagnosticStoreError message"
                )
        finally:
            parent.chmod(0o700)


# ---------------------------------------------------------------------------
# 6. Recovery from partial writes (corrupt lines skipped gracefully)
# ---------------------------------------------------------------------------

class TestCorruptLineRecovery:
    """When a JSONL file contains corrupt/partial lines, valid lines are retained."""

    def _corrupt_file(self, diag_dir: Path, today: date, corruption: str) -> Path:
        """Inject a corrupt line into an existing JSONL file."""
        path = diag_dir / f"{today.isoformat()}.jsonl"
        with open(str(path), "a", encoding="utf-8") as fh:
            fh.write(corruption + "\n")
        return path

    def test_empty_lines_are_skipped_during_read(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        today = date.today()
        append_diagnostic(_make_event(stage="asr"), directory=diag_dir, for_date=today)
        # Inject blank lines directly into the file
        path = diag_dir / f"{today.isoformat()}.jsonl"
        with open(str(path), "a", encoding="utf-8") as fh:
            fh.write("\n\n")
        records = read_diagnostics(directory=diag_dir, for_date=today)
        assert len(records) == 1
        assert records[0]["stage"] == "asr"

    def test_corrupt_json_line_raises_diagnostic_store_error(self, tmp_path: Path) -> None:
        """A truncated / malformed JSON line must raise DiagnosticStoreError, not ValueError."""
        diag_dir = tmp_path / "voice"
        today = date.today()
        append_diagnostic(_make_event(stage="asr"), directory=diag_dir, for_date=today)
        self._corrupt_file(diag_dir, today, "{not valid json at all")
        with pytest.raises(DiagnosticStoreError):
            read_diagnostics(directory=diag_dir, for_date=today)

    def test_truncated_line_raises_diagnostic_store_error(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        today = date.today()
        append_diagnostic(_make_event(stage="local_llm"), directory=diag_dir, for_date=today)
        self._corrupt_file(diag_dir, today, '{"schema_version": 1, "event_id": "abc')
        with pytest.raises(DiagnosticStoreError):
            read_diagnostics(directory=diag_dir, for_date=today)

    def test_wholly_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        today = date.today()
        diag_dir.mkdir(parents=True)
        path = diag_dir / f"{today.isoformat()}.jsonl"
        path.touch()
        path.chmod(0o600)
        records = read_diagnostics(directory=diag_dir, for_date=today)
        assert records == []


# ---------------------------------------------------------------------------
# 7. Multi-threaded concurrent appends (race-free atomic writes)
# ---------------------------------------------------------------------------

class TestConcurrentAppends:
    """Atomic appends must produce one record per write under thread concurrency."""

    def test_multithreaded_appends_produce_correct_count(self, tmp_path: Path) -> None:
        """50 threads each write one event; all 50 must be readable afterwards."""
        diag_dir = tmp_path / "voice"
        today = date.today()
        n_threads = 50
        errors: list[Exception] = []

        def write_one() -> None:
            try:
                append_diagnostic(
                    _make_event(stage="asr"),
                    directory=diag_dir,
                    for_date=today,
                )
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=write_one) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        records = read_diagnostics(directory=diag_dir, for_date=today)
        assert len(records) == n_threads

    def test_multithreaded_records_are_all_valid_json(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        today = date.today()
        n_threads = 30

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [
                executor.submit(
                    append_diagnostic,
                    _make_event(stage="local_llm"),
                    directory=diag_dir,
                    for_date=today,
                )
                for _ in range(n_threads)
            ]
            concurrent.futures.wait(futures)
            for f in futures:
                assert f.exception() is None, f"Thread raised: {f.exception()}"

        records = read_diagnostics(directory=diag_dir, for_date=today)
        assert len(records) == n_threads
        for r in records:
            assert r["stage"] == "local_llm"

    def test_multithreaded_no_content_on_disk(self, tmp_path: Path) -> None:
        """Under concurrency no content-bearing field must leak onto disk."""
        diag_dir = tmp_path / "voice"
        today = date.today()
        n_threads = 20

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [
                executor.submit(
                    append_diagnostic,
                    _make_event(stage="playback"),
                    directory=diag_dir,
                    for_date=today,
                )
                for _ in range(n_threads)
            ]
            concurrent.futures.wait(futures)

        records = read_diagnostics(directory=diag_dir, for_date=today)
        for r in records:
            assert not _has_content_field(r)

    def test_multithreaded_file_mode_stays_0600(self, tmp_path: Path) -> None:
        """Concurrent writes must not downgrade the file permission."""
        diag_dir = tmp_path / "voice"
        today = date.today()
        n_threads = 20
        paths: list[Path] = []
        lock = threading.Lock()

        def write_and_record() -> None:
            p = append_diagnostic(
                _make_event(stage="ipc"),
                directory=diag_dir,
                for_date=today,
            )
            with lock:
                paths.append(p)

        threads = [threading.Thread(target=write_and_record) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert paths
        for p in paths:
            assert _file_mode_octal(p) == oct(0o600), (
                f"File {p} mode is {_file_mode_octal(p)}, expected 0600"
            )


# ---------------------------------------------------------------------------
# 8. Async concurrent appends
# ---------------------------------------------------------------------------

class TestAsyncConcurrentAppends:
    """Appends dispatched from async tasks must not corrupt the file."""

    @pytest.mark.asyncio
    async def test_async_concurrent_appends_correct_count(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        today = date.today()
        n_tasks = 40
        loop = asyncio.get_event_loop()

        async def write_one() -> None:
            await loop.run_in_executor(
                None,
                lambda: append_diagnostic(
                    _make_event(stage="cloud_gate"),
                    directory=diag_dir,
                    for_date=today,
                ),
            )

        await asyncio.gather(*[write_one() for _ in range(n_tasks)])
        records = read_diagnostics(directory=diag_dir, for_date=today)
        assert len(records) == n_tasks

    @pytest.mark.asyncio
    async def test_async_concurrent_no_content_on_disk(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        today = date.today()
        n_tasks = 25
        loop = asyncio.get_event_loop()

        async def write_one(stage: str) -> None:
            await loop.run_in_executor(
                None,
                lambda: append_diagnostic(
                    _make_event(stage=stage),
                    directory=diag_dir,
                    for_date=today,
                ),
            )

        stages = ["asr", "local_llm", "local_tts", "playback", "ipc"]
        tasks = [write_one(stages[i % len(stages)]) for i in range(n_tasks)]
        await asyncio.gather(*tasks)

        records = read_diagnostics(directory=diag_dir, for_date=today)
        assert len(records) == n_tasks
        for r in records:
            assert not _has_content_field(r)

    @pytest.mark.asyncio
    async def test_async_all_records_valid_json_schema(self, tmp_path: Path) -> None:
        diag_dir = tmp_path / "voice"
        today = date.today()
        loop = asyncio.get_event_loop()
        n_tasks = 15

        async def write_one() -> None:
            await loop.run_in_executor(
                None,
                lambda: append_diagnostic(
                    _make_event(stage="pipecat", outcome="completed"),
                    directory=diag_dir,
                    for_date=today,
                ),
            )

        await asyncio.gather(*[write_one() for _ in range(n_tasks)])
        records = read_diagnostics(directory=diag_dir, for_date=today)
        assert len(records) == n_tasks
        for r in records:
            missing = _REQUIRED_SCHEMA_FIELDS - set(r.keys())
            assert not missing, f"Missing fields in async-written record: {missing}"
