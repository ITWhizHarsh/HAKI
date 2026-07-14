# Implementation Plan: HAKI Brain Memory Processing Pipeline

## Overview

This plan implements the Architect's Hybrid Pipeline in incremental steps: dependency
setup, the Fast Pass extractor, the memory note writer/link validator, the process
tracker, the `LLMRouter` `prefer_local` flag, the Heavy Pass extractor, the `HAKIBrain`
hybrid routing refactor, the conversation processing flow, the `PipelineScheduler`, and
finally service wiring in `haki_core_service.py`. Property-based tests (Hypothesis,
Properties 1–9) and unit tests are added alongside each component so regressions are
caught as early as possible. All code is Python, matching the existing `Core/` codebase.

## Tasks

- [x] 1. Set up dependencies and configuration
  - Add `spacy>=3.7`, `apscheduler>=3.10`, `psutil>=5.9`, `hypothesis>=6.100` to the
    project's dependency file (`requirements.txt` / `pyproject.toml`, matching existing style)
  - Document the `python -m spacy download en_core_web_sm` post-install step
  - Fix `Core/.env`: set `HAKI_OBSIDIAN_VAULT=/Users/harshkumarroy/Downloads/HKR/HAKI/HAKI_Brain`
  - Add new configuration keys with defaults documented in code/comments:
    `HAKI_PIPELINE_RAW_INTERVAL_MINUTES` (30), `HAKI_PIPELINE_CONV_RUN_TIME` (02:00),
    `HAKI_PIPELINE_LOW_MEMORY_THRESHOLD_MB` (500), `HAKI_PIPELINE_LLM_TIMEOUT_SECS` (30)
  - _Requirements: 8.4, 9.3, 9.7_

- [x] 2. Implement `FastPassExtractor` (`core/memory/fast_pass.py`)
  - [x] 2.1 Create `FastPassStatus` enum, `Entity` and `FastPassResult` dataclasses
    - _Requirements: 1.2_
  - [x] 2.2 Implement `FastPassExtractor` with lazy `spaCy` loading (`_load_spacy`),
    regex patterns (EMAIL, URL, PHONE, HAKI_MARKER), Hindi/Hinglish fallback, and
    entity deduplication by `(text, label)`
    - Return `SUCCESS` when ≥1 unique entity found, `NO_ENTITIES` when none found,
      `ERROR` with diagnostic message on exception
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.7_
  - [x] 2.3 Implement `_entities_to_markdown()` helper for pre-formatted note bodies
    - _Requirements: 1.4_
  - [ ]* 2.4 Write unit tests for `FastPassExtractor`
    - Cover: PERSON entity extraction, email regex match, empty content →
      `NO_ENTITIES`, Hinglish fallback trigger, deduplication, spaCy-unavailable
      fallback to regex-only mode
    - _Requirements: 1.2, 1.3, 1.5, 1.7_
  - [ ]* 2.5 Write property test for Fast Pass idempotency
    - **Property 1: Fast Pass Idempotency**
    - **Validates: Requirements 1.2, 1.3**
  - [ ]* 2.6 Write property test for no-LLM-call guarantee on Fast Pass success
    - **Property 2: No LLM Call During Fast Pass Success**
    - **Validates: Requirements 1.1, 1.3, 1.6, 7.1**

- [x] 3. Checkpoint - ensure Fast Pass tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement `MemoryNoteWriter` and link validation (`core/memory/memory_note_writer.py`)
  - [x] 4.1 Create `MemoryNotePath` dataclass, `_slugify()`, and
    `validate_wiki_link(target, vault_root)` helper function
    - _Requirements: 11.1, 11.2, 11.5_
  - [x] 4.2 Implement `MemoryNoteWriter.write_fast_pass_note()` with provenance-link
    validation before write (skip note creation and log diagnostic if source file
    does not resolve)
    - _Requirements: 3.1, 3.3, 3.4, 1.4_
  - [x] 4.3 Implement `MemoryNoteWriter.write_heavy_pass_note()` with provenance-link
    and evolutionary-link validation before write
    - _Requirements: 3.2, 3.3, 3.4, 5.1, 5.3, 5.4, 5.5_
  - [x] 4.4 Implement `_write_atomic()` (temp file + `fsync` + `rename`, cleanup temp
    file on failure) and `_embed_note()` (ChromaDB upsert after successful write)
    - _Requirements: 10.2_
  - [x] 4.5 Implement duplicate-link detection so the writer never writes two links of
    the same type targeting the same destination note within one Memory_Note
    - _Requirements: 11.3_
  - [ ]* 4.6 Write unit tests for `MemoryNoteWriter`
    - Cover: file appears in `wiki/` after write, no `.tmp` file left after a
      simulated write failure, `validate_wiki_link` true/false cases, evolutionary
      link skipped when target missing, provenance link skipped when source missing
    - _Requirements: 3.3, 3.4, 5.4, 10.2, 11.2, 11.5_
  - [ ]* 4.7 Write property test for provenance link completeness
    - **Property 3: Provenance Link Completeness**
    - **Validates: Requirements 3.1, 3.2, 3.3, 11.1, 11.2, 11.4**
  - [ ]* 4.8 Write property test for entity-to-note count on Fast Pass
    - **Property 4: Entity-to-Note Count (Fast Pass)**
    - **Validates: Requirements 1.4**
  - [ ]* 4.9 Write property test for evolutionary link validity
    - **Property 6: Evolutionary Link Validity**
    - **Validates: Requirements 5.1, 5.3, 5.4, 11.2, 11.5**
  - [ ]* 4.10 Write property test for link resolution (no dangling wiki links)
    - **Property 7: Link Resolution**
    - **Validates: Requirements 11.1, 11.2, 11.4, 11.5**

- [x] 5. Checkpoint - ensure MemoryNoteWriter tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement `ProcessTracker` (`core/memory/process_tracker.py`)
  - [x] 6.1 Create SQLite-backed `ProcessTracker` with `processed_logs` table (WAL mode),
    `mark_processed()`, and `is_processed()`
    - _Requirements: 6.5_
  - [x] 6.2 Implement `get_unprocessed_conversations(cutoff_date, conversations_dir)`
    returning `YYYY-MM-DD.md` filenames on or before cutoff, not yet processed,
    sorted chronologically oldest-first
    - _Requirements: 6.1, 6.4, 6.6_
  - [ ]* 6.3 Write unit tests for `ProcessTracker`
    - Cover: mark-then-query returns true, unmarked query returns false, unprocessed
      list is sorted oldest-first, today's log excluded by cutoff filtering
    - _Requirements: 6.1, 6.4, 6.5, 6.6_

- [x] 7. Add `prefer_local` flag to `LLMRouter` (`core/model_provider/llm_router.py`)
  - [x] 7.1 Add `prefer_local: bool = False` parameter to `chat()` and `stream_chat()`
    - _Requirements: 2.3, 2.4, 2.6_
  - [x] 7.2 Update `_routing_order()` to return `[LLMTier.LOCAL_MLX]` exclusively when
    `prefer_local=True`, bypassing cloud tiers
    - _Requirements: 2.3, 2.4, 2.6_
  - [x] 7.3 Update `_stream_local_mlx()` to accept `prefer_standby` and try
    `mlx_standby_model` (Bonsai-8B) before `mlx_primary_model` when set, falling back
    to the primary model if the standby model fails to load
    - _Requirements: 2.6, 2.7, 2.8_
  - [ ]* 7.4 Write unit tests for `prefer_local` routing
    - Cover: `prefer_local=True` returns only `LOCAL_MLX` tier, standby model tried
      first, fallback to primary model on standby failure
    - _Requirements: 2.3, 2.4, 2.6, 2.7_

- [x] 8. Implement `HeavyPassExtractor` (`core/memory/heavy_pass.py`)
  - [x] 8.1 Create `HeavyPassStatus` enum and `HeavyPassResult` dataclass
    - _Requirements: 2.5_
  - [x] 8.2 Implement ChromaDB top-3 semantic query step, selecting the top-1 result as
    the candidate old Memory_Note for evolutionary linking
    - _Requirements: 2.2, 2.3, 5.5_
  - [x] 8.3 Implement evolutionary prompt (`_EVOLUTIONARY_PROMPT`, `_HEAVY_PASS_SYSTEM`)
    used when an old note is found, and fresh synthesis prompt
    (`_FRESH_SYNTHESIS_PROMPT`) used when ChromaDB returns no results
    - _Requirements: 2.3, 2.4, 2.6_
  - [x] 8.4 Implement `extract()`: call `LLMRouter.chat(prefer_local=True)` wrapped in
    `asyncio.wait_for()` with configurable timeout; return `TIMEOUT` on
    `asyncio.TimeoutError`, `LLM_ERROR` on empty/unparseable response, `SUCCESS`
    otherwise with the `EVOLVED_FROM:` line parsed out and stripped from the content
    - _Requirements: 2.5, 2.7, 2.8_
  - [ ]* 8.5 Write unit tests for `HeavyPassExtractor`
    - Cover: evolutionary prompt built when ChromaDB has results, fresh synthesis
      prompt built when ChromaDB is empty, timeout returns `TIMEOUT` status, empty
      LLM response returns `LLM_ERROR`, `EVOLVED_FROM:` line correctly parsed and
      stripped
    - _Requirements: 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 5.5_
  - [ ]* 8.6 Write property test for evolutionary link validity end-to-end with
    `HeavyPassExtractor` outputs feeding `MemoryNoteWriter`
    - **Property 6: Evolutionary Link Validity**
    - **Validates: Requirements 5.1, 5.3, 5.4, 11.2, 11.5**

- [x] 9. Checkpoint - ensure Heavy Pass and LLMRouter tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Refactor `HAKIBrain._ingest_file()` to hybrid routing (`core/memory/haki_brain.py`)
  - [x] 10.1 In `HAKIBrain.__init__()`, instantiate `FastPassExtractor` and
    `ProcessTracker`, and read `HAKI_PIPELINE_LLM_TIMEOUT_SECS` and
    `HAKI_PIPELINE_LOW_MEMORY_THRESHOLD_MB` from environment
    - _Requirements: 8.1, 8.2_
  - [x] 10.2 Implement `_check_low_memory()` using `psutil.virtual_memory().available`,
    returning `False` (no guard) if `psutil` is not installed
    - _Requirements: 7.6_
  - [x] 10.3 Rewrite `_ingest_file()` body: read content, run `FastPassExtractor`, on
    `SUCCESS` write one Memory_Note per entity via `MemoryNoteWriter` and move the
    source file to `processed/` only after all writes succeed; on `NO_ENTITIES` or
    `ERROR` fall through to the Heavy Pass
    - _Requirements: 1.1, 1.3, 1.4, 1.5, 1.6, 1.7, 4.1, 4.4, 4.5, 10.1, 10.2, 10.3_
  - [x] 10.4 In the Heavy Pass branch, call `_check_low_memory()` before constructing
    `HeavyPassExtractor`; if low memory, leave the file in `raw/` and record a
    diagnostic instead of loading the LLM
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_
  - [x] 10.5 Implement `_get_processed_dest()` to generate a collision-safe destination
    filename (stem + timestamp + original extension) when a same-named file already
    exists in `processed/`
    - _Requirements: 4.3_
  - [x] 10.6 Ensure filesystem move failures after successful writes are caught,
    logged as diagnostics, and leave the source file in `raw/` without marking it
    processed
    - _Requirements: 4.6, 10.5_
  - [x] 10.7 Wrap each `_ingest_file()` call site (`ingest_pending()`) in a
    per-file try/except so one file's failure does not stop processing of the
    remaining files in the same run, and append a run summary (files processed,
    notes created, Fast Pass successes, Heavy Pass invocations, errors) to the
    diagnostic log at run end
    - _Requirements: 4.2, 10.1, 10.4_
  - [ ]* 10.8 Write unit tests for `HAKIBrain` ingestion routing
    - Cover: Fast Pass success moves file to `processed/`, Heavy Pass invoked when
      Fast Pass yields nothing, both-pass failure leaves file in `raw/`, low-memory
      mock defers Heavy Pass, collision-safe destination naming
    - _Requirements: 1.5, 1.6, 4.1, 4.3, 4.4, 4.6, 7.6, 10.1_
  - [ ]* 10.9 Write property test for file movement atomicity
    - **Property 5: File Movement Atomicity**
    - **Validates: Requirements 4.1, 4.4, 4.5, 4.6, 10.2, 10.3, 10.5**

- [x] 11. Checkpoint - ensure HAKIBrain ingestion tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Implement conversation processing (`core/memory/haki_brain.py`)
  - [x] 12.1 Implement `process_pending_conversations()`: compute cutoff as
    yesterday's date, fetch unprocessed logs from `ProcessTracker` in chronological
    order, process each via `_process_conversation_log()`, call `mark_processed()`
    only on success, and accumulate a run summary
    - _Requirements: 6.1, 6.4, 6.5, 6.6, 6.8_
  - [x] 12.2 Implement `_process_conversation_log()`: apply Fast Pass then Heavy Pass
    using `source_vault_rel = "conversations/{filename}"`, and never move or modify
    the log file
    - _Requirements: 6.2, 6.3, 6.7_
  - [ ]* 12.3 Write unit tests for conversation processing
    - Cover: unprocessed logs are picked up in order, today's log is excluded,
      already-processed logs are skipped, conversation files remain in
      `conversations/` after processing, log stays unmarked when both passes fail
    - _Requirements: 6.1, 6.3, 6.4, 6.5, 6.8_
  - [ ]* 12.4 Write property test for conversation log immutability
    - **Property 8: Conversation Log Immutability**
    - **Validates: Requirements 6.3, 6.4**
  - [ ]* 12.5 Write property test for conversation log processing idempotency
    - **Property 9: Conversation Log Processing Idempotency**
    - **Validates: Requirements 6.5**

- [x] 13. Checkpoint - ensure conversation processing tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. Implement `PipelineScheduler` (`core/memory/pipeline_scheduler.py`)
  - [x] 14.1 Implement `PipelineScheduler.__init__()` and `start()` using
    `AsyncIOScheduler`, registering `haki_raw_pipeline` (`IntervalTrigger`, default 30
    minutes) and `haki_conv_pipeline` (`CronTrigger`, default 02:00), both with
    `max_instances=1` and `coalesce=True`
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.6_
  - [x] 14.2 Implement `_raw_job()` and `_conv_job()` wrappers with an in-flight guard
    that skips a new trigger if the previous run of the same job is still active, and
    `stop()` to shut down the scheduler
    - _Requirements: 9.4, 9.6_
  - [x] 14.3 Implement `_parse_time()` to parse `HH:MM`, falling back to the default
    `02:00` and logging an error on an invalid or out-of-range value
    - _Requirements: 9.7_
  - [ ]* 14.4 Write unit tests for `PipelineScheduler`
    - Cover: jobs registered with correct trigger types and `max_instances=1`,
      overlapping trigger is skipped while a job is running, invalid time string
      falls back to default and logs an error
    - _Requirements: 9.1, 9.2, 9.6, 9.7_

- [-] 15. Wire startup validation and live-turn deferral into `HAKIBrain`
  - [x] 15.1 Implement vault path startup validation in `HAKIBrain.init()`: abort with
    a diagnostic message if `HAKI_OBSIDIAN_VAULT` is unset, empty, not absolute, does
    not exist, or is not read/write accessible
    - _Requirements: 8.1, 8.2, 8.3, 8.5_
  - [x] 15.2 Implement `_wait_for_live_turn()` and call it before Vault-modifying
    operations in `_ingest_file()` and `_process_conversation_log()`; abort the
    deferred operation and log a warning if the wait exceeds 10 minutes
    - _Requirements: 9.5, 9.8_
  - [ ]* 15.3 Write unit tests for startup validation and live-turn deferral
    - Cover: missing/relative/non-existent vault path aborts startup with a
      diagnostic, live-turn wait times out after the configured deferral window
    - _Requirements: 8.3, 8.5, 9.5, 9.8_

- [x] 16. Wire `PipelineScheduler` into `haki_core_service.py`
  - [x] 16.1 Replace `haki_brain.start_watching()` with construction and `start()` of
    `PipelineScheduler`, reading `HAKI_PIPELINE_RAW_INTERVAL_MINUTES` and
    `HAKI_PIPELINE_CONV_RUN_TIME` from the environment with a `_safe_int()` helper
    that falls back to defaults and logs on out-of-range/invalid values
    - _Requirements: 9.1, 9.2, 9.3, 9.7_
  - [x] 16.2 Ensure `Core/.env` validation failure (from task 15.1) prevents the
    service from starting the scheduler and performing any Vault modifications
    - _Requirements: 8.3, 8.5_
  - [ ]* 16.3 Write an integration/smoke test verifying `haki_core_service.py` starts
    `PipelineScheduler` with the correct interval and cron configuration when
    `HAKI_OBSIDIAN_VAULT` is valid, and does not start it when the vault path is
    invalid
    - _Requirements: 8.3, 8.5, 9.1, 9.2_

- [x] 17. Final checkpoint - ensure full pipeline test suite passes
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP; they are not
  implemented by the coding agent by default.
- Properties 1–9 correspond directly to the Correctness Properties defined in
  `design.md` and use Hypothesis with a minimum of 100 iterations per test (50 for
  Properties 8 and 9 per the design's testing strategy table).
- Each property test must be tagged with a comment in the form:
  `Feature: haki-brain-memory-processing-pipeline, Property {N}: {property_text}`
- Checkpoints are placed after each major component (Fast Pass, MemoryNoteWriter,
  Heavy Pass/LLMRouter, HAKIBrain ingestion, conversation processing) to validate
  incremental progress before moving to integration and scheduler wiring.
