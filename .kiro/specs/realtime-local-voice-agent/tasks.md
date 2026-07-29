# Implementation Plan: Realtime Local Voice Agent

## Overview

Implement the full-duplex, local-first Swift/Python voice runtime in safe layers. First create and enforce an inert, sanitized legacy archive; then build the versioned same-UID transport and local components; then integrate and validate the replacement path behind an internal-only development gate. Only after the automated isolation, failure, hardware, privacy, and resource gates pass may live voice wiring be cut over and all old live voice routes removed. No task may add a legacy runtime fallback, compatibility mode, or archive execution path.

## Tasks

- [x] 1. Freeze, archive, and statically isolate the legacy voice pipeline
  - [x] 1.1 Implement deterministic legacy-voice discovery, classification, sanitization, and inventory generation
    - **Targets:** `tools/archive_legacy_voice.py`, `legacy_voice_manifest.yaml`, `legacy_pipeline_backup/inventory.jsonl`.
    - Discover every script, handler, configuration, dependency declaration, startup path, and routing rule; classify each item; record repository-relative source path, category, source SHA-256, archive path, archive SHA-256, and sanitizer status.
    - Parse supported config formats fail-closed; redact credential values to `__REDACTED_LEGACY_SECRET__` while preserving keys and paths. Archive source as inert `.txt` content without package markers or executable entrypoints.
    - **Dependencies:** None. **Requirements:** 1.1, 1.3, 1.4. **Design:** §§1, 10; Property 1. **Focused validation:** Fixture inventories cover all categories, supported config parsers, unparseable probable secrets, and deterministic digest output.
  - [x] 1.2 Generate and repository-track the complete static legacy archive
    - **Targets:** `legacy_pipeline_backup/`, `legacy_pipeline_backup/README.md`, generated `inventory.jsonl` and archived static copies.
    - Run the migration tool against the approved manifest; preserve original-relative layout below archive root, verify one inert sanitized copy per inventory record, and ensure archive contents contain no importable modules, launch scripts, symlinks, or package discovery markers.
    - **Dependencies:** 1.1. **Requirements:** 1.1–1.4. **Design:** §1. **Focused validation:** Check all inventory mappings, source/archive digests, tracked archive paths, permissions, and absence of executable/package artifacts.
  - [x] 1.3 Implement zero-coupling archive enforcement and its CI policy
    - **Targets:** `tools/check_voice_archive.py`, `.github/workflows/voice-archive-policy.yml` (or repository-equivalent CI), package/build configuration.
    - Rebuild the inventory in check-only mode; AST- and metadata-scan runtime, tests, startup, service registration, dependency declarations, dynamic import/subprocess paths, generated-code targets, and symlinks. Reject archive imports, discovery, execution, bundling, provider selection, and user/developer fallback switches; exclude the archive from app/package data.
    - **Dependencies:** 1.1, 1.2. **Requirements:** 1.5. **Design:** §1. **Focused validation:** CI fixtures demonstrate failures for static imports, `sys.path`, dynamic imports, subprocesses, launch paths, package data, and secret residue.
  - [ ]* 1.4 Write property tests for archive inventory and redaction preservation
    - **Targets:** `Core/tests/voice/test_archive_properties.py`.
    - **Property 1: Archive inventory and redaction preservation.** Generate supported artifact/config sets and assert exactly one mapped inert archive record/copy, category and source-digest preservation, and complete value redaction with keys retained; run at least 100 cases.
    - **Dependencies:** 1.1–1.3. **Validates:** Requirements 1.1, 1.3, 1.4. **Design:** Property 1; `V-ARCHIVE`. **Focused validation:** Seed malformed/ambiguous configs and verify fail-closed behavior.

- [x] 2. Establish pinned local-only dependencies, model manifests, and voice package foundations
  - [x] 2.1 Pin direct production dependencies and isolate retired voice dependencies
    - **Targets:** `Core/pyproject.toml`, `Core/requirements.txt`, generated lockfile, `HAKI/Package.resolved`.
    - Add matching exact direct pins for `pipecat-ai==1.4.0`, `mlx-lm==0.18.1`, `TTS==0.22.0`, and `silero-vad==5.1.2`, plus the design-pinned supporting packages. Remove old voice SDKs/providers and unbounded voice dependency ranges; retain a dependency only in an explicit non-voice extra when independently required.
    - **Dependencies:** None. **Requirements:** 9.6. **Design:** §10. **Focused validation:** Resolve from clean environments; assert lockfile consistency and that `core.voice` has no legacy/cloud voice dependency imports.
  - [x] 2.2 Implement local model/voice-asset availability manifests and non-blocking startup checks
    - **Targets:** `Core/core/voice/resources.py`, model-manifest helpers, Swift voice configuration, startup health-check wiring.
    - Define non-secret local artifact manifests/hashes for CoreML Qwen3 ASR and Qwen3-4B-Instruct-4bit; validate the readable user-supplied `my_voice.wav`; make startup report missing assets/permissions without downloading or converting models during a turn.
    - **Dependencies:** 2.1. **Requirements:** 3.1, 6.2, 7.1–7.2, 9.6. **Design:** §§3, 6, 7, 10. **Focused validation:** Mock present, missing, hash-mismatched, and unreadable assets; verify failure is actionable and never selects legacy or cloud voice components.
  - [x] 2.3 Create the typed `core.voice` package boundary and prohibit reuse of legacy runtime routers
    - **Targets:** `Core/core/voice/__init__.py`, package exports, import-boundary checks.
    - Establish the new voice namespace and explicit interfaces only; prevent `core.voice` from importing legacy STT/TTS paths, `LLMRouter._routing_order()`, archive files, or process-wide `Orchestrator` conversation mutation.
    - **Dependencies:** 1.3, 2.1. **Requirements:** 1.5, 4.1, 6.1, 7.1. **Design:** Overview, Current Integration Constraints, §10. **Focused validation:** Import-boundary/static checks show the new package has only permitted local/capability dependencies.

- [x] 3. Implement the secure, versioned Swift/Python text-and-control UDS protocol
  - [x] 3.1 Define strict versioned protocol schemas, framing, and peer-security primitives
    - **Targets:** `Core/core/ipc/voice_protocol.py`, protocol fixtures/shared schema documentation embedded in code.
    - Implement v1 JSONL schemas for transcript/control events, acknowledgements, PCM output metadata with length-prefixed output payloads, and terminal playback events. Reject unknown fields, microphone payload fields, malformed IDs, invalid language/finality/event sequences, duplicate finals, stale sessions, and invalid binary lengths.
    - **Dependencies:** 2.3. **Requirements:** 3.2–3.6, 3.8, 5.6–5.7. **Design:** §§2–3. **Focused validation:** Schema unit tests cover valid partial/final sequences, rejection reasons, no raw-microphone fields, and protocol-version incompatibility.
  - [x] 3.2 Implement the same-UID `VoiceUnixServer` with secure lifecycle and terminal turn handling
    - **Targets:** `Core/core/ipc/voice_unix_server.py`, UDS lifecycle helpers.
    - Create session-random socket paths/capabilities, atomic creation, `0700` parents and `0600` sockets, peer UID verification, stale-socket type/ownership validation, bounded reconnect state, ACK semantics, and disconnect-before-final discard/cancel behavior. Keep this server separate from non-voice handlers and do not use it to transport microphone PCM.
    - **Dependencies:** 3.1. **Requirements:** 3.4–3.6, 3.8. **Design:** §3. **Focused validation:** Temporary same-UID integration tests cover modes, wrong UID/capability, symlink refusal, stale files, reconnect backoff, late events, and disconnect before final.
  - [x] 3.3 Implement the Swift UDS client and output/control event handling
    - **Targets:** `HAKI/Sources/Subsystems/IPC/VoiceSocketClient.swift`, Swift protocol test fixtures.
    - Encode ordered transcript events, ACK/reconnect handling, session capability negotiation, `CAPTURE_INTERRUPTED`, `PCM_CHUNK`, `STOP_PLAYBACK`, and exactly-one playback terminal events. On disconnect, discard the unfinished final turn rather than replay it after reconnect.
    - **Dependencies:** 3.1, 3.2. **Requirements:** 3.3–3.6, 3.8, 5.2, 5.6–5.7. **Design:** §§2–3. **Focused validation:** Swift/Python transport contract tests prove final-event discard, idempotent stop, sequence enforcement, and prohibited microphone fields.
  - [ ]* 3.4 Write the sequenced capture/transcript/Pipecat property test
    - **Targets:** `Core/tests/voice/test_frame_protocol_properties.py`.
    - **Property 3: Sequenced local capture, transcript, and Pipecat frame preservation.** Generate at least 100 ordered/gapped frame and partial/final event streams; assert monotonic capture metadata, no PCM over UDS, at-most-one ordered final, mandated Pipecat frame types, and per-turn terminal ordering.
    - **Dependencies:** 3.1–3.3, 4.2, 5.2. **Validates:** Requirements 2.6, 3.2, 3.3, 3.6, 4.2–4.6, 4.8. **Design:** Property 3; `V-FRAME-PROP`. **Focused validation:** Include reconnect, duplicate-final, out-of-order, and cancellation-generation examples.

- [x] 4. Build the Swift full-duplex native audio, local ASR, ephemeral ring, and renderer
  - [x] 4.1 Implement `VoiceAudioController` VoiceProcessingIO lifecycle and capture recovery
    - **Targets:** `HAKI/Sources/Subsystems/Audio/VoiceAudioController.swift`.
    - Use one session-owned `AVAudioEngine`; set and verify `inputNode.isVoiceProcessingEnabled` before installing the tap or accepting frames. Normalize input to 16 kHz mono, assign monotonic sequence/timestamps before dispatch, keep capture active during playback, and transition to actionable unavailable/retry states on permission, route, engine, tap, or VoiceProcessingIO failure.
    - **Dependencies:** 2.2. **Requirements:** 2.1–2.6. **Design:** §2. **Focused validation:** Instrument start order, state transitions, bounded reset retries, and prove no frame passes when voice processing is false.
  - [x] 4.2 Implement the same-UID transient `AudioFrameRing` and native PCM playback renderer
    - **Targets:** `HAKI/Sources/Subsystems/Audio/AudioFrameRing.swift`, `HAKI/Sources/Subsystems/Audio/PCMPlaybackRenderer.swift`.
    - Implement session-random, bounded shared-memory descriptors usable only by same UID for Pipecat `InputAudioRawFrame`/Silero; zero/unlink at session end; drop only oldest non-final frames without reordering. Schedule TTS PCM on the controller’s same-engine player node; send enqueue acknowledgements and exactly one confirmed/cancelled/failed terminal event per sentence; make stop high-priority/idempotent.
    - **Dependencies:** 4.1, 3.1. **Requirements:** 2.3, 2.5–2.6, 4.2, 5.2, 5.6–5.7. **Design:** §§2, 4–5. **Focused validation:** Assert no microphone bytes enter UDS/diagnostics; test ring access/modes/zeroization, ordered descriptors, renderer terminal uniqueness, and stop acknowledgement.
  - [x] 4.3 Implement the CoreML Qwen3 local ASR adapter and transcript generation contract
    - **Targets:** `HAKI/Sources/Subsystems/Audio/LocalASRAdapter.swift`, CoreML adapter/configuration and model-manifest integration.
    - Implement `LocalASRAdapter` with `CoreMLQwen3ASRAdapter` as production default: consume 16 kHz frames, manage partial/final hypotheses, NFC/whitespace/control-character normalization, `hi|en|hinglish` classification, discontinuity partial reset, empty-final diagnostic/repeat behavior, and local-only model selection.
    - **Dependencies:** 2.2, 4.1, 3.3. **Requirements:** 3.1–3.3, 3.7. **Design:** §3. **Focused validation:** Synthetic Hindi, English, Hinglish, empty, gap, and cancelled-turn fixtures demonstrate ordered partial-before-final output and no LLM creation for empty finals.
  - [ ]* 4.4 Add macOS native-audio integration tests **[hardware/macOS gate]**
    - **Targets:** `HAKI/Tests/Voice/VoiceAudioControllerTests.swift`, `HAKI/Tests/Voice/PCMPlaybackRendererTests.swift`.
    - Exercise instrumented enable→verify→tap ordering, duplex capture/playback, route/media-reset recovery, unavailable VoiceProcessingIO, capture persistence while playing, and renderer stop-terminal behavior on macOS 14+.
    - **Dependencies:** 4.1–4.3. **Validates:** Requirements 2.1–2.5, 5.2, 5.6–5.7. **Design:** §2; `V-SWIFT-AUDIO`. **Focused validation:** Label tests requiring microphone/audio route entitlement and keep them outside non-macOS CI.

- [x] 5. Implement the session-owned Pipecat graph and bounded frame transport
  - [x] 5.1 Implement voice session state, typed frame metadata, and turn registry
    - **Targets:** `Core/core/voice/session.py`, `Core/core/voice/frames.py`.
    - Define one `VoiceSession` per active session, linear turn states, `TurnRegistry`, per-turn `asyncio.Lock`, sequence/cancellation generation metadata, bounded queues, and terminal rejection of late frames. Keep voice context session-owned rather than using `Orchestrator._conversation_history`.
    - **Dependencies:** 2.3, 3.1. **Requirements:** 4.1, 4.6, 4.8, 5.8. **Design:** §§4–5. **Focused validation:** Unit-test legal/illegal state transitions, interleaved turns, late terminal frames, and per-turn ordering.
  - [x] 5.2 Implement ASR/ring ingress adapters for mandatory Pipecat input/transcription frames
    - **Targets:** `Core/core/voice/asr_bridge.py`, `Core/core/voice/pipeline.py` ingress processors.
    - Temporarily map/release authenticated ring slots into `InputAudioRawFrame` only for Silero; validate and ACK transcript socket messages only after sequencing acceptance; attach typed metadata and produce `TranscriptionFrame` without embedding metadata in text. Ring failure must leave capture/VAD unavailable rather than serialize microphone data over the socket.
    - **Dependencies:** 3.1–3.2, 4.2, 5.1. **Requirements:** 3.4–3.6, 4.2–4.3, 4.6. **Design:** §4. **Focused validation:** Exercise ring-unavailable, descriptor gap, socket rejection, and immediate release behavior with mocked Pipecat adapters.
  - [x] 5.3 Implement `VoiceSessionPipeline` initialization, frame graph, and backpressure policy
    - **Targets:** `Core/core/voice/pipeline.py`.
    - Build a single `PipelineTask` in the existing asyncio loop using `TaskGroup`, bounded queues, cancellation-aware single-worker executors only for blocking libraries, and the required `InputAudioRawFrame`, `TranscriptionFrame`, `LLMTextFrame`, and `TTSTextFrame` path. Make initialization all-or-nothing and forbid custom-thread/subprocess playback substitutes.
    - **Dependencies:** 2.1, 5.1–5.2. **Requirements:** 4.1–4.6, 4.8–4.9. **Design:** §4. **Focused validation:** Async integration confirms initialization failure leaves voice unavailable, queue policies preserve non-droppable controls/order, and no legacy/custom-thread path is instantiated.
  - [ ]* 5.4 Write Pipecat pipeline integration tests
    - **Targets:** `Core/tests/voice/test_pipeline_integration.py`.
    - Verify final transcript ingress to the mandatory frame graph, bounded partial/final/control/LLM/PCM queues, terminal ordering, all-or-nothing initialization, and replacement-only failure outcomes.
    - **Dependencies:** 5.1–5.3. **Validates:** Requirements 4.1–4.6, 4.8–4.9. **Design:** §4; `V-PIPELINE`. **Focused validation:** Inject initialization, queue, executor, and sink failures and assert no custom or legacy runtime begins.

- [x] 6. Implement smart-turn VAD, immediate barge-in, cancellation, and playback-confirmed context
  - [x] 6.1 Implement Silero VAD smart-turn state and turn joining
    - **Targets:** `Core/core/voice/vad.py`, `Core/core/voice/pipeline.py` turn-join processor.
    - Consume 16 kHz `InputAudioRawFrame`s with configurable hysteresis; apply 800 ms continuous post-speech silence only when no barge-in is active; correlate final transcripts and VAD state by turn ID; update partial UI state without beginning LLM generation until the final smart-turn condition is met.
    - **Dependencies:** 5.2–5.3. **Requirements:** 4.7. **Design:** §§4–5. **Focused validation:** Deterministic synthetic timeline tests cover silence resets, no transcript-only turn start, and final eligibility exactly at threshold.
  - [x] 6.2 Implement `BargeInCoordinator` cancellation generations and `PlaybackLedger`
    - **Targets:** `Core/core/voice/vad.py`, `Core/core/voice/session.py`, `Core/core/voice/pipeline.py` cancellation/ledger processors.
    - Declare barge-in after 200 ms continuous speech during playback; atomically increment generation, high-priority stop renderer, cancel LLM/synthesis, drain matching queued TTS/PCM work, mark provisional sentences cancelled, and resume new capture without awaiting cleanup. Append assistant text only on normal renderer confirmation, in confirmation order.
    - **Dependencies:** 4.2, 5.1, 5.3, 6.1. **Requirements:** 5.1–5.8. **Design:** §5. **Focused validation:** Race tests prove late token/chunk/confirmation cannot revive cancelled output and unconfirmed text never enters later prompt context.
  - [x]* 6.3 Write the smart-turn silence property test
    - **Targets:** `Core/tests/voice/test_smart_turn_properties.py`.
    - **Property 4: Smart-turn silence rule.** Generate at least 100 VAD speech/silence/barge-in timelines and assert finalization iff continuous silence reaches 800 ms while barge-in is inactive.
    - **Dependencies:** 6.1. **Validates:** Requirement 4.7. **Design:** Property 4; `V-TURN-PROP`. **Focused validation:** Include clock boundaries, recurrent speech, VAD hysteresis changes, and active-barge cases.
  - [x]* 6.4 Write the barge-in invalidation property and deterministic latency test
    - **Targets:** `Core/tests/voice/test_barge_in_properties.py`, `Core/tests/voice/test_barge_in_latency.py`.
    - **Property 5: Barge-in invalidates interrupted assistant work.** Across at least 100 scheduled work/generation cases, verify 200 ms speech cancels matching LLM/TTS/PCM work and capture proceeds before cleanup. Use a controllable renderer clock to assert threshold-to-declaration and declaration-to-stop acknowledgement are each at most 200 ms.
    - **Dependencies:** 6.2. **Validates:** Requirements 5.1–5.5. **Design:** Property 5; `V-TURN-PROP`, `V-BARGE-LATENCY`. **Focused validation:** Verify idempotent repeated signals and stale-generation producer rejection.
  - [x]* 6.5 Write the confirmed-playback context property test
    - **Targets:** `Core/tests/voice/test_playback_ledger_properties.py`.
    - **Property 6: Confirmed-playback context ledger.** Generate at least 100 ordered schedules with confirmations, failures, cancellations, and interruptions; assert user turns plus exactly confirmed assistant sentences appear in completion order and nothing provisional is exposed to later turns.
    - **Dependencies:** 6.2. **Validates:** Requirements 5.6–5.8. **Design:** Property 6; `V-TURN-PROP`. **Focused validation:** Include out-of-order confirmations and duplicate terminal renderer events.

- [x] 7. Add local MLX Qwen routing and memory admission controls
  - [x] 7.1 Implement `VoiceResourceManager` warm-up, measurement, draining, and recovery admission
    - **Targets:** `Core/core/voice/resources.py`.
    - Warm required local components, measure MLX resident footprint and union pipeline memory using process-specific macOS metrics without double counting, and enforce `>=2.5 GB` model or `>5 GB` pipeline draining. Reject new turns, release idle XTTS then idle Qwen/cache, preserve active capture safety, and reopen only when both recovery thresholds are met.
    - **Dependencies:** 2.1–2.2, 5.1. **Requirements:** 9.1, 9.2, 9.4–9.5. **Design:** §6. **Focused validation:** Mock boundary samples, resource release order, double-count prevention, rejection diagnostics, and dual-threshold recovery.
  - [x] 7.2 Implement the fixed local `VoiceLLMRouter` and MLX streaming service
    - **Targets:** `Core/core/voice/llm.py`.
    - Load only `Qwen/Qwen3-4B-Instruct-4bit` through `mlx-lm==0.18.1` with Metal enabled, a one-model cache, 16,384 context cap, 1,024 generation cap, and one generation semaphore. Stream ordered LLM frames with decode-step cancellation; use the cloud gate only as explicit routing input and never call legacy or broad `LLMRouter` fallback chains.
    - **Dependencies:** 2.1–2.2, 5.1, 7.1, 10.1. **Requirements:** 6.1–6.2, 6.7, 8.5–8.6, 9.1. **Design:** §6. **Focused validation:** Mock MLX load/stream/terminal failures and assert local error reporting rather than provider fallback.
  - [x]* 7.3 Write the memory admission and recovery property test
    - **Targets:** `Core/tests/voice/test_resource_properties.py`.
    - **Property 11: Memory admission and recovery.** Generate at least 100 model/pipeline measurements and assert limit-triggered rejection/release/diagnostic behavior, with admission resuming only below 2.5 GB model and at-or-below 5 GB pipeline memory.
    - **Dependencies:** 7.1. **Validates:** Requirements 9.4–9.5. **Design:** Property 11; `V-RESOURCE`. **Focused validation:** Cover exact thresholds, incomplete release, active work, and repeated samples.
  - [x]* 7.4 Write local-route and MLX configuration tests
    - **Targets:** `Core/tests/voice/test_local_llm.py`, `Core/tests/voice/test_local_route_properties.py`.
    - **Property 7: Local-route default.** For at least 100 non-eligible gate decisions, assert routing selects configured local Qwen/Metal only. Test exact model/runtime configuration and load/generation error terminal handling.
    - **Dependencies:** 7.2, 10.1. **Validates:** Requirements 6.1–6.2, 6.7, 8.5–8.6. **Design:** Property 7; `V-LLM`. **Focused validation:** Assert no Groq, Cerebras, Gemini, or legacy route is selected in any non-eligible/failure case.

- [x] 8. Implement strict voice tool schemas and capability adapters
  - [x] 8.1 Implement registered Pydantic tool validation and turn-correlated capability execution
    - **Targets:** `Core/core/voice/tools.py`, `Core/core/voice/llm.py` tool integration.
    - Define strict, versioned allowlisted schemas with `extra="forbid"` for `obsidian_rag.search` and `screen_control.run`; parse model output as data, validate before invocation, bound results, route to `HAKIBrain`/`ScreenAgent`, preserve all existing ScreenAgent permission/consequential-action confirmation, and return valid results only to the originating turn.
    - **Dependencies:** 5.1, 7.2. **Requirements:** 6.3–6.6. **Design:** §6. **Focused validation:** Verify invalid data has zero target calls, valid results are turn-correlated, and voice-originated screen requests never auto-confirm.
  - [ ]* 8.2 Write the tool-safety and turn-correlation property test
    - **Targets:** `Core/tests/voice/test_tool_properties.py`.
    - **Property 8: Tool-call safety and turn correlation.** Fuzz at least 100 valid/invalid tool objects, asserting only registered schema-conforming calls invoke their target, invalid calls emit diagnostics, and valid results resume the same LLM turn.
    - **Dependencies:** 8.1. **Validates:** Requirements 6.3–6.6. **Design:** Property 8; `V-TOOLS-PROP`. **Focused validation:** Include unknown tools/fields, bounds violations, malformed JSON, injection-shaped strings, and ScreenAgent confirmation preservation.

- [x] 9. Implement sentence-streaming XTTS and reliable native playback delivery
  - [x] 9.1 Implement sentence boundaries and the local XTTS v2 adapter
    - **Targets:** `Core/core/voice/tts.py`, sentence-boundary processor in `Core/core/voice/pipeline.py`.
    - Segment non-empty output on terminal punctuation/explicit EOR with abbreviation protection; emit each ordered `TTSTextFrame` promptly; map `en` to `en` and Hindi/Hinglish to `hi`; initialize one `TTS==0.22.0` XTTS v2 session conditioned on validated `my_voice.wav`; permit next-sentence synthesis during prior playback under bounded backpressure.
    - **Dependencies:** 2.1–2.2, 5.3, 7.1–7.2. **Requirements:** 7.1–7.5, 7.7. **Design:** §7. **Focused validation:** Test segmentation, language mapping, ordered incremental frame emission, overlap scheduling, missing asset rejection, and no system/legacy TTS selection.
  - [x] 9.2 Implement the Python PCM sink, TTFA measurement, and synthesis failure text fallback
    - **Targets:** `Core/core/voice/tts.py`, `Core/core/voice/pipeline.py`, `Core/core/ipc/voice_protocol.py` output integration.
    - Emit ordered bounded PCM chunks with `(turn_id, sentence_id, chunk_seq)` to Swift, measure TTFA from completed TTS text frame to first non-empty renderer-received chunk, and on synthesis failure cancel remaining unplayed work, record diagnostics, and surface the complete generated response as UI text without an alternate speech engine.
    - **Dependencies:** 3.1–3.2, 4.2, 6.2, 9.1, 11.1. **Requirements:** 7.5–7.8, 10.2, 10.6. **Design:** §§2, 7, 9. **Focused validation:** Mock PCM delivery/cancellation/failure paths and verify only confirmed sentences reach the ledger.
  - [ ]* 9.3 Write the sentence streaming language/overlap property test
    - **Targets:** `Core/tests/voice/test_tts_properties.py`.
    - **Property 10: Sentence streaming language and overlap.** Generate at least 100 language/classification/chunk cases and assert prompt non-empty frame emission, exact `en`/`hi` mapping, sentence order, and permitted N+1 synthesis while N plays subject to backpressure.
    - **Dependencies:** 9.1–9.2. **Validates:** Requirements 7.3–7.5, 7.7. **Design:** Property 10; `V-TTS-PROP`. **Focused validation:** Include Devanagari punctuation, explicit EOR, abbreviations, empty buffers, and cancellation generations.
  - [ ]* 9.4 Add warmed XTTS TTFA benchmark **[hardware/macOS gate]**
    - **Targets:** `Core/tests/voice/test_tts_ttfa_benchmark.py`, benchmark configuration.
    - Execute reproducible warmed representative English/Hindi/Hinglish sentences with provisioned `my_voice.wav`; record first delivered non-empty PCM and assert TTFA is at most 500 ms under Nominal_Test_Profile.
    - **Dependencies:** 4.2, 9.1–9.2. **Validates:** Requirements 7.1–7.2, 7.6. **Design:** §7; `V-TTS-TTFA`. **Focused validation:** Separate hardware results from unit CI and report environment/profile metadata without recording speech content.

- [x] 10. Implement the explicit session-scoped Gemini Live eligibility gate
  - [x] 10.1 Implement `CloudEscalationGate` session state, conditions, and decision diagnostics
    - **Targets:** `Core/core/voice/cloud_gate.py`, `Core/core/voice/session.py`, `Core/core/voice/diagnostics.py` gate integration.
    - Initialize every voice session disabled; require an explicit active-session UI action to enable and show state; remove enablement at session end. Evaluate battery/power, thermal, prompt-token, and validated-tool-count conditions exactly as designed; return an eligibility decision for the router to invoke Gemini only when enabled and qualifying; report an eligible invocation failure without fallback to Qwen, other cloud providers, or legacy.
    - **Dependencies:** 5.1, 11.1. **Requirements:** 8.1–8.7, 10.3. **Design:** §8. **Focused validation:** Exhaustive unit truth table checks condition boundaries, session isolation/end, diagnostic values, and no implicit retries.
  - [x]* 10.2 Write the session-scoped cloud-gate truth-table property test
    - **Targets:** `Core/tests/voice/test_cloud_gate_properties.py`.
    - **Property 9: Session-scoped cloud gate truth table.** Generate at least 100 combinations of sessions, enable/end events, battery/power, thermal state, prompt size, and validated tool count; assert eligibility iff explicit active-session enablement and one qualifying condition are both present, plus complete gate diagnostics.
    - **Dependencies:** 10.1. **Validates:** Requirements 8.1–8.6, 10.3. **Design:** Property 9; `V-GATE-PROP`. **Focused validation:** Cover 20% battery, external-power, serious/critical thermal, 16,000-token, and six-tool boundaries.
  - [x]* 10.3 Add eligible-cloud failure integration coverage
    - **Targets:** `Core/tests/voice/test_cloud_failure.py`.
    - Mock an eligible Gemini Live invocation failure and verify a terminal reported cloud error with no same-turn local, other-cloud, archive, or legacy fallback.
    - **Dependencies:** 10.1. **Validates:** Requirements 1.6, 8.7. **Design:** §§1, 8; `V-CLOUD-FAIL`. **Focused validation:** Assert route-selection telemetry and invoked-provider spies remain single-route.

- [x] 11. Add privacy-preserving local voice diagnostics and failure reporting
  - [x] 11.1 Implement versioned diagnostics schema, redaction, and atomic local store
    - **Targets:** `Core/core/voice/diagnostics.py`.
    - Create atomic JSONL diagnostics under the local HAKI directory with `0700`/`0600` permissions. Record required identifiers, stages, component IDs, timestamps, routes, gate data, TTFA, memory, terminal/failure/recovery fields; omit PCM, transcript/response/prompt/tool content by default. Implement session-scoped content capture defaulting false and expiring at session end; never allow raw-audio capture.
    - **Dependencies:** 2.3, 5.1. **Requirements:** 10.1–10.6. **Design:** §9. **Focused validation:** Serialize start/completion/cancellation/failure paths and assert default fields exclude every content-bearing/raw-byte field.
  - [x]* 11.2 Write the privacy-preserving diagnostic completeness property test
    - **Targets:** `Core/tests/voice/test_diagnostics_properties.py`.
    - **Property 12: Privacy-preserving diagnostic completeness.** Generate at least 100 starts, terminal outcomes, gate decisions, and failures; assert required identifiers/measurements/errors exist while default records contain no audio or full transcript, with content fields permitted only by active session control.
    - **Dependencies:** 10.1, 11.1. **Validates:** Requirements 10.1–10.3, 10.5–10.6. **Design:** Property 12; `V-DIAG-PROP`. **Focused validation:** Include cancellation, missing optional metrics, redaction hashes, content-control expiry, and serialization round trips.
  - [x]* 11.3 Add local diagnostic-store integration tests
    - **Targets:** `Core/tests/voice/test_diagnostics_store.py`.
    - Test atomic append/recovery, owner/mode enforcement, date rotation, schema retrieval, and injected storage failure conversion to a stable non-content-bearing outcome.
    - **Dependencies:** 11.1. **Validates:** Requirements 10.4, 10.6. **Design:** §9; `V-DIAG-STORE`. **Focused validation:** Inspect file modes and assert raw audio/full text are absent from every on-disk fixture.

- [x] 12. Integrate the replacement runtime behind a non-production internal development gate
  - [ ] 12.1 Wire Swift audio, secure transport, and Python `VoiceSessionPipeline` into an isolated replacement-only session path
    - **Targets:** `Core/core/ipc/voice_unix_server.py`, `Core/core/voice/pipeline.py`, `Core/core/voice/session.py`, Swift audio/IPC composition root, internal development configuration.
    - Connect the completed components through an internal-only development replacement gate for automated integration. The gate may enable only the new local path; it must not select, wrap, invoke, or fall back to legacy live routes/archive artifacts, and must preserve existing non-voice IPC handlers.
    - **Dependencies:** 3.1–3.3, 4.1–4.3, 5.1–5.3, 6.1–6.2, 7.1–7.2, 8.1, 9.1–9.2, 10.1, 11.1. **Requirements:** 1.5–1.6, 2.1–2.6, 3.1–3.8, 4.1–4.9, 5.1–5.8, 6.1–6.7, 7.1–7.8, 8.1–8.7, 9.4–9.6, 10.1–10.6. **Design:** Overview, §§2–10. **Focused validation:** Mocked end-to-end component test exercises transcript→turn→PCM→confirmation and each stage failure without starting live legacy code.
  - [ ]* 12.2 Add replacement-path integration and fault-injection tests
    - **Targets:** `Core/tests/voice/test_replacement_integration.py`, `HAKI/Tests/Voice/VoiceProtocolIntegrationTests.swift`.
    - Validate transcript-only UDS, ring-only microphone transport, Pipecat ordering/backpressure, VAD/barge-in, local tool/TTS/renderer flows, and every replacement-stage terminal failure with spies proving no archive/legacy process, import, provider, route, or fallback is used.
    - **Dependencies:** 12.1. **Validates:** Requirements 1.5–1.6, 3.4–3.8, 4.1–4.9, 5.1–5.8, 6.7, 7.8, 8.7. **Design:** §§1–9; `V-ASR-IPC`, `V-PIPELINE`, fault-injection matrix. **Focused validation:** Assert non-voice IPC remains functional and no raw microphone content appears in protocol/diagnostics.

- [ ] 13. Checkpoint — verify isolation and replacement safety before live cutover
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. Cut over live voice wiring and remove all executable legacy voice routes
  - [ ] 14.1 Replace live voice service wiring with `VoiceUnixServer` and `VoiceSessionPipeline`
    - **Targets:** `Core/core/ipc/server.py`, `Core/haki_core_service.py`, voice startup/service registration, Swift application composition root.
    - Route live voice only to the replacement UDS/session pipeline while retaining non-voice IPC behavior. Remove the internal development gate in favor of the local-first production route; initialize Gemini disabled; surface replacement unavailable/errors directly rather than attempting any fallback.
    - **Dependencies:** 1.3–1.4, 3.4, 4.4, 5.4, 6.3–6.5, 7.3–7.4, 8.2, 9.3–9.4, 10.2–10.3, 11.2–11.3, 12.2, 15.2. **Requirements:** 1.5–1.6, 4.1, 8.1, 8.5–8.7. **Design:** Current Integration Constraints, §11 migration steps 5–6. **Focused validation:** Startup integration verifies a live voice session creates only replacement components and normal session default is local Qwen.
  - [ ] 14.2 Remove live legacy buffering, providers, subprocess playback, and route selections
    - **Targets:** `Core/core/ipc/server.py`, legacy `STTEngine`/`TTSEngine` runtime selection, `Core/core/orchestrator/orchestrator.py` voice coupling, `Core/core/model_provider/llm_router.py` voice coupling, startup scripts/service configuration, voice dependency declarations.
    - Delete old raw-audio buffering, Deepgram/Groq/Cartesia/Edge/Kokoro/ChatTTS/system-speech selections, `afplay`/`say` playback subprocess paths, and voice-specific cloud fallback order. Preserve only static archive references and independently required non-voice functionality; do not move deleted code into a compatibility switch.
    - **Dependencies:** 1.2–1.3, 14.1. **Requirements:** 1.2, 1.5–1.6, 6.1, 6.7, 7.1, 7.8, 8.7, 9.6. **Design:** Current Integration Constraints, §§1, 6–7, 10–11. **Focused validation:** Run archive/static checker against runtime, manifests, startup and tests; prove no old provider/playback command remains reachable from voice.
  - [ ]* 14.3 Write replacement-failure zero-legacy-fallback property coverage
    - **Targets:** `Core/tests/voice/test_no_legacy_fallback_properties.py`.
    - **Property 2: Replacement failures cannot select legacy behavior.** Generate at least 100 terminal failures across capture, ASR, IPC, Pipecat, local LLM, tool, TTS, renderer, memory, and eligible cloud stages; assert only unavailable/retry/error/cancel/text outcomes and no legacy import, execution, route, suggestion, or user-selected fallback.
    - **Dependencies:** 12.1–12.2, 14.1–14.2. **Validates:** Requirements 1.6, 4.9, 6.7, 8.7. **Design:** Property 2; `V-ARCHIVE`, fault-injection matrix. **Focused validation:** Execute with import/subprocess/provider spies and enforce archive checker in the test command.

- [ ] 15. Automate final test, benchmark, and CI release-readiness gates
  - [ ] 15.1 Implement the voice CI matrix and deterministic validation runners
    - **Targets:** `.github/workflows/voice-runtime.yml` (or repository-equivalent CI), `Core/tests/voice/` test markers/configuration, benchmark runner configuration.
    - Wire archive policy, Python lint/type/unit/integration/property suites, Swift protocol/native tests, and dependency-lock verification into CI. Separate macOS hardware/entitlement and Nominal_Test_Profile performance jobs from portable mocks; configure every property suite for at least 100 iterations and preserve failure artifacts without audio/transcript content.
    - **Dependencies:** 1.3, 2.1, 3.4, 4.4, 5.4, 6.3–6.5, 7.3–7.4, 8.2, 9.3–9.4, 10.2–10.3, 11.2–11.3, 12.2, 14.3. **Requirements:** 1.1–1.6, 2.1–2.6, 3.1–3.8, 4.1–4.9, 5.1–5.8, 6.1–6.7, 7.1–7.8, 8.1–8.7, 9.1–9.6, 10.1–10.6. **Design:** §11 Testing Strategy. **Focused validation:** Confirm job partitioning, no test route can execute archive content, and failed jobs report only safe structured diagnostics.
  - [ ]* 15.2 Add the Nominal_Test_Profile responsiveness/resource release verification **[hardware/macOS gate]**
    - **Targets:** `Core/tests/voice/test_resource_benchmark.py`, CI hardware-job configuration.
    - On warmed macOS 14+ Apple M2/8 GB hardware with the declared profile, automatically verify model footprint `<2.5 GB`, concurrent pipeline memory `<=5 GB`, and final transcript-to-first assistant PCM `<=1.5 s`; exercise drain/release/re-admission and record profile/measurement metadata locally.
    - **Dependencies:** 4.4, 7.1–7.2, 9.2, 12.1. **Validates:** Requirements 9.1–9.5. **Design:** §6, §11; `V-RESOURCE`. **Focused validation:** Fail closed on unavailable profile/measurement quality and never change routes to cloud/legacy to satisfy the benchmark.
  - [ ]* 15.3 Run the complete automated release-readiness suite and enforce final static inventory checks
    - **Targets:** CI release workflow and `tools/check_voice_archive.py` invocation configuration.
    - Run all archive, protocol, mock integration, property, native macOS, performance, privacy, dependency, and fault-injection gates after cutover; regenerate inventory in check-only mode and verify the release bundle/package excludes `legacy_pipeline_backup/`.
    - **Dependencies:** 14.1–14.3, 15.1–15.2. **Validates:** Requirements 1.1–1.6, 2.1–2.6, 3.1–3.8, 4.1–4.9, 5.1–5.8, 6.1–6.7, 7.1–7.8, 8.1–8.7, 9.1–9.6, 10.1–10.6. **Design:** §11. **Focused validation:** Require green automated results, static zero coupling, exact dependency pins, privacy-safe artifacts, and all hardware gates before marking the migration ready.

- [ ] 16. Final checkpoint — verify release readiness
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional in the task UI, but they are the required automated validation work for a production release and must be included in the final release-readiness suite.
- **[hardware/macOS gate]** tasks require macOS 14+ audio/model hardware and the stated Nominal_Test_Profile; they are intentionally separate from portable unit/property CI to prevent flaky substitutions.
- Every implementation and test task must preserve the absolute rule: no legacy runtime fallback, no archive execution, no user/developer compatibility switch, and no raw microphone audio in the transcript/control UDS or default diagnostics.
- The archive is a static auditable reference only. Cutover happens only in Task 14, after the archive policy, replacement implementation, and fault-injection coverage are in place.
- The dependency graph schedules only incomplete leaf tasks. Completed `[x]` prerequisites are resolved before wave 0; `[~]` remains active work, and `[ ]`/`[ ]*` leaves remain pending. Direct dependencies in each task remain authoritative.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.4", "3.4", "4.4", "5.4", "6.1", "7.3", "10.1", "11.3"] },
    { "id": 1, "tasks": ["6.2", "6.3", "7.2", "10.2", "10.3", "11.2"] },
    { "id": 2, "tasks": ["6.4", "6.5", "7.4", "8.1", "9.1"] },
    { "id": 3, "tasks": ["8.2", "9.2"] },
    { "id": 4, "tasks": ["9.3", "9.4"] },
    { "id": 5, "tasks": ["12.1"] },
    { "id": 6, "tasks": ["12.2", "15.2"] },
    { "id": 7, "tasks": ["14.1"] },
    { "id": 8, "tasks": ["14.2"] },
    { "id": 9, "tasks": ["14.3"] },
    { "id": 10, "tasks": ["15.1"] },
    { "id": 11, "tasks": ["15.3"] }
  ]
}
```
