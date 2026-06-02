# Implementation Plan: HAKI — Personal AI Assistant

## Overview

This plan converts the HAKI design into incremental, dependency-ordered coding tasks across the two-process hybrid architecture: a native **Swift / SwiftUI macOS app shell** ("Body", owns TCC permissions, capture, OS actuators, audio I/O, UI, notifications, secure store) and a local **Python orchestration service** ("Mind"/Core, owns the orchestrator, Model Provider, RAG/memory, learning, planner, dialogue) communicating over a streaming gRPC channel on a UNIX domain socket.

Tasks follow the design's 7-phase rollout (Phase 0 Foundations → Phase 6 Creative & automations). Each task builds on prior tasks and ends by wiring new code into the running system; there is no orphaned code. Each task cites the requirements and/or design components it implements.

### Testing conventions (applied to every test sub-task)

- **Property-based tests** implement the design's 76 Correctness Properties, one property per single property test. Python Core properties use **Hypothesis**; Swift shell logic properties use **SwiftCheck**.
- Every property test runs a **minimum of 100 iterations** and is tagged with a comment of the form: `Feature: haki-personal-ai-assistant, Property {number}: {property_text}`.
- Model-backed capabilities (STT, LLM, TTS, mood, image, embeddings) are **mocked/stubbed** inside property tests so routing, gating, ordering, filtering, and atomicity logic is exercised cheaply.
- **Unit/example tests** cover concrete or subjective behaviors; **integration tests** cover OS/app/website wiring; **performance tests** assert the real-time latency budgets; **smoke tests** cover setup/configuration.
- Sub-tasks marked `*` are optional test tasks and are not implemented by the implementation agent automatically.

## Tasks

### Phase 0 — Foundations

- [x] 1. Set up two-process project structure and streaming IPC
  - [x] 1.1 Scaffold the Swift / SwiftUI app shell project
    - Create the `HAKI.app` Xcode/SwiftPM project (menu-bar app target), folder layout for shell subsystems, and the build/signing config that bundles the Core as a child process
    - Add the SwiftCheck test target
    - _Design: Architecture (Swift shell). Requirements: 20.1_
  - [x] 1.2 Scaffold the Python Core service project
    - Create the Core package layout (orchestrator, model_provider, memory, learning, planner, dialogue modules), dependency/venv setup, and the Hypothesis + pytest test harness
    - _Design: Architecture (local Python core)_
  - [x] 1.3 Define the gRPC/IPC contract and streaming transport
    - Author the `.proto` (or JSON-RPC schema) for bidirectional streaming of audio frames, partial/final transcripts, LLM tokens, TTS audio chunks, and control/cancel events
    - Generate Swift and Python stubs
    - _Design: Process & Threading Model, Voice Pipeline. Requirements: 3.1_
  - [x] 1.4 Implement IPC client (Swift) and server (Core) with child-process lifecycle
    - Connect over a UNIX domain socket scoped to the app, manage Core spawn/health/shutdown, and expose a cancellable bidirectional stream API on both sides
    - _Design: Architecture, Security Considerations (local IPC only). Requirements: 3.1_
  - [ ]* 1.5 Write integration test for IPC streaming round trip
    - Verify frames/tokens/control events stream and cancel across the socket
    - _Design: Process & Threading Model_

- [x] 2. Implement encrypted app store and secret handling
  - [x] 2.1 Implement the SQLite app store and Settings/PrivacyState models
    - Create schema/migrations for tasks, automations, settings, reminders, dismissed-suggestion state; implement `Settings` and `PrivacyState` models
    - _Design: Data Models (App store, Settings & Privacy). Requirements: 4.2, 6.3, 8.4, 16.6, 2.4_
  - [x] 2.2 Implement Keychain-backed secret references
    - Store OAuth tokens and API keys in the macOS Keychain referenced by `keyRef`; never persist secrets in the vault, notes, logs, or plan state
    - _Design: Security Considerations (secret handling). Requirements: 20.2_
  - [ ]* 2.3 Write unit tests for settings persistence and secret handling
    - Round-trip settings; assert secrets are referenced by handle, never stored in plaintext stores
    - _Design: Data Models, Security Considerations_

- [x] 3. Implement the Clock subsystem (Req 14)
  - [x] 3.1 Implement Clock date/time/timezone provider and unavailability path
    - Provide `now()` returning date/time/timezone or an `Unavailable` result surfaced to subsystems and the user
    - _Design: Clock. Requirements: 14.1, 14.2, 14.3, 14.5_
  - [x] 3.2 Implement the timezone-change watcher and propagation
    - Detect device timezone changes and propagate updated timezone to subscribers within 5 s
    - _Design: Clock. Requirements: 14.4_
  - [ ]* 3.3 Write unit tests for clock accessors and unavailability
    - Cover now() values and the time-unavailable messaging path
    - _Design: Clock. Requirements: 14.1, 14.2, 14.3, 14.5_
  - [ ]* 3.4 Write integration test for timezone-change propagation
    - Simulate a timezone change and assert propagation within budget
    - _Design: Clock. Requirements: 14.4_

- [x] 4. Implement the Permission_Manager (Reqs 2, 21.15)
  - [x] 4.1 Implement permission status/request and capability dependency mapping
    - Wrap TCC status/request for Screen Recording, Accessibility, Automation; implement `missingFor(capability)` and the System Settings deep-link/guidance messaging within 2 s
    - _Design: Permission_Manager. Requirements: 2.1, 2.2, 2.6, 21.15_
  - [x] 4.2 Implement the screen-access user toggle and revocation watcher
    - Add the always-reachable `screenAccessEnabled` toggle and a watcher that disables dependent capabilities within 5 s on revocation
    - _Design: Permission_Manager. Requirements: 2.3, 2.4, 2.5, 2.7_
  - [ ]* 4.3 Write property test for permission gating and messaging
    - **Property 4: Permission gating and messaging** (SwiftCheck)
    - **Validates: Requirements 2.2, 2.3, 21.15**
  - [ ]* 4.4 Write property test for permission-to-capability dependency mapping
    - **Property 5: Permission-to-capability dependency mapping** (SwiftCheck)
    - **Validates: Requirements 2.6**
  - [ ]* 4.5 Write property test for the screen-access toggle gate
    - **Property 6: Screen-access toggle gate** (SwiftCheck)
    - **Validates: Requirements 2.5**
  - [ ]* 4.6 Write integration test for permission prompts and revocation propagation
    - Exercise prompt flow and revocation-driven capability disablement
    - _Requirements: 2.1, 2.7_

- [x] 5. Implement the Model_Provider abstraction skeleton (Req 20)
  - [x] 5.1 Implement the ModelProvider registry, CapabilityConfig, and per-invocation mode resolution
    - Read `CapabilityConfig` at the start of each invocation so a mode change applies on the next call; expose `invoke`/`invokeStream`
    - _Design: Model Provider Abstraction. Requirements: 20.2, 20.3, 20.4_
  - [x] 5.2 Implement one local + one API backend stub per capability with disclosure and failure handling
    - Wire local and API backends for STT/LLM/TTS/mood/image/embeddings; gate first API use on disclosure acknowledgement; on API-unavailable or local-load-failure inform the user with no silent fallback
    - _Design: Model Provider Abstraction (backend recommendations, mode-switching). Requirements: 20.5, 20.6, 20.7_
  - [ ]* 5.3 Write property test for mode applied at invocation time
    - **Property 62: Mode applied at invocation time** (Hypothesis)
    - **Validates: Requirements 20.3**
  - [ ]* 5.4 Write property test for API disclosure precedes first external use
    - **Property 63: API disclosure precedes first external use** (Hypothesis)
    - **Validates: Requirements 20.5**
  - [ ]* 5.5 Write property test for no silent fallback between processing modes
    - **Property 64: No silent fallback between processing modes** (Hypothesis)
    - **Validates: Requirements 20.6, 20.7**
  - [ ]* 5.6 Write smoke tests for macOS run and per-capability mode setting
    - Assert the app runs on macOS and the per-capability mode control is present/reachable
    - _Requirements: 20.1, 20.2_

- [-] 6. Checkpoint - Phase 0 foundations
  - Ensure all tests pass, ask the user if questions arise.

### Phase 1 — Voice spine

- [ ] 7. Implement the Voice_Engine streaming pipeline (Req 3)
  - [ ] 7.1 Implement Swift audio I/O, realtime VAD, end-of-speech and barge-in detection
    - AVAudioEngine mic tap producing 20 ms frames; on-thread VAD detecting 800 ms end-of-speech and ≥200 ms barge-in; acoustic echo cancellation on the mic path
    - _Design: Voice Pipeline. Requirements: 3.2, 3.3_
  - [ ] 7.2 Implement streaming STT through the Model Provider with partial transcripts and failure handling
    - Stream speech frames to STT, emit partial/final transcripts plus audio features; on no recognizable speech, prompt repeat and dispatch nothing
    - _Design: Voice Pipeline, Voice_Engine. Requirements: 3.4, 3.6_
  - [ ] 7.3 Implement sentence-chunked streaming TTS with cancellation and text fallback
    - Segment LLM token stream into clauses, play first audio while the rest generates; cancel on barge-in; on TTS failure render response as on-screen text and notify
    - _Design: Voice Pipeline, Voice_Engine. Requirements: 3.1, 3.5, 3.7_
  - [ ]* 7.4 Write property test for unrecognized speech is never dispatched
    - **Property 7: Unrecognized speech is never dispatched** (Hypothesis)
    - **Validates: Requirements 3.6**
  - [ ]* 7.5 Write property test for TTS failure falls back to text
    - **Property 8: TTS failure falls back to text** (Hypothesis)
    - **Validates: Requirements 3.7**
  - [ ]* 7.6 Write performance tests for voice latency budgets
    - First-words→first-audio ≤ 300 ms; end-of-speech→playback ≤ 1.5 s; barge-in stop ≤ 200 ms (measured on Apple Silicon)
    - _Design: Voice Pipeline (latency budget). Requirements: 3.1, 3.2, 3.3_
  - [ ]* 7.7 Write integration test for STT/TTS model wiring on representative audio/text
    - Verify capture→transcript and text→audio behavior end-to-end with representative inputs
    - _Requirements: 3.4, 3.5_

- [ ] 8. Implement the Language_Engine and Hinglish handling (Req 5)
  - [ ] 8.1 Implement language composition analysis and per-token origin tagging
    - Tokenize and tag each token's origin (script + lexicon heuristics + model); classify composition as hindi/english/hinglish/unknown; accept without a language picker; on uninterpretable input, not-understood + prompt rephrase
    - _Design: Language_Engine, Voice Pipeline (Hinglish). Requirements: 5.1, 5.5_
  - [ ] 8.2 Implement generation language constraints
    - Build the response-language constraint for the LLM prompt: Hinglish in → mixed out (≥1 Hindi-origin, ≥1 English-origin, never fully Hindi); monolingual in → same language out
    - _Design: Language_Engine. Requirements: 5.2, 5.3_
  - [ ] 8.3 Wire per-token origin map into TTS pronunciation routing
    - Pass the origin map with the response so each Hindi-origin token uses the Hindi voice and each English-origin token uses the English voice
    - _Design: Voice Pipeline (pronunciation). Requirements: 5.4_
  - [ ]* 8.4 Write property test for language acceptance
    - **Property 11: Language acceptance** (Hypothesis)
    - **Validates: Requirements 5.1**
  - [ ]* 8.5 Write property test for Hinglish response composition
    - **Property 12: Hinglish response composition** (Hypothesis)
    - **Validates: Requirements 5.2**
  - [ ]* 8.6 Write property test for monolingual response composition
    - **Property 13: Monolingual response composition** (Hypothesis)
    - **Validates: Requirements 5.3**
  - [ ]* 8.7 Write property test for per-word pronunciation routing
    - **Property 14: Per-word pronunciation routing** (Hypothesis)
    - **Validates: Requirements 5.4**
  - [ ]* 8.8 Write unit test for the uninterpretable-language edge
    - Cover the not-understood + rephrase path
    - _Requirements: 5.5_

- [ ] 9. Implement the Mood_Detector (Req 4)
  - [ ] 9.1 Implement prosodic mood classification with confidence and duration gating
    - Classify one primary mood from {angry, sad, happy, neutral} with confidence in [0.0,1.0] from pitch/volume features via the Model Provider; clips < 1 s return unclassifiable; emit exactly one result per request
    - _Design: Mood_Detector. Requirements: 4.1, 4.7, 4.8_
  - [ ]* 9.2 Write property test for mood classification output contract
    - **Property 9: Mood classification output contract** (Hypothesis)
    - **Validates: Requirements 4.1, 4.7, 4.8**
  - [ ]* 9.3 Write unit test for the mood threshold default and range validation
    - Default 0.6, configurable within [0.0,1.0]
    - _Requirements: 4.2_

- [ ] 10. Implement the Persona_Engine (Reqs 6, 4.3–4.6)
  - [ ] 10.1 Implement identity/intensity system-prompt shaping and mood/memory tone integration
    - Apply consistent HAKI identity at ≥3 ordered intensity levels; integrate mood and memory context into tone; at minimum intensity prefer conciseness; proceed with whatever inputs are available
    - _Design: Persona_Engine. Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  - [ ] 10.2 Implement the mood-to-tone mapping function
    - Pure mapping over (mood, confidence, threshold): angry≥t→calming, sad≥t→encouraging, otherwise/below-threshold/unclassifiable→neutral
    - _Design: Persona_Engine. Requirements: 4.3, 4.4, 4.5, 4.6_
  - [ ]* 10.3 Write property test for mood-to-tone mapping
    - **Property 10: Mood-to-tone mapping** (Hypothesis)
    - **Validates: Requirements 4.3, 4.4, 4.5, 4.6**
  - [ ]* 10.4 Write property test for response produced regardless of optional inputs
    - **Property 15: Response produced regardless of optional inputs** (Hypothesis)
    - **Validates: Requirements 6.5**
  - [ ]* 10.5 Write unit tests for personality identity, intensity, and conciseness
    - Spot-check identity presence and min-intensity conciseness
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 11. Implement the Orchestrator turn loop and intent routing
  - [ ] 11.1 Implement the cancellable turn loop with parallel mood/language/memory dispatch
    - Sequence Voice_Engine → parallel(Mood, Language, Memory placeholder) → intent classification → capability dispatch → Persona shaping → TTS; cancellable at every await for barge-in
    - _Design: The Orchestrator, Intent Routing. Requirements: 3.1, 4.8, 5.1, 6.5_
  - [ ] 11.2 Wire intent routing to subsystem entry points
    - Classify each turn into an intent (chat/recall/read_aloud/mac_command/run_automation/image/schedule/task/meta) and route to the owning subsystem, deferring side effects to the dialogue gate
    - _Design: Intent Routing. Requirements: 6.1_
  - [ ]* 11.3 Write integration test for an end-to-end chat turn
    - Drive a mocked turn from transcript to shaped TTS through the orchestrator
    - _Design: The Orchestrator_

- [ ] 12. Checkpoint - Phase 1 voice spine
  - Ensure all tests pass, ask the user if questions arise.

### Phase 2 — Memory

- [ ] 13. Implement the Memory_Brain vault and Note model (Req 7)
  - [ ] 13.1 Implement the Note model and Obsidian-style Markdown serializer/parser
    - Implement the `Note` and `Chunk` models and the YAML-front-matter Markdown file format (id, timestamps, source, tags, topics, superseded_by, private, body)
    - _Design: Data Models (Note), Memory, RAG & Learning. Requirements: 7.8_
  - [ ] 13.2 Implement vault init and durable note writes with atomicity
    - Initialize the vault directory and empty index on startup even when empty; confirm a store only after a successful durable write; on failure leave no partial note and inform the user
    - _Design: Vault + RAG design. Requirements: 7.1, 7.2, 7.4, 7.5_
  - [ ]* 13.3 Write property test for note serialization round trip
    - **Property 21: Note serialization round trip** (Hypothesis)
    - **Validates: Requirements 7.8**
  - [ ]* 13.4 Write property test for store/retrieve round trip with durable-confirm ordering
    - **Property 16: Store/retrieve round trip with durable-confirm ordering** (Hypothesis)
    - **Validates: Requirements 7.1**
  - [ ]* 13.5 Write property test for store failure atomicity
    - **Property 17: Store failure atomicity** (Hypothesis)
    - **Validates: Requirements 7.2**
  - [ ]* 13.6 Write property test for notes persist across restart
    - **Property 19: Notes persist across restart** (Hypothesis)
    - **Validates: Requirements 7.5**
  - [ ]* 13.7 Write smoke test for empty-vault initialization
    - Assert vault + empty index are created when no notes exist
    - _Requirements: 7.4_

- [ ] 14. Implement RAG indexing and retrieval (Req 7)
  - [ ] 14.1 Implement chunk/embed indexing into the local vector index
    - Chunk notes, embed via the embeddings ModelProvider, store in a rebuildable local vector index sidecar
    - _Design: Vault + RAG design (indexing). Requirements: 7.3_
  - [ ] 14.2 Implement hybrid retrieval with term/topic filtering and superseded exclusion
    - Combine vector similarity with a term/topic filter so results share ≥1 matching term/topic, exclude non-matching and superseded notes, return within 2 s; implement "what do you know about X"
    - _Design: Vault + RAG design (retrieval). Requirements: 7.3, 7.7_
  - [ ] 14.3 Wire Memory_Brain retrieval into the orchestrator turn loop
    - Replace the Phase-1 memory placeholder with real retrieval feeding Persona context
    - _Design: The Orchestrator. Requirements: 7.3_
  - [ ]* 14.4 Write property test for retrieval term/topic filtering excludes non-matching and superseded notes
    - **Property 18: Retrieval term/topic filtering excludes non-matching and superseded notes** (Hypothesis)
    - **Validates: Requirements 7.3, 7.7, 8.3**
  - [ ]* 14.5 Write performance test for memory retrieval latency
    - Retrieval and topic queries return within 2 s
    - _Requirements: 7.3, 7.7_

- [ ] 15. Implement memory deletion, export, and privacy controls (Reqs 7, 9)
  - [ ] 15.1 Implement single-note delete, delete-all, and export with failure atomicity
    - `forget(noteId)`, `forgetAll()`, and `export()` to a single user-accessible file; confirm only on success; on failure leave data intact / produce no partial file and inform the user
    - _Design: Vault + RAG design (delete/export). Requirements: 7.6, 9.3, 9.4, 9.5, 9.6, 9.8_
  - [ ] 15.2 Implement local-only storage guarantee and the privacy-designation control
    - Ensure notes never leave the device; add the always-accessible "designate conversation private" control writing `PrivacyState`
    - _Design: Settings & Privacy, Security Considerations. Requirements: 9.2, 9.7_
  - [ ]* 15.3 Write property test for deletion removes and confirms
    - **Property 20: Deletion removes and confirms** (Hypothesis)
    - **Validates: Requirements 7.6, 8.5**
  - [ ]* 15.4 Write property test for export completeness round trip
    - **Property 27: Export completeness round trip** (Hypothesis)
    - **Validates: Requirements 9.3**
  - [ ]* 15.5 Write property test for delete-all empties the vault
    - **Property 28: Delete-all empties the vault** (Hypothesis)
    - **Validates: Requirements 9.5**
  - [ ]* 15.6 Write property test for deletion/export failure atomicity
    - **Property 29: Deletion/export failure atomicity** (Hypothesis)
    - **Validates: Requirements 9.6, 9.8**
  - [ ]* 15.7 Write smoke/integration tests for privacy control and data locality
    - Privacy-designation control reachable; assert no network egress for a capability in local mode
    - _Requirements: 9.2, 9.7, 20.4_

- [ ] 16. Implement the Learning_Engine (Reqs 8, 9.1)
  - [ ] 16.1 Implement conversation-end detection and durable-item extraction with privacy gate
    - Trigger on explicit end or 300 s idle; skip private conversations; extract durable facts/preferences via the LLM; record learning incomplete when nothing is extractable
    - _Design: Autonomous Learning loop. Requirements: 8.1, 8.6, 9.1_
  - [ ] 16.2 Implement conflict supersede and per-item write atomicity
    - On a conflicting value, write a new note and mark exactly the prior note superseded; on per-item write failure retain no partial note, leave prior notes unchanged, record that item incomplete
    - _Design: Autonomous Learning loop. Requirements: 8.2, 8.3, 8.7_
  - [ ] 16.3 Implement the recently-learned record and mark-incorrect correction
    - Tag items with `learned_session`; query learned items over a configurable 1–90 day window (default 7); mark-incorrect removes and confirms
    - _Design: Autonomous Learning loop. Requirements: 8.4, 8.5_
  - [ ]* 16.4 Write property test for conflict supersede
    - **Property 22: Conflict supersede** (Hypothesis)
    - **Validates: Requirements 8.2**
  - [ ]* 16.5 Write property test for recently-learned window filter
    - **Property 23: Recently-learned window filter** (Hypothesis)
    - **Validates: Requirements 8.4**
  - [ ]* 16.6 Write property test for no-extraction leaves notes unchanged
    - **Property 24: No-extraction leaves notes unchanged** (Hypothesis)
    - **Validates: Requirements 8.6**
  - [ ]* 16.7 Write property test for per-item learning write atomicity
    - **Property 25: Per-item learning write atomicity** (Hypothesis)
    - **Validates: Requirements 8.7**
  - [ ]* 16.8 Write property test for private conversations write nothing
    - **Property 26: Private conversations write nothing** (Hypothesis)
    - **Validates: Requirements 9.1**
  - [ ]* 16.9 Write unit test for durable-item extraction over example conversations
    - Extract known facts from example transcripts
    - _Requirements: 8.1_

- [ ] 17. Checkpoint - Phase 2 memory
  - Ensure all tests pass, ask the user if questions arise.

### Phase 3 — Read & comprehend

- [ ] 18. Implement the Screen_Reader capture and read-aloud (Req 1)
  - [ ] 18.1 Implement layered content capture with OCR fallback
    - AX focused-window text in reading order (primary); PDFKit extraction for PDFs; ScreenCaptureKit + Vision OCR fallback when selectable text is empty/errors or content is image-only; resolve named apps and decline when not running/found
    - _Design: Screen_Reader (capture strategy). Requirements: 1.1, 1.2, 1.3, 1.4, 1.7, 1.9_
  - [ ] 18.2 Implement the read-aloud playback handoff and ordered command queue
    - Hand captured content to the Voice_Engine; process pause/resume/stop in receipt order with stop>pause>resume priority within a 200 ms window; when no text after extraction+OCR, do not play and inform the user
    - _Design: Screen_Reader (playback control). Requirements: 1.5, 1.6, 1.8_
  - [ ] 18.3 Wire Screen_Reader to Permission_Manager and the screen-access toggle
    - Gate capture on Screen Recording/Accessibility permission and the user toggle; decline with guidance when blocked
    - _Design: Screen_Reader, Permission_Manager. Requirements: 2.5_
  - [ ]* 18.4 Write property test for screen-read OCR fallback selection
    - **Property 1: Screen-read OCR fallback selection** (SwiftCheck)
    - **Validates: Requirements 1.3, 1.4**
  - [ ]* 18.5 Write property test for no playback without text
    - **Property 2: No playback without text** (SwiftCheck)
    - **Validates: Requirements 1.6**
  - [ ]* 18.6 Write property test for playback command ordering and priority
    - **Property 3: Playback command ordering and priority** (SwiftCheck)
    - **Validates: Requirements 1.5, 1.8**
  - [ ]* 18.7 Write unit tests for reading-order PDF extraction and named-app edges
    - Known-document reading order; named-app capture and unavailable-app edge
    - _Requirements: 1.2, 1.7, 1.9_
  - [ ]* 18.8 Write performance test for capture-to-playback latency
    - ≤ 3 s for ≤ 10,000 characters
    - _Requirements: 1.1_

- [ ] 19. Implement the Text_Assistant (Req 16)
  - [ ] 19.1 Implement AX field observation with confidence-gated inline correction
    - Observe supported input fields via Accessibility; apply a spelling correction inline only at/above the configured confidence threshold
    - _Design: Text_Assistant. Requirements: 16.1_
  - [ ] 19.2 Implement single context-aware completion with dismissal memory and disabled inertness
    - Offer at most one completion from Memory_Brain + recent input on 500 ms pause or focus-without-typing; insert on accept; never re-offer a dismissed suggestion for the same input state; when disabled do no background work
    - _Design: Text_Assistant. Requirements: 16.2, 16.3, 16.4, 16.5, 16.6_
  - [ ]* 19.3 Write property test for confidence-gated inline correction
    - **Property 46: Confidence-gated inline correction** (SwiftCheck)
    - **Validates: Requirements 16.1**
  - [ ]* 19.4 Write property test for at most one completion suggestion
    - **Property 47: At most one completion suggestion** (SwiftCheck)
    - **Validates: Requirements 16.2**
  - [ ]* 19.5 Write property test for dismissed suggestions are not re-offered
    - **Property 48: Dismissed suggestions are not re-offered** (SwiftCheck)
    - **Validates: Requirements 16.5**
  - [ ]* 19.6 Write property test for disabled Text_Assistant is inert
    - **Property 49: Disabled Text_Assistant is inert** (SwiftCheck)
    - **Validates: Requirements 16.6**
  - [ ]* 19.7 Write unit test for completion insertion at cursor
    - Accepted completion inserts at the cursor position
    - _Requirements: 16.4_

- [ ] 20. Checkpoint - Phase 3 read & comprehend
  - Ensure all tests pass, ask the user if questions arise.

### Phase 4 — Agentic core

- [ ] 21. Implement the planner and Command_Plan model (Req 21)
  - [ ] 21.1 Implement the CommandPlan/Step data model and dependency graph
    - Implement `CommandPlan` and `Step` (intent, actuator, args, dependsOn, classification, requiredSlots, status) with the dependency-graph semantics for ordering and parallelism
    - _Design: Data Models (CommandPlan & Step). Requirements: 17.2, 21.1_
  - [ ] 21.2 Implement the LLM planner with memory-backed slot filling
    - Convert a natural-language command into an ordered, dependency-aware plan; annotate each step with actuator and safety classification; fill slots that reference stored facts from Memory_Brain instead of prompting
    - _Design: Planning. Requirements: 21.1, 21.7_
  - [ ]* 21.3 Write property test for execution respects step dependencies and order
    - **Property 51: Execution respects step dependencies and order** (Hypothesis)
    - **Validates: Requirements 17.2**
  - [ ]* 21.4 Write property test for memory-backed slot filling
    - **Property 65: Memory-backed slot filling** (Hypothesis)
    - **Validates: Requirements 21.7**

- [ ] 22. Implement the Safety_Gate (Req 22)
  - [ ] 22.1 Implement action classification and confirmation gating
    - Classify steps CONSEQUENTIAL/REVERSIBLE/UNKNOWN; require confirmation describing the action before consequential/unknown steps; run reversible steps without confirmation; treat unknown as consequential
    - _Design: Safety_Gate. Requirements: 22.1, 22.4, 22.7_
  - [ ] 22.2 Implement mid-plan pause/confirm/resume and no-response handling
    - Pause at a consequential step reached mid-plan, request confirmation, resume only on confirm; on rejection or no response, do not perform the step and stop dependents, reporting completed vs not
    - _Design: Safety_Gate, Execution loop. Requirements: 22.2, 22.3, 22.5, 22.6, 22.8_
  - [ ]* 22.3 Write property test for consequential gating vs reversible pass-through
    - **Property 69: Consequential actions are gated by confirmation; reversible actions are not** (Hypothesis)
    - **Validates: Requirements 22.1, 22.2, 22.3, 22.4, 22.5, 22.7**
  - [ ]* 22.4 Write property test for no confirmation response means no execution
    - **Property 70: No confirmation response means no execution** (Hypothesis)
    - **Validates: Requirements 22.8**

- [ ] 23. Implement the Execution_Engine (Reqs 17, 21)
  - [ ] 23.1 Implement the plan→gate→execute→verify loop with parallelism and postcondition checks
    - Execute ready steps respecting dependencies, run independent steps in parallel, verify postconditions, report executed steps on completion
    - _Design: Execution loop. Requirements: 17.2, 21.8_
  - [ ] 23.2 Implement cancellation and failure/rejection propagation
    - Cancel stops unstarted steps and interrupts the in-progress step within the bound, partitioning the report; failure/rejection/unreachable-site/missing-element/app-not-installed stops transitive dependents while independent steps continue, reporting completed/not-performed/failed-step-with-reason
    - _Design: ExecutionEngine, Execution loop. Requirements: 17.5, 17.6, 17.7, 21.9, 21.12, 21.13, 21.14_
  - [ ]* 23.3 Write property test for cancellation stops unstarted steps and partitions the report
    - **Property 53: Cancellation stops unstarted steps and partitions the report** (Hypothesis)
    - **Validates: Requirements 17.6**
  - [ ]* 23.4 Write property test for failure and rejection propagation
    - **Property 54: Failure and rejection propagation** (Hypothesis)
    - **Validates: Requirements 17.7, 21.9, 21.12, 21.13, 21.14, 22.6**
  - [ ]* 23.5 Write property test for completed plan reports executed steps
    - **Property 66: Completed plan reports executed steps** (Hypothesis)
    - **Validates: Requirements 21.8**

- [ ] 24. Implement the Mac_Controller actuators (Req 21)
  - [ ] 24.1 Implement app launch/focus and contact-resolution actuators
    - NSWorkspace/open/AppleScript launch+frontmost within budget; open closed apps before dependents; AppleScript/AX message-send and call-placement; never act on a wrong/ambiguous contact — prompt to disambiguate
    - _Design: Mac_Controller (actuation backends). Requirements: 21.2, 21.3, 21.4, 21.10, 21.11, 21.16_
  - [ ] 24.2 Implement the Arc/CDP web actuator and Vision computer-use fallback
    - Open result tabs, navigate, click elements, fill forms via Chrome DevTools Protocol over loopback; when no AX/CDP selector exists, use the OCR+click vision loop gated by the Safety_Gate
    - _Design: Mac_Controller, Security Considerations (computer-use containment). Requirements: 21.5, 21.6, 21.12, 21.13_
  - [ ] 24.3 Wire Mac_Controller to Permission_Manager for control permissions
    - Without Automation/Accessibility permission, do not run control steps; inform which permission is missing, which steps are unavailable, and how to grant it
    - _Design: Mac_Controller (permissions). Requirements: 21.15_
  - [ ]* 24.4 Write property test for closed required app is opened before dependents
    - **Property 67: Closed required app is opened before dependents** (Hypothesis)
    - **Validates: Requirements 21.10**
  - [ ]* 24.5 Write property test for never act on the wrong contact
    - **Property 68: Never act on the wrong contact** (Hypothesis)
    - **Validates: Requirements 21.11, 21.16**
  - [ ]* 24.6 Write integration tests for Mac control wiring
    - App launch/focus, message send, call placement, opening result tabs in Arc via CDP, website operation
    - _Requirements: 21.2, 21.3, 21.4, 21.5, 21.6_

- [ ] 25. Implement the Dialogue_Manager (Req 23)
  - [ ] 25.1 Implement memory-first slot assessment and pre-execution gating
    - Assess sufficiency/missing slots; resolve missing slots from Memory_Brain first; ask only for genuinely missing slots; do not start execution until required slots are resolved and do not re-ask resolved slots
    - _Design: Dialogue_Manager integration. Requirements: 23.1, 23.2, 23.4, 23.5_
  - [ ] 25.2 Implement mid-task pause/resume, declines, and candidate-choice presentation
    - Pause on mid-task decision points retaining state and resume without re-executing completed steps; declined-with-default proceeds and reports the default; declined-without-default abandons only the dependent step; present gathered candidates without auto-selecting
    - _Design: Dialogue_Manager integration. Requirements: 23.3, 23.6, 23.7, 23.8, 23.9_
  - [ ] 25.3 Wire the Dialogue_Manager into the orchestrator and execution loop
    - Route the intent-routing ambiguity gate and per-step slot resolution through the Dialogue_Manager
    - _Design: Intent Routing, Execution loop. Requirements: 23.1_
  - [ ]* 25.4 Write property test for no execution until required slots are resolved
    - **Property 71: No execution until required slots are resolved, and resolved slots are not re-asked** (Hypothesis)
    - **Validates: Requirements 23.1, 23.4, 23.5**
  - [ ]* 25.5 Write property test for clarification is memory-first
    - **Property 72: Clarification is memory-first** (Hypothesis)
    - **Validates: Requirements 23.2**
  - [ ]* 25.6 Write property test for mid-task clarification pauses and resumes without re-execution
    - **Property 73: Mid-task clarification pauses and resumes without re-execution** (Hypothesis)
    - **Validates: Requirements 23.3, 23.9**
  - [ ]* 25.7 Write property test for declined slot with a default proceeds using the default
    - **Property 74: Declined slot with a default proceeds using the default** (Hypothesis)
    - **Validates: Requirements 23.6**
  - [ ]* 25.8 Write property test for declined slot without a default abandons only the dependent step
    - **Property 75: Declined slot without a default abandons only the dependent step** (Hypothesis)
    - **Validates: Requirements 23.7**
  - [ ]* 25.9 Write property test for candidate choices are never auto-selected
    - **Property 76: Candidate choices are never auto-selected** (Hypothesis)
    - **Validates: Requirements 23.8**

- [ ] 26. Checkpoint - Phase 4 agentic core
  - Ensure all tests pass, ask the user if questions arise.

### Phase 5 — Productivity

- [ ] 27. Implement the Comms_Reader (Req 10)
  - [ ] 27.1 Implement account connect/disconnect and message polling for WhatsApp and email
    - Per-account grant/revoke controls; WhatsApp via AX/CDP of the desktop app or web; email via IMAP/Gmail API; begin reading on grant, stop within 5 s on revoke
    - _Design: Comms_Reader (integration approach). Requirements: 10.5, 10.8, 10.9_
  - [ ] 27.2 Implement Actionable_Item extraction and clarification flagging
    - Implement the `ActionableItem` model; LLM extracts date/time/location/description within the 60 s budget; flag items missing explicit date or time for clarification and surface to the user
    - _Design: Comms_Reader, Data Models (ActionableItem). Requirements: 10.1, 10.2, 10.3, 10.4_
  - [ ] 27.3 Implement bounded retry with account-identified failure notification
    - Retry a failing account up to 3 additional times at 30 s intervals, stop early on success, then notify identifying the specific account
    - _Design: Comms_Reader. Requirements: 10.6, 10.7_
  - [ ]* 27.4 Write property test for actionable clarification flagging
    - **Property 30: Actionable clarification flagging** (Hypothesis)
    - **Validates: Requirements 10.4**
  - [ ]* 27.5 Write property test for bounded comms retries with account-identified failure
    - **Property 31: Bounded comms retries with account-identified failure** (Hypothesis)
    - **Validates: Requirements 10.6, 10.7**
  - [ ]* 27.6 Write unit test for field extraction from example messages/emails
    - Extract date/time/location/description from known samples
    - _Requirements: 10.3_
  - [ ]* 27.7 Write integration test for comms read and grant/revoke transitions
    - Incoming WhatsApp/email read on grant; reads stop on revoke
    - _Requirements: 10.1, 10.2, 10.8, 10.9_

- [ ] 28. Implement the Scheduler calendar automation (Req 11)
  - [ ] 28.1 Implement event proposal from actionables and the CalendarProposal model
    - Implement `CalendarProposal`; propose an event carrying extracted date/time/description within 5 s of identification
    - _Design: Scheduler, Data Models (CalendarProposal). Requirements: 11.1_
  - [ ] 28.2 Implement confirmation-gated EventKit creation with edit validation and failure atomicity
    - Require explicit confirmation before creating; apply user edits over extracted values; block creation on omitted/invalid date-time and prompt correction retaining other details; on creation failure create no partial event, inform, retain details for retry
    - _Design: Scheduler. Requirements: 11.2, 11.3, 11.4, 11.5, 11.6, 11.7_
  - [ ]* 28.3 Write property test for event proposal carries extracted details
    - **Property 32: Event proposal carries extracted details** (Hypothesis)
    - **Validates: Requirements 11.1**
  - [ ]* 28.4 Write property test for calendar creation requires confirmation and uses confirmed details
    - **Property 33: Calendar creation requires confirmation and uses confirmed details** (Hypothesis)
    - **Validates: Requirements 11.2, 11.3, 11.4, 11.5**
  - [ ]* 28.5 Write property test for invalid edited date/time blocks creation
    - **Property 34: Invalid edited date/time blocks creation** (Hypothesis)
    - **Validates: Requirements 11.6**
  - [ ]* 28.6 Write property test for calendar creation failure atomicity
    - **Property 35: Calendar creation failure atomicity** (Hypothesis)
    - **Validates: Requirements 11.7**
  - [ ]* 28.7 Write performance test for proposal latency
    - Proposal within 5 s of identification
    - _Requirements: 11.1_

- [ ] 29. Implement severity-based reminders (Req 12)
  - [ ] 29.1 Implement task creation with severity assignment and the Reminder/ReminderPolicy models
    - Implement `Task`, `Reminder`, `ReminderPolicy`; assign every task a Severity; indeterminate severity → default + notify
    - _Design: Scheduler, Data Models (Task & Reminder). Requirements: 12.1, 12.11_
  - [ ] 29.2 Implement effective reminder-offset computation using the Clock
    - Compute reminder times = due date + effective offsets (custom-when-valid else defaults: assignment/exam {−7d,−3d}, birthday {−14d,−1d}); fire elapsed-window reminders immediately and schedule future ones normally; all times from the Clock
    - _Design: Scheduler. Requirements: 12.2, 12.3, 12.4, 12.7, 12.8, 12.10_
  - [ ] 29.3 Implement dual-channel issuance, per-reminder failure isolation, and birthday day-of prompt
    - Issue each reminder via Voice_Engine and on-screen notification; a single reminder failure still issues the rest and notifies; prompt on the day of a birthday to confirm wishes sent
    - _Design: Scheduler. Requirements: 12.5, 12.6, 12.9_
  - [ ]* 29.4 Write property test for every task has a severity
    - **Property 36: Every task has a severity** (Hypothesis)
    - **Validates: Requirements 12.1, 12.11**
  - [ ]* 29.5 Write property test for effective reminder offsets
    - **Property 37: Effective reminder offsets** (Hypothesis)
    - **Validates: Requirements 12.2, 12.3, 12.4, 12.7, 12.8**
  - [ ]* 29.6 Write property test for reminders fire on both channels
    - **Property 38: Reminders fire on both channels** (Hypothesis)
    - **Validates: Requirements 12.6**
  - [ ]* 29.7 Write property test for single reminder failure does not block the rest
    - **Property 39: Single reminder failure does not block the rest** (Hypothesis)
    - **Validates: Requirements 12.9**
  - [ ]* 29.8 Write property test for elapsed-window reminders fire immediately
    - **Property 40: Elapsed-window reminders fire immediately** (Hypothesis)
    - **Validates: Requirements 12.10**
  - [ ]* 29.9 Write unit test for the birthday day-of prompt
    - Confirm-wishes prompt fires on the birthday
    - _Requirements: 12.5_

- [ ] 30. Implement the Task_Tracker (Req 13)
  - [ ] 30.1 Implement the task list with sorted incomplete listing and persistence atomicity
    - Maintain tasks with due dates and severity; return incomplete tasks ordered by due date within 2 s; on add failure add no partial entry, inform, retain details for retry
    - _Design: Task_Tracker. Requirements: 13.1, 13.2, 13.7_
  - [ ] 30.2 Implement due-date prompting, completion transitions, and prerequisite tracking
    - Ask within 60 s of a due date passing; mark complete on user report/detection and stop reminders; report incomplete prerequisites on status request
    - _Design: Task_Tracker. Requirements: 13.3, 13.4, 13.5, 13.6_
  - [ ]* 30.3 Write property test for incomplete tasks listed sorted by due date
    - **Property 41: Incomplete tasks listed sorted by due date** (Hypothesis)
    - **Validates: Requirements 13.2**
  - [ ]* 30.4 Write property test for completed tasks stop reminders
    - **Property 42: Completed tasks stop reminders** (Hypothesis)
    - **Validates: Requirements 13.5**
  - [ ]* 30.5 Write property test for incomplete prerequisites reported
    - **Property 43: Incomplete prerequisites reported** (Hypothesis)
    - **Validates: Requirements 13.6**
  - [ ]* 30.6 Write property test for task add failure atomicity
    - **Property 44: Task add failure atomicity** (Hypothesis)
    - **Validates: Requirements 13.7**
  - [ ]* 30.7 Write unit tests for task list invariants and completion transitions
    - Data invariants (13.1) and completion state changes (13.4)
    - _Requirements: 13.1, 13.4_
  - [ ]* 30.8 Write performance tests for task list and due-date prompt latency
    - Task list ≤ 2 s; due-date prompt within 60 s
    - _Requirements: 13.2, 13.3_

- [ ] 31. Checkpoint - Phase 5 productivity
  - Ensure all tests pass, ask the user if questions arise.

### Phase 6 — Creative & automations

- [ ] 32. Implement the Image_Studio (Req 15)
  - [ ] 32.1 Implement image generation and edit-target resolution over session history
    - Generate via the image ModelProvider, display in the UI; maintain session image history so an unspecified edit resolves to the most recent image and an explicit reference resolves to that image
    - _Design: Image_Studio. Requirements: 15.1, 15.2, 15.3_
  - [ ] 32.2 Implement save-with-confirm and failure messaging
    - Save to the designated user-accessible folder and confirm; on save failure keep the image in-session and inform; on generation failure inform with reason
    - _Design: Image_Studio. Requirements: 15.4, 15.5, 15.6_
  - [ ]* 32.3 Write property test for image edit target resolution
    - **Property 45: Image edit target resolution** (Hypothesis)
    - **Validates: Requirements 15.2, 15.3**
  - [ ]* 32.4 Write unit tests for save/confirm, save-failure, and generation-failure messaging
    - Save+confirm path and both failure messages
    - _Requirements: 15.4, 15.5, 15.6_

- [ ] 33. Implement the Automation_Library (Req 17)
  - [ ] 33.1 Implement named-automation storage and the NamedAutomation model
    - Implement `NamedAutomation`; store name + ordered steps; load by name yields equal name and ordered steps
    - _Design: Automation_Library, Data Models (NamedAutomation). Requirements: 17.1, 17.3_
  - [ ] 33.2 Implement exact-name invocation, nearest-name suggestion, and progress/cancel reporting over the Execution_Engine
    - Run by exact name on the shared Execution_Engine; on no exact match run nothing and suggest the nearest stored name; report the currently executing step and support cancel
    - _Design: Automation_Library + Execution_Engine. Requirements: 17.2, 17.4, 17.5, 17.6, 17.7_
  - [ ]* 33.3 Write property test for automation definition round trip
    - **Property 50: Automation definition round trip** (Hypothesis)
    - **Validates: Requirements 17.1, 17.3**
  - [ ]* 33.4 Write property test for unknown automation name does not execute and suggests nearest
    - **Property 52: Unknown automation name does not execute and suggests nearest** (Hypothesis)
    - **Validates: Requirements 17.4**

- [ ] 34. Implement the question-paper analysis built-in automation (Req 18)
  - [ ] 34.1 Implement the question-paper analysis plan steps
    - Require ≥1 paper (else don't start); extract topics per paper; identify topics recurring in ≥2 papers; cross-reference recurring topics against available course content via Memory_Brain/RAG and annotate; present a prioritized chapter list ordered by recurrence; complete only when the list is presented; on partial processing complete over processed papers and report which failed and why
    - _Design: Built-in automations (Question-Paper Analysis). Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 18.6_
  - [ ]* 34.2 Write property test for question-paper analysis requires at least one paper
    - **Property 55: Question-paper analysis requires at least one paper** (Hypothesis)
    - **Validates: Requirements 18.1**
  - [ ]* 34.3 Write property test for recurring topics are those in two or more papers
    - **Property 56: Recurring topics are those in two or more papers** (Hypothesis)
    - **Validates: Requirements 18.2**
  - [ ]* 34.4 Write property test for recurring topics annotated with matching course content
    - **Property 57: Recurring topics annotated with matching course content** (Hypothesis)
    - **Validates: Requirements 18.3**
  - [ ]* 34.5 Write property test for prioritized list ordered by recurrence, computed over processed papers
    - **Property 58: Prioritized list ordered by recurrence, computed over processed papers** (Hypothesis)
    - **Validates: Requirements 18.4, 18.5**

- [ ] 35. Implement the document-humanization built-in automation (Req 19)
  - [ ] 35.1 Implement LaTeX prose/markup separation and prose segmentation
    - Parse LaTeX and separate prose from markup; on unparseable/no-prose don't start and inform with reason; segment prose into 800–1200 word chunks with a possibly-shorter final segment
    - _Design: Built-in automations (Document Humanization). Requirements: 19.1, 19.2, 19.3_
  - [ ] 35.2 Implement segment humanization and atomic write-back preserving markup
    - Humanize each segment via the Language_Engine preserving meaning; write humanized prose back preserving original markup; complete only when every segment is written; on per-segment write failure report not-complete, warn which failed, never overwrite saved segments with unsaved content
    - _Design: Built-in automations, Language_Engine. Requirements: 19.4, 19.5, 19.6_
  - [ ]* 35.3 Write property test for prose segmentation bounds
    - **Property 59: Prose segmentation bounds** (Hypothesis)
    - **Validates: Requirements 19.3**
  - [ ]* 35.4 Write property test for LaTeX markup preservation round trip
    - **Property 60: LaTeX markup preservation round trip** (Hypothesis)
    - **Validates: Requirements 19.1, 19.5**
  - [ ]* 35.5 Write property test for humanization segment write atomicity
    - **Property 61: Humanization segment write atomicity** (Hypothesis)
    - **Validates: Requirements 19.6**
  - [ ]* 35.6 Write unit tests for unparseable-LaTeX edge and meaning-preservation spot check
    - Unparseable/no-prose path (19.2) and humanization meaning preservation (19.4)
    - _Requirements: 19.2, 19.4_

- [ ] 36. Final integration and wiring
  - [ ] 36.1 Wire all Phase 5–6 intents into the orchestrator and menu-bar UI
    - Connect schedule/task/image/run_automation intents through intent routing and the Dialogue_Manager gate; surface proposals, reminders, automation progress, and images in the SwiftUI panels
    - _Design: Intent Routing, The Orchestrator. Requirements: 6.1_
  - [ ]* 36.2 Write end-to-end integration test across a multi-subsystem command
    - Drive a command that spans Mac control, memory, scheduling, and dialogue with mocked models
    - _Design: Architecture_

- [ ] 37. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test tasks and can be skipped for a faster MVP; core implementation tasks are never optional.
- Each task references the specific requirements and/or design components it implements for traceability.
- The plan follows the design's 7-phase rollout; each phase ends with a checkpoint and is independently demoable.
- All 76 correctness properties are implemented as single property-based tests with a minimum of 100 iterations, tagged `Feature: haki-personal-ai-assistant, Property {number}: {property_text}`, using Hypothesis (Python Core) or SwiftCheck (Swift shell) with model-backed capabilities mocked.
- Unit/example, integration, performance (latency budgets), and smoke tests complement the property tests per the design's Testing Strategy.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["1.3"] },
    { "id": 2, "tasks": ["1.4", "1.5", "2.1", "2.2", "3.1"] },
    { "id": 3, "tasks": ["2.3", "3.2", "3.3", "3.4", "4.1", "5.1"] },
    { "id": 4, "tasks": ["4.2", "4.3", "4.4", "4.5", "4.6", "5.2", "5.6"] },
    { "id": 5, "tasks": ["5.3", "5.4", "5.5", "7.1", "8.1", "9.1", "10.2"] },
    { "id": 6, "tasks": ["7.2", "7.3", "8.2", "8.3", "9.2", "9.3", "10.1", "10.3", "10.5"] },
    { "id": 7, "tasks": ["7.4", "7.5", "7.6", "7.7", "8.4", "8.5", "8.6", "8.7", "8.8", "11.1"] },
    { "id": 8, "tasks": ["10.4", "11.2", "11.3", "13.1"] },
    { "id": 9, "tasks": ["13.2", "13.3", "13.4", "13.5", "13.6", "13.7", "14.1"] },
    { "id": 10, "tasks": ["14.2", "14.4", "14.5", "15.1", "15.2"] },
    { "id": 11, "tasks": ["14.3", "15.3", "15.4", "15.5", "15.6", "15.7", "16.1"] },
    { "id": 12, "tasks": ["16.2", "16.3", "18.1", "19.1"] },
    { "id": 13, "tasks": ["16.4", "16.5", "16.6", "16.7", "16.8", "16.9", "18.2", "18.3", "19.2"] },
    { "id": 14, "tasks": ["18.4", "18.5", "18.6", "18.7", "18.8", "19.3", "19.4", "19.5", "19.6", "19.7", "21.1"] },
    { "id": 15, "tasks": ["21.2", "22.1"] },
    { "id": 16, "tasks": ["21.3", "21.4", "22.2", "22.3", "22.4", "23.1"] },
    { "id": 17, "tasks": ["23.2", "24.1", "24.2", "25.1"] },
    { "id": 18, "tasks": ["23.3", "23.4", "23.5", "24.3", "24.4", "24.5", "24.6", "25.2"] },
    { "id": 19, "tasks": ["25.3", "27.1"] },
    { "id": 20, "tasks": ["25.4", "25.5", "25.6", "25.7", "25.8", "25.9", "27.2", "27.3", "28.1"] },
    { "id": 21, "tasks": ["27.4", "27.5", "27.6", "27.7", "28.2", "29.1"] },
    { "id": 22, "tasks": ["28.3", "28.4", "28.5", "28.6", "28.7", "29.2", "30.1"] },
    { "id": 23, "tasks": ["29.3", "30.2", "32.1", "33.1"] },
    { "id": 24, "tasks": ["29.4", "29.5", "29.6", "29.7", "29.8", "29.9", "30.3", "30.4", "30.5", "30.6", "30.7", "30.8", "32.2", "33.2"] },
    { "id": 25, "tasks": ["32.3", "32.4", "33.3", "33.4", "34.1", "35.1"] },
    { "id": 26, "tasks": ["34.2", "34.3", "34.4", "34.5", "35.2"] },
    { "id": 27, "tasks": ["35.3", "35.4", "35.5", "35.6", "36.1"] },
    { "id": 28, "tasks": ["36.2"] }
  ]
}
```
