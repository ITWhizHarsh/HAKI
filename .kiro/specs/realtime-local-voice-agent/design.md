# Technical Design: Realtime Local Voice Agent

## Overview

HAKI will replace the live, turn-based cloud voice path with a full-duplex local-first voice runtime. Swift exclusively owns microphone capture, VoiceProcessingIO, local ASR, and PCM rendering. A user-owned UNIX-domain socket carries normalized transcript and control messages—not microphone PCM—to Python. Python owns the asynchronous Pipecat turn graph, local-model routing, tool execution, sentence streaming, cancellation, playback-ledger context, and privacy-preserving diagnostics.

The normal route is:

```text
Swift AVAudioEngine + VoiceProcessingIO
  ├─ PCM frames (process memory only) → CoreML Qwen3-ASR adapter → partial/final transcript
  ├─ PCM frame mirror (protected local shared-memory ring) → Python InputAudioRawFrame → Silero VAD
  └─ final Transcript_Event (JSONL UDS; text only) ──────────────────────────┐
                                                                          Python Core
                                                               ┌──────────────┴──────────────┐
                                                               │ Pipecat VoiceSessionPipeline│
                                                               │ VAD/turn → LLM → sentences  │
                                                               └──────────────┬──────────────┘
                                                                              │ TTS PCM chunks
Swift PCM renderer ← output/control UDS ← XTTS v2 adapter ← TTSTextFrame ────┘
      │
      └─ Playback_Confirmation / cancellation → Python ledger and turn coordinator
```

The shared-memory ring is deliberately separate from `TranscriptSocket`: it exists only because Python Pipecat and Silero VAD require ephemeral `InputAudioRawFrame` payloads. It is allocated locally at session start, access-controlled to the same UID, never serialized into JSON, never logged, and unlinked/zeroed at session end. The only Swift-to-Python UNIX socket carrying application messages is the text/control protocol below; it never carries microphone samples. This preserves the stated clean-text IPC boundary while allowing Python to use the prescribed Pipecat audio frame type.

### Goals

- Local, full-duplex Hindi/Hinglish/English interaction on macOS 14+ / Apple M2 / 8 GB unified memory.
- Native echo-cancelled capture remains active during playback, enabling immediate barge-in.
- `Qwen3-4B-Instruct-4bit` through MLX/Metal and Coqui XTTS v2 are the normal response path.
- Existing `HAKIBrain` retrieval and `ScreenAgent` safety/permission semantics remain authoritative.
- Gemini Live is an explicit, per-session escalation only; it is never an availability fallback.
- The old cloud/turn-based implementation remains auditable but impossible to import, select, or run.

### Non-goals

- Redesigning `HAKIBrain`, Screen_Control safety policy, macOS permission policy, or non-voice orchestration.
- Persisting microphone audio or transcript content by default.
- Supporting legacy voice code as an optional mode, developer switch, test fixture runtime, or recovery path.
- Requiring network access for an ordinary voice session.

## Current Integration Constraints

`Core/core/ipc/server.py` currently accepts `AUDIO_FRAME`, buffers microphone samples, invokes the legacy STT path, streams `Orchestrator.stream_turn`, then calls Edge TTS/`afplay` with `say` fallback. That direct path must be removed from the replacement server rather than wrapped. `Orchestrator` already exposes cancellable `asyncio` turn execution and `set_current_task`; its process-wide mutable conversation history is not safe for playback-confirmed voice context. The new voice pipeline will use a session-owned context ledger and call reusable orchestration/capability interfaces, not `Orchestrator.run_turn()`’s history-mutating end-to-end path.

`LLMRouter` currently intentionally cascades Groq, Cerebras, Gemini, and local MLX, and its local configuration is not the required Qwen model/Metal configuration. Voice must use a new `VoiceLLMRouter`, not `_routing_order()` or an adapted cloud fallback list. `HAKIBrain.search`/`search_and_format` and `ScreenAgent.run` remain the authoritative tool targets. `IntentRouter` remains available for non-tool intent/capability behavior but is not trusted as a schema validator for model-emitted tools.

## Architecture and Responsibilities

| Boundary | Owner | Responsibility | Explicit exclusion |
|---|---|---|---|
| `VoiceAudioController` | Swift | AVAudioEngine lifecycle, VoiceProcessingIO, capture tap, monotonically sequenced `AudioFrame`, ASR feed, barge-in capture | No LLM, no Python turn ordering, no archive execution |
| `LocalASRAdapter` | Swift | Default CoreML Qwen3-ASR model inference, partial/final normalized text, language classification | No cloud ASR, no raw audio socket payload |
| `AudioFrameRing` | Swift + Python | Same-UID, session-random shared memory with bounded PCM frame descriptors for `InputAudioRawFrame`/Silero only | No persistence, no diagnostics, no UDS serialization |
| `TranscriptSocketClient` / `VoiceUnixServer` | Swift + Python | Authenticated text/control UDS protocol, reconnect state, output PCM/control delivery | No microphone sample fields |
| `VoiceSessionPipeline` | Python | Pipecat frame graph, per-turn order, VAD smart turn, cancellation/backpressure | No custom threads or subprocess playback |
| `VoiceLLMRouter` | Python | resource guard, local Qwen MLX generation, explicit cloud gate | No Groq/Cerebras/legacy route selection |
| `VoiceToolAdapter` | Python | strict registered-schema validation and turn-correlated results | No bypass of Screen_Control safeguards |
| `XTTSSentenceAdapter` | Python | sentence synthesis and PCM chunk streaming | No Edge/Kokoro/ChatTTS/system-speech fallback |
| `PlaybackLedger` | Python | append only renderer-confirmed assistant sentences to voice context | Never records partial/cancelled text |
| `VoiceDiagnosticsStore` | Python | local structured measurements and failures | No audio/full transcript by default |

### Proposed files and module layout

| Path | Change | Purpose |
|---|---|---|
| `legacy_pipeline_backup/` | new tracked static directory | Sanitized legacy artifacts plus `inventory.jsonl`, `README.md`, and checksums; no package markers or executable startup paths |
| `tools/archive_legacy_voice.py` | new one-shot migrator | Discover, classify, hash, redact, and copy legacy artifacts deterministically |
| `tools/check_voice_archive.py` | new CI/static checker | Enforce archive immutability, inventory completeness, redaction, and zero runtime coupling |
| `.github/workflows/voice-archive-policy.yml` | new | Runs archive and forbidden-reference policy on PRs; use the repository CI equivalent if GitHub Actions is not active |
| `Core/core/ipc/voice_protocol.py` | new | Versioned message schema, framing, UID/permission checks, acknowledgements, output protocol |
| `Core/core/ipc/voice_unix_server.py` | new | Dedicated transcript/control server; replaces voice behavior in `JSONIPCServer` while leaving non-voice IPC handlers intact |
| `Core/core/voice/session.py` | new | `VoiceSession`, lifecycle state, cancellation scope, context ledger ownership |
| `Core/core/voice/pipeline.py` | new | Pipecat source/processors/sinks and bounded queues |
| `Core/core/voice/frames.py` | new | typed metadata carried with Pipecat frames and protocol conversion helpers |
| `Core/core/voice/vad.py` | new | Silero VAD + smart-turn/barge-in state machine |
| `Core/core/voice/asr_bridge.py` | new | transcript event ingress and audio-ring source contract; no ASR implementation in Python |
| `Core/core/voice/llm.py` | new | Qwen MLX service, session gate integration, sentence stream generation |
| `Core/core/voice/tools.py` | new | Pydantic tool schemas and adapters to `HAKIBrain`/`ScreenAgent` |
| `Core/core/voice/tts.py` | new | XTTS v2 session, segmentation, PCM streamer, TTFA measurement |
| `Core/core/voice/resources.py` | new | model warm-up, resident/pipeline memory accounting and admission guard |
| `Core/core/voice/diagnostics.py` | new | diagnostic schema, redaction, local atomic JSONL storage |
| `Core/tests/voice/` | new | unit, property, transport, integration, and performance tests |
| `HAKI/Sources/Subsystems/Audio/VoiceAudioController.swift` | new/replaces legacy audio implementation | native capture/VoiceProcessingIO lifecycle |
| `HAKI/Sources/Subsystems/Audio/LocalASRAdapter.swift` | new | protocol and default CoreML Qwen3 ASR adapter |
| `HAKI/Sources/Subsystems/Audio/AudioFrameRing.swift` | new | same-user ephemeral PCM ring descriptors |
| `HAKI/Sources/Subsystems/Audio/PCMPlaybackRenderer.swift` | new | queueing, stop acknowledgement, sentence completion events |
| `HAKI/Sources/Subsystems/IPC/VoiceSocketClient.swift` | new | UDS reconnect, protocol sequencing, control/output handling |
| `HAKI/Tests/Voice/` | new | AVFoundation lifecycle and protocol integration tests |
| `Core/requirements.txt`, `Core/pyproject.toml`, lockfile | update | direct pinned production voice dependencies and test extras |

The first implementation PR must introduce `core.voice` alongside current code, but the cutover PR must remove the old voice branches from `JSONIPCServer`, `STTEngine` runtime selection, `TTSEngine` runtime selection, startup scripts, and service wiring. Non-voice uses of existing components are out of scope.

## 1. Static Legacy Archive and Zero-Coupling Policy

### Inventory and archive procedure

`archive_legacy_voice.py` accepts an explicit `legacy_voice_manifest.yaml` discovery policy. It scans repository-tracked artifacts in these categories: `script`, `handler`, `configuration`, `dependency_declaration`, `startup_path`, and `routing_rule`. Match rules include all code/config/launch/dependency references to Deepgram, Groq voice routing, Cartesia, Edge TTS, Kokoro, ChatTTS, `say`, `afplay`, old microphone buffering, old STT/TTS service classes, and their environment keys. The inventory is generated before any runtime deletion.

For every discovered file the migrator writes a record to `legacy_pipeline_backup/inventory.jsonl`:

```json
{
  "schema_version": 1,
  "original_path": "Core/core/ipc/server.py",
  "archive_path": "legacy_pipeline_backup/Core/core/ipc/server.py.txt",
  "category": "handler",
  "source_sha256": "<64 lowercase hex chars>",
  "archive_sha256": "<sanitized-copy digest>",
  "sanitized": true,
  "discovered_by": "legacy_voice_manifest.yaml"
}
```

The archive preserves all original repository-relative locations underneath its own root. It stores source snippets as static reference content (`.py.txt`, `.sh.txt`, etc.) rather than importable modules, contains no `__init__.py`, and is excluded from package discovery. Config sanitizers parse `.env`, JSON, YAML, TOML, plist, and line-oriented shell settings. Any key matching case-insensitive `*KEY`, `*TOKEN`, `*SECRET`, `*PASSWORD`, `*CREDENTIAL`, or provider-specific credential name is changed to `__REDACTED_LEGACY_SECRET__`; keys, comments where safe, original path, and non-secret values are retained. The tool fails closed for a config it cannot safely parse or a probable secret it cannot redact.

### Enforcement

`check_voice_archive.py` runs in CI and local pre-commit/verification mode. It:

1. Rebuilds the inventory in check-only mode and fails for missing, untracked, renamed, or digest-mismatched archive records.
2. Scans archive configuration for credential patterns and rejects values other than the redaction placeholder.
3. Scans executable source, imports, package manifests, environment discovery, launch agents, `start_haki.sh`, service registration, and route configuration for `legacy_pipeline_backup`, archived module names, or legacy provider selections in the voice runtime.
4. Parses Python AST imports/calls and package metadata rather than relying only on grep; it rejects `sys.path` additions, dynamic imports, subprocess execution, symlinks, and code generation that target the archive.
5. Runs a fault-injection test matrix that proves all replacement failures terminate as an error/degraded UI state and never select a legacy component.

The release build excludes `legacy_pipeline_backup/` from application bundles and Python package data. Developers may inspect it but no command may execute from it. CI’s static rule applies to tests and manual commands too.

**Decision:** archive as inert text, not a compatibility package. **Rejected:** retaining importable code behind a feature flag, because it makes both accidental and user-selected fallback possible.

## 2. Swift Full-Duplex Audio Subsystem

### Lifecycle and ownership

`VoiceAudioController` is an `actor` (or a serial dispatch-bound object where AVFoundation requires it) with states:

```text
idle → configuring → capturing ↔ playing → stopping → idle
                 ↘ unavailable(error) → retry/configuring
```

There is exactly one `AVAudioEngine` per voice session. That engine owns both input capture and output playback. `PCMPlaybackRenderer` attaches an `AVAudioPlayerNode` and scheduling queue to that same engine; it must not create a second engine, use `afplay`, or use `say`. Keeping input and output in the same VoiceProcessingIO topology is required for AEC reference alignment.

Start order is non-negotiable:

```swift
func startCapture() async throws {
    guard state == .idle else { throw VoiceAudioError.invalidState }
    state = .configuring
    let input = audioEngine.inputNode
    try input.setVoiceProcessingEnabled(true)
    guard input.isVoiceProcessingEnabled else {
        throw VoiceAudioError.voiceProcessingUnavailable
    }
    try configureEngineFormat(input.inputFormat(forBus: 0))
    input.installTap(onBus: 0, bufferSize: 960, format: nil, block: captureTap)
    try audioEngine.start()
    state = .capturing
}
```

The production implementation additionally handles route/sample-rate changes by serially stopping the tap, stopping the engine, re-enabling and re-verifying voice processing, reinstalling the tap, and starting again. It never accepts a microphone `AudioFrame` while `isVoiceProcessingEnabled == false`. Any enablement, verification, route, permission, engine-start, or tap-install failure transitions to `unavailable`, publishes an actionable UI error (permission, route reset, or retry), and emits `Voice_Diagnostic_Event(stage="voice_processing")`.

`VoiceProcessingIO` supplies AEC, noise suppression, and AGC for the microphone path. While assistant PCM is playing, capture remains active; output samples are scheduled on the player node while the input tap continues to feed local ASR/VAD. Application code must not try to subtract rendered PCM from the microphone signal because that would undermine the platform AEC reference.

### Frame model and failure recovery

Swift creates an in-memory `AudioFrame` per tap callback:

```swift
struct AudioFrame: Sendable {
    let sessionID: UUID
    let sequence: UInt64             // strictly incrementing per capture session
    let capturedAtMonotonicNs: UInt64
    let sampleRateHz: Int            // normalized to 16_000 for ASR/VAD
    let channels: UInt8              // 1
    let pcmS16LE: Data               // process memory only
}
```

The capture actor assigns sequence and timestamp before handing a frame to `LocalASRAdapter` and bounded `AudioFrameRing`. If either consumer is slow, the oldest *non-final, unprocessed* ring frame may be dropped with a diagnostic counter; the capture callback never blocks. A ring overrun cannot reorder frames. The ASR adapter sees discontinuity metadata and resets only its partial hypothesis, never fabricates a final transcript.

On interruption, media-services reset, or engine error: stop the tap, invalidate the current ASR partial/turn, send `CAPTURE_INTERRUPTED`, emit a diagnostic, and reconfigure with bounded exponential retries (250 ms, 500 ms, 1 s; maximum three). Capture stays unavailable after exhaustion until user retry. Playback renderer errors cancel only the affected assistant turn and preserve active capture.

### Native PCM playback protocol

Python sends `PCM_CHUNK` records containing only TTS output (never microphone audio): `{session_id, turn_id, sentence_id, sequence, sample_rate_hz, channels, format:"s16le", byte_length}` followed by a length-prefixed binary PCM payload. The renderer acknowledges `PCM_ACCEPTED` after enqueueing and sends exactly one terminal event per sentence:

- `PLAYBACK_CONFIRMED` only after all samples reached normal player-node completion;
- `PLAYBACK_CANCELLED` if stopped, route-lost, or replaced before completion;
- `PLAYBACK_FAILED` with error class for renderer failure.

`STOP_PLAYBACK(turn_id, generation)` is high priority and idempotent. The renderer removes queued buffers for that generation, calls `playerNode.stop()`, clears scheduled state, then emits stop acknowledgement. The barge-in timing test measures declaration-to-acknowledgement and requires no more than 200 ms.

## 3. Local ASR and Secure Transcript IPC

### ASR selection boundary

The initial production adapter is `CoreMLQwen3ASRAdapter`: a locally provisioned Qwen3-ASR CoreML artifact selected in `VoiceASRConfig` with a model manifest (model ID, artifact path, SHA-256, sample rate, vocabulary/version). It runs in Swift against normalized 16 kHz mono PCM, supports partial hypotheses, and classifies the final language as `hi`, `en`, or `hinglish`. `MLXQwen3ASRAdapter` is a future-compatible implementation of the same protocol for a locally provisioned MLX worker; it may replace CoreML only through explicit build/session configuration and the same conformance tests. No normal configuration may select Deepgram, Whisper cloud, Groq transcription, or other cloud ASR.

```swift
protocol LocalASRAdapter: Sendable {
    func startTurn(_ id: UUID) async throws
    func consume(_ frame: AudioFrame) async throws -> [ASRHypothesis]
    func finalize(turnID: UUID) async throws -> ASRHypothesis
    func cancel(turnID: UUID) async
}

struct ASRHypothesis: Sendable {
    let turnID: UUID
    let text: String
    let isFinal: Bool
    let language: TranscriptLanguage // hi | en | hinglish
    let captureStartedNs: UInt64
    let captureEndedNs: UInt64
}
```

Text normalization performs Unicode NFC, trims/collapses whitespace, removes control characters, preserves Devanagari and Latin characters, and rejects an empty final result. Empty final output emits an ASR diagnostic, clears the turn without emitting an LLM frame, and presents a repeat prompt.

### Transcript event contract

The newline-delimited JSON protocol is versioned and strict:

```json
{
  "version": 1,
  "type": "TRANSCRIPT_EVENT",
  "event_id": "uuid",
  "session_id": "uuid",
  "turn_id": "uuid",
  "event_seq": 17,
  "text": "Kal meeting reschedule kar do",
  "is_final": false,
  "language": "hinglish",
  "capture_started_monotonic_ns": 123,
  "capture_ended_monotonic_ns": 456
}
```

`text` is the only content field. The schema rejects `audio`, `samples`, `pcm`, `samples_b64`, arbitrary binary fields, unknown extensions, malformed IDs, invalid language, out-of-order `event_seq`, or a second final event. Partial events use one turn ID and precede its exactly-one final event. Python replies with `EVENT_ACK(event_id, accepted|discarded, reason?)`; Swift may resend unacknowledged partials after reconnect but never replays an acknowledged final event.

The socket lives at `$XDG_RUNTIME_DIR/haki/voice/<session-id>.sock` (fallback `~/Library/Application Support/HAKI/runtime/voice/`), with parent directory mode `0700`, socket mode `0600`, owner UID check, no group/other access, and a random 128-bit session capability exchanged only through the inherited launch configuration. The Python server validates `getpeereid`/equivalent UID before protocol negotiation. Socket filenames are created atomically, stale files are unlinked only after ownership/type validation, and symlinks are refused.

On connection loss, Swift enters `reconnecting`: it retains only the current *non-final* ASR hypothesis for UI display, discards the unfinished turn, emits an IPC diagnostic, and shows a reconnectable service error. It reconnects with bounded backoff (100 ms, 250 ms, 500 ms, 1 s, then user retry). It never transmits the missed final event after a disconnect because the turn was explicitly discarded; the user may speak again after connection recovery. Python marks any outstanding same-session turn terminal/cancelled and ignores late messages by `(session_id, turn_id, event_seq)`.

## 4. Python Pipecat Voice Pipeline

### Concrete processors and frame flow

`VoiceSessionPipeline` runs one Pipecat `PipelineTask` per active session in the existing asyncio loop. It uses `asyncio.TaskGroup`, cancellation scopes, and bounded `asyncio.Queue`s—not custom threads, `ensure_future` fire-and-forget chains, or blocking subprocess playback. Blocking MLX/XTTS calls are isolated behind a single-worker executor only where a library lacks async APIs; the executor is owned, bounded, and cancellation-aware.

```text
AudioFrameRingSource
  → InputAudioRawFrame
  → SileroVADSmartTurnProcessor ───────────────────┐
                                                     ├→ TurnJoinProcessor
TranscriptSocketSource → Transcript_Event → TranscriptionFrame ┘
  → VoiceTurnProcessor → CloudEscalationGate → VoiceLLMService
  → LLMTextFrame → SentenceBoundaryProcessor → TTSTextFrame
  → XTTSSentenceAdapter → PCMChunkFrame → SwiftPlaybackSink
  ← PlaybackEventSource (confirmed/cancelled/failed) → PlaybackLedgerProcessor
```

1. `AudioFrameRingSource` reads authenticated, bounded ring descriptors in sequence order, maps a frame temporarily, creates `InputAudioRawFrame(audio, sample_rate, num_channels)`, immediately releases that slot after VAD processing, and emits no diagnostic content from PCM. If the ring is unavailable, it transitions capture/VAD to unavailable instead of serializing raw samples to `TranscriptSocket`.
2. `SileroVADSmartTurnProcessor` consumes audio frames, produces internal speech/silence transitions and `BargeInEvent`; no VAD result makes a transcript on its own.
3. `TranscriptSocketSource` decodes and validates `Transcript_Event`, emits a `TranscriptionFrame` with metadata `{session_id, turn_id, event_seq, is_final, language, timestamps}`, and ACKs only after sequencing acceptance.
4. `TurnJoinProcessor` associates VAD state and final transcription by `turn_id`. A final transcript becomes eligible for LLM only after smart-turn finalization. It may update partial UI state but never starts an LLM turn for partial text.
5. `VoiceTurnProcessor` creates a turn cancellation scope, applies the resource admission guard and Cloud Escalation Gate, builds context only from `PlaybackLedger`, and invokes the selected LLM service.
6. `VoiceLLMService` emits ordered `LLMTextFrame`s and structured tool requests/results tagged with the turn ID. `SentenceBoundaryProcessor` turns only terminal-punctuation or explicit-end-of-response content into non-empty sentences and promptly emits each `TTSTextFrame`.
7. `XTTSSentenceAdapter` may synthesize sentence N+1 while the renderer plays N. It produces ordered `PCMChunkFrame`s with `(turn_id, sentence_id, chunk_seq)`; `SwiftPlaybackSink` preserves sentence order at the renderer.
8. `PlaybackEventSource` converts Swift terminal renderer events to internal confirmation/cancellation frames. `PlaybackLedgerProcessor` is the sole writer of assistant content to voice conversation context.

The Pipecat types `InputAudioRawFrame`, `TranscriptionFrame`, `LLMTextFrame`, and `TTSTextFrame` are mandatory in this path. Project metadata is stored in typed frame attributes/wrappers rather than encoded into text. The pipeline’s frame adapter must compile against `pipecat-ai==1.4.0`; if a compatible Pipecat release changes an import or frame constructor, the adapter shields the rest of HAKI and the pin updates only after compatibility tests pass.

### Ordering, backpressure, and terminality

A `TurnRegistry` holds `{turn_id, state, next_event_seq, cancellation_generation, queues}`. State transitions are linear:

```text
capturing → partial → final_pending_silence → reasoning → synthesizing → playing → completed
       └──────────────────────────────────────────────────────────────→ cancelled | failed
```

For one `turn_id`, processing is serialized by an `asyncio.Lock`; each stage accepts only the next sequence and terminal states reject all later frames. Interleaving different turns is allowed, but an older assistant turn cannot schedule output after a new barge-in generation. Bounded capacity policy is:

- capture ring: drop oldest non-final audio and report a counter; preserve sequence discontinuity;
- partial transcript UI queue: latest-wins/coalesced;
- final transcription, barge-in, cancellation, and playback terminal events: non-droppable control queue;
- LLM and sentence queues: bounded; upstream awaits capacity to avoid response reordering;
- PCM queue: bounded by milliseconds of audio; synthesis pauses before renderer overrun.

Pipecat initialization is an all-or-nothing session precondition. Failure leaves voice unavailable, emits a `pipecat` diagnostic, and does not start a substitute thread-based pipeline.

## 5. Smart Turn, Barge-In, and Confirmed-Playback Context

### Silero VAD state machine

Silero VAD receives 16 kHz mono frames and applies configured hysteresis (initial values: speech probability >= 0.60; release < 0.35) with monotonic timestamps. Values are configuration, not magic constants in callbacks. The state machine is:

```text
LISTENING -- voiced accumulation >=200ms --> USER_SPEAKING
USER_SPEAKING -- continuous silence <800ms --> USER_SPEAKING
USER_SPEAKING -- continuous silence >=800ms --> FINALIZE_TURN
PLAYING + voiced accumulation >=200ms --> BARGE_IN_DECLARED
BARGE_IN_DECLARED --> CAPTURING_NEW_TURN (immediately)
```

During assistant playback, 200 ms of *continuous voiced* user audio is the threshold. The VAD processor records `threshold_reached_ns`; `BargeInCoordinator` must publish the declaration within 200 ms from that point and emit high-priority cancellation. The 800 ms end-of-turn silence applies only when no barge-in is active; silence restarts whenever speech resumes.

On barge-in, atomically increment the session cancellation generation, then concurrently:

1. tell Swift `STOP_PLAYBACK` and require acknowledgement within 200 ms;
2. cancel active local/eligible-cloud LLM tasks and await only non-blocking cancellation registration;
3. drain every queued `TTSTextFrame`, pending synthesis job, and unscheduled `PCMChunkFrame` whose `(turn_id, generation)` matches the interrupted turn;
4. mark unconfirmed sentences cancelled;
5. return the capture/VAD path to `CAPTURING_NEW_TURN` immediately, without awaiting LLM/XTTS cleanup.

All producers compare their captured generation before emitting, so a late token, synthesis chunk, or playback completion cannot resurrect interrupted output.

### Playback ledger and context

`PlaybackLedger` is per session and append-only for confirmed assistant sentences:

```python
@dataclass(frozen=True)
class PlayedSentence:
    turn_id: UUID
    sentence_id: UUID
    text: str
    playback_completed_monotonic_ns: int

@dataclass(frozen=True)
class VoiceContext:
    user_turns: tuple[ContextMessage, ...]
    assistant_sentences: tuple[PlayedSentence, ...]
```

User final transcripts are appended when their turns are accepted. Assistant text is provisional until `PLAYBACK_CONFIRMED`; only then is that exact sentence appended in renderer completion order. `PLAYBACK_CANCELLED`, `PLAYBACK_FAILED`, interrupted output, raw LLM tokens, partial sentence buffers, and merely enqueued/synthesized text are never added. Later prompts are rendered from ordered user messages plus confirmed assistant sentences only. This deliberately differs from `Orchestrator._conversation_history`; the voice session adapter must not let that process-wide history append full generated replies before playback confirms them.

## 6. Local LLM, Tools, and Resource Lifecycle

### Local Qwen service

`VoiceLocalMLXService` is the normal LLM provider. Its fixed configuration is:

```python
VoiceMLXConfig(
    model_id="Qwen/Qwen3-4B-Instruct-4bit",
    runtime="mlx-lm==0.18.1",
    use_metal=True,
    max_context_tokens=16_384,
    max_generation_tokens=1_024,
    model_cache_capacity=1,
)
```

The exact local model artifact is provisioned before session start and recorded in a non-secret model manifest with repository-independent artifact hash. The service loads once per warm voice session through `mlx_lm.load`, uses Metal acceleration (do not pass `metal=False`), serializes concurrent generation through a single model semaphore, and streams incremental generation chunks to `LLMTextFrame`. It supports cancellation between decode steps and destroys/evicts the model only through `VoiceResourceManager`.

`VoiceLLMRouter` calls `CloudEscalationGate` first. It invokes Qwen for every non-eligible turn and does not delegate to the existing broad `LLMRouter` fallback order. A Qwen load or terminal generation error ends the affected voice turn with a `local_llm` diagnostic and user-facing error; it never selects Groq, Cerebras, Gemini, or a legacy voice route as an implicit retry.

### Structured tool adapter

The model is prompted with an allowlisted JSON tool grammar. Parsed tool calls are data, never Python dispatch strings. Pydantic schemas are strict (`extra="forbid"`) and include a tool name/version:

```python
class ObsidianRAGCall(BaseModel):
    tool: Literal["obsidian_rag.search"]
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    limit: Annotated[int, Field(ge=1, le=5)] = 3

class ScreenControlCall(BaseModel):
    tool: Literal["screen_control.run"]
    goal: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    confirmation_context: Literal["voice"]
```

`VoiceToolAdapter` parses/validates before invocation. `obsidian_rag.search` calls existing `HAKIBrain.search` or `search_and_format`; it returns a bounded structured result. `screen_control.run` calls existing `ScreenAgent.run` and passes through the existing accessibility permission and consequential-action confirmation workflow—it cannot auto-confirm because it originated from voice. Valid results return as a structured, turn-correlated tool-result message to the same Qwen generation before final response text resumes. Schema failure produces `stage="tool_call"`, no target call, and a safe clarification/error response.

The gate’s `>6 validated tools` condition uses the validated planned/executed tool-call count, never untrusted model text. Tool calls that are invalid do not count.

### Resource budget guard

`VoiceResourceManager` warms local ASR, Qwen, XTTS, and Pipecat before timing tests. It measures the MLX model process resident footprint after warm-up and before a turn (`Model_Resident_Footprint`), and measures the union of Swift, Core Python, and local ASR worker resident memory (`Pipeline_Memory`) from process-specific macOS metrics. It records sampling method/version and avoids summing child memory twice.

Admission state is:

```text
ADMITTING -- model >=2.5GB OR pipeline >5GB --> DRAINING
DRAINING: reject new turns, release idle XTTS then idle Qwen/cache, keep active capture safe
DRAINING -- model <2.5GB AND pipeline <=5GB --> ADMITTING
```

A resource breach reports `memory_budget`, stops admitting new turns, frees only idle resources, and does not route to cloud/legacy. A later sample must satisfy both recovery thresholds before new turns resume. The warmed deployment acceptance gates are `< 2.5 GB` model resident footprint, `<= 5 GB` pipeline memory, and final user utterance to first assistant PCM `<= 1.5 s` under the Nominal_Test_Profile.

## 7. XTTS Sentence Streaming

`XTTSSentenceAdapter` constructs a single local `TTS==0.22.0` / Coqui XTTS v2 session per warmed voice session using a readable, user-provided `my_voice.wav` path. It rejects missing/unreadable assets at initialization rather than silently choosing a system, Edge, Kokoro, ChatTTS, or Cartesia voice.

`SentenceBoundaryProcessor` buffers `LLMTextFrame` chunks until terminal `.`, `?`, `!`, Devanagari `।`, or explicit end-of-response; it trims and rejects empty text. Abbreviation-aware rules prevent common false boundaries. A complete sentence immediately creates `TTSTextFrame(turn_id, sentence_id, text, language)`, without waiting for later generation. Classification selects `en` only for `en`; `hi` for `hi` and `hinglish`:

```python
xtts_language = "en" if sentence.language == "en" else "hi"
```

XTTS chunk generation is sequential per sentence but may pipeline sentence N+1 synthesis while sentence N is playing. A bounded synthesis semaphore and PCM backpressure prevent memory growth. TTFA starts when the complete `TTSTextFrame` reaches the adapter and ends when Swift receives the first non-empty `PCM_CHUNK`; target is `<= 500 ms` for warmed representative sentences on Nominal_Test_Profile. If any sentence fails synthesis, cancel unplayed synthesis for its assistant turn, record `local_tts`, send the entire generated response as a UI text event, and do not substitute any old or system TTS engine.

## 8. Explicit Session-Scoped Gemini Live Gate

```python
@dataclass(frozen=True)
class GateInput:
    session_id: UUID
    gemini_enabled_for_session: bool
    battery_percent: int | None
    external_power_connected: bool | None
    thermal_state: Literal["nominal", "fair", "serious", "critical"]
    assembled_prompt_tokens: int
    validated_tool_count: int

@dataclass(frozen=True)
class GateDecision:
    route: Literal["local_qwen", "gemini_live"]
    qualifying_conditions: tuple[Literal["low_battery", "thermal_throttling", "ultra_complex_reasoning"], ...]
```

The gate starts disabled for every `VoiceSession`. Only an explicit user action, recorded with that active session ID and reflected in the UI, enables it. Ending the session deletes the enablement. Eligibility is true only when enablement is true **and** at least one condition holds:

- `battery_percent <= 20` **and** external power is disconnected;
- thermal state is `serious` or `critical`;
- assembled prompt is greater than 16,000 tokens **or** validated tool plan requires more than six calls.

Disabled plus any condition routes locally. Enabled plus no condition routes locally. Eligible Gemini Live invocation failure returns a reported cloud error; it may not fall through to legacy, local Qwen, or another cloud provider for that already-selected turn. Gemini Live is therefore an explicitly chosen escalation, not an availability fallback. Gate diagnostics record enabled state, every evaluated condition/value class, qualifying conditions, and selected route, but not prompt text.

## 9. Diagnostics, Errors, and Privacy

### Schema and local storage

`VoiceDiagnosticEvent` is versioned JSONL stored atomically at `~/Library/Application Support/HAKI/diagnostics/voice/<local-date>.jsonl`, directory mode `0700`, file mode `0600`:

```json
{
  "schema_version": 1,
  "event_id": "uuid",
  "session_id": "uuid",
  "turn_id": "uuid",
  "stage": "asr|ipc|pipecat|voice_processing|local_llm|tool_call|local_tts|memory_budget|cloud_gate|playback",
  "outcome": "started|completed|cancelled|failed|rejected",
  "started_monotonic_ns": 0,
  "transcription_completed_monotonic_ns": 0,
  "first_llm_text_monotonic_ns": 0,
  "first_tts_text_monotonic_ns": 0,
  "first_pcm_delivered_monotonic_ns": 0,
  "ttfa_ms": 0,
  "selected_route": "local_qwen|gemini_live",
  "asr_engine": "qwen3_asr_coreml",
  "tts_engine": "xtts_v2",
  "model_resident_bytes": 0,
  "pipeline_memory_bytes": 0,
  "gate": {"enabled": false, "evaluated": ["low_battery"], "qualifying": []},
  "error_class": null,
  "recovery_outcome": null,
  "content_capture_enabled": false
}
```

Diagnostics record identifiers, routing, component IDs, stage timestamps, memory values, terminal result, and error/recovery data. Default serialization excludes PCM, byte data, transcript text, LLM response text, tool arguments/results containing content, and full prompt text. It may include non-reversible text length and a per-session salted correlation hash when needed. A user-controlled session-scoped `content_capture_enabled` may add explicitly labeled transcript/response diagnostic fields; it never enables raw-audio recording. The control defaults false, is displayed in UI, expires at session end, and is captured in each event.

### Failure policy

Every named failure is converted to a stable error class and one terminal outcome. Native capture failures leave voice unavailable/retryable; empty ASR asks for repetition; IPC loss discards the current turn and reconnects; Pipecat start failure leaves voice unavailable; tool validation refuses execution; local LLM/TTS failure shows an error/text result; renderer error excludes unconfirmed sentences; memory breach drains/rejects; cloud failure reports failure. No failure policy invokes an archived or legacy component.

## 10. Dependencies and Deployment Configuration

The production dependency declarations must contain matching exact direct pins in both `Core/pyproject.toml` and `Core/requirements.txt` (and a generated lockfile):

```text
pipecat-ai==1.4.0
mlx-lm==0.18.1
TTS==0.22.0
silero-vad==5.1.2
```

Supporting packages required by the selected adapter must also be locked, not ranged: `torch==2.5.1`, `torchaudio==2.5.1`, `soundfile==0.12.1`, `numpy==1.26.4`, `psutil==6.0.0`, and a lockfile-resolved `pydantic==2.8.2`. Replace the existing unbounded `httpx`, `numpy`, `torch`, `torchaudio`, `psutil`, cloud voice SDK, Deepgram, Kokoro, Edge TTS, Cartesia, and ChatTTS declarations from the **voice runtime** dependency set during cutover. If any are still needed by a non-voice product feature, isolate them in an explicit non-voice optional extra; they must not be imported or selected from `core.voice`.

Model provisioning stores artifact manifests and hashes under a user-local HAKI model directory, never in the repository archive. The Core startup performs a non-blocking model availability/voice permission check; it does not download or convert models during a turn. Swift package dependencies remain pinned in `HAKI/Package.resolved`.

## 11. Testing Strategy and Migration Sequence

### Test layers

| Test ID | Scope | Method and required assertions | Requirements |
|---|---|---|---|
| `V-ARCHIVE` | archive | fixture migration, inventory digest/path/category, parser redaction, CI AST/static forbidden-reference scan | 1.1–1.6 |
| `V-SWIFT-AUDIO` | native macOS | instrumented input node proves enable → verify → tap order; duplex capture/playback; VoiceProcessingIO unavailable and engine-reset recovery | 2.1–2.5 |
| `V-FRAME-PROP` | property, >=100 cases | randomized frame/transcript/event interleavings verify sequence monotonicity, normalized schema, no PCM in UDS, prescribed Pipecat frames, and per-turn order | 2.6, 3.2–3.3, 3.6, 4.2–4.6, 4.8 |
| `V-ASR-IPC` | integration/security | local Qwen/CoreML adapter fixture, temporary UDS same-UID transport, UID/mode validation, disconnect before final, empty final | 3.1, 3.4–3.5, 3.7–3.8 |
| `V-PIPELINE` | async integration | Pipecat initialization failure, final transcript to frame graph, bounded queues, no legacy/custom-thread runtime | 4.1, 4.9 |
| `V-TURN-PROP` | property, >=100 cases | synthetic 16 kHz VAD timelines, 800 ms silence condition, 200 ms barge threshold, cancellation generations, queues and context projection | 4.7, 5.1, 5.3–5.8 |
| `V-BARGE-LATENCY` | deterministic integration | controllable renderer clock verifies threshold-to-declaration and declaration-to-stop acknowledgement each <=200 ms; capture accepts new frames before cleanup completes | 5.1–5.2, 5.5 |
| `V-TOOLS-PROP` | property, >=100 cases | fuzz valid/invalid Pydantic tool calls; valid result correlation; zero invocation on invalid schemas; ScreenAgent confirmation is preserved | 6.3–6.6 |
| `V-LLM` | mocked MLX/gate integration | exact Qwen model/Metal config, local default, load/generation failure has no legacy route | 6.1–6.2, 6.7 |
| `V-TTS-PROP` | property, >=100 cases | language mapping, punctuation/EOR segmentation, immediate sentence frames, overlap scheduling, error-to-full-text UI | 7.3–7.5, 7.7–7.8 |
| `V-TTS-TTFA` | Nominal_Test_Profile benchmark | warmed XTTS with `my_voice.wav`, representative en/hi/Hinglish sentences; first delivered PCM <=500 ms | 7.1–7.2, 7.6 |
| `V-GATE-PROP` | property, >=100 truth-table cases | session enable/end scope, all battery/power/thermal/prompt/tool combinations, route and diagnostic reason | 8.1–8.6 |
| `V-CLOUD-FAIL` | integration | eligible mocked Gemini Live failure reports error and does not select legacy or a fallback route | 8.7 |
| `V-RESOURCE` | Nominal_Test_Profile benchmark | warmed footprint <2.5 GB, concurrent <=5 GB, final transcript-to-first PCM <=1.5 s, drain/recover admission behavior | 9.1–9.6 |
| `V-DIAG-PROP` | property, >=100 cases | start/terminal/gate/failure completeness and default content redaction | 10.1–10.3, 10.5–10.6 |
| `V-DIAG-STORE` | integration | atomic local storage, modes/ownership, retrieval | 10.4 |

Property tests use the tag format `Feature: realtime-local-voice-agent, Property N: <property title>` and a minimum of 100 iterations. Synthetic audio fixtures include silence, voice-like samples, Hindi/English/Hinglish labels, frame gaps, route changes, and cancellation races; they do not require real microphone recording. Hardware-dependent VoiceProcessingIO, memory, and latency gates run separately on the declared Nominal_Test_Profile and are not made flaky unit-test requirements.

### Safe migration sequence

1. Freeze and classify existing legacy voice artifacts; run the archive tool, review redaction, commit only static archive and inventory.
2. Add static CI enforcement before new runtime code. Ensure current legacy paths are still runnable only until cutover, never from archive.
3. Introduce Swift audio/ASR/renderer and Python `core.voice` behind an internal development-only replacement feature gate that cannot reference the archive.
4. Implement and validate protocol, Pipecat, VAD/barge-in, Qwen/XTTS, tool schemas, diagnostics, resource guard, and all mock/property tests.
5. Run macOS hardware integration, memory, TTFA, end-to-end responsiveness, and a clean-session Gemini gate test on Nominal_Test_Profile.
6. Cut over startup/service wiring to `VoiceUnixServer` and `VoiceSessionPipeline`; remove the old raw-audio buffering, Deepgram/Groq/Cartesia/Edge/Kokoro/ChatTTS/system-speech selections and playback subprocesses from live voice paths.
7. Re-run archive inventory against the removed artifacts, enforce zero coupling, and exercise each injected replacement failure to prove no legacy fallback.
8. Release with normal voice route local-only and Gemini disabled. Observe only privacy-preserving local diagnostics; enable diagnostic content only through the per-session user control.

## Correctness Properties

*A property is a behavior that must hold across all valid executions. These properties complement example, integration, smoke, and hardware performance tests described above.*

### Property reflection

The transcript event ordering, frame type mapping, and per-turn frame ordering requirements are combined into Property 3 because a single correlated event stream subsumes individual type-only checks. Playback confirmation, interrupted-sentence exclusion, and later-context visibility are combined into Property 6 because the ledger invariant determines all three. The six cloud-gate routing criteria are combined into Property 9 as one complete truth table. Tool schema validation/rejection is combined into Property 8. This avoids duplicate generators while retaining each distinct requirement reference.

### Property 1: Archive inventory and redaction preservation

For all discovered legacy artifact sets and supported configuration secret values, migration produces exactly one inventory entry and one inert archived copy per source path, preserves the artifact category and source SHA-256 mapping, and replaces every credential value with the approved placeholder without changing its configuration key.

**Validates: Requirements 1.1, 1.3, 1.4**

### Property 2: Replacement failures cannot select legacy behavior

For all replacement voice stages and injected terminal failures, the resulting recovery decision is an unavailable, retry, error, cancellation, or text-only outcome and never imports, executes, selects, or proposes a Legacy_Voice_Pipeline component.

**Validates: Requirements 1.6, 4.9, 6.7, 8.7**

### Property 3: Sequenced local capture, transcript, and Pipecat frame preservation

For all accepted monotonically captured audio-frame sequences and all valid partial/final transcript event streams, frames retain strictly increasing capture sequence metadata, transcript events contain no microphone payload, each turn has zero or more non-final events followed by at most one final event, and the corresponding `InputAudioRawFrame`, `TranscriptionFrame`, `LLMTextFrame`, and `TTSTextFrame` values retain their turn order until terminal completion or cancellation.

**Validates: Requirements 2.6, 3.2, 3.3, 3.6, 4.2, 4.3, 4.4, 4.5, 4.6, 4.8**

### Property 4: Smart-turn silence rule

For all Silero VAD speech/silence timelines and barge-in states, an active user utterance is finalized if and only if it has 800 milliseconds of continuous post-speech silence while no barge-in is active.

**Validates: Requirements 4.7**

### Property 5: Barge-in invalidates interrupted assistant work

For all active assistant generations and arbitrary queued TTS/audio work, once 200 milliseconds of continuous user speech during playback reaches the barge-in threshold, the pipeline cancels unfinished generation and removes every queued or unscheduled output item belonging to the interrupted turn generation, while accepting frames for the new user utterance without waiting for cleanup.

**Validates: Requirements 5.1, 5.3, 5.4, 5.5**

### Property 6: Confirmed-playback context ledger

For all ordered assistant sentence schedules, playback confirmations, failures, and cancellations, conversation context contains every user turn plus exactly the assistant sentences that received normal `Playback_Confirmation`, in confirmation order, and contains no partial, merely generated, failed, or interrupted sentence text.

**Validates: Requirements 5.6, 5.7, 5.8**

### Property 7: Local-route default

For all voice turns that the Cloud_Escalation_Gate does not mark eligible, routing selects the Qwen local MLX service and no cloud or legacy voice route.

**Validates: Requirements 6.1, 8.5, 8.6**

### Property 8: Tool-call safety and turn correlation

For all model-produced tool-call objects, only calls conforming to the registered Obsidian_RAG or Screen_Control schema invoke their target; every invalid call produces a tool-call diagnostic with no target invocation; and every valid result is returned to the same originating LLM turn before its final response continues.

**Validates: Requirements 6.3, 6.4, 6.5, 6.6**

### Property 9: Session-scoped cloud gate truth table

For all session identifiers, enablement actions, end events, battery/power states, thermal states, prompt token counts, and validated tool counts, Gemini Live is eligible exactly when it is explicitly enabled for the active session and at least one qualifying condition holds; otherwise the route is local, and the gate diagnostic records its enablement, evaluated conditions, qualifying conditions, and selected route.

**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 10.3**

### Property 10: Sentence streaming language and overlap

For all complete non-empty response sentences and classified languages, sentence completion emits an ordered TTS-text frame without waiting for later response content, selects `en` exactly for English and `hi` exactly for Hindi or Hinglish, and starts available next-sentence synthesis while prior-sentence playback remains active subject to bounded backpressure.

**Validates: Requirements 7.3, 7.4, 7.5, 7.7**

### Property 11: Memory admission and recovery

For all sampled model-resident and pipeline-memory measurements, a measurement at or beyond the model limit or above the pipeline limit rejects new voice turns, requests idle resource release, and records a memory-budget diagnostic; admission resumes only after model resident memory is below 2.5 GB and pipeline memory is at or below 5 GB.

**Validates: Requirements 9.4, 9.5**

### Property 12: Privacy-preserving diagnostic completeness

For all voice turn starts, terminal outcomes, gate evaluations, and stage failures, diagnostics contain the required IDs, selected components/routes, applicable timing/resource/error/recovery fields, while default serialization excludes raw microphone data and full transcript text unless the session-scoped content control is enabled.

**Validates: Requirements 10.1, 10.2, 10.5, 10.6**

## Requirements Traceability

| Requirement | Design coverage | Test coverage |
|---|---|---|
| 1.1–1.6 | Static Legacy Archive and Zero-Coupling Policy | `V-ARCHIVE`, Properties 1–2 |
| 2.1–2.6 | Swift Full-Duplex Audio Subsystem | `V-SWIFT-AUDIO`, `V-FRAME-PROP`, Property 3 |
| 3.1–3.8 | Local ASR and Secure Transcript IPC | `V-ASR-IPC`, `V-FRAME-PROP`, Property 3 |
| 4.1–4.9 | Python Pipecat Voice Pipeline; Smart Turn | `V-PIPELINE`, `V-FRAME-PROP`, `V-TURN-PROP`, Properties 3–4 |
| 5.1–5.8 | Smart Turn, Barge-In, and Confirmed-Playback Context | `V-TURN-PROP`, `V-BARGE-LATENCY`, Properties 5–6 |
| 6.1–6.7 | Local LLM, Tools, and Resource Lifecycle | `V-LLM`, `V-TOOLS-PROP`, Properties 2, 7–8 |
| 7.1–7.8 | XTTS Sentence Streaming | `V-TTS-PROP`, `V-TTS-TTFA`, Property 10 |
| 8.1–8.7 | Explicit Session-Scoped Gemini Live Gate | `V-GATE-PROP`, `V-CLOUD-FAIL`, Properties 2, 7, 9 |
| 9.1–9.6 | Resource Lifecycle; Dependencies and Deployment Configuration | `V-RESOURCE`, Property 11 |
| 10.1–10.6 | Diagnostics, Errors, and Privacy | `V-DIAG-PROP`, `V-DIAG-STORE`, Properties 9, 12 |
