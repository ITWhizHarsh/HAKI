# HAKI — Personal AI Assistant

**Heuristic Augmented Knowledge Interface** — A local-first, privacy-respecting personal AI assistant for macOS Apple Silicon (M2), built as a two-process hybrid architecture optimized for 8 GB unified memory. HAKI is a **fully local-first** voice and text AI that runs your data on-device, routes to cloud APIs only when beneficial, and gives you a production-quality macOS GUI — no subscriptions required.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [New — Realtime Local Voice Pipeline](#new--realtime-local-voice-pipeline)
3. [New — HAKI Brain Memory Processing Pipeline](#new--haki-brain-memory-processing-pipeline)
4. [New — Gemini Sidecar GUI Agent](#new--gemini-sidecar-gui-agent)
5. [New — Persistent Memory Vault Fix](#new--persistent-memory-vault-fix)
6. [New — SwiftUI Frontend (HAKIFrontend)](#new--swiftui-frontend-hakifrontend)
7. [LLM Routing Strategy](#llm-routing-strategy)
8. [Embeddings and RAG](#embeddings-and-rag)
9. [Mood Detection](#mood-detection)
10. [System Requirements](#system-requirements)
11. [Quick Start](#quick-start)
12. [Project Structure](#project-structure)
13. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

HAKI uses a **heterogeneous compute scheduling** architecture — intelligently routing between cloud APIs and local ANE/Metal models to maintain sub-300 ms latency inside a strict 8 GB RAM budget.

The system is split into two cooperating processes:

**Swift "Body" (HAKI / HAKIFrontend)**
- `HAKIFrontend` — SwiftUI `@main` app, glassmorphism UI, 6-state observable model
- `VoiceAudioController` — `AVAudioEngine` + `VoiceProcessingIO` (AEC, NS, AGC)
- `LocalASRAdapter` — CoreML Qwen3-ASR, Hindi/Hinglish/English code-switching
- `PCMPlaybackRenderer` — XTTS v2 PCM playback, barge-in acknowledgement
- `AudioFrameRing` — same-UID shared-memory ring for Silero VAD frames
- `VoiceSocketClient` — UNIX-domain transcript/control protocol to Python
- `MoodDSP` — RMS + autocorrelation pitch detection (Accelerate framework)
- `ScreenReader` — ScreenCaptureKit JPEG sidecar for Gemini GUI agent
- `PermissionManager` — TCC gates (mic, screen, accessibility)

**Python "Mind" (Core)**
- `VoiceSessionPipeline` — Pipecat asynchronous frame graph; Silero VAD smart-turn
- `VoiceLLMRouter` — Qwen3-4B-Instruct-4bit via `mlx-lm` + Metal; Gemini Live gate
- `XTTSSentenceAdapter` — XTTS v2 sentence streaming via `TTS==0.22.0`
- `PlaybackLedger` — confirmed-playback-only conversation context
- `HAKIBrain` — 3-folder Obsidian wiki with Hybrid Fast/Heavy pipeline
- `FastPassExtractor` — spaCy NER + regex, zero LLM cost
- `HeavyPassExtractor` — Bonsai-8B MLX evolutionary memory synthesis
- `PipelineScheduler` — APScheduler background ingestion jobs
- `SidecarAgentLoop` — Gemini Vision See→Think→Act→Verify GUI agent
- `GeminiVisionClient` — Gemini 2.5 Flash + SCKit frame stream
- `MacQuartzExecutor` — `pyobjc-framework-Quartz` HID event dispatch
- `LLMRouter` — Groq / Cerebras / Gemini / MLX (non-voice text turns)
- `EmbeddingsEngine` — Granite ModernBERT 32k + ChromaDB
- `JSONIPCServer` — UNIX socket server with `AGENT_EVENT` broadcast

---

## New — Realtime Local Voice Pipeline

The legacy Deepgram → Groq → Cartesia turn-based voice path has been **completely replaced** with a full-duplex, local-first runtime. The old code is archived to `legacy_pipeline_backup/` as inert text — it cannot be imported or executed.

### How it Works

Swift owns the entire audio path. Python owns the Pipecat turn graph, LLM inference, and TTS streaming. The only data that crosses the process boundary is clean transcript text — **no microphone audio ever leaves the Swift process over the socket**.

```
Swift AVAudioEngine + VoiceProcessingIO
  |-- PCM frames --> CoreML Qwen3-ASR --> partial/final transcript
  |-- PCM mirror --> AudioFrameRing --> Python Silero VAD (ephemeral, same-UID)
  +-- TRANSCRIPT_EVENT (text only, UNIX socket) -------------------------->
                                                             Python Core
                                              Pipecat VoiceSessionPipeline
                                              VAD/turn --> LLM --> sentences
                                              XTTS v2 PCM chunks
Swift PCMPlaybackRenderer <-- output socket <-----------------------------+
  +-- PLAYBACK_CONFIRMED / PLAYBACK_CANCELLED --> PlaybackLedger
```

### Swift Audio Subsystem

- `VoiceAudioController` actor: single `AVAudioEngine` per session, VoiceProcessingIO AEC always on during playback (enables barge-in)
- Start order enforced: enable VoiceProcessingIO → verify → install tap → start engine
- Each `AudioFrame` carries a monotonic sequence number and capture timestamp
- Route-change / media-services-reset recovery with bounded retries (250 ms, 500 ms, 1 s)
- `PCMPlaybackRenderer` uses the same engine — no `afplay`, no `say`, no second engine

### Local ASR (Qwen3-ASR CoreML)

- Default adapter: `CoreMLQwen3ASRAdapter` — runs on `.cpuAndNeuralEngine`, zero Metal GPU usage
- Supports partial hypotheses; language classification: `hi` / `en` / `hinglish`
- Text normalization: Unicode NFC, Devanagari + Latin preserved, empty finals trigger repeat prompt
- Future: `MLXQwen3ASRAdapter` available through explicit build config only

### Pipecat Voice Pipeline

One `PipelineTask` per voice session runs in the existing asyncio event loop:

```
AudioFrameRingSource --> InputAudioRawFrame
  --> SileroVADSmartTurnProcessor  (speech/silence + barge-in events)
  --> TurnJoinProcessor             (pairs VAD state with final transcript)
TranscriptSocketSource --> TranscriptionFrame
  --> VoiceTurnProcessor --> CloudEscalationGate --> VoiceLLMService
  --> LLMTextFrame --> SentenceBoundaryProcessor --> TTSTextFrame
  --> XTTSSentenceAdapter --> PCMChunkFrame --> SwiftPlaybackSink
  <-- PlaybackEventSource  --> PlaybackLedgerProcessor
```

- Frame types: `InputAudioRawFrame`, `TranscriptionFrame`, `LLMTextFrame`, `TTSTextFrame` (all Pipecat native)
- Bounded `asyncio.Queue` at every stage; no custom threads; no fire-and-forget chains
- `TurnRegistry` serializes each `turn_id` through states: `capturing → partial → final_pending_silence → reasoning → synthesizing → playing → completed`

### Smart Turn Taking and Barge-In

- Silero VAD (16 kHz mono): speech threshold 0.60, release 0.35
- End-of-turn: 800 ms continuous silence after speech
- Barge-in: 200 ms continuous voiced speech **during playback** triggers cancellation within 200 ms
- On barge-in: increment cancellation generation, stop playback, cancel LLM generation, flush TTS queue, start new capture — all concurrently, new capture does not wait for cleanup

### Confirmed-Playback Context (PlaybackLedger)

The `PlaybackLedger` is per-session and append-only. A sentence enters context **only** after `PLAYBACK_CONFIRMED` arrives from Swift — interrupted, cancelled, or merely generated text is never added. This means the LLM always reasons from what the user actually heard.

### Local LLM — Qwen3-4B-Instruct-4bit

- Model: `Qwen/Qwen3-4B-Instruct-4bit` loaded via `mlx-lm==0.18.1` with Metal acceleration
- Context: 16,384 tokens max; single model semaphore serializes concurrent generation
- No Groq, Cerebras, or legacy route as implicit fallback — a load failure ends the turn with an error
- Memory budget: model resident < 2.5 GB; full pipeline <= 5 GB; first PCM <= 1.5 s on M2

### XTTS v2 Sentence Streaming

- Library: `TTS==0.22.0` (Coqui XTTS v2), conditioned on user-provided `my_voice.wav`
- Language routing: `en` → XTTS `en`; `hi` or `hinglish` → XTTS `hi`
- Pipelining: sentence N+1 synthesizes while sentence N plays (bounded backpressure)
- TTFA target: <= 500 ms from complete sentence to first PCM chunk (warmed, M2)
- TTS failure: cancel turn synthesis, send full generated text as on-screen fallback

### Gemini Live Gate

Gemini Live is **disabled by default** for every new voice session. It is an explicit opt-in, not an availability fallback. A turn is only eligible when:
1. The user explicitly enables Gemini Live for the active session, **AND**
2. At least one qualifying condition is present: battery ≤ 20% on external disconnect, thermal state `serious`/`critical`, or prompt > 16k tokens / > 6 validated tool calls

The gate resets when the session ends. Gemini Live failure reports the error and does not fall through to any other route.

### Voice Diagnostics (Privacy-Preserving)

Stored locally at `~/Library/Application Support/HAKI/diagnostics/voice/<date>.jsonl` (mode 0600). Records: turn IDs, stage, outcome, timing (TTFA, transcription, first LLM token, first PCM), memory measurements, selected route, error class. **Raw audio and full transcript text are excluded by default.** A session-scoped user control can enable transcript-level diagnostics; it expires at session end.

---

## New — HAKI Brain Memory Processing Pipeline

The previous `HAKIBrain._ingest_file()` sent every file to the LLM unconditionally. The new **Architect's Hybrid Pipeline** routes files through a fast deterministic pass first — the LLM is only invoked when the fast pass finds nothing.

### Two-Pass Design

**Fast Pass — `FastPassExtractor`** (CPU-only, zero LLM cost)
- Uses `spaCy en_core_web_sm` to extract: PERSON, ORG, GPE, DATE, EVENT, PRODUCT
- Regex patterns: EMAIL, URL, PHONE, HAKI_MARKER (`#haki:tag`)
- Hindi/Hinglish fallback regex when spaCy finds nothing in non-English content
- Deduplicates by `(text, label)` — deterministic, idempotent
- Writes one `Memory_Note` per entity to `wiki/` and upserts into ChromaDB
- spaCy model is ~15 MB, loaded once at startup, resident for the session lifetime

**Heavy Pass — `HeavyPassExtractor`** (invoked only when Fast Pass yields nothing)
- Queries ChromaDB for top-3 semantically similar existing Memory_Notes
- Builds an **evolutionary prompt**: merges old memory + new content, asks Bonsai-8B to produce an updated note with an `EVOLVED_FROM: [[old-note-name]]` link
- When ChromaDB is empty, uses a fresh-synthesis prompt instead
- Calls `LLMRouter.chat(prefer_local=True)` → routes to Bonsai-8B MLX (~1.28 GB) only, no cloud
- 30 s timeout; file stays in `raw/` on timeout or empty response (retried next run)
- Low-memory guard: if `psutil` reports < 500 MB available, Heavy Pass is deferred

### MemoryNoteWriter

- Writes atomically: `tempfile.mkstemp` → `fsync` → `os.rename` (no partial notes on disk)
- Validates all wiki links before writing — provenance link and evolutionary link must resolve to existing vault files
- Note names: `{concept_slug}_{YYYYMMDD_HHMMSS}.md` (UTC timestamp, collision-safe)
- Updates ChromaDB after every successful write

### PipelineScheduler (APScheduler)

Replaces the old `start_watching()` polling loop. Two independent background jobs:

| Job | Trigger | Calls |
|-----|---------|-------|
| Raw processing | Every 30 min (configurable via `HAKI_PIPELINE_RAW_INTERVAL_MINUTES`) | `HAKIBrain.ingest_pending()` |
| Conversation processing | Daily at 02:00 (configurable via `HAKI_PIPELINE_CONV_RUN_TIME`) | `HAKIBrain.process_pending_conversations()` |

Both jobs use `max_instances=1` + `coalesce=True` — if the previous run is still in progress, the next trigger is skipped, not queued.

### Conversation Processing

Daily conversation logs in `conversations/YYYY-MM-DD.md` are processed by the same Fast/Heavy pass pipeline at the scheduled time. Unlike raw files, **conversation logs are never moved** — they stay in `conversations/`. A SQLite `ProcessTracker` (`~/.haki/pipeline_tracker.db`) records which logs have been processed to prevent re-ingestion.

### Memory Note Format

Fast Pass notes include entity type, surface form, extraction source, and a provenance link. Heavy Pass notes include the Bonsai-8B synthesized content plus an `evolved_from` frontmatter field linking to the previous memory note that was updated. Both types follow Obsidian wiki-link conventions.

### Hardware Budget (M2 / 8 GB)

| Component | Memory | Pattern |
|-----------|--------|---------|
| spaCy `en_core_web_sm` | ~15 MB | Resident after first call |
| ChromaDB + embeddings | ~85 MB | Always resident (shared) |
| Bonsai-8B MLX 1-bit | ~1.28 GB | On-demand, GC'd after each call |
| **Pipeline peak** | **~1.38 GB** | During Heavy Pass only |

### Configuration

```bash
# Core/.env
HAKI_PIPELINE_RAW_INTERVAL_MINUTES=30   # 1-1440
HAKI_PIPELINE_CONV_RUN_TIME=02:00       # HH:MM 24h
HAKI_PIPELINE_LOW_MEMORY_THRESHOLD_MB=500
HAKI_PIPELINE_LLM_TIMEOUT_SECS=30
```

---

## New — Gemini Sidecar GUI Agent

HAKI can now autonomously control your Mac screen using a **See → Think → Act → Verify** cognitive loop powered by Gemini 2.5 Flash vision. The legacy `ScreenAgent` / `MacController` stack has been replaced.

### Architecture

```
User voice intent
  --> VAD/STT --> Qwen3 LLM emits {"tool": "gui_agent.spawn", "task_description": "..."}
  --> VoiceToolAdapter validates SpawnGuiAgentCall schema
  --> XTTS ack: "On it, starting task..."  (<= 500 ms, Pipecat NOT blocked)
  --> threading.Thread(daemon=True) with its own asyncio event loop
       --> SidecarAgentLoop.run(task_description)
             --> See:    GeminiVisionClient.request_frame() from Swift sidecar
             --> Think:  Gemini 2.5 Flash --> GeminiAction (bbox + action_type)
             --> Act:    MacQuartzExecutor.dispatch() --> CGEventPost HID events
             --> Verify: request_frame() again, broadcast AGENT_STEP over IPC
             --> repeat up to 20 steps or action_type == "done"
```

### Swift ScreenCaptureKit Sidecar

- Swift process captures the display via `SCStreamConfiguration` at 2560×1600 (native Retina)
- Maintains a single-slot `FrameBuffer` actor — always the latest frame, no queue buildup
- Listens on `~/.haki/sidecar_frames.sock` (permissions: `0600`, same-UID only)
- Handshake: sends `{"display_scale": 2.0, "width": 2560, "height": 1600}` on connect
- Protocol: client sends `REQUEST_FRAME\n` → server replies with 4-byte length prefix + JPEG bytes
- JPEG quality 85% by default (configurable via `--jpeg-quality`); frames deleted from memory immediately after use

### GeminiVisionClient

- Model: `gemini-2.5-flash` via `google-generativeai` SDK
- API key read exclusively from `HAKI_GEMINI_API_KEY` env var — never passed as parameter, never logged
- 15 s timeout per Gemini call; `GeminiAPIError` (retriable) on non-200 or timeout
- Bounding box conversion: Gemini returns 0–1000 normalized coords → converted to native 2560×1600 pixels
- Frame bytes deleted in the same coroutine scope that created them (`del frame_bytes` in `finally`)

### MacQuartzExecutor

- Uses `pyobjc-framework-Quartz` — raises `ExecutorUnavailableError` at instantiation if not installed
- Queries `CGMainDisplayID` at runtime for live `DisplayScale` (2560 / 1280 = 2.0 on M2 Air)
- Converts native pixel center to logical coordinate by dividing by `DisplayScale`
- Dispatches `kCGEventLeftMouseDown` + `kCGEventLeftMouseUp` via `CGEventPost(kCGHIDEventTap, ...)`
- Keyboard input via `CGEventCreateKeyboardEvent` — no AppleScript, no pyautogui

### SidecarAgentLoop

- Runs in an isolated daemon thread with its own `asyncio` event loop — Pipecat voice pipeline is never blocked
- Up to 20 steps per task; 3 retries per step on retriable errors (2 s delay)
- Broadcasts `AGENT_EVENT` messages over `JSONIPCServer` to all connected Swift clients:
  `agent_start`, `agent_step`, `agent_done`, `agent_error`, `agent_max_steps_reached`, `agent_hitl_pause`, `agent_hitl_resume`

### Human-in-the-Loop (HITL) Bridge

When the agent detects an `AXSecureTextField` (e.g. an OTP or password field):
1. Loop pauses; broadcasts `agent_hitl_pause` over IPC
2. Swift UI forwards the pause to the Pipecat bus
3. HAKI speaks: "Please tell me the OTP"
4. User speaks the answer; Pipecat STT transcribes it
5. `HITLBridge.inject_response(text)` resumes the loop
6. `MacQuartzExecutor.type_text()` types the injected text
7. The injected text is held in memory only — **never logged, never persisted**
8. 60 s timeout: if no response, loop emits `agent_error` and terminates

---

## New — Persistent Memory Vault Fix

The previous code fell back to `~/Obsidian/HAKI_Brain` when `HAKI_OBSIDIAN_VAULT` was unset. That is an external path that most users never have. All `log_conversation` writes succeeded but landed in the wrong folder — the project-local `HAKI_Brain/` that you have open in Obsidian never received any files.

### What Changed

`Core/haki_core_service.py` now resolves the vault path via a private helper:

```python
def _resolve_haki_brain_vault() -> Path:
    env_value = os.environ.get("HAKI_OBSIDIAN_VAULT", "").strip()
    if env_value:
        return Path(env_value)
    # Default: project-local HAKI_Brain/ folder
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "HAKI_Brain"
```

- If `HAKI_OBSIDIAN_VAULT` is set and non-empty → that path is used (no change to existing behavior)
- If unset or empty → defaults to `{project_root}/HAKI_Brain` (the folder inside the repo)
- The resolved path is logged at startup so misconfigurations are immediately visible

### Core/.env Update

```bash
# The env var can now be left unset — the project-local HAKI_Brain/ is the default
# HAKI_OBSIDIAN_VAULT=/Users/harshkumarroy/Downloads/HKR/HAKI/HAKI_Brain  # optional override
```

### Vault Structure (unchanged)

```
HAKI_Brain/
|-- raw/           <- Drop PDFs, images, text files here
|-- processed/     <- HAKI moves files here after ingestion
|-- wiki/          <- HAKI writes Memory_Notes here
|-- conversations/ <- Daily YYYY-MM-DD.md conversation logs
```

`HAKIBrain.init()` creates all four subdirectories on startup if they do not exist.

---

## New — SwiftUI Frontend (HAKIFrontend)

A production-quality SwiftUI 5 / AppKit frontend has been added as the `HAKIFrontend` SPM executable target. It replaces the legacy `AppDelegate` + `main.swift` entry point with a proper `@main` SwiftUI app.

### State Machine (HAKIStateModel)

`HAKIState` is the single source of truth for all visual surfaces. `HAKIStateModel` is `@MainActor @Observable` — mutations from background tasks use `Task { @MainActor in ... }` trampolines.

| State | Accent Color | Particle Rate | Trigger |
|-------|-------------|---------------|---------|
| `.idle` | Cyan | 20/s | Default |
| `.listening` | Green | 80/s | VAD speech / mic tap |
| `.thinking` | Purple | 60/s | LLM tokens flowing |
| `.speaking` | Light blue | 100/s | `speakingStarted` IPC event |
| `.agent` | Orange | 50/s | GUI agent spawned |
| `.error` | Red | 120/s | Error from any subsystem |

State transitions post `haki.agentModeActivated` / `haki.agentModeDeactivated` notifications automatically, which `ScreenOverlayManager` observes.

### Window Layers

**1. Main Workspace (`WindowGroup`)**
- `NavigationSplitView`: date-grouped conversation sidebar + detail column
- Detail: `JARVISParticleView` (SceneKit 3D JARVIS HUD, 240 pt tall) above conversation timeline
- Floating command bar at bottom: `TextField`, file drop target, `WaveformView`, mic toggle
- Non-modal error banner overlay when `currentState == .error`
- Background: `.ultraThinMaterial` throughout

**2. Menu Bar Dropdown (`MenuBarExtra`)**
- Brain icon in menu bar; style `.window`
- 300 pt wide panel: IPC connection status, audio input/output device names, voice mode toggle, Open HAKI, Quit
- Audio device names fetched live via `AVCaptureDevice` and `AVAudioSession`

**3. Global Hotkey HUD (`FloatingPanelManager`)**
- `Option + Space` registered via Carbon `RegisterEventHotKey`
- `NSPanel` (`.nonactivatingPanel`, borderless, `.floating` level, transparent background)
- 480 × 72 pt `NSHostingView<HotkeyPanelView>` with auto-focused `TextField`
- `Return` → post `haki.hotkeyCommand` notification + dismiss; `Escape` → dismiss only

**4. Screen Control Overlay (`ScreenOverlayManager`)**
- Full-screen `NSWindow` at `.screenSaver` level, `ignoresMouseEvents: true`
- Transparent background; orange 6 pt `RoundedRectangle` stroke pulsates (0.6 ↔ 1.0 opacity, 0.8 s)
- Shown automatically when GUI agent activates; hidden when agent finishes

### JARVISParticleView (SceneKit)

- `NSViewRepresentable` wrapping `SCNView`, `SCNSceneRendererDelegate` for per-frame updates
- Three torus rings (radii 1.0, 1.4, 1.8) rotating continuously on distinct axes
- Central sphere with `SCNParticleSystem`; birth rate and scale driven by `audioLevel` × `particleEmissionRate`
- `.thinking` state: faster ring rotations; `.error` state: jitter action; `.idle` state: gentle ambient pulse
- All ring emission colors match `HAKIState.accentColor`; no external 3D model files required

### Audio Reactivity

AVAudioEngine input tap (1024 samples) computes RMS amplitude on the realtime thread, clamps to [0, 1], and writes to `stateModel.audioLevel` on the main actor. When not in `.listening` state, `audioLevel` is forced to `0.0` so the particle system stays calm.

### IPC Integration

Background `Task` reads `JSONIPCClient.inbound` async stream and maps messages to state:

```
speakingStarted  --> .speaking
speakingStopped  --> .idle
partialTranscript --> .listening
llmToken(isLast: true) --> .speaking
AGENT_EVENT(agent_start) --> .agent
AGENT_EVENT(agent_done / agent_error) --> .idle
```

Connection retry: 12 attempts × 5 s intervals. After 12 failures, logs a terminal error and stops.

### SPM Target

```swift
// Package.swift
.executableTarget(
    name: "HAKIFrontend",
    dependencies: ["HAKIIPC", "HAKIAudio", "HAKIPermissions"],
    path: "Sources/HAKIFrontend"
)
```

All existing subsystem targets (`HAKIAudio`, `HAKICapture`, `HAKIIPC`, `HAKIStore`, `HAKIPermissions`, `HAKIOSActions`, `HAKIControl`, `HAKIScheduler`, `HAKITextInput`, `HAKIUI`) are retained.

---

## LLM Routing Strategy

### Voice Path (VoiceLLMRouter)
Used exclusively for live voice sessions. Does NOT use the broad `LLMRouter`.

- **Default**: `Qwen3-4B-Instruct-4bit` via `mlx-lm` + Metal (fully local, no cloud)
- **Gemini Live escalation**: explicit user opt-in per session + qualifying condition required

### Non-Voice / Text Path (LLMRouter)

| Scenario | Route |
|----------|-------|
| Fast conversations (< 8k tokens) | Groq → Cerebras (Llama-3.3-70B / Llama-4-Scout) |
| Large context (> 8k tokens) | Gemini 2.0 Flash (1M token window, free tier) |
| Memory pipeline (prefer_local=True) | Bonsai-8B MLX first, xLAM-2-3b fallback |
| Offline / no API keys | mlx-community/xLAM-2-3b-fc-r-4bit (~1.85 GB) |

**Zero GPU usage** — all local models run on ANE + CPU via MLX framework.

---

## Embeddings and RAG

### Local Engine
- **Model**: IBM Granite ModernBERT (~97 MB), context 32,768 tokens
- **Languages**: Multilingual, Hindi/Hinglish code-mix supported
- **Storage**: ChromaDB persistent (SQLite/Parquet on SSD) at `~/.haki/chroma_db`

### API Fallback
- Gemini Embedding 2 (`text-embedding-004`), free tier, for large batch operations

### 3-Folder HAKI Brain

```
HAKI_Brain/
|-- raw/           <- Drop files here (PDF, images, markdown, text)
|-- processed/     <- HAKI moves files here after successful ingestion
|-- wiki/          <- HAKI writes Memory_Notes (Fast Pass + Heavy Pass)
|-- conversations/ <- Auto-generated daily conversation logs
```

Ingestion flow:
1. `PipelineScheduler` triggers `ingest_pending()` every 30 min
2. `FastPassExtractor` runs spaCy + regex (milliseconds, free)
3. If entities found: write one Memory_Note per entity → embed → move to `processed/`
4. If nothing found: `HeavyPassExtractor` queries top-3 similar notes, calls Bonsai-8B with evolutionary prompt → write Heavy Pass note → embed → move to `processed/`
5. Daily at 02:00: `process_pending_conversations()` processes yesterday's conversation logs through the same pipeline (files stay in `conversations/`)

---

## Mood Detection

**Hardware-native DSP** — no separate ML model.

Swift Audio subsystem uses Apple Accelerate framework to analyze each captured audio frame:
- **RMS energy** via `vDSP_rmsqv`
- **Fundamental frequency** via autocorrelation

| Tag | Condition | HAKI Response |
|-----|-----------|---------------|
| `ANGRY_SHOUT` | High RMS + High f0 | Calming Hinglish: "Ok chill out boss..." |
| `SAD_LOW_ENERGY` | Low RMS + Low f0 | Encouraging tone |

Mood tags are injected as hidden metadata in the transcript (`[METADATA: MOOD=ANGRY_SHOUT] ...`). The Orchestrator strips the tag and adjusts the system prompt.

---

## System Requirements

### Hardware
- **macOS 14+** (Sonoma or later)
- **Apple Silicon** (M1/M2/M3 — tested on M2)
- **8 GB unified memory minimum** (architecture is specifically optimized for this constraint)

### Software
- **Xcode Command Line Tools**: `xcode-select --install`
- **Python 3.11+**: Built-in with macOS Sonoma
- **espeak-ng**: `brew install espeak-ng` (required for TTS phonemization)
- **Swift 5.9+**: Included with Xcode

### Optional (but recommended) API Keys

| Service | Use | Free Tier |
|---------|-----|-----------|
| Groq | Fast LLM (text turns) | 30 req/min |
| Cerebras | LLM fallback | Rate-limit backup |
| Gemini | Large context + Gemini Live gate + GUI agent vision | 1M token window |
| Cartesia | TTS cloud fallback (legacy, now optional) | Free tier |
| Deepgram | STT cloud (legacy, now optional) | $200 credits |

---

## Quick Start

### 1. Install System Dependencies
```bash
brew install espeak-ng
xcode-select --install
```

### 2. Install Python Dependencies
```bash
cd /path/to/HAKI/Core
pip3 install -r requirements.txt

# For the memory pipeline (spaCy model)
python3 -m spacy download en_core_web_sm

# For the Gemini GUI agent
pip3 install pyobjc-framework-Quartz
```

### 3. Configure API Keys
```bash
# Edit Core/.env
HAKI_GROQ_API_KEY=your_key
HAKI_CEREBRAS_API_KEY=your_key
HAKI_GEMINI_API_KEY=your_key

# Vault path — leave unset to use the project-local HAKI_Brain/ folder (recommended)
# HAKI_OBSIDIAN_VAULT=/custom/path  # only set if you want a different location
```

### 4. Provide your voice file for XTTS
```bash
# Place a clean 6-30 second WAV recording of your voice at:
Core/my_voice.wav
```

### 5. Run HAKI
```bash
cd /path/to/HAKI
./start_haki.sh
```

The startup script will:
1. Load API keys from `Core/.env`
2. Validate and create `HAKI_Brain/` subfolder structure
3. Install Python dependencies if missing
4. Build the Swift app (release mode)
5. Start the Python Core service (JSON IPC server + voice server)
6. Wait for the UNIX socket to appear
7. Launch the `HAKIFrontend` SwiftUI app

### 6. Manual Startup (Debugging)

**Terminal 1 — Python Core**
```bash
cd HAKI/Core
source ../Misc/haki_env.sh
python3 haki_core_service.py --socket ~/.haki/haki_core.sock --transport json
```

**Terminal 2 — Swift Frontend**
```bash
cd HAKI/HAKI
swift run HAKIFrontend
```

### 7. First Run Checklist

1. **Grant permissions**: Microphone, Screen Recording, Accessibility in System Settings
2. **Model downloads** (automatic, ~300 MB total):
   - Granite ModernBERT embeddings (~97 MB)
   - Qwen3-4B-Instruct-4bit voice LLM (~2.3 GB, downloads to `~/.cache/huggingface`)
   - XTTS v2 model (~1.8 GB)
3. **Test voice**: Speak naturally — HAKI routes to Qwen3 locally, no cloud required
4. **Test memory**: Drop a PDF in `HAKI_Brain/raw/` — within 30 min it appears in `wiki/`
5. **Test GUI agent**: Say "Open Safari and search for the weather in Delhi"

---

## Project Structure

```
HAKI/
|-- HAKI/                              # Swift package
|   |-- Sources/
|   |   |-- HAKI/                      # Legacy entry point (retained)
|   |   |-- HAKIFrontend/              # NEW: SwiftUI @main app
|   |   |   |-- HAKIApp.swift          # @main, WindowGroup, MenuBarExtra
|   |   |   |-- MainWorkspaceView.swift
|   |   |   |-- JARVISParticleView.swift  # SceneKit 3D JARVIS HUD
|   |   |   |-- FloatingPanelManager.swift  # Option+Space HUD
|   |   |   +-- ScreenOverlayManager.swift  # Agent mode full-screen overlay
|   |   +-- Subsystems/
|   |       |-- Audio/                 # VoiceAudioController, LocalASRAdapter,
|   |       |                          # AudioFrameRing, PCMPlaybackRenderer
|   |       |-- Capture/              # ScreenReader (SCKit sidecar)
|   |       |-- IPC/                  # JSONIPCClient, VoiceSocketClient
|   |       |-- Permissions/          # PermissionManager
|   |       |-- UI/                   # Menu bar (legacy)
|   |       |-- Store/                # HAKIStore (SQLite.swift)
|   |       |-- Control/              # HAKIControl
|   |       |-- Scheduler/            # HAKIScheduler
|   |       +-- TextInput/            # HAKITextInput
|   |-- Tests/
|   |   |-- HAKITests/                # Unit + integration
|   |   |-- HAKIPropertyTests/        # SwiftCheck property tests
|   |   +-- Voice/                    # AVFoundation voice hardware tests
|   +-- Package.swift
|
|-- Core/                             # Python package
|   |-- core/
|   |   |-- voice/                    # NEW: full voice pipeline
|   |   |   |-- session.py            # VoiceSession lifecycle
|   |   |   |-- pipeline.py           # Pipecat frame graph
|   |   |   |-- vad.py                # Silero VAD state machine
|   |   |   |-- llm.py                # VoiceLLMRouter, VoiceLocalMLXService
|   |   |   |-- tts.py                # XTTSSentenceAdapter
|   |   |   |-- tools.py              # Pydantic tool schemas
|   |   |   |-- asr_bridge.py         # Transcript ingress, ring slot reader
|   |   |   |-- resources.py          # Memory budget guard
|   |   |   +-- diagnostics.py        # Local JSONL diagnostic store
|   |   |-- gui_agent/                # NEW: Gemini sidecar GUI agent
|   |   |   |-- sidecar_agent_loop.py # See->Think->Act->Verify loop
|   |   |   |-- gemini_vision_client.py  # Gemini 2.5 Flash + sidecar frames
|   |   |   |-- mac_quartz_executor.py   # Quartz HID event dispatch
|   |   |   +-- hitl_bridge.py           # Secure field HITL pause/resume
|   |   |-- memory/                   # HAKIBrain + memory pipeline
|   |   |   |-- haki_brain.py         # 3-folder wiki, conversation logging
|   |   |   |-- memory_brain.py       # Vault-based note store
|   |   |   |-- fast_pass.py          # NEW: spaCy + regex extractor
|   |   |   |-- heavy_pass.py         # NEW: Bonsai-8B evolutionary synthesis
|   |   |   |-- memory_note_writer.py # NEW: atomic wiki note writer
|   |   |   |-- process_tracker.py    # NEW: SQLite conversation log tracker
|   |   |   +-- pipeline_scheduler.py # NEW: APScheduler background jobs
|   |   |-- model_provider/
|   |   |   |-- llm_router.py         # Non-voice LLM routing + prefer_local
|   |   |   +-- embeddings_engine.py  # Granite ModernBERT + ChromaDB
|   |   |-- ipc/
|   |   |   |-- server.py             # JSONIPCServer + AGENT_EVENT broadcast
|   |   |   +-- voice_unix_server.py  # NEW: dedicated voice transcript server
|   |   |-- orchestrator/             # Non-voice intent routing, turn mgmt
|   |   +-- scheduler/               # Task reminder scheduler (unchanged)
|   |-- haki_core_service.py          # Main service entry point
|   +-- requirements.txt             # Pinned production dependencies
|
|-- HAKI_Brain/                       # Local Obsidian vault (project-local)
|   |-- raw/                          # Drop files here for ingestion
|   |-- processed/                    # Files after successful ingestion
|   |-- wiki/                         # Memory_Notes (Fast + Heavy Pass)
|   +-- conversations/               # Daily YYYY-MM-DD.md logs
|
|-- legacy_pipeline_backup/           # Archived legacy voice artifacts (inert)
|   |-- inventory.jsonl              # SHA-256 manifest of all archived files
|   +-- README.md                    # Archive provenance notes
|
|-- Misc/
|   |-- haki_env.sh                   # API key exports
|   +-- api_keys_setup.md
|
+-- start_haki.sh                    # Unified startup script
```

---

## Pinned Production Dependencies

```
# Core/requirements.txt
grpcio==1.64.1
grpcio-tools==1.64.1
protobuf==5.29.6
PyYAML==6.0.3
python-dotenv==1.0.1

# Voice runtime (pinned, do not range-relax)
pipecat-ai==1.4.0
mlx-lm==0.18.1
TTS==0.22.0
silero-vad==5.1.2
torch==2.5.1
torchaudio==2.5.1
soundfile==0.12.1
numpy==1.26.4
psutil==6.0.0
pydantic==2.8.2

# Memory pipeline (add to requirements.txt if not present)
spacy>=3.7          # + python -m spacy download en_core_web_sm
apscheduler>=3.10

# GUI agent
pyobjc-framework-Quartz  # install separately: pip install pyobjc-framework-Quartz
google-generativeai
```

---

## Troubleshooting

### "Socket not found" or IPC connection failed
```bash
# Check Python Core logs
tail -f Core/logs/haki_core.log

# Kill stale socket
pkill -f haki_core_service
rm -f ~/.haki/haki_core.sock
```

### Voice not working / "voice pipeline unavailable"
- Confirm `my_voice.wav` exists at `Core/my_voice.wav`
- Confirm Qwen3-4B model is fully downloaded: check `~/.cache/huggingface`
- Check diagnostics: `cat "~/Library/Application Support/HAKI/diagnostics/voice/$(date +%Y-%m-%d).jsonl"`
- Verify microphone permission in System Settings > Privacy > Microphone

### HAKI Brain not writing to HAKI_Brain/wiki/
- Confirm `HAKI_OBSIDIAN_VAULT` is unset in `Core/.env` (project-local default will be used)
- Or set it explicitly: `HAKI_OBSIDIAN_VAULT=/path/to/HAKI/HAKI_Brain`
- Check logs for `[HAKIBrain]` lines; vault path is logged at startup
- Verify spaCy model is installed: `python3 -c "import spacy; spacy.load(\"en_core_web_sm\")"`

### Gemini GUI agent not starting
- Confirm `HAKI_GEMINI_API_KEY` is set in `Core/.env`
- Confirm Screen Recording permission granted (System Settings > Privacy > Screen Recording)
- Confirm `pyobjc-framework-Quartz` is installed: `pip3 install pyobjc-framework-Quartz`
- Check for `[SidecarAgentLoop]` lines in the Core service logs

### Swift build errors
```bash
cd HAKI/HAKI
rm -rf .build
swift build --configuration release
```

### Memory pressure / voice turns being rejected
- Check `stage: memory_budget` entries in voice diagnostics JSONL
- Close other memory-intensive apps during voice sessions
- Bonsai-8B (~1.28 GB) loads only during Heavy Pass ingestion — avoid running ingestion and voice simultaneously on 8 GB machines if possible

---

## Privacy and Security

- **Microphone audio never leaves the Swift process** — only normalized transcript text crosses the IPC boundary
- **Frame bytes deleted immediately** after Gemini API call returns (same coroutine scope)
- **HITL injected text** (passwords, OTPs) held in memory only; never logged, never written to disk
- **API keys** read from environment only; never passed as function parameters, never interpolated into logs
- **Voice diagnostics** exclude raw audio and full transcripts by default; session-scoped opt-in required
- **UNIX sockets** created with `0700` parent directory and `0600` socket permissions, owner UID verified
- **Legacy voice artifacts** archived as inert `.txt` files with credentials redacted; no `__init__.py`, not importable
