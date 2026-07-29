# Requirements Document

## Introduction

This feature replaces HAKI's legacy, turn-based Deepgram → Groq → Cartesia voice path with a full-duplex, local-first voice agent for Hindi, Hinglish, and English on an Apple M2 Mac with 8 GB unified memory. The replacement keeps native microphone processing in Swift, transmits clean transcript events over a local UNIX domain socket, and uses a Pipecat asynchronous frame pipeline for turn orchestration, local inference, interruption, and streaming playback.

The normal runtime path is entirely local: VoiceProcessingIO acoustic echo cancellation, local code-switching ASR, Qwen3-4B-Instruct-4bit through `mlx-lm` and Metal, and XTTS v2 through `TTS==0.22.0` using `my_voice.wav`. Gemini Live is an opt-in, session-scoped escalation tier; the system must not invoke Gemini Live unless the User has enabled the tier for the active session and a defined qualifying condition is present. The legacy pipeline is retained only as a tracked static archive and has no executable relationship with the replacement runtime.

## Glossary

- **Realtime_Voice_Agent**: The HAKI subsystem that captures speech, recognizes speech, orchestrates conversational turns, generates responses, synthesizes speech, and manages interruption.
- **Legacy_Voice_Pipeline**: The project-owned runtime code, handlers, scripts, configuration, dependencies, startup paths, and routing that implement or select the Deepgram, Groq, Cartesia, Edge TTS, Kokoro, ChatTTS, or system-speech turn-based voice path.
- **Legacy_Archive**: The repository-tracked `legacy_pipeline_backup/` directory containing static reference copies of Legacy_Voice_Pipeline artifacts and an artifact inventory.
- **Voice_Processing_IO**: The macOS VoiceProcessingIO audio path selected by enabling voice processing on the `AVAudioEngine` input node; Voice_Processing_IO supplies acoustic echo cancellation, noise suppression, and automatic gain control.
- **ASR_Engine**: A local automatic-speech-recognition model selected from Qwen3-ASR or an optimized CoreML or MLX implementation that supports Hindi, Hinglish, and English code-switching.
- **Transcript_Event**: A UNIX-domain-socket message containing a turn identifier, normalized recognized text, finality state, language classification, and capture timestamps, with no raw microphone sample payload.
- **IPC_Channel**: The process-owner-only UNIX domain socket communication path between the Swift audio subsystem and the Python Realtime_Voice_Agent.
- **Pipecat_Pipeline**: The asynchronous Pipecat pipeline that owns voice-turn ordering and uses audio-input, transcription, LLM-text, and TTS-text frames.
- **Silero_VAD**: The local Silero voice-activity detector used to identify speech and silence states.
- **Smart_Turn_Taking**: Turn-end evaluation that combines Silero_VAD speech state, a configurable silence interval, and interruption state before finalizing a User turn.
- **Barge_In**: User speech detected while Assistant audio is playing that cancels the active assistant response and begins capture of the new User utterance.
- **Sentence**: A non-empty response text unit delimited by terminal punctuation or an explicit end-of-response marker and selected for independent TTS synthesis.
- **Playback_Confirmation**: A renderer event proving that all audio samples for one Sentence reached normal playback completion without cancellation.
- **Conversation_Context**: The ordered record of completed User turns and only Assistant Sentences with a Playback_Confirmation that is supplied to later LLM turns.
- **Local_LLM**: `Qwen3-4B-Instruct-4bit` loaded through `mlx-lm` and executed with Metal acceleration on the local Mac.
- **Tool_Call**: A model-produced structured tool name and arguments that conform to a registered tool schema.
- **Obsidian_RAG**: The existing local Obsidian retrieval-augmented-generation capability used to search HAKI's knowledge vault.
- **Screen_Control**: The existing HAKI capability that inspects or controls the macOS screen subject to its existing permission and confirmation rules.
- **Local_TTS**: Coqui XTTS v2 provided by the installed `TTS==0.22.0` package and conditioned with `my_voice.wav`.
- **TTFA**: Time to first audio, measured from the time the Realtime_Voice_Agent receives a complete Sentence to the time the Swift audio renderer receives the first non-empty PCM chunk for that Sentence.
- **Gemini_Live**: The external cloud reasoning tier that may process an eligible turn only after explicit session enablement and a Qualifying_Condition.
- **Cloud_Escalation_Gate**: The component that decides whether Gemini_Live is eligible for a turn.
- **Qualifying_Condition**: One of: Low_Battery, Thermal_Throttling, or Ultra_Complex_Reasoning.
- **Low_Battery**: A battery state of charge at or below 20 percent while external power is disconnected.
- **Thermal_Throttling**: A macOS thermal state reported as `serious` or `critical`.
- **Ultra_Complex_Reasoning**: A request whose assembled prompt exceeds 16,000 tokens or whose validated execution plan requires more than six Tool_Calls.
- **Model_Resident_Footprint**: The memory attributed to loaded Local_LLM model weights and runtime cache, measured after warm-up and before a voice turn begins.
- **Pipeline_Memory**: The resident memory used by the concurrent local ASR, Local_LLM, Local_TTS, Pipecat_Pipeline, and supporting voice-process components.
- **Nominal_Test_Profile**: A warmed macOS 14-or-later Apple M2 computer with 8 GB unified memory, connected to external power, a thermal state below `serious`, and no other HAKI model workload running.
- **Voice_Diagnostic_Event**: A locally recorded structured event containing a turn identifier, stage, outcome, timestamps, selected model route, resource measurements, and failure reason when applicable.

## Assumptions

- The deployment target is macOS 14 or later on an Apple M2 Mac with 8 GB unified memory, and the User grants the microphone permission required by the existing HAKI permission flow.
- `my_voice.wav` is a local, readable voice-conditioning asset supplied by the User before Local_TTS is enabled.
- Existing Obsidian_RAG and Screen_Control services remain the authoritative implementations of retrieval, macOS permission, and consequential-action confirmation behavior.
- Model files are provisioned locally before a normal local voice session; cloud connectivity is not required for a normal local voice session.

## Scope Boundaries

- This feature replaces only the live voice path. It does not redesign the existing Obsidian_RAG data model, Screen_Control safety policy, or non-voice HAKI capabilities.
- Legacy_Archive is static reference material, not a compatibility layer, executable mode, or recovery option.
- Gemini_Live processing is outside the local-default path and is limited to a single explicitly enabled session and a Qualifying_Condition.
- The feature does not add automatic upload, retention, or third-party telemetry for raw microphone audio or transcripts.

## Requirements

### Requirement 1: Legacy Voice-Pipeline Archival and Isolation

**User Story:** As the HAKI owner, I want the previous cloud voice pipeline retained only as a traceable static archive, so that the replacement runtime cannot silently execute or recover through legacy behavior.

#### Acceptance Criteria

1. WHEN the migration inventory is generated, THE Realtime_Voice_Agent SHALL record the repository-relative path, artifact category, and SHA-256 digest for every Legacy_Voice_Pipeline script, handler, configuration artifact, dependency declaration, startup path, and routing rule.
2. WHEN the migration is completed, THE Realtime_Voice_Agent SHALL place a static copy of every inventoried Legacy_Voice_Pipeline artifact under the repository-tracked `legacy_pipeline_backup/` directory.
3. WHEN an inventoried configuration artifact contains a credential, THE Realtime_Voice_Agent SHALL replace the credential value in the Legacy_Archive copy with a non-secret placeholder while preserving the configuration key and artifact path.
4. THE Realtime_Voice_Agent SHALL include an inventory file in Legacy_Archive that maps every archived artifact to its original repository-relative path and SHA-256 digest.
5. WHILE the replacement runtime is installed, THE Realtime_Voice_Agent SHALL exclude Legacy_Archive from runtime imports, dependency loading, configuration discovery, startup commands, service registration, and route selection.
6. IF a replacement voice component fails, THEN THE Realtime_Voice_Agent SHALL report the failure without invoking a Legacy_Voice_Pipeline component through an automatic or User-selected fallback.

### Requirement 2: Native Full-Duplex Capture and Echo Cancellation

**User Story:** As a voice user, I want microphone capture to remain active while HAKI speaks without recognizing HAKI's own output as my speech, so that I can interrupt naturally.

#### Acceptance Criteria

1. WHEN the Swift audio subsystem starts microphone capture, THE Realtime_Voice_Agent SHALL set `inputNode.isVoiceProcessingEnabled` to `true` before installing the microphone tap.
2. WHEN the Swift audio subsystem starts microphone capture, THE Realtime_Voice_Agent SHALL verify that `inputNode.isVoiceProcessingEnabled` is `true` before accepting microphone frames.
3. WHILE microphone capture and Local_TTS playback are active, THE Realtime_Voice_Agent SHALL route microphone capture through Voice_Processing_IO.
4. IF Voice_Processing_IO cannot be enabled or cannot be verified, THEN THE Realtime_Voice_Agent SHALL keep voice capture unavailable, emit a Voice_Diagnostic_Event with stage `voice_processing`, and present an actionable microphone-audio error to the User.
5. WHEN Local_TTS playback starts, THE Realtime_Voice_Agent SHALL continue microphone capture through Voice_Processing_IO for Barge_In detection.
6. WHEN the Swift audio subsystem emits a microphone frame, THE Realtime_Voice_Agent SHALL label the frame with a monotonically increasing sequence number and capture timestamp.

### Requirement 3: Local Code-Switching Recognition and Transcript IPC

**User Story:** As a Hindi-English bilingual user, I want HAKI to recognize Hindi, Hinglish, and English locally and pass clean text to the conversational service, so that voice interactions remain private and natural.

#### Acceptance Criteria

1. THE Realtime_Voice_Agent SHALL use an ASR_Engine selected from Qwen3-ASR or an optimized CoreML or MLX ASR implementation for every normal voice turn.
2. WHEN the ASR_Engine finalizes an utterance containing Hindi, English, or both, THE Realtime_Voice_Agent SHALL emit one final Transcript_Event containing the recognized text and one of the language classifications `hi`, `en`, or `hinglish`.
3. WHEN the ASR_Engine emits a partial recognition result, THE Realtime_Voice_Agent SHALL emit a non-final Transcript_Event with the same turn identifier before the final Transcript_Event for that turn.
4. WHEN the Swift audio subsystem sends a final Transcript_Event, THE IPC_Channel SHALL deliver the Transcript_Event to the Python Realtime_Voice_Agent through a UNIX domain socket.
5. THE IPC_Channel SHALL restrict socket access to the operating-system user that owns the HAKI process.
6. WHILE a normal voice turn is processed, THE IPC_Channel SHALL transmit Transcript_Event payloads without raw microphone sample payloads.
7. IF the ASR_Engine produces empty recognized text for a finalized utterance, THEN THE Realtime_Voice_Agent SHALL emit a Voice_Diagnostic_Event with stage `asr`, refrain from creating an LLM turn, and prompt the User to repeat the utterance.
8. IF the IPC_Channel disconnects before delivery of a final Transcript_Event, THEN THE Realtime_Voice_Agent SHALL discard the unfinished turn, emit a Voice_Diagnostic_Event with stage `ipc`, and present a reconnectable voice-service error to the User.

### Requirement 4: Pipecat Frame-Oriented Turn Orchestration

**User Story:** As a user, I want a responsive asynchronous voice pipeline, so that speech, reasoning, and speech playback do not block each other.

#### Acceptance Criteria

1. THE Realtime_Voice_Agent SHALL use Pipecat_Pipeline asynchronous frames as the authoritative orchestration path for live voice turns.
2. THE Pipecat_Pipeline SHALL represent captured microphone input with audio-input frames.
3. THE Pipecat_Pipeline SHALL represent ASR results with transcription frames.
4. THE Pipecat_Pipeline SHALL represent generated model output with LLM-text frames.
5. THE Pipecat_Pipeline SHALL represent sentence-ready synthesis input with TTS-text frames.
6. WHEN a final Transcript_Event arrives through the IPC_Channel, THE Pipecat_Pipeline SHALL create a transcription frame with the Transcript_Event turn identifier and normalized text.
7. WHEN Silero_VAD detects speech followed by 800 milliseconds of continuous silence, THE Smart_Turn_Taking component SHALL finalize the active User utterance unless a Barge_In is active.
8. WHILE a voice turn is active, THE Realtime_Voice_Agent SHALL preserve frame order for each turn identifier from transcription through terminal completion or cancellation.
9. IF Pipecat_Pipeline initialization fails, THEN THE Realtime_Voice_Agent SHALL keep voice turns unavailable, emit a Voice_Diagnostic_Event with stage `pipecat`, and refrain from starting a custom-thread replacement pipeline.

### Requirement 5: Barge-In, Queue Cancellation, and Played-Sentence Context

**User Story:** As a user, I want to interrupt HAKI immediately without losing the conversational state that I actually heard, so that the conversation behaves like a natural two-way exchange.

#### Acceptance Criteria

1. WHEN Silero_VAD detects at least 200 milliseconds of User speech during Local_TTS playback, THE Realtime_Voice_Agent SHALL declare a Barge_In within 200 milliseconds of the detection threshold.
2. WHEN a Barge_In is declared, THE Realtime_Voice_Agent SHALL stop active Local_TTS playback within 200 milliseconds.
3. WHEN a Barge_In is declared, THE Pipecat_Pipeline SHALL cancel unfinished LLM generation for the interrupted Assistant turn.
4. WHEN a Barge_In is declared, THE Pipecat_Pipeline SHALL flush every queued TTS-text frame and unscheduled audio frame for the interrupted Assistant turn.
5. WHEN a Barge_In is declared, THE Realtime_Voice_Agent SHALL begin collecting the new User utterance without waiting for cancelled synthesis or generation tasks to finish.
6. WHEN the audio renderer emits a Playback_Confirmation for a Sentence, THE Realtime_Voice_Agent SHALL append that Sentence to Conversation_Context in playback order.
7. IF a Sentence is interrupted before a Playback_Confirmation, THEN THE Realtime_Voice_Agent SHALL exclude that Sentence from Conversation_Context.
8. WHILE an Assistant turn is incomplete, THE Realtime_Voice_Agent SHALL provide later Local_LLM turns only the User messages and Assistant Sentences present in Conversation_Context.

### Requirement 6: Local LLM and Structured Tool Calls

**User Story:** As a user, I want local reasoning and existing HAKI tools to remain available during voice conversations, so that private conversation can retrieve knowledge and control the screen without a cloud default.

#### Acceptance Criteria

1. THE Realtime_Voice_Agent SHALL use Local_LLM as the default LLM route for every voice turn that is not eligible for Gemini_Live.
2. WHEN Local_LLM is loaded for a voice session, THE Realtime_Voice_Agent SHALL load `Qwen3-4B-Instruct-4bit` through `mlx-lm` with Metal acceleration.
3. WHEN Local_LLM emits a Tool_Call for Obsidian_RAG, THE Realtime_Voice_Agent SHALL validate the Tool_Call against the registered Obsidian_RAG schema before invoking Obsidian_RAG.
4. WHEN Local_LLM emits a Tool_Call for Screen_Control, THE Realtime_Voice_Agent SHALL validate the Tool_Call against the registered Screen_Control schema before invoking Screen_Control.
5. WHEN a Tool_Call is validated and executed, THE Realtime_Voice_Agent SHALL supply the structured tool result to the originating Local_LLM turn before creating the final response text.
6. IF a Tool_Call does not match a registered schema, THEN THE Realtime_Voice_Agent SHALL reject the Tool_Call, emit a Voice_Diagnostic_Event with stage `tool_call`, and refrain from invoking Obsidian_RAG or Screen_Control.
7. IF Local_LLM cannot load or generates a terminal error, THEN THE Realtime_Voice_Agent SHALL emit a Voice_Diagnostic_Event with stage `local_llm`, present the error to the User, and refrain from selecting Legacy_Voice_Pipeline as a fallback.

### Requirement 7: Local XTTS Sentence Streaming

**User Story:** As a user, I want HAKI to begin speaking complete response sentences promptly in the appropriate language and voice, so that replies feel conversational.

#### Acceptance Criteria

1. THE Realtime_Voice_Agent SHALL use Local_TTS through `TTS==0.22.0` and Coqui XTTS v2 for every normal voice response.
2. WHEN Local_TTS initializes a voice session, THE Realtime_Voice_Agent SHALL condition Local_TTS with the local `my_voice.wav` asset.
3. WHEN a Sentence is classified as English, THE Realtime_Voice_Agent SHALL synthesize the Sentence using the Local_TTS language selection `en`.
4. WHEN a Sentence is classified as Hindi or Hinglish, THE Realtime_Voice_Agent SHALL synthesize the Sentence using the Local_TTS language selection `hi`.
5. WHEN Pipecat_Pipeline produces a Sentence, THE Realtime_Voice_Agent SHALL create a TTS-text frame for that Sentence before waiting for subsequent Sentence generation.
6. WHEN a warmed Local_TTS session receives a complete Sentence under Nominal_Test_Profile, THE Realtime_Voice_Agent SHALL deliver the first non-empty PCM chunk for that Sentence to the Swift audio renderer within 500 milliseconds.
7. WHILE a prior Sentence is playing, THE Realtime_Voice_Agent SHALL permit synthesis of the next complete Sentence without delaying the prior Sentence playback.
8. IF Local_TTS cannot synthesize a Sentence, THEN THE Realtime_Voice_Agent SHALL stop synthesis for the affected Assistant turn, emit a Voice_Diagnostic_Event with stage `local_tts`, and present the complete generated response as on-screen text.

### Requirement 8: Cloud Escalation Eligibility

**User Story:** As a privacy-conscious user, I want Gemini Live to remain disabled unless I explicitly enable it for a justified session condition, so that cloud processing is deliberate rather than an implicit fallback.

#### Acceptance Criteria

1. THE Cloud_Escalation_Gate SHALL initialize Gemini_Live as disabled for every new voice session.
2. WHEN the User explicitly enables Gemini_Live for the active voice session, THE Cloud_Escalation_Gate SHALL record the enablement with the active session identifier and present the enabled state to the User.
3. WHEN the active voice session ends, THE Cloud_Escalation_Gate SHALL disable Gemini_Live for that session.
4. WHERE Gemini_Live is enabled for the active voice session, WHEN Low_Battery, Thermal_Throttling, or Ultra_Complex_Reasoning is present for a turn, THE Cloud_Escalation_Gate SHALL mark Gemini_Live eligible for that turn and record the Qualifying_Condition in a Voice_Diagnostic_Event.
5. IF Gemini_Live is disabled for the active voice session, THEN THE Cloud_Escalation_Gate SHALL route the turn to Local_LLM regardless of Low_Battery, Thermal_Throttling, or Ultra_Complex_Reasoning.
6. IF no Qualifying_Condition is present for a turn, THEN THE Cloud_Escalation_Gate SHALL route the turn to Local_LLM regardless of active-session Gemini_Live enablement.
7. IF Gemini_Live is eligible and invocation fails, THEN THE Cloud_Escalation_Gate SHALL report the failure and SHALL not invoke a Legacy_Voice_Pipeline component.

### Requirement 9: Apple M2 Memory and Responsiveness Limits

**User Story:** As an owner of an 8 GB M2 Mac, I want the complete local voice stack to stay within a predictable memory budget, so that HAKI remains usable alongside macOS.

#### Acceptance Criteria

1. WHEN Local_LLM warm-up completes under Nominal_Test_Profile, THE Realtime_Voice_Agent SHALL measure Model_Resident_Footprint at less than 2.5 GB.
2. WHILE local ASR, Local_LLM, Local_TTS, and Pipecat_Pipeline are concurrently active under Nominal_Test_Profile, THE Realtime_Voice_Agent SHALL maintain Pipeline_Memory at or below 5 GB.
3. WHEN a complete User utterance is finalized under Nominal_Test_Profile, THE Realtime_Voice_Agent SHALL begin playback of the first Assistant Sentence within 1.5 seconds of utterance finalization.
4. IF Model_Resident_Footprint reaches 2.5 GB or Pipeline_Memory exceeds 5 GB, THEN THE Realtime_Voice_Agent SHALL stop admitting new voice turns, release idle model resources, and emit a Voice_Diagnostic_Event with stage `memory_budget`.
5. WHEN resource release restores Model_Resident_Footprint below 2.5 GB and Pipeline_Memory to 5 GB or less, THE Realtime_Voice_Agent SHALL resume admitting new voice turns.
6. THE Realtime_Voice_Agent SHALL declare Pipecat, `mlx-lm`, `TTS==0.22.0`, and Silero_VAD as direct, version-pinned production dependencies.

### Requirement 10: Local Observability and Failure Reporting

**User Story:** As the HAKI owner, I want local diagnostics that explain voice behavior without retaining private audio, so that I can validate performance and resolve failures.

#### Acceptance Criteria

1. WHEN a voice turn starts, THE Realtime_Voice_Agent SHALL create a Voice_Diagnostic_Event containing the turn identifier, selected LLM route, ASR_Engine identifier, Local_TTS identifier, and start timestamp.
2. WHEN a voice turn reaches playback completion or cancellation, THE Realtime_Voice_Agent SHALL record transcription completion, first LLM-text frame, first TTS-text frame, TTFA, terminal outcome, Model_Resident_Footprint, and Pipeline_Memory in the Voice_Diagnostic_Event.
3. WHEN Cloud_Escalation_Gate evaluates a voice turn, THE Realtime_Voice_Agent SHALL record Gemini_Live enablement state, each evaluated Qualifying_Condition, and the selected route in the Voice_Diagnostic_Event.
4. THE Realtime_Voice_Agent SHALL store Voice_Diagnostic_Event records locally.
5. THE Realtime_Voice_Agent SHALL exclude raw microphone audio and full transcript text from Voice_Diagnostic_Event records unless the User enables a session-scoped diagnostic-content control.
6. IF any voice-stage failure occurs, THEN THE Realtime_Voice_Agent SHALL record the stage, error class, turn identifier, and recovery outcome in a Voice_Diagnostic_Event.

## Verification Boundaries

- The requirements require automated tests with synthetic audio, mocked model adapters, and mocked Playback_Confirmation events; hardware-dependent Voice_Processing_IO validation additionally requires a macOS integration test.
- Memory and TTFA limits are evaluated under Nominal_Test_Profile after ASR, Local_LLM, and Local_TTS warm-up. Cold model download time and first-time model conversion are excluded from per-turn timing limits.
- No requirement permits a test fixture, a manual developer command, or a production error path to execute Legacy_Archive as a runtime fallback.
