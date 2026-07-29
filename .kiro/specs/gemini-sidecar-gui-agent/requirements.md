# Requirements Document

## Introduction

This feature replaces HAKI's legacy screen automation stack with a Gemini-Sidecar Architecture for an Autonomous GUI Screen Control Agent on macOS (M2 MacBook Air, 8 GB RAM, 0 GB available for local VLMs). The legacy `ScreenAgent` and `MacController` are moved to a backup directory. A new `SidecarAgentLoop` module becomes the sole GUI automation runtime. It runs in an isolated async background thread, never blocking Pipecat's WebRTC voice loop. Screen frames are captured via a Swift ScreenCaptureKit sidecar over a UNIX socket and analysed by Gemini 2.5 Flash. Physical input is dispatched through pyobjc-framework-Quartz HID events with runtime display-scale correction. Human-in-the-loop (HITL) pauses are communicated over the existing JSON IPC socket using new `AGENT_EVENT` message types, and Pipecat asks the user verbally via XTTS.

---

## Glossary

- **SidecarAgentLoop**: The new autonomous GUI agent loop that replaces `ScreenAgent`. Runs in an isolated async background thread and coordinates the See→Think→Act→Verify cognitive cycle.
- **GeminiVisionClient**: Module responsible for requesting compressed display frames from the Swift sidecar and submitting them to the Gemini 2.5 Flash multimodal API.
- **MacQuartzExecutor**: Module responsible for dispatching physical mouse and keyboard HID events via `pyobjc-framework-Quartz` `CGEventCreateMouseEvent` and related APIs.
- **HITLBridge**: Module responsible for detecting HITL trigger conditions (e.g. OTP/secure input fields), pausing `SidecarAgentLoop`, and forwarding `AGENT_EVENT` messages over the JSON IPC socket so Pipecat can ask the user verbally.
- **SwiftSidecar**: The Swift ScreenCaptureKit process that captures Retina display frames, compresses them to JPEG, and serves them over a UNIX domain socket.
- **Pipecat**: The existing WebRTC voice pipeline (`Core/core/voice/pipeline.py`) that must never be blocked by GUI agent activity.
- **JSON IPC Socket**: The existing `JSONIPCServer` UNIX domain socket transport (`Core/core/ipc/server.py`) extended with new `AGENT_EVENT` message types.
- **DisplayScale**: The macOS Retina scale factor queried at runtime via `CGMainDisplayID` / `CGDisplayScreenSize`. Expected value is `2.0` on the target M2 MacBook Air (2560×1600 native, 1280×800 logical).
- **NativeResolution**: The physical pixel dimensions of the built-in display: 2560×1600.
- **LogicalCoordinate**: A UI coordinate in the macOS logical (points) space, equal to `NativeCoordinate / DisplayScale`.
- **NativeCoordinate**: A raw pixel coordinate from the Gemini bounding box response, in the `[ymin, xmin, ymax, xmax]` format returned by the Gemini 2.5 Flash API.
- **HITL**: Human-in-the-loop. A pause condition in `SidecarAgentLoop` triggered when the agent detects a secure input field, OTP prompt, or explicit uncertainty requiring verbal user confirmation.
- **AGENT_EVENT**: A new JSON IPC message type added to `JSONIPCServer` for communicating real-time GUI agent state changes to the Swift UI and Pipecat bus.
- **LegacyBackup**: The directory `Core/legacy_screen_control_backup/` where `mac_controller.py`, `screen_agent.py`, and related automation modules are archived before replacement.
- **spawn_gui_agent tool**: A new Pipecat LLM tool registered in `Core/core/voice/tools.py` that triggers `SidecarAgentLoop` in a background thread without blocking Pipecat.
- **XTTS**: The existing text-to-speech engine used by Pipecat to deliver verbal responses to the user.

---

## Requirements

### Requirement 1 — Legacy Code Archival

**User Story:** As a developer, I want the legacy screen automation code safely backed up before replacement, so that the original implementation is recoverable without a git revert.

#### Acceptance Criteria

1. THE SidecarAgentLoop System SHALL move `Core/core/automation/mac_controller.py` to `Core/legacy_screen_control_backup/mac_controller.py` before any new GUI agent module is written to `Core/core/`.
2. THE SidecarAgentLoop System SHALL move `Core/core/automation/screen_agent.py` to `Core/legacy_screen_control_backup/screen_agent.py` before any new GUI agent module is written to `Core/core/`.
3. THE SidecarAgentLoop System SHALL create a `Core/legacy_screen_control_backup/README.md` that records the date of archival, the originating file paths, and a one-line description of why the files were archived.
4. WHEN the backup directory `Core/legacy_screen_control_backup/` does not exist, THE SidecarAgentLoop System SHALL create it before moving any files.
5. IF a file already exists at the backup destination path, THEN THE SidecarAgentLoop System SHALL abort the move and log an error rather than overwrite the existing backup.

---

### Requirement 2 — Swift ScreenCaptureKit Sidecar

**User Story:** As a developer, I want a Swift sidecar process that captures Retina display frames and serves them over a UNIX socket, so that Python can receive compressed frames without polling or screen-recording subprocess calls.

#### Acceptance Criteria

1. THE SwiftSidecar SHALL capture the built-in display at NativeResolution (2560×1600) using ScreenCaptureKit's `SCStreamConfiguration`.
2. THE SwiftSidecar SHALL compress each captured frame to JPEG with a quality setting configurable at launch (default: 85%).
3. WHEN a Python client connects to the UNIX domain socket, THE SwiftSidecar SHALL serve the most recently captured JPEG frame as a binary response within 200 ms of the request.
4. THE SwiftSidecar SHALL expose a UNIX domain socket at a path scoped to the application container (e.g. `~/.haki/sidecar_frames.sock`), never a network port.
5. WHILE a Python client is connected, THE SwiftSidecar SHALL maintain a rolling single-frame buffer, replacing the previous frame on each new capture tick.
6. IF ScreenCaptureKit requires `Screen Recording` permission and that permission is not granted, THEN THE SwiftSidecar SHALL exit with a non-zero status code and write a human-readable error to stderr.
7. THE SwiftSidecar SHALL query the `DisplayScale` factor via `CGMainDisplayID` at startup and include it in the initial handshake JSON message sent to connecting clients.

---

### Requirement 3 — GeminiVisionClient

**User Story:** As a developer, I want a Python module that requests frames from the SwiftSidecar and queries Gemini 2.5 Flash, so that the agent can see the screen and receive structured next-action JSON.

#### Acceptance Criteria

1. THE GeminiVisionClient SHALL request a JPEG frame from the SwiftSidecar UNIX socket for each See phase of the cognitive loop.
2. THE GeminiVisionClient SHALL submit the JPEG frame and a structured text prompt to the `gemini-2.5-flash` model using the `HAKI_GEMINI_API_KEY` environment variable as the API key.
3. WHEN the Gemini API responds, THE GeminiVisionClient SHALL parse the response and extract a structured JSON object containing at minimum: `action_type` (string) and `bbox` (`[ymin, xmin, ymax, xmax]` normalized 0–1000 integers as returned by the Gemini 2.5 Flash API).
4. IF the Gemini API returns a non-200 HTTP status or a network timeout after 15 seconds, THEN THE GeminiVisionClient SHALL raise a retriable `GeminiAPIError` with the status code and a content-free error message (no user data in the exception message).
5. THE GeminiVisionClient SHALL never log or store the raw JPEG frame bytes outside of the in-memory request lifecycle.
6. THE GeminiVisionClient SHALL convert Gemini normalized bounding box coordinates (`[ymin, xmin, ymax, xmax]` in 0–1000 space) to NativeCoordinate pixel values by multiplying each component by `NativeResolution / 1000`.

---

### Requirement 4 — MacQuartzExecutor

**User Story:** As a developer, I want a Python module that dispatches physical HID events via pyobjc-framework-Quartz with runtime display-scale correction, so that click coordinates are accurate on Retina displays without pyautogui.

#### Acceptance Criteria

1. THE MacQuartzExecutor SHALL use `CGEventCreateMouseEvent` from `pyobjc-framework-Quartz` to dispatch mouse click events, and SHALL NOT use pyautogui or any subprocess-based click mechanism.
2. WHEN dispatching a click action, THE MacQuartzExecutor SHALL query the DisplayScale at runtime using `CGMainDisplayID` and related Quartz APIs before computing LogicalCoordinates.
3. WHEN converting NativeCoordinates to LogicalCoordinates, THE MacQuartzExecutor SHALL divide each pixel coordinate by the runtime DisplayScale value (expected 2.0).
4. THE MacQuartzExecutor SHALL dispatch a `kCGEventLeftMouseDown` event followed immediately by a `kCGEventLeftMouseUp` event at the computed LogicalCoordinate for every click action.
5. THE MacQuartzExecutor SHALL support keyboard input dispatch via `CGEventCreateKeyboardEvent` for typing and keystroke actions.
6. IF `pyobjc-framework-Quartz` is not importable, THEN THE MacQuartzExecutor SHALL raise an `ExecutorUnavailableError` at instantiation time with a message indicating the missing dependency.
7. THE MacQuartzExecutor SHALL post all CGEvents to the HID event tap using `CGEventPost(kCGHIDEventTap, event)`.

---

### Requirement 5 — SidecarAgentLoop

**User Story:** As a user, I want HAKI to autonomously control my Mac GUI to complete a task I described verbally, so that I do not have to interact with the screen myself.

#### Acceptance Criteria

1. WHEN `spawn_gui_agent(task_description)` is called, THE SidecarAgentLoop SHALL start in an isolated `asyncio` event loop running in a dedicated daemon thread, separate from the Pipecat voice loop thread.
2. THE SidecarAgentLoop SHALL execute a repeating See→Think→Act→Verify cognitive cycle: capture a frame via GeminiVisionClient, submit to Gemini 2.5 Flash for next-action reasoning, execute the action via MacQuartzExecutor, then verify the result with a follow-up frame capture.
3. WHILE the SidecarAgentLoop is running, THE Pipecat voice pipeline (VoiceSessionPipeline) SHALL remain available to accept and process new voice turns without blocking or queue saturation.
4. THE SidecarAgentLoop SHALL limit each task run to a maximum of 20 cognitive loop iterations before emitting a `AGENT_EVENT` of type `agent_max_steps_reached` and terminating the loop.
5. WHEN the SidecarAgentLoop determines the task is complete, THE SidecarAgentLoop SHALL emit an `AGENT_EVENT` of type `agent_done` with a `success: true` flag and a brief natural-language summary.
6. WHEN a cognitive loop iteration fails due to a retriable error (e.g. `GeminiAPIError`), THE SidecarAgentLoop SHALL retry that iteration up to 3 times with a 2-second delay between retries before treating the step as failed.
7. IF the SidecarAgentLoop encounters a non-retriable error, THEN THE SidecarAgentLoop SHALL emit an `AGENT_EVENT` of type `agent_error` and terminate the loop without affecting the Pipecat voice pipeline.

---

### Requirement 6 — spawn_gui_agent Tool Registration

**User Story:** As a user, I want to trigger GUI agent tasks by voice, so that HAKI starts automating the screen immediately after I speak my intent.

#### Acceptance Criteria

1. THE VoiceToolAdapter SHALL register a new tool named `gui_agent.spawn` with the Pipecat LLM service via a `SpawnGuiAgentCall` Pydantic schema in `Core/core/voice/tools.py`.
2. THE SpawnGuiAgentCall schema SHALL require a `task_description` field of type `str` with `min_length=1` and `max_length=500`, and SHALL use `extra="forbid"`.
3. WHEN the LLM emits a `gui_agent.spawn` tool call, THE VoiceToolAdapter SHALL immediately trigger an XTTS audio response acknowledging the task before starting the SidecarAgentLoop.
4. WHEN the LLM emits a `gui_agent.spawn` tool call, THE VoiceToolAdapter SHALL spin up the SidecarAgentLoop in an isolated async background thread and return a `ToolCallResult` with `success=True` within 500 ms of receiving the tool call.
5. THE VoiceToolAdapter SHALL update `VOICE_TOOL_GRAMMAR` in `Core/core/voice/tools.py` to include the `gui_agent.spawn` tool JSON format example.
6. WHILE the SidecarAgentLoop is running in the background, THE Pipecat WebRTC voice loop SHALL continue to accept new audio frames and voice turns without degradation.

---

### Requirement 7 — JSON IPC AGENT_EVENT Extension

**User Story:** As a developer, I want the existing JSON IPC socket to carry structured AGENT_EVENT messages, so that the Swift UI and Pipecat bus receive real-time GUI agent state updates.

#### Acceptance Criteria

1. THE JSONIPCServer SHALL define and handle a new message type constant `MSG_TYPE_AGENT_EVENT = "AGENT_EVENT"` in `Core/core/ipc/server.py`.
2. THE JSONIPCServer SHALL define the following `AGENT_EVENT` sub-types as string constants: `agent_start`, `agent_step`, `agent_done`, `agent_error`, `agent_max_steps_reached`, `agent_hitl_pause`, `agent_hitl_resume`.
3. WHEN an `AGENT_EVENT` message is received from an internal component (e.g. HITLBridge or SidecarAgentLoop), THE JSONIPCServer SHALL broadcast the message to all currently connected IPC clients.
4. THE JSONIPCServer SHALL validate that each inbound `AGENT_EVENT` message contains a `payload.event_type` field matching one of the defined sub-type constants before broadcasting.
5. IF an `AGENT_EVENT` message contains an unrecognised `event_type`, THEN THE JSONIPCServer SHALL log a warning and drop the message without broadcasting it or returning an error to the sender.

---

### Requirement 8 — HITLBridge

**User Story:** As a user, I want HAKI to pause GUI automation and ask me verbally when it needs my input (e.g. an OTP or password), so that secure information is never filled in without my explicit spoken consent.

#### Acceptance Criteria

1. WHEN the SidecarAgentLoop's Verify phase detects a password field, OTP input, or any element whose AX role is `AXSecureTextField`, THE HITLBridge SHALL pause the SidecarAgentLoop and emit an `AGENT_EVENT` of sub-type `agent_hitl_pause` over the JSON IPC socket.
2. WHEN the Pipecat bus receives an `agent_hitl_pause` event, THE Pipecat LLM service SHALL synthesise a verbal XTTS prompt asking the user for the required input (e.g. "Please tell me the OTP.").
3. WHEN the user speaks a response to the HITL prompt, THE HITLBridge SHALL receive the transcribed text from Pipecat's STT pipeline and inject it into the paused SidecarAgentLoop as the value to type.
4. AFTER injecting the user's spoken response, THE HITLBridge SHALL emit an `AGENT_EVENT` of sub-type `agent_hitl_resume` and resume the SidecarAgentLoop from the paused step.
5. IF no spoken response is received within 60 seconds of the `agent_hitl_pause` event, THEN THE HITLBridge SHALL emit an `AGENT_EVENT` of sub-type `agent_error` with message `"HITL timeout: no user response"` and terminate the SidecarAgentLoop.
6. WHILE the SidecarAgentLoop is paused for HITL, THE Pipecat voice pipeline SHALL remain fully active for unrelated voice commands.

---

### Requirement 9 — No Blocking of Pipecat Voice Loop

**User Story:** As a user, I want voice interaction to remain responsive while the GUI agent is running in the background, so that I can still talk to HAKI during automation tasks.

#### Acceptance Criteria

1. THE SidecarAgentLoop SHALL run exclusively in a daemon thread that owns a separate `asyncio` event loop and SHALL NOT schedule coroutines on the Pipecat event loop.
2. THE spawn_gui_agent tool handler SHALL use `threading.Thread(daemon=True)` to launch the SidecarAgentLoop thread and SHALL return control to the Pipecat frame dispatcher within 500 ms.
3. WHILE the SidecarAgentLoop thread is running, THE Pipecat VoiceSessionPipeline audio queue SHALL not drop frames due to GUI agent activity.
4. THE MacQuartzExecutor SHALL NOT call any blocking `asyncio.sleep` or synchronous network call on the Pipecat event loop thread.
5. THE GeminiVisionClient SHALL perform all HTTP requests in the SidecarAgentLoop's own event loop and SHALL NOT share connection pools or state with Pipecat's async context.

---

### Requirement 10 — Security and Privacy

**User Story:** As a user, I want captured screen frames and user input handled securely, so that sensitive information is not persisted or leaked.

#### Acceptance Criteria

1. THE GeminiVisionClient SHALL delete in-memory JPEG frame bytes immediately after the Gemini API response is received, within the same coroutine scope.
2. THE SidecarAgentLoop SHALL NOT write captured frames to disk at any point during normal operation.
3. THE HITLBridge SHALL NOT log, persist, or transmit user-spoken HITL responses (e.g. OTPs) outside of the in-memory injection into the SidecarAgentLoop.
4. THE SwiftSidecar SHALL restrict the UNIX socket file permissions to `0600` (owner read/write only) at creation time.
5. THE GeminiVisionClient SHALL read `HAKI_GEMINI_API_KEY` exclusively from the process environment and SHALL NOT accept the key via any function argument that could be logged.
