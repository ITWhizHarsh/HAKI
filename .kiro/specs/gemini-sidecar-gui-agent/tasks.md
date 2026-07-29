# Implementation Plan: Gemini-Sidecar GUI Screen Control Agent

## Overview

Replace HAKI's legacy `ScreenAgent` / `MacController` stack with a
Gemini-Sidecar Architecture. A Swift ScreenCaptureKit sidecar streams Retina
JPEG frames over a UNIX socket; a Python `SidecarAgentLoop` drives a
See→Think→Act→Verify cognitive cycle using `GeminiVisionClient` and
`MacQuartzExecutor`; the Pipecat voice pipeline is never blocked; and a
`HITLBridge` handles secure-field pauses over the existing JSON IPC socket.

All implementation is in Python (with one Swift component for the sidecar).
Tests use Hypothesis for property-based tests and pytest for unit tests.

---

## Tasks

- [x] 1. Legacy archival — back up and document the replaced modules
  - [x] 1.1 Create archival script and run migration
    - Create `Core/legacy_screen_control_backup/` directory if it does not exist
    - Write a Python script (or inline migration in `__init__.py`) that calls `shutil.move` for `Core/core/automation/mac_controller.py` → `Core/legacy_screen_control_backup/mac_controller.py` and `Core/core/automation/screen_agent.py` → `Core/legacy_screen_control_backup/screen_agent.py`
    - Abort with `FileExistsError` log (no overwrite) if destination already exists
    - After moving both files, write `Core/legacy_screen_control_backup/README.md` with archival date, original paths, and one-line reason
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 1.2 Write unit tests for legacy archival
    - Test: source files are moved, destination created, README fields present
    - Test: `FileExistsError` is raised (and source file unchanged) when backup destination already exists
    - _Requirements: 1.5_

- [x] 2. JSON IPC AGENT_EVENT extension — add constants and broadcast API to `server.py`
  - [x] 2.1 Add `MSG_TYPE_AGENT_EVENT` constant and `AgentEventType` class
    - Add `MSG_TYPE_AGENT_EVENT = "AGENT_EVENT"` constant after existing `MSG_TYPE_*` constants in `Core/core/ipc/server.py`
    - Add `AgentEventType` class with all 7 sub-type string constants: `agent_start`, `agent_step`, `agent_done`, `agent_error`, `agent_max_steps_reached`, `agent_hitl_pause`, `agent_hitl_resume`
    - Add `_VALID` frozenset and `is_valid(cls, event_type)` classmethod
    - _Requirements: 7.1, 7.2_

  - [x] 2.2 Add `_connected_writers` set and `broadcast_agent_event` method to `JSONIPCServer`
    - Add `self._connected_writers: set[asyncio.StreamWriter] = set()` in `JSONIPCServer.__init__`
    - Update `_handle_client` to add writer to `_connected_writers` on connect and remove in `finally`
    - Implement `async def broadcast_agent_event(self, event_type: str, payload: dict) -> None` that validates event type with `AgentEventType.is_valid()`, logs warning and returns on unknown type, serialises and writes to all connected writers, silently discards dead connections
    - _Requirements: 7.3, 7.4, 7.5_

  - [x] 2.3 Handle inbound `AGENT_EVENT` messages in `_dispatch`
    - Add `elif msg_type == MSG_TYPE_AGENT_EVENT:` branch in `_dispatch` that extracts `payload.event_type`, validates with `AgentEventType.is_valid()`, drops unknown sub-types with a warning, and calls `await self.broadcast_agent_event(...)` for valid ones
    - _Requirements: 7.3, 7.4, 7.5_

  - [ ]* 2.4 Write property tests for AGENT_EVENT broadcast (Properties 16 and 17)
    - **Property 16: AGENT_EVENT broadcast reaches all connected clients**
    - **Validates: Requirements 7.3, 7.4**
    - **Property 17: Valid event types pass; unknown types are silently dropped**
    - **Validates: Requirements 7.4, 7.5**

  - [ ]* 2.5 Write unit tests for IPC constants
    - Test that `MSG_TYPE_AGENT_EVENT == "AGENT_EVENT"` and all 7 `AgentEventType` string constants match expected values
    - Test that inbound `AGENT_EVENT` with unknown `event_type` is dropped without broadcasting
    - _Requirements: 7.1, 7.2, 7.5_

- [ ] 3. Checkpoint — IPC extension is complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. GeminiVisionClient — core frame-fetching and Gemini query module
  - [x] 4.1 Create `Core/core/gui_agent/__init__.py` with public exports
    - Create the `Core/core/gui_agent/` package directory
    - Write `__init__.py` that exports: `GeminiVisionClient`, `GeminiAPIError`, `BoundingBox`, `NativePixelBox`, `GeminiAction`, `MacQuartzExecutor`, `ExecutorUnavailableError`, `SidecarAgentLoop`, `HITLBridge`
    - _Requirements: 3.1, 4.1, 5.1, 8.1_

  - [x] 4.2 Implement data classes and `GeminiAPIError` in `gemini_vision_client.py`
    - Create `Core/core/gui_agent/gemini_vision_client.py`
    - Implement frozen dataclasses: `BoundingBox(ymin, xmin, ymax, xmax)` (0–1000 ints), `NativePixelBox(ymin, xmin, ymax, xmax)` (pixel ints), `GeminiAction(action_type, bbox, text, summary)`
    - Implement `GeminiAPIError(status_code, message)` with `retriable = True`
    - _Requirements: 3.3, 3.4, 3.6_

  - [x] 4.3 Implement `GeminiVisionClient.__init__` and `bbox_to_native`
    - Read `HAKI_GEMINI_API_KEY` exclusively from `os.environ` (never as a function argument)
    - Configure `google.generativeai` with the key and store `GenerativeModel("gemini-2.5-flash")`
    - Implement `bbox_to_native(bbox: BoundingBox) -> NativePixelBox` using `NATIVE_WIDTH=2560`, `NATIVE_HEIGHT=1600`, formula: `int(component * dimension / 1000)`
    - _Requirements: 3.2, 3.6, 10.5_

  - [x] 4.4 Implement `perform_handshake` and `request_frame` (async UNIX socket)
    - Implement `async perform_handshake() -> float`: open asyncio UNIX socket to `~/.haki/sidecar_frames.sock`, read first newline-delimited JSON, extract and return `display_scale` float
    - Implement `async request_frame() -> bytes`: send `REQUEST_FRAME\n`, read 4-byte big-endian length prefix, read that many bytes for the JPEG payload
    - _Requirements: 3.1, 2.3_

  - [x] 4.5 Implement `query_gemini` with frame lifecycle and timeout
    - Implement `async query_gemini(task_description: str) -> GeminiAction`
    - Fetch frame via `request_frame()`, submit to Gemini with `asyncio.wait_for(..., timeout=15)`, parse JSON response into `GeminiAction`
    - In `finally` block: `del frame_bytes` (explicit in-scope deletion per Req 10.1)
    - On `asyncio.TimeoutError`: raise `GeminiAPIError(None, "Gemini API timeout")`
    - On non-200 HTTP / network error: raise `GeminiAPIError(status_code, "<fixed error string>")`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 10.1_

  - [ ]* 4.6 Write property tests for GeminiVisionClient (Properties 5, 6, 7)
    - **Property 5: Gemini response parse round-trip**
    - **Validates: Requirements 3.3**
    - **Property 6: Retriable error on non-200 or timeout**
    - **Validates: Requirements 3.4**
    - **Property 7: Bounding box coordinate conversion correctness**
    - **Validates: Requirements 3.6**

  - [ ]* 4.7 Write property test for API key sourcing (Property 20)
    - **Property 20: Gemini API key is sourced exclusively from environment**
    - **Validates: Requirements 10.5**

  - [ ]* 4.8 Write property test for no disk writes (Property 21)
    - **Property 21: No disk writes during frame capture or agent execution**
    - **Validates: Requirements 10.2**

- [x] 5. MacQuartzExecutor — HID event dispatch with runtime display-scale
  - [x] 5.1 Implement `MacQuartzExecutor` class skeleton and `ExecutorUnavailableError`
    - Create `Core/core/gui_agent/mac_quartz_executor.py`
    - Implement `ExecutorUnavailableError(RuntimeError)` with descriptive message
    - In `__init__`: attempt `import Quartz`; on `ImportError` raise `ExecutorUnavailableError` immediately
    - _Requirements: 4.1, 4.6_

  - [x] 5.2 Implement `_get_display_scale` and `_to_logical` helper
    - Implement `_get_display_scale(self) -> float` using `Quartz.CGMainDisplayID()`, `CGDisplayScreenSize`, `CGDisplayBounds`; fall back to `2.0` if `phys.width == 0`
    - Implement `_to_logical(cx_native, cy_native, display_scale) -> tuple[float, float]` as `(cx_native / display_scale, cy_native / display_scale)`
    - _Requirements: 4.2, 4.3_

  - [x] 5.3 Implement `click`, `_post_click`, `type_text`, and `scroll`
    - Implement `click(native_box: NativePixelBox) -> None`: compute center, call `_get_display_scale()`, call `_to_logical`, call `_post_click`
    - Implement `_post_click(lx, ly) -> None`: create `CGPointMake(lx, ly)`, dispatch `kCGEventLeftMouseDown` then `kCGEventLeftMouseUp` via `CGEventPost(kCGHIDEventTap, ...)`
    - Implement `type_text(text: str) -> None`: iterate characters, dispatch via `CGEventCreateKeyboardEvent`
    - Implement `scroll(native_box: NativePixelBox, direction: str, amount: int) -> None`
    - _Requirements: 4.1, 4.4, 4.5, 4.7_

  - [ ]* 5.4 Write property tests for MacQuartzExecutor (Properties 8 and 9)
    - **Property 8: Logical coordinate conversion correctness**
    - **Validates: Requirements 4.3**
    - **Property 9: Every click dispatches exactly two HID events in order**
    - **Validates: Requirements 4.4, 4.7**

  - [ ]* 5.5 Write unit tests for MacQuartzExecutor
    - Test: `ExecutorUnavailableError` raised when `Quartz` import fails (mock importlib)
    - Test: `type_text` calls `CGEventCreateKeyboardEvent` once per character
    - Test: MacQuartzExecutor operations occur on the agent thread, not the Pipecat thread
    - _Requirements: 4.6_

- [x] 6. HITLBridge — secure field detection, pause/resume, and timeout
  - [x] 6.1 Implement `HITLBridge` class with pause/resume event logic
    - Create `Core/core/gui_agent/hitl_bridge.py`
    - Implement `HITLBridge.__init__(ipc_server)` with `asyncio.Event` for pause and resume
    - Implement `should_pause(self) -> bool` returning `True` when current screen state indicates `AXSecureTextField` (use Accessibility API or Gemini metadata flag)
    - _Requirements: 8.1_

  - [x] 6.2 Implement `pause_and_wait` with 60 s timeout
    - Implement `async pause_and_wait(self) -> str | None`: broadcast `agent_hitl_pause`, call `asyncio.wait_for(self._resume_event.wait(), timeout=60.0)`, on `asyncio.TimeoutError` broadcast `agent_error` with `"HITL timeout: no user response"` and raise `HITLTimeoutError`
    - _Requirements: 8.1, 8.5_

  - [x] 6.3 Implement `inject_response`
    - Implement `inject_response(self, text: str) -> None`: store text in `self._injected_text` (never logged/persisted), call `self._resume_event.set()`, then broadcast `agent_hitl_resume`
    - _Requirements: 8.3, 8.4, 10.3_

  - [ ]* 6.4 Write property tests for HITLBridge (Properties 18 and 19)
    - **Property 18: HITL pause occurs for any AXSecureTextField detection**
    - **Validates: Requirements 8.1, 8.5**
    - **Property 19: Injected HITL text is returned unchanged**
    - **Validates: Requirements 8.3**

  - [ ]* 6.5 Write unit tests for HITLBridge
    - Test: `agent_hitl_resume` emitted after `inject_response()`
    - Test: `agent_error` emitted and `HITLTimeoutError` raised after 60 s timeout (mock `asyncio.wait_for`)
    - _Requirements: 8.4, 8.5_

- [ ] 7. Checkpoint — all core modules are complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. SidecarAgentLoop — See→Think→Act→Verify cognitive cycle
  - [x] 8.1 Implement `SidecarAgentLoop` class with state machine and `run` method
    - Create `Core/core/gui_agent/sidecar_agent_loop.py`
    - Implement `SidecarAgentLoop.__init__(ipc_server, vision_client=None, executor=None, hitl_bridge=None)` — accept injected collaborators for testability
    - Implement `async run(task_description: str) -> None` wrapping `_run_loop` and broadcasting `agent_start` at entry and appropriate terminal events on exit
    - Implement `abort(self) -> None` to signal early termination via an internal flag
    - _Requirements: 5.1, 5.5_

  - [x] 8.2 Implement the `_run_loop` with See→Think→Act→Verify and max-steps enforcement
    - Implement `async _run_loop(task_description: str) -> None` iterating `for step in range(1, MAX_STEPS + 1)` (MAX_STEPS = 20)
    - Each iteration: call `_see_think()` → `executor.dispatch(action)` → broadcast `agent_step` → call `_see_think(verify=True)` → check `hitl_bridge.should_pause()`
    - After 20 iterations without `action_type == "done"`, broadcast `agent_max_steps_reached` and return
    - On `action_type == "done"`, broadcast `agent_done` with `success=True` and `summary`
    - _Requirements: 5.2, 5.4, 5.5_

  - [x] 8.3 Implement `_see_think` with 3-retry / 2 s delay on `GeminiAPIError`
    - Implement `async _see_think(task, step, verify=False) -> GeminiAction | None`: retry up to `RETRY_LIMIT=3` times with `await asyncio.sleep(RETRY_DELAY=2.0)` between attempts on `GeminiAPIError`
    - On exhausted retries return `None`; on non-retriable error propagate immediately
    - When `None` returned, caller broadcasts `agent_error` and terminates loop
    - _Requirements: 5.6, 5.7_

  - [ ]* 8.4 Write property tests for SidecarAgentLoop (Properties 11, 13, 14)
    - **Property 11: Cognitive cycle step order is always See→Think→Act→Verify**
    - **Validates: Requirements 5.2**
    - **Property 13: Loop terminates at exactly step 20 with agent_max_steps_reached**
    - **Validates: Requirements 5.4**
    - **Property 14: Retry count and delay for retriable errors**
    - **Validates: Requirements 5.6**

- [x] 9. spawn_gui_agent tool — Pipecat LLM tool registration and thread launch
  - [x] 9.1 Add `SpawnGuiAgentCall` schema and update `VOICE_TOOL_GRAMMAR` in `tools.py`
    - Add `SpawnGuiAgentCall(BaseModel)` with `model_config = ConfigDict(extra="forbid")`, `tool: Literal["gui_agent.spawn"]`, `schema_version: int = 1`, `task_description: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]`
    - Update `VOICE_TOOL_GRAMMAR` to append `{"tool": "gui_agent.spawn", "schema_version": 1, "task_description": "<task description>"}` example line
    - Update `_validate_tool_call` to handle `tool == "gui_agent.spawn"` and return `SpawnGuiAgentCall`
    - _Requirements: 6.1, 6.2, 6.5_

  - [x] 9.2 Add `_start_agent_in_thread` helper and `VoiceToolAdapter._execute_gui_agent_spawn`
    - Add module-level `_start_agent_in_thread(task_description, ipc_server)` that creates `asyncio.new_event_loop()`, wraps it in a `threading.Thread(daemon=True, name="haki-gui-agent")`, and starts it
    - Update `VoiceToolAdapter.__init__` to accept `ipc_server` and `xtts_sink` optional kwargs
    - Implement `async _execute_gui_agent_spawn(call, *, turn_id, session_id) -> ToolCallResult`: first `await self._xtts_sink(ack_text)` if configured, then call `_start_agent_in_thread`, return `ToolCallResult(success=True)` — entire method completes within 500 ms
    - Route `SpawnGuiAgentCall` instances in `execute_tool_call` to `_execute_gui_agent_spawn`
    - _Requirements: 6.1, 6.3, 6.4, 9.2_

  - [ ]* 9.3 Write property tests for SpawnGuiAgentCall (Property 15)
    - **Property 15: SpawnGuiAgentCall rejects invalid task_description lengths**
    - **Validates: Requirements 6.2**

  - [ ]* 9.4 Write property tests for thread isolation (Property 10)
    - **Property 10: Agent runs in isolated daemon thread with its own event loop**
    - **Validates: Requirements 5.1, 9.1**

  - [ ]* 9.5 Write unit tests for spawn_gui_agent tool handler
    - Test: XTTS ack fires before agent thread starts (mock sequencing with `threading.Barrier`)
    - Test: `ToolCallResult(success=True)` returned within 500 ms of receiving a valid `SpawnGuiAgentCall`
    - Test: `VOICE_TOOL_GRAMMAR` contains string `"gui_agent.spawn"`
    - _Requirements: 6.3, 6.4, 6.5_

- [ ] 10. Checkpoint — all Python modules are wired end-to-end
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Swift ScreenCaptureKit sidecar (`ScreenReader.swift`)
  - [x] 11.1 Implement `FrameBuffer` actor and `SCStream` integration
    - Create `HAKI/Sources/Subsystems/Capture/ScreenReader.swift`
    - Implement `FrameBuffer` actor with `update(_ frame: Data)` and `latest() -> Data?`
    - Implement `SCStreamConfiguration` targeting 2560×1600 pixels; check and exit non-zero if `Screen Recording` permission is denied
    - Implement JPEG compression callback (default quality 85%, `--jpeg-quality` CLI override) and push each frame into `FrameBuffer`
    - _Requirements: 2.1, 2.2, 2.6_

  - [x] 11.2 Implement `SidecarServer` UNIX socket with handshake protocol
    - Bind UNIX domain socket to `~/.haki/sidecar_frames.sock` and set file permissions to `0600` after bind
    - On client connect: send handshake JSON `{"display_scale": <float>, "width": 2560, "height": 1600}\n`
    - Display scale is queried via `CGMainDisplayID` at startup
    - On `REQUEST_FRAME\n` from client: read latest frame from `FrameBuffer`, send 4-byte big-endian length prefix followed by JPEG bytes; respond within 200 ms
    - _Requirements: 2.3, 2.4, 2.5, 2.7, 10.4_

- [ ] 12. Property-based tests for pipeline-isolation properties (Properties 4 and 12)
  - [ ]* 12.1 Write property test for handshake DisplayScale (Property 4)
    - **Property 4: Handshake always contains DisplayScale**
    - **Validates: Requirements 2.7**
    - Use a mock sidecar server that emits configurable handshake JSON

  - [ ]* 12.2 Write property test for Pipecat pipeline availability (Property 12)
    - **Property 12: Pipecat pipeline availability is unaffected while agent runs**
    - **Validates: Requirements 5.3, 6.6, 8.6, 9.3**
    - Poll `VoiceSessionPipeline.availability` every 100 ms while a mock `SidecarAgentLoop` runs

- [x] 13. Integration wiring — connect all components in `haki_core_service.py`
  - [x] 13.1 Wire `JSONIPCServer` with `_connected_writers` into `SidecarAgentLoop` and `HITLBridge`
    - Update `haki_core_service.py` (or the service entry point) to pass the running `JSONIPCServer` instance when constructing `SidecarAgentLoop` and `HITLBridge`
    - Update `VoiceToolAdapter` construction to receive `ipc_server` reference so `_execute_gui_agent_spawn` can pass it to `_start_agent_in_thread`
    - _Requirements: 5.1, 7.3, 8.1_

  - [x] 13.2 Ensure `gui_agent` package is importable from `Core/`
    - Verify `Core/core/gui_agent/__init__.py` exports all public symbols
    - Add any missing `__all__` entries; ensure no circular imports with `core.ipc.server`
    - _Requirements: 3.1, 4.1, 5.1, 8.1_

  - [ ]* 13.3 Write integration smoke tests
    - Test: `Core/legacy_screen_control_backup/` exists with both legacy files after archival
    - Test: `VOICE_TOOL_GRAMMAR` string contains `"gui_agent.spawn"` after module import
    - Test: end-to-end mock — `spawn_gui_agent` triggers thread start and first `AGENT_STEP` event is broadcast (using a fully mocked `GeminiVisionClient` and `MacQuartzExecutor`)
    - _Requirements: 1.1–1.3, 5.1–5.2, 6.5_

- [ ] 14. Final checkpoint — full test suite passes
  - Ensure all tests pass, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at major integration boundaries
- Property tests validate 21 universal correctness properties from the design document using Hypothesis `@settings(max_examples=200)`
- Unit tests validate specific examples, edge cases, and error conditions
- The Swift sidecar (task 11) is independent of the Python tasks and can be developed in parallel with tasks 4–9
- The legacy archival (task 1) MUST be executed before any new `Core/core/gui_agent/` files are written

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1"] },
    { "id": 1, "tasks": ["1.2", "2.2", "2.3", "4.1"] },
    { "id": 2, "tasks": ["2.4", "2.5", "4.2", "11.1"] },
    { "id": 3, "tasks": ["4.3", "4.4", "5.1", "11.2"] },
    { "id": 4, "tasks": ["4.5", "5.2", "6.1"] },
    { "id": 5, "tasks": ["4.6", "4.7", "4.8", "5.3", "6.2", "6.3"] },
    { "id": 6, "tasks": ["5.4", "5.5", "6.4", "6.5", "8.1"] },
    { "id": 7, "tasks": ["8.2", "9.1"] },
    { "id": 8, "tasks": ["8.3", "9.2"] },
    { "id": 9, "tasks": ["8.4", "9.3", "9.4", "9.5", "12.1", "12.2"] },
    { "id": 10, "tasks": ["13.1"] },
    { "id": 11, "tasks": ["13.2"] },
    { "id": 12, "tasks": ["13.3"] }
  ]
}
```
