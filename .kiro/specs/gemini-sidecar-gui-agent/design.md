# Design Document — Gemini-Sidecar Architecture for HAKI GUI Screen Control Agent

## Overview

This design replaces HAKI's legacy `ScreenAgent` / `MacController` stack with a
**Gemini-Sidecar Architecture** — a Swift ScreenCaptureKit sidecar streams Retina
JPEG frames over a UNIX socket; a Python `SidecarAgentLoop` drives a
See→Think→Act→Verify cognitive cycle using `GeminiVisionClient` and
`MacQuartzExecutor`; the Pipecat voice pipeline is never blocked; and a
`HITLBridge` handles secure-field pauses over the existing JSON IPC socket.

Target hardware: **8 GB M2 MacBook Air**, 2560×1600 native, 1280×800 logical
(DisplayScale = 2.0). No local VLMs, no OmniParser, no pyautogui.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Main Process (Python)                                                  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Pipecat Voice Thread (asyncio loop A)                           │  │
│  │   VoiceSessionPipeline → STT → LLM → XTTS                       │  │
│  │   VoiceToolAdapter.execute_tool_call("gui_agent.spawn", ...)     │  │
│  └──────────────────────┬───────────────────────────────────────────┘  │
│                         │ threading.Thread(daemon=True)                 │
│  ┌──────────────────────▼───────────────────────────────────────────┐  │
│  │  SidecarAgentLoop Thread (asyncio loop B — isolated)             │  │
│  │   ┌────────────────────────────────────────────────────────┐     │  │
│  │   │  See → GeminiVisionClient.request_frame()              │     │  │
│  │   │       → GeminiVisionClient.query_gemini()              │     │  │
│  │   │  Think → parse action JSON                             │     │  │
│  │   │  Act  → MacQuartzExecutor.dispatch_action()            │     │  │
│  │   │  Verify → GeminiVisionClient.request_frame()           │     │  │
│  │   └────────────────────────────────────────────────────────┘     │  │
│  │   HITLBridge (AXSecureTextField detection, pause/resume)         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  JSONIPCServer (asyncio loop A)                                  │  │
│  │   MSG_TYPE_AGENT_EVENT broadcast → all connected Swift clients   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
         │ UNIX socket ~/.haki/sidecar_frames.sock
┌────────▼──────────────────────────────────────────────────┐
│  Swift Sidecar Process                                     │
│   ScreenCaptureKit → JPEG compress → frame buffer          │
│   UNIX socket server (0600 perms)                          │
│   Handshake JSON: {"display_scale": 2.0, "width": 2560,   │
│                    "height": 1600}                         │
└────────────────────────────────────────────────────────────┘
```

---

## Directory Layout

```
Core/
├── core/
│   ├── automation/          ← untouched (other automation modules remain)
│   ├── gui_agent/
│   │   ├── __init__.py
│   │   ├── gemini_vision_client.py
│   │   ├── mac_quartz_executor.py
│   │   ├── sidecar_agent_loop.py
│   │   └── hitl_bridge.py
│   ├── ipc/
│   │   └── server.py        ← extended with AGENT_EVENT
│   └── voice/
│       └── tools.py         ← extended with SpawnGuiAgentCall
└── legacy_screen_control_backup/
    ├── mac_controller.py
    ├── screen_agent.py
    └── README.md

HAKI/Sources/Subsystems/
├── Capture/ScreenReader.swift  ← frame streaming sidecar
└── IPC/JSONIPCClient.swift     ← AGENT_EVENT routing
```

---

## Components

### 1. Legacy Archival (`Core/legacy_screen_control_backup/`)

A one-time migration step moves `mac_controller.py` and `screen_agent.py` from
`Core/core/automation/` into a backup directory before any new module is written.

**Archival logic (Python pseudo-code):**

```python
import shutil, os
from pathlib import Path

BACKUP_DIR = Path("Core/legacy_screen_control_backup")
SOURCES = [
    Path("Core/core/automation/mac_controller.py"),
    Path("Core/core/automation/screen_agent.py"),
]

def archive_legacy() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        dest = BACKUP_DIR / src.name
        if dest.exists():
            raise FileExistsError(f"Backup already exists at {dest}; aborting to prevent overwrite.")
        shutil.move(str(src), str(dest))
    _write_readme(BACKUP_DIR)

def _write_readme(directory: Path) -> None:
    from datetime import date
    readme = directory / "README.md"
    readme.write_text(
        f"# Legacy Screen Control Backup\n\n"
        f"Archived: {date.today().isoformat()}\n\n"
        f"## Original paths\n"
        f"- `Core/core/automation/mac_controller.py`\n"
        f"- `Core/core/automation/screen_agent.py`\n\n"
        f"## Reason\n"
        f"Replaced by Gemini-Sidecar Architecture (SidecarAgentLoop). "
        f"Files preserved for recovery without a git revert.\n"
    )
```

---

### 2. Swift ScreenCaptureKit Sidecar (`HAKI/Sources/Subsystems/Capture/ScreenReader.swift`)

The sidecar is a Swift executable that:
1. Checks `Screen Recording` permission via ScreenCaptureKit; exits non-zero with stderr message if denied.
2. Queries `CGMainDisplayID` at startup to compute `DisplayScale`.
3. Creates a `SCStreamConfiguration` targeting 2560×1600.
4. Compresses each frame to JPEG (default quality 85%, configurable via `--jpeg-quality` arg).
5. Maintains a single-slot `actor`-protected frame buffer (always holds the latest frame).
6. Listens on `~/.haki/sidecar_frames.sock` (created with `0600` file permissions).
7. On each client connection: sends handshake JSON `{"display_scale": 2.0, "width": 2560, "height": 1600}\n`, then waits for `REQUEST_FRAME\n` line; responds with 4-byte big-endian length prefix followed by JPEG bytes.

**Handshake protocol (newline-delimited):**

```
Client connects
  → Server sends: {"display_scale":2.0,"width":2560,"height":1600}\n
  → Client sends: REQUEST_FRAME\n
  → Server sends: <4-byte big-endian uint32 frame length><JPEG bytes>
  → Client may send REQUEST_FRAME\n again for next frame
```

**Swift skeleton:**

```swift
// ScreenReader.swift — relevant additions
actor FrameBuffer {
    private var latestFrame: Data?
    func update(_ frame: Data) { latestFrame = frame }
    func latest() -> Data? { latestFrame }
}

final class SidecarServer {
    static let socketPath = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".haki/sidecar_frames.sock").path

    private let buffer = FrameBuffer()

    func start() throws {
        // 1. Verify Screen Recording permission
        // 2. Set socket permissions to 0o600 after bind
        // 3. Start SCStream with SCStreamConfiguration at 2560x1600
        // 4. Accept connections, send handshake, serve frames on REQUEST_FRAME
    }
}
```

---

### 3. GeminiVisionClient (`Core/core/gui_agent/gemini_vision_client.py`)

Responsibilities:
- Connect to the sidecar UNIX socket and request JPEG frames (async).
- Submit frames + prompt to `gemini-2.5-flash` via Google's `google-generativeai` SDK.
- Parse response into a structured `GeminiAction` dataclass.
- Convert normalized bbox coordinates to native pixel values.
- Raise `GeminiAPIError` on non-200 or timeout (15 s).
- Never log or persist frame bytes outside the request coroutine scope.

**Data models:**

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Normalized Gemini bbox in 0–1000 integer space."""
    ymin: int  # 0–1000
    xmin: int  # 0–1000
    ymax: int  # 0–1000
    xmax: int  # 0–1000

@dataclass(frozen=True, slots=True)
class NativePixelBox:
    """Pixel coordinates in NativeResolution (2560×1600) space."""
    ymin: int
    xmin: int
    ymax: int
    xmax: int

@dataclass(frozen=True, slots=True)
class GeminiAction:
    action_type: str        # e.g. "click", "type", "scroll", "done"
    bbox: NativePixelBox
    text: str | None = None  # populated for "type" actions
    summary: str | None = None  # populated for "done" actions

class GeminiAPIError(Exception):
    """Retriable error from the Gemini API."""
    def __init__(self, status_code: int | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retriable = True
```

**Key methods:**

```python
class GeminiVisionClient:
    NATIVE_WIDTH = 2560
    NATIVE_HEIGHT = 1600
    SOCKET_PATH = os.path.expanduser("~/.haki/sidecar_frames.sock")
    GEMINI_MODEL = "gemini-2.5-flash"
    TIMEOUT_SECONDS = 15

    def __init__(self) -> None:
        # API key is read exclusively from environment — never passed as argument
        api_key = os.environ["HAKI_GEMINI_API_KEY"]
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(self.GEMINI_MODEL)
        self._display_scale: float | None = None

    async def perform_handshake(self) -> float:
        """Connect to sidecar, receive DisplayScale, return it."""
        ...

    async def request_frame(self) -> bytes:
        """Send REQUEST_FRAME to sidecar socket, receive JPEG bytes."""
        ...

    async def query_gemini(self, task_description: str) -> GeminiAction:
        """Run one See phase: request frame, query Gemini, parse, return action."""
        frame_bytes: bytes = await self.request_frame()
        try:
            action = await asyncio.wait_for(
                self._call_gemini(frame_bytes, task_description),
                timeout=self.TIMEOUT_SECONDS,
            )
            return action
        except asyncio.TimeoutError:
            raise GeminiAPIError(None, "Gemini API timeout")
        finally:
            del frame_bytes  # explicit deletion within coroutine scope (Req 10.1)

    def bbox_to_native(self, bbox: BoundingBox) -> NativePixelBox:
        """Convert 0–1000 normalized bbox to 2560×1600 pixel coordinates."""
        return NativePixelBox(
            ymin=int(bbox.ymin * self.NATIVE_HEIGHT / 1000),
            xmin=int(bbox.xmin * self.NATIVE_WIDTH / 1000),
            ymax=int(bbox.ymax * self.NATIVE_HEIGHT / 1000),
            xmax=int(bbox.xmax * self.NATIVE_WIDTH / 1000),
        )
```

---

### 4. MacQuartzExecutor (`Core/core/gui_agent/mac_quartz_executor.py`)

Responsibilities:
- Import `Quartz` from `pyobjc-framework-Quartz`; raise `ExecutorUnavailableError` at instantiation if unavailable.
- Query `DisplayScale` at runtime via `CGMainDisplayID` before each click.
- Convert `NativePixelBox` center to `LogicalCoordinate` by dividing by `DisplayScale`.
- Dispatch `kCGEventLeftMouseDown` + `kCGEventLeftMouseUp` via `CGEventPost(kCGHIDEventTap, ...)`.
- Dispatch keyboard events via `CGEventCreateKeyboardEvent`.

**Interface:**

```python
class ExecutorUnavailableError(RuntimeError):
    """Raised at instantiation when pyobjc-framework-Quartz is not importable."""

class MacQuartzExecutor:
    def __init__(self) -> None:
        try:
            import Quartz  # noqa: F401
            self._Quartz = Quartz
        except ImportError as exc:
            raise ExecutorUnavailableError(
                "pyobjc-framework-Quartz is not installed. "
                "Install with: pip install pyobjc-framework-Quartz"
            ) from exc

    def _get_display_scale(self) -> float:
        """Query DisplayScale at runtime. Returns 2.0 on M2 MacBook Air."""
        Q = self._Quartz
        display_id = Q.CGMainDisplayID()
        phys = Q.CGDisplayScreenSize(display_id)
        pix = Q.CGDisplayBounds(display_id)
        # DisplayScale = pixel width / point width
        # For M2 MBA: 2560 / 1280 = 2.0
        return pix.size.width / (phys.width / 25.4 * 72) if phys.width > 0 else 2.0

    def click(self, native_box: NativePixelBox) -> None:
        """Click center of native_box after converting to logical coordinates."""
        scale = self._get_display_scale()
        cx_native = (native_box.xmin + native_box.xmax) / 2
        cy_native = (native_box.ymin + native_box.ymax) / 2
        lx = cx_native / scale
        ly = cy_native / scale
        self._post_click(lx, ly)

    def type_text(self, text: str) -> None:
        """Type a string using CGEventCreateKeyboardEvent."""
        ...

    def _post_click(self, lx: float, ly: float) -> None:
        Q = self._Quartz
        point = Q.CGPointMake(lx, ly)
        down = Q.CGEventCreateMouseEvent(None, Q.kCGEventLeftMouseDown, point, Q.kCGMouseButtonLeft)
        up   = Q.CGEventCreateMouseEvent(None, Q.kCGEventLeftMouseUp,   point, Q.kCGMouseButtonLeft)
        Q.CGEventPost(Q.kCGHIDEventTap, down)
        Q.CGEventPost(Q.kCGHIDEventTap, up)
```

---

### 5. SidecarAgentLoop (`Core/core/gui_agent/sidecar_agent_loop.py`)

Responsibilities:
- Run in an isolated daemon `threading.Thread` with its own `asyncio` event loop.
- Execute the See→Think→Act→Verify cycle up to 20 iterations.
- On each step: call `GeminiVisionClient.query_gemini()`, dispatch via `MacQuartzExecutor`, verify by calling `query_gemini()` again.
- Retry retriable errors (e.g. `GeminiAPIError`) up to 3 times with a 2 s delay.
- Emit `AGENT_EVENT` messages via `JSONIPCServer.broadcast_agent_event()` for state transitions.
- Check `HITLBridge.should_pause()` after each Verify phase.

**State machine:**

```
IDLE → RUNNING → (each step) → CHECK_HITL → RUNNING
                             → HITL_PAUSED → HITL_RESUMED → RUNNING
                             → DONE        → IDLE
                             → ERROR       → IDLE
                             → MAX_STEPS   → IDLE
```

**Core loop (pseudo-code):**

```python
MAX_STEPS = 20
RETRY_LIMIT = 3
RETRY_DELAY = 2.0  # seconds

async def _run_loop(self, task_description: str) -> None:
    await self._broadcast(AgentEventType.AGENT_START, {"task": task_description})
    for step in range(1, MAX_STEPS + 1):
        # --- See + Think (with retry) ---
        action = await self._see_think(task_description, step)
        if action is None:
            await self._broadcast(AgentEventType.AGENT_ERROR, {"step": step})
            return

        # --- Act ---
        self._executor.dispatch(action)
        await self._broadcast(AgentEventType.AGENT_STEP, {"step": step, "action": action.action_type})

        # --- Verify ---
        verify_action = await self._see_think(task_description, step, verify=True)

        # --- HITL check ---
        if self._hitl_bridge.should_pause():
            await self._hitl_bridge.pause_and_wait()

        # --- Done check ---
        if action.action_type == "done":
            await self._broadcast(AgentEventType.AGENT_DONE, {"success": True, "summary": action.summary})
            return

    await self._broadcast(AgentEventType.AGENT_MAX_STEPS_REACHED, {"steps": MAX_STEPS})

async def _see_think(self, task: str, step: int, verify: bool = False) -> GeminiAction | None:
    for attempt in range(RETRY_LIMIT):
        try:
            return await self._vision_client.query_gemini(task)
        except GeminiAPIError:
            if attempt < RETRY_LIMIT - 1:
                await asyncio.sleep(RETRY_DELAY)
    return None
```

**Thread launch (in `spawn_gui_agent`):**

```python
import threading

def _start_agent_in_thread(task_description: str, ipc_server: JSONIPCServer) -> None:
    loop = asyncio.new_event_loop()

    def _thread_main() -> None:
        asyncio.set_event_loop(loop)
        agent = SidecarAgentLoop(ipc_server=ipc_server)
        loop.run_until_complete(agent.run(task_description))
        loop.close()

    t = threading.Thread(target=_thread_main, daemon=True, name="haki-gui-agent")
    t.start()
```

---

### 6. HITLBridge (`Core/core/gui_agent/hitl_bridge.py`)

Responsibilities:
- Detect `AXSecureTextField` in the current screen state (via Accessibility API or Gemini's response metadata).
- Pause `SidecarAgentLoop` via an `asyncio.Event`.
- Emit `agent_hitl_pause` over IPC.
- Receive user's spoken answer from Pipecat STT and inject into the paused loop.
- Emit `agent_hitl_resume` and set the resume event.
- Implement a 60 s timeout; on expiry emit `agent_error` and terminate the loop.

**Interface:**

```python
class HITLBridge:
    TIMEOUT_SECONDS = 60.0

    def __init__(self, ipc_server: JSONIPCServer) -> None:
        self._ipc_server = ipc_server
        self._pause_event = asyncio.Event()
        self._resume_event = asyncio.Event()
        self._injected_text: str | None = None

    def should_pause(self) -> bool:
        """Return True if current screen state requires HITL."""
        ...

    async def pause_and_wait(self) -> str | None:
        """Pause loop, emit hitl_pause, wait for user answer (60s timeout)."""
        await self._ipc_server.broadcast_agent_event(
            AgentEventType.AGENT_HITL_PAUSE, {}
        )
        try:
            await asyncio.wait_for(self._resume_event.wait(), timeout=self.TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            await self._ipc_server.broadcast_agent_event(
                AgentEventType.AGENT_ERROR,
                {"message": "HITL timeout: no user response"},
            )
            raise HITLTimeoutError("HITL timeout: no user response")
        return self._injected_text

    def inject_response(self, text: str) -> None:
        """Called by Pipecat STT handler when user speaks HITL answer."""
        self._injected_text = text
        self._resume_event.set()
        # Note: text is held only in memory; never logged or persisted (Req 10.3)
```

---

### 7. JSON IPC AGENT_EVENT Extension (`Core/core/ipc/server.py`)

New constants added to `server.py`:

```python
MSG_TYPE_AGENT_EVENT = "AGENT_EVENT"

class AgentEventType:
    AGENT_START              = "agent_start"
    AGENT_STEP               = "agent_step"
    AGENT_DONE               = "agent_done"
    AGENT_ERROR              = "agent_error"
    AGENT_MAX_STEPS_REACHED  = "agent_max_steps_reached"
    AGENT_HITL_PAUSE         = "agent_hitl_pause"
    AGENT_HITL_RESUME        = "agent_hitl_resume"

    _VALID: frozenset[str] = frozenset({
        AGENT_START, AGENT_STEP, AGENT_DONE, AGENT_ERROR,
        AGENT_MAX_STEPS_REACHED, AGENT_HITL_PAUSE, AGENT_HITL_RESUME,
    })

    @classmethod
    def is_valid(cls, event_type: str) -> bool:
        return event_type in cls._VALID
```

`JSONIPCServer` gains a `_connected_writers` set and a `broadcast_agent_event` method:

```python
# In JSONIPCServer:
def __init__(self, ...) -> None:
    ...
    self._connected_writers: set[asyncio.StreamWriter] = set()

async def _handle_client(self, reader, writer) -> None:
    self._connected_writers.add(writer)
    try:
        ...  # existing logic
    finally:
        self._connected_writers.discard(writer)
        ...

async def broadcast_agent_event(self, event_type: str, payload: dict) -> None:
    """Broadcast an AGENT_EVENT to all connected IPC clients."""
    if not AgentEventType.is_valid(event_type):
        logger.warning("JSONIPCServer: unknown AGENT_EVENT type %r — dropped", event_type)
        return
    msg = {
        "type": MSG_TYPE_AGENT_EVENT,
        "payload": {"event_type": event_type, **payload},
    }
    for writer in list(self._connected_writers):
        try:
            await self._write_message(writer, msg)
        except Exception:
            logger.debug("JSONIPCServer: failed to write AGENT_EVENT to client — removing")
            self._connected_writers.discard(writer)
```

Inbound `AGENT_EVENT` messages from internal components are also routed through `_dispatch`:

```python
elif msg_type == MSG_TYPE_AGENT_EVENT:
    event_type = payload.get("event_type", "")
    if not AgentEventType.is_valid(event_type):
        logger.warning("JSONIPCServer: unknown AGENT_EVENT sub-type %r — dropped", event_type)
        return
    await self.broadcast_agent_event(event_type, {k: v for k, v in payload.items() if k != "event_type"})
```

---

### 8. spawn_gui_agent Tool (`Core/core/voice/tools.py`)

New schema and grammar addition:

```python
class SpawnGuiAgentCall(BaseModel):
    """Schema for a gui_agent.spawn tool call emitted by the voice LLM."""

    model_config = ConfigDict(extra="forbid")

    tool: Literal["gui_agent.spawn"]
    schema_version: int = 1
    task_description: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
    ]

# VOICE_TOOL_GRAMMAR updated to include:
# {"tool": "gui_agent.spawn", "schema_version": 1, "task_description": "<task description>"}
```

In `VoiceToolAdapter._execute_gui_agent_spawn`:

```python
async def _execute_gui_agent_spawn(
    self,
    call: SpawnGuiAgentCall,
    *,
    turn_id: UUID,
    session_id: UUID,
) -> ToolCallResult:
    # 1. Immediately synthesize XTTS acknowledgement (before spawning agent)
    if self._xtts_sink is not None:
        await self._xtts_sink(f"On it, starting task: {call.task_description[:80]}")

    # 2. Spin up SidecarAgentLoop in isolated daemon thread
    _start_agent_in_thread(call.task_description, self._ipc_server)

    # 3. Return success within 500ms
    return ToolCallResult(
        tool_name="gui_agent.spawn",
        turn_id=turn_id,
        success=True,
        data={"message": "GUI agent started", "task": call.task_description},
    )
```

---

## Data Flow

### Voice → GUI Agent trigger

```
User speaks intent
  → VAD / STT (Pipecat loop A)
  → LLM emits {"tool": "gui_agent.spawn", "task_description": "..."}
  → VoiceToolAdapter.execute_tool_call()
    → validates SpawnGuiAgentCall schema
    → XTTS ack: "On it, starting task..."         ← ≤500ms total return
    → threading.Thread(daemon=True).start()
      → asyncio.new_event_loop() on thread B
      → SidecarAgentLoop.run(task_description)
  → ToolCallResult(success=True) returned to Pipecat   ← Pipecat never blocked
```

### See→Think→Act→Verify cycle

```
SidecarAgentLoop (loop B):
  1. See:    GeminiVisionClient.request_frame()
             → UNIX socket to Swift sidecar
             → recv JPEG bytes (≤200ms)
  2. Think:  GeminiVisionClient._call_gemini(frame, task)
             → google-generativeai SDK HTTP (≤15s timeout)
             → parse JSON → GeminiAction
             → del frame_bytes
  3. Act:    MacQuartzExecutor.dispatch(action)
             → _get_display_scale() via CGMainDisplayID
             → NativeCoord / DisplayScale → LogicalCoord
             → CGEventPost(kCGHIDEventTap, mouseDown)
             → CGEventPost(kCGHIDEventTap, mouseUp)
  4. Verify: GeminiVisionClient.request_frame() (again)
             → broadcast AGENT_STEP via JSONIPCServer
  5. HITLBridge.should_pause()?
             → if yes: broadcast agent_hitl_pause, await user response (60s)
             → inject response, broadcast agent_hitl_resume
  6. action_type == "done"?
             → broadcast agent_done, terminate loop
  7. step >= 20?
             → broadcast agent_max_steps_reached, terminate loop
```

### HITL pause/resume flow

```
SidecarAgentLoop detects AXSecureTextField (via AX API or Gemini metadata)
  → HITLBridge.pause_and_wait()
    → broadcast agent_hitl_pause over JSON IPC
    → JSONIPCServer sends to all clients (Swift UI)
  → Swift UI forwards to Pipecat bus
  → Pipecat LLM synthesizes "Please tell me the OTP."
  → User speaks OTP
  → Pipecat STT transcribes → HITLBridge.inject_response(text)
    → text held in memory only (never logged)
    → resume_event.set()
  → SidecarAgentLoop receives injected text
  → broadcast agent_hitl_resume
  → MacQuartzExecutor.type_text(injected_text)
```

---

## Interfaces

### GeminiVisionClient public API

| Method | Signature | Description |
|--------|-----------|-------------|
| `perform_handshake` | `async () → float` | Connect to sidecar, return DisplayScale |
| `request_frame` | `async () → bytes` | Get latest JPEG frame from sidecar |
| `query_gemini` | `async (task: str) → GeminiAction` | Full See phase |
| `bbox_to_native` | `(bbox: BoundingBox) → NativePixelBox` | Coord conversion |

### MacQuartzExecutor public API

| Method | Signature | Description |
|--------|-----------|-------------|
| `click` | `(native_box: NativePixelBox) → None` | Mouse click at box center |
| `type_text` | `(text: str) → None` | Keyboard input dispatch |
| `scroll` | `(native_box: NativePixelBox, direction: str, amount: int) → None` | Scroll gesture |

### SidecarAgentLoop public API

| Method | Signature | Description |
|--------|-----------|-------------|
| `run` | `async (task: str) → None` | Start the cognitive loop |
| `abort` | `() → None` | Request early termination |

### JSONIPCServer new public API

| Method | Signature | Description |
|--------|-----------|-------------|
| `broadcast_agent_event` | `async (event_type: str, payload: dict) → None` | Broadcast to all clients |

---

## Error Handling

| Error class | Source | Handling |
|-------------|--------|----------|
| `GeminiAPIError` | GeminiVisionClient | Retriable: up to 3 retries, 2 s delay |
| `ExecutorUnavailableError` | MacQuartzExecutor | Non-retriable: emit `agent_error`, stop loop |
| `HITLTimeoutError` | HITLBridge | Non-retriable: emit `agent_error`, stop loop |
| `FileExistsError` | Legacy archival | Abort move, log error (no overwrite) |
| `asyncio.TimeoutError` | GeminiVisionClient | Wrapped as `GeminiAPIError(None, "timeout")` |
| `ImportError` (Quartz) | MacQuartzExecutor | `ExecutorUnavailableError` at instantiation |
| Network error | GeminiVisionClient | Wrapped as `GeminiAPIError`, retriable |
| Sidecar permission denied | Swift sidecar | Exit non-zero, write stderr, Python sees socket error |

**Non-retriable vs. retriable:**
- `GeminiAPIError.retriable = True` — transient network, rate-limit, or 5xx.
- Any exception not of type `GeminiAPIError` is non-retriable.

---

## Security Model

1. **No frame persistence**: JPEG bytes are deleted in the same coroutine scope that created them (`del frame_bytes` after `finally`).
2. **API key isolation**: `HAKI_GEMINI_API_KEY` is read exclusively from `os.environ` during `__init__`; never passed as a parameter, never interpolated into log messages.
3. **UNIX socket permissions**: Swift sidecar calls `chmod(path, 0o600)` after binding; only the owning user process can connect.
4. **HITL answer in-memory only**: `HITLBridge._injected_text` is a plain Python string held in the object; it is never written to disk, logged at any level, or transmitted over IPC.
5. **Content-free error messages**: `GeminiAPIError` carries only the status code integer and a fixed string; no user data, no frame content.

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Backup idempotence — existing backup is never overwritten

*For any* file that already exists at the backup destination path, executing the legacy archival procedure SHALL leave that file's content unchanged (i.e. the destination bytes before and after an attempted archive are identical, and the source file remains in place).

**Validates: Requirements 1.5**

---

### Property 2: JPEG output validity

*For any* display frame captured by the SwiftSidecar, the bytes delivered to a connected Python client SHALL be parseable as a valid JPEG image (i.e. JPEG magic bytes `FF D8` at offset 0 and `FF D9` at end).

**Validates: Requirements 2.2**

---

### Property 3: Rolling buffer holds only latest frame

*For any* sequence of N ≥ 2 frame captures by the SwiftSidecar, a `REQUEST_FRAME` command issued after all N captures SHALL return bytes that are byte-for-byte equal to the Nth (most recent) frame and NOT equal to any of the first N−1 frames.

**Validates: Requirements 2.5**

---

### Property 4: Handshake always contains DisplayScale

*For any* Python client that connects to the UNIX domain socket, the first newline-delimited message received SHALL be valid JSON containing a `display_scale` field whose value is a positive finite float.

**Validates: Requirements 2.7**

---

### Property 5: Gemini response parse round-trip

*For any* valid Gemini 2.5 Flash response JSON object containing an `action_type` string and a `bbox` array of four integers in [0, 1000], `GeminiVisionClient._parse_response(json_text)` SHALL return a `GeminiAction` whose `action_type` equals the input field and whose `bbox` components equal the input array values (before coordinate conversion).

**Validates: Requirements 3.3**

---

### Property 6: Retriable error on non-200 or timeout

*For any* HTTP status code ≠ 200 returned by the Gemini API mock, or *for any* simulated network timeout exceeding 15 seconds, `GeminiVisionClient.query_gemini()` SHALL raise a `GeminiAPIError` with `retriable == True` and a status code matching the mocked response.

**Validates: Requirements 3.4**

---

### Property 7: Bounding box coordinate conversion correctness

*For any* bounding box `[ymin, xmin, ymax, xmax]` with all components in [0, 1000], `GeminiVisionClient.bbox_to_native(bbox)` SHALL return a `NativePixelBox` where:
- `result.ymin == int(ymin * 1600 / 1000)`
- `result.xmin == int(xmin * 2560 / 1000)`
- `result.ymax == int(ymax * 1600 / 1000)`
- `result.xmax == int(xmax * 2560 / 1000)`

**Validates: Requirements 3.6**

---

### Property 8: Logical coordinate conversion correctness

*For any* `NativePixelBox` center point `(cx_native, cy_native)` and *for any* `display_scale > 0` returned by the runtime query, `MacQuartzExecutor._to_logical(cx_native, cy_native, display_scale)` SHALL return `(cx_native / display_scale, cy_native / display_scale)`.

**Validates: Requirements 4.3**

---

### Property 9: Every click dispatches exactly two HID events in order

*For any* `NativePixelBox` passed to `MacQuartzExecutor.click()`, exactly two `CGEventPost(kCGHIDEventTap, ...)` calls SHALL occur: the first with event type `kCGEventLeftMouseDown` and the second with event type `kCGEventLeftMouseUp`, both at the same logical coordinate.

**Validates: Requirements 4.4, 4.7**

---

### Property 10: Agent runs in isolated daemon thread with its own event loop

*For any* `task_description` string, calling `spawn_gui_agent(task_description)` SHALL start the `SidecarAgentLoop` in a thread whose `threading.current_thread()` is NOT the Pipecat voice pipeline thread, and the `asyncio.get_event_loop()` inside that thread SHALL be a different object from the Pipecat event loop.

**Validates: Requirements 5.1, 9.1**

---

### Property 11: Cognitive cycle step order is always See→Think→Act→Verify

*For any* task description and *for any* step 1 ≤ n ≤ 20 of the `SidecarAgentLoop`, the sequence of operations in step n SHALL be: (1) `request_frame`, (2) `query_gemini`, (3) `MacQuartzExecutor.dispatch`, (4) `request_frame` again — with no other `dispatch` calls interleaved.

**Validates: Requirements 5.2**

---

### Property 12: Pipecat pipeline availability is unaffected while agent runs

*For any* `SidecarAgentLoop` task running in its daemon thread, the `VoiceSessionPipeline.availability` property SHALL remain `PipelineAvailability.RUNNING` throughout the duration of the agent task (measured by polling availability every 100 ms during agent execution).

**Validates: Requirements 5.3, 6.6, 8.6, 9.3**

---

### Property 13: Loop terminates at exactly step 20 with agent_max_steps_reached

*For any* task that never produces a `done` action, the `SidecarAgentLoop` SHALL emit exactly one `AGENT_EVENT` of sub-type `agent_max_steps_reached` after exactly 20 iterations, and then terminate without any further `AGENT_STEP` events.

**Validates: Requirements 5.4**

---

### Property 14: Retry count and delay for retriable errors

*For any* cognitive loop step where `GeminiVisionClient.query_gemini()` raises `GeminiAPIError` on every call, the `SidecarAgentLoop` SHALL call `query_gemini()` exactly 3 times for that step (1 initial + 2 retries), with each inter-attempt delay ≥ 1.5 s (allowing for timing tolerance), before treating the step as permanently failed.

**Validates: Requirements 5.6**

---

### Property 15: SpawnGuiAgentCall rejects invalid task_description lengths

*For any* string `s`, `SpawnGuiAgentCall.model_validate({"tool": "gui_agent.spawn", "schema_version": 1, "task_description": s})` SHALL succeed when `1 ≤ len(s.strip()) ≤ 500` and SHALL raise `ValidationError` when `len(s.strip()) == 0` or `len(s) > 500`.

**Validates: Requirements 6.2**

---

### Property 16: AGENT_EVENT broadcast reaches all connected clients

*For any* number N ≥ 1 of simultaneously connected IPC clients, calling `JSONIPCServer.broadcast_agent_event(event_type, payload)` with a valid `event_type` SHALL deliver the message to all N client writers before returning, and no message SHALL be delivered for an invalid `event_type`.

**Validates: Requirements 7.3, 7.4**

---

### Property 17: Valid event types pass; unknown types are silently dropped

*For any* string `event_type`, `AgentEventType.is_valid(event_type)` SHALL return `True` iff `event_type` is one of the seven defined constants (`agent_start`, `agent_step`, `agent_done`, `agent_error`, `agent_max_steps_reached`, `agent_hitl_pause`, `agent_hitl_resume`), and for any unknown type the broadcast function SHALL make zero calls to any client writer.

**Validates: Requirements 7.4, 7.5**

---

### Property 18: HITL pause occurs for any AXSecureTextField detection

*For any* screen state in which `HITLBridge.should_pause()` returns `True`, calling `HITLBridge.pause_and_wait()` SHALL: (1) emit `agent_hitl_pause` before yielding, (2) block until either `inject_response()` is called or 60 s elapses, and (3) emit `agent_hitl_resume` after successful injection or `agent_error` after timeout.

**Validates: Requirements 8.1, 8.5**

---

### Property 19: Injected HITL text is returned unchanged

*For any* non-empty string `text` passed to `HITLBridge.inject_response(text)`, the value returned by `HITLBridge.pause_and_wait()` SHALL be byte-for-byte equal to `text` (the spoken answer is injected without modification).

**Validates: Requirements 8.3**

---

### Property 20: Gemini API key is sourced exclusively from environment

*For any* `GeminiVisionClient` instantiation in a process where `HAKI_GEMINI_API_KEY` is set, the key value forwarded to the `google-generativeai` SDK SHALL equal `os.environ["HAKI_GEMINI_API_KEY"]`, and no call site in `gui_agent/` SHALL accept an API key as a function parameter.

**Validates: Requirements 10.5**

---

### Property 21: No disk writes during frame capture or agent execution

*For any* sequence of See and Verify phase calls during `SidecarAgentLoop` execution, the set of file paths opened for writing (as captured by mocking `builtins.open` and `pathlib.Path.write_*`) SHALL be empty — no frame bytes or intermediate state are persisted to disk.

**Validates: Requirements 10.2**

---

## Testing Strategy

### Unit Tests (example-based)

- Archival script: source removed, destination created, README fields present.
- `MacQuartzExecutor` raises `ExecutorUnavailableError` when `Quartz` import fails.
- `VoiceToolAdapter` returns `ToolCallResult` within 500 ms for `gui_agent.spawn`.
- XTTS ack fires before agent thread starts (mock sequencing with `threading.Barrier`).
- `JSONIPCServer` constants: `MSG_TYPE_AGENT_EVENT`, all 7 sub-type strings.
- `HITLBridge` emits `agent_hitl_resume` after `inject_response()`.
- `MacQuartzExecutor.type_text()` calls `CGEventCreateKeyboardEvent` for each character.
- `MacQuartzExecutor` operations occur on agent thread, not Pipecat thread.

### Property-Based Tests (Hypothesis)

All 21 correctness properties above are implemented as Hypothesis property tests with `@settings(max_examples=200)`. Key generators:

- `bbox_strategy = st.builds(BoundingBox, ymin=st.integers(0,1000), xmin=st.integers(0,1000), ymax=st.integers(0,1000), xmax=st.integers(0,1000))`
- `display_scale_strategy = st.floats(min_value=0.1, max_value=4.0, allow_nan=False)`
- `task_description_strategy = st.text(min_size=1, max_size=500)`
- `event_type_strategy = st.text()` (for valid/invalid discrimination tests)
- `client_count_strategy = st.integers(min_value=1, max_value=10)`

### Integration Tests (1–3 examples each)

- Swift sidecar starts, handshake JSON received, JPEG frame returned within 200 ms.
- Gemini API key auth header is present in mock HTTP intercept.
- End-to-end: `spawn_gui_agent` → thread starts → first `AGENT_STEP` event broadcast.
- HITL 60 s timeout with mocked `asyncio.sleep` advance.

### Smoke Tests

- `Core/legacy_screen_control_backup/` exists with expected files after archival.
- UNIX socket at `~/.haki/sidecar_frames.sock` has `0600` permissions after sidecar startup.
- `VOICE_TOOL_GRAMMAR` contains `"gui_agent.spawn"`.
