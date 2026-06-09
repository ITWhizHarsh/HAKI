# HAKI — Heuristic Augmented Knowledge Interface

Heuristic Augmented Knowledge Interface
A local-first, privacy-respecting personal AI assistant for macOS, built as a two-process hybrid architecture.

---

## Architecture Overview

HAKI splits its responsibilities across two processes that talk over a UNIX domain socket:

| Process | Language | Nickname | Responsibilities |
|---|---|---|---|
| `HAKI.app` | Swift / SwiftUI | **Body** | TCC permissions, ScreenCaptureKit, Accessibility, Vision OCR, EventKit, AppleScript, global hotkey, audio I/O, menu-bar UI, notifications, encrypted on-device store |
| `haki_core_service` | Python | **Mind** | Orchestrator, Model Provider, RAG/memory, learning engine, planner, dialogue manager, gRPC/IPC server |

Communication happens over a **bidirectional gRPC/JSON-RPC streaming channel** via a UNIX socket at:
```
~/Library/Application Support/HAKI/haki_core.sock
```

---

## Repository Layout

```
HAKI/
├── HAKI/              ← Swift / SwiftUI shell ("Body") — SwiftPM package
│   ├── Package.swift
│   ├── Sources/
│   │   ├── HAKI/      ← Executable entry point (AppDelegate, CoreProcessManager)
│   │   └── Subsystems/ ← Audio, Capture, OSActions, Permissions, IPC, UI, Store
│   └── Tests/
├── Core/              ← Python orchestration service ("Mind")
│   ├── core/          ← Main Python package (orchestrator, model_provider, memory, …)
│   ├── tests/
│   ├── haki_core_service.py
│   ├── pyproject.toml
│   └── requirements.txt
├── proto/             ← Shared .proto schema for the IPC channel
│   └── haki_ipc.proto
└── .kiro/             ← Kiro AI specs (requirements, design, tasks)
```

---

## Getting Started

### Prerequisites

- macOS 13+ (Ventura or later)
- Xcode 15+ (full installation, not just CLT)
- Python 3.11+

### Swift Shell (Body)

```bash
cd HAKI
swift build
swift test
```

### Python Core (Mind)

```bash
cd Core

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# or editable install
pip install -e ".[dev]"

# Run tests
pytest
```

---

## Key Subsystems

| Subsystem | Location | Description |
|---|---|---|
| Orchestrator | `Core/core/orchestrator/` | Turn loop, intent routing, cancellation |
| Model Provider | `Core/core/model_provider/` | STT, LLM, TTS, embeddings backend abstraction |
| Memory Brain | `Core/core/memory/` | RAG, vector index, vault I/O |
| Learning Engine | `Core/core/learning/` | Conversation-end detection, knowledge extraction |
| Planner | `Core/core/planner/` | CommandPlan, step execution, Safety Gate |
| Dialogue Manager | `Core/core/dialogue/` | Slot filling, clarifying questions |
| IPC Server | `Core/core/ipc/` | gRPC/JSON-RPC server over UNIX socket |
| Audio | `HAKI/Sources/Subsystems/Audio/` | AVAudioEngine tap, VAD, AEC |
| Capture | `HAKI/Sources/Subsystems/Capture/` | ScreenCaptureKit, AXUIElement, Vision OCR |
| OS Actions | `HAKI/Sources/Subsystems/OSActions/` | AppleScript, EventKit, UserNotifications |
| Store | `HAKI/Sources/Subsystems/Store/` | Encrypted SQLite + Keychain |

---

## Privacy

All processing is **local by default**. No data is sent to external servers unless the user explicitly configures a remote model backend. The IPC socket is bound to the loopback filesystem and is never exposed over the network.

---

## Design Documents

Full requirements, architecture design, and implementation tasks are in:
- `.kiro/specs/haki-personal-ai-assistant/requirements.md`
- `.kiro/specs/haki-personal-ai-assistant/design.md`
- `.kiro/specs/haki-personal-ai-assistant/tasks.md`

---

## License

MIT
