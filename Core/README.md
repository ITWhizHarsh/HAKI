# HAKI Core — the Mind

HAKI Core is the local Python orchestration service that powers the HAKI personal AI assistant.  It is the second process of the two-process hybrid architecture described in the design document:

```
HAKI.app (Swift/SwiftUI "Body")  ←→  HAKI Core (Python "Mind")
         gRPC stream over UNIX socket
```

The Core owns the **Orchestrator**, **Model Provider abstraction**, **RAG/memory engine**, **Learning Engine**, **agentic Planner**, **Dialogue Manager**, and the **IPC server** that the Swift shell connects to.  It never listens on a network port reachable off-device.

---

## Module Layout

```
Core/
├── core/                      # Main Python package
│   ├── __init__.py
│   ├── orchestrator/          # Turn loop, intent routing, cancellation
│   │   ├── __init__.py
│   │   └── orchestrator.py
│   ├── model_provider/        # STT, LLM, TTS, mood, image, embeddings backends
│   │   ├── __init__.py
│   │   └── model_provider.py
│   ├── memory/                # Memory_Brain, vault I/O, RAG, vector index
│   │   ├── __init__.py
│   │   └── memory_brain.py
│   ├── learning/              # Learning_Engine, conversation-end detection, extraction
│   │   ├── __init__.py
│   │   └── learning_engine.py
│   ├── planner/               # CommandPlan, Step, LLM planner, Safety_Gate
│   │   ├── __init__.py
│   │   └── planner.py
│   ├── dialogue/              # Dialogue_Manager, slot filling, clarifying questions
│   │   ├── __init__.py
│   │   └── dialogue_manager.py
│   └── ipc/                   # gRPC/JSON-RPC server stub, UNIX socket server
│       ├── __init__.py
│       └── server.py
├── tests/
│   ├── conftest.py            # Hypothesis profiles + shared fixtures
│   └── test_smoke.py          # Import smoke tests + Hypothesis harness verification
├── pyproject.toml
├── requirements.txt
├── .python-version            # Python 3.11
└── README.md                  # This file
```

---

## Prerequisites

- **Python 3.11+** — enforced by `.python-version` (works with pyenv or mise)
- **pip** ≥ 23 or **uv**

---

## Setup

### 1 — Create and activate a virtual environment

```bash
cd Core

# Using the standard library venv
python3.11 -m venv .venv
source .venv/bin/activate

# Or using uv (faster)
uv venv --python 3.11
source .venv/bin/activate
```

### 2 — Install dependencies

```bash
# Pinned requirements (production + dev)
pip install -r requirements.txt

# Or install the package in editable mode (picks up pyproject.toml extras)
pip install -e ".[dev]"
```

---

## Running the Tests

```bash
# All tests with default Hypothesis profile (100 examples per property)
pytest

# Verbose output
pytest -v

# With a specific Hypothesis profile
HYPOTHESIS_PROFILE=ci pytest

# Coverage report
pytest --cov=core --cov-report=term-missing
```

---

## Hypothesis Profiles

| Profile | `max_examples` | Deadline | Use case |
|---------|---------------|----------|----------|
| `default` | 100 | 2 s | Local development |
| `ci` | 200 | 5 s | CI / pre-merge |
| `dev` | 20 | None | Fast inner-loop feedback |

Activate with the `HYPOTHESIS_PROFILE` environment variable or by calling `settings.load_profile("ci")` at the top of a test module.

---

## IPC / gRPC

The `.proto` schema and generated Swift + Python stubs are authored in **Task 1.3**.  The `core/ipc/server.py` module is a stub that sets up the server skeleton so the package remains importable before the proto is finalized.

---

## Design Reference

- **Architecture** — [design.md → Architecture](../.kiro/specs/haki-personal-ai-assistant/design.md)
- **Correctness Properties** — 76 properties defined in the design document, implemented as Hypothesis tests throughout the task list (Tasks 5.3 – 28+).
- **Requirements** — [requirements.md](../.kiro/specs/haki-personal-ai-assistant/requirements.md)
