# HAKI IPC — Proto schema and stub regeneration

## Schema overview

**File:** `haki_ipc.proto`  
**Package:** `haki`  
**Transport:** gRPC over a UNIX domain socket (`~/Library/Application Support/HAKI/haki_core.sock`)

The schema defines a **single bidirectional streaming RPC** — `HAKICore.StreamTurn` — that carries the entire voice/command pipeline in one long-lived stream per session.

### Why one bidirectional stream?

- Eliminates per-turn connection overhead that would violate the **300 ms first-audio budget** (Req 3.1).
- Both sides can push at any time: the server sends partial transcripts and control events without waiting for a `TurnRequest`.
- Barge-in (`CANCEL` / `BARGE_IN` `ControlEvent`) can be sent the instant AVAudioEngine detects continuous speech, with no request/response boundary to race.

### Message types

| Message | Direction | Purpose |
|---|---|---|
| `AudioFrame` | C → S | 20 ms PCM audio frames (Int16 LE), 16 kHz mono |
| `PartialTranscript` | Both | Incremental or final STT output |
| `AudioFeatures` | C → S | Pitch / energy for Mood_Detector (Req 4.1) |
| `LLMToken` | S → C | One output token — streamed immediately (Req 3.1) |
| `TTSAudioChunk` | S → C | Clause-sized PCM chunks for streaming playback (Req 3.1) |
| `ControlEvent` | Both | `CANCEL`, `BARGE_IN`, `END_OF_SPEECH`, `HEARTBEAT` |
| `TurnRequest` | C → S | Committed transcript + audio features for a turn |
| `TurnResponse` | S → C | Per-turn response fragment (token / chunk / event) |
| `ClientMessage` | C → S | Top-level envelope (oneof AudioFrame / PartialTranscript / TurnRequest / ControlEvent) |
| `ServerMessage` | S → C | Top-level envelope (oneof PartialTranscript / LLMToken / TTSAudioChunk / ControlEvent / error) |

### Service

```protobuf
service HAKICore {
  rpc StreamTurn(stream ClientMessage) returns (stream ServerMessage);
}
```

---

## Regenerating Python stubs

The generated files live in `Core/core/ipc/proto/`.

```bash
# From the repo root
python3 -m grpc_tools.protoc \
  -I proto \
  --python_out=Core/core/ipc/proto \
  --grpc_python_out=Core/core/ipc/proto \
  proto/haki_ipc.proto
```

Requirements:
- `grpcio-tools` installed (listed in `Core/requirements.txt`)
- Python ≥ 3.11

---

## Regenerating Swift stubs

Swift stubs are **not yet generated** (as of Phase 0, Task 1.3).  
The current `HAKI/Sources/Subsystems/IPC/IPCClient.swift` hand-mirrors the proto types as native Swift structs.  
Once `protoc-gen-grpc-swift` and `swift-protobuf` are available, replace the hand-written types with the generated ones.

### Install prerequisites (macOS, Homebrew)

```bash
brew install swift-protobuf grpc-swift
```

`swift-protobuf` installs `protoc-gen-swift`.  
`grpc-swift` installs `protoc-gen-grpc-swift`.

### Generate Swift stubs

```bash
# From the repo root
protoc \
  -I proto \
  --swift_out=HAKI/Sources/Subsystems/IPC/Generated \
  --grpc-swift_out=HAKI/Sources/Subsystems/IPC/Generated \
  proto/haki_ipc.proto
```

This produces:
- `haki_ipc.pb.swift` — message types
- `haki_ipc.grpc.swift` — `HAKICoreAsyncClient` and `HAKICoreProvider` (service protocol)

### Add Swift Package dependencies

Add to `HAKI/Package.swift`:

```swift
.package(url: "https://github.com/apple/swift-protobuf.git", from: "1.27.0"),
.package(url: "https://github.com/grpc/grpc-swift.git", from: "1.23.0"),
```

And to the target:
```swift
.product(name: "SwiftProtobuf",  package: "swift-protobuf"),
.product(name: "GRPC",           package: "grpc-swift"),
```

Once wired, delete the hand-written Swift type mirrors in `IPCClient.swift` and import the generated types instead.

---

## Socket path

| Side | Path |
|---|---|
| Default (both) | `~/.haki/core.sock` (development) |
| App sandbox (production) | `~/Library/Application Support/HAKI/haki_core.sock` |

The path is scoped to the local device and is never reachable from outside the machine (Req 20.4 / Security Considerations).
