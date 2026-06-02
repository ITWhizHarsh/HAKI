# HAKI — Swift Shell ("Body")

> Heuristic Augmented Knowledge Interface — macOS native shell

This directory contains the **Swift / SwiftUI app shell** component of the two-process HAKI architecture.

---

## Architecture Overview

HAKI is a **two-process hybrid**:

| Process | Language | Nickname | Owns |
|---|---|---|---|
| `HAKI.app` | Swift / SwiftUI | **Body** | TCC permissions, ScreenCaptureKit, Accessibility, Vision OCR, EventKit, AppleScript/Apple Events, global hotkey, audio I/O (AVAudioEngine), menu-bar UI, notifications, encrypted on-device store |
| `haki_core_service` | Python | **Mind** | Orchestrator, Model Provider, RAG/memory, learning, planner, dialogue manager |

The two processes communicate over a **UNIX domain socket** (path: `~/Library/Application Support/HAKI/haki_core.sock`) via a **bidirectional gRPC/JSON-RPC streaming channel**.

---

## Project Structure

```
HAKI/                          ← SwiftPM package root
├── Package.swift              ← Package manifest (all targets declared here)
├── README.md                  ← This file
│
├── Sources/
│   ├── HAKI/                  ← Executable entry point
│   │   ├── main.swift         ← NSApplication bootstrap
│   │   ├── AppDelegate.swift  ← Menu-bar NSStatusItem, Core process lifecycle
│   │   ├── CoreProcessManager.swift  ← Spawn / health-check / restart of Python Core
│   │   └── Resources/
│   │       ├── Info.plist     ← TCC usage descriptions, bundle metadata
│   │       └── HAKI.entitlements ← Hardened-runtime entitlements
│   │
│   └── Subsystems/            ← One library target per subsystem
│       ├── Audio/             ← AVAudioEngine tap, VAD, AEC (HAKIAudio)
│       │   ├── AudioEngine.swift
│       │   └── VAD.swift
│       ├── Capture/           ← ScreenCaptureKit, AXUIElement, Vision OCR (HAKICapture)
│       │   ├── ScreenReader.swift
│       │   └── VisionOCR.swift
│       ├── OSActions/         ← AppleScript, EventKit, UserNotifications (HAKIOSActions)
│       │   ├── AppleScriptBridge.swift
│       │   ├── EventKitBridge.swift
│       │   └── NotificationManager.swift
│       ├── Permissions/       ← TCC status/request, screen-access toggle, revocation watcher (HAKIPermissions)
│       │   └── PermissionManager.swift
│       ├── IPC/               ← gRPC/JSON-RPC client stub for Core channel (HAKIIPC)
│       │   └── IPCClient.swift
│       ├── UI/                ← SwiftUI menu-bar views, settings panel (HAKIUI)
│       │   └── MenuBarUI.swift
│       └── Store/             ← Encrypted SQLite store + Keychain (HAKIStore)
│           ├── AppStore.swift
│           └── KeychainStore.swift
│
└── Tests/
    ├── HAKITests/             ← Unit/example tests
    │   └── AppShellTests.swift
    └── HAKIPropertyTests/     ← SwiftCheck property-based tests
        └── VADPropertyTests.swift
```

---

## Running the Tests

```bash
# From the HAKI/ package directory:
swift test
```

> **Requirements:** Running tests requires a full **Xcode.app** installation (not just Command Line Tools) because both XCTest and SwiftCheck depend on the XCTest framework, which is only bundled with Xcode. Install Xcode from the App Store and run `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer` before running tests.

Property-based tests use **SwiftCheck** and run a minimum of 100 iterations per property.

---

## Bundling the Python Core (Build Config Notes)

The Python `haki_core_service` executable is bundled inside the `.app` at:

```
HAKI.app/Contents/Resources/haki_core/haki_core_service
```

### How to bundle (development workflow)

1. Build the Core with PyInstaller (or `nuitka`) from the `../Core/` directory:

   ```bash
   cd ../Core
   pyinstaller --onefile haki_core_service.py \
               --name haki_core_service \
               --distpath ../HAKI/.build/core_bundle
   ```

2. Copy the resulting binary into the app bundle after Xcode/SwiftPM builds the `.app`:

   ```bash
   # Run script phase (add to Xcode target → Build Phases → Run Script):
   CORE_BUNDLE="$BUILT_PRODUCTS_DIR/$PRODUCT_NAME.app/Contents/Resources/haki_core"
   mkdir -p "$CORE_BUNDLE"
   cp -R "$SRCROOT/../Core/.build/core_bundle/haki_core_service" "$CORE_BUNDLE/"
   ```

   If using SwiftPM only (no Xcode), add a `Makefile` target that performs the same copy after `swift build`.

3. `CoreProcessManager.start()` resolves the path at runtime and spawns the process passing `--socket <socket_path>`.

### Signing & notarization

- The bundled `haki_core_service` binary **must be ad-hoc or Developer-ID signed** before notarization.
- Add it to the `OTHER_CODE_SIGN_FLAGS` deep-sign list or sign it explicitly:

  ```bash
  codesign --force --sign "Developer ID Application: ..." \
           HAKI.app/Contents/Resources/haki_core/haki_core_service
  ```

- The entitlements file (`HAKI.entitlements`) covers the shell binary. The Core binary requires its own entitlements if it calls any system APIs directly (currently none — it uses the shell as its only OS interface).

---

## TCC Permissions Required

| Permission | Usage description key | Required for |
|---|---|---|
| Screen Recording | `NSScreenRecordingUsageDescription` | `ScreenCaptureKit`, OCR fallback (Req 1, 2) |
| Accessibility | `NSAccessibilityUsageDescription` | AX text extraction, Mac Control (Req 1, 21) |
| Microphone | `NSMicrophoneUsageDescription` | Voice pipeline (Req 3) |
| Calendars | `NSCalendarsUsageDescription` | EventKit calendar events (Req 11, 12) |
| Reminders | `NSRemindersUsageDescription` | EventKit reminders (Req 12) |
| Contacts | `NSContactsUsageDescription` | Contact resolution for messaging (Req 21.11) |
| Automation | `NSAppleEventsUsageDescription` | AppleScript/Apple Events bridge (Req 21) |

---

## Design References

- **Requirements**: `.kiro/specs/haki-personal-ai-assistant/requirements.md`
- **Design**: `.kiro/specs/haki-personal-ai-assistant/design.md`
- **Tasks**: `.kiro/specs/haki-personal-ai-assistant/tasks.md`

Task **1.1** (this scaffold) implements: Design §Architecture (Swift shell), Requirement 20.1.
