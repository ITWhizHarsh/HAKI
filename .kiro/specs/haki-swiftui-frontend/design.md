# Design Document — HAKI SwiftUI Frontend

## Overview

The `HAKIFrontend` SPM executable target delivers a production-quality SwiftUI/AppKit frontend for the HAKI macOS AI agent. It replaces the existing `AppDelegate` + `main.swift` entry point with a SwiftUI `@main HAKIApp`, introducing four distinct window layers — a primary workspace, a menu-bar dropdown, a global floating hotkey HUD, and an autonomous screen-control overlay — all driven by a six-state observable state machine (`HAKIStateModel`) wired to the existing IPC and audio subsystems.

The design targets macOS 14+ (Sonoma/Sequoia), uses Swift 5.9 / SwiftUI 5, and introduces zero external 3D model dependencies. All surfaces use Apple HIG glassmorphism (`.ultraThinMaterial`).

---

## Architecture

### High-Level Component Map

```
┌─────────────────────────────────────────────────────────────┐
│                        HAKIApp (@main)                      │
│  ┌──────────────────────┐  ┌──────────────────────────────┐ │
│  │    WindowGroup        │  │      MenuBarExtra            │ │
│  │  MainWorkspaceView    │  │    StatusBarMenuView         │ │
│  └──────────────────────┘  └──────────────────────────────┘ │
│  ┌──────────────────────┐  ┌──────────────────────────────┐ │
│  │  FloatingPanelManager│  │   ScreenOverlayManager       │ │
│  │  (NSPanel HUD)       │  │   (NSWindow overlay)         │ │
│  └──────────────────────┘  └──────────────────────────────┘ │
│                                                             │
│         HAKIStateModel (@Observable @MainActor)             │
│   currentState: HAKIState   audioLevel: Float               │
│   ipcConnected: Bool                                        │
│                                                             │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  HAKIIPC       │  │  HAKIAudio   │  │ HAKIPermissions│  │
│  │ JSONIPCClient  │  │ AVAudioEngine│  │ PermissionMgr  │  │
│  └────────────────┘  └──────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Process & Threading Model

- **Main Actor**: `HAKIStateModel`, all SwiftUI views, `FloatingPanelManager`, `ScreenOverlayManager`, `HAKIApp` init side-effects.
- **Background Task**: IPC listener (`Task { }` from `HAKIApp.init`) — reads `JSONIPCClient.inbound` async stream, dispatches state updates via `Task { @MainActor in ... }`.
- **AVAudioEngine realtime thread**: Audio tap callback — computes RMS, dispatches `Task { @MainActor in ... }` to write `audioLevel`.
- **NWConnection queue** (`haki.ipc.json`): Internal to `JSONIPCClient`; no direct coupling to frontend.

---

## SPM Target Layout

### `Package.swift` Additions

```swift
// New executable target
.executableTarget(
    name: "HAKIFrontend",
    dependencies: ["HAKIIPC", "HAKIAudio", "HAKIPermissions"],
    path: "Sources/HAKIFrontend"
),
```

The existing `HAKI` executable target, all library targets, and all test targets are retained verbatim. The `HAKIFrontend` target does **not** depend on `HAKIUI` to avoid pulling in `ConversationWindowController` and legacy `AppDelegate` references.

### Source Files in `Sources/HAKIFrontend/`

| File | Responsibility |
|---|---|
| `HAKIApp.swift` | `@main` entry, `WindowGroup`, `MenuBarExtra`, manager construction, IPC + audio setup |
| `MainWorkspaceView.swift` | `NavigationSplitView`, conversation scaffold, `JARVISParticleView`, command bar |
| `JARVISParticleView.swift` | SceneKit 3D JARVIS HUD, audio-reactive particles, state-driven visuals |
| `FloatingPanelManager.swift` | NSPanel hotkey HUD, Carbon `RegisterEventHotKey`, dismiss/show logic |
| `ScreenOverlayManager.swift` | NSWindow screen-control overlay, notification observers, pulsating stroke |

---

## Component Design

### HAKIState + HAKIStateModel

`HAKIState` is a plain Swift enum with six cases acting as the single source of truth for all visual surfaces. `HAKIStateModel` is `@MainActor @Observable` — it can be observed by SwiftUI views without `@Published`; mutations happen only on the main actor via `Task { @MainActor in ... }` trampolines from background threads.

```swift
// HAKIApp.swift

enum HAKIState: Equatable, CaseIterable {
    case idle, listening, thinking, speaking, agent, error

    var accentColor: Color {
        switch self {
        case .idle:      return .cyan
        case .listening: return .green
        case .thinking:  return Color(red: 0.5, green: 0.2, blue: 1.0)
        case .speaking:  return Color(red: 0.4, green: 0.9, blue: 1.0)
        case .agent:     return .orange
        case .error:     return .red
        }
    }

    var particleEmissionRate: Float {
        switch self {
        case .idle:      return 20.0
        case .listening: return 80.0
        case .thinking:  return 60.0
        case .speaking:  return 100.0
        case .agent:     return 50.0
        case .error:     return 120.0
        }
    }
}

@MainActor
@Observable
final class HAKIStateModel {
    var currentState: HAKIState = .idle {
        didSet { handleStateTransition(from: oldValue, to: currentState) }
    }
    var audioLevel: Float = 0.0
    var ipcConnected: Bool = false

    private func handleStateTransition(from old: HAKIState, to new: HAKIState) {
        if new == .agent {
            NotificationCenter.default.post(name: .hakiAgentModeActivated, object: nil)
        } else if old == .agent {
            NotificationCenter.default.post(name: .hakiAgentModeDeactivated, object: nil)
        }
    }
}

extension Notification.Name {
    static let hakiAgentModeActivated   = Notification.Name("haki.agentModeActivated")
    static let hakiAgentModeDeactivated = Notification.Name("haki.agentModeDeactivated")
    static let hakiHotkeyCommand        = Notification.Name("haki.hotkeyCommand")
}
```

**State transition diagram:**

```
idle ←──────────────────────────────────┐
 │ (user taps mic / partialTranscript)  │
 ▼                                      │
listening ──(LLM token isLast)──► speaking ──(speakingStopped)──► idle
 │                                      │
 │  (llmToken flowing)                  │
 ▼                                      │
thinking ──────────────────────────────►┘
 │
 └──► agent  (haki.agentModeActivated notification)
 │
 └──► error  (error banner shown)
```

---

### HAKIApp Entry Point

`HAKIApp` is the `@main` struct. It owns all manager instances and injects `HAKIStateModel` into both SwiftUI scenes via `.environment(stateModel)`.

```swift
@main
struct HAKIApp: App {
    private let stateModel = HAKIStateModel()
    private let floatingPanelManager: FloatingPanelManager
    private let screenOverlayManager: ScreenOverlayManager
    private let ipcClient: JSONIPCClient
    private let audioEngine = AVAudioEngine()

    init() {
        NSApp.setActivationPolicy(.regular)

        // Construct managers
        floatingPanelManager = FloatingPanelManager()
        floatingPanelManager.registerGlobalHotkey()
        screenOverlayManager = ScreenOverlayManager()

        // IPC socket path
        let appSupport = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first!
            .appendingPathComponent("HAKI/haki_core.sock")
        ipcClient = JSONIPCClient(socketPath: appSupport)

        // Start IPC listener and audio tap
        Task { await startIPCListener() }
        setupAudioTap()
    }

    var body: some Scene {
        WindowGroup {
            MainWorkspaceView()
                .environment(stateModel)
                .frame(minWidth: 800, minHeight: 600)
        }
        MenuBarExtra("HAKI", systemImage: "brain.head.profile") {
            StatusBarMenuView()
                .environment(stateModel)
        }
        .menuBarExtraStyle(.window)
    }
}
```

**IPC retry logic** (Req 14.1): The `startIPCListener` function retries connection every 5 seconds up to 12 attempts before logging a terminal failure. Each failed attempt sets `stateModel.ipcConnected = false`; success sets it to `true`.

**Audio tap** (Req 10): Installed on `AVAudioEngine.inputNode` with `bufferSize: 1024`. The tap block computes RMS of the PCM channel data, clamps to `[0,1]`, and writes to `stateModel.audioLevel` on the main actor. When `stateModel.currentState != .listening`, the tap writes `0.0` instead.

---

### MainWorkspaceView

Root layout is a `NavigationSplitView` with:
- **Sidebar column**: date-grouped `List` of `ConversationEntry` items, `Section` headers using ISO 8601 date strings, "No conversations yet" placeholder when empty.
- **Detail column**: `JARVISParticleView` (240 pt tall, full width) above a scrollable conversation timeline, above the floating command bar.

```swift
struct MainWorkspaceView: View {
    @Environment(HAKIStateModel.self) private var stateModel
    @State private var conversations: [ConversationEntry] = ConversationEntry.mockData
    @State private var selectedEntry: ConversationEntry?
    @State private var commandText: String = ""
    @State private var columnVisibility = NavigationSplitViewVisibility.all

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            sidebarContent
        } detail: {
            detailContent
        }
        .background(.ultraThinMaterial)
        .preferredColorScheme(nil)
        .overlay(alignment: .top) {
            if stateModel.currentState == .error {
                errorBanner
            }
        }
    }
}
```

**Conversation grouping** uses `Dictionary(grouping:by:)` keyed on `Calendar.current.dateComponents([.year, .month, .day], from: entry.timestamp)` — `DateComponents` is `Hashable` in Swift, enabling clean grouping. Groups are sorted descending by date.

**Command bar** contains:
- `TextField("Ask HAKI…", text: $commandText)` with `.onSubmit` handler
- File attachment `Button` with `.onDrop(of: [.fileURL], isTargeted:perform:)` 
- `WaveformView(level: stateModel.audioLevel)` — a simple `HStack` of capsule-shaped bars scaled by the audio level
- Microphone toggle button that switches `stateModel.currentState` between `.listening` and `.idle`

---

### JARVISParticleView

`NSViewRepresentable` wrapping an `SCNView`. The `Coordinator` class acts as `SCNSceneRendererDelegate` to update particle birth rate and sphere scale every frame.

```swift
struct JARVISParticleView: NSViewRepresentable {
    @Binding var audioLevel: Float
    @Environment(HAKIStateModel.self) private var stateModel

    func makeNSView(context: Context) -> SCNView {
        let scnView = SCNView()
        scnView.antialiasingMode = .multisampling4X
        scnView.backgroundColor = .clear
        scnView.allowsCameraControl = false
        scnView.delegate = context.coordinator

        do {
            try context.coordinator.buildScene(in: scnView)
        } catch {
            print("[JARVISParticleView] Scene setup failed: \(error)")
            scnView.scene = SCNScene() // fallback
        }
        return scnView
    }
}
```

**Scene construction** (`Coordinator.buildScene`):
1. Create camera node at `(0, 0, 5)`.
2. Create three `SCNTorus` ring nodes (radii 1.0, 1.4, 1.8; pipeRadius 0.04) each with `SCNMaterial` emission set to `currentState.accentColor` and a `repeatForever(rotateBy)` action with durations 6.0, 4.5, 8.0 s and distinct axes.
3. Create central `SCNSphere(radius: 0.3)` with an attached `SCNParticleSystem`.
4. Set up `.idle` ambient scale pulse (0.95 ↔ 1.05, 2s loop).

**Per-frame update** (`renderer(_:updateAtTime:)` on the coordinator):
- Read `audioLevel` and `currentState` (captured as properties on the coordinator, updated via `updateNSView`).
- Set `particleSystem.birthRate = Double(audioLevel * currentState.particleEmissionRate + currentState.particleEmissionRate)`.
- Set sphere node `scale = SCNVector3(s, s, s)` where `s = 1.0 + Double(audioLevel) * 0.5`.
- If state changed, update ring material emission colors and swap rotation actions as needed.

**State-specific overrides**:
- `.thinking`: Replace ring rotation actions with faster durations (1.5, 1.1, 2.0 s).
- `.error`: Apply jitter action (±0.05 pt translation, 1.0 s sequence, then return to origin).
- `.idle`: Ensure ambient pulse action running on sphere.

---

### StatusBarMenuView

Fixed-width (300 pt) `VStack` rendered inside `MenuBarExtra(.window)`. Uses `@Environment(HAKIStateModel.self)` for live state.

```swift
struct StatusBarMenuView: View {
    @Environment(HAKIStateModel.self) private var stateModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ipcStatusRow
            audioInputRow
            audioOutputRow
            Divider()
            voiceModeToggle
            Divider()
            openHAKIButton
            quitButton
        }
        .padding()
        .frame(width: 300)
        .frame(maxHeight: 400)
        .background(.ultraThinMaterial)
    }
}
```

Audio device names are fetched once on appear via `AVCaptureDevice.default(for: .audio)?.localizedName` and `AVAudioSession.sharedInstance().currentRoute.outputs.first?.portName`. The voice mode `Toggle` is bound to a computed `Binding<Bool>` that reads/writes `stateModel.currentState`.

---

### FloatingPanelManager

`NSObject` subclass managing one `NSPanel`. Hotkey registration uses Carbon's `RegisterEventHotKey` (keyCode 49 = Space, modifier `optionKey`).

```swift
final class FloatingPanelManager: NSObject {
    private var panel: NSPanel!
    private var hotKeyRef: EventHotKeyRef?
    private var eventHandler: EventHandlerRef?

    func registerGlobalHotkey() {
        var hotKeyID = EventHotKeyID(signature: OSType("HKFP"), id: 1)
        var ref: EventHotKeyRef?
        let status = RegisterEventHotKey(49, UInt32(optionKey), hotKeyID,
                                         GetApplicationEventTarget(), 0, &ref)
        if status != noErr {
            print("[FloatingPanelManager] Failed to register hotkey: \(status)")
            return
        }
        hotKeyRef = ref
        // Install Carbon event handler for kEventHotKeyPressed
        installCarbonHandler()
    }
}
```

The Carbon `EventHandlerUPP` callback is a C function pointer; it is bridged to the Swift object via `userData` carrying an `Unmanaged<FloatingPanelManager>` pointer. When fired, it calls `toggle()` on the manager.

**Panel content view** is an `NSHostingView<HotkeyPanelView>` — a SwiftUI `VStack` inside a `RoundedRectangle(.ultraThinMaterial)` background with a focused `TextField("Ask HAKI…", text: $commandText)`. `@FocusState` is set `true` when the panel becomes visible. Pressing `Return` posts `haki.hotkeyCommand` with key `"commandText"` and dismisses the panel. Pressing `Escape` dismisses immediately.

**Panel configuration**: `NSPanel(contentRect: NSRect(x: 0, y: 0, width: 480, height: 72), styleMask: [.nonactivatingPanel, .borderless], backing: .buffered, defer: false)`. Level `.floating`, `isOpaque = false`, `backgroundColor = .clear`, `hasShadow = true`.

---

### ScreenOverlayManager

`NSObject` subclass managing one full-screen `NSWindow`. Observes `haki.agentModeActivated` / `haki.agentModeDeactivated` notifications on `NotificationCenter.default`.

```swift
final class ScreenOverlayManager: NSObject {
    private var overlayWindow: NSWindow!

    override init() {
        super.init()
        buildOverlayWindow()
        observeNotifications()
    }

    func show() {
        guard let screen = NSScreen.main else {
            print("[ScreenOverlayManager] No main screen available")
            return
        }
        overlayWindow.setFrame(screen.frame, display: false)
        overlayWindow.orderFront(nil)
    }

    func hide() {
        overlayWindow.orderOut(nil)
    }
}
```

**Window config**: `styleMask: .borderless`, `level: .screenSaver`, `isOpaque: false`, `backgroundColor: .clear`, `ignoresMouseEvents: true`, `collectionBehavior: [.canJoinAllSpaces, .fullScreenAuxiliary]`.

**Content view**: `NSHostingView<OverlayStrokeView>` containing a `ZStack` with a `RoundedRectangle(cornerRadius: 16).stroke(Color.orange, lineWidth: 6)` filling the full frame. The opacity animates between 0.6 and 1.0 with `.easeInOut(duration: 0.8).repeatForever(autoreverses: true)`.

---

### IPC Integration Data Flow

```
JSONIPCClient.inbound (AsyncStream<ServerMessage>)
    │
    ▼  (background Task)
for await message in ipcClient.inbound {
    switch message {
    case .controlEvent(let ce) where ce.eventType == .speakingStarted:
        → stateModel.currentState = .speaking
    case .controlEvent(let ce) where ce.eventType == .speakingStopped:
        → stateModel.currentState = .idle
    case .partialTranscript:
        → stateModel.currentState = .listening
    case .llmToken(let t) where t.isLast:
        → stateModel.currentState = .speaking
    }
}
```

**Retry loop** (up to 12 attempts, 5-second intervals):

```swift
private func startIPCListener() async {
    for attempt in 1...12 {
        do {
            try await ipcClient.connect()
            await MainActor.run { stateModel.ipcConnected = true }
            for await message in ipcClient.inbound {
                await handleServerMessage(message)
            }
            // Stream ended cleanly — connection lost
            await MainActor.run { stateModel.ipcConnected = false }
            return
        } catch {
            await MainActor.run { stateModel.ipcConnected = false }
            print("[HAKIFrontend] IPC attempt \(attempt)/12 failed: \(error)")
            if attempt == 12 {
                print("[HAKIFrontend] IPC: terminal failure after 12 attempts")
                return
            }
            try? await Task.sleep(nanoseconds: 5_000_000_000)
        }
    }
}
```

---

### Audio Reactivity Pipeline

```
AVAudioEngine.inputNode tap (bufferSize: 1024, realtime thread)
    │
    ▼
RMS = sqrt( Σ(sample²) / n )  (Float32 channel data)
clamped = min(max(rms, 0.0), 1.0)
    │
    ▼
Task { @MainActor in
    stateModel.audioLevel = (stateModel.currentState == .listening) ? clamped : 0.0
}
```

**RMS computation** over `bufferSize: 1024` samples at the hardware sample rate (typically 44.1 / 48 kHz). The value is normalised assuming Float32 range `[-1, 1]` so RMS is already in `[0, 1]` after clamping. On `AVAudioEngine` start failure, error is logged and `audioLevel` is set to `0.0`.

---

### ConversationEntry Scaffold

```swift
// In MainWorkspaceView.swift

enum ConversationRole { case user, assistant }

struct ConversationEntry: Identifiable {
    let id: UUID
    let role: ConversationRole
    let text: String
    let timestamp: Date
}
```

Mock data spans two calendar days. Grouping is computed via:

```swift
var groupedConversations: [(key: DateComponents, value: [ConversationEntry])] {
    let grouped = Dictionary(grouping: conversations) { entry in
        Calendar.current.dateComponents([.year, .month, .day], from: entry.timestamp)
    }
    return grouped.sorted { a, b in
        (a.key.year ?? 0) > (b.key.year ?? 0) ||
        (a.key.month ?? 0) > (b.key.month ?? 0) ||
        (a.key.day ?? 0) > (b.key.day ?? 0)
    }
}
```

---

### Glassmorphism Design System

All surfaces use a consistent layering approach:

| Layer | Material | Corner Radius | Shadow |
|---|---|---|---|
| Main workspace background | `.ultraThinMaterial` | — | — |
| Message bubbles | `.ultraThinMaterial` | 16 | — |
| Command bar | `.ultraThinMaterial` | 20 | `.black.opacity(0.15), r:8, y:4` |
| Status bar menu | `.ultraThinMaterial` | — | — |
| Hotkey HUD panel | `.ultraThinMaterial` | 16 | (NSPanel hasShadow) |

No hardcoded colour scheme. All text uses `Color.primary` for user messages and `HAKIState.accentColor` for assistant messages. `.preferredColorScheme(nil)` defers to system appearance.

---

### Error Handling Summary

| Failure | Response |
|---|---|
| IPC socket not found | Retry every 5s up to 12 times, set `ipcConnected = false`, log terminal failure |
| `RegisterEventHotKey` fails | Log `"[FloatingPanelManager] Failed to register hotkey: \(status)"`, no crash |
| `NSScreen.main == nil` | Log `"[ScreenOverlayManager] No main screen available"`, return without showing |
| `AVAudioEngine.start()` fails | Log error, set `audioLevel = 0.0` |
| SceneKit node creation fails | Catch in `do/catch`, fall back to empty `SCNScene`, log error |
| `currentState == .error` | Show non-modal error banner at top of detail column |

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: audioLevel is always clamped to [0, 1]

*For any* Float value computed as the RMS amplitude from the audio tap — regardless of hardware noise, buffer anomalies, or computed values outside the normal range — the value assigned to `HAKIStateModel.audioLevel` SHALL always satisfy `0.0 <= audioLevel <= 1.0`.

**Validates: Requirements 3.3, 10.2**

---

### Property 2: Entering .agent always posts agentModeActivated

*For any* prior `HAKIState` value (including all six cases), setting `HAKIStateModel.currentState` to `.agent` SHALL always result in exactly one `haki.agentModeActivated` notification being posted to `NotificationCenter.default`.

**Validates: Requirements 3.5**

---

### Property 3: Leaving .agent always posts agentModeDeactivated

*For any* destination `HAKIState` value that is not `.agent`, transitioning `HAKIStateModel.currentState` away from `.agent` SHALL always result in exactly one `haki.agentModeDeactivated` notification being posted to `NotificationCenter.default`.

**Validates: Requirements 3.6**

---

### Property 4: HAKIState.accentColor is always defined and correct

*For any* `HAKIState` case, invoking `accentColor` SHALL return a non-nil `Color` value matching the specification: `.cyan` for `idle`, `.green` for `listening`, the specified purple for `thinking`, the specified light-blue for `speaking`, `.orange` for `agent`, and `.red` for `error`.

**Validates: Requirements 3.7**

---

### Property 5: HAKIState.particleEmissionRate is always defined and positive

*For any* `HAKIState` case, invoking `particleEmissionRate` SHALL return a positive `Float` matching the specification: 20.0, 80.0, 60.0, 100.0, 50.0, 120.0 for `idle`, `listening`, `thinking`, `speaking`, `agent`, `error` respectively.

**Validates: Requirements 3.8**

---

### Property 6: JARVISParticleView birthRate formula is correct

*For any* `audioLevel` in `[0, 1]` and any `HAKIState`, the `SCNParticleSystem.birthRate` SHALL equal `Double(audioLevel * state.particleEmissionRate + state.particleEmissionRate)`, which is always positive and bounded above by `2 * state.particleEmissionRate`.

**Validates: Requirements 5.6**

---

### Property 7: Date grouping is stable and complete

*For any* non-empty array of `ConversationEntry` items with arbitrary `timestamp` values, the date-grouping algorithm SHALL produce groups such that: (a) every entry appears in exactly one group, (b) all entries in a group share the same `(year, month, day)` calendar components, and (c) no entries are lost or duplicated across groups.

**Validates: Requirements 4.2, 11.3**

---

### Property 8: IPC state transitions are correct for all message types

*For any* sequence of `ServerMessage` values received from the IPC stream, each of the following mappings SHALL hold deterministically: `controlEvent(.speakingStarted)` → `currentState == .speaking`; `controlEvent(.speakingStopped)` → `currentState == .idle`; `partialTranscript(_)` → `currentState == .listening`; `llmToken(isLast: true)` → `currentState == .speaking`.

**Validates: Requirements 9.4, 9.5, 9.6, 9.7**

---

### Property 9: audioLevel is zero when not listening

*For any* `HAKIState` other than `.listening`, regardless of the actual microphone RMS amplitude, the value written to `HAKIStateModel.audioLevel` by the audio tap SHALL be `0.0`.

**Validates: Requirements 10.3**

---

### Property 10: IPC retry terminates after exactly 12 attempts

*For any* sequence of consecutive IPC connection failures, the retry loop SHALL attempt reconnection at most 12 times (with 5-second intervals) before logging a terminal failure message and stopping — never exceeding 12 attempts regardless of error type.

**Validates: Requirements 14.1**
