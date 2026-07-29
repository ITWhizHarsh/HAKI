  # Implementation Plan: HAKI SwiftUI Frontend

## Overview

Implement a production-ready SwiftUI/AppKit frontend for the HAKI macOS AI agent. The implementation adds a new `HAKIFrontend` SPM executable target with five source files: `HAKIApp.swift`, `MainWorkspaceView.swift`, `JARVISParticleView.swift`, `FloatingPanelManager.swift`, and `ScreenOverlayManager.swift`. The frontend introduces a six-state visual state machine, a 3D SceneKit JARVIS HUD with audio reactivity, a Carbon global hotkey HUD panel, a full-screen agent overlay, and a menu-bar dropdown — all connected to the existing IPC and audio subsystems.

Language: **Swift 5.9 / SwiftUI 5**, macOS 14+ (Sonoma/Sequoia).

---

## Tasks

- [x] 1. Add `HAKIFrontend` SPM executable target and create source directory scaffold
  - Add the new `.executableTarget(name: "HAKIFrontend", dependencies: ["HAKIIPC", "HAKIAudio", "HAKIPermissions"], path: "Sources/HAKIFrontend")` block to `Package.swift`, retaining all existing targets verbatim
  - Create the `Sources/HAKIFrontend/` directory
  - Create stub/empty Swift files for all five required files so the target compiles: `HAKIApp.swift`, `MainWorkspaceView.swift`, `JARVISParticleView.swift`, `FloatingPanelManager.swift`, `ScreenOverlayManager.swift`
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Implement `HAKIState` enum and `HAKIStateModel` observable
  - [x] 2.1 Implement `HAKIState` enum with all six cases and computed properties
    - Define `enum HAKIState: Equatable, CaseIterable` with cases: `idle`, `listening`, `thinking`, `speaking`, `agent`, `error`
    - Implement `var accentColor: Color` returning `.cyan`, `.green`, `Color(red:0.5,green:0.2,blue:1.0)`, `Color(red:0.4,green:0.9,blue:1.0)`, `.orange`, `.red` for each case
    - Implement `var particleEmissionRate: Float` returning 20.0, 80.0, 60.0, 100.0, 50.0, 120.0 for each case
    - Define `Notification.Name` extensions for `hakiAgentModeActivated`, `hakiAgentModeDeactivated`, and `hakiHotkeyCommand`
    - Place all of this at the top of `HAKIApp.swift`
    - _Requirements: 3.1, 3.7, 3.8_

  - [ ]* 2.2 Write property test for `HAKIState.accentColor` correctness
    - **Property 4: HAKIState.accentColor is always defined and correct**
    - **Validates: Requirements 3.7**
    - Use Hypothesis or Swift property test library to verify each HAKIState case returns its specified Color

  - [ ]* 2.3 Write property test for `HAKIState.particleEmissionRate` correctness
    - **Property 5: HAKIState.particleEmissionRate is always defined and positive**
    - **Validates: Requirements 3.8**

  - [x] 2.4 Implement `HAKIStateModel` `@MainActor @Observable` class
    - Define `@MainActor @Observable final class HAKIStateModel`
    - Add `var currentState: HAKIState = .idle` with `didSet` that calls `handleStateTransition(from:to:)`
    - Add `var audioLevel: Float = 0.0` and `var ipcConnected: Bool = false`
    - Implement `handleStateTransition(from:to:)` posting `.hakiAgentModeActivated` when `new == .agent` and `.hakiAgentModeDeactivated` when `old == .agent`
    - _Requirements: 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ]* 2.5 Write property test for agent notification posting
    - **Property 2: Entering .agent always posts agentModeActivated**
    - **Property 3: Leaving .agent always posts agentModeDeactivated**
    - **Validates: Requirements 3.5, 3.6**

- [x] 3. Implement `HAKIApp` entry point with scene body and manager construction
  - [x] 3.1 Implement `HAKIApp` struct with `@main`, `App` conformance, and `init()`
    - Annotate with `@main`, conform to `App`
    - In `init()`: call `NSApp.setActivationPolicy(.regular)`, construct `HAKIStateModel`, `FloatingPanelManager`, `ScreenOverlayManager`, `AVAudioEngine`, and `JSONIPCClient` with socket path resolved from `FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!.appendingPathComponent("HAKI/haki_core.sock")`
    - Call `floatingPanelManager.registerGlobalHotkey()`
    - Launch IPC listener via `Task { await startIPCListener() }`
    - Call `setupAudioTap()` to install the audio tap
    - _Requirements: 2.1, 2.2, 2.5, 2.6, 2.7, 9.1, 10.1_

  - [x] 3.2 Implement `HAKIApp.body` with `WindowGroup` and `MenuBarExtra` scenes
    - Declare `WindowGroup` rendering `MainWorkspaceView().environment(stateModel).frame(minWidth: 800, minHeight: 600)`
    - Declare `MenuBarExtra("HAKI", systemImage: "brain.head.profile")` with `.menuBarExtraStyle(.window)` rendering `StatusBarMenuView().environment(stateModel)`
    - _Requirements: 2.3, 2.4, 2.5, 4.11_

  - [x] 3.3 Implement IPC retry loop `startIPCListener()`
    - Loop from attempt 1 to 12; on each iteration call `ipcClient.connect()`
    - On success: `await MainActor.run { stateModel.ipcConnected = true }`, then iterate `ipcClient.inbound` calling `handleServerMessage(_:)` for each message
    - On stream end: set `ipcConnected = false` and return
    - On catch: set `ipcConnected = false`, log attempt number, sleep `5_000_000_000` nanoseconds; after 12 failures log terminal failure and return
    - _Requirements: 9.2, 9.3, 9.8, 14.1_

  - [x] 3.4 Implement `handleServerMessage(_:)` IPC state dispatch
    - Switch on message type: `.controlEvent` where `eventType == .speakingStarted` → `stateModel.currentState = .speaking`; `speakingStopped` → `.idle`; `.partialTranscript` → `.listening`; `.llmToken` where `isLast == true` → `.speaking`
    - All assignments via `Task { @MainActor in ... }`
    - _Requirements: 9.4, 9.5, 9.6, 9.7_

  - [ ]* 3.5 Write property test for IPC state transition correctness
    - **Property 8: IPC state transitions are correct for all message types**
    - **Validates: Requirements 9.4, 9.5, 9.6, 9.7**

  - [ ]* 3.6 Write property test for IPC retry termination
    - **Property 10: IPC retry terminates after exactly 12 attempts**
    - **Validates: Requirements 14.1**

  - [x] 3.7 Implement `setupAudioTap()` with RMS computation
    - Install tap on `audioEngine.inputNode` with `bufferSize: 1024`
    - In the tap block: if `stateModel.currentState != .listening`, write `0.0` to `stateModel.audioLevel` and return; otherwise compute RMS via `sqrt(sum(sample²) / n)`, clamp to `[0.0, 1.0]`, dispatch `Task { @MainActor in stateModel.audioLevel = clamped }`
    - On `audioEngine.start()` failure: log `"[HAKIFrontend] AVAudioEngine error: \(error)"` and set `audioLevel = 0.0`
    - Add `applicationDidResignActive` / audio-session-interrupted handler that calls `audioEngine.stop()` and resets `audioLevel = 0.0`
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [ ]* 3.8 Write property test for audioLevel clamping
    - **Property 1: audioLevel is always clamped to [0, 1]**
    - **Validates: Requirements 3.3, 10.2**

  - [ ]* 3.9 Write property test for audioLevel zero gate
    - **Property 9: audioLevel is zero when not listening**
    - **Validates: Requirements 10.3**

- [ ] 4. Checkpoint — Ensure app target compiles and state model tests pass
  - Verify `swift build -target HAKIFrontend` completes with zero errors and zero warnings
  - Ensure all property tests for state model pass
  - Ask the user if questions arise.

- [x] 5. Implement `ConversationEntry` scaffold and `MainWorkspaceView`
  - [~] 5.1 Implement `ConversationEntry`, `ConversationRole`, and mock data
    - Define `enum ConversationRole` with cases `user` and `assistant`
    - Define `struct ConversationEntry: Identifiable` with fields `id: UUID`, `role: ConversationRole`, `text: String`, `timestamp: Date`
    - Add `static var mockData: [ConversationEntry]` with at least three entries spanning two distinct calendar days
    - _Requirements: 11.1, 11.2_

  - [ ]* 5.2 Write property test for date grouping correctness
    - **Property 7: Date grouping is stable and complete**
    - **Validates: Requirements 4.2, 11.3**
    - Test that for any array of `ConversationEntry`, grouping produces no lost or duplicated entries and each group shares the same `(year, month, day)` components

  - [x] 5.3 Implement `MainWorkspaceView` `NavigationSplitView` skeleton and sidebar
    - Define `MainWorkspaceView` conforming to `View`, reading `@Environment(HAKIStateModel.self)`
    - Add `@State` for `conversations: [ConversationEntry]`, `selectedEntry: ConversationEntry?`, `commandText: String`, `columnVisibility: NavigationSplitViewVisibility`
    - Render `NavigationSplitView` with sidebar column: date-grouped `List` using `groupedConversations` computed property, `Section` headers using ISO 8601 date string, `lineLimit(1)` truncation to 40 chars, `"HH:mm"` `DateFormatter` for timestamp, and `"No conversations yet"` placeholder when empty
    - On tap: set `selectedEntry` to tapped entry
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 11.2, 11.3, 11.4, 11.5_

  - [x] 5.4 Implement `MainWorkspaceView` detail column with conversation timeline
    - Render `JARVISParticleView(audioLevel: $stateModel.audioLevel)` at top of detail column with `.frame(height: 240)` and full width
    - Render scrollable `ScrollView` below it displaying `ConversationEntry` items as alternating user/assistant message bubbles with `.background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))`, user messages in `Color.primary`, assistant messages in `stateModel.currentState.accentColor`
    - Apply `.accessibilityLabel("HAKI status: \(stateModel.currentState)")` and `.accessibilityHidden(false)` to `JARVISParticleView` wrapper
    - _Requirements: 4.5, 4.6, 12.2, 12.7, 13.3_

  - [x] 5.5 Implement floating bottom command bar
    - Render a `ZStack`-based `RoundedRectangle` bar at the bottom of the detail column using `.background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20))` and `shadow(color: .black.opacity(0.15), radius: 8, y: 4)`
    - Add `TextField("Ask HAKI…", text: $commandText)`
    - Add file attachment `Button` with `.onDrop(of: [.fileURL], isTargeted:perform:)`, `.accessibilityLabel("Attach file")`
    - Add `WaveformView(level: stateModel.audioLevel)` — `HStack` of capsule bars scaled by audio level
    - Add microphone toggle `Button` that switches `stateModel.currentState` between `.listening` and `.idle`, with `.accessibilityLabel("Toggle microphone")` and `.accessibilityHint("Activates or deactivates voice listening mode")`
    - _Requirements: 4.7, 4.8, 4.9, 12.3, 13.1, 13.2_

  - [x] 5.6 Apply glassmorphism, color scheme, error banner, and accessibility
    - Apply `ZStack { Color.clear.background(.ultraThinMaterial) }` to the `NavigationSplitView` container
    - Apply `.preferredColorScheme(nil)` to `MainWorkspaceView`
    - Add `.overlay(alignment: .top)` that shows a `Text("HAKI encountered an error")` banner with `.red.opacity(0.15)` background `ZStack` when `stateModel.currentState == .error`
    - Add `.accessibilityLabel("Toggle conversation history sidebar")` to the sidebar toggle affordance
    - _Requirements: 4.10, 12.1, 12.6, 13.4, 14.4_

- [x] 6. Implement `JARVISParticleView`
  - [x] 6.1 Implement `JARVISParticleView` `NSViewRepresentable` wrapper and `Coordinator`
    - Define `struct JARVISParticleView: NSViewRepresentable` with `@Binding var audioLevel: Float` and `@Environment(HAKIStateModel.self)`
    - Implement `makeCoordinator()` returning a `Coordinator` holding references to particle system, sphere node, ring nodes, and current state
    - In `makeNSView(context:)`: configure `SCNView` with `antialiasingMode: .multisampling4X`, `backgroundColor: .clear`, `allowsCameraControl: false`, set `delegate = context.coordinator`; call `context.coordinator.buildScene(in:)` inside a `do/catch` block — on error log and set `scnView.scene = SCNScene()`
    - In `updateNSView(_:context:)`: update coordinator's `audioLevel` and `currentState` snapshot properties so per-frame logic picks them up
    - _Requirements: 5.1, 14.5_

  - [x] 6.2 Implement `Coordinator.buildScene(in:)` — camera, ring tori, and central sphere
    - Camera node at position `(0, 0, 5)` set as `scene.rootNode.camera`
    - Create three `SCNTorus` nodes with `ringRadius` 1.0 / 1.4 / 1.8 and `pipeRadius: 0.04`; each with `SCNMaterial.emission.contents = currentState.accentColor` and a `SCNAction.repeatForever(rotateBy(...))` with durations 6.0, 4.5, 8.0 s and distinct axis combinations
    - Create `SCNSphere(radius: 0.3)` central node at origin; attach `SCNParticleSystem` with `particleSize: 0.05`, `particleColor = currentState.accentColor`, `emissionDuration: 0`
    - Start `.idle` ambient pulse action on sphere (scale 0.95 ↔ 1.05, 2s, `repeatForever`, `autoreverses: true`)
    - _Requirements: 5.2, 5.3, 5.4, 5.5, 5.8_

  - [x] 6.3 Implement `SCNSceneRendererDelegate.renderer(_:updateAtTime:)` for per-frame reactivity
    - Read coordinator's `audioLevel` and `currentState` snapshots
    - Set `particleSystem.birthRate = Double(audioLevel * currentState.particleEmissionRate + currentState.particleEmissionRate)`
    - Set `sphereNode.scale = SCNVector3(s, s, s)` where `s = 1.0 + Double(audioLevel) * 0.5`
    - On state change: update ring material emission colours to `currentState.accentColor`; if `.thinking`, replace ring rotation actions with durations 1.5, 1.1, 2.0 s; if `.error`, run position-jitter sequence (±0.05 pt, 1.0 s, then return to origin); if returning to `.idle`, restore default rotation durations and ambient pulse
    - _Requirements: 5.4, 5.6, 5.7, 5.9, 5.10, 5.11_

  - [ ]* 6.4 Write property test for `birthRate` formula bounds
    - **Property 6: JARVISParticleView birthRate formula is correct**
    - **Validates: Requirements 5.6**
    - For any `audioLevel` in `[0,1]` and any `HAKIState`, verify `birthRate == Double(audioLevel * state.particleEmissionRate + state.particleEmissionRate)` and `birthRate > 0` and `birthRate <= 2 * state.particleEmissionRate`

- [ ] 7. Checkpoint — Verify SceneKit rendering compiles, no errors
  - Run `swift build -target HAKIFrontend` and confirm zero errors
  - Ensure all property tests introduced so far pass
  - Ask the user if questions arise.

- [x] 8. Implement `StatusBarMenuView`
  - [x] 8.1 Implement `StatusBarMenuView` SwiftUI view
    - Define `struct StatusBarMenuView: View` reading `@Environment(HAKIStateModel.self)`
    - IPC status row: `Circle()` fill `.green` / `.red` based on `stateModel.ipcConnected`, `Text("Core: Connected")` or `"Core: Disconnected"`
    - Audio input row: display `AVCaptureDevice.default(for: .audio)?.localizedName ?? "No mic"`
    - Audio output row: display `AVAudioSession.sharedInstance().currentRoute.outputs.first?.portName ?? "Default"`
    - Voice mode `Toggle("Voice Mode", isOn: voiceModeBinding)` where `voiceModeBinding` reads/writes `stateModel.currentState` between `.listening` and `.idle`
    - `Button("Open HAKI")` calling `NSApp.activate(ignoringOtherApps: true)`
    - `Button("Quit")` calling `NSApplication.shared.terminate(nil)`
    - Wrap in `VStack(alignment: .leading, spacing: 12)` with `.padding()`, `.frame(width: 300)`, `.frame(maxHeight: 400)`, `.background(.ultraThinMaterial)`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 12.4_

- [x] 9. Implement `FloatingPanelManager`
  - [x] 9.1 Implement `FloatingPanelManager` `NSPanel` setup
    - Define `final class FloatingPanelManager: NSObject`
    - In `init()` create `NSPanel(contentRect: NSRect(x:0,y:0,width:480,height:72), styleMask: [.nonactivatingPanel,.borderless], backing: .buffered, defer: false)`
    - Set `panel.level = .floating`, `panel.isOpaque = false`, `panel.backgroundColor = .clear`, `panel.hasShadow = true`
    - Set panel content view to `NSHostingView<HotkeyPanelView>` — a SwiftUI `VStack` inside `RoundedRectangle(.ultraThinMaterial)` with a `@FocusState`-focused `TextField("Ask HAKI…")` and a one-line status label
    - Wire `TextField.onSubmit` to post `Notification.Name.hakiHotkeyCommand` with `userInfo: ["commandText": text]` then call `panel.orderOut(nil)`
    - Wire `Escape` key handler to call `panel.orderOut(nil)`
    - _Requirements: 7.1, 7.2, 7.3, 7.9, 12.5, 13.5_

  - [x] 9.2 Implement `registerGlobalHotkey()` using Carbon `RegisterEventHotKey`
    - Import `Carbon.framework`; define `EventHotKeyID` with signature `"HKFP"`, id `1`
    - Call `RegisterEventHotKey(49, UInt32(optionKey), hotKeyID, GetApplicationEventTarget(), 0, &ref)`
    - If `status != noErr`: log `"[FloatingPanelManager] Failed to register hotkey: \(status)"` and return without crashing
    - Install a Carbon `EventHandlerUPP` callback bridged via `userData = Unmanaged.passUnretained(self)` that calls `toggle()` on `kEventHotKeyPressed`
    - _Requirements: 7.4, 14.2_

  - [x] 9.3 Implement `toggle()` show/hide logic
    - If panel is not visible: call `panel.center()` and `panel.orderFront(nil)`, set `@FocusState` to `true`
    - If panel is already visible: call `panel.orderOut(nil)`
    - _Requirements: 7.5, 7.6, 7.7, 7.8_

- [x] 10. Implement `ScreenOverlayManager`
  - [x] 10.1 Implement `ScreenOverlayManager` `NSWindow` setup and overlay view
    - Define `final class ScreenOverlayManager: NSObject`
    - In `init()` create `NSWindow(contentRect: .zero, styleMask: .borderless, backing: .buffered, defer: false)`
    - Set `overlayWindow.level = .screenSaver`, `isOpaque = false`, `backgroundColor = .clear`, `ignoresMouseEvents = true`, `collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]`
    - Set content view to `NSHostingView<OverlayStrokeView>`: a `ZStack` with `RoundedRectangle(cornerRadius: 16).stroke(Color.orange, lineWidth: 6)` filling the full frame
    - Apply pulsating `.opacity` animation between 0.6 and 1.0 via `.animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: isVisible)`
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 13.6_

  - [x] 10.2 Implement `show()`, `hide()`, and notification observers
    - `show()`: guard `NSScreen.main` else log `"[ScreenOverlayManager] No main screen available"` and return; call `overlayWindow.setFrame(screen.frame, display: false)` and `overlayWindow.orderFront(nil)`
    - `hide()`: call `overlayWindow.orderOut(nil)`
    - In `observeNotifications()`: register on `NotificationCenter.default` for `.hakiAgentModeActivated` → call `DispatchQueue.main.async { self.show() }` and `.hakiAgentModeDeactivated` → `DispatchQueue.main.async { self.hide() }`
    - _Requirements: 8.5, 8.6, 8.7, 8.8, 8.9, 14.3_

- [x] 11. Final integration and wiring
  - [x] 11.1 Wire `HAKIApp` environment injection and verify all scenes compile end-to-end
    - Confirm `HAKIStateModel` is passed via `.environment(stateModel)` to `MainWorkspaceView`, `StatusBarMenuView`, and the `NSHostingView` inside `FloatingPanelManager`
    - Confirm `JARVISParticleView` reads `HAKIStateModel` from environment and `@Binding var audioLevel` from `MainWorkspaceView`
    - Confirm `ScreenOverlayManager` is constructed and retained for app lifetime in `HAKIApp`
    - Remove all placeholder/stub code left from Task 1
    - _Requirements: 2.5, 2.7, 5.11_

  - [x] 11.2 Verify `Package.swift` target correctness and run full build
    - Confirm `HAKIFrontend` target has exactly the five required source files
    - Confirm existing `HAKI` target and library targets are not modified
    - Run `swift build -target HAKIFrontend` and fix any remaining compiler errors or warnings
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 11.3 Write integration tests for full state flow
    - Test that toggling microphone in `MainWorkspaceView` transitions `currentState` between `.idle` and `.listening`
    - Test that receiving `haki.agentModeActivated` notification causes `ScreenOverlayManager` to call `show()` (mock `NSScreen.main`)
    - Test that `FloatingPanelManager.toggle()` shows and hides the panel correctly on successive calls
    - _Requirements: 4.9, 8.7, 8.8, 7.5, 7.6_

- [ ] 12. Final Checkpoint — Ensure all tests pass and build is clean
  - Run `swift build -target HAKIFrontend` with zero errors and zero warnings
  - Ensure all property tests and integration tests pass
  - Ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- The design document's 10 correctness properties are covered by property test sub-tasks in tasks 2, 3, 5, and 6
- Checkpoints in tasks 4, 7, and 12 provide incremental build validation
- All Carbon `RegisterEventHotKey` usage requires linking `Carbon.framework`; add it to the `HAKIFrontend` target's `linkerSettings` in `Package.swift` if not already present via the `HAKIIPC` dependency
- `AVAudioSession` is iOS/macOS cross-platform; on macOS 14 the output route API may require `AVAudioSession.sharedInstance()` workaround — fall back to `"Default"` as required by Req 6.4
- Property tests should be written in a separate `Tests/HAKIFrontendTests/` SPM test target using the Swift Testing framework or XCTest with Hypothesis-style generation

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1", "2.4"] },
    { "id": 2, "tasks": ["2.2", "2.3", "2.5", "3.1"] },
    { "id": 3, "tasks": ["3.2", "3.3", "3.4", "5.1"] },
    { "id": 4, "tasks": ["3.5", "3.6", "3.7", "5.2", "5.3"] },
    { "id": 5, "tasks": ["3.8", "3.9", "5.4", "5.5", "6.1"] },
    { "id": 6, "tasks": ["5.6", "6.2", "8.1"] },
    { "id": 7, "tasks": ["6.3", "9.1", "10.1"] },
    { "id": 8, "tasks": ["6.4", "9.2", "10.2"] },
    { "id": 9, "tasks": ["9.3", "11.1"] },
    { "id": 10, "tasks": ["11.2"] },
    { "id": 11, "tasks": ["11.3"] }
  ]
}
```
