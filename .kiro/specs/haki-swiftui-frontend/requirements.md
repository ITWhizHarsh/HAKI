# Requirements Document

## Introduction

This feature replaces the existing AppKit `AppDelegate` / `main.swift` entry point with a production-ready SwiftUI `@main HAKIApp` frontend for the HAKI macOS application (macOS 14+, Sonoma/Sequoia). The frontend introduces a new SPM executable target `HAKIFrontend` that shares the existing `HAKIIPC`, `HAKIAudio`, and `HAKIPermissions` library targets. It implements four distinct window layers: a primary `NavigationSplitView` Main Workspace, a `MenuBarExtra` status-bar dropdown, a global floating hotkey `NSPanel` HUD, and an autonomous screen-control `NSWindow` overlay. A 3D SceneKit JARVIS HUD (`JARVISParticleView`) with audio reactivity and a six-state visual state machine (`HAKIState`) is embedded in the Main Workspace center canvas. All surfaces use Apple HIG-compliant dark/light mode glassmorphism (`.ultraThinMaterial`) with zero external 3D model dependencies.

---

## Glossary

- **HAKIApp**: The `@main` SwiftUI application entry point that replaces `main.swift` and `AppDelegate.swift` for the `HAKIFrontend` target.
- **HAKIFrontend**: The new SPM `executableTarget` housing all SwiftUI/AppKit frontend source files.
- **HAKIState**: The six-value enum encoding HAKI's operating mode — `idle`, `listening`, `thinking`, `speaking`, `agent`, `error`.
- **JARVISParticleView**: The `NSViewRepresentable`-wrapped SceneKit view rendering the 3D animated JARVIS HUD.
- **MainWorkspaceView**: The primary `WindowGroup` scene body containing the `NavigationSplitView`, conversation timeline, and bottom command bar.
- **StatusBarMenuView**: The `MenuBarExtra` scene content view showing IPC connection status, audio device selectors, and voice-mode toggles.
- **FloatingPanelManager**: The `NSPanel`-backed AppKit coordinator managing the global hotkey HUD overlay.
- **ScreenOverlayManager**: The `NSWindow`-backed AppKit coordinator managing the autonomous screen-control perimeter overlay.
- **ConversationEntry**: A single persisted or in-memory turn in the conversation timeline (role, text, timestamp).
- **AudioLevel**: A `Float` in the range `[0.0, 1.0]` representing the normalised real-time microphone RMS amplitude from `AVAudioEngine`.
- **IPC Socket**: The UNIX domain socket at `~/Library/Application Support/HAKI/haki_core.sock` used for full-duplex JSON-RPC messaging with the HAKI Core Python process.
- **Glassmorphism**: A visual treatment combining `NSVisualEffectView` / `.ultraThinMaterial` backgrounds with translucent, blurred layers per Apple HIG.
- **SCNTorus**: A SceneKit torus geometry node used for the three nested wireframe ring primitives in `JARVISParticleView`.
- **SCNParticleSystem**: A SceneKit particle system attached to the central sphere node in `JARVISParticleView`.
- **NSPanel**: An AppKit panel window subclass used for the non-activating global hotkey HUD overlay.
- **ScreenSaver level**: `NSWindow.Level.screenSaver`, the window stacking level used for the screen-control perimeter overlay.

---

## Requirements

### Requirement 1 — HAKIFrontend SPM Target

**User Story:** As a developer, I want a dedicated `HAKIFrontend` SPM executable target, so that the new SwiftUI frontend compiles independently while sharing existing subsystem libraries.

#### Acceptance Criteria

1. THE `Package.swift` SHALL define a new `.executableTarget` named `HAKIFrontend` with source path `Sources/HAKIFrontend` and dependencies `["HAKIIPC", "HAKIAudio", "HAKIPermissions"]`.
2. THE `Package.swift` SHALL retain the existing `HAKI` executable target and all existing library targets without modification to their declarations.
3. WHEN the `HAKIFrontend` target is built with `swift build -target HAKIFrontend`, THE Swift compiler SHALL produce zero errors and zero warnings under the `macOS(.v14)` platform constraint.
4. THE `HAKIFrontend` target SHALL contain exactly the files: `HAKIApp.swift`, `JARVISParticleView.swift`, `MainWorkspaceView.swift`, `FloatingPanelManager.swift`, and `ScreenOverlayManager.swift`.
5. THE `HAKIFrontend` target SHALL NOT import any SPM target that is not `HAKIIPC`, `HAKIAudio`, or `HAKIPermissions`.

---

### Requirement 2 — SwiftUI App Entry Point (`HAKIApp.swift`)

**User Story:** As a macOS user, I want HAKI to launch as a regular app with a dock icon and open its main workspace on launch, so that I can interact with the full UI immediately.

#### Acceptance Criteria

1. THE `HAKIApp` struct SHALL be annotated with `@main` and conform to the SwiftUI `App` protocol, replacing the `MainActor.assumeIsolated` `NSApplicationMain` block in `main.swift` for the `HAKIFrontend` target.
2. WHEN `HAKIApp` initialises, THE `HAKIApp` SHALL set `NSApp.setActivationPolicy(.regular)` so that HAKI appears in the macOS Dock.
3. THE `HAKIApp.body` SHALL declare a `WindowGroup` scene rendering `MainWorkspaceView` as its content.
4. THE `HAKIApp.body` SHALL declare a `MenuBarExtra("HAKI", systemImage: "brain.head.profile")` scene with `.menuBarExtraStyle(.window)` rendering `StatusBarMenuView`.
5. WHEN `HAKIApp` initialises, THE `HAKIApp` SHALL construct a single `HAKIStateModel` observable object and inject it into the SwiftUI environment via `.environmentObject(_:)` on both scenes.
6. WHEN `HAKIApp` initialises, THE `HAKIApp` SHALL construct a `FloatingPanelManager` and call `FloatingPanelManager.registerGlobalHotkey()` to register the `Option + Space` global key combination.
7. WHEN `HAKIApp` initialises, THE `HAKIApp` SHALL construct a `ScreenOverlayManager` and store it as a retained property for the lifetime of the application.
8. THE `HAKIApp` SHALL NOT reference `AppDelegate`, `ConversationWindowController`, or `CoreProcessManager` from the existing `HAKI` target.

---

### Requirement 3 — HAKIState Visual State Machine

**User Story:** As a developer, I want a `HAKIState` enum and `HAKIStateModel` observable, so that all UI surfaces react to the same authoritative HAKI operating mode.

#### Acceptance Criteria

1. THE `HAKIState` enum SHALL declare exactly six cases: `idle`, `listening`, `thinking`, `speaking`, `agent`, `error`.
2. THE `HAKIStateModel` class SHALL be annotated `@MainActor` and `@Observable` and SHALL expose a `var currentState: HAKIState` property initialised to `.idle`.
3. THE `HAKIStateModel` SHALL expose a `var audioLevel: Float` property in the range `[0.0, 1.0]` representing live microphone RMS amplitude, initialised to `0.0`.
4. THE `HAKIStateModel` SHALL expose a `var ipcConnected: Bool` property initialised to `false`, updated to `true` when the IPC socket handshake completes successfully.
5. WHEN `HAKIStateModel.currentState` transitions to `.agent`, THE `HAKIStateModel` SHALL post a `Notification` named `haki.agentModeActivated` on `NotificationCenter.default`.
6. WHEN `HAKIStateModel.currentState` transitions away from `.agent`, THE `HAKIStateModel` SHALL post a `Notification` named `haki.agentModeDeactivated` on `NotificationCenter.default`.
7. THE `HAKIState` enum SHALL provide a `var accentColor: Color` computed property returning: `.cyan` for `idle`, `.green` for `listening`, `Color(red: 0.5, green: 0.2, blue: 1.0)` for `thinking`, `Color(red: 0.4, green: 0.9, blue: 1.0)` for `speaking`, `.orange` for `agent`, `.red` for `error`.
8. THE `HAKIState` enum SHALL provide a `var particleEmissionRate: Float` computed property returning: `20.0` for `idle`, `80.0` for `listening`, `60.0` for `thinking`, `100.0` for `speaking`, `50.0` for `agent`, `120.0` for `error`.

---

### Requirement 4 — Main Workspace View (`MainWorkspaceView.swift`)

**User Story:** As a user, I want a full macOS workspace window with a collapsible sidebar for conversation history and a central AI interaction canvas, so that I can browse past conversations and interact with HAKI in one place.

#### Acceptance Criteria

1. THE `MainWorkspaceView` SHALL render a `NavigationSplitView` with a collapsible sidebar column and a detail column as the root layout.
2. THE sidebar column SHALL display a `List` of date-grouped `ConversationEntry` items read from a `conversations/` directory scaffold, with each group headed by a `Section` whose title is the ISO 8601 date string of the group.
3. WHEN the sidebar contains no conversation entries, THE `MainWorkspaceView` SHALL render a placeholder `Text("No conversations yet")` view in the sidebar.
4. THE `NavigationSplitView` sidebar SHALL support toggling via `Cmd + S` keyboard shortcut, animating the column collapse with the default `NavigationSplitView` transition.
5. THE detail column SHALL render `JARVISParticleView` centred at the top of the canvas, with a fixed height of `240 pt` and full column width.
6. THE detail column SHALL render a scrollable conversation timeline below `JARVISParticleView`, displaying `ConversationEntry` items as alternating user/assistant message bubbles with `.ultraThinMaterial` backgrounds and `cornerRadius(16)`.
7. THE detail column SHALL render a floating bottom command bar occupying full column width with a `ZStack`-based `RoundedRectangle` background using `.ultraThinMaterial` and `cornerRadius(20)`.
8. THE bottom command bar SHALL contain a `TextField` for text input, a file-attachment button accepting drag-and-drop of file URLs via `.onDrop(of: [.fileURL], ...)`, a live audio waveform visualiser view driven by `HAKIStateModel.audioLevel`, and a microphone toggle button.
9. WHEN the microphone toggle button is tapped, THE `MainWorkspaceView` SHALL toggle `HAKIStateModel.currentState` between `.listening` and `.idle`.
10. THE `MainWorkspaceView` SHALL support both light mode and dark mode by using only SwiftUI semantic colours and `.ultraThinMaterial` for all backgrounds.
11. THE `MainWorkspaceView` SHALL have a minimum window size of `800 × 600 pt` enforced via `.frame(minWidth: 800, minHeight: 600)` on the `WindowGroup` scene.

---

### Requirement 5 — JARVIS Particle HUD (`JARVISParticleView.swift`)

**User Story:** As a user, I want an animated 3D holographic HUD at the top of the main canvas that reacts visually to my voice and HAKI's current state, so that I have clear real-time feedback about what HAKI is doing.

#### Acceptance Criteria

1. THE `JARVISParticleView` SHALL conform to `NSViewRepresentable` and wrap a `SCNView` configured with `antialiasingMode: .multisampling4X`, `backgroundColor: .clear`, and `allowsCameraControl: false`.
2. THE `JARVISParticleView` SHALL create exactly three `SCNTorus` geometry nodes with `pipeRadius: 0.04`, `ringRadius` values of `1.0`, `1.4`, and `1.8` respectively, each wrapped in an `SCNNode` added as a child of the scene's `rootNode`.
3. THE three torus ring nodes SHALL each carry a permanently running `SCNAction.repeatForever(SCNAction.rotateBy(x:y:z:duration:))` action with distinct axis combinations and durations of `6.0`, `4.5`, and `8.0` seconds respectively.
4. THE torus ring nodes SHALL each use an `SCNMaterial` with `emission.contents` set to the `HAKIState.accentColor` of `HAKIStateModel.currentState`, updated via a `SceneKit` render delegate or `Combine` subscription whenever `HAKIStateModel.currentState` changes.
5. THE `JARVISParticleView` SHALL create one `SCNSphere` geometry node with `radius: 0.3` at the scene origin, with an `SCNParticleSystem` attached, configured with `particleSize: 0.05`, `particleColor` matching `HAKIState.accentColor`, and `emissionDuration: 0` (continuous).
6. THE `SCNParticleSystem` `birthRate` SHALL be bound to `HAKIStateModel.audioLevel * HAKIState.particleEmissionRate + HAKIState.particleEmissionRate`, updated every render frame via the `SCNSceneRendererDelegate.renderer(_:updateAtTime:)` callback.
7. THE central `SCNSphere` node `scale` SHALL be set to `SCNVector3(s, s, s)` where `s = 1.0 + Double(HAKIStateModel.audioLevel) * 0.5`, updated every render frame.
8. WHILE `HAKIStateModel.currentState == .idle`, THE `JARVISParticleView` SHALL apply a slow ambient `SCNAction` that scales the central sphere between `0.95` and `1.05` over `2.0` seconds, repeating forever.
9. WHILE `HAKIStateModel.currentState == .thinking`, THE three torus ring rotation durations SHALL be reduced to `1.5`, `1.1`, and `2.0` seconds by replacing the running actions.
10. WHILE `HAKIStateModel.currentState == .error`, THE `JARVISParticleView` SHALL apply a position-jitter `SCNAction` sequence of ±`0.05 pt` translations repeating for `1.0` second before returning to origin.
11. THE `JARVISParticleView` SHALL accept a `@Binding var audioLevel: Float` parameter and SHALL read `HAKIStateModel` from the SwiftUI environment to derive state-dependent visual parameters.

---

### Requirement 6 — Status Bar Menu (`StatusBarMenuView`)

**User Story:** As a user, I want a native menu-bar dropdown that shows the IPC connection status, audio device selection, and voice-mode toggles, so that I can monitor and control HAKI without opening the main window.

#### Acceptance Criteria

1. THE `StatusBarMenuView` SHALL be a SwiftUI `View` rendered inside the `MenuBarExtra` scene with `.menuBarExtraStyle(.window)`.
2. THE `StatusBarMenuView` SHALL display an IPC connection status indicator: a `Circle()` fill of `.green` when `HAKIStateModel.ipcConnected == true` and `.red` when `false`, accompanied by a `Text` label reading `"Core: Connected"` or `"Core: Disconnected"` respectively.
3. THE `StatusBarMenuView` SHALL display the name of the currently selected audio input device using `AVCaptureDevice.default(for: .audio)?.localizedName` or `"No mic"` if unavailable.
4. THE `StatusBarMenuView` SHALL display the name of the currently selected audio output device using `AVAudioSession.sharedInstance().currentRoute.outputs.first?.portName` or `"Default"` if unavailable.
5. THE `StatusBarMenuView` SHALL render a toggle labelled `"Voice Mode"` bound to a `Bool` that, when set to `true`, transitions `HAKIStateModel.currentState` to `.listening` and, when set to `false`, transitions it to `.idle`.
6. THE `StatusBarMenuView` SHALL render a `Button("Open HAKI")` that calls `NSApp.activate(ignoringOtherApps: true)` and brings the `MainWorkspaceView` window to front.
7. THE `StatusBarMenuView` SHALL render a `Button("Quit")` that calls `NSApplication.shared.terminate(nil)`.
8. THE `StatusBarMenuView` SHALL have a fixed width of `300 pt` and a maximum height of `400 pt`.

---

### Requirement 7 — Global Floating Hotkey HUD (`FloatingPanelManager.swift`)

**User Story:** As a user, I want to press `Option + Space` from anywhere in macOS and instantly see a minimal HAKI command input overlay, so that I can interact with HAKI without switching away from my current app.

#### Acceptance Criteria

1. THE `FloatingPanelManager` class SHALL subclass `NSObject` and SHALL manage one `NSPanel` instance created with `styleMask: [.nonactivatingPanel, .borderless]`, `backing: .buffered`, and `defer: false`.
2. THE `NSPanel` SHALL have `level` set to `.floating`, `isOpaque` set to `false`, `backgroundColor` set to `.clear`, and `hasShadow` set to `true`.
3. THE `NSPanel` content view SHALL render a `NSHostingView` containing a SwiftUI `VStack` with a `RoundedRectangle` background using `.ultraThinMaterial`, a `TextField` placeholder `"Ask HAKI…"` auto-focused on show, and a one-line status text label.
4. THE `FloatingPanelManager.registerGlobalHotkey()` method SHALL register a Carbon `EventHotKey` for `Option + Space` (keyCode `49`, modifier `optionKey`) using `RegisterEventHotKey` from `Carbon.framework`.
5. WHEN the `Option + Space` hotkey fires and the `NSPanel` is not visible, THE `FloatingPanelManager` SHALL call `panel.center()` and `panel.orderFront(nil)` to display the panel centred on the main screen.
6. WHEN the `Option + Space` hotkey fires and the `NSPanel` is already visible, THE `FloatingPanelManager` SHALL call `panel.orderOut(nil)` to dismiss the panel.
7. WHEN the `Escape` key is pressed while the `NSPanel` is key, THE `FloatingPanelManager` SHALL dismiss the panel by calling `panel.orderOut(nil)`.
8. WHEN the user presses `Return` in the `TextField` while the panel is visible, THE `FloatingPanelManager` SHALL post the entered text as a `Notification` named `haki.hotkeyCommand` on `NotificationCenter.default` with the key `"commandText"`, then dismiss the panel.
9. THE `NSPanel` SHALL have a fixed size of `480 × 72 pt`.

---

### Requirement 8 — Agent Screen Takeover Overlay (`ScreenOverlayManager.swift`)

**User Story:** As a user, I want a visual indicator drawn around the entire screen perimeter when HAKI enters Autonomous Screen Control mode, so that I always know when HAKI is actively controlling the display.

#### Acceptance Criteria

1. THE `ScreenOverlayManager` class SHALL subclass `NSObject` and SHALL manage one `NSWindow` instance created with `styleMask: .borderless`, `backing: .buffered`, and `defer: false`.
2. THE overlay `NSWindow` SHALL have `level` set to `.screenSaver`, `isOpaque` set to `false`, `backgroundColor` set to `.clear`, `ignoresMouseEvents` set to `true`, and `collectionBehavior` set to `[.canJoinAllSpaces, .fullScreenAuxiliary]`.
3. THE overlay `NSWindow` content view SHALL be an `NSHostingView` containing a SwiftUI `ZStack` with a single `RoundedRectangle(cornerRadius: 16).stroke(Color.orange, lineWidth: 6)` that fills the full frame with zero padding.
4. THE `RoundedRectangle` stroke SHALL animate with a pulsating opacity between `0.6` and `1.0` using `.animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: true)`.
5. WHEN `ScreenOverlayManager.show()` is called, THE `ScreenOverlayManager` SHALL set the overlay `NSWindow` frame to `NSScreen.main?.frame ?? .zero` and call `overlayWindow.orderFront(nil)`.
6. WHEN `ScreenOverlayManager.hide()` is called, THE `ScreenOverlayManager` SHALL call `overlayWindow.orderOut(nil)`.
7. WHEN `ScreenOverlayManager` receives the `haki.agentModeActivated` notification from `NotificationCenter.default`, THE `ScreenOverlayManager` SHALL call `ScreenOverlayManager.show()` on the main thread.
8. WHEN `ScreenOverlayManager` receives the `haki.agentModeDeactivated` notification from `NotificationCenter.default`, THE `ScreenOverlayManager` SHALL call `ScreenOverlayManager.hide()` on the main thread.
9. IF `NSScreen.main` is `nil` at the time `ScreenOverlayManager.show()` is called, THEN THE `ScreenOverlayManager` SHALL log a warning message and return without displaying the overlay window.

---

### Requirement 9 — IPC Integration

**User Story:** As a developer, I want the frontend to connect to the HAKI Core socket and update `HAKIStateModel` based on inbound IPC messages, so that the visual state machine reflects the real operating state of the Core.

#### Acceptance Criteria

1. THE `HAKIApp` SHALL instantiate a `JSONIPCClient` with `socketPath` equal to the path of `~/Library/Application Support/HAKI/haki_core.sock` resolved via `FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first`.
2. WHEN the `JSONIPCClient` successfully completes the socket handshake, THE `HAKIStateModel.ipcConnected` SHALL be set to `true` on the `@MainActor`.
3. WHEN the `JSONIPCClient` loses the socket connection or throws a connection error, THE `HAKIStateModel.ipcConnected` SHALL be set to `false` on the `@MainActor`.
4. WHEN a `ServerMessage.controlEvent` with `eventType == .speakingStarted` is received from the IPC stream, THE `HAKIStateModel.currentState` SHALL be set to `.speaking`.
5. WHEN a `ServerMessage.controlEvent` with `eventType == .speakingStopped` is received from the IPC stream, THE `HAKIStateModel.currentState` SHALL be set to `.idle`.
6. WHEN a `ServerMessage.partialTranscript` is received from the IPC stream, THE `HAKIStateModel.currentState` SHALL be set to `.listening`.
7. WHEN a `ServerMessage.llmToken` with `isLast == true` is received from the IPC stream, THE `HAKIStateModel.currentState` SHALL be set to `.speaking`.
8. THE IPC listener task SHALL be started within a `Task { }` block launched from the `HAKIApp` initialisation path and SHALL not block the main actor.

---

### Requirement 10 — Audio Reactivity Pipeline

**User Story:** As a developer, I want `HAKIStateModel.audioLevel` updated in real time from `AVAudioEngine`, so that the JARVIS HUD and waveform visualiser react to live microphone input.

#### Acceptance Criteria

1. THE `HAKIApp` SHALL install a tap on the `AVAudioEngine.inputNode` via `installTap(onBus:bufferSize:format:block:)` with `bufferSize: 1024`.
2. WHEN the audio tap block fires, THE tap block SHALL compute the RMS amplitude of the buffer's channel data and write the result clamped to `[0.0, 1.0]` into `HAKIStateModel.audioLevel` on the `@MainActor` using `Task { @MainActor in ... }`.
3. WHILE `HAKIStateModel.currentState != .listening`, THE tap block SHALL write `0.0` into `HAKIStateModel.audioLevel` rather than the measured RMS, so that the HUD returns to baseline when HAKI is not actively listening.
4. IF the `AVAudioEngine` fails to start, THEN THE `HAKIApp` SHALL log the error via `print("[HAKIFrontend] AVAudioEngine error: \(error)")` and SHALL set `HAKIStateModel.audioLevel` to `0.0`.
5. WHEN the application enters the background or the audio session is interrupted, THE `HAKIApp` SHALL call `AVAudioEngine.stop()` and reset `HAKIStateModel.audioLevel` to `0.0`.

---

### Requirement 11 — Conversation History Sidebar Scaffold

**User Story:** As a user, I want the sidebar to display date-grouped placeholder conversation entries at launch, so that the layout and interaction model are validated before live file I/O is wired.

#### Acceptance Criteria

1. THE `MainWorkspaceView` SHALL define a `ConversationEntry` struct with fields `id: UUID`, `role: ConversationRole`, `text: String`, and `timestamp: Date`, where `ConversationRole` is an enum with cases `user` and `assistant`.
2. THE `MainWorkspaceView` SHALL initialise an `@State var conversations: [ConversationEntry]` array with at least three mock entries spanning two distinct calendar days.
3. THE sidebar `List` SHALL group `ConversationEntry` items by calendar day using `Calendar.current.dateComponents([.year, .month, .day], from: entry.timestamp)` as the grouping key.
4. WHEN a sidebar entry is tapped, THE `MainWorkspaceView` SHALL set a `@State var selectedEntry: ConversationEntry?` to the tapped entry and scroll the detail timeline to that entry.
5. THE sidebar SHALL display each entry's `text` truncated to 40 characters with a `lineLimit(1)` modifier and the entry's `timestamp` formatted as `"HH:mm"` using `DateFormatter`.

---

### Requirement 12 — Glassmorphism and Visual Design

**User Story:** As a user, I want all HAKI surfaces to use a consistent premium glassmorphism aesthetic that adapts to macOS light and dark mode, so that the app feels native and polished.

#### Acceptance Criteria

1. THE `MainWorkspaceView` background SHALL use `ZStack { Color.clear.background(.ultraThinMaterial) }` applied to the `NavigationSplitView` container.
2. THE message bubble backgrounds in the conversation timeline SHALL use `.background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))`.
3. THE bottom command bar container SHALL use `.background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20))` with a `shadow(color: .black.opacity(0.15), radius: 8, y: 4)` modifier.
4. THE `StatusBarMenuView` background SHALL use `.background(.ultraThinMaterial)` on the root `VStack`.
5. THE floating hotkey `NSPanel` content SwiftUI view SHALL use `.background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))`.
6. THE `MainWorkspaceView` SHALL apply `.preferredColorScheme(nil)` to respect the system appearance and SHALL NOT hardcode a specific colour scheme.
7. ALL `Text` elements displaying user messages SHALL use `Color.primary` and all `Text` elements displaying assistant messages SHALL use the `HAKIState.accentColor` of `HAKIStateModel.currentState`.

---

### Requirement 13 — Accessibility

**User Story:** As a user with accessibility needs, I want all interactive HAKI controls to be accessible via keyboard navigation and VoiceOver, so that the app meets macOS accessibility standards.

#### Acceptance Criteria

1. THE microphone toggle button in the bottom command bar SHALL carry an `.accessibilityLabel("Toggle microphone")` and `.accessibilityHint("Activates or deactivates voice listening mode")` modifier.
2. THE file attachment button SHALL carry an `.accessibilityLabel("Attach file")` modifier.
3. THE `JARVISParticleView` SHALL carry an `.accessibilityLabel("HAKI status: \(HAKIStateModel.currentState)")` modifier and `.accessibilityHidden(false)`.
4. THE sidebar toggle button SHALL carry an `.accessibilityLabel("Toggle conversation history sidebar")` modifier.
5. THE floating hotkey `NSPanel` `TextField` SHALL be focused via `@FocusState` set to `true` immediately when the panel becomes visible, so that keyboard input is captured without a mouse click.
6. THE `ScreenOverlayManager` overlay `NSWindow` SHALL set `ignoresMouseEvents = true` so that the perimeter overlay does not intercept any accessibility or user mouse events.

---

### Requirement 14 — Error Handling and Resilience

**User Story:** As a developer, I want all IPC, audio, and window management errors to be handled gracefully without crashing the frontend, so that the app remains stable across network interruptions and hardware changes.

#### Acceptance Criteria

1. IF the IPC socket file does not exist at connection time, THEN THE `HAKIApp` SHALL set `HAKIStateModel.ipcConnected` to `false` and retry the connection every `5.0` seconds up to `12` attempts before logging a terminal failure message.
2. IF `RegisterEventHotKey` returns a non-zero `OSStatus`, THEN THE `FloatingPanelManager` SHALL log `"[FloatingPanelManager] Failed to register hotkey: \(status)"` and SHALL NOT crash.
3. IF `NSScreen.main` returns `nil` when `ScreenOverlayManager.show()` is called, THEN THE `ScreenOverlayManager` SHALL log `"[ScreenOverlayManager] No main screen available"` and return without calling `overlayWindow.orderFront(nil)`.
4. WHEN `HAKIStateModel.currentState` transitions to `.error`, THE `MainWorkspaceView` SHALL display a non-modal `Text("HAKI encountered an error")` banner at the top of the detail column using a `ZStack` overlay with `.red.opacity(0.15)` background.
5. THE `JARVISParticleView` SceneKit scene setup SHALL be performed inside a `do/catch` block; IF any scene node fails to create, THEN `JARVISParticleView` SHALL fall back to rendering an empty `SCNScene` and SHALL log the error.
