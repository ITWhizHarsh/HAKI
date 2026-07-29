// AppDelegate.swift
// HAKI — Swift / SwiftUI Shell
//
// Bootstraps the menu-bar NSStatusItem and manages the lifecycle of the
// HAKI Core child process.

import AppKit
import AVFoundation
import SwiftUI
import HAKIIPC
import HAKIUI
import HAKIAudio
import HAKIPermissions

// Import CoreAudioPlayer for TTS playback
import class HAKIAudio.CoreAudioPlayer

/// Root application delegate.
///
/// Responsibilities:
/// - Create and own the `NSStatusItem` (menu-bar icon/menu).
/// - Spawn the Python Core child process on launch (see `CoreProcessManager`).
/// - Create and own the `JSONIPCClient`; connect it once the Core socket is ready.
/// - Tear down the Core process on termination.
final class AppDelegate: NSObject, NSApplicationDelegate {

    // MARK: - Properties

    /// The menu-bar status item shown in the system menu bar.
    private var statusItem: NSStatusItem?

    /// Manages the lifecycle of the HAKI Core (Python) child process.
    private let coreProcessManager = CoreProcessManager()

    /// The IPC client connected to the Core over a UNIX domain socket.
    /// Retained here so its lifetime matches the app.
    private var ipcClient: JSONIPCClient?

    /// Voice engine — microphone capture, VAD, STT, TTS.
    private var voiceEngine: VoiceEngine?

    /// Permission manager — TCC gates.
    private var permissionManager: PermissionManager?
    
    /// Audio player for TTS responses from Core
    private var audioPlayer: CoreAudioPlayer?

    // MARK: - NSApplicationDelegate

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Prevent a dock icon — this is a menu-bar-only app.
        NSApp.setActivationPolicy(.accessory)

        // Initialize PermissionManager on main actor (it is @MainActor-isolated).
        permissionManager = PermissionManager()

        setupMenuBarItem()
        reportLocalVoiceAvailabilityAtStartup()
        // Open the conversation window immediately so the user can see
        // what HAKI is hearing and saying.
        ConversationWindowController.shared.open()
        setupIPC()
        coreProcessManager.start()
    }

    func applicationWillTerminate(_ notification: Notification) {
        // Disconnect IPC before terminating the Core process.
        let client = ipcClient
        Task { await client?.disconnect() }
        coreProcessManager.stop()
    }

    // MARK: - Private helpers

    /// Inspect provisioned local voice assets on a utility queue. This only
    /// reports actionable availability status; it never downloads/converts a
    /// model or chooses a cloud/legacy fallback while the app is starting.
    private func reportLocalVoiceAvailabilityAtStartup() {
        DispatchQueue.global(qos: .utility).async {
            let availability = VoiceLocalAssetConfiguration().availability()
            if availability.isReady {
                print("[AppDelegate] ✓ Local voice assets are available.")
            } else {
                print("[AppDelegate] Local voice unavailable: \(availability.actionableSummary)")
            }
        }
    }

    private func setupIPC() {
        // Create the IPC client pointing at the same socket the Core will use.
        let client = JSONIPCClient(socketPath: coreProcessManager.socketPath)
        ipcClient = client

        // Wire the CoreProcessManager callback so we connect only once the
        // socket file exists.
        coreProcessManager.onCoreReady = { [weak self] in
            guard let self, let client = self.ipcClient else { return }
            print("[AppDelegate] Core socket is ready, attempting IPC connection...")
            Task {
                do {
                    try await client.connect()
                    print("[AppDelegate] ✓ IPC connected to Core successfully!")
                    
                    print("")
                    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                    print("  ✅ HAKI IS READY — speak to interact!")
                    print("  🎤 Requesting microphone permission...")
                    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                    print("")

                    // Request microphone permission and wait for the actual result.
                    // AVCaptureDevice.requestAccess is the authoritative async call —
                    // it returns true once the user grants, or immediately if already granted.
                    let granted = await AVCaptureDevice.requestAccess(for: .audio)
                    guard granted else {
                        print("[AppDelegate] ✗ Microphone permission denied. Enable HAKI in System Settings → Privacy & Security → Microphone.")
                        return
                    }
                    print("[AppDelegate] ✓ Microphone permission granted.")

                    // Start the IPC listener in its own task FIRST. It plays
                    // TTS audio chunks and routes LLM tokens / proposals /
                    // reminders to the UI. startVoiceEngine below blocks on its
                    // own event loop forever, so the listener MUST run
                    // concurrently or audio playback and UI updates never run.
                    Task { [weak self] in
                        await self?.listenForIPCMessages(client: client)
                    }

                    await self.startVoiceEngine(ipcClient: client)
                } catch {
                    print("[AppDelegate] ✗ IPC error: \(error)")
                }
            }
        }
    }

    /// Create the VoiceEngine and start listening for speech.
    /// Retries once after a short delay if the first attempt fails with
    /// hardwareUnavailable — this handles the window between TCC grant and
    /// AVAudioEngine being ready.
    private func startVoiceEngine(ipcClient: any IPCClientProtocol) async {
        let engine = VoiceEngineFactory.makeLive(ipcClient: ipcClient)
        self.voiceEngine = engine

        for attempt in 1...3 {
            do {
                let events = try engine.listen()
                print("[AppDelegate] 🎤 VoiceEngine started — HAKI is listening! (attempt \(attempt))")
                for await event in events {
                    switch event {
                    case .finalTranscript(let text, _):
                        print("[AppDelegate] 🗣️ Heard: \(text)")
                        UIState.postTranscriptUpdate(text)
                    case .partialTranscript(let text):
                        UIState.postTranscriptUpdate(text)
                    case .bargeIn:
                        // User started talking while HAKI was speaking. Stop our
                        // own playback locally and tell the Core to kill its
                        // audio + cancel the in-flight turn, then keep listening.
                        print("[AppDelegate] ✋ Barge-in detected — stopping HAKI to listen.")
                        engine.bargeInStop()
                        try? await ipcClient.send(
                            .controlEvent(HAKIControlEvent(eventType: .bargeIn, sequenceNum: 0))
                        )
                    default:
                        break
                    }
                }
                return // stream ended cleanly
            } catch {
                print("[AppDelegate] ✗ VoiceEngine failed to start (attempt \(attempt)): \(error)")
                if attempt < 3 {
                    print("[AppDelegate] Retrying in 1s…")
                    try? await Task.sleep(nanoseconds: 1_000_000_000)
                }
            }
        }
        print("[AppDelegate] ✗ VoiceEngine could not start after 3 attempts. Check microphone permission in System Settings → Privacy & Security → Microphone.")
    }

    private func listenForIPCMessages(client: JSONIPCClient) async {
        print("[AppDelegate] 🎧 Starting IPC message listener...")
        
        // Create audio player for TTS responses
        if audioPlayer == nil {
            audioPlayer = CoreAudioPlayer()
            print("[AppDelegate] ✓ CoreAudioPlayer created")
        }
        
        for await message in client.inbound {
            // Handle TTS audio chunks from Python Core
            if case .ttsAudioChunk(let chunk) = message {
                print("[AppDelegate] 🔊 Received TTS chunk from Python: \(chunk.samples.count) bytes")
                audioPlayer?.playChunk(chunk)
            }

            // When the Core starts/stops speaking (its TTS plays via afplay on
            // the Python side), arm/disarm the VAD's barge-in detection so HAKI
            // does not hear its own voice as a new user turn, and so a real
            // interruption is detected as a barge-in.
            if case .controlEvent(let ce) = message {
                switch ce.eventType {
                case .speakingStarted:
                    self.voiceEngine?.notifyTTSStarted()
                case .speakingStopped:
                    self.voiceEngine?.notifyTTSStopped()
                default:
                    break
                }
            }

            // Route image / proposal / reminder / automation-progress messages
            // to the appropriate UIState helpers so the SwiftUI panels update.
            routeIPCServerMessage(message)
        }
    }

    private func setupMenuBarItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)

        if let button = statusItem?.button {
            button.image = NSImage(
                systemSymbolName: "brain.head.profile",
                accessibilityDescription: "HAKI"
            )
            button.toolTip = "HAKI — Personal AI Assistant"
        }

        let menu = buildMenu()
        statusItem?.menu = menu
    }

    private func buildMenu() -> NSMenu {
        let menu = NSMenu()

        menu.addItem(
            withTitle: "HAKI is running",
            action: nil,
            keyEquivalent: ""
        )
        menu.addItem(.separator())

        menu.addItem(
            withTitle: "Toggle Screen Access",
            action: #selector(toggleScreenAccess),
            keyEquivalent: ""
        )
        menu.addItem(
            withTitle: "Privacy: Mark conversation private",
            action: #selector(markPrivate),
            keyEquivalent: ""
        )
        menu.addItem(.separator())

        menu.addItem(
            withTitle: "Settings…",
            action: #selector(openSettings),
            keyEquivalent: ","
        )
        menu.addItem(.separator())

        menu.addItem(
            withTitle: "Quit HAKI",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q"
        )

        return menu
    }

    // MARK: - Menu actions

    /// Toggle the user-facing screen-content-access control (Req 2.4).
    @objc private func toggleScreenAccess() {
        // TODO: wire to PermissionManager.screenAccessEnabled toggle in Phase 0 Task 4
    }

    /// Mark the current conversation as private (Req 9.7).
    @objc private func markPrivate() {
        // TODO: wire to PrivacyState in Phase 0 Task 2
    }

    /// Open the settings panel (Req 20.2).
    @objc private func openSettings() {
        // TODO: open SwiftUI settings panel in Phase 1
    }
}
