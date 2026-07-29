// HAKIApp.swift
// HAKIFrontend — SwiftUI @main application entry point

import SwiftUI
import AVFoundation
import AppKit
import HAKIIPC

// MARK: - HAKIState

enum HAKIState: Equatable, CaseIterable, CustomStringConvertible {
    case idle, listening, thinking, speaking, agent, error

    var description: String {
        switch self {
        case .idle:      return "idle"
        case .listening: return "listening"
        case .thinking:  return "thinking"
        case .speaking:  return "speaking"
        case .agent:     return "agent"
        case .error:     return "error"
        }
    }

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

// MARK: - Notification.Name Extensions

extension Notification.Name {
    static let hakiAgentModeActivated   = Notification.Name("haki.agentModeActivated")
    static let hakiAgentModeDeactivated = Notification.Name("haki.agentModeDeactivated")
    static let hakiHotkeyCommand        = Notification.Name("haki.hotkeyCommand")
}

// MARK: - HAKIStateModel

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

// MARK: - HAKIApp

@main
struct HAKIApp: App {
    private let stateModel = HAKIStateModel()
    private let floatingPanelManager: FloatingPanelManager
    private let screenOverlayManager: ScreenOverlayManager
    private let ipcClient: JSONIPCClient
    private let audioEngine = AVAudioEngine()

    @MainActor
    init() {
        NSApp.setActivationPolicy(.regular)

        // Construct managers
        floatingPanelManager = FloatingPanelManager()
        floatingPanelManager.registerGlobalHotkey()
        screenOverlayManager = ScreenOverlayManager()

        // IPC socket path: ~/Library/Application Support/HAKI/haki_core.sock
        let appSupport = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first!
            .appendingPathComponent("HAKI/haki_core.sock")
        ipcClient = JSONIPCClient(socketPath: appSupport)

        // Start IPC listener and audio tap
        // Capture reference-type members explicitly to avoid capturing mutating self
        let _ipcClient = ipcClient
        let _stateModel = stateModel
        Task { @MainActor in
            await HAKIApp.startIPCListenerStatic(ipcClient: _ipcClient, stateModel: _stateModel)
        }
        setupAudioTap()
    }

    // MARK: - App body

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

    // MARK: - IPC retry loop

    private func startIPCListener() async {
        await HAKIApp.startIPCListenerStatic(ipcClient: ipcClient, stateModel: stateModel)
    }

    @MainActor
    private static func startIPCListenerStatic(ipcClient: JSONIPCClient, stateModel: HAKIStateModel) async {
        for attempt in 1...12 {
            do {
                try await ipcClient.connect()
                stateModel.ipcConnected = true
                for await message in ipcClient.inbound {
                    await handleServerMessageStatic(message, stateModel: stateModel)
                }
                // Stream ended cleanly — connection lost
                stateModel.ipcConnected = false
                return
            } catch {
                stateModel.ipcConnected = false
                print("[HAKIFrontend] IPC attempt \(attempt)/12 failed: \(error)")
                if attempt == 12 {
                    print("[HAKIFrontend] IPC: terminal failure after 12 attempts")
                    return
                }
                try? await Task.sleep(nanoseconds: 5_000_000_000)
            }
        }
    }

    // MARK: - IPC message handler

    private func handleServerMessage(_ message: ServerMessage) async {
        await HAKIApp.handleServerMessageStatic(message, stateModel: stateModel)
    }

    @MainActor
    private static func handleServerMessageStatic(_ message: ServerMessage, stateModel: HAKIStateModel) async {
        switch message {
        case .controlEvent(let ce) where ce.eventType == .speakingStarted:
            stateModel.currentState = .speaking
        case .controlEvent(let ce) where ce.eventType == .speakingStopped:
            stateModel.currentState = .idle
        case .partialTranscript:
            stateModel.currentState = .listening
        case .llmToken(let t) where t.isLast:
            stateModel.currentState = .speaking
        default:
            break
        }
    }

    // MARK: - Audio tap

    private func setupAudioTap() {
        let inputNode = audioEngine.inputNode
        let format = inputNode.inputFormat(forBus: 0)

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak audioEngine] buffer, _ in
            guard let channelData = buffer.floatChannelData else { return }
            let frameLength = Int(buffer.frameLength)
            guard frameLength > 0 else { return }

            // Compute RMS amplitude
            let channel = channelData[0]
            var sum: Float = 0.0
            for i in 0..<frameLength { sum += channel[i] * channel[i] }
            let rms = sqrt(sum / Float(frameLength))
            let clamped = min(max(rms, 0.0), 1.0)

            Task { @MainActor [stateModel] in
                if stateModel.currentState == .listening {
                    stateModel.audioLevel = clamped
                } else {
                    stateModel.audioLevel = 0.0
                }
            }
        }

        do {
            try audioEngine.start()
        } catch {
            print("[HAKIFrontend] AVAudioEngine error: \(error)")
            Task { @MainActor [stateModel] in
                stateModel.audioLevel = 0.0
            }
        }

        // Register for application resign-active / audio interruption to stop the engine
        NotificationCenter.default.addObserver(
            forName: NSApplication.didResignActiveNotification,
            object: nil,
            queue: .main
        ) { [weak audioEngine] _ in
            audioEngine?.stop()
            Task { @MainActor [stateModel] in
                stateModel.audioLevel = 0.0
            }
        }

        #if !os(macOS)
        NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification,
            object: nil,
            queue: .main
        ) { [weak audioEngine] _ in
            audioEngine?.stop()
            Task { @MainActor [stateModel] in
                stateModel.audioLevel = 0.0
            }
        }
        #endif
    }
}

// MARK: - StatusBarMenuView

/// Status bar menu dropdown rendered inside the `MenuBarExtra` scene.
/// Displays IPC connection status, audio device names, voice mode toggle,
/// and quick-action buttons — all driven by `HAKIStateModel`.
struct StatusBarMenuView: View {
    @Environment(HAKIStateModel.self) private var stateModel

    // Audio device names resolved once on appear and cached in @State.
    @State private var audioInputName: String = "No mic"
    @State private var audioOutputName: String = "Default"

    // MARK: - Voice mode binding

    /// Computed `Binding<Bool>` that maps `.listening` ↔ `true` and `.idle` ↔ `false`
    /// against `stateModel.currentState`.
    private var voiceModeBinding: Binding<Bool> {
        Binding<Bool>(
            get: { stateModel.currentState == .listening },
            set: { isOn in
                stateModel.currentState = isOn ? .listening : .idle
            }
        )
    }

    // MARK: - View body

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {

            // IPC connection status row (Req 6.2)
            HStack(spacing: 8) {
                Circle()
                    .fill(stateModel.ipcConnected ? Color.green : Color.red)
                    .frame(width: 10, height: 10)
                Text(stateModel.ipcConnected ? "Core: Connected" : "Core: Disconnected")
                    .font(.subheadline)
            }

            Divider()

            // Audio input device row (Req 6.3)
            HStack(spacing: 8) {
                Image(systemName: "mic.fill")
                    .foregroundStyle(.secondary)
                    .frame(width: 16)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Input")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(audioInputName)
                        .font(.subheadline)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
            }

            // Audio output device row (Req 6.4)
            HStack(spacing: 8) {
                Image(systemName: "speaker.wave.2.fill")
                    .foregroundStyle(.secondary)
                    .frame(width: 16)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Output")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(audioOutputName)
                        .font(.subheadline)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
            }

            Divider()

            // Voice mode toggle (Req 6.5)
            Toggle("Voice Mode", isOn: voiceModeBinding)
                .toggleStyle(.switch)

            Divider()

            // Open HAKI button (Req 6.6)
            Button("Open HAKI") {
                NSApp.activate(ignoringOtherApps: true)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.primary)

            // Quit button (Req 6.7)
            Button("Quit") {
                NSApplication.shared.terminate(nil)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.red)
        }
        .padding()
        .frame(width: 300)                  // fixed width — Req 6.8
        .frame(maxHeight: 400)              // max height — Req 6.8
        .background(.ultraThinMaterial)     // glassmorphism — Req 12.4
        .onAppear {
            resolveAudioDevices()
        }
    }

    // MARK: - Audio device resolution

    /// Resolves the current audio input and output device names.
    /// Falls back to `"No mic"` and `"Default"` respectively when unavailable (Req 6.3, 6.4).
    private func resolveAudioDevices() {
        // Audio input — AVCaptureDevice (available on macOS)
        audioInputName = AVCaptureDevice.default(for: .audio)?.localizedName ?? "No mic"

        // Audio output — AVAudioSession is not available on macOS.
        // Fall back to "Default" as required by Req 6.4.
        #if false
        let outputs = AVAudioSession.sharedInstance().currentRoute.outputs
        audioOutputName = outputs.first?.portName ?? "Default"
        #else
        audioOutputName = "Default"
        #endif
    }
}
