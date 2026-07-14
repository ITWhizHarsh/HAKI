// MenuBarUI.swift
// HAKI — UI Subsystem
//
// SwiftUI views and view models for the HAKI menu-bar interface.
//
// The UI subsystem owns:
//   • The status bar button and popover/panel.
//   • The chat / image panel (Phase 1+).
//   • Settings panel (Phase 0 Task 2+).
//   • Privacy and screen-access toggles (always accessible, Req 2.4, 9.7).
//   • Event proposal cards (Phase 5, Req 11.1 – 11.7).
//   • In-app reminder strip (Phase 5, Req 12.6).
//   • Automation progress panel (Phase 6, Req 17.5).
//   • Image gallery panel (Phase 6, Req 15.1 – 15.6).
//
// Full implementation: Phase 0–6 (Task 36.1).
// Implements: Req 20.1 (macOS native UI), Req 2.4, 9.7, 11.1, 12.6, 15.1, 17.5

import SwiftUI
import HAKIIPC

// MARK: - HAKIApp

/// The top-level SwiftUI app scene, used when building with the SwiftUI App
/// lifecycle (alternative to the AppKit `AppDelegate` entry-point above).
/// Currently the AppKit delegate is used; this struct is provided for future
/// migration to the SwiftUI App lifecycle.
@available(macOS 14.0, *)
public struct HAKIApp: App {
    public var body: some Scene {
        // Menu Extra is the SwiftUI equivalent of NSStatusItem (macOS 13+).
        MenuBarExtra("HAKI", systemImage: "brain.head.profile") {
            HAKIMenuContent()
        }
        .menuBarExtraStyle(.window)
    }

    public init() {}
}

// MARK: - Panel selection

/// Which tab is currently visible in the HAKI popover.
public enum HAKIPanel: String, CaseIterable {
    case chat        = "Chat"
    case proposals   = "Proposals"
    case reminders   = "Reminders"
    case automation  = "Automation"
    case images      = "Images"
}

// MARK: - HAKIMenuContent

/// The content rendered inside the menu-bar popover.
///
/// Phase 5–6 wiring (Task 36.1):
///   • Proposals tab — shows `ProposalPanel` with confirm/reject callbacks
///   • Reminders tab — shows `ReminderStrip` with recent reminder banners
///   • Automation tab — shows `AutomationProgressPanel` with cancel callback
///   • Images tab — shows `ImageStudioPanel` gallery
///
/// Badge counts on proposal/reminder tabs are driven by `UIState.shared`.
public struct HAKIMenuContent: View {

    // TODO: Phase 1 — inject real view model / environment objects
    @State private var screenAccessEnabled: Bool = true
    @State private var isPrivateConversation: Bool = false
    @State private var selectedPanel: HAKIPanel = .chat
    @ObservedObject private var uiState: UIState

    public init(uiState: UIState) {
        self.uiState = uiState
    }

    @MainActor
    public init() {
        self.uiState = UIState.shared
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // ── Header ──────────────────────────────────────────────
            HStack {
                Text("HAKI")
                    .font(.headline)

                Spacer()

                Toggle("Screen Access", isOn: $screenAccessEnabled)
                    .toggleStyle(.switch)
                    .controlSize(.mini)
                    .help("Enable or disable screen content access (Req 2.4).")

                Toggle("Private", isOn: $isPrivateConversation)
                    .toggleStyle(.switch)
                    .controlSize(.mini)
                    .help("Mark conversation private — HAKI won't learn from it (Req 9.7).")
            }
            .padding(.horizontal)
            .padding(.top, 10)
            .padding(.bottom, 6)

            Divider()

            // ── Panel tab bar ───────────────────────────────────────
            HStack(spacing: 0) {
                ForEach(HAKIPanel.allCases, id: \.self) { panel in
                    panelTabButton(panel)
                }
            }
            .padding(.horizontal, 4)
            .padding(.vertical, 4)

            Divider()

            // ── Active panel ────────────────────────────────────────
            Group {
                switch selectedPanel {
                case .chat:
                    chatPlaceholder

                case .proposals:
                    ProposalPanel(uiState: uiState) { proposalId, action in
                        // TODO: send PROPOSAL_ACTION message back to Core via IPC
                        print("[HAKIMenuContent] Proposal \(proposalId) action: \(action)")
                        uiState.dismissProposal(proposalId)
                    }

                case .reminders:
                    ReminderStrip(uiState: uiState)

                case .automation:
                    AutomationProgressPanel(uiState: uiState) {
                        // TODO: send BARGE_IN control event to Core via IPC to cancel automation
                        print("[HAKIMenuContent] Automation cancel requested")
                    }

                case .images:
                    ImageStudioPanel(uiState: uiState)
                }
            }
            .frame(minHeight: 160)

            Divider()

            // ── Footer ──────────────────────────────────────────────
            HStack {
                Button("Settings…") {
                    // TODO: open settings panel
                }
                .keyboardShortcut(",")
                .font(.caption)

                Spacer()

                Button("Quit HAKI") {
                    NSApplication.shared.terminate(nil)
                }
                .keyboardShortcut("q")
                .font(.caption)
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
        }
        .frame(width: 320)
    }

    // MARK: Private

    @ViewBuilder
    private func panelTabButton(_ panel: HAKIPanel) -> some View {
        Button {
            selectedPanel = panel
        } label: {
            ZStack(alignment: .topTrailing) {
                VStack(spacing: 2) {
                    Image(systemName: panelIcon(panel))
                        .font(.system(size: 14))
                    Text(panel.rawValue)
                        .font(.system(size: 9))
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 5)
                .background(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(selectedPanel == panel ? Color.accentColor.opacity(0.15) : Color.clear)
                )

                // Badge for pending counts
                if let count = panelBadgeCount(panel), count > 0 {
                    Text("\(min(count, 9))")
                        .font(.system(size: 8, weight: .bold))
                        .foregroundColor(.white)
                        .frame(width: 14, height: 14)
                        .background(Color.red)
                        .clipShape(Circle())
                        .offset(x: -2, y: 2)
                }
            }
        }
        .buttonStyle(.plain)
        .foregroundColor(selectedPanel == panel ? .accentColor : .secondary)
    }

    private func panelIcon(_ panel: HAKIPanel) -> String {
        switch panel {
        case .chat:       return "bubble.left.and.bubble.right"
        case .proposals:  return "calendar.badge.plus"
        case .reminders:  return "bell.badge"
        case .automation: return "gearshape.2"
        case .images:     return "photo.on.rectangle.angled"
        }
    }

    private func panelBadgeCount(_ panel: HAKIPanel) -> Int? {
        switch panel {
        case .proposals:  return uiState.pendingProposals.count
        case .reminders:  return uiState.recentReminders.count
        case .automation: return uiState.currentAutomationName != nil ? 1 : nil
        default:          return nil
        }
    }

    private var chatPlaceholder: some View {
        VStack(spacing: 8) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.largeTitle)
                .foregroundColor(.secondary)
            Text("Chat panel")
                .foregroundColor(.secondary)
            Text("Voice interactions appear here.")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

// MARK: - IPC inbound handler helper

/// Routes an inbound `ServerMessage` to the appropriate `UIState` helper.
///
/// Call this from the IPC stream observer (e.g. in `AppDelegate` or a
/// dedicated `IPCCoordinator`) for each message received from the Core.
///
/// Covers the Phase 5–6 messages wired by Task 36.1:
///   `.imageResponse`      → `UIState.postImageResponse(_:)`
///   `.proposalReceived`   → `UIState.postCalendarProposal(_:)`
///   `.reminderFired`      → `UIState.postReminder(_:)`
///   `.automationProgress` → `UIState.postAutomationProgress(_:)`
///
/// Design: Intent Routing, The Orchestrator.
/// Requirements: 6.1, 11.1, 12.6, 15.1, 17.5.
public func routeIPCServerMessage(_ message: ServerMessage) {
    switch message {
    case .imageResponse(let response):
        UIState.postImageResponse(response)

    case .proposalReceived(let proposal):
        UIState.postCalendarProposal(proposal)

    case .reminderFired(let reminder):
        UIState.postReminder(reminder)

    case .automationProgress(let event):
        UIState.postAutomationProgress(event)

    case .partialTranscript(let transcript):
        UIState.postTranscriptUpdate(transcript.text)

    case .llmToken(let token):
        UIState.postLLMResponseUpdate(token.text, isFinal: token.isLast)

    case .ttsAudioChunk, .controlEvent, .error:
        // Handled by the voice pipeline / audio layer; not routed to UIState here.
        break
    }
}
