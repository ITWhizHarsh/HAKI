// UIState.swift
// HAKI — UI Subsystem
//
// Shared observable state for the HAKI UI layer.
//
// `UIState` is the single source of truth for transient UI notifications that
// originate from non-UI subsystems (e.g. the TTS pipeline posting a fallback
// text response when audio playback fails — Req 3.7).
//
// Usage:
//   - Subsystems post events via `NotificationCenter` with the names defined
//     below, OR call the `UIState.shared` methods directly.
//   - SwiftUI views observe `@EnvironmentObject var uiState: UIState`.
//
// Implements: Req 3.7 (TTS failure → on-screen text + notify user)
// Phase 1 Task 7.3

import Foundation
import SwiftUI
import Combine
import HAKIIPC

// MARK: - Notification names

public extension Notification.Name {
    /// Posted when the TTS pipeline fails and the response must be shown as text.
    /// `userInfo` keys: `UIState.Keys.responseText`, `UIState.Keys.turnId`.
    static let ttsFailedShowText = Notification.Name("haki.ttsFailedShowText")

    /// Posted when the Image_Studio delivers a new image (Req 15.1, 15.2, 15.3).
    /// `userInfo` keys: `UIState.Keys.imageResponse`.
    static let imageStudioResponse = Notification.Name("haki.imageStudioResponse")

    /// Posted when Core sends a calendar event proposal (Req 11.1).
    /// `userInfo` keys: `UIState.Keys.calendarProposal`.
    static let calendarProposalReceived = Notification.Name("haki.calendarProposalReceived")

    /// Posted when a reminder fires (Req 12.6).
    /// `userInfo` keys: `UIState.Keys.reminderNotification`.
    static let reminderFired = Notification.Name("haki.reminderFired")

    /// Posted when an automation step progresses (Req 17.5).
    /// `userInfo` keys: `UIState.Keys.automationProgress`.
    static let automationProgressUpdated = Notification.Name("haki.automationProgressUpdated")
    
    /// Posted when the speech-to-text transcript is updated.
    /// `userInfo` keys: `UIState.Keys.transcriptText`.
    static let transcriptUpdated = Notification.Name("haki.transcriptUpdated")
    
    /// Posted when the LLM generates response tokens.
    /// `userInfo` keys: `UIState.Keys.responseText`, `UIState.Keys.isFinal`.
    static let llmResponseUpdated = Notification.Name("haki.llmResponseUpdated")
}

// MARK: - UIState

/// Thread-safe, `@Observable`-compatible shared state for HAKI's transient UI.
///
/// Subsystems that cannot hold a direct reference to SwiftUI views post
/// `NotificationCenter` notifications; `UIState` converts them to
/// `@Published` properties that views can bind to.
@MainActor
public final class UIState: ObservableObject {

    // MARK: - Singleton

    public static let shared = UIState()

    // MARK: - Notification user-info keys

    public enum Keys {
        public static let responseText = "responseText"
        public static let turnId       = "turnId"
        public static let imageResponse = "imageResponse"
        public static let calendarProposal = "calendarProposal"
        public static let reminderNotification = "reminderNotification"
        public static let automationProgress = "automationProgress"
        public static let transcriptText = "transcriptText"
        public static let isFinal = "isFinal"
    }

    // MARK: - Published properties

    /// The live user speech transcript for the floating HUD.
    @Published public var currentTranscript: String = ""
    
    /// The live AI response tokens for the floating HUD.
    @Published public var currentResponse: String = ""

    /// When non-nil, a TTS failure occurred and the UI should display this text
    /// as the assistant's response (Req 3.7).
    @Published public var ttsFailbackText: String? = nil

    /// Human-readable notification message shown alongside the fallback text
    /// (e.g. "Audio playback unavailable.").
    @Published public var ttsFailbackNotice: String? = nil

    /// `true` while TTS audio is actively playing.
    @Published public var isTTSPlaying: Bool = false

    /// Session image history delivered by the Image_Studio (Req 15.1, 15.2, 15.3).
    /// Ordered oldest-first; the UI binds to this to render the image panel.
    @Published public var sessionImages: [HAKIImageResponse] = []

    /// Pending calendar event proposals awaiting user confirmation (Req 11.1).
    /// Each proposal is shown as a card in the Proposals panel.
    @Published public var pendingProposals: [HAKICalendarProposal] = []

    /// Recent in-app reminder notifications (Req 12.6).
    /// Newest first; shown in the Reminders strip.
    @Published public var recentReminders: [HAKIReminderNotification] = []

    /// Current automation run progress events (Req 17.5).
    /// Cleared when a new automation starts; step-by-step events are appended.
    @Published public var automationProgressEvents: [HAKIAutomationProgress] = []

    /// Name of the currently-running automation (nil when idle).
    @Published public var currentAutomationName: String? = nil
    
    /// Timer to automatically hide the floating HUD after inactivity.
    private var hudHideTimer: AnyCancellable?

    // MARK: - Init

    private var cancellables = Set<AnyCancellable>()

    private init() {
        // Listen for TTS-failed notifications posted from background tasks.
        NotificationCenter.default.publisher(for: .ttsFailedShowText)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] notification in
                guard let self else { return }
                let text   = notification.userInfo?[Keys.responseText] as? String ?? ""
                self.ttsFailbackText   = text
                self.ttsFailbackNotice = "Audio playback was unavailable. Showing response as text."
            }
            .store(in: &cancellables)

        // Listen for Image_Studio responses (Req 15.1, 15.4, 15.5).
        NotificationCenter.default.publisher(for: .imageStudioResponse)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] notification in
                guard let self else { return }
                if let response = notification.userInfo?[Keys.imageResponse] as? HAKIImageResponse {
                    self.sessionImages.append(response)
                }
            }
            .store(in: &cancellables)

        // Listen for calendar proposals (Req 11.1).
        NotificationCenter.default.publisher(for: .calendarProposalReceived)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] notification in
                guard let self else { return }
                if let proposal = notification.userInfo?[Keys.calendarProposal] as? HAKICalendarProposal {
                    // Replace an existing proposal with the same ID (status update) or append.
                    if let idx = self.pendingProposals.firstIndex(where: { $0.proposalId == proposal.proposalId }) {
                        self.pendingProposals[idx] = proposal
                    } else {
                        self.pendingProposals.append(proposal)
                    }
                }
            }
            .store(in: &cancellables)

        // Listen for reminder notifications (Req 12.6).
        NotificationCenter.default.publisher(for: .reminderFired)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] notification in
                guard let self else { return }
                if let reminder = notification.userInfo?[Keys.reminderNotification] as? HAKIReminderNotification {
                    // Newest first; cap at 20 recent reminders.
                    self.recentReminders.insert(reminder, at: 0)
                    if self.recentReminders.count > 20 {
                        self.recentReminders = Array(self.recentReminders.prefix(20))
                    }
                }
            }
            .store(in: &cancellables)

        // Listen for automation progress events (Req 17.5).
        NotificationCenter.default.publisher(for: .automationProgressUpdated)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] notification in
                guard let self else { return }
                if let event = notification.userInfo?[Keys.automationProgress] as? HAKIAutomationProgress {
                    // If a new automation started or the name changed, clear previous events.
                    if self.currentAutomationName != event.automationName {
                        self.automationProgressEvents = []
                        self.currentAutomationName = event.automationName
                    }
                    self.automationProgressEvents.append(event)
                    // Clear completed automation after plan_complete.
                    if event.status == "plan_complete" {
                        self.currentAutomationName = nil
                    }
                }
            }
            .store(in: &cancellables)

        // Listen for real-time STT transcripts.
        NotificationCenter.default.publisher(for: .transcriptUpdated)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] notification in
                guard let self else { return }
                if let text = notification.userInfo?[Keys.transcriptText] as? String {
                    self.currentTranscript = text
                    // Clear the old response when user starts speaking
                    if text.count > 0 && self.currentResponse.count > 0 {
                        self.currentResponse = ""
                    }
                    self.resetHUDHideTimer()
                }
            }
            .store(in: &cancellables)

        // Listen for LLM response updates.
        NotificationCenter.default.publisher(for: .llmResponseUpdated)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] notification in
                guard let self else { return }
                if let text = notification.userInfo?[Keys.responseText] as? String {
                    let isFinal = notification.userInfo?[Keys.isFinal] as? Bool ?? false
                    if isFinal {
                        self.currentResponse = text
                    } else {
                        self.currentResponse += text
                    }
                    self.resetHUDHideTimer()
                }
            }
            .store(in: &cancellables)
    }

    private func resetHUDHideTimer() {
        hudHideTimer?.cancel()
        hudHideTimer = Just(())
            .delay(for: .seconds(8), scheduler: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.currentTranscript = ""
                self?.currentResponse = ""
            }
    }

    // MARK: - Public helpers

    /// Post a TTS-failure event from any thread.
    ///
    /// This is the preferred call site inside non-UI code; it keeps the
    /// notification format consistent.
    public nonisolated static func postTTSFailure(responseText: String, turnId: String = "") {
        NotificationCenter.default.post(
            name: .ttsFailedShowText,
            object: nil,
            userInfo: [
                Keys.responseText: responseText,
                Keys.turnId: turnId,
            ]
        )
    }

    /// Deliver an Image_Studio response from any thread (Req 15.1, 15.4, 15.5).
    ///
    /// Call this from the IPC inbound handler whenever a `ServerMessage.imageResponse`
    /// arrives so that the SwiftUI image panel automatically updates.
    public nonisolated static func postImageResponse(_ response: HAKIImageResponse) {
        NotificationCenter.default.post(
            name: .imageStudioResponse,
            object: nil,
            userInfo: [Keys.imageResponse: response]
        )
    }

    /// Deliver a calendar event proposal from any thread (Req 11.1).
    ///
    /// Call this from the IPC inbound handler whenever a `ServerMessage.proposalReceived`
    /// arrives so that the SwiftUI proposals panel automatically updates.
    public nonisolated static func postCalendarProposal(_ proposal: HAKICalendarProposal) {
        NotificationCenter.default.post(
            name: .calendarProposalReceived,
            object: nil,
            userInfo: [Keys.calendarProposal: proposal]
        )
    }

    /// Deliver a reminder notification from any thread (Req 12.6).
    ///
    /// Call this from the IPC inbound handler whenever a `ServerMessage.reminderFired`
    /// arrives so that the SwiftUI reminders strip updates.
    public nonisolated static func postReminder(_ reminder: HAKIReminderNotification) {
        NotificationCenter.default.post(
            name: .reminderFired,
            object: nil,
            userInfo: [Keys.reminderNotification: reminder]
        )
    }

    /// Deliver an automation progress event from any thread (Req 17.5).
    ///
    /// Call this from the IPC inbound handler whenever a `ServerMessage.automationProgress`
    /// arrives so that the SwiftUI automation-progress panel updates.
    public nonisolated static func postAutomationProgress(_ event: HAKIAutomationProgress) {
        NotificationCenter.default.post(
            name: .automationProgressUpdated,
            object: nil,
            userInfo: [Keys.automationProgress: event]
        )
    }

    /// Deliver a live transcript update from any thread.
    public nonisolated static func postTranscriptUpdate(_ text: String) {
        NotificationCenter.default.post(
            name: .transcriptUpdated,
            object: nil,
            userInfo: [Keys.transcriptText: text]
        )
    }

    /// Deliver an LLM token or response block from any thread.
    public nonisolated static func postLLMResponseUpdate(_ text: String, isFinal: Bool = false) {
        NotificationCenter.default.post(
            name: .llmResponseUpdated,
            object: nil,
            userInfo: [
                Keys.responseText: text,
                Keys.isFinal: isFinal
            ]
        )
    }

    /// Dismiss the current TTS fallback message (e.g. after the user
    /// acknowledges it).
    public func dismissTTSFailback() {
        ttsFailbackText   = nil
        ttsFailbackNotice = nil
    }

    /// Clear all session images (e.g. at conversation end or new session start).
    public func clearSessionImages() {
        sessionImages = []
    }

    /// Dismiss a calendar proposal by ID (after user confirms/rejects it).
    public func dismissProposal(_ proposalId: String) {
        pendingProposals.removeAll { $0.proposalId == proposalId }
    }

    /// Clear all recent reminders.
    public func clearReminders() {
        recentReminders = []
    }

    /// Clear automation progress events (e.g. when starting a new session).
    public func clearAutomationProgress() {
        automationProgressEvents = []
        currentAutomationName = nil
    }
}
