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

// MARK: - Notification names

public extension Notification.Name {
    /// Posted when the TTS pipeline fails and the response must be shown as text.
    /// `userInfo` keys: `UIState.Keys.responseText`, `UIState.Keys.turnId`.
    static let ttsFailedShowText = Notification.Name("haki.ttsFailedShowText")
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
    }

    // MARK: - Published properties

    /// When non-nil, a TTS failure occurred and the UI should display this text
    /// as the assistant's response (Req 3.7).
    @Published public var ttsFailbackText: String? = nil

    /// Human-readable notification message shown alongside the fallback text
    /// (e.g. "Audio playback unavailable.").
    @Published public var ttsFailbackNotice: String? = nil

    /// `true` while TTS audio is actively playing.
    @Published public var isTTSPlaying: Bool = false

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

    /// Dismiss the current TTS fallback message (e.g. after the user
    /// acknowledges it).
    public func dismissTTSFailback() {
        ttsFailbackText   = nil
        ttsFailbackNotice = nil
    }
}
