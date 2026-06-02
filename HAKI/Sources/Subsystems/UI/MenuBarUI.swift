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
//
// Full implementation: Phase 0–1.
// Implements: Req 20.1 (macOS native UI), Req 2.4, 9.7

import SwiftUI

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

// MARK: - HAKIMenuContent

/// The content rendered inside the menu-bar popover.
public struct HAKIMenuContent: View {

    // TODO: Phase 1 — inject real view model / environment objects
    @State private var screenAccessEnabled: Bool = true
    @State private var isPrivateConversation: Bool = false

    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("HAKI")
                .font(.headline)

            Divider()

            Toggle("Screen Content Access", isOn: $screenAccessEnabled)
                .help("Enable or disable HAKI's ability to read on-screen content (Req 2.4).")

            Toggle("Private Conversation", isOn: $isPrivateConversation)
                .help("Mark the current conversation as private — HAKI will not learn from it (Req 9.7).")

            Divider()

            Button("Settings…") {
                // TODO: Phase 1 — open settings panel
            }
            .keyboardShortcut(",")

            Button("Quit HAKI") {
                NSApplication.shared.terminate(nil)
            }
            .keyboardShortcut("q")
        }
        .padding()
        .frame(width: 240)
    }

    public init() {}
}
