// AppDelegate.swift
// HAKI — Swift / SwiftUI Shell
//
// Bootstraps the menu-bar NSStatusItem and manages the lifecycle of the
// HAKI Core child process.

import AppKit
import SwiftUI
import HAKIIPC

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

    // MARK: - NSApplicationDelegate

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Prevent a dock icon — this is a menu-bar-only app.
        NSApp.setActivationPolicy(.accessory)

        setupMenuBarItem()
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

    private func setupIPC() {
        // Create the IPC client pointing at the same socket the Core will use.
        let client = JSONIPCClient(socketPath: coreProcessManager.socketPath)
        ipcClient = client

        // Wire the CoreProcessManager callback so we connect only once the
        // socket file exists.
        coreProcessManager.onCoreReady = { [weak self] in
            guard let self, let client = self.ipcClient else { return }
            Task {
                do {
                    try await client.connect()
                    print("[AppDelegate] IPC connected to Core.")
                } catch {
                    print("[AppDelegate] IPC connect failed: \(error).")
                }
            }
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
