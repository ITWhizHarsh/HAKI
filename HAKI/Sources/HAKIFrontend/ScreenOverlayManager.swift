// ScreenOverlayManager.swift
// HAKIFrontend — NSWindow screen-control overlay manager

import Foundation
import AppKit
import SwiftUI

// MARK: - OverlayStrokeView

/// SwiftUI view that renders an orange pulsating stroke border over the full screen.
/// Satisfies Requirements 8.3 and 8.4.
struct OverlayStrokeView: View {
    @State private var isVisible: Bool = true

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 16)
                .stroke(Color.orange, lineWidth: 6)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .opacity(isVisible ? 1.0 : 0.6)
                .animation(
                    .easeInOut(duration: 0.8).repeatForever(autoreverses: true),
                    value: isVisible
                )
        }
        .onAppear {
            isVisible = true
        }
    }
}

// MARK: - ScreenOverlayManager

/// Manages the full-screen perimeter overlay window.
///
/// Requirements satisfied:
/// - 8.1: Subclasses NSObject, owns one NSWindow with styleMask .borderless, backing .buffered, defer false
/// - 8.2: Window level = .screenSaver, isOpaque = false, backgroundColor = .clear,
///         ignoresMouseEvents = true, collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
/// - 8.3: Content view is NSHostingView wrapping OverlayStrokeView (ZStack + RoundedRectangle stroke)
/// - 8.4: Pulsating opacity animation easeInOut(duration: 0.8).repeatForever(autoreverses: true)
/// - 13.6: ignoresMouseEvents = true — overlay never intercepts accessibility or user mouse events
final class ScreenOverlayManager: NSObject {

    private var overlayWindow: NSWindow!

    override init() {
        super.init()
        setupOverlayWindow()
        observeNotifications()
    }

    // MARK: - Private setup

    private func setupOverlayWindow() {
        // Req 8.1: borderless, buffered, non-deferred window
        overlayWindow = NSWindow(
            contentRect: .zero,
            styleMask: .borderless,
            backing: .buffered,
            defer: false
        )

        // Req 8.2: window appearance and behavior
        overlayWindow.level = .screenSaver
        overlayWindow.isOpaque = false
        overlayWindow.backgroundColor = .clear
        overlayWindow.ignoresMouseEvents = true                          // Req 13.6
        overlayWindow.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]

        // Req 8.3: SwiftUI overlay content
        overlayWindow.contentView = NSHostingView(rootView: OverlayStrokeView())
    }

    // MARK: - Notification observers

    /// Registers for agent-mode notifications and dispatches show/hide on the main thread.
    ///
    /// Requirements satisfied:
    /// - 8.7: .hakiAgentModeActivated → show() on main thread
    /// - 8.8: .hakiAgentModeDeactivated → hide() on main thread
    private func observeNotifications() {
        NotificationCenter.default.addObserver(
            forName: .hakiAgentModeActivated,
            object: nil,
            queue: nil
        ) { [weak self] _ in
            DispatchQueue.main.async { self?.show() }
        }

        NotificationCenter.default.addObserver(
            forName: .hakiAgentModeDeactivated,
            object: nil,
            queue: nil
        ) { [weak self] _ in
            DispatchQueue.main.async { self?.hide() }
        }
    }

    // MARK: - Public interface

    /// Shows the full-screen perimeter overlay on the main screen.
    ///
    /// Requirements satisfied:
    /// - 8.5: Sets overlay NSWindow frame to NSScreen.main?.frame and calls orderFront(nil)
    /// - 8.9 / 14.3: If NSScreen.main is nil, logs warning and returns without displaying overlay
    func show() {
        guard let screen = NSScreen.main else {
            print("[ScreenOverlayManager] No main screen available")
            return
        }
        overlayWindow.setFrame(screen.frame, display: false)
        overlayWindow.orderFront(nil)
    }

    /// Hides the full-screen perimeter overlay.
    ///
    /// Requirements satisfied:
    /// - 8.6: Calls orderOut(nil) to remove overlay from screen
    func hide() {
        overlayWindow.orderOut(nil)
    }
}
