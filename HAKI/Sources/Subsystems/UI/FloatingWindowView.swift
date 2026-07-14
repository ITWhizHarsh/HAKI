// ConversationWindow.swift
// HAKI — UI Subsystem
//
// A simple, always-visible conversation window showing what HAKI
// is hearing and what it is responding — like the Gemini desktop app.
// Uses plain AppKit (NSWindow + NSTextView) so it works reliably
// without any SwiftUI / Combine binding complexity.

import AppKit
import Foundation

// MARK: - ConversationWindowController

/// Opens and manages the HAKI conversation window.
/// Call `open()` once from AppDelegate at launch.
public final class ConversationWindowController: NSObject {

    // MARK: - Singleton
    public static let shared = ConversationWindowController()

    // MARK: - UI
    private var window: NSWindow?
    private weak var youLabel: NSTextField?
    private weak var youText: NSTextView?
    private weak var hakiLabel: NSTextField?
    private weak var hakiText: NSTextView?
    private weak var statusDot: NSTextField?

    // MARK: - Notification observers
    private var observers: [NSObjectProtocol] = []

    // MARK: - Public API

    /// Create and show the conversation window. Call once on launch.
    @MainActor
    public func open() {
        let win = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 440, height: 340),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        win.title = "HAKI"
        win.minSize = NSSize(width: 340, height: 240)
        win.level = .floating               // always on top
        win.isReleasedWhenClosed = false    // keep alive when user closes
        win.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]

        // Build content
        let content = buildContent()
        win.contentView = content

        // Center on screen
        win.center()

        win.makeKeyAndOrderFront(nil)
        self.window = win

        // Wire up notifications
        subscribeToNotifications()
    }

    // MARK: - Private: build pure-AppKit layout

    @MainActor
    private func buildContent() -> NSView {
        let root = NSView()
        root.wantsLayer = true
        root.layer?.backgroundColor = NSColor.windowBackgroundColor.cgColor

        // ── Status row ─────────────────────────────────────────────
        let dot = NSTextField(labelWithString: "● Listening…")
        dot.textColor = .systemGreen
        dot.font = .systemFont(ofSize: 11, weight: .medium)
        dot.translatesAutoresizingMaskIntoConstraints = false
        root.addSubview(dot)
        self.statusDot = dot

        // ── You section ────────────────────────────────────────────
        let youLabel = sectionLabel("🎤  You")
        let youScroll = makeScrollableTextView(placeholder: "Waiting for speech…", fontSize: 14)
        let youTV = (youScroll.documentView as! NSTextView)
        youTV.textColor = .labelColor
        self.youLabel = youLabel
        self.youText = youTV
        root.addSubview(youLabel)
        root.addSubview(youScroll)

        // ── Divider ────────────────────────────────────────────────
        let divider = NSBox()
        divider.boxType = .separator
        divider.translatesAutoresizingMaskIntoConstraints = false
        root.addSubview(divider)

        // ── HAKI section ───────────────────────────────────────────
        let hakiLabel = sectionLabel("🧠  HAKI")
        let hakiScroll = makeScrollableTextView(placeholder: "Response will appear here…", fontSize: 14)
        let hakiTV = (hakiScroll.documentView as! NSTextView)
        hakiTV.textColor = .systemPurple
        self.hakiLabel = hakiLabel
        self.hakiText = hakiTV
        root.addSubview(hakiLabel)
        root.addSubview(hakiScroll)

        // ── Layout ─────────────────────────────────────────────────
        youLabel.translatesAutoresizingMaskIntoConstraints = false
        youScroll.translatesAutoresizingMaskIntoConstraints = false
        hakiLabel.translatesAutoresizingMaskIntoConstraints = false
        hakiScroll.translatesAutoresizingMaskIntoConstraints = false

        let pad: CGFloat = 14
        NSLayoutConstraint.activate([
            // Status dot at top
            dot.topAnchor.constraint(equalTo: root.topAnchor, constant: pad),
            dot.leadingAnchor.constraint(equalTo: root.leadingAnchor, constant: pad),

            // You label
            youLabel.topAnchor.constraint(equalTo: dot.bottomAnchor, constant: 10),
            youLabel.leadingAnchor.constraint(equalTo: root.leadingAnchor, constant: pad),
            youLabel.trailingAnchor.constraint(equalTo: root.trailingAnchor, constant: -pad),

            // You text scroll
            youScroll.topAnchor.constraint(equalTo: youLabel.bottomAnchor, constant: 6),
            youScroll.leadingAnchor.constraint(equalTo: root.leadingAnchor, constant: pad),
            youScroll.trailingAnchor.constraint(equalTo: root.trailingAnchor, constant: -pad),
            youScroll.heightAnchor.constraint(equalTo: root.heightAnchor, multiplier: 0.3),

            // Divider
            divider.topAnchor.constraint(equalTo: youScroll.bottomAnchor, constant: 10),
            divider.leadingAnchor.constraint(equalTo: root.leadingAnchor, constant: pad),
            divider.trailingAnchor.constraint(equalTo: root.trailingAnchor, constant: -pad),

            // HAKI label
            hakiLabel.topAnchor.constraint(equalTo: divider.bottomAnchor, constant: 10),
            hakiLabel.leadingAnchor.constraint(equalTo: root.leadingAnchor, constant: pad),
            hakiLabel.trailingAnchor.constraint(equalTo: root.trailingAnchor, constant: -pad),

            // HAKI text scroll (fills remaining space)
            hakiScroll.topAnchor.constraint(equalTo: hakiLabel.bottomAnchor, constant: 6),
            hakiScroll.leadingAnchor.constraint(equalTo: root.leadingAnchor, constant: pad),
            hakiScroll.trailingAnchor.constraint(equalTo: root.trailingAnchor, constant: -pad),
            hakiScroll.bottomAnchor.constraint(equalTo: root.bottomAnchor, constant: -pad),
        ])

        return root
    }

    // MARK: - Helpers

    private func sectionLabel(_ title: String) -> NSTextField {
        let lbl = NSTextField(labelWithString: title)
        lbl.font = .systemFont(ofSize: 12, weight: .semibold)
        lbl.textColor = .secondaryLabelColor
        return lbl
    }

    private func makeScrollableTextView(placeholder: String, fontSize: CGFloat) -> NSScrollView {
        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.hasHorizontalScroller = false
        scroll.autohidesScrollers = true
        scroll.borderType = .noBorder
        scroll.drawsBackground = false

        let tv = NSTextView()
        tv.isEditable = false
        tv.isSelectable = true
        tv.backgroundColor = NSColor.controlBackgroundColor.withAlphaComponent(0.5)
        tv.textContainerInset = NSSize(width: 8, height: 8)
        tv.font = .systemFont(ofSize: fontSize)
        tv.textContainer?.widthTracksTextView = true
        tv.textContainer?.lineBreakMode = .byWordWrapping
        tv.isVerticallyResizable = true
        tv.isHorizontallyResizable = false
        tv.autoresizingMask = [.width]
        tv.wantsLayer = true
        tv.layer?.cornerRadius = 6
        tv.string = placeholder

        scroll.documentView = tv
        return scroll
    }

    // MARK: - Notification wiring

    private func subscribeToNotifications() {
        let nc = NotificationCenter.default

        let t1 = nc.addObserver(
            forName: Notification.Name("haki.transcriptUpdated"),
            object: nil,
            queue: nil   // we hop to main manually below
        ) { [weak self] note in
            guard let self,
                  let text = note.userInfo?["transcriptText"] as? String else { return }
            DispatchQueue.main.async { self.updateYouText(text) }
        }

        let t2 = nc.addObserver(
            forName: Notification.Name("haki.llmResponseUpdated"),
            object: nil,
            queue: nil
        ) { [weak self] note in
            guard let self,
                  let text = note.userInfo?["responseText"] as? String else { return }
            let isFinal = note.userInfo?["isFinal"] as? Bool ?? false
            DispatchQueue.main.async { self.updateHAKIText(text, isFinal: isFinal) }
        }

        observers = [t1, t2]
    }

    // MARK: - Text update helpers (always called on main thread via DispatchQueue.main.async)

    private var responseAccumulator = ""

    private func updateYouText(_ text: String) {
        guard let tv = youText else { return }
        if text.isEmpty { return }
        tv.string = text
        tv.textColor = .labelColor
        if let hakiTV = hakiText, !tv.string.isEmpty {
            hakiTV.string = "Thinking…"
            hakiTV.textColor = .tertiaryLabelColor
        }
        statusDot?.stringValue = "● Hearing you…"
        statusDot?.textColor = .systemOrange
        bringToFront()
    }

    private func updateHAKIText(_ text: String, isFinal: Bool) {
        guard let tv = hakiText else { return }
        if isFinal {
            responseAccumulator = text
        } else {
            responseAccumulator += text
        }
        tv.string = responseAccumulator
        tv.textColor = .systemPurple
        statusDot?.stringValue = "● Responding…"
        statusDot?.textColor = .systemBlue
        tv.scrollToEndOfDocument(nil)
        bringToFront()
    }

    private func bringToFront() {
        if let win = window, !win.isVisible {
            win.makeKeyAndOrderFront(nil)
        }
    }

    deinit {
        observers.forEach { NotificationCenter.default.removeObserver($0) }
    }
}

