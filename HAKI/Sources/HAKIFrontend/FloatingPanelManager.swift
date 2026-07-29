// FloatingPanelManager.swift
// HAKIFrontend — NSPanel global hotkey HUD manager

import AppKit
import Carbon
import SwiftUI

// MARK: - HotkeyPanelState

/// Observable state object driving the HotkeyPanelViewBound.
final class HotkeyPanelState: ObservableObject {
    @Published var commandText: String = ""
    @Published var statusText: String = "Ready"
}

// MARK: - HotkeyPanelViewBound

/// SwiftUI view that owns the @ObservedObject and forwards submit/dismiss callbacks.
struct HotkeyPanelViewBound: View {
    @ObservedObject var state: HotkeyPanelState
    var onSubmit: () -> Void
    var onDismiss: () -> Void

    @FocusState private var isFocused: Bool

    var body: some View {
        VStack(spacing: 6) {
            TextField("Ask HAKI…", text: $state.commandText)
                .textFieldStyle(.plain)
                .font(.system(size: 16, weight: .regular))
                .focused($isFocused)
                .onSubmit {
                    let text = state.commandText
                    NotificationCenter.default.post(
                        name: .hakiHotkeyCommand,
                        object: nil,
                        userInfo: ["commandText": text]
                    )
                    state.commandText = ""
                    onSubmit()
                }

            Text(state.statusText)
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .frame(width: 480, height: 72)
        // Req 12.5
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .onAppear {
            // Req 13.5: auto-focus TextField when panel appears
            isFocused = true
        }
        // Req 7.7: Escape key dismisses panel (macOS 14+)
        .onKeyPress(.escape) {
            onDismiss()
            return .handled
        }
        .focusable()
    }
}

// MARK: - FloatingPanelManager

/// Manages the floating NSPanel HUD triggered by the global hotkey.
final class FloatingPanelManager: NSObject {

    // MARK: - Properties

    let panel: NSPanel

    /// Observable state for the hosted SwiftUI panel view.
    private let panelState = HotkeyPanelState()

    /// Carbon hotkey reference (Req 7.4)
    private var hotKeyRef: EventHotKeyRef?

    /// Carbon event handler reference (Req 7.4)
    private var eventHandler: EventHandlerRef?

    // MARK: - Init

    @MainActor
    override init() {
        // Req 7.1: nonactivatingPanel + borderless, buffered, defer false
        panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 480, height: 72),
            styleMask: [.nonactivatingPanel, .borderless],
            backing: .buffered,
            defer: false
        )

        super.init()

        configurePanel()
        configureContentView()
    }

    // MARK: - Panel configuration

    private func configurePanel() {
        // Req 7.2
        panel.level = .floating
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        // Req 7.9: fixed 480×72
        panel.setContentSize(NSSize(width: 480, height: 72))
    }

    private func configureContentView() {
        let panel = self.panel

        let dismissAction: () -> Void = {
            panel.orderOut(nil)
        }

        let submitAction: () -> Void = {
            panel.orderOut(nil)
        }

        let rootView = HotkeyPanelViewBound(
            state: panelState,
            onSubmit: submitAction,
            onDismiss: dismissAction
        )

        // Req 7.3: NSHostingView wrapping the SwiftUI HUD view
        let hv = NSHostingView(rootView: rootView)
        hv.frame = NSRect(x: 0, y: 0, width: 480, height: 72)
        panel.contentView = hv
    }

    // MARK: - Stubs (implemented in tasks 9.2 and 9.3)

    /// Registers the Option + Space global hotkey.
    /// Req 7.4: Carbon RegisterEventHotKey for keyCode 49 (Space) + optionKey.
    /// Req 14.2: Logs on failure without crashing.
    func registerGlobalHotkey() {
        // Build the FourCharCode signature "HKFP"
        let h: UInt32 = UInt32(Character("H").asciiValue!) << 24
        let k: UInt32 = UInt32(Character("K").asciiValue!) << 16
        let f: UInt32 = UInt32(Character("F").asciiValue!) << 8
        let p: UInt32 = UInt32(Character("P").asciiValue!)
        let sig: FourCharCode = h | k | f | p

        let hotKeyID = EventHotKeyID(signature: sig, id: 1)
        var ref: EventHotKeyRef?

        // keyCode 49 = Space, optionKey modifier
        let status = RegisterEventHotKey(
            49,
            UInt32(optionKey),
            hotKeyID,
            GetApplicationEventTarget(),
            0,
            &ref
        )

        // Req 14.2: graceful failure
        guard status == noErr else {
            print("[FloatingPanelManager] Failed to register hotkey: \(status)")
            return
        }

        hotKeyRef = ref

        // Install Carbon event handler via a C-compatible free function bridged
        // through userData carrying an unretained reference to self.
        var eventSpec = EventTypeSpec(
            eventClass: OSType(kEventClassKeyboard),
            eventKind:  UInt32(kEventHotKeyPressed)
        )

        // Bridge self into the C callback via userData.
        let userData = Unmanaged.passUnretained(self).toOpaque()

        // C-compatible callback: retrieves the FloatingPanelManager and calls toggle().
        let callback: EventHandlerUPP = { _, _, userData -> OSStatus in
            guard let ptr = userData else { return OSStatus(eventNotHandledErr) }
            let manager = Unmanaged<FloatingPanelManager>.fromOpaque(ptr)
                .takeUnretainedValue()
            DispatchQueue.main.async { manager.toggle() }
            return noErr
        }

        var handlerRef: EventHandlerRef?
        InstallEventHandler(
            GetApplicationEventTarget(),
            callback,
            1,
            &eventSpec,
            userData,
            &handlerRef
        )
        eventHandler = handlerRef
    }

    /// Toggles the floating panel visibility.
    /// Req 7.5: If not visible, centre and show panel; set focus.
    /// Req 7.6: If already visible, dismiss panel.
    func toggle() {
        if panel.isVisible {
            // Req 7.6: panel is visible — dismiss it
            panel.orderOut(nil)
        } else {
            // Req 7.5: panel is hidden — centre and present it
            panel.center()
            panel.orderFront(nil)
            // Req 13.5: @FocusState is set to true via onAppear in HotkeyPanelViewBound,
            // which fires each time the panel is ordered front.
        }
    }
}
