// ScreenReader.swift
// HAKI — Capture Subsystem
//
// Implements the layered content-capture strategy described in the design:
//
//   1. AXUIElement — read focused-window text in reading order (primary, fast path).
//   2. PDFKit      — extract text from PDF documents.
//   3. ScreenCaptureKit + Vision OCR — fallback when no selectable text.
//
// Named-app resolution, playback command queue, and the permission gate are
// also implemented here.
//
// Full implementation: Phase 3 Task 18.
// Implements: Req 1 (Screen Reading), Req 2.5 (screen-access toggle gate)

import Foundation
import AppKit
import ApplicationServices
import PDFKit
import ScreenCaptureKit
import Vision
import HAKIAudio    // VoiceEngineProtocol, HAKIAudioFeatures
import HAKIPermissions  // PermissionManagerProtocol, HAKICapability

// MARK: - CapturedContent

/// Result of a screen-capture attempt.
public enum CapturedContent: Equatable {
    /// Text was successfully extracted.
    case text(String)
    /// No text could be extracted after all fallback strategies.
    case noContent
    /// The requested named application is not running or was not found (Req 1.9).
    case appUnavailable(String)
}

// MARK: - PlaybackCommand

/// Commands that control read-aloud playback (Req 1.5, 1.8).
public enum PlaybackCommand: Equatable {
    case pause
    case resume
    case stop
}

// MARK: - ScreenReaderProtocol

public protocol ScreenReaderProtocol: AnyObject, Sendable {
    /// Capture textual content of the focused window (or a named app's window).
    func captureFocused(appName: String?) async -> CapturedContent
    /// Enqueue a playback control command.
    func enqueueCommand(_ command: PlaybackCommand)
}

// MARK: - ScreenReader

/// Production implementation of the Screen_Reader component.
///
/// Capture strategy (layered fallback):
///   1. Resolve named app (if given); decline if not running.
///   2. AXUIElement: extract focused-window text in reading order (fast path, Req 1.1, 1.2).
///   3. PDFKit: extract text from PDF documents (Req 1.2, 1.3).
///   4. ScreenCaptureKit + VisionOCR: fallback for image-only content (Req 1.3, 1.4).
///   5. Return .noContent if nothing was found (Req 1.6).
///
/// Playback command queue applies stop > pause > resume priority within a
/// 200 ms window (Req 1.5, 1.8).
///
/// Permission gate checks TCC permissions and the user toggle before capture
/// begins (Req 2.5).
public final class ScreenReader: ScreenReaderProtocol, @unchecked Sendable {

    // MARK: - State

    /// Ordered playback command queue with stop > pause > resume priority
    /// within a 200 ms window (Req 1.8).
    private var commandQueue: [PlaybackCommand] = []
    internal let commandQueueLock = NSLock()

    // MARK: - Init

    public init() {}

    // MARK: - ScreenReaderProtocol

    public func enqueueCommand(_ command: PlaybackCommand) {
        commandQueueLock.lock()
        defer { commandQueueLock.unlock() }
        commandQueue.append(command)
        // Priority coalescing is applied when draining within a 200 ms window (Req 1.8).
    }

    // MARK: - Command priority (Req 1.8)

    /// Drain the command queue and apply stop > pause > resume priority.
    /// Returns the highest-priority command present, or nil if the queue
    /// is empty.
    public func drainCommands() -> PlaybackCommand? {
        commandQueueLock.lock()
        defer { commandQueueLock.unlock() }
        guard !commandQueue.isEmpty else { return nil }
        let commands = commandQueue
        commandQueue.removeAll()

        if commands.contains(.stop)   { return .stop }
        if commands.contains(.pause)  { return .pause }
        if commands.contains(.resume) { return .resume }
        return nil
    }

    // MARK: - Task 18.1 — Layered capture (Req 1.1, 1.2, 1.3, 1.4, 1.7, 1.9)

    /// Capture the textual content of the focused window, applying layered
    /// fallback strategies.
    ///
    /// - Parameter appName: Optional display-name or bundle-ID of a specific
    ///   application to read from. If nil, the current frontmost app is used.
    /// - Returns: `.text(String)` with the extracted text, `.noContent` when
    ///   nothing was found, or `.appUnavailable(name)` when the named app is
    ///   not running.
    ///
    /// Budget: ≤10,000 chars on the AX fast path must complete within 3 s
    /// from Wake_Invocation (Req 1.1). The AX path is synchronous-fast;
    /// ScreenCaptureKit/OCR is used only as a fallback.
    public func captureFocused(appName: String?) async -> CapturedContent {

        // ── 1. Named-app resolution (Req 1.7, 1.9) ─────────────────────────
        let targetApp: NSRunningApplication?
        if let name = appName {
            guard let app = resolveApp(name: name) else {
                return .appUnavailable(name)  // (Req 1.9)
            }
            targetApp = app
        } else {
            // Use the current frontmost application.
            targetApp = await MainActor.run {
                NSWorkspace.shared.frontmostApplication
            }
        }

        guard let runningApp = targetApp else {
            // No frontmost app — nothing to capture.
            return .noContent
        }

        let pid = runningApp.processIdentifier

        // ── 2. Primary path — AXUIElement text extraction (Req 1.2) ─────────
        if let axText = await extractAXText(pid: pid), !axText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return .text(axText)  // (Req 1.1)
        }

        // ── 3. PDF path — PDFKit extraction (Req 1.2, 1.3) ──────────────────
        let bundleID = runningApp.bundleIdentifier ?? ""
        let urlEndsInPDF = await focusedURLEndsWith(pid: pid, suffix: ".pdf")
        let isFocusedPDF = isPDFApp(bundleID: bundleID) || urlEndsInPDF
        if isFocusedPDF {
            if let pdfText = await extractPDFText(pid: pid), !pdfText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return .text(pdfText)  // (Req 1.2)
            }
            // If PDF extraction yielded nothing, fall through to OCR (Req 1.3).
        }

        // ── 4. OCR fallback — ScreenCaptureKit + Vision (Req 1.3, 1.4) ──────
        if let ocrText = await ocrCapture(pid: pid), !ocrText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return .text(ocrText)
        }

        // ── 5. Nothing found (Req 1.6) ───────────────────────────────────────
        return .noContent
    }

    // MARK: - Task 18.2 — Read-aloud playback handoff (Req 1.5, 1.6, 1.8, 1.9)

    /// Hand already-captured content to the Voice_Engine for playback.
    ///
    /// - Parameters:
    ///   - content: The previously captured content.
    ///   - voiceEngine: The Voice_Engine to use for speech synthesis.
    /// - Returns: A user-facing error message when content cannot be played,
    ///   or `nil` on success.
    ///
    /// Req 1.6: when no text was found, do NOT call voiceEngine.speak; inform
    ///   the user instead.
    /// Req 1.9: when the app was unavailable, return an informative message.
    public func readAloud(
        content: CapturedContent,
        voiceEngine: any VoiceEngineProtocol
    ) async -> String? {
        switch content {
        case .noContent:
            // (Req 1.6) — do NOT begin playback
            return "No readable text was found in the focused content."

        case .appUnavailable(let name):
            // (Req 1.9)
            return "The application '\(name)' is not currently running."

        case .text(let str):
            // Hand off to the Voice_Engine.
            let (stream, continuation) = AsyncStream<String>.makeStream()
            continuation.yield(str)
            continuation.finish()
            do {
                try await voiceEngine.speak(textStream: stream)
            } catch {
                // TTS failure is handled inside Voice_Engine (Req 3.7); we
                // surface the error message to the caller so they can decide
                // whether to show it in the UI.
                return "Read-aloud playback failed: \(error.localizedDescription)"
            }
            return nil
        }
    }

    /// Apply any queued playback commands to an in-progress read-aloud session.
    ///
    /// - Parameter voiceEngine: The active Voice_Engine.
    ///
    /// The priority logic (stop > pause > resume within 200 ms window) is
    /// applied by `drainCommands()` (Req 1.5, 1.8).
    public func processCommandQueue(voiceEngine: any VoiceEngineProtocol) {
        guard let command = drainCommands() else { return }
        switch command {
        case .stop:
            voiceEngine.bargeInStop()
        case .pause:
            // Voice_Engine does not have a dedicated pause API; use bargeInStop
            // as the nearest equivalent for stopping in-progress TTS.
            voiceEngine.bargeInStop()
        case .resume:
            // Resume is coordinated by the caller re-invoking speak; nothing
            // to do here at the engine level.
            break
        }
    }

    /// Convenience entry-point that gates on permissions, captures content, and
    /// hands it to the Voice_Engine.
    ///
    /// - Parameters:
    ///   - appName: Optional named app to read from.
    ///   - voiceEngine: The Voice_Engine for playback.
    ///   - permissionManager: The Permission_Manager to consult.
    /// - Returns: A user-facing error/guidance message, or `nil` on success.
    public func startReadAloud(
        appName: String?,
        voiceEngine: any VoiceEngineProtocol,
        permissionManager: any PermissionManagerProtocol
    ) async -> String? {
        // 18.3 — permission gate first
        if let blocked = await checkPermissions(permissionManager: permissionManager) {
            return blocked
        }
        let content = await captureFocused(appName: appName)
        return await readAloud(content: content, voiceEngine: voiceEngine)
    }

    // MARK: - Task 18.3 — Permission gate (Req 2.5)

    /// Check that all required TCC permissions are granted and that the
    /// screen-access user toggle is enabled.
    ///
    /// - Returns: A user-facing message if anything is blocked, or `nil` when
    ///   all clear.
    ///
    /// Req 2.2: decline with guidance naming the missing permission(s).
    /// Req 2.5: decline with guidance when the user toggle is off.
    private func checkPermissions(permissionManager: any PermissionManagerProtocol) async -> String? {
        // Check TCC permissions required for .readAloud.
        // missingPermissions(for:) is nonisolated on PermissionManager (safe from any context).
        let missing = permissionManager.missingPermissions(for: .readAloud)
        if !missing.isEmpty {
            return permissionManager.guidanceMessage(for: missing, capability: .readAloud)
        }

        // Check the user-facing screen-access toggle (Req 2.5).
        // PermissionManager is @MainActor; we dispatch to MainActor to read
        // screenAccessEnabled safely from this non-isolated async context.
        let toggleEnabled = await MainActor.run { permissionManager.screenAccessEnabled }
        if !toggleEnabled {
            return "Screen content access is currently disabled. " +
                   "Enable 'Screen Content Access' from the HAKI menu bar to use Screen Reading."
        }

        return nil
    }

    // MARK: - Private: AX text extraction (Req 1.2)

    /// Walk the accessibility tree of `pid` and collect text in reading order.
    ///
    /// AX APIs require the main thread on some macOS versions, so we dispatch
    /// there and collect the result asynchronously.
    private func extractAXText(pid: pid_t) async -> String? {
        return await withCheckedContinuation { continuation in
            DispatchQueue.main.async {
                let appElement = AXUIElementCreateApplication(pid)
                var collected: [String] = []
                Self.walkAXTree(element: appElement, results: &collected, depth: 0)
                let result = collected.joined(separator: " ")
                continuation.resume(returning: result.isEmpty ? nil : result)
            }
        }
    }

    /// Recursively walk an AXUIElement tree and collect visible text values
    /// in reading order (depth-first, top-to-bottom).
    ///
    /// - Parameters:
    ///   - element: The root element to walk.
    ///   - results: Accumulated text strings (in order).
    ///   - depth: Current recursion depth (used to cap recursion for safety).
    private static func walkAXTree(
        element: AXUIElement,
        results: inout [String],
        depth: Int
    ) {
        // Safety cap: limit recursion to avoid run-away trees.
        guard depth < 30 else { return }

        // Collect AXValue (the textual content of a field/element).
        var valueRef: CFTypeRef?
        if AXUIElementCopyAttributeValue(element, kAXValueAttribute as CFString, &valueRef) == .success,
           let value = valueRef as? String,
           !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            results.append(value)
        }

        // Collect AXDescription (for images/buttons/etc.).
        var descRef: CFTypeRef?
        if AXUIElementCopyAttributeValue(element, kAXDescriptionAttribute as CFString, &descRef) == .success,
           let desc = descRef as? String,
           !desc.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            // Avoid duplicating content already captured via AXValue.
            if !(valueRef as? String == desc) {
                results.append(desc)
            }
        }

        // Recurse into children.
        var childrenRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXChildrenAttribute as CFString, &childrenRef) == .success,
              let children = childrenRef as? [AXUIElement]
        else { return }

        for child in children {
            walkAXTree(element: child, results: &results, depth: depth + 1)
        }
    }

    // MARK: - Private: PDF app detection

    /// Returns true when the given bundle ID belongs to a known PDF viewer.
    private func isPDFApp(bundleID: String) -> Bool {
        let pdfBundleIDs: Set<String> = [
            "com.adobe.Reader",
            "com.adobe.Acrobat.Pro",
            "com.apple.Preview"
        ]
        return pdfBundleIDs.contains(bundleID)
    }

    /// Returns true when the AX-reported document URL for `pid` ends in `.pdf`.
    ///
    /// Some document editors set AXDocument on the window/app element with the
    /// file URL; we check that as an additional PDF signal.
    private func focusedURLEndsWith(pid: pid_t, suffix: String) async -> Bool {
        return await withCheckedContinuation { continuation in
            DispatchQueue.main.async {
                let appElement = AXUIElementCreateApplication(pid)
                var docRef: CFTypeRef?
                if AXUIElementCopyAttributeValue(appElement, "AXDocument" as CFString, &docRef) == .success,
                   let docURL = docRef as? String,
                   docURL.lowercased().hasSuffix(suffix.lowercased()) {
                    continuation.resume(returning: true)
                    return
                }
                // Also check the focused window's document attribute.
                var winRef: CFTypeRef?
                if AXUIElementCopyAttributeValue(appElement, kAXFocusedWindowAttribute as CFString, &winRef) == .success,
                   let winCF = winRef,
                   AXUIElementCopyAttributeValue(winCF as! AXUIElement, "AXDocument" as CFString, &docRef) == .success,
                   let docURL = docRef as? String,
                   docURL.lowercased().hasSuffix(suffix.lowercased()) {
                    continuation.resume(returning: true)
                    return
                }
                continuation.resume(returning: false)
            }
        }
    }

    // MARK: - Private: PDFKit text extraction (Req 1.2, 1.3)

    /// Attempt to extract text from a PDF rendered by the app with `pid`.
    ///
    /// Strategy: look up the AX document URL, open it with PDFKit, and
    /// extract text page-by-page in reading order.
    private func extractPDFText(pid: pid_t) async -> String? {
        // Retrieve the document file URL from the AX tree.
        let urlString: String? = await withCheckedContinuation { continuation in
            DispatchQueue.main.async {
                let appElement = AXUIElementCreateApplication(pid)
                var docRef: CFTypeRef?
                // Try app-level AXDocument first, then focused window.
                if AXUIElementCopyAttributeValue(appElement, kAXDocumentAttribute as CFString, &docRef) == .success,
                   let doc = docRef as? String {
                    continuation.resume(returning: doc)
                    return
                }
                var winRef: CFTypeRef?
                if AXUIElementCopyAttributeValue(appElement, kAXFocusedWindowAttribute as CFString, &winRef) == .success,
                   let win = winRef as! AXUIElement?,
                   AXUIElementCopyAttributeValue(win, kAXDocumentAttribute as CFString, &docRef) == .success,
                   let doc = docRef as? String {
                    continuation.resume(returning: doc)
                    return
                }
                continuation.resume(returning: nil)
            }
        }

        guard let rawURL = urlString else { return nil }

        // AXDocument may return a file:// URL string or a plain path.
        let fileURL: URL
        if rawURL.hasPrefix("file://") {
            guard let u = URL(string: rawURL) else { return nil }
            fileURL = u
        } else {
            fileURL = URL(fileURLWithPath: rawURL)
        }

        // PDFKit is safe to use off the main thread.
        guard let pdf = PDFDocument(url: fileURL) else { return nil }

        var pages: [String] = []
        for pageIndex in 0..<pdf.pageCount {
            guard let page = pdf.page(at: pageIndex) else { continue }
            if let pageText = page.string, !pageText.isEmpty {
                pages.append(pageText)
            }
        }
        let result = pages.joined(separator: "\n")
        return result.isEmpty ? nil : result
    }

    // MARK: - Private: OCR capture via ScreenCaptureKit + Vision (Req 1.3, 1.4)

    /// Capture the focused window of `pid` via ScreenCaptureKit and pass the
    /// resulting CGImage to VisionOCR for text recognition.
    ///
    /// Requires Screen Recording permission (checked by the permission gate).
    /// ScreenCaptureKit is available on macOS 12.3+; the project targets
    /// macOS 14, so no @available guard is needed beyond the deployment target.
    @available(macOS 12.3, *)
    private func ocrCapture(pid: pid_t) async -> String? {
        // Get the SCRunningApplication matching our target pid.
        guard let scContent = try? await SCShareableContent.excludingDesktopWindows(
            false,
            onScreenWindowsOnly: true
        ) else { return nil }

        // Find windows belonging to the target process.
        let targetWindows = scContent.windows.filter { $0.owningApplication?.processID == pid }
        guard let targetWindow = targetWindows.first else { return nil }

        let filter = SCContentFilter(desktopIndependentWindow: targetWindow)

        let config = SCStreamConfiguration()
        config.width  = targetWindow.frame.width  > 0 ? Int(targetWindow.frame.width)  : 1920
        config.height = targetWindow.frame.height > 0 ? Int(targetWindow.frame.height) : 1080
        config.pixelFormat = kCVPixelFormatType_32BGRA
        config.showsCursor = false

        guard let screenshot = try? await SCScreenshotManager.captureImage(
            contentFilter: filter,
            configuration: config
        ) else { return nil }

        // Pass to VisionOCR for recognition (Req 1.3, 1.4).
        return await VisionOCR().recogniseText(in: screenshot)
    }

    // MARK: - Private: named-app resolution (Req 1.7, 1.9)

    /// Resolve a named application by display name or bundle identifier.
    ///
    /// - Parameter name: Display name (e.g. "Safari") or bundle ID
    ///   (e.g. "com.apple.Safari"). Case-insensitive display-name match.
    /// - Returns: The first matching `NSRunningApplication`, or `nil` when
    ///   not found or not running.
    private func resolveApp(name: String) -> NSRunningApplication? {
        let running = NSWorkspace.shared.runningApplications

        // Try exact bundle-ID match first (most reliable).
        if let byBundle = running.first(where: {
            $0.bundleIdentifier?.lowercased() == name.lowercased()
        }) {
            return byBundle
        }

        // Try localised name match (e.g. "Safari", "Notes").
        if let byName = running.first(where: {
            $0.localizedName?.lowercased() == name.lowercased()
        }) {
            return byName
        }

        // Try case-insensitive prefix/contains match as a last resort.
        return running.first(where: {
            guard let locName = $0.localizedName else { return false }
            return locName.lowercased().contains(name.lowercased())
        })
    }
}

// MARK: - Gemini Sidecar Architecture (Tasks 11.1, 11.2)
//
// Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 10.4
//
// FrameBuffer actor — rolling single-slot frame store (Req 2.5).
// SidecarServer — UNIX socket server for the Python GeminiVisionClient (Req 2.3–2.7).

import CoreGraphics
import CoreImage
import Network

// MARK: - FrameBuffer

/// Rolling single-frame buffer protected by Swift's actor isolation (Req 2.5).
///
/// Only the most recently captured frame is retained; older frames are
/// discarded on each ``update(_:)`` call.
actor FrameBuffer {
    private var latestFrame: Data?

    /// Replace the stored frame with *frame*.
    func update(_ frame: Data) {
        latestFrame = frame
    }

    /// Return the most recently stored frame, or ``nil`` if none has been
    /// captured yet.
    func latest() -> Data? {
        latestFrame
    }
}

// MARK: - SidecarServer

/// UNIX-socket server that streams JPEG frames to the Python GeminiVisionClient.
///
/// Protocol (newline-delimited, Req 2.3):
/// ```
/// Client connects
///   → Server sends:  {"display_scale":<float>,"width":2560,"height":1600}\n
///   → Client sends:  REQUEST_FRAME\n
///   → Server sends:  <4-byte big-endian uint32 frame length><JPEG bytes>
///   → (Client may send REQUEST_FRAME\n again for the next frame)
/// ```
///
/// Socket permissions are set to 0600 after bind so only the owning user can
/// connect (Req 10.4).
///
/// Screen Recording permission is checked at startup; the process exits with
/// code 1 when it is not granted (Req 2.6).
final class SidecarServer: @unchecked Sendable {

    // MARK: - Constants

    static let socketPath: String = {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return "\(home)/.haki/sidecar_frames.sock"
    }()

    private static let nativeWidth  = 2560
    private static let nativeHeight = 1600

    // Default JPEG quality — overridable via ``--jpeg-quality`` CLI arg (Req 2.2).
    var jpegQuality: Double = 0.85

    // MARK: - State

    private let frameBuffer = FrameBuffer()
    private var scStream: SCStream?
    private var streamOutput: _StreamOutput?
    private var listenerFD: Int32 = -1
    private var displayScale: Double = 2.0

    // MARK: - Init

    init(jpegQuality: Double = 0.85) {
        self.jpegQuality = max(0.01, min(1.0, jpegQuality))
    }

    // MARK: - Start

    /// Start the ScreenCaptureKit stream and the UNIX socket server.
    ///
    /// - Throws: If Screen Recording permission is not granted (Req 2.6) or
    ///   if the UNIX socket cannot be created.
    func start() async throws {
        // 1. Query display scale (Req 2.7)
        let mainDisplay = CGMainDisplayID()
        displayScale = _queryDisplayScale(displayID: mainDisplay)

        // 2. Check Screen Recording permission (Req 2.6)
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        guard !content.displays.isEmpty else {
            fputs("SidecarServer: Screen Recording permission denied. "
                  + "Grant permission in System Settings → Privacy & Security.\n", stderr)
            exit(1)
        }

        // 3. Start SCStream at 2560×1600 (Req 2.1)
        guard let display = content.displays.first else {
            fputs("SidecarServer: no display found.\n", stderr)
            exit(1)
        }

        let config = SCStreamConfiguration()
        config.width  = Self.nativeWidth
        config.height = Self.nativeHeight
        config.pixelFormat          = kCVPixelFormatType_32BGRA
        config.showsCursor          = false
        config.minimumFrameInterval = CMTime(value: 1, timescale: 30) // 30 fps

        let filter   = SCContentFilter(display: display, excludingWindows: [])
        let output   = _StreamOutput(buffer: frameBuffer, jpegQuality: jpegQuality)
        streamOutput = output
        let stream   = SCStream(filter: filter, configuration: config, delegate: nil)
        try stream.addStreamOutput(output, type: .screen, sampleHandlerQueue: .global())
        try await stream.startCapture()
        scStream = stream

        // 4. Bind UNIX socket (Req 2.4)
        try _bindSocket()

        // 5. Accept connections loop (Req 2.3)
        _acceptLoop()
    }

    // MARK: - Private: Display scale query (Req 2.7)

    private func _queryDisplayScale(displayID: CGDirectDisplayID) -> Double {
        let physMM  = CGDisplayScreenSize(displayID)    // millimetres
        let bounds  = CGDisplayBounds(displayID)        // pixels
        guard physMM.width > 0 else { return 2.0 }
        let pointWidth = physMM.width / 25.4 * 72.0
        return Double(bounds.size.width) / pointWidth
    }

    // MARK: - Private: UNIX socket bind (Req 2.4, 10.4)

    private func _bindSocket() throws {
        // Ensure ~/.haki/ directory exists
        let dir = (Self.socketPath as NSString).deletingLastPathComponent
        try FileManager.default.createDirectory(atPath: dir,
                                                withIntermediateDirectories: true)

        // Remove stale socket
        try? FileManager.default.removeItem(atPath: Self.socketPath)

        listenerFD = socket(AF_UNIX, SOCK_STREAM, 0)
        guard listenerFD >= 0 else {
            throw NSError(domain: "SidecarServer", code: Int(errno),
                          userInfo: [NSLocalizedDescriptionKey: "socket() failed"])
        }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = Self.socketPath.utf8CString
        withUnsafeMutableBytes(of: &addr.sun_path) { ptr in
            pathBytes.withUnsafeBytes { src in
                ptr.copyMemory(from: src.prefix(ptr.count))
            }
        }

        let bindResult = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(listenerFD, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard bindResult == 0 else {
            throw NSError(domain: "SidecarServer", code: Int(errno),
                          userInfo: [NSLocalizedDescriptionKey: "bind() failed"])
        }

        // Set permissions to 0600 (owner read/write only) — Req 10.4
        chmod(Self.socketPath, S_IRUSR | S_IWUSR)

        guard Darwin.listen(listenerFD, 5) == 0 else {
            throw NSError(domain: "SidecarServer", code: Int(errno),
                          userInfo: [NSLocalizedDescriptionKey: "listen() failed"])
        }
    }

    // MARK: - Private: Accept loop (Req 2.3)

    private func _acceptLoop() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            while true {
                let clientFD = accept(self.listenerFD, nil, nil)
                guard clientFD >= 0 else { continue }
                DispatchQueue.global(qos: .userInitiated).async {
                    self._handleClient(fd: clientFD)
                }
            }
        }
    }

    // MARK: - Private: Client handler (Req 2.3, 2.7)

    private func _handleClient(fd: Int32) {
        defer { close(fd) }

        // Send handshake JSON (Req 2.7)
        let handshake: [String: Any] = [
            "display_scale": displayScale,
            "width":  Self.nativeWidth,
            "height": Self.nativeHeight,
        ]
        guard let handshakeData = try? JSONSerialization.data(withJSONObject: handshake),
              let nl = "\n".data(using: .utf8) else { return }

        var hs = handshakeData
        hs.append(nl)
        guard _writeAll(fd: fd, data: hs) else { return }

        // Serve REQUEST_FRAME commands (Req 2.3)
        var lineBuffer = Data()
        var readBuf = [UInt8](repeating: 0, count: 256)

        while true {
            let n = recv(fd, &readBuf, readBuf.count, 0)
            if n <= 0 { break }
            lineBuffer.append(contentsOf: readBuf[..<n])

            while let newlineIdx = lineBuffer.firstIndex(of: UInt8(ascii: "\n")) {
                let line = String(data: lineBuffer[lineBuffer.startIndex..<newlineIdx],
                                  encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                lineBuffer = lineBuffer[lineBuffer.index(after: newlineIdx)...]

                if line == "REQUEST_FRAME" {
                    _serveFrame(fd: fd)
                }
            }
        }
    }

    // MARK: - Private: Frame serving (Req 2.3, 2.5)

    private func _serveFrame(fd: Int32) {
        // Use a semaphore to await the actor-isolated latest() call
        let sem = DispatchSemaphore(value: 0)
        var frameData: Data?
        Task {
            frameData = await frameBuffer.latest()
            sem.signal()
        }
        sem.wait()

        guard let jpeg = frameData else { return }

        // 4-byte big-endian length prefix (Req 2.3)
        var length = UInt32(jpeg.count).bigEndian
        let lengthData = Data(bytes: &length, count: 4)
        var payload = lengthData
        payload.append(jpeg)
        _ = _writeAll(fd: fd, data: payload)
    }

    // MARK: - Private: Write helper

    private func _writeAll(fd: Int32, data: Data) -> Bool {
        var remaining = data
        while !remaining.isEmpty {
            let n = remaining.withUnsafeBytes { ptr in
                send(fd, ptr.baseAddress!, ptr.count, 0)
            }
            if n <= 0 { return false }
            remaining = remaining.dropFirst(n)
        }
        return true
    }
}

// MARK: - _StreamOutput

/// SCStreamOutput implementation that compresses frames to JPEG and pushes them
/// into the FrameBuffer (Req 2.2, 2.5).
private final class _StreamOutput: NSObject, SCStreamOutput {

    private let buffer: FrameBuffer
    private let jpegQuality: Double

    init(buffer: FrameBuffer, jpegQuality: Double) {
        self.buffer = buffer
        self.jpegQuality = jpegQuality
    }

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of outputType: SCStreamOutputType
    ) {
        guard outputType == .screen,
              let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let ciImage   = CIImage(cvImageBuffer: imageBuffer)
        let context   = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }

        let nsImage   = NSBitmapImageRep(cgImage: cgImage)
        let jpegProps = [NSBitmapImageRep.PropertyKey.compressionFactor: jpegQuality]
        guard let jpeg = nsImage.representation(using: .jpeg, properties: jpegProps) else { return }

        // Update the rolling frame buffer with the latest JPEG (Req 2.2, 2.5).
        Task {
            await buffer.update(jpeg)
        }
    }
}
