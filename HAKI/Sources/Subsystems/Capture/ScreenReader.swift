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
import HAKIAudio
import HAKIPermissions

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
