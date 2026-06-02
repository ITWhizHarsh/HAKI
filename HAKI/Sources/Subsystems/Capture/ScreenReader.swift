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
// also scaffolded here.
//
// Full implementation: Phase 3 Task 18.
// Implements: Req 1 (Screen Reading), Req 2.5 (screen-access toggle gate)

import Foundation

// MARK: - CapturedContent

/// Result of a screen-capture attempt.
public enum CapturedContent {
    /// Text was successfully extracted.
    case text(String)
    /// No text could be extracted after all fallback strategies.
    case noContent
    /// The requested named application is not running or was not found (Req 1.9).
    case appUnavailable(String)
}

// MARK: - PlaybackCommand

/// Commands that control read-aloud playback (Req 1.5, 1.8).
public enum PlaybackCommand {
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

// MARK: - ScreenReader (placeholder)

/// Production implementation — Phase 3 Task 18.
public final class ScreenReader: ScreenReaderProtocol, @unchecked Sendable {

    // MARK: - State

    /// Ordered playback command queue with stop > pause > resume priority
    /// within a 200 ms window (Req 1.8).
    private var commandQueue: [PlaybackCommand] = []
    private let commandQueueLock = NSLock()

    // MARK: - ScreenReaderProtocol

    public func captureFocused(appName: String?) async -> CapturedContent {
        // TODO: Phase 3 Task 18.1 — implement layered capture strategy
        //
        // Pseudocode:
        //   if let appName, !isRunning(appName) { return .appUnavailable(appName) }
        //   if let text = axExtract(focused: appName), !text.isEmpty { return .text(text) }
        //   if let text = pdfExtract(focused: appName), !text.isEmpty { return .text(text) }
        //   if let text = ocrCapture(focused: appName), !text.isEmpty { return .text(text) }
        //   return .noContent
        return .noContent
    }

    public func enqueueCommand(_ command: PlaybackCommand) {
        commandQueueLock.lock()
        defer { commandQueueLock.unlock() }
        commandQueue.append(command)
        // Priority coalescing is applied when draining within a 200 ms window (Req 1.8).
    }

    // MARK: - Command priority (Req 1.8)

    /// Drain the command queue and apply stop > pause > resume priority.
    func drainCommands() -> PlaybackCommand? {
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
}
