// AppleScriptBridge.swift
// HAKI — OSActions Subsystem
//
// Thin wrapper around `NSAppleScript` for running ad-hoc AppleScript
// commands as part of Mac_Controller automation steps.
//
// The bridge is used for:
//   • Launching / bringing apps to the foreground (Req 21.2, 21.10)
//   • Sending messages via scriptable apps (Req 21.3, 21.4)
//
// Full implementation: Phase 4 Task 24.1.
// Implements: Req 21 (Mac Control)

import Foundation

// MARK: - AppleScriptBridge

public struct AppleScriptBridge: Sendable {

    public init() {}

    // MARK: - Public API

    /// Execute an AppleScript source string.
    ///
    /// - Parameter source: The AppleScript source code to execute.
    /// - Returns: The string value of the result descriptor, if any.
    /// - Throws: `AppleScriptError` if the script fails.
    public func run(source: String) async throws -> String? {
        // AppleScript execution must occur on the main thread.
        return try await MainActor.run {
            var error: NSDictionary?
            let script = NSAppleScript(source: source)
            guard let descriptor = script?.executeAndReturnError(&error) else {
                let message = (error?[NSAppleScript.errorMessage] as? String) ?? "Unknown error"
                throw AppleScriptError.executionFailed(message)
            }
            return descriptor.stringValue
        }
    }
}

// MARK: - AppleScriptError

public enum AppleScriptError: Error {
    case executionFailed(String)
    case permissionDenied
}
