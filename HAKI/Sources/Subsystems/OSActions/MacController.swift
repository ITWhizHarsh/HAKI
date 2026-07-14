// MacController.swift
// HAKI — OSActions Subsystem
//
// Mac_Controller: the actuator layer that performs agentic control of the
// macOS environment on behalf of the Execution_Engine.
//
// PERMISSION GATE (Req 21.15, 2.2)
// ---------------------------------
// Before executing any control step, `MacController` calls
// `permissionManager.missingPermissions(for: .macControl)`.
// `.macControl` requires both `.accessibility` and `.automation`.
//
// If any permission is missing:
//   • The step is NOT executed.
//   • `ActuatorResult.permissionDenied(message:)` is returned immediately,
//     within 2 seconds (Req 2.2).
//   • The message names:
//       a. Which permission(s) are missing.
//       b. Which actuator / step is blocked.
//       c. System Settings path to grant each missing permission.
//
// Design reference: Mac_Controller component (design.md)
// Implements: Req 21 (Mac Control), Req 21.15, Req 2.2

import Foundation
import HAKIPermissions

// MARK: - MacControllerProtocol

/// Interface consumed by the Execution_Engine to invoke Mac actuator steps.
///
/// Every method returns an `ActuatorResult`.  If the required macOS
/// permissions are not granted, all methods return `.permissionDenied`
/// without executing the underlying action (Req 21.15).
public protocol MacControllerProtocol: Sendable {

    /// Launch or bring to the foreground the application identified by `name`.
    ///
    /// Requires: `.automation` (AppleScript / Apple Events).
    /// Req 21.2, 21.10
    func launchApp(name: String) async -> ActuatorResult

    /// Bring the named application to the foreground (focus) without relaunching.
    ///
    /// Requires: `.automation`.
    /// Req 21.10
    func bringToFront(appName: String) async -> ActuatorResult

    /// Execute an AppleScript source string.
    ///
    /// Requires: `.automation` (Apple Events).
    /// Req 21.3, 21.4, 21.15
    func runAppleScript(source: String) async -> ActuatorResult

    /// Activate an accessibility element identified by `selector` in `appName`.
    ///
    /// Requires: `.accessibility` (AXUIElement).
    /// Req 21.6, 21.12
    func activateElement(appName: String, selector: String) async -> ActuatorResult

    /// Fill a field identified by `selector` in `appName` with `value`.
    ///
    /// Requires: `.accessibility` (AXUIElement).
    /// Req 21.6, 21.12
    func fillField(appName: String, selector: String, value: String) async -> ActuatorResult
}

// MARK: - MacController

/// Production implementation of `MacControllerProtocol`.
///
/// Injects a `PermissionManagerProtocol` at creation so the permission gate
/// is testable without spawning real TCC dialogs.
///
/// Threading: all `async` methods can be called from any actor context.
/// The permission check is synchronous (nonisolated read) so no await is
/// needed for the gate itself — the 2 s requirement (Req 2.2) is met
/// trivially because the gate is a pure in-memory lookup.
///
/// Req 21, 21.15, 2.2
public struct MacController: MacControllerProtocol {

    // MARK: - Dependencies

    private let permissionManager: PermissionManagerProtocol
    private let scriptBridge: AppleScriptBridge

    // MARK: - Init

    /// Creates a `MacController`.
    ///
    /// - Parameter permissionManager: The permission manager to check before
    ///   executing any control step.  Defaults to `PermissionManager.shared`
    ///   when not supplied (convenient for production callers).
    public init(
        permissionManager: PermissionManagerProtocol,
        scriptBridge: AppleScriptBridge = AppleScriptBridge()
    ) {
        self.permissionManager = permissionManager
        self.scriptBridge = scriptBridge
    }

    // MARK: - MacControllerProtocol

    /// Launch or bring to the foreground the named application.
    ///
    /// Performs the permission gate for `.macControl` (needs `.automation`
    /// and `.accessibility`) before running any action.  Req 21.2, 21.15.
    public func launchApp(name: String) async -> ActuatorResult {
        if let denied = permissionGateResult(stepName: "app launch") {
            return denied
        }
        // AppleScript: tell application "<name>" to activate
        let source = "tell application \"\(name)\" to activate"
        return await runScript(source: source, stepName: "launch app '\(name)'")
    }

    /// Bring the named application to the foreground.
    ///
    /// Req 21.10, 21.15
    public func bringToFront(appName: String) async -> ActuatorResult {
        if let denied = permissionGateResult(stepName: "bring app to front") {
            return denied
        }
        let source = "tell application \"\(appName)\" to activate"
        return await runScript(source: source, stepName: "bring '\(appName)' to front")
    }

    /// Execute an arbitrary AppleScript string.
    ///
    /// This is the primary path for AppleScript-based app control
    /// (messaging, calling, etc.).  Req 21.3, 21.4, 21.15.
    public func runAppleScript(source: String) async -> ActuatorResult {
        if let denied = permissionGateResult(stepName: "AppleScript-based app control") {
            return denied
        }
        return await runScript(source: source, stepName: "AppleScript execution")
    }

    /// Activate an accessibility element.
    ///
    /// Req 21.6, 21.12, 21.15
    public func activateElement(appName: String, selector: String) async -> ActuatorResult {
        if let denied = permissionGateResult(stepName: "UI element activation (AX action)") {
            return denied
        }
        // Full AX implementation deferred; stub confirms gate works.
        return .failure(error: "AX element activation not yet implemented for '\(appName)'/'\(selector)'")
    }

    /// Fill an accessibility text field.
    ///
    /// Req 21.6, 21.12, 21.15
    public func fillField(appName: String, selector: String, value: String) async -> ActuatorResult {
        if let denied = permissionGateResult(stepName: "UI field fill (AX action)") {
            return denied
        }
        // Full AX implementation deferred; stub confirms gate works.
        return .failure(error: "AX field fill not yet implemented for '\(appName)'/'\(selector)'")
    }

    // MARK: - Private helpers

    /// Checks whether all permissions required for `.macControl` are granted.
    ///
    /// If any permission is missing, returns `ActuatorResult.permissionDenied`
    /// with a guidance message naming:
    ///   (a) the missing permission(s),
    ///   (b) the blocked step, and
    ///   (c) the System Settings path to grant each missing permission.
    ///
    /// Returns `nil` when all required permissions are present (proceed).
    ///
    /// This check is synchronous — no I/O — so it completes well within
    /// the 2 s budget required by Req 2.2.
    ///
    /// Req 21.15, 2.2
    private func permissionGateResult(stepName: String) -> ActuatorResult? {
        let missing = permissionManager.missingPermissions(for: .macControl)
        guard !missing.isEmpty else {
            return nil   // All permissions present — let the step proceed.
        }

        // Build a clear, actionable guidance message (Req 21.15, 2.2):
        //   a. Which permission(s) are missing.
        //   b. Which step is blocked.
        //   c. How to grant each missing permission.
        let permissionList = missing
            .map { $0.displayName }
            .joined(separator: " and ")

        let instructions = missing
            .map { "• \($0.displayName): \($0.settingsPath)" }
            .joined(separator: "\n")

        let message = """
        \(permissionList) permission\(missing.count > 1 ? "s are" : " is") not granted. \
        \(stepName.capitalizingFirstLetter()) requires \(permissionList).

        To enable Mac Control, grant access in macOS System Settings:
        \(instructions)
        """

        return .permissionDenied(message: message)
    }

    /// Runs an AppleScript string through the bridge and converts the result
    /// to `ActuatorResult`.
    private func runScript(source: String, stepName: String) async -> ActuatorResult {
        do {
            let output = try await scriptBridge.run(source: source)
            return .success(value: output)
        } catch AppleScriptError.permissionDenied {
            // The OS rejected the Apple Event at runtime (e.g. TCC revoked
            // mid-session).  Synthesise the same permissionDenied response
            // so callers get a consistent interface.
            let instructions = HAKIPermission.automation.settingsPath
            return .permissionDenied(
                message: """
                Automation permission was denied at runtime while executing \(stepName).
                • Automation: \(instructions)
                """
            )
        } catch {
            return .failure(error: "\(stepName) failed: \(error.localizedDescription)")
        }
    }
}

// MARK: - String helper

private extension String {
    /// Capitalises only the first character, leaving the rest unchanged.
    func capitalizingFirstLetter() -> String {
        guard let first = first else { return self }
        return String(first).uppercased() + dropFirst()
    }
}
