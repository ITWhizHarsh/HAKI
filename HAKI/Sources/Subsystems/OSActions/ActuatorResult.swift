// ActuatorResult.swift
// HAKI — OSActions Subsystem
//
// Represents the outcome of a single Mac_Controller actuator invocation.
//
// A control step may succeed, fail for a runtime reason, or be blocked
// because a required macOS permission (Accessibility / Automation) is not
// granted.  The `.permissionDenied` case carries a human-readable message
// that names which permission is missing, which capability/step is blocked,
// and how to grant the missing permission — satisfying Req 2.2 and 21.15.
//
// Design reference: Mac_Controller component (design.md — Req 21.15)
// Implements: Req 21.15, 2.2

import Foundation

// MARK: - ActuatorResult

/// The result returned by every Mac_Controller actuator method.
///
/// - `.success(value:)`:        The actuator step completed without error.
///   `value` is the optional string result from the operation (e.g.
///   the return value of an AppleScript execution).
///
/// - `.failure(error:)`:        The actuator step encountered a runtime
///   error after permissions were confirmed to be present.
///
/// - `.permissionDenied(message:)`:  The step was NOT executed because at
///   least one required macOS permission (`Accessibility` and/or
///   `Automation`) is not granted.  `message` names:
///     a. Which permission(s) are missing.
///     b. Which capability/step is blocked.
///     c. How to grant the missing permission(s) in macOS System Settings.
///   This result is returned within 2 seconds of the attempt (Req 2.2).
///
/// Requirements: 21.15, 2.2
public enum ActuatorResult: Sendable, Equatable {

    /// Actuator step completed successfully.
    ///
    /// - Parameter value: Optional string output from the operation.
    case success(value: String? = nil)

    /// Actuator step failed for a runtime reason unrelated to permissions.
    ///
    /// - Parameter error: Human-readable description of the failure.
    case failure(error: String)

    /// Actuator step was blocked because a required macOS permission is
    /// not granted.  The step was NOT executed.
    ///
    /// - Parameter message: Human-readable guidance describing:
    ///   (a) which permission(s) are missing,
    ///   (b) which capability / step is blocked, and
    ///   (c) how to grant the missing permission(s).
    ///
    ///   Req 21.15, 2.2
    case permissionDenied(message: String)

    // MARK: - Convenience

    /// Returns `true` when the result is `.success`.
    public var isSuccess: Bool {
        if case .success = self { return true }
        return false
    }

    /// Returns `true` when the result is `.permissionDenied`.
    public var isPermissionDenied: Bool {
        if case .permissionDenied = self { return true }
        return false
    }

    /// Returns the message string for `.permissionDenied`, or `nil` for
    /// other variants.
    public var permissionDeniedMessage: String? {
        if case .permissionDenied(let msg) = self { return msg }
        return nil
    }
}
