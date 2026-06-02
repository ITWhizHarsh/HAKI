// HAKIPermission.swift
// HAKI — Permissions Subsystem
//
// Defines the HAKIPermission enum (the macOS TCC permissions HAKI manages),
// the PermissionStatus enum, the HAKICapability enum (every feature that
// depends on a permission), and the static capability → permission dependency
// map.
//
// Design reference: Permission_Manager component (design.md)
// Implements: Req 2.1, 2.2, 2.6, 21.15

import Foundation

// MARK: - HAKIPermission

/// The set of macOS TCC permissions managed by HAKI.
///
/// Each case maps 1:1 to a macOS permission category:
/// - `.screenRecording` — ScreenCaptureKit / CGPreflightScreenCaptureAccess
/// - `.accessibility`   — AX trusted-process check (AXIsProcessTrusted)
/// - `.automation`      — Apple Events / AEDeterminePermissionToAutomateTarget
///
/// Requirements: 2.1, 2.2, 2.6
public enum HAKIPermission: CaseIterable, Hashable, Sendable, CustomStringConvertible {
    /// Screen Recording (ScreenCaptureKit). Req 1, 2.
    case screenRecording
    /// Accessibility (AXUIElement / AX API). Req 1, 16, 21.
    case accessibility
    /// Automation / Apple Events (Mac_Controller). Req 21, 21.15.
    case automation

    // MARK: Human-readable names

    /// Human-readable name of the permission, used in guidance messages.
    public var displayName: String {
        switch self {
        case .screenRecording: return "Screen Recording"
        case .accessibility:   return "Accessibility"
        case .automation:      return "Automation"
        }
    }

    /// The macOS System Settings deep-link URL that opens the relevant
    /// privacy pane for this permission.  Used when requesting or
    /// directing the user to grant/restore a permission.
    public var systemSettingsURL: URL {
        switch self {
        case .screenRecording:
            // Privacy & Security → Screen Recording
            return URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")!
        case .accessibility:
            // Privacy & Security → Accessibility
            return URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")!
        case .automation:
            // Privacy & Security → Automation
            return URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Automation")!
        }
    }

    /// The System Settings navigation path shown in guidance messages,
    /// e.g. "System Settings → Privacy & Security → Screen Recording".
    public var settingsPath: String {
        "System Settings → Privacy & Security → \(displayName)"
    }

    public var description: String { displayName }
}

// MARK: - PermissionStatus

/// The TCC grant state of a single `HAKIPermission`.
///
/// - `.granted`:       The permission has been approved by the user.
/// - `.denied`:        The user has explicitly denied the permission.
/// - `.undetermined`:  The permission has never been requested (first-run).
///
/// Requirements: 2.1, 2.2
public enum PermissionStatus: Hashable, Sendable, CustomStringConvertible {
    case granted
    case denied
    case undetermined

    public var description: String {
        switch self {
        case .granted:       return "granted"
        case .denied:        return "denied"
        case .undetermined:  return "undetermined"
        }
    }

    /// Convenience: returns true only when `.granted`.
    public var isGranted: Bool { self == .granted }
}

// MARK: - HAKICapability

/// Every user-facing HAKI capability that requires at least one macOS
/// permission to function.
///
/// The `capabilityPermissionMap` below defines which permissions each
/// capability requires.  The `PermissionManager.missingPermissions(for:)`
/// method uses that map to gate capabilities.
///
/// Requirements: 2.2, 2.6, 21.15
public enum HAKICapability: CaseIterable, Hashable, Sendable, CustomStringConvertible {

    // Phase 0 / Phase 1 capabilities
    /// Read-aloud of on-screen content (Req 1, 2).
    case readAloud
    /// Context-aware text autocorrect and autocompletion (Req 16).
    case textAssist
    /// Agentic macOS control — app launch, UI interaction (Req 21).
    case macControl
    /// Sending messages or placing calls via macOS apps (Req 21.3, 21.4).
    case messaging
    /// WhatsApp / email reading via Accessibility (Req 10).
    case commsReading

    // MARK: Human-readable display name

    public var displayName: String {
        switch self {
        case .readAloud:     return "Screen Reading"
        case .textAssist:    return "Smart Text Input"
        case .macControl:    return "Mac Control"
        case .messaging:     return "Messaging & Calling"
        case .commsReading:  return "Communications Reading"
        }
    }

    public var description: String { displayName }
}

// MARK: - Capability → Permission map

/// Maps each `HAKICapability` to the set of macOS permissions it requires.
///
/// This is the single authoritative source consulted by
/// `PermissionManager.missingPermissions(for:)` and
/// `PermissionManager.guidanceMessage(for:capability:)`.
///
/// Req 2.2 ("identify each capability that remains unavailable without that
/// permission"), Req 2.6 ("identify each capability that remains unavailable
/// without that permission"), Req 21.15 ("without Automation/Accessibility
/// permission, control steps don't run and the user is told what's missing").
extension HAKICapability {

    /// The macOS permissions required for this capability to function.
    /// An empty array means the capability has no permission dependency.
    ///
    /// Req 2.2, 2.6, 21.15
    public static let capabilityPermissionMap: [HAKICapability: [HAKIPermission]] = [
        // Read-aloud needs both screen capture (ScreenCaptureKit / OCR fallback)
        // and accessibility (AX text extraction primary path).  Req 1, 2.
        .readAloud:    [.screenRecording, .accessibility],

        // Text assistant uses AX to observe and write into input fields.  Req 16.
        .textAssist:   [.accessibility],

        // Agentic Mac control needs accessibility for UI actions and automation
        // for AppleScript/Apple Events.  Req 21, 21.15.
        .macControl:   [.accessibility, .automation],

        // Sending messages / placing calls via a scriptable app (WhatsApp, Mail)
        // uses both Apple Events and AX UI actions.  Req 21.3, 21.4.
        .messaging:    [.accessibility, .automation],

        // Reading WhatsApp / email via the AX tree of the desktop app.  Req 10.
        .commsReading: [.accessibility],
    ]

    /// Convenience accessor: permissions required for this capability.
    public var requiredPermissions: [HAKIPermission] {
        HAKICapability.capabilityPermissionMap[self] ?? []
    }
}

// MARK: - PermissionChangeEvent

/// An event emitted by `PermissionManager.watch()` when the TCC grant state
/// of a permission changes while HAKI is running.
///
/// The watcher polls every 3 seconds (≤5 s budget, Req 2.7).
///
/// Requirements: 2.7
public struct PermissionChangeEvent: Sendable {
    /// Which permission changed.
    public let permission: HAKIPermission
    /// The new TCC status after the change.
    public let newStatus: PermissionStatus

    public init(permission: HAKIPermission, newStatus: PermissionStatus) {
        self.permission = permission
        self.newStatus = newStatus
    }
}
