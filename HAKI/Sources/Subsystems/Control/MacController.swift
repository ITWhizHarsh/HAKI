// MacController.swift
// HAKI — Control Subsystem
//
// Implements the Mac_Controller actuation backends described in the design:
//
//   • App launch/focus   — NSWorkspace / open(1) / AppleScript  (Req 21.2, 21.10)
//   • Contact resolution — Contacts framework + AppleScript     (Req 21.11, 21.16)
//   • Message-send       — AppleScript / AX actions             (Req 21.3)
//   • Call-placement     — AppleScript / AX actions             (Req 21.4)
//
// Safety rules (from Requirements 21.9, 21.11, 21.16):
//   - If an app is not installed, return .notInstalled and do NOT proceed
//     with dependent steps.
//   - If a contact name is ambiguous (multiple matches) return .ambiguous so
//     the Dialogue_Manager can present options to the user — never auto-pick.
//   - If a contact is not found, return .notFound and inform the user.
//
// Full implementation: Phase 4 Task 24.1.
// Implements: Req 21.2, 21.3, 21.4, 21.10, 21.11, 21.16

import Foundation
import AppKit
import Contacts
import HAKIOSActions

// MARK: - Contact

/// Minimal contact descriptor used by the Mac_Controller.
///
/// `handle` is the phone number, email address, or in-app handle (e.g.
/// WhatsApp number) that the *target app* understands for addressing
/// messages or calls.
public struct Contact: Sendable, Equatable {
    /// Stable system-level identifier (CNContact.identifier or synthetic UUID).
    public let id: String
    /// Human-readable display name shown to the user.
    public let displayName: String
    /// The phone/email/handle the target app uses to route messages or calls.
    public let handle: String

    public init(id: String, displayName: String, handle: String) {
        self.id = id
        self.displayName = displayName
        self.handle = handle
    }
}

// MARK: - ActuatorResult

/// The result of a Mac_Controller actuation attempt.
public enum ActuatorResult: Sendable {
    /// The action completed successfully.
    case success
    /// The named application is not installed on this device (Req 21.9).
    case notInstalled(String)
    /// The named contact was not found in the target application (Req 21.11).
    case notFound(String)
    /// Multiple contacts matched the given name; the user must disambiguate
    /// before any action is taken (Req 21.16).
    case ambiguousContact([Contact])
    /// A required macOS permission was not granted (Req 21.15).
    case permissionDenied(String)
    /// A lower-level error occurred during actuation.
    case failed(Error)
}

// MARK: - ContactResolutionResult

/// The result of a contact-resolution query.
public enum ContactResolutionResult: Sendable {
    /// Exactly one contact matched; safe to proceed.
    case resolved(Contact)
    /// Multiple contacts matched — must NOT auto-pick; surface to user (Req 21.16).
    case ambiguous([Contact])
    /// No contact was found with the given name (Req 21.11).
    case notFound
}

// MARK: - MacController

/// Actuation backend for Mac_Controller.
///
/// This class provides the low-level actuators consumed by the Execution_Engine
/// when it evaluates steps in a CommandPlan that target native macOS apps.
///
/// Thread-safety: all async methods are safe to call from any Swift concurrency
/// context. Methods that call AppKit or AX APIs dispatch to the main actor
/// internally.
public final class MacController: @unchecked Sendable {

    // MARK: - Dependencies

    private let bridge: AppleScriptBridge

    // MARK: - Init

    public init(bridge: AppleScriptBridge = AppleScriptBridge()) {
        self.bridge = bridge
    }

    // =========================================================================
    // MARK: - App Launch / Focus  (Req 21.2, 21.10)
    // =========================================================================

    /// Launch a named application.
    ///
    /// Strategy (in order):
    ///   1. Check whether the app is already running — if so, skip launch.
    ///   2. Locate the app bundle with `NSWorkspace.shared.urlForApplication`.
    ///   3. Launch via `NSWorkspace.shared.openApplication(at:configuration:)`.
    ///   4. Wait up to `launchTimeoutSeconds` for the app to become running.
    ///
    /// Req 21.9: if the app is not installed, return `.notInstalled` and do
    /// not run dependent steps.
    /// Req 21.10: if the app is closed, open it *before* performing dependent
    /// steps — this method must succeed before the caller continues.
    ///
    /// - Parameter name: Display name (e.g. "Safari") or bundle ID
    ///   (e.g. "com.apple.Safari").
    /// - Returns: `.success` when the app is running after this call;
    ///   `.notInstalled` when the app cannot be found on disk.
    public func launchApp(name: String) async -> ActuatorResult {
        // 1. Already running?
        if await resolveRunning(name: name) != nil {
            return .success
        }

        // 2. Find bundle on disk.
        guard let appURL = await locateAppBundle(name: name) else {
            return .notInstalled(name)
        }

        // 3. Launch.
        await MainActor.run {
            let config = NSWorkspace.OpenConfiguration()
            config.activates = true
            // openApplication is async; we fire-and-forget here and poll below.
            NSWorkspace.shared.openApplication(at: appURL, configuration: config)
        }

        // 4. Poll until running or timeout.
        let deadline = Date().addingTimeInterval(launchTimeoutSeconds)
        while Date() < deadline {
            if await resolveRunning(name: name) != nil {
                return .success
            }
            try? await Task.sleep(nanoseconds: 300_000_000) // 0.3 s
        }

        // Final check.
        if await resolveRunning(name: name) != nil {
            return .success
        }

        return .failed(MacControllerError.launchTimeout(name))
    }

    /// Bring a running application to the foreground.
    ///
    /// Strategy:
    ///   1. Find the running app via `NSRunningApplication`.
    ///   2. Call `activate(options: .activateIgnoringOtherApps)`.
    ///   3. Fall back to AppleScript `tell application X to activate` if the
    ///      direct activate returns false (some sandbox targets ignore it).
    ///
    /// Req 21.2: must bring the window to front within 5 s of the command for
    /// an installed, running app.
    ///
    /// - Parameter name: Display name or bundle ID of the target app.
    /// - Returns: `.success` when the app is frontmost; `.notInstalled` when
    ///   not running (caller should call `launchApp` first).
    public func bringToFront(name: String) async -> ActuatorResult {
        guard let app = await resolveRunning(name: name) else {
            // Not running — treat as not installed for focus purposes.
            return .notInstalled(name)
        }

        // Attempt direct activation.
        let activated = await MainActor.run {
            app.activate()
        }

        if activated {
            return .success
        }

        // Fallback: AppleScript activate.
        let appName = app.localizedName ?? name
        let script = "tell application \"\(appName.appleScriptEscaped)\" to activate"
        do {
            _ = try await bridge.run(source: script)
            return .success
        } catch let e as AppleScriptError {
            if case .permissionDenied = e {
                return .permissionDenied("Automation permission required to activate \(appName).")
            }
            return .failed(e)
        } catch {
            return .failed(error)
        }
    }

    // =========================================================================
    // MARK: - Contact Resolution  (Req 21.11, 21.16)
    // =========================================================================

    /// Resolve a contact name within the scope of a named target application.
    ///
    /// Resolution strategy:
    ///   1. Query the system Contacts framework (CNContactStore) for all
    ///      contacts whose full name contains `name` (case-insensitive).
    ///   2. Build `Contact` records, choosing the best handle for the target
    ///      app (phone for FaceTime/Phone, email for Mail, etc.).
    ///   3. Return `.resolved`, `.ambiguous`, or `.notFound` based on the
    ///      match count.
    ///
    /// NEVER auto-picks when multiple matches exist (Req 21.16).
    ///
    /// - Parameters:
    ///   - name: The contact name as spoken/typed by the user.
    ///   - app:  The target application name (influences handle selection).
    /// - Returns: A `ContactResolutionResult`.
    public func resolveContact(name: String, app: String) async -> ContactResolutionResult {
        // Request Contacts access.
        let store = CNContactStore()
        let authStatus = CNContactStore.authorizationStatus(for: .contacts)

        if authStatus == .notDetermined {
            do {
                try await store.requestAccess(for: .contacts)
            } catch {
                return .notFound
            }
        }

        guard CNContactStore.authorizationStatus(for: .contacts) == .authorized else {
            return .notFound
        }

        // Build a predicate that matches any part of the full name.
        let predicate = CNContact.predicateForContacts(matchingName: name)
        let keysToFetch: [CNKeyDescriptor] = [
            CNContactIdentifierKey as CNKeyDescriptor,
            CNContactGivenNameKey as CNKeyDescriptor,
            CNContactFamilyNameKey as CNKeyDescriptor,
            CNContactPhoneNumbersKey as CNKeyDescriptor,
            CNContactEmailAddressesKey as CNKeyDescriptor
        ]

        let cnContacts: [CNContact]
        do {
            cnContacts = try store.unifiedContacts(matching: predicate, keysToFetch: keysToFetch)
        } catch {
            return .notFound
        }

        guard !cnContacts.isEmpty else {
            return .notFound
        }

        let contacts = cnContacts.compactMap { cn -> Contact? in
            let displayName = [cn.givenName, cn.familyName]
                .filter { !$0.isEmpty }
                .joined(separator: " ")
            guard !displayName.isEmpty else { return nil }
            let handle = bestHandle(for: cn, app: app)
            return Contact(id: cn.identifier, displayName: displayName, handle: handle)
        }

        guard !contacts.isEmpty else { return .notFound }

        if contacts.count == 1 {
            return .resolved(contacts[0])
        } else {
            // Multiple matches — must not auto-pick (Req 21.16).
            return .ambiguous(contacts)
        }
    }

    // =========================================================================
    // MARK: - Message-Send Actuator  (Req 21.3, 21.11, 21.16)
    // =========================================================================

    /// Compose and send a message to a contact through a named application.
    ///
    /// Steps (in order):
    ///   1. Resolve contact — abort if ambiguous or not found (Req 21.11, 21.16).
    ///   2. Launch/focus the target app.
    ///   3. Compose and send via AppleScript (preferred) or AX fallback.
    ///
    /// Supported apps (AppleScript dictionaries):
    ///   - Messages.app    — `send text to buddy`
    ///   - WhatsApp Desktop — best-effort AX/AppleScript fallback
    ///
    /// - Parameters:
    ///   - app:     Target application name (e.g. "Messages", "WhatsApp").
    ///   - contact: The contact name as supplied by the user.
    ///   - text:    The message text to send.
    /// - Returns: An `ActuatorResult` indicating success or the failure reason.
    public func sendMessage(app: String, contact: String, text: String) async -> ActuatorResult {
        // 1. Resolve contact.
        switch await resolveContact(name: contact, app: app) {
        case .notFound:
            return .notFound(contact)
        case .ambiguous(let candidates):
            return .ambiguousContact(candidates)
        case .resolved(let resolved):
            // 2. Ensure app is running.
            let launchResult = await launchApp(name: app)
            guard case .success = launchResult else { return launchResult }

            // 3. Bring to front.
            _ = await bringToFront(name: app)

            // 4. Send via AppleScript.
            return await sendMessageAppleScript(
                app: app,
                handle: resolved.handle,
                text: text
            )
        }
    }

    // =========================================================================
    // MARK: - Call-Placement Actuator  (Req 21.4, 21.11, 21.16)
    // =========================================================================

    /// Initiate a call to a contact through a named application.
    ///
    /// Steps (in order):
    ///   1. Resolve contact — abort if ambiguous or not found (Req 21.11, 21.16).
    ///   2. Launch/focus the target app.
    ///   3. Initiate call via AppleScript.
    ///
    /// Supported apps:
    ///   - FaceTime — `tell application "FaceTime" to call "handle"`
    ///   - Phone (macOS Continuity) — same pattern
    ///
    /// - Parameters:
    ///   - app:     Target application name (e.g. "FaceTime", "Phone").
    ///   - contact: The contact name as supplied by the user.
    /// - Returns: An `ActuatorResult` indicating success or the failure reason.
    public func placeCall(app: String, contact: String) async -> ActuatorResult {
        // 1. Resolve contact.
        switch await resolveContact(name: contact, app: app) {
        case .notFound:
            return .notFound(contact)
        case .ambiguous(let candidates):
            return .ambiguousContact(candidates)
        case .resolved(let resolved):
            // 2. Ensure app is running.
            let launchResult = await launchApp(name: app)
            guard case .success = launchResult else { return launchResult }

            // 3. Bring to front.
            _ = await bringToFront(name: app)

            // 4. Initiate call via AppleScript.
            return await placeCallAppleScript(app: app, handle: resolved.handle)
        }
    }

    // =========================================================================
    // MARK: - Private: AppleScript helpers
    // =========================================================================

    /// Send a message via the target app's AppleScript dictionary.
    ///
    /// Messages.app dictionary: `send "text" to buddy "handle" of service "SMS"`
    /// WhatsApp Desktop: no standard dictionary; we fall back to UI scripting.
    private func sendMessageAppleScript(
        app: String,
        handle: String,
        text: String
    ) async -> ActuatorResult {
        let escapedApp    = app.appleScriptEscaped
        let escapedHandle = handle.appleScriptEscaped
        let escapedText   = text.appleScriptEscaped

        // Build the script depending on the target app.
        let script: String
        let lowerApp = app.lowercased()

        if lowerApp.contains("messages") || lowerApp.contains("imessage") {
            // Messages.app AppleScript dictionary (most reliable path).
            script = """
            tell application "Messages"
                activate
                set targetService to 1st service whose service type = iMessage
                set targetBuddy to buddy "\(escapedHandle)" of targetService
                send "\(escapedText)" to targetBuddy
            end tell
            """
        } else if lowerApp.contains("whatsapp") {
            // WhatsApp Desktop: use UI scripting via System Events as fallback.
            // Note: this requires Accessibility permission.
            script = """
            tell application "\(escapedApp)" to activate
            delay 0.5
            tell application "System Events"
                tell process "\(escapedApp)"
                    keystroke "n" using command down
                    delay 0.3
                    keystroke "\(escapedHandle)"
                    delay 0.5
                    key code 36
                    delay 0.3
                    keystroke "\(escapedText)"
                    key code 36
                end tell
            end tell
            """
        } else {
            // Generic: try a simple `send` tell — works for scriptable IM apps.
            script = """
            tell application "\(escapedApp)"
                activate
                send "\(escapedText)" to buddy "\(escapedHandle)"
            end tell
            """
        }

        do {
            _ = try await bridge.run(source: script)
            return .success
        } catch let e as AppleScriptError {
            if case .permissionDenied = e {
                return .permissionDenied("Automation permission required to send messages via \(app).")
            }
            return .failed(e)
        } catch {
            return .failed(error)
        }
    }

    /// Initiate a call via the target app's AppleScript dictionary.
    private func placeCallAppleScript(app: String, handle: String) async -> ActuatorResult {
        let escapedApp    = app.appleScriptEscaped
        let escapedHandle = handle.appleScriptEscaped

        let lowerApp = app.lowercased()
        let script: String

        if lowerApp.contains("facetime") {
            script = """
            tell application "FaceTime"
                activate
                call "\(escapedHandle)"
            end tell
            """
        } else if lowerApp.contains("phone") {
            // macOS Continuity Phone app.
            script = """
            tell application "Phone"
                activate
                call "\(escapedHandle)"
            end tell
            """
        } else {
            // Generic fallback for other calling apps.
            script = """
            tell application "\(escapedApp)"
                activate
                call "\(escapedHandle)"
            end tell
            """
        }

        do {
            _ = try await bridge.run(source: script)
            return .success
        } catch let e as AppleScriptError {
            if case .permissionDenied = e {
                return .permissionDenied("Automation permission required to place calls via \(app).")
            }
            return .failed(e)
        } catch {
            return .failed(error)
        }
    }

    // =========================================================================
    // MARK: - Private: NSWorkspace helpers
    // =========================================================================

    /// Maximum seconds to wait for an app to appear in the running-app list
    /// after a launch attempt (Req 21.2 budget: 5 s end-to-end).
    private let launchTimeoutSeconds: TimeInterval = 4.5

    /// Find a running application by display name or bundle identifier.
    ///
    /// - Parameter name: Display name (e.g. "Safari") or bundle ID.
    /// - Returns: The matching `NSRunningApplication`, or `nil`.
    private func resolveRunning(name: String) async -> NSRunningApplication? {
        return await MainActor.run {
            let running = NSWorkspace.shared.runningApplications
            let lower = name.lowercased()

            // Exact bundle ID match.
            if let byBundle = running.first(where: {
                $0.bundleIdentifier?.lowercased() == lower
            }) { return byBundle }

            // Exact localised-name match.
            if let byName = running.first(where: {
                $0.localizedName?.lowercased() == lower
            }) { return byName }

            // Partial name match (last resort).
            return running.first(where: {
                guard let locName = $0.localizedName else { return false }
                return locName.lowercased().contains(lower)
            })
        }
    }

    /// Locate an application bundle by name or bundle ID on disk.
    ///
    /// Uses `NSWorkspace.urlForApplication(withBundleIdentifier:)` for bundle-ID
    /// queries and `NSWorkspace.urlForApplication(toOpen:)` heuristics for
    /// display-name queries.
    ///
    /// - Parameter name: Display name or bundle ID.
    /// - Returns: The file URL of the app bundle, or `nil` when not found.
    private func locateAppBundle(name: String) async -> URL? {
        return await MainActor.run {
            // Try as a bundle identifier first.
            if let url = NSWorkspace.shared.urlForApplication(
                withBundleIdentifier: name
            ) { return url }

            // Try resolving via the `open` scheme with a display-name search.
            // NSWorkspace doesn't have a direct "search by display name" API
            // before macOS 12, so we search well-known locations.
            let searchDirs = [
                "/Applications",
                "/Applications/Utilities",
                (NSHomeDirectory() as NSString).appendingPathComponent("Applications")
            ]
            let lower = name.lowercased()
            let fm = FileManager.default

            for dir in searchDirs {
                guard let entries = try? fm.contentsOfDirectory(atPath: dir) else { continue }
                for entry in entries {
                    guard entry.hasSuffix(".app") else { continue }
                    let appName = (entry as NSString).deletingPathExtension.lowercased()
                    if appName == lower || appName.contains(lower) {
                        return URL(fileURLWithPath: (dir as NSString).appendingPathComponent(entry))
                    }
                }
            }
            return nil
        }
    }

    // =========================================================================
    // MARK: - Private: Contact handle selection
    // =========================================================================

    /// Choose the best communication handle for a CNContact given the target app.
    ///
    /// - phone number  → for FaceTime, Phone, Messages (SMS), WhatsApp
    /// - email address → for Mail, FaceTime (email-based FaceTime IDs)
    ///
    /// Falls back to the first available value of either kind.
    private func bestHandle(for contact: CNContact, app: String) -> String {
        let lower = app.lowercased()
        let prefersPhone = lower.contains("facetime") ||
                           lower.contains("phone")    ||
                           lower.contains("messages") ||
                           lower.contains("whatsapp")

        if prefersPhone {
            if let phone = contact.phoneNumbers.first {
                return phone.value.stringValue
            }
        }
        if let email = contact.emailAddresses.first {
            return email.value as String
        }
        // Last resort: first phone number.
        if let phone = contact.phoneNumbers.first {
            return phone.value.stringValue
        }
        return ""
    }
}

// MARK: - MacControllerError

/// Domain-specific errors thrown by MacController internals.
public enum MacControllerError: Error {
    /// The app did not appear in the running-app list within the timeout.
    case launchTimeout(String)
    /// An AX-based UI scripting action failed.
    case axActionFailed(String)
}

// MARK: - String + AppleScript escaping

private extension String {
    /// Escape a string for safe embedding inside an AppleScript quoted literal.
    ///
    /// AppleScript uses `"` as the string delimiter; we escape embedded quotes
    /// by replacing `"` with `\" ` (backslash-quote).  Backslashes themselves
    /// are not treated as escape characters by the AppleScript runtime inside
    /// `NSAppleScript`, so we only need to handle the quote character.
    var appleScriptEscaped: String {
        self.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }
}
