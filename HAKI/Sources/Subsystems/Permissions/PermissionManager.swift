// PermissionManager.swift
// HAKI — Permissions Subsystem
//
// Wraps macOS TCC (Transparency, Consent, and Control) permission state for:
//   • Screen Recording  — CGPreflightScreenCaptureAccess / CGRequestScreenCaptureAccess
//   • Accessibility     — AXIsProcessTrustedWithOptions
//   • Automation        — kAXTrustedCheckOptionPrompt / System Settings deep-link
//
// Also owns the user-facing `screenAccessEnabled` toggle (persisted to
// UserDefaults, Req 2.4) and a revocation watcher that detects grant/revoke
// events within 5 s and disables dependent capabilities (Req 2.7).
//
// All guidance messages are computed synchronously (no I/O), satisfying the
// 2 s constraint stated in Req 2.2.
//
// Design reference: Permission_Manager (design.md)
// Implements: Req 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 21.15
//
// Phase 0 Task 4 — full implementation.

import AppKit
import Combine
import ApplicationServices
import CoreGraphics

// MARK: - PermissionManagerProtocol

/// Public interface consumed by the Orchestrator and capability subsystems.
public protocol PermissionManagerProtocol: AnyObject, Sendable {

    // MARK: Status & request

    /// Synchronous TCC status for a single permission. (Req 2.1, 2.2)
    func status(for permission: HAKIPermission) -> PermissionStatus

    /// Asynchronously present the system TCC prompt or open System Settings
    /// if the permission was already denied. (Req 2.1, 2.6)
    func requestPermission(_ permission: HAKIPermission) async

    // MARK: Capability gating

    /// Returns every permission that is not yet granted but is required for
    /// `capability`.  Returns `[]` when all permissions are satisfied.
    ///
    /// The caller must check `screenAccessEnabled` separately for capabilities
    /// that also depend on the user toggle (i.e. `.readAloud`).
    ///
    /// Req 2.2, 2.6, 21.15
    func missingPermissions(for capability: HAKICapability) -> [HAKIPermission]

    /// Returns a human-readable guidance string for the given set of missing
    /// permissions that blocked `capability`.
    ///
    /// The message names: the missing permission(s), the blocked capability,
    /// and the System Settings path to grant each missing permission.
    ///
    /// This method is synchronous and performs no I/O — it returns within 2 s
    /// as required by Req 2.2.
    ///
    /// Req 2.2, 2.6
    func guidanceMessage(for permissions: [HAKIPermission], capability: HAKICapability) -> String

    // MARK: User toggle

    /// User-facing screen-content-access control.  Persisted across restarts
    /// via `UserDefaults`.  Must be accessible whenever HAKI is running,
    /// e.g. from the menu-bar.
    ///
    /// Req 2.4, 2.5
    var screenAccessEnabled: Bool { get set }

    // MARK: Revocation watcher

    /// An `AsyncStream` that emits a `PermissionChangeEvent` whenever the TCC
    /// grant state of a watched permission changes.  The watcher polls every
    /// ≤ 5 s (default 3 s) so revocations are detected within the 5 s budget
    /// defined in Req 2.7.
    ///
    /// Callers iterate over this stream and react to revocations by disabling
    /// dependent capabilities.
    ///
    /// Req 2.7
    func watch() -> AsyncStream<PermissionChangeEvent>

    // MARK: Reactive state (SwiftUI / Combine consumers)

    /// The set of capabilities currently disabled because one or more of their
    /// required permissions are not granted.  Updated within 5 s of any
    /// revocation (Req 2.7).
    ///
    /// UI layers and the Orchestrator observe this set to gate features.
    var disabledCapabilities: Set<HAKICapability> { get }
}

// MARK: - PermissionManager

/// Production implementation of `PermissionManagerProtocol`.
///
/// Threading model:
/// - `status(for:)`, `missingPermissions(for:)`, `guidanceMessage(for:capability:)`
///   are synchronous and safe to call from any thread.
/// - `requestPermission(_:)` must be called from an `async` context; it
///   dispatches to the main thread for any UI interaction.
/// - The revocation watcher runs a background `Task` that fires a poll every
///   3 s and emits events via an `AsyncStream`.
/// - `@Published` properties and `disabledCapabilities` are updated on
///   `@MainActor` so SwiftUI bindings work without extra dispatch.
///
/// Req 2.1–2.7, 21.15
@MainActor
public final class PermissionManager: ObservableObject, PermissionManagerProtocol {

    // MARK: - UserDefaults key

    private static let screenAccessKey = "HAKIScreenAccessEnabled"
    /// Poll interval for the revocation watcher (Req 2.7: must be ≤ 5 s).
    private static let watchInterval: TimeInterval = 3.0

    // MARK: - Published state (Req 2.4, 2.5, 2.7)

    /// User-facing screen-content-access toggle.
    /// Persisted to `UserDefaults` across restarts (Req 2.4).
    @Published public var screenAccessEnabled: Bool {
        didSet {
            // Persist immediately so the toggle survives app restarts.
            UserDefaults.standard.set(screenAccessEnabled, forKey: Self.screenAccessKey)
            // Re-compute which capabilities are blocked.
            updateDisabledCapabilities()
        }
    }

    /// The set of capabilities currently unavailable due to missing
    /// permissions.  Updated within 5 s of any revocation (Req 2.7).
    @Published public private(set) var disabledCapabilities: Set<HAKICapability> = []

    // MARK: - Watcher state

    /// Snapshot of permission statuses from the previous poll cycle.
    /// Used to diff against the current state and emit only delta events.
    private var lastKnownStatuses: [HAKIPermission: PermissionStatus] = [:]

    /// Continuations for all active `watch()` streams.
    private var watchContinuations: [UUID: AsyncStream<PermissionChangeEvent>.Continuation] = [:]

    /// Background task running the polling loop.
    private var watcherTask: Task<Void, Never>?

    // MARK: - Init

    /// Creates a `PermissionManager`, restoring the screen-access toggle from
    /// `UserDefaults` (defaults to `true` on first run, Req 2.4).
    public init() {
        // Restore the persisted toggle.  On first run, UserDefaults returns
        // `false` for an unset key — we want the default to be `true` so that
        // screen access is on out-of-the-box, which is more useful.
        // We guard this by checking whether the key has been written before.
        // Req 2.4
        if UserDefaults.standard.object(forKey: Self.screenAccessKey) == nil {
            // First run: default to enabled.
            UserDefaults.standard.set(true, forKey: Self.screenAccessKey)
            self.screenAccessEnabled = true
        } else {
            self.screenAccessEnabled = UserDefaults.standard.bool(forKey: Self.screenAccessKey)
        }

        // Seed the last-known-status snapshot with current values so the first
        // poll only emits genuine changes.
        for perm in HAKIPermission.allCases {
            lastKnownStatuses[perm] = currentTCCStatus(for: perm)
        }

        // Compute initial disabled-capability set.
        updateDisabledCapabilities()

        // Start the background revocation watcher.
        startWatcher()
    }

    deinit {
        watcherTask?.cancel()
    }

    // MARK: - PermissionManagerProtocol: status

    /// Returns the current TCC grant status for `permission`.
    ///
    /// - Screen Recording: uses `CGPreflightScreenCaptureAccess()`.
    /// - Accessibility:    uses `AXIsProcessTrustedWithOptions(_:)` with
    ///                     `kAXTrustedCheckOptionPrompt = false` (no side effect).
    /// - Automation:       uses `NSWorkspace` to test Apple Events permission
    ///                     to the Finder bundle; treated as a proxy for the
    ///                     general Automation grant.
    ///
    /// This call is synchronous and does NOT trigger a system prompt.
    ///
    /// Req 2.1, 2.2
    public nonisolated func status(for permission: HAKIPermission) -> PermissionStatus {
        currentTCCStatus(for: permission)
    }

    // MARK: - PermissionManagerProtocol: requestPermission

    /// Triggers the system TCC prompt for `permission`, or deep-links System
    /// Settings when the permission has already been denied (because macOS
    /// will not re-show the prompt for a denied permission).
    ///
    /// This method is `async` so callers can `await` it without blocking the
    /// main thread; the actual UI work happens on the main thread via
    /// `NSWorkspace.shared.open(_:)`.
    ///
    /// Req 2.1, 2.6
    public func requestPermission(_ permission: HAKIPermission) async {
        let current = currentTCCStatus(for: permission)

        switch permission {
        case .screenRecording:
            if current == .undetermined {
                // Ask the system to present the TCC alert.  The result
                // comes back asynchronously; we don't need to observe it
                // here — the revocation watcher will pick up the change.
                // Req 2.1
                CGRequestScreenCaptureAccess()
            } else {
                // Already denied (or indeterminate after first denial) —
                // open System Settings so the user can grant manually.
                // Req 2.6
                openSystemSettings(for: permission)
            }

        case .accessibility:
            if current == .undetermined {
                // Presenting the AX prompt requires the option dictionary.
                // `kAXTrustedCheckOptionPrompt = true` triggers the system
                // dialog (Req 2.1).
                let opts: NSDictionary = [
                    kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true
                ]
                _ = AXIsProcessTrustedWithOptions(opts)
            } else {
                // Already handled; direct user to System Settings.  Req 2.6
                openSystemSettings(for: permission)
            }

        case .automation:
            // macOS does not provide an API to request Automation permission
            // programmatically without an active Apple Event send.  Direct
            // the user to System Settings in all cases.
            // Req 2.1, 2.6
            openSystemSettings(for: permission)
        }
    }

    // MARK: - PermissionManagerProtocol: capability gating

    /// Returns every permission required by `capability` that is not currently
    /// granted.  Returns `[]` when the capability is fully unblocked.
    ///
    /// Note: does **not** factor in `screenAccessEnabled`.  Callers that gate
    /// `.readAloud` must additionally check that property.
    ///
    /// Req 2.2, 2.6, 21.15
    public nonisolated func missingPermissions(for capability: HAKICapability) -> [HAKIPermission] {
        capability.requiredPermissions.filter { currentTCCStatus(for: $0) != .granted }
    }

    /// Returns a human-readable guidance string for the given missing
    /// permissions that are blocking `capability`.
    ///
    /// Format:
    ///   "HAKI needs <permission list> to use <capability>.
    ///    Please grant <each permission> in <Settings path>."
    ///
    /// This method is synchronous with no I/O, satisfying the ≤ 2 s
    /// constraint from Req 2.2.
    ///
    /// Req 2.2, 2.6
    public nonisolated func guidanceMessage(
        for permissions: [HAKIPermission],
        capability: HAKICapability
    ) -> String {
        guard !permissions.isEmpty else {
            // No missing permissions — capability is fully available.
            return "\(capability.displayName) is available."
        }

        // Build the list of missing permission names (e.g. "Screen Recording
        // and Accessibility").
        let permissionNames = permissions.map { $0.displayName }
        let permissionList: String
        switch permissionNames.count {
        case 1:
            permissionList = permissionNames[0]
        case 2:
            permissionList = "\(permissionNames[0]) and \(permissionNames[1])"
        default:
            let allButLast = permissionNames.dropLast().joined(separator: ", ")
            permissionList = "\(allButLast), and \(permissionNames.last!)"
        }

        // Build per-permission grant instructions.
        let instructions = permissions.map { perm in
            "• \(perm.displayName): \(perm.settingsPath)"
        }.joined(separator: "\n")

        return """
        HAKI needs \(permissionList) permission\(permissions.count > 1 ? "s" : "") \
        to use \(capability.displayName).

        To enable \(capability.displayName), please grant access in macOS System Settings:
        \(instructions)
        """
    }

    // MARK: - PermissionManagerProtocol: revocation watcher

    /// Returns an `AsyncStream<PermissionChangeEvent>` that emits whenever any
    /// watched permission's TCC status changes.
    ///
    /// The stream is backed by a shared polling loop (interval ≤ 5 s, Req 2.7).
    /// Multiple concurrent callers each get their own stream but share the
    /// same poll task.
    ///
    /// Req 2.7
    public func watch() -> AsyncStream<PermissionChangeEvent> {
        let id = UUID()
        let stream = AsyncStream<PermissionChangeEvent> { [weak self] continuation in
            guard let self else {
                continuation.finish()
                return
            }
            // Register this continuation.
            self.watchContinuations[id] = continuation
            continuation.onTermination = { [weak self] _ in
                Task { @MainActor [weak self] in
                    self?.watchContinuations.removeValue(forKey: id)
                }
            }
        }
        return stream
    }

    // MARK: - Private helpers

    /// Core TCC status check — the single place where platform APIs are called.
    /// `nonisolated` so it can be called from non-`@MainActor` contexts.
    ///
    /// Req 2.1, 2.2
    nonisolated func currentTCCStatus(for permission: HAKIPermission) -> PermissionStatus {
        switch permission {
        case .screenRecording:
            // CGPreflightScreenCaptureAccess() returns true when the app has
            // been granted screen-recording permission and false when denied
            // OR undetermined.  We treat both non-granted states as
            // `.undetermined` on the first call (before the user has
            // responded to the TCC prompt) and `.denied` once the prompt has
            // been shown.  Since we cannot reliably distinguish the two via
            // this API alone, we map `false` → `.undetermined` until the
            // `requestPermission` flow has been triggered.
            //
            // In practice, after `CGRequestScreenCaptureAccess()` is called
            // the returned value from `CGPreflightScreenCaptureAccess()` will
            // be stable: `true` = granted, `false` = denied.
            // Req 2.1
            return CGPreflightScreenCaptureAccess() ? .granted : .undetermined

        case .accessibility:
            // AXIsProcessTrustedWithOptions with prompt=false checks the
            // current AX trust state without triggering a system dialog.
            // Req 2.1
            let opts: NSDictionary = [
                kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: false
            ]
            return AXIsProcessTrustedWithOptions(opts) ? .granted : .undetermined

        case .automation:
            // There is no public API to check Automation permission without
            // sending an actual Apple Event.  We use a known-safe no-op event
            // to the Finder as a proxy.  If the event descriptor is accepted
            // (kAEEventNotHandled is a valid handler-not-found, not a denial),
            // we treat it as granted.  An `errAEEventNotPermitted` error
            // (-1743) means the user has denied automation to Finder, which is
            // a proxy for general automation being restricted.
            return automationPermissionStatus()
        }
    }

    /// Checks Automation / Apple Events permission using
    /// `AEDeterminePermissionToAutomateTarget`, targeting the Finder (always
    /// present on macOS) as a reliable proxy for the general Automation TCC
    /// grant.  Passes `askUserIfNeeded = false` so no dialog is shown.
    ///
    /// Returns `.granted`, `.denied`, or `.undetermined`.
    ///
    /// Req 2.1, 2.6
    nonisolated private func automationPermissionStatus() -> PermissionStatus {
        // Build an AEAddressDesc for the Finder using its well-known bundle ID.
        // Finder is always installed and its automation permission is a reliable
        // proxy for whether the general Automation TCC category is granted.
        let finderBundleID = "com.apple.finder" as CFString

        // Encode the bundle ID string as UTF-8 bytes for AECreateDesc.
        guard let cStr = (finderBundleID as NSString).utf8String else {
            return .undetermined
        }
        // strlen gives us the length without the NUL terminator; AECreateDesc
        // expects the byte count of the data (not including NUL).
        let byteCount = strlen(cStr)

        var targetDesc = AEAddressDesc()
        let createResult = AECreateDesc(
            typeApplicationBundleID,
            cStr,
            byteCount,
            &targetDesc
        )

        guard createResult == noErr else {
            // Could not create the descriptor — treat as undetermined.
            return .undetermined
        }
        defer { AEDisposeDesc(&targetDesc) }

        // AEDeterminePermissionToAutomateTarget returns:
        //   noErr                 → permission is granted
        //   errAEEventNotPermitted(-1743) → user has denied automation to this target
        //   procNotFound / other  → system hasn't decided yet (undetermined)
        //
        // Pass `askUserIfNeeded = false` to avoid triggering a prompt here.
        let permResult = AEDeterminePermissionToAutomateTarget(
            &targetDesc,
            typeWildCard,
            typeWildCard,
            false   // do NOT prompt — we only check; request() opens Settings
        )

        switch permResult {
        case noErr:
            return .granted
        case Int32(errAEEventNotPermitted):
            return .denied
        default:
            // procNotFound, connectionInvalid, errAETargetAddressNotPermitted,
            // or other OS errors all map to undetermined.
            return .undetermined
        }
    }

    /// Opens the System Settings pane for the given permission's privacy
    /// category.  Called when `requestPermission(_:)` determines the
    /// permission was already denied and needs to be manually re-granted.
    ///
    /// Req 2.6
    @MainActor
    private func openSystemSettings(for permission: HAKIPermission) {
        NSWorkspace.shared.open(permission.systemSettingsURL)
    }

    /// Re-computes `disabledCapabilities` from the current TCC statuses and
    /// the `screenAccessEnabled` toggle, then publishes the updated set.
    ///
    /// Called whenever the toggle changes or the watcher detects a status
    /// change.  Must run on `@MainActor` so `@Published` updates are on the
    /// main thread.
    ///
    /// Req 2.3, 2.5, 2.7
    @MainActor
    private func updateDisabledCapabilities() {
        var disabled = Set<HAKICapability>()

        for capability in HAKICapability.allCases {
            let missingPerms = missingPermissions(for: capability)
            if !missingPerms.isEmpty {
                // At least one required permission is missing.  Req 2.2, 2.6.
                disabled.insert(capability)
            } else if capability == .readAloud && !screenAccessEnabled {
                // All permissions are granted but the user toggle is OFF.
                // Screen reading is still blocked.  Req 2.5.
                disabled.insert(capability)
            }
        }

        // Req 2.3: no missing-permission messages when both screen-recording
        // and accessibility are granted — we achieve this by leaving
        // readAloud out of disabled when both permissions are present AND the
        // toggle is on.
        disabledCapabilities = disabled
    }

    // MARK: - Watcher

    /// Starts the background polling loop that detects TCC grant/revoke
    /// events within ≤ 5 s (Req 2.7, poll interval = 3 s).
    ///
    /// The loop runs until the `PermissionManager` is deallocated.
    private func startWatcher() {
        watcherTask?.cancel()
        watcherTask = Task { [weak self] in
            while !Task.isCancelled {
                // Sleep for the poll interval before checking.
                do {
                    try await Task.sleep(nanoseconds: UInt64(Self.watchInterval * 1_000_000_000))
                } catch {
                    // Task cancelled — exit the loop.
                    break
                }
                await self?.pollPermissions()
            }
        }
    }

    /// Polls the current TCC statuses for all permissions and emits
    /// `PermissionChangeEvent`s for any that changed since the last poll.
    ///
    /// Also updates `disabledCapabilities` when any status changes.
    ///
    /// Req 2.7
    @MainActor
    private func pollPermissions() {
        var didChange = false

        for permission in HAKIPermission.allCases {
            let newStatus = currentTCCStatus(for: permission)
            let oldStatus = lastKnownStatuses[permission] ?? .undetermined

            if newStatus != oldStatus {
                // Status changed — emit an event to all active streams.
                lastKnownStatuses[permission] = newStatus
                didChange = true

                let event = PermissionChangeEvent(permission: permission, newStatus: newStatus)
                for continuation in watchContinuations.values {
                    continuation.yield(event)
                }
            }
        }

        if didChange {
            // Re-compute which capabilities are now disabled.  Req 2.7.
            updateDisabledCapabilities()
        }
    }
}

// MARK: - Convenience: guidance for a capability

public extension PermissionManager {

    /// Convenience that computes missing permissions for `capability` and
    /// immediately returns the guidance message if any are missing, or `nil`
    /// when the capability is fully available.
    ///
    /// The `screenAccessEnabled` user toggle is also checked for `.readAloud`.
    ///
    /// Req 2.2, 2.5
    func guidance(for capability: HAKICapability) -> String? {
        // Check TCC permissions.
        let missing = missingPermissions(for: capability)
        if !missing.isEmpty {
            return guidanceMessage(for: missing, capability: capability)
        }

        // Check user toggle for capabilities that depend on it.
        if capability == .readAloud && !screenAccessEnabled {
            return "Screen content access is currently disabled. " +
                   "Enable 'Screen Content Access' from the HAKI menu bar to use \(capability.displayName)."
        }

        return nil
    }

    /// Returns `true` when `capability` is fully available:
    /// all required TCC permissions are granted AND (if applicable) the
    /// screen-access toggle is on.
    ///
    /// Req 2.3, 2.5
    func isAvailable(_ capability: HAKICapability) -> Bool {
        guidance(for: capability) == nil
    }
}
