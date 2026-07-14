// MacControllerTests.swift
// HAKI — Tests
//
// Unit tests for the Mac_Controller permission gate (Req 21.15, 2.2).
//
// Design: Every MacController actuator method must check
// PermissionManager.missingPermissions(for: .macControl) before executing.
// If any permission is missing:
//   • The step MUST NOT run.
//   • An ActuatorResult.permissionDenied(message:) MUST be returned.
//   • The message MUST name the missing permission(s), the blocked step,
//     and the System Settings path to grant it.
//
// These tests use a lightweight stub PermissionManagerProtocol so no
// real TCC dialogs are triggered.

#if canImport(XCTest)
import XCTest
@testable import HAKIOSActions
@testable import HAKIPermissions

// MARK: - Stub PermissionManager

/// A minimal stub that lets tests configure which permissions are granted.
///
/// Conforms to `PermissionManagerProtocol` so it can be injected into
/// `MacController` without triggering real TCC calls.
final class StubPermissionManager: PermissionManagerProtocol, @unchecked Sendable {

    // MARK: - Configurable state

    /// Which permissions are currently "granted".  All others are `.undetermined`.
    var grantedPermissions: Set<HAKIPermission> = []

    // MARK: - PermissionManagerProtocol

    var screenAccessEnabled: Bool = true
    var disabledCapabilities: Set<HAKICapability> = []

    func status(for permission: HAKIPermission) -> PermissionStatus {
        grantedPermissions.contains(permission) ? .granted : .undetermined
    }

    func requestPermission(_ permission: HAKIPermission) async {}

    func missingPermissions(for capability: HAKICapability) -> [HAKIPermission] {
        capability.requiredPermissions.filter { !grantedPermissions.contains($0) }
    }

    func guidanceMessage(for permissions: [HAKIPermission], capability: HAKICapability) -> String {
        guard !permissions.isEmpty else { return "\(capability.displayName) is available." }
        let list = permissions.map { $0.displayName }.joined(separator: ", ")
        return "Missing: \(list) for \(capability.displayName)"
    }

    func watch() -> AsyncStream<PermissionChangeEvent> {
        AsyncStream { $0.finish() }
    }
}

// MARK: - MacControllerTests

final class MacControllerTests: XCTestCase {

    // MARK: - Helpers

    private func makeController(granted: Set<HAKIPermission>) -> (MacController, StubPermissionManager) {
        let stub = StubPermissionManager()
        stub.grantedPermissions = granted
        let controller = MacController(permissionManager: stub)
        return (controller, stub)
    }

    // MARK: - Permission Gate: no permissions granted

    /// When both Accessibility and Automation are missing, every actuator
    /// must return .permissionDenied without executing the step.
    ///
    /// Req 21.15
    func test_launchApp_neitherPermission_returnsPermissionDenied() async {
        let (controller, _) = makeController(granted: [])
        let result = await controller.launchApp(name: "Safari")

        XCTAssertTrue(result.isPermissionDenied,
                      "Expected .permissionDenied but got \(result)")
    }

    func test_bringToFront_neitherPermission_returnsPermissionDenied() async {
        let (controller, _) = makeController(granted: [])
        let result = await controller.bringToFront(appName: "Finder")

        XCTAssertTrue(result.isPermissionDenied)
    }

    func test_runAppleScript_neitherPermission_returnsPermissionDenied() async {
        let (controller, _) = makeController(granted: [])
        let result = await controller.runAppleScript(source: "return 1")

        XCTAssertTrue(result.isPermissionDenied)
    }

    func test_activateElement_neitherPermission_returnsPermissionDenied() async {
        let (controller, _) = makeController(granted: [])
        let result = await controller.activateElement(appName: "Mail", selector: "sendButton")

        XCTAssertTrue(result.isPermissionDenied)
    }

    func test_fillField_neitherPermission_returnsPermissionDenied() async {
        let (controller, _) = makeController(granted: [])
        let result = await controller.fillField(appName: "Mail", selector: "toField", value: "test@example.com")

        XCTAssertTrue(result.isPermissionDenied)
    }

    // MARK: - Permission Gate: only one permission granted

    /// Only Accessibility granted (Automation missing) → still denied (Req 21.15).
    func test_runAppleScript_onlyAccessibility_returnsPermissionDenied() async {
        let (controller, _) = makeController(granted: [.accessibility])
        let result = await controller.runAppleScript(source: "return 1")

        XCTAssertTrue(result.isPermissionDenied,
                      "Automation is still missing; expected .permissionDenied")
    }

    /// Only Automation granted (Accessibility missing) → still denied (Req 21.15).
    func test_activateElement_onlyAutomation_returnsPermissionDenied() async {
        let (controller, _) = makeController(granted: [.automation])
        let result = await controller.activateElement(appName: "Mail", selector: "sendButton")

        XCTAssertTrue(result.isPermissionDenied,
                      "Accessibility is still missing; expected .permissionDenied")
    }

    // MARK: - Permission Gate: all required permissions granted

    /// With both Accessibility and Automation granted, the gate must NOT block
    /// the step.  Because the step itself is a stub (AX not implemented), we
    /// expect .failure (not .permissionDenied) — confirming the gate passed.
    ///
    /// Req 21.15 — gate doesn't fire when permissions are present.
    func test_activateElement_allPermissionsGranted_doesNotReturnPermissionDenied() async {
        let (controller, _) = makeController(granted: [.accessibility, .automation])
        let result = await controller.activateElement(appName: "Mail", selector: "btn")

        // The AX stub returns .failure("not yet implemented"), NOT .permissionDenied.
        XCTAssertFalse(result.isPermissionDenied,
                       "Gate should not fire when all permissions are granted; got \(result)")
    }

    func test_fillField_allPermissionsGranted_doesNotReturnPermissionDenied() async {
        let (controller, _) = makeController(granted: [.accessibility, .automation])
        let result = await controller.fillField(appName: "Mail", selector: "to", value: "x@y.com")

        XCTAssertFalse(result.isPermissionDenied)
    }

    // MARK: - Guidance message content (Req 21.15, 2.2)

    /// The permissionDenied message must name the missing permission.
    func test_permissionDeniedMessage_containsMissingPermissionName() async {
        let (controller, _) = makeController(granted: [])
        let result = await controller.runAppleScript(source: "return 1")

        guard case .permissionDenied(let msg) = result else {
            XCTFail("Expected .permissionDenied"); return
        }
        // Message must mention what's missing.
        XCTAssertTrue(
            msg.contains("Automation") || msg.contains("Accessibility"),
            "Message should name the missing permission(s). Got:\n\(msg)"
        )
    }

    /// The permissionDenied message must reference the System Settings path.
    func test_permissionDeniedMessage_containsSettingsPath() async {
        let (controller, _) = makeController(granted: [])
        let result = await controller.runAppleScript(source: "return 1")

        guard case .permissionDenied(let msg) = result else {
            XCTFail("Expected .permissionDenied"); return
        }
        // Must include navigation guidance.
        XCTAssertTrue(
            msg.contains("System Settings"),
            "Message should reference System Settings. Got:\n\(msg)"
        )
    }

    /// The permissionDenied message must describe which step is blocked.
    func test_permissionDeniedMessage_containsBlockedStep() async {
        let (controller, _) = makeController(granted: [])
        let result = await controller.launchApp(name: "Finder")

        guard case .permissionDenied(let msg) = result else {
            XCTFail("Expected .permissionDenied"); return
        }
        // "app launch" should appear in the message.
        XCTAssertTrue(
            msg.lowercased().contains("app launch") || msg.lowercased().contains("mac control"),
            "Message should name the blocked step. Got:\n\(msg)"
        )
    }

    // MARK: - ActuatorResult helpers

    func test_actuatorResult_success_isSuccess() {
        let r = ActuatorResult.success(value: "ok")
        XCTAssertTrue(r.isSuccess)
        XCTAssertFalse(r.isPermissionDenied)
        XCTAssertNil(r.permissionDeniedMessage)
    }

    func test_actuatorResult_permissionDenied_accessors() {
        let msg = "Automation permission is not granted."
        let r = ActuatorResult.permissionDenied(message: msg)
        XCTAssertFalse(r.isSuccess)
        XCTAssertTrue(r.isPermissionDenied)
        XCTAssertEqual(r.permissionDeniedMessage, msg)
    }

    func test_actuatorResult_failure_isNotSuccess() {
        let r = ActuatorResult.failure(error: "something went wrong")
        XCTAssertFalse(r.isSuccess)
        XCTAssertFalse(r.isPermissionDenied)
    }

    // MARK: - macControl capability maps to correct permissions

    func test_macControlCapability_requiresAccessibilityAndAutomation() {
        let required = HAKICapability.macControl.requiredPermissions
        XCTAssertTrue(required.contains(.accessibility),
                      ".macControl must require .accessibility")
        XCTAssertTrue(required.contains(.automation),
                      ".macControl must require .automation")
    }
}
#endif // canImport(XCTest)
