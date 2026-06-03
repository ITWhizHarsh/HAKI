// ScreenReaderTests.swift
// HAKI — Unit tests for ScreenReader
//
// Covers:
//  - captureFocused: app not running → .appUnavailable (Req 1.9)
//  - readAloud: .noContent → returns error message, does NOT call voiceEngine (Req 1.6)
//  - readAloud: .appUnavailable → returns appropriate message (Req 1.9)
//  - readAloud: .text → calls voiceEngine.speak, returns nil (Req 1.1)
//  - drainCommands priority: stop > pause > resume (Req 1.8)
//  - drainCommands: empty queue → nil
//  - Permission gate: missing permission → returns guidance message (Req 2.2)
//  - Permission gate: toggle disabled → returns toggle-off message (Req 2.5)
//  - Permission gate: all clear → returns nil (Req 2.3)
//  - startReadAloud: permission blocked → returns guidance before capture
//
// Phase 3 Task 18
// Requirements: 1.1, 1.5, 1.6, 1.8, 1.9, 2.2, 2.3, 2.5

#if canImport(XCTest)
import XCTest
@testable import HAKICapture
@testable import HAKIAudio
@testable import HAKIPermissions

// MARK: - Mocks

// MARK: MockVoiceEngine

/// A minimal mock of `VoiceEngineProtocol` that records calls.
private final class MockVoiceEngine: VoiceEngineProtocol, @unchecked Sendable {

    var speakCallCount = 0
    var bargeInStopCallCount = 0
    var notifyTTSStartedCallCount = 0
    var notifyTTSStoppedCallCount = 0
    var lastSpokenText: String?
    var shouldThrowOnSpeak = false

    let noSpeechPrompt: String = "Sorry, I didn't catch that."

    func listen() throws -> AsyncStream<VoiceEvent> {
        AsyncStream { _ in }
    }

    func stopListening() {}

    func notifyTTSStarted() { notifyTTSStartedCallCount += 1 }
    func notifyTTSStopped() { notifyTTSStoppedCallCount += 1 }

    func bargeInStop() { bargeInStopCallCount += 1 }

    func speak(textStream: AsyncStream<String>) async throws {
        speakCallCount += 1
        // Collect all text from the stream.
        var collected = ""
        for await chunk in textStream {
            collected += chunk
        }
        lastSpokenText = collected
        if shouldThrowOnSpeak {
            throw NSError(domain: "MockTTS", code: 1, userInfo: [NSLocalizedDescriptionKey: "TTS failed"])
        }
    }
}

// MARK: MockPermissionManager

/// A minimal mock of `PermissionManagerProtocol` that returns configurable values.
private final class MockPermissionManager: PermissionManagerProtocol, @unchecked Sendable {

    // Configurable test doubles
    var stubbedMissing: [HAKIPermission] = []
    var screenAccessEnabled: Bool = true
    var disabledCapabilities: Set<HAKICapability> = []

    func status(for permission: HAKIPermission) -> PermissionStatus {
        stubbedMissing.contains(permission) ? .undetermined : .granted
    }

    func requestPermission(_ permission: HAKIPermission) async {}

    func missingPermissions(for capability: HAKICapability) -> [HAKIPermission] {
        stubbedMissing
    }

    func guidanceMessage(for permissions: [HAKIPermission], capability: HAKICapability) -> String {
        let names = permissions.map { $0.displayName }.joined(separator: ", ")
        return "HAKI needs \(names) to use \(capability.displayName)."
    }

    func watch() -> AsyncStream<PermissionChangeEvent> {
        AsyncStream { _ in }
    }
}

// MARK: - ScreenReaderTests

final class ScreenReaderTests: XCTestCase {

    // MARK: captureFocused — app not running (Req 1.9)

    /// When an app name is provided that has no running application, 
    /// captureFocused should return .appUnavailable(name).
    ///
    /// We use a UUID-based name guaranteed to not be running.
    func test_captureFocused_unknownAppName_returnsAppUnavailable() async {
        let reader = ScreenReader()
        let fakeName = "HAKI-NonExistentApp-\(UUID().uuidString)"
        let result = await reader.captureFocused(appName: fakeName)

        if case .appUnavailable(let name) = result {
            XCTAssertEqual(name, fakeName,
                           "appUnavailable should carry the original requested app name")
        } else {
            XCTFail("Expected .appUnavailable, got \(result)")
        }
    }

    // MARK: readAloud — .noContent does NOT call voiceEngine (Req 1.6)

    func test_readAloud_noContent_returnsErrorMessage_andDoesNotSpeak() async {
        let reader = ScreenReader()
        let voiceEngine = MockVoiceEngine()

        let result = await reader.readAloud(content: .noContent, voiceEngine: voiceEngine)

        XCTAssertNotNil(result, "readAloud(.noContent) should return an error message (Req 1.6)")
        XCTAssertEqual(voiceEngine.speakCallCount, 0,
                       "voiceEngine.speak must NOT be called when content is .noContent (Req 1.6)")
        // Verify message content is user-friendly
        XCTAssertTrue(result?.contains("No readable text") == true,
                      "Error message should mention 'No readable text'")
    }

    // MARK: readAloud — .appUnavailable returns message (Req 1.9)

    func test_readAloud_appUnavailable_returnsAppMessage_andDoesNotSpeak() async {
        let reader = ScreenReader()
        let voiceEngine = MockVoiceEngine()
        let appName = "TestApp"

        let result = await reader.readAloud(content: .appUnavailable(appName), voiceEngine: voiceEngine)

        XCTAssertNotNil(result, "readAloud(.appUnavailable) should return a message (Req 1.9)")
        XCTAssertTrue(result?.contains(appName) == true,
                      "Error message should mention the app name")
        XCTAssertEqual(voiceEngine.speakCallCount, 0,
                       "voiceEngine.speak must NOT be called for .appUnavailable (Req 1.9)")
    }

    // MARK: readAloud — .text calls voiceEngine (Req 1.1)

    func test_readAloud_text_callsSpeakAndReturnsNil() async {
        let reader = ScreenReader()
        let voiceEngine = MockVoiceEngine()
        let sampleText = "Hello, world!"

        let result = await reader.readAloud(content: .text(sampleText), voiceEngine: voiceEngine)

        XCTAssertNil(result, "readAloud(.text) should return nil on success")
        XCTAssertEqual(voiceEngine.speakCallCount, 1, "voiceEngine.speak should be called once")
        XCTAssertEqual(voiceEngine.lastSpokenText, sampleText,
                       "voiceEngine should receive the full text")
    }

    // MARK: readAloud — TTS failure returns error message

    func test_readAloud_ttsFailure_returnsErrorMessage() async {
        let reader = ScreenReader()
        let voiceEngine = MockVoiceEngine()
        voiceEngine.shouldThrowOnSpeak = true

        let result = await reader.readAloud(content: .text("some text"), voiceEngine: voiceEngine)

        XCTAssertNotNil(result, "readAloud should return an error message when TTS throws")
        XCTAssertTrue(result?.contains("Read-aloud playback failed") == true,
                      "Error message should indicate playback failed")
    }

    // MARK: drainCommands — empty queue returns nil

    func test_drainCommands_emptyQueue_returnsNil() {
        let reader = ScreenReader()
        XCTAssertNil(reader.drainCommands(),
                     "drainCommands on empty queue should return nil")
    }

    // MARK: drainCommands — stop overrides pause and resume (Req 1.8)

    func test_drainCommands_stopBeforePauseResume_returnsStop() {
        let reader = ScreenReader()
        reader.enqueueCommand(.resume)
        reader.enqueueCommand(.pause)
        reader.enqueueCommand(.stop)

        let command = reader.drainCommands()
        XCTAssertEqual(command, .stop,
                       "stop must have higher priority than pause and resume (Req 1.8)")
    }

    // MARK: drainCommands — pause overrides resume (Req 1.8)

    func test_drainCommands_pauseBeforeResume_returnsPause() {
        let reader = ScreenReader()
        reader.enqueueCommand(.resume)
        reader.enqueueCommand(.pause)

        let command = reader.drainCommands()
        XCTAssertEqual(command, .pause,
                       "pause must have higher priority than resume (Req 1.8)")
    }

    // MARK: drainCommands — resume alone returns resume

    func test_drainCommands_resumeOnly_returnsResume() {
        let reader = ScreenReader()
        reader.enqueueCommand(.resume)

        let command = reader.drainCommands()
        XCTAssertEqual(command, .resume)
    }

    // MARK: drainCommands — multiple stops deduped to one stop

    func test_drainCommands_multipleStops_returnsStopOnce() {
        let reader = ScreenReader()
        reader.enqueueCommand(.stop)
        reader.enqueueCommand(.stop)
        reader.enqueueCommand(.pause)

        let first = reader.drainCommands()
        XCTAssertEqual(first, .stop)

        // Queue is now empty after drain.
        let second = reader.drainCommands()
        XCTAssertNil(second, "Queue should be empty after drain")
    }

    // MARK: drainCommands — stop priority with all three commands

    func test_drainCommands_allThreeCommands_stopWins() {
        let reader = ScreenReader()
        // Enqueue in different orders to confirm priority is deterministic.
        reader.enqueueCommand(.pause)
        reader.enqueueCommand(.resume)
        reader.enqueueCommand(.stop)

        XCTAssertEqual(reader.drainCommands(), .stop, "stop must always win (Req 1.8)")
    }

    // MARK: Permission gate — missing permission returns guidance (Req 2.2)

    func test_startReadAloud_missingPermission_returnsGuidance() async {
        let reader = ScreenReader()
        let voiceEngine = MockVoiceEngine()
        let permManager = MockPermissionManager()
        permManager.stubbedMissing = [.screenRecording, .accessibility]

        let result = await reader.startReadAloud(
            appName: nil,
            voiceEngine: voiceEngine,
            permissionManager: permManager
        )

        XCTAssertNotNil(result, "Should return guidance when permissions are missing (Req 2.2)")
        XCTAssertTrue(result?.contains("Screen Recording") == true ||
                      result?.contains("Accessibility") == true,
                      "Guidance should mention the missing permission(s)")
        XCTAssertEqual(voiceEngine.speakCallCount, 0,
                       "voiceEngine.speak must not be called when permissions are missing")
    }

    // MARK: Permission gate — toggle disabled returns toggle-off message (Req 2.5)

    func test_startReadAloud_toggleDisabled_returnsToggleOffMessage() async {
        let reader = ScreenReader()
        let voiceEngine = MockVoiceEngine()
        let permManager = MockPermissionManager()
        permManager.stubbedMissing = []           // All TCC permissions granted
        permManager.screenAccessEnabled = false   // Toggle is off

        let result = await reader.startReadAloud(
            appName: nil,
            voiceEngine: voiceEngine,
            permissionManager: permManager
        )

        XCTAssertNotNil(result, "Should return a message when toggle is disabled (Req 2.5)")
        XCTAssertTrue(result?.contains("disabled") == true ||
                      result?.contains("Screen Content Access") == true,
                      "Message should mention the toggle")
        XCTAssertEqual(voiceEngine.speakCallCount, 0,
                       "voiceEngine.speak must not be called when toggle is off")
    }

    // MARK: Permission gate — all clear returns nil (Req 2.3)

    /// When permissions are all granted and the toggle is on, checkPermissions
    /// should return nil (no blocking message). We verify this via startReadAloud
    /// with a named app that doesn't exist so we can predict the capture result
    /// without relying on the current macOS environment.
    func test_startReadAloud_allPermissionsGranted_doesNotReturnPermissionMessage() async {
        let reader = ScreenReader()
        let voiceEngine = MockVoiceEngine()
        let permManager = MockPermissionManager()
        permManager.stubbedMissing = []
        permManager.screenAccessEnabled = true

        // Use a definitely-not-running app name so capture returns .appUnavailable
        // which means we got past the permission gate (it returned nil).
        let fakeName = "HAKI-NoPermBlock-\(UUID().uuidString)"
        let result = await reader.startReadAloud(
            appName: fakeName,
            voiceEngine: voiceEngine,
            permissionManager: permManager
        )

        // The result here is the .appUnavailable message, NOT a permission guidance.
        // So it should NOT contain "HAKI needs" (which is our mock guidance format).
        XCTAssertFalse(result?.contains("HAKI needs") == true,
                       "Should not return permission guidance when all permissions are clear (Req 2.3)")
        // But it SHOULD contain the app name (confirming we reached the capture stage).
        XCTAssertTrue(result?.contains(fakeName) == true,
                      "Result should mention the unavailable app name, confirming we passed the permission gate")
    }

    // MARK: enqueueCommand + drainCommands — queue is drained after drain

    func test_enqueueAndDrain_queueIsEmptyAfterDrain() {
        let reader = ScreenReader()
        reader.enqueueCommand(.pause)
        reader.enqueueCommand(.stop)

        _ = reader.drainCommands()

        XCTAssertNil(reader.drainCommands(), "Queue should be empty after drain")
    }
}
#endif // canImport(XCTest)
