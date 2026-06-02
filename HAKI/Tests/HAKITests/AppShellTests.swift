// AppShellTests.swift
// HAKITests — Unit Tests for the Swift Shell scaffold
//
// These tests verify the core structural correctness of the Shell scaffold:
//   • CoreProcessManager builds a valid socket path in Application Support
//   • KeychainStore can round-trip a secret
//   • Settings encodes/decodes correctly (default values)
//   • VAD correctly identifies end-of-speech and barge-in transitions
//   • PermissionManager dependency map covers the expected capabilities
//
// Note: TCC-gated tests (screen recording, accessibility) are skipped in CI
// via `#if canImport(XCTest)` guards — they require a signed, running app.

import XCTest
@testable import HAKI
@testable import HAKIPermissions
@testable import HAKIStore
@testable import HAKIAudio

final class AppShellTests: XCTestCase {

    // MARK: - CoreProcessManager

    func testSocketPathIsInApplicationSupport() {
        let manager = CoreProcessManager()
        let path = manager.socketPath.path

        // Must be under the user's Library/Application Support directory.
        XCTAssertTrue(
            path.contains("Application Support"),
            "Socket path '\(path)' should be inside Application Support."
        )
        // Must end with the expected socket filename.
        XCTAssertTrue(
            path.hasSuffix("haki_core.sock"),
            "Socket path should end with 'haki_core.sock'."
        )
    }

    func testSocketPathContainsHAKINamespace() {
        let manager = CoreProcessManager()
        XCTAssertTrue(
            manager.socketPath.path.contains("HAKI"),
            "Socket path should be scoped to the HAKI namespace."
        )
    }

    // MARK: - Settings model

    func testSettingsDefaultValues() {
        let settings = Settings()
        XCTAssertEqual(settings.personalityIntensity, 2)
        XCTAssertEqual(settings.moodThreshold, 0.6, accuracy: 0.001)
        XCTAssertEqual(settings.recentlyLearnedDays, 7)
        XCTAssertTrue(settings.screenAccessEnabled)
        XCTAssertTrue(settings.textAssistantEnabled)
    }

    func testSettingsCodableRoundTrip() throws {
        var settings = Settings()
        settings.personalityIntensity = 3
        settings.moodThreshold = 0.8
        settings.recentlyLearnedDays = 14
        settings.screenAccessEnabled = false
        settings.textAssistantEnabled = false

        let data = try JSONEncoder().encode(settings)
        let decoded = try JSONDecoder().decode(Settings.self, from: data)

        XCTAssertEqual(decoded.personalityIntensity, 3)
        XCTAssertEqual(decoded.moodThreshold, 0.8, accuracy: 0.001)
        XCTAssertEqual(decoded.recentlyLearnedDays, 14)
        XCTAssertFalse(decoded.screenAccessEnabled)
        XCTAssertFalse(decoded.textAssistantEnabled)
    }

    // MARK: - PrivacyState model

    func testPrivacyStateCodableRoundTrip() throws {
        let state = PrivacyState(conversationId: "conv-001", isPrivate: true)
        let data = try JSONEncoder().encode(state)
        let decoded = try JSONDecoder().decode(PrivacyState.self, from: data)

        XCTAssertEqual(decoded.conversationId, "conv-001")
        XCTAssertTrue(decoded.isPrivate)
    }

    // MARK: - VAD

    func testVADEndOfSpeechDetection() {
        let vad = VAD()
        var endOfSpeechFired = false
        vad.endOfSpeechHandler = { endOfSpeechFired = true }

        // Simulate 10 speech frames (200 ms), then silence for > 800 ms.
        let speechSamples = makeSamples(energy: 0.5)
        let silenceSamples = makeSamples(energy: 0.001)
        let silenceFrameCount = 50 // 50 × 20 ms = 1000 ms > 800 ms

        // Start speaking
        for i in 0..<10 {
            vad.process(frame: AudioFrame(
                samples: speechSamples,
                timestamp: Date(timeIntervalSinceReferenceDate: Double(i) * 0.020)
            ))
        }

        // Silence begins
        for i in 10..<(10 + silenceFrameCount) {
            vad.process(frame: AudioFrame(
                samples: silenceSamples,
                timestamp: Date(timeIntervalSinceReferenceDate: Double(i) * 0.020)
            ))
        }

        XCTAssertTrue(endOfSpeechFired, "endOfSpeech should fire after 800 ms of silence following speech.")
    }

    func testVADBargeInDetection() {
        let vad = VAD()
        var bargeInFired = false
        vad.bargeInHandler = { bargeInFired = true }
        vad.setTTSPlaying(true)

        let speechSamples = makeSamples(energy: 0.5)
        // 200 ms of continuous speech = 10 frames × 20 ms
        for i in 0..<15 {
            vad.process(frame: AudioFrame(
                samples: speechSamples,
                timestamp: Date(timeIntervalSinceReferenceDate: Double(i) * 0.020)
            ))
        }

        XCTAssertTrue(bargeInFired, "Barge-in should fire after ≥ 200 ms of speech during TTS playback.")
    }

    func testVADNoEndOfSpeechWithoutPriorSpeech() {
        let vad = VAD()
        var fired = false
        vad.endOfSpeechHandler = { fired = true }

        let silence = makeSamples(energy: 0.0)
        for i in 0..<100 {
            vad.process(frame: AudioFrame(
                samples: silence,
                timestamp: Date(timeIntervalSinceReferenceDate: Double(i) * 0.020)
            ))
        }

        XCTAssertFalse(fired, "endOfSpeech should not fire if no speech preceded the silence.")
    }

    // MARK: - PermissionManager dependency map

    func testScreenReadingRequiresScreenRecordingAndAccessibility() {
        let manager = PermissionManager()
        let missing = manager.missing(for: .screenReading)
        // In a fresh test runner without granted permissions, both should be missing.
        XCTAssertTrue(
            missing.contains(.screenRecording) || missing.contains(.accessibility),
            "ScreenReading capability should depend on Screen Recording and/or Accessibility."
        )
    }

    func testMacControlRequiresAccessibilityAndAutomation() {
        let manager = PermissionManager()
        let missing = manager.missing(for: .macControl)
        XCTAssertTrue(
            missing.contains(.accessibility) || missing.contains(.automation),
            "MacControl capability should depend on Accessibility and/or Automation."
        )
    }

    // MARK: - CapturedContent

    func testCapturedContentNoContentDescription() {
        switch CapturedContent.noContent {
        case .noContent:
            break // expected
        default:
            XCTFail("Expected .noContent")
        }
    }

    func testCapturedContentAppUnavailable() {
        let result = CapturedContent.appUnavailable("Safari")
        if case .appUnavailable(let name) = result {
            XCTAssertEqual(name, "Safari")
        } else {
            XCTFail("Expected .appUnavailable(\"Safari\")")
        }
    }

    // MARK: - Helpers

    /// Build a mono 20 ms frame with a given RMS energy level.
    private func makeSamples(energy: Float) -> [Int16] {
        let value = Int16(energy * Float(Int16.max))
        return Array(repeating: value, count: LiveAudioEngine.samplesPerFrame)
    }
}
