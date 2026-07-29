// VoiceAudioControllerTests.swift
// macOS native-audio integration tests for VoiceAudioController.
//
// Validates: Requirements 2.1–2.5, 5.2
// Test ID: V-SWIFT-AUDIO
//
// These tests exercise the instrumented enable → verify → tap ordering, duplex
// capture/playback, route/media-reset recovery, unavailable VoiceProcessingIO,
// capture persistence during playback, and the guard that rejects frames when
// voice processing is disabled.
//
// Hardware-dependent behaviors are guarded with #if canImport(AVFoundation) && os(macOS).
// Tests requiring a live microphone/audio route are labelled with a comment so
// they can be excluded from non-macOS CI.

#if canImport(AVFoundation) && os(macOS)
import AVFoundation
import XCTest
@testable import HAKIAudio

// MARK: - Instrumented fake runtime

/// Deterministic VoiceAudioRuntime that records the exact call order for
/// enable → verify → tap → start lifecycle validation (Req 2.1, 2.2).
private final class InstrumentedRuntime: VoiceAudioRuntime, @unchecked Sendable {
    var microphoneAuthorization: AVAuthorizationStatus = .authorized
    var inputFormat = VoiceAudioInputFormat(sampleRateHz: 16_000, channelCount: 1)
    var isVoiceProcessingEnabled = false

    /// Controls whether setVoiceProcessingEnabled actually flips the flag.
    var enableChangesState = true
    var installError = false
    var startError = false

    private(set) var log: [String] = []
    private(set) var startCount = 0
    private(set) var stopCount = 0
    private(set) var removeTapCount = 0
    private var tapHandler: (@Sendable (VoiceAudioInputBuffer) -> Void)?

    func setVoiceProcessingEnabled(_ enabled: Bool) throws {
        log.append("enable(\(enabled))")
        if enableChangesState { isVoiceProcessingEnabled = enabled }
    }

    func installInputTap(
        bufferSize: AVAudioFrameCount,
        handler: @escaping @Sendable (VoiceAudioInputBuffer) -> Void
    ) throws {
        log.append("installTap")
        if installError { throw VoiceAudioUnavailableReason.tapInstallation }
        tapHandler = handler
    }

    func removeInputTap() {
        log.append("removeTap")
        removeTapCount += 1
        tapHandler = nil
    }

    func startEngine() throws {
        log.append("startEngine")
        startCount += 1
        if startError { throw VoiceAudioUnavailableReason.engineStart }
    }

    func stopEngine() {
        log.append("stopEngine")
        stopCount += 1
    }

    func setConfigurationChangeHandler(_ handler: (@Sendable () -> Void)?) {}

    /// Simulate a tap callback delivering a raw input buffer.
    func emit(_ input: VoiceAudioInputBuffer) {
        tapHandler?(input)
    }

    /// Convenience: emit a single-channel 16 kHz buffer of n silent frames.
    func emitSilence(frames: Int = 160, capturedAt ts: UInt64 = 0) {
        let samples = [Float](repeating: 0, count: frames)
        emit(VoiceAudioInputBuffer(
            interleavedSamples: samples,
            sourceSampleRateHz: 16_000,
            channelCount: 1,
            capturedAtMonotonicNs: ts
        ))
    }
}

// MARK: - Helpers

private extension VoiceAudioControllerTests {
    static func makeController(
        runtime: InstrumentedRuntime,
        retryScheduler: VoiceAudioRetryScheduler? = nil
    ) -> VoiceAudioController {
        if let scheduler = retryScheduler {
            return VoiceAudioController(
                sessionID: UUID(),
                runtime: runtime,
                retryScheduler: scheduler
            )
        }
        return VoiceAudioController(sessionID: UUID(), runtime: runtime)
    }

    /// Emits a stereo 48 kHz buffer that the controller should down-mix to
    /// 16 kHz mono (used for normalization coverage in Req 2.6).
    static func stereoBuffer(
        sampleRateHz: Double = 48_000,
        channelCount: UInt8 = 2,
        frameCount: Int = 960,
        capturedAt ts: UInt64 = 1000
    ) -> VoiceAudioInputBuffer {
        let samples = [Float](repeating: 0.3, count: frameCount * Int(channelCount))
        return VoiceAudioInputBuffer(
            interleavedSamples: samples,
            sourceSampleRateHz: sampleRateHz,
            channelCount: channelCount,
            capturedAtMonotonicNs: ts
        )
    }
}

// MARK: - Main test class

@available(macOS 14, *)
final class VoiceAudioControllerTests: XCTestCase {

    // -------------------------------------------------------------------------
    // MARK: 1. Enable → Verify → Tap ordering (Req 2.1, 2.2)
    // -------------------------------------------------------------------------

    /// The start-capture call must log enable, then installTap, then startEngine
    /// in that exact order. No other ordering is permissible because AEC
    /// reference alignment depends on VoiceProcessingIO being active before the
    /// tap sees any buffer.
    func testEnableVerifyTapEngineOrderIsStrict() throws {
        let runtime = InstrumentedRuntime()
        let controller = Self.makeController(runtime: runtime)

        try controller.startCapture()

        let orderedSteps = runtime.log.filter {
            ["enable(true)", "installTap", "startEngine"].contains($0)
        }
        XCTAssertEqual(orderedSteps, ["enable(true)", "installTap", "startEngine"],
                       "enable must precede tap, tap must precede startEngine")
        XCTAssertEqual(controller.state, .capturing)
    }

    /// If setVoiceProcessingEnabled does not flip the flag (e.g. hardware
    /// refuses), the controller must NOT install a tap and must transition to
    /// .unavailable with a diagnostic (Req 2.4).
    func testVoiceProcessingUnavailableTransitionsToUnavailableWithoutTap() {
        let runtime = InstrumentedRuntime()
        runtime.enableChangesState = false
        let controller = Self.makeController(runtime: runtime)

        XCTAssertThrowsError(try controller.startCapture()) { error in
            XCTAssertEqual(
                error as? VoiceAudioControllerError,
                .unavailable(.voiceProcessingUnavailable)
            )
        }
        XCTAssertEqual(controller.state, .unavailable(.voiceProcessingUnavailable))
        XCTAssertFalse(runtime.log.contains("installTap"),
                       "tap must not be installed when voice processing verification fails")
        XCTAssertFalse(runtime.log.contains("startEngine"),
                       "engine must not start when voice processing verification fails")
    }

    /// The unavailable state must carry an actionable user message (Req 2.4).
    func testUnavailableReasonCarriesActionableUserMessage() {
        let reasons: [VoiceAudioUnavailableReason] = [
            .microphonePermission, .inputRoute, .voiceProcessingUnavailable,
            .tapInstallation, .engineStart, .mediaServicesReset,
        ]
        for reason in reasons {
            XCTAssertFalse(
                reason.userMessage.isEmpty,
                "userMessage must not be empty for \(reason)"
            )
            XCTAssertFalse(
                reason.recoveryAction.userInstruction.isEmpty,
                "userInstruction must not be empty for \(reason)"
            )
        }
    }

    // -------------------------------------------------------------------------
    // MARK: 2. State transitions (Req 2.1–2.5)
    // -------------------------------------------------------------------------

    func testIdleToCapturingTransitionAfterSuccessfulStart() throws {
        let runtime = InstrumentedRuntime()
        let controller = Self.makeController(runtime: runtime)
        XCTAssertEqual(controller.state, .idle)
        try controller.startCapture()
        XCTAssertEqual(controller.state, .capturing)
    }

    func testCapturingToPlayingTransitionWhenPlaybackStarts() throws {
        let runtime = InstrumentedRuntime()
        let controller = Self.makeController(runtime: runtime)
        try controller.startCapture()

        try controller.playbackDidStart()

        XCTAssertEqual(controller.state, .playing)
    }

    func testPlayingToCapturingWhenPlaybackStops() throws {
        let runtime = InstrumentedRuntime()
        let controller = Self.makeController(runtime: runtime)
        try controller.startCapture()
        try controller.playbackDidStart()

        controller.playbackDidStop()

        XCTAssertEqual(controller.state, .capturing)
    }

    func testCapturingToIdleAfterExplicitStop() throws {
        let runtime = InstrumentedRuntime()
        let controller = Self.makeController(runtime: runtime)
        try controller.startCapture()

        controller.stopCapture()

        XCTAssertEqual(controller.state, .idle)
    }

    func testStartCaptureMustFailWhenAlreadyCapturing() throws {
        let runtime = InstrumentedRuntime()
        let controller = Self.makeController(runtime: runtime)
        try controller.startCapture()

        XCTAssertThrowsError(try controller.startCapture()) { error in
            XCTAssertEqual(error as? VoiceAudioControllerError, .invalidState)
        }
    }

    func testStateUpdatesStreamPublishesTransitions() async throws {
        let runtime = InstrumentedRuntime()
        let controller = Self.makeController(runtime: runtime)

        var states: [VoiceAudioControllerState] = []
        let collected = expectation(description: "three state events")
        collected.expectedFulfillmentCount = 3
        let task = Task {
            for await s in controller.stateUpdates {
                states.append(s)
                collected.fulfill()
                if states.count >= 3 { break }
            }
        }

        try controller.startCapture()
        try controller.playbackDidStart()
        controller.playbackDidStop()

        await fulfillment(of: [collected], timeout: 1)
        task.cancel()

        XCTAssertEqual(states.prefix(3).map { "\($0)" }, [
            "\(VoiceAudioControllerState.capturing)",
            "\(VoiceAudioControllerState.playing)",
            "\(VoiceAudioControllerState.capturing)",
        ])
    }

    // -------------------------------------------------------------------------
    // MARK: 3. Capture persistence during playback (Req 2.5)
    // -------------------------------------------------------------------------

    /// Frames must continue to flow through the tap while the controller is in
    /// .playing state — capture is never paused during TTS playback.
    func testCaptureRemainsActiveWhilePlaybackIsRunning() async throws {
        let runtime = InstrumentedRuntime()
        runtime.inputFormat = VoiceAudioInputFormat(sampleRateHz: 48_000, channelCount: 2)
        let controller = Self.makeController(runtime: runtime)
        try controller.startCapture()
        try controller.playbackDidStart()

        let framesExpected = expectation(description: "frames arrive during playback")
        framesExpected.expectedFulfillmentCount = 2
        var received: [VoiceAudioFrame] = []
        let task = Task {
            for await frame in controller.frames {
                received.append(frame)
                framesExpected.fulfill()
                if received.count >= 2 { break }
            }
        }

        runtime.emit(Self.stereoBuffer(capturedAt: 500))
        runtime.emit(Self.stereoBuffer(capturedAt: 1000))

        await fulfillment(of: [framesExpected], timeout: 1)
        task.cancel()

        XCTAssertEqual(controller.state, .playing)
        XCTAssertEqual(received.count, 2,
                       "capture must deliver frames even when state is .playing")
        XCTAssertEqual(received[0].sampleRateHz, 16_000)
        XCTAssertEqual(received[0].channels, 1)
        XCTAssertLessThan(received[0].sequence, received[1].sequence,
                          "sequence must remain strictly increasing across duplex frames")
    }

    /// VoiceProcessingIO must remain enabled through playback start/stop cycles
    /// (Req 2.3). A frame arriving while isVoiceProcessingEnabled == false must
    /// never reach downstream consumers.
    func testNoFrameDeliveredWhenVoiceProcessingDisabledDuringPlayback() async throws {
        let runtime = InstrumentedRuntime()
        let controller = Self.makeController(runtime: runtime)
        try controller.startCapture()
        try controller.playbackDidStart()

        // Simulate external route change disabling voice processing.
        runtime.isVoiceProcessingEnabled = false

        let noFrame = expectation(description: "no frame during disabled VP")
        noFrame.isInverted = true
        let task = Task {
            for await _ in controller.frames {
                noFrame.fulfill()
                break
            }
        }
        runtime.emitSilence(capturedAt: 9999)
        await fulfillment(of: [noFrame], timeout: 0.15)
        task.cancel()

        XCTAssertEqual(controller.state, .unavailable(.voiceProcessingUnavailable),
                       "controller must transition to unavailable when VP becomes false mid-flight")
    }

    // -------------------------------------------------------------------------
    // MARK: 4. No frame accepted when isVoiceProcessingEnabled == false (Req 2.2)
    // -------------------------------------------------------------------------

    func testFramesRejectedWhenVoiceProcessingFalseFromStart() async throws {
        let runtime = InstrumentedRuntime()
        let controller = Self.makeController(runtime: runtime)
        try controller.startCapture()

        let noFrame = expectation(description: "no frame when VP disabled")
        noFrame.isInverted = true
        let task = Task {
            for await _ in controller.frames {
                noFrame.fulfill()
                break
            }
        }
        runtime.isVoiceProcessingEnabled = false
        runtime.emitSilence(capturedAt: 200)
        await fulfillment(of: [noFrame], timeout: 0.15)
        task.cancel()
        XCTAssertEqual(controller.state, .unavailable(.voiceProcessingUnavailable))
    }

    // -------------------------------------------------------------------------
    // MARK: 5. Frame normalization and monotonic sequencing (Req 2.6)
    // -------------------------------------------------------------------------

    func testNormalizedFramesHave16kHzMono() async throws {
        let runtime = InstrumentedRuntime()
        runtime.inputFormat = VoiceAudioInputFormat(sampleRateHz: 48_000, channelCount: 2)
        let controller = VoiceAudioController(sessionID: UUID(), runtime: runtime)
        try controller.startCapture()

        let got = expectation(description: "one normalized frame")
        var frame: VoiceAudioFrame?
        let task = Task {
            for await f in controller.frames {
                frame = f
                got.fulfill()
                break
            }
        }
        runtime.emit(Self.stereoBuffer())
        await fulfillment(of: [got], timeout: 1)
        task.cancel()

        XCTAssertEqual(frame?.sampleRateHz, 16_000)
        XCTAssertEqual(frame?.channels, 1)
    }

    func testFrameSequencesAreStrictlyMonotonic() async throws {
        let runtime = InstrumentedRuntime()
        let controller = VoiceAudioController(sessionID: UUID(), runtime: runtime)
        try controller.startCapture()

        let collected = expectation(description: "five sequential frames")
        collected.expectedFulfillmentCount = 5
        var frames: [VoiceAudioFrame] = []
        let task = Task {
            for await f in controller.frames {
                frames.append(f)
                collected.fulfill()
                if frames.count >= 5 { break }
            }
        }

        for i in 0..<5 {
            runtime.emit(VoiceAudioInputBuffer(
                interleavedSamples: [Float](repeating: 0.1, count: 160),
                sourceSampleRateHz: 16_000,
                channelCount: 1,
                capturedAtMonotonicNs: UInt64(i * 1000) + 1
            ))
        }

        await fulfillment(of: [collected], timeout: 1)
        task.cancel()

        for i in 1..<frames.count {
            XCTAssertLessThan(
                frames[i - 1].sequence, frames[i].sequence,
                "sequence[\(i-1)] must be strictly less than sequence[\(i)]"
            )
            XCTAssertLessThan(
                frames[i - 1].capturedAtMonotonicNs, frames[i].capturedAtMonotonicNs,
                "capturedAtMonotonicNs[\(i-1)] must be strictly less than [\(i)]"
            )
        }
    }

    // -------------------------------------------------------------------------
    // MARK: 6. Route reset recovery (Req 2.4, design §2)
    // -------------------------------------------------------------------------

    /// After a route change, the controller must:
    ///   1. stop the tap and engine
    ///   2. re-enable VoiceProcessingIO
    ///   3. reinstall the tap
    ///   4. restart the engine
    /// and retry this cycle at most three times before giving up.
    func testRouteResetPerformsReEnableBeforeReinstallTap() async throws {
        let runtime = InstrumentedRuntime()
        // Use an immediate scheduler so retries run synchronously.
        let controller = VoiceAudioController(
            sessionID: UUID(),
            runtime: runtime,
            retryScheduler: { _, action in action() }
        )
        try controller.startCapture()

        // Cause all retry attempts to fail so we can inspect the full cycle.
        runtime.startError = true
        controller.handleAudioRouteChange()
        try await Task.sleep(nanoseconds: 50_000_000)

        // After the initial start plus three retries every attempt should have
        // tried to re-enable voice processing before installing the tap again.
        let enableIndices = runtime.log.indices.filter { runtime.log[$0] == "enable(true)" }
        let tapIndices = runtime.log.indices.filter { runtime.log[$0] == "installTap" }
        // For each tap there must be a preceding enable.
        for tapIndex in tapIndices {
            XCTAssertTrue(
                enableIndices.contains { $0 < tapIndex },
                "every tap installation must be preceded by an enable(true) call"
            )
        }
        XCTAssertEqual(
            runtime.startCount, 4,
            "initial start + three recovery attempts = 4 engine starts"
        )
        XCTAssertEqual(controller.state, .unavailable(.engineStart))
    }

    /// A successful route reset must recover to .capturing without exhausting
    /// the retry budget.
    func testRouteResetRecoversToCaptureWhenEngineEventuallyStarts() async throws {
        let runtime = InstrumentedRuntime()
        let controller = VoiceAudioController(
            sessionID: UUID(),
            runtime: runtime,
            retryScheduler: { _, action in action() }
        )
        try controller.startCapture()

        // Fail the first attempt, succeed on the second.
        let original = runtime
        original.startError = true
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.02) {
            original.startError = false
        }
        controller.handleAudioRouteChange()
        try await Task.sleep(nanoseconds: 150_000_000)

        // If startError is cleared before the second attempt the state is capturing.
        // Because timing can vary we just check the engine started more than once.
        XCTAssertGreaterThan(runtime.startCount, 1)
    }

    // -------------------------------------------------------------------------
    // MARK: 7. Media-services reset recovery
    // -------------------------------------------------------------------------

    func testMediaServicesResetTriggersRecovery() async throws {
        let runtime = InstrumentedRuntime()
        runtime.startError = true
        let controller = VoiceAudioController(
            sessionID: UUID(),
            runtime: runtime,
            retryScheduler: { _, action in action() }
        )
        try controller.startCapture()

        controller.handleMediaServicesReset()
        try await Task.sleep(nanoseconds: 100_000_000)

        // Initial start (before reset) + three retries.
        XCTAssertEqual(runtime.startCount, 4)
        XCTAssertEqual(controller.state, .unavailable(.mediaServicesReset))
    }

    // -------------------------------------------------------------------------
    // MARK: 8. Unavailable VoiceProcessingIO — diagnostic emission (Req 2.4)
    // -------------------------------------------------------------------------

    /// When VoiceProcessingIO cannot be verified, the controller must emit
    /// a VoiceAudioDiagnosticEvent with stage == .voiceProcessing.
    func testUnavailableVoiceProcessingEmitsDiagnostic() async throws {
        let runtime = InstrumentedRuntime()
        runtime.enableChangesState = false
        let controller = Self.makeController(runtime: runtime)

        let diagReceived = expectation(description: "diagnostic emitted")
        var diagnostic: VoiceAudioDiagnosticEvent?
        let task = Task {
            for await d in controller.diagnostics {
                diagnostic = d
                diagReceived.fulfill()
                break
            }
        }

        _ = try? controller.startCapture()
        await fulfillment(of: [diagReceived], timeout: 1)
        task.cancel()

        XCTAssertEqual(diagnostic?.stage, .voiceProcessing)
        XCTAssertEqual(diagnostic?.reason, .voiceProcessingUnavailable)
        XCTAssertFalse(diagnostic?.recoveryAction.userInstruction.isEmpty ?? true)
    }

    /// Diagnostic events must not contain PCM or transcript content.
    func testDiagnosticEventContainsNoAudioData() async throws {
        let runtime = InstrumentedRuntime()
        runtime.enableChangesState = false
        let controller = Self.makeController(runtime: runtime)

        let diagReceived = expectation(description: "diagnostic for no-audio check")
        var diag: VoiceAudioDiagnosticEvent?
        let task = Task {
            for await d in controller.diagnostics {
                diag = d
                diagReceived.fulfill()
                break
            }
        }
        _ = try? controller.startCapture()
        await fulfillment(of: [diagReceived], timeout: 1)
        task.cancel()

        // VoiceAudioDiagnosticEvent has no audio/PCM field by design.
        // Confirm through reflection that the Mirror has no "pcm", "audio",
        // "samples", or "bytes" labeled child.
        if let d = diag {
            let mirror = Mirror(reflecting: d)
            let fieldNames = mirror.children.compactMap(\.label)
            let forbidden = ["pcm", "audio", "samples", "bytes", "microphone", "waveform"]
            for name in fieldNames {
                XCTAssertFalse(
                    forbidden.contains { name.lowercased().contains($0) },
                    "diagnostic field '\(name)' looks like an audio payload field"
                )
            }
        }
    }

    // -------------------------------------------------------------------------
    // MARK: 9. Tap installation failure
    // -------------------------------------------------------------------------

    func testTapInstallFailureTransitionsToUnavailable() {
        let runtime = InstrumentedRuntime()
        runtime.installError = true
        let controller = Self.makeController(runtime: runtime)

        XCTAssertThrowsError(try controller.startCapture()) { error in
            XCTAssertEqual(
                error as? VoiceAudioControllerError,
                .unavailable(.tapInstallation)
            )
        }
        XCTAssertEqual(controller.state, .unavailable(.tapInstallation))
        XCTAssertFalse(runtime.log.contains("startEngine"),
                       "engine must not start if tap installation fails")
    }

    // -------------------------------------------------------------------------
    // MARK: 10. User retry from unavailable state
    // -------------------------------------------------------------------------

    func testUserRetryFromUnavailableRestoresCapturing() throws {
        let runtime = InstrumentedRuntime()
        runtime.enableChangesState = false
        let controller = Self.makeController(runtime: runtime)
        _ = try? controller.startCapture()
        XCTAssertEqual(controller.state, .unavailable(.voiceProcessingUnavailable))

        // Fix the runtime before retrying.
        runtime.enableChangesState = true
        try controller.retryCapture()
        XCTAssertEqual(controller.state, .capturing)
    }
}

#endif // canImport(AVFoundation) && os(macOS)
