// VoiceAudioControllerTests.swift
// Focused lifecycle coverage for realtime-local-voice-agent task 4.1.

#if canImport(XCTest)
import AVFoundation
import XCTest
@testable import HAKIAudio

private final class FakeVoiceAudioRuntime: VoiceAudioRuntime, @unchecked Sendable {
    var microphoneAuthorization: AVAuthorizationStatus = .authorized
    var inputFormat = VoiceAudioInputFormat(sampleRateHz: 48_000, channelCount: 1)
    var isVoiceProcessingEnabled = false
    var enableChangesState = true
    var installError = false
    var startError = false
    var log: [String] = []
    var startCount = 0
    private var tapHandler: (@Sendable (VoiceAudioInputBuffer) -> Void)?

    func setVoiceProcessingEnabled(_ enabled: Bool) throws {
        log.append("enable")
        if enableChangesState { isVoiceProcessingEnabled = enabled }
    }

    func installInputTap(
        bufferSize: AVAudioFrameCount,
        handler: @escaping @Sendable (VoiceAudioInputBuffer) -> Void
    ) throws {
        log.append("tap")
        if installError { throw VoiceAudioUnavailableReason.tapInstallation }
        tapHandler = handler
    }

    func removeInputTap() {
        log.append("removeTap")
        tapHandler = nil
    }

    func startEngine() throws {
        log.append("start")
        startCount += 1
        if startError { throw VoiceAudioUnavailableReason.engineStart }
    }

    func stopEngine() {
        log.append("stop")
    }

    func setConfigurationChangeHandler(_ handler: (@Sendable () -> Void)?) {}

    func emit(_ input: VoiceAudioInputBuffer) {
        tapHandler?(input)
    }
}

final class VoiceAudioControllerTests: XCTestCase {
    func testStartEnablesAndVerifiesVoiceProcessingBeforeTapAndEngineStart() throws {
        let runtime = FakeVoiceAudioRuntime()
        let controller = VoiceAudioController(runtime: runtime)

        try controller.startCapture()

        XCTAssertEqual(runtime.log.prefix(3), ["enable", "tap", "start"])
        XCTAssertEqual(controller.state, .capturing)
    }

    func testVoiceProcessingVerificationFailureIsActionableAndNeverInstallsTap() {
        let runtime = FakeVoiceAudioRuntime()
        runtime.enableChangesState = false
        let controller = VoiceAudioController(runtime: runtime)

        XCTAssertThrowsError(try controller.startCapture()) { error in
            XCTAssertEqual(error as? VoiceAudioControllerError, .unavailable(.voiceProcessingUnavailable))
        }
        XCTAssertEqual(controller.state, .unavailable(.voiceProcessingUnavailable))
        XCTAssertFalse(runtime.log.contains("tap"))
        XCTAssertTrue(VoiceAudioUnavailableReason.voiceProcessingUnavailable.userMessage.contains("Retry voice capture"))
    }

    func testNormalizedFramesAreSequencedAndCapturePersistsDuringPlayback() async throws {
        let runtime = FakeVoiceAudioRuntime()
        runtime.inputFormat = VoiceAudioInputFormat(sampleRateHz: 48_000, channelCount: 2)
        let controller = VoiceAudioController(sessionID: UUID(), runtime: runtime)
        try controller.startCapture()
        try controller.playbackDidStart()

        let received = expectation(description: "two normalized frames")
        received.expectedFulfillmentCount = 2
        var frames: [VoiceAudioFrame] = []
        let collectTask = Task {
            for await frame in controller.frames {
                frames.append(frame)
                received.fulfill()
                if frames.count == 2 { break }
            }
        }

        let stereoSamples = Array(repeating: Float(0.5), count: 960 * 2)
        runtime.emit(VoiceAudioInputBuffer(
            interleavedSamples: stereoSamples,
            sourceSampleRateHz: 48_000,
            channelCount: 2,
            capturedAtMonotonicNs: 100
        ))
        runtime.emit(VoiceAudioInputBuffer(
            interleavedSamples: stereoSamples,
            sourceSampleRateHz: 48_000,
            channelCount: 2,
            capturedAtMonotonicNs: 100
        ))

        await fulfillment(of: [received], timeout: 1)
        collectTask.cancel()

        XCTAssertEqual(controller.state, .playing)
        XCTAssertEqual(frames.map(\.sampleRateHz), [16_000, 16_000])
        XCTAssertEqual(frames.map(\.channels), [1, 1])
        XCTAssertEqual(frames.map(\.pcmS16LE.count), [640, 640])
        XCTAssertLessThan(frames[0].sequence, frames[1].sequence)
        XCTAssertLessThan(frames[0].capturedAtMonotonicNs, frames[1].capturedAtMonotonicNs)
    }

    func testNoFramePassesWhenVoiceProcessingBecomesFalse() async throws {
        let runtime = FakeVoiceAudioRuntime()
        let controller = VoiceAudioController(runtime: runtime)
        try controller.startCapture()

        let noFrame = expectation(description: "no frame is delivered")
        noFrame.isInverted = true
        let collectTask = Task {
            for await _ in controller.frames {
                noFrame.fulfill()
                break
            }
        }

        runtime.isVoiceProcessingEnabled = false
        runtime.emit(VoiceAudioInputBuffer(
            interleavedSamples: [0.25, 0.25],
            sourceSampleRateHz: 16_000,
            channelCount: 1,
            capturedAtMonotonicNs: 200
        ))

        await fulfillment(of: [noFrame], timeout: 0.15)
        collectTask.cancel()
        XCTAssertEqual(controller.state, .unavailable(.voiceProcessingUnavailable))
    }

    func testRouteResetRetriesAtMostThreeTimesThenBecomesUnavailable() async throws {
        let runtime = FakeVoiceAudioRuntime()
        let controller = VoiceAudioController(
            runtime: runtime,
            retryScheduler: { _, action in action() }
        )
        try controller.startCapture()
        runtime.startError = true

        controller.handleAudioRouteChange()
        try await Task.sleep(nanoseconds: 100_000_000)

        // One successful initial start plus exactly three reset attempts.
        XCTAssertEqual(runtime.startCount, 4)
        XCTAssertEqual(controller.state, .unavailable(.engineStart))
    }
}
#endif
