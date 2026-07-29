// PCMPlaybackRendererTests.swift
// Focused coverage for realtime-local-voice-agent task 4.2.
// Validates: Requirements 2.5, 5.2, 5.6–5.7

#if canImport(XCTest)
import AVFoundation
import Foundation
import XCTest
@testable import HAKIAudio
@testable import HAKIIPC

private final class FixturePlaybackLifecycle: PCMPlaybackLifecycle, @unchecked Sendable {
    private(set) var starts = 0
    private(set) var stops = 0

    func playbackDidStart() throws { starts += 1 }
    func playbackDidStop() { stops += 1 }
}

private final class ManualPCMPlaybackNode: PCMPlaybackNode, @unchecked Sendable {
    private(set) var attached = false
    private(set) var stopCount = 0
    private var completions: [@Sendable (Result<Void, Error>) -> Void] = []

    func attach(to engine: AVAudioEngine?) throws { attached = true }

    func schedule(
        pcmS16LE: Data,
        sampleRateHz: Int,
        channels: UInt8,
        completion: @escaping @Sendable (Result<Void, Error>) -> Void
    ) throws {
        completions.append(completion)
    }

    func play() throws {}
    func stop() { stopCount += 1 }

    func completeNext() {
        completions.removeFirst()(.success(()))
    }
}

final class PCMPlaybackRendererTests: XCTestCase {
    func testAcknowledgesAfterEnqueueAndEmitsOneConfirmedTerminalAfterCompletion() throws {
        let node = ManualPCMPlaybackNode()
        let lifecycle = FixturePlaybackLifecycle()
        var terminals: [VoicePlaybackTerminal] = []
        let sessionID = UUID()
        let renderer = PCMPlaybackRenderer(
            sessionID: sessionID,
            engineProvider: { nil },
            lifecycle: lifecycle,
            player: node,
            terminalSink: { terminals.append($0) }
        )
        let chunk = chunk(sessionID: sessionID)

        let acknowledgement = try renderer.enqueue(chunk, pcmS16LE: Data([0, 0, 1, 0]), generation: 7)
        XCTAssertEqual(acknowledgement.sequence, chunk.sequence)
        XCTAssertTrue(node.attached)
        XCTAssertEqual(terminals.count, 0)

        try renderer.finishSentence(turnID: chunk.turnID, sentenceID: chunk.sentenceID, generation: 7)
        node.completeNext()
        waitForRenderer()

        XCTAssertEqual(terminals.map(\.kind), [.confirmed])
        XCTAssertEqual(terminals.first?.turnID, chunk.turnID)
        XCTAssertEqual(lifecycle.starts, 1)
        XCTAssertGreaterThanOrEqual(lifecycle.stops, 1)
    }

    func testStopIsHighPriorityIdempotentAndOnlyEmitsOneCancelledTerminal() throws {
        let node = ManualPCMPlaybackNode()
        let lifecycle = FixturePlaybackLifecycle()
        var terminals: [VoicePlaybackTerminal] = []
        let sessionID = UUID()
        let renderer = PCMPlaybackRenderer(
            sessionID: sessionID,
            engineProvider: { nil },
            lifecycle: lifecycle,
            player: node,
            terminalSink: { terminals.append($0) }
        )
        let chunk = chunk(sessionID: sessionID)
        _ = try renderer.enqueue(chunk, pcmS16LE: Data([0, 0, 1, 0]), generation: 9)

        let stop = VoiceStopPlayback(sessionID: sessionID, turnID: chunk.turnID, generation: 9)
        renderer.stop(stop)
        renderer.stop(stop)
        waitForRenderer()

        XCTAssertEqual(node.stopCount, 1)
        XCTAssertEqual(terminals.map(\.kind), [.cancelled])
        XCTAssertThrowsError(try renderer.finishSentence(
            turnID: chunk.turnID,
            sentenceID: chunk.sentenceID,
            generation: 9
        )) { error in
            XCTAssertEqual(error as? PCMPlaybackRendererError, .sentenceAlreadyTerminal)
        }
    }

    private func chunk(sessionID: UUID) -> VoicePCMChunk {
        VoicePCMChunk(
            sessionID: sessionID,
            turnID: UUID(),
            sentenceID: UUID(),
            sequence: 0,
            sampleRateHz: 24_000,
            channels: 1,
            byteLength: 4
        )
    }

    private func waitForRenderer() {
        RunLoop.current.run(until: Date().addingTimeInterval(0.05))
    }
}
#endif
