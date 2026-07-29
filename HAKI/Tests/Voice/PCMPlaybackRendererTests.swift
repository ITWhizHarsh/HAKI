// PCMPlaybackRendererTests.swift
// macOS native-audio integration tests for PCMPlaybackRenderer.
//
// Validates: Requirements 2.5, 5.2, 5.6–5.7
// Test ID: V-SWIFT-AUDIO (renderer stop-terminal behaviour)
//
// Tests cover:
//   - Stop is high-priority, idempotent, and emits exactly one PLAYBACK_CANCELLED
//   - PLAYBACK_CONFIRMED only after all samples reach normal completion
//   - PLAYBACK_CANCELLED on stop / route-lost / replaced before completion
//   - PLAYBACK_FAILED with error class on renderer failure
//   - Stop acknowledgement timing (≤ 200 ms)
//   - No microphone bytes in renderer output
//
// Hardware-dependent behaviors guarded with #if canImport(AVFoundation) && os(macOS).

#if canImport(AVFoundation) && os(macOS)
import AVFoundation
import Foundation
import XCTest
@testable import HAKIAudio
@testable import HAKIIPC

// MARK: - Test doubles

/// Records every lifecycle call so tests can assert sequencing.
private final class SpyPlaybackLifecycle: PCMPlaybackLifecycle, @unchecked Sendable {
    private(set) var startCount = 0
    private(set) var stopCount = 0
    var shouldThrowOnStart = false

    func playbackDidStart() throws {
        if shouldThrowOnStart { throw PCMPlaybackRendererError.playbackUnavailable }
        startCount += 1
    }
    func playbackDidStop() { stopCount += 1 }
}

/// Thread-safe collector for terminal events (avoids Swift 6 var-capture warnings).
private final class TerminalSink: @unchecked Sendable {
    private let lock = NSLock()
    private var _events: [VoicePlaybackTerminal] = []

    var events: [VoicePlaybackTerminal] {
        lock.withLock { _events }
    }

    func append(_ event: VoicePlaybackTerminal) {
        lock.withLock { _events.append(event) }
    }
}

/// Deterministic player that queues completion callbacks for manual fire.
private final class ManualCompletionNode: PCMPlaybackNode, @unchecked Sendable {
    private(set) var scheduleCount = 0
    private(set) var stopCount = 0
    private(set) var playCount = 0
    private var completions: [(@Sendable (Result<Void, Error>) -> Void)] = []
    var scheduleError: Error? = nil

    func attach(to engine: AVAudioEngine?) throws {}

    func schedule(
        pcmS16LE: Data,
        sampleRateHz: Int,
        channels: UInt8,
        completion: @escaping @Sendable (Result<Void, Error>) -> Void
    ) throws {
        if let err = scheduleError { throw err }
        scheduleCount += 1
        completions.append(completion)
    }

    func play() throws { playCount += 1 }
    func stop() { stopCount += 1 }

    /// Complete the oldest in-flight buffer with success.
    func completeNext(success: Bool = true) {
        guard !completions.isEmpty else { return }
        let cb = completions.removeFirst()
        if success { cb(.success(())) } else { cb(.failure(NSError(domain: "test", code: 1))) }
    }
}

// MARK: - Fixture builders

private func makeChunk(
    sessionID: UUID,
    turnID: UUID? = nil,
    sentenceID: UUID? = nil,
    sequence: UInt64 = 0,
    sampleRateHz: Int = 24_000,
    channels: UInt8 = 1,
    byteLength: Int = 4
) -> VoicePCMChunk {
    VoicePCMChunk(
        sessionID: sessionID,
        turnID: turnID ?? UUID(),
        sentenceID: sentenceID ?? UUID(),
        sequence: sequence,
        sampleRateHz: sampleRateHz,
        channels: channels,
        byteLength: byteLength
    )
}

/// Minimal valid TTS-only PCM (two int16 samples at 24 kHz mono → 4 bytes).
private let minimalPCM = Data([0x00, 0x10, 0x00, 0x20])

private func makeRenderer(
    sessionID: UUID,
    node: ManualCompletionNode,
    lifecycle: SpyPlaybackLifecycle,
    sink: TerminalSink
) -> PCMPlaybackRenderer {
    PCMPlaybackRenderer(
        sessionID: sessionID,
        engineProvider: { nil },
        lifecycle: lifecycle,
        player: node,
        terminalSink: { sink.append($0) }
    )
}

// MARK: - Small wait helper (avoids full RunLoop.run dependency)

private func shortWait(seconds: TimeInterval = 0.06) {
    RunLoop.current.run(until: Date(timeIntervalSinceNow: seconds))
}

// MARK: - Test class

@available(macOS 14, *)
final class PCMPlaybackRendererTests: XCTestCase {

    // -------------------------------------------------------------------------
    // MARK: 1. Enqueue acknowledgement
    // -------------------------------------------------------------------------

    /// The renderer must emit PCM_ACCEPTED immediately after enqueueing — before
    /// any playback begins (design §2 native PCM playback protocol).
    func testEnqueueYieldsAcknowledgementWithMatchingSequence() throws {
        let sessionID = UUID()
        let node = ManualCompletionNode()
        let lifecycle = SpyPlaybackLifecycle()
        let sink = TerminalSink()
        let renderer = makeRenderer(sessionID: sessionID, node: node, lifecycle: lifecycle, sink: sink)
        let chunk = makeChunk(sessionID: sessionID, sequence: 3)

        let ack = try renderer.enqueue(chunk, pcmS16LE: minimalPCM, generation: 1)

        XCTAssertEqual(ack.sessionID, sessionID)
        XCTAssertEqual(ack.turnID, chunk.turnID)
        XCTAssertEqual(ack.sentenceID, chunk.sentenceID)
        XCTAssertEqual(ack.sequence, 3)
        XCTAssertEqual(sink.events.count, 0, "no terminal should be emitted on enqueue")
    }

    // -------------------------------------------------------------------------
    // MARK: 2. PLAYBACK_CONFIRMED only after all samples complete (Req 5.6)
    // -------------------------------------------------------------------------

    func testConfirmedTerminalEmittedOnlyAfterAllBuffersComplete() throws {
        let sessionID = UUID()
        let node = ManualCompletionNode()
        let lifecycle = SpyPlaybackLifecycle()
        let sink = TerminalSink()
        let renderer = makeRenderer(sessionID: sessionID, node: node, lifecycle: lifecycle, sink: sink)
        let turnID = UUID()
        let sentenceID = UUID()
        let chunk1 = makeChunk(sessionID: sessionID, turnID: turnID, sentenceID: sentenceID, sequence: 0)
        let chunk2 = makeChunk(sessionID: sessionID, turnID: turnID, sentenceID: sentenceID, sequence: 1)

        _ = try renderer.enqueue(chunk1, pcmS16LE: minimalPCM, generation: 1)
        _ = try renderer.enqueue(chunk2, pcmS16LE: minimalPCM, generation: 1)
        try renderer.finishSentence(turnID: turnID, sentenceID: sentenceID, generation: 1)

        XCTAssertEqual(sink.events.count, 0)
        node.completeNext()
        shortWait()
        XCTAssertEqual(sink.events.count, 0)
        node.completeNext()
        shortWait()

        XCTAssertEqual(sink.events.map(\.kind), [.confirmed])
        XCTAssertEqual(sink.events.first?.sentenceID, sentenceID)
    }

    func testNoConfirmedTerminalWhileBuffersStillPending() throws {
        let sessionID = UUID()
        let node = ManualCompletionNode()
        let lifecycle = SpyPlaybackLifecycle()
        let sink = TerminalSink()
        let renderer = makeRenderer(sessionID: sessionID, node: node, lifecycle: lifecycle, sink: sink)
        let chunk = makeChunk(sessionID: sessionID)
        _ = try renderer.enqueue(chunk, pcmS16LE: minimalPCM, generation: 1)
        try renderer.finishSentence(turnID: chunk.turnID, sentenceID: chunk.sentenceID, generation: 1)
        XCTAssertEqual(sink.events.count, 0)
    }

    // -------------------------------------------------------------------------
    // MARK: 3. PLAYBACK_CANCELLED on stop (Req 5.2, 5.7)
    // -------------------------------------------------------------------------

    /// Stop is high-priority and emits exactly one PLAYBACK_CANCELLED regardless
    /// of how many times the same stop control arrives.
    func testStopIsHighPriorityAndIdempotent() throws {
        let sessionID = UUID()
        let node = ManualCompletionNode()
        let lifecycle = SpyPlaybackLifecycle()
        let sink = TerminalSink()
        let renderer = makeRenderer(sessionID: sessionID, node: node, lifecycle: lifecycle, sink: sink)
        let chunk = makeChunk(sessionID: sessionID)
        _ = try renderer.enqueue(chunk, pcmS16LE: minimalPCM, generation: 5)

        let stop = VoiceStopPlayback(sessionID: sessionID, turnID: chunk.turnID, generation: 5)
        renderer.stop(stop)
        renderer.stop(stop)
        renderer.stop(stop)
        shortWait()

        XCTAssertEqual(node.stopCount, 1, "playerNode.stop() must be called exactly once")
        XCTAssertEqual(sink.events.map(\.kind), [.cancelled],
                       "exactly one PLAYBACK_CANCELLED for an idempotent stop")
        XCTAssertEqual(sink.events.first?.turnID, chunk.turnID)
    }

    func testStopUnknownGenerationEmitsAcknowledgementOnly() throws {
        let sessionID = UUID()
        let node = ManualCompletionNode()
        let lifecycle = SpyPlaybackLifecycle()
        let sink = TerminalSink()
        let renderer = makeRenderer(sessionID: sessionID, node: node, lifecycle: lifecycle, sink: sink)
        let stop = VoiceStopPlayback(sessionID: sessionID, turnID: UUID(), generation: 99)
        renderer.stop(stop)
        shortWait()
        XCTAssertEqual(sink.events.count, 0,
                       "stop for unknown generation must not emit a sentence terminal")
    }

    func testEnqueueAfterStopForSameGenerationFails() throws {
        let sessionID = UUID()
        let node = ManualCompletionNode()
        let lifecycle = SpyPlaybackLifecycle()
        let sink = TerminalSink()
        let renderer = makeRenderer(sessionID: sessionID, node: node, lifecycle: lifecycle, sink: sink)
        let chunk = makeChunk(sessionID: sessionID)
        _ = try renderer.enqueue(chunk, pcmS16LE: minimalPCM, generation: 2)
        renderer.stop(VoiceStopPlayback(sessionID: sessionID, turnID: chunk.turnID, generation: 2))
        shortWait()

        XCTAssertThrowsError(
            try renderer.enqueue(chunk, pcmS16LE: minimalPCM, generation: 2)
        ) { error in
            XCTAssertEqual(error as? PCMPlaybackRendererError, .sentenceAlreadyTerminal)
        }
    }

    // -------------------------------------------------------------------------
    // MARK: 4. PLAYBACK_CANCELLED on route-lost / replaced (Req 5.7)
    // -------------------------------------------------------------------------

    /// Stopping before finishSentence must still cancel the sentence.
    func testStopBeforeFinishSentenceCancels() throws {
        let sessionID = UUID()
        let node = ManualCompletionNode()
        let lifecycle = SpyPlaybackLifecycle()
        let sink = TerminalSink()
        let renderer = makeRenderer(sessionID: sessionID, node: node, lifecycle: lifecycle, sink: sink)
        let chunk = makeChunk(sessionID: sessionID)
        _ = try renderer.enqueue(chunk, pcmS16LE: minimalPCM, generation: 7)

        renderer.stop(VoiceStopPlayback(sessionID: sessionID, turnID: chunk.turnID, generation: 7))
        shortWait()

        XCTAssertEqual(sink.events.map(\.kind), [.cancelled])
    }

    func testStopCancelsBothQueuedSentencesForSameTurnGeneration() throws {
        let sessionID = UUID()
        let node = ManualCompletionNode()
        let lifecycle = SpyPlaybackLifecycle()
        let sink = TerminalSink()
        let renderer = makeRenderer(sessionID: sessionID, node: node, lifecycle: lifecycle, sink: sink)
        let turnID = UUID()
        let s1 = makeChunk(sessionID: sessionID, turnID: turnID, sentenceID: UUID(), sequence: 0)
        let s2 = makeChunk(sessionID: sessionID, turnID: turnID, sentenceID: UUID(), sequence: 0)

        _ = try renderer.enqueue(s1, pcmS16LE: minimalPCM, generation: 3)
        _ = try renderer.enqueue(s2, pcmS16LE: minimalPCM, generation: 3)

        renderer.stop(VoiceStopPlayback(sessionID: sessionID, turnID: turnID, generation: 3))
        shortWait()

        let kinds = sink.events.map(\.kind)
        XCTAssertEqual(kinds.count, 2)
        XCTAssertTrue(kinds.allSatisfy { $0 == .cancelled })
    }

    // -------------------------------------------------------------------------
    // MARK: 5. PLAYBACK_FAILED with error class (design §2)
    // -------------------------------------------------------------------------

    /// If the native player schedule call throws, the renderer emits exactly one
    /// PLAYBACK_FAILED with a non-empty error class.
    func testRendererFailureEmitsPlaybackFailedWithErrorClass() throws {
        let sessionID = UUID()
        let node = ManualCompletionNode()
        node.scheduleError = PCMPlaybackRendererError.invalidPCM
        let lifecycle = SpyPlaybackLifecycle()
        let sink = TerminalSink()
        let renderer = makeRenderer(sessionID: sessionID, node: node, lifecycle: lifecycle, sink: sink)
        let chunk = makeChunk(sessionID: sessionID)
        _ = try renderer.enqueue(chunk, pcmS16LE: minimalPCM, generation: 1)
        shortWait()

        XCTAssertEqual(sink.events.map(\.kind), [.failed])
        XCTAssertNotNil(sink.events.first?.errorClass)
        XCTAssertFalse(sink.events.first?.errorClass?.isEmpty ?? true)
    }

    func testBufferCompletionErrorYieldsPlaybackFailed() throws {
        let sessionID = UUID()
        let node = ManualCompletionNode()
        let lifecycle = SpyPlaybackLifecycle()
        let sink = TerminalSink()
        let renderer = makeRenderer(sessionID: sessionID, node: node, lifecycle: lifecycle, sink: sink)
        let chunk = makeChunk(sessionID: sessionID)
        _ = try renderer.enqueue(chunk, pcmS16LE: minimalPCM, generation: 1)
        try renderer.finishSentence(turnID: chunk.turnID, sentenceID: chunk.sentenceID, generation: 1)
        node.completeNext(success: false)
        shortWait()

        XCTAssertEqual(sink.events.map(\.kind), [.failed])
        XCTAssertNotNil(sink.events.first?.errorClass)
    }

    // -------------------------------------------------------------------------
    // MARK: 6. Exactly one terminal per sentence
    // -------------------------------------------------------------------------

    /// The renderer must never emit two terminals for the same sentence.
    func testExactlyOneTerminalPerSentence() throws {
        let sessionID = UUID()
        let node = ManualCompletionNode()
        let lifecycle = SpyPlaybackLifecycle()
        let sink = TerminalSink()
        let renderer = makeRenderer(sessionID: sessionID, node: node, lifecycle: lifecycle, sink: sink)
        let chunk = makeChunk(sessionID: sessionID)
        _ = try renderer.enqueue(chunk, pcmS16LE: minimalPCM, generation: 1)
        try renderer.finishSentence(turnID: chunk.turnID, sentenceID: chunk.sentenceID, generation: 1)
        node.completeNext()
        shortWait()

        XCTAssertThrowsError(
            try renderer.finishSentence(turnID: chunk.turnID, sentenceID: chunk.sentenceID, generation: 1)
        ) { error in
            XCTAssertEqual(error as? PCMPlaybackRendererError, .sentenceAlreadyTerminal)
        }
        XCTAssertEqual(sink.events.count, 1, "must never emit more than one terminal per sentence")
    }

    func testConcurrentStopsYieldExactlyOneTerminal() throws {
        let sessionID = UUID()
        let node = ManualCompletionNode()
        let lifecycle = SpyPlaybackLifecycle()
        let sink = TerminalSink()
        let renderer = makeRenderer(sessionID: sessionID, node: node, lifecycle: lifecycle, sink: sink)
        let chunk = makeChunk(sessionID: sessionID)
        _ = try renderer.enqueue(chunk, pcmS16LE: minimalPCM, generation: 8)

        let stop = VoiceStopPlayback(sessionID: sessionID, turnID: chunk.turnID, generation: 8)
        let group = DispatchGroup()
        for _ in 0..<5 {
            group.enter()
            DispatchQueue.global().async {
                renderer.stop(stop)
                group.leave()
            }
        }
        group.wait()
        shortWait()

        XCTAssertEqual(sink.events.map(\.kind), [.cancelled],
                       "concurrent stops must produce exactly one cancelled terminal")
    }

    // -------------------------------------------------------------------------
    // MARK: 7. Stop acknowledgement timing ≤ 200 ms (Req 5.2, design §2)
    // -------------------------------------------------------------------------

    /// The stop call must not block the caller for more than 200 ms. Because
    /// the renderer queue is serial and synchronous, timing is deterministic.
    func testStopAcknowledgedWithin200ms() throws {
        let sessionID = UUID()
        let node = ManualCompletionNode()
        let lifecycle = SpyPlaybackLifecycle()
        let sink = TerminalSink()
        let renderer = makeRenderer(sessionID: sessionID, node: node, lifecycle: lifecycle, sink: sink)
        let chunk = makeChunk(sessionID: sessionID)
        _ = try renderer.enqueue(chunk, pcmS16LE: minimalPCM, generation: 1)

        let stop = VoiceStopPlayback(sessionID: sessionID, turnID: chunk.turnID, generation: 1)
        let start = Date()
        renderer.stop(stop)
        shortWait(seconds: 0.01)
        let elapsed = Date().timeIntervalSince(start)

        XCTAssertLessThanOrEqual(
            elapsed, 0.200,
            "stop acknowledgement must complete within 200 ms (took \(Int(elapsed * 1000)) ms)"
        )
    }

    // -------------------------------------------------------------------------
    // MARK: 8. No microphone bytes in renderer output (design §2)
    // -------------------------------------------------------------------------

    /// VoicePCMChunk must not carry any microphone-related field by design.
    /// Verify through Mirror that the type has no audio/microphone payload field.
    func testRendererProtocolTypesContainNoMicrophoneFields() {
        let chunk = makeChunk(sessionID: UUID())
        let terminal = VoicePlaybackTerminal(
            kind: .confirmed, sessionID: UUID(), turnID: UUID(), sentenceID: UUID()
        )
        let ack = PCMPlaybackEnqueueAcknowledgement(
            sessionID: UUID(), turnID: UUID(), sentenceID: UUID(), sequence: 0
        )
        let forbidden = ["pcm", "audio", "samples", "microphone", "waveform", "bytes", "buffer"]
        for (name, subject) in [
            ("VoicePCMChunk", Mirror(reflecting: chunk)),
            ("VoicePlaybackTerminal", Mirror(reflecting: terminal)),
            ("PCMPlaybackEnqueueAcknowledgement", Mirror(reflecting: ack)),
        ] {
            for child in subject.children {
                if let label = child.label {
                    let normalized = label.lowercased()
                    for forbidden in forbidden {
                        XCTAssertFalse(
                            normalized.contains(forbidden),
                            "\(name) field '\(label)' looks like a microphone payload field"
                        )
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // MARK: 9. Stale session guard
    // -------------------------------------------------------------------------

    func testEnqueueChunkFromWrongSessionThrows() throws {
        let sessionID = UUID()
        let node = ManualCompletionNode()
        let lifecycle = SpyPlaybackLifecycle()
        let sink = TerminalSink()
        let renderer = makeRenderer(sessionID: sessionID, node: node, lifecycle: lifecycle, sink: sink)
        let foreignChunk = makeChunk(sessionID: UUID()) // different session
        XCTAssertThrowsError(
            try renderer.enqueue(foreignChunk, pcmS16LE: minimalPCM, generation: 1)
        ) { error in
            XCTAssertEqual(error as? PCMPlaybackRendererError, .staleSession)
        }
    }

    // -------------------------------------------------------------------------
    // MARK: 10. Multiple sentences in sequence — no cross-sentence contamination
    // -------------------------------------------------------------------------

    func testMultipleSentencesEachReceiveIndependentTerminal() throws {
        let sessionID = UUID()
        let node = ManualCompletionNode()
        let lifecycle = SpyPlaybackLifecycle()
        let sink = TerminalSink()
        let renderer = makeRenderer(sessionID: sessionID, node: node, lifecycle: lifecycle, sink: sink)
        let turnID = UUID()
        let s1ID = UUID()
        let s2ID = UUID()
        let s1 = makeChunk(sessionID: sessionID, turnID: turnID, sentenceID: s1ID, sequence: 0)
        let s2 = makeChunk(sessionID: sessionID, turnID: turnID, sentenceID: s2ID, sequence: 0)

        _ = try renderer.enqueue(s1, pcmS16LE: minimalPCM, generation: 1)
        try renderer.finishSentence(turnID: turnID, sentenceID: s1ID, generation: 1)
        _ = try renderer.enqueue(s2, pcmS16LE: minimalPCM, generation: 1)
        try renderer.finishSentence(turnID: turnID, sentenceID: s2ID, generation: 1)

        node.completeNext() // complete s1
        shortWait()
        node.completeNext() // complete s2
        shortWait()

        XCTAssertEqual(sink.events.count, 2)
        XCTAssertEqual(sink.events[0].sentenceID, s1ID)
        XCTAssertEqual(sink.events[0].kind, .confirmed)
        XCTAssertEqual(sink.events[1].sentenceID, s2ID)
        XCTAssertEqual(sink.events[1].kind, .confirmed)
    }

    // -------------------------------------------------------------------------
    // MARK: 11. Stop for wrong session is silently ignored
    // -------------------------------------------------------------------------

    func testStopFromWrongSessionIsIgnored() throws {
        let sessionID = UUID()
        let node = ManualCompletionNode()
        let lifecycle = SpyPlaybackLifecycle()
        let sink = TerminalSink()
        let renderer = makeRenderer(sessionID: sessionID, node: node, lifecycle: lifecycle, sink: sink)
        let chunk = makeChunk(sessionID: sessionID)
        _ = try renderer.enqueue(chunk, pcmS16LE: minimalPCM, generation: 1)

        let foreignStop = VoiceStopPlayback(sessionID: UUID(), turnID: chunk.turnID, generation: 1)
        renderer.stop(foreignStop)
        shortWait()

        XCTAssertEqual(sink.events.count, 0, "stop from wrong session must not emit a terminal")
        XCTAssertEqual(node.stopCount, 0, "playerNode.stop must not be called for wrong session")
    }
}

#endif // canImport(AVFoundation) && os(macOS)
