// VoiceSocketClientTests.swift
// Transport contract coverage for realtime-local-voice-agent task 3.3.
// Validates: Requirements 3.3–3.6, 3.8, 5.2, 5.6–5.7

import Foundation
import XCTest
@testable import HAKIIPC

private final class FixtureVoiceSocketTransport: VoiceSocketTransport, @unchecked Sendable {
    private let lock = NSLock()
    private let incomingStream: AsyncStream<Data>
    private let incomingContinuation: AsyncStream<Data>.Continuation
    private var connectedPath: String?
    private var sentStorage: [Data] = []

    init() {
        let stream = AsyncStream<Data>.makeStream(bufferingPolicy: .unbounded)
        incomingStream = stream.stream
        incomingContinuation = stream.continuation
    }

    func connect(path: String) async throws {
        lock.withLock { connectedPath = path }
    }

    func send(_ data: Data) async throws {
        lock.withLock { sentStorage.append(data) }
    }

    func receive() async throws -> Data? {
        for await data in incomingStream { return data }
        return nil
    }

    func close() {}

    func enqueue(_ data: Data) {
        incomingContinuation.yield(data)
    }

    func finish() {
        incomingContinuation.finish()
    }

    var sent: [Data] {
        lock.withLock { sentStorage }
    }
}

private final class TransportSequence: @unchecked Sendable {
    private let lock = NSLock()
    private var transports: [FixtureVoiceSocketTransport]

    init(_ transports: [FixtureVoiceSocketTransport]) {
        self.transports = transports
    }

    func next() -> any VoiceSocketTransport {
        lock.withLock {
            precondition(!transports.isEmpty, "missing fixture transport")
            return transports.removeFirst()
        }
    }
}

private final class AsyncCallbackRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var pcm: [(VoicePCMChunk, Data)] = []
    private(set) var stops: [VoiceStopPlayback] = []

    func recordPCM(_ chunk: VoicePCMChunk, payload: Data) {
        lock.withLock { pcm.append((chunk, payload)) }
    }

    func recordStop(_ stop: VoiceStopPlayback) {
        lock.withLock { stops.append(stop) }
    }

    var pcmCount: Int { lock.withLock { pcm.count } }
    var stopCount: Int { lock.withLock { stops.count } }
}

final class VoiceSocketClientTests: XCTestCase {
    func testMirroredPythonTranscriptFixtureUsesTextOnlyStrictContract() async throws {
        let transport = FixtureVoiceSocketTransport()
        let client = try makeClient(using: [transport])
        try await client.connect()
        try await client.sendTranscript(VoiceSocketProtocolFixtures.transcriptPartial())

        let sent = transport.sent
        XCTAssertEqual(String(decoding: sent[0], as: UTF8.self), "VOICE_AUTH \(VoiceSocketProtocolFixtures.capability)\n")
        let event = try decodedJSONObject(sent[1])
        XCTAssertEqual(event["version"] as? Int, 1)
        XCTAssertEqual(event["type"] as? String, "TRANSCRIPT_EVENT")
        XCTAssertEqual(event["event_id"] as? String, VoiceSocketProtocolFixtures.partialEventID.uuidString.lowercased())
        XCTAssertEqual(event["session_id"] as? String, VoiceSocketProtocolFixtures.sessionID.uuidString.lowercased())
        XCTAssertEqual(event["turn_id"] as? String, VoiceSocketProtocolFixtures.turnID.uuidString.lowercased())
        XCTAssertEqual(event["event_seq"] as? Int, 17)
        XCTAssertEqual(event["language"] as? String, "hinglish")
        XCTAssertEqual(Set(event.keys), [
            "version", "type", "event_id", "session_id", "turn_id", "event_seq", "text", "is_final",
            "language", "capture_started_monotonic_ns", "capture_ended_monotonic_ns",
        ])
        XCTAssertFalse(event.keys.contains { key in
            ["audio", "microphone", "samples", "pcm", "bytes", "base64"].contains(where: key.lowercased().contains)
        })
    }

    func testSequenceEnforcementRejectsGapAndPostFinalEvent() async throws {
        let transport = FixtureVoiceSocketTransport()
        let client = try makeClient(using: [transport])
        try await client.connect()

        try await client.sendTranscript(VoiceSocketProtocolFixtures.transcriptPartial())
        await XCTAssertThrowsErrorAsync(try await client.sendTranscript(VoiceSocketProtocolFixtures.transcriptFinal(sequence: 19))) { error in
            XCTAssertEqual(error as? VoiceSocketClientError, .invalidTranscriptSequence)
        }
        try await client.sendTranscript(VoiceSocketProtocolFixtures.transcriptFinal())
        await XCTAssertThrowsErrorAsync(try await client.sendTranscript(VoiceSocketProtocolFixtures.transcriptFinal(sequence: 19))) { error in
            XCTAssertEqual(error as? VoiceSocketClientError, .turnAlreadyFinalized)
        }
    }

    func testDisconnectDiscardsUnacknowledgedFinalAndNeverReplaysItAfterReconnect() async throws {
        let first = FixtureVoiceSocketTransport()
        let second = FixtureVoiceSocketTransport()
        let client = try makeClient(using: [first, second], reconnectDelays: [10_000_000])
        try await client.connect()
        try await client.sendTranscript(VoiceSocketProtocolFixtures.transcriptPartial(sequence: 0))
        try await client.sendTranscript(VoiceSocketProtocolFixtures.transcriptFinal(sequence: 1))

        let discarded = expectation(description: "unfinished turn discarded")
        let observeDiscard = Task {
            for await turnID in await client.discardedTurns {
                if turnID == VoiceSocketProtocolFixtures.turnID {
                    discarded.fulfill()
                    break
                }
            }
        }
        first.finish()
        await fulfillment(of: [discarded], timeout: 1)
        observeDiscard.cancel()
        try await Task.sleep(nanoseconds: 100_000_000)

        XCTAssertEqual(second.sent.count, 1, "a reconnect sends only the capability preface")
        XCTAssertEqual(String(decoding: second.sent[0], as: UTF8.self), "VOICE_AUTH \(VoiceSocketProtocolFixtures.capability)\n")
        await XCTAssertThrowsErrorAsync(try await client.sendTranscript(VoiceSocketProtocolFixtures.transcriptFinal(sequence: 1))) { error in
            XCTAssertEqual(error as? VoiceSocketClientError, .turnDiscarded)
        }
    }

    func testPCMFramingStopIdempotenceAndExactlyOnePlaybackTerminal() async throws {
        let transport = FixtureVoiceSocketTransport()
        let recorder = AsyncCallbackRecorder()
        let client = try makeClient(using: [transport], recorder: recorder)
        try await client.connect()

        let pcmPayload = Data([0x00, 0x00, 0x01, 0x00])
        transport.enqueue(try VoiceSocketProtocolFixtures.pcmFrame(
            VoiceSocketProtocolFixtures.pcmMetadata(),
            payload: pcmPayload
        ))
        transport.enqueue(try VoiceSocketProtocolFixtures.jsonLine(VoiceSocketProtocolFixtures.stopPlayback()))
        transport.enqueue(try VoiceSocketProtocolFixtures.jsonLine(VoiceSocketProtocolFixtures.stopPlayback()))
        try await waitUntil { recorder.pcmCount == 1 && recorder.stopCount == 1 }

        let outbound = transport.sent.compactMap { try? self.decodedJSONObject($0) }
        XCTAssertTrue(outbound.contains { $0["type"] as? String == "PCM_ACCEPTED" })
        XCTAssertEqual(outbound.filter { $0["type"] as? String == "STOP_PLAYBACK_ACK" }.count, 2)

        try await client.sendPlaybackConfirmed(
            turnID: VoiceSocketProtocolFixtures.turnID,
            sentenceID: VoiceSocketProtocolFixtures.sentenceID
        )
        await XCTAssertThrowsErrorAsync(try await client.sendPlaybackCancelled(
            turnID: VoiceSocketProtocolFixtures.turnID,
            sentenceID: VoiceSocketProtocolFixtures.sentenceID
        )) { error in
            XCTAssertEqual(error as? VoiceSocketClientError, .duplicatePlaybackTerminal)
        }
    }

    func testRejectsProhibitedMicrophoneFieldInIncomingPCMMetadata() async throws {
        let transport = FixtureVoiceSocketTransport()
        let recorder = AsyncCallbackRecorder()
        let client = try makeClient(
            using: [transport],
            reconnectDelays: [1_000_000_000],
            recorder: recorder
        )
        try await client.connect()

        var prohibited = VoiceSocketProtocolFixtures.pcmMetadata()
        prohibited["samples_b64"] = "never-permitted"
        transport.enqueue(try VoiceSocketProtocolFixtures.jsonLine(prohibited))
        try await Task.sleep(nanoseconds: 50_000_000)

        XCTAssertEqual(recorder.pcmCount, 0)
        let currentState = await client.state
        XCTAssertNotEqual(currentState, VoiceSocketConnectionState.connected)
    }

    private func makeClient(
        using transports: [FixtureVoiceSocketTransport],
        reconnectDelays: [UInt64] = VoiceSocketClient.reconnectDelaysNanoseconds,
        recorder: AsyncCallbackRecorder? = nil
    ) throws -> VoiceSocketClient {
        let sequence = TransportSequence(transports)
        return try VoiceSocketClient(
            socketPath: URL(fileURLWithPath: "/tmp/haki-voice-fixture.sock"),
            sessionID: VoiceSocketProtocolFixtures.sessionID,
            sessionCapability: VoiceSocketProtocolFixtures.capability,
            transportFactory: { sequence.next() },
            reconnectDelaysNanoseconds: reconnectDelays,
            onPCMChunk: { chunk, payload in recorder?.recordPCM(chunk, payload: payload) },
            onStopPlayback: { stop in recorder?.recordStop(stop) }
        )
    }

    private func decodedJSONObject(_ data: Data) throws -> [String: Any] {
        let line = data.last == 0x0A ? data.dropLast() : data[...]
        return try XCTUnwrap(JSONSerialization.jsonObject(with: line) as? [String: Any])
    }

    private func waitUntil(
        timeoutNanoseconds: UInt64 = 1_000_000_000,
        condition: @escaping @Sendable () -> Bool
    ) async throws {
        let deadline = DispatchTime.now().uptimeNanoseconds + timeoutNanoseconds
        while !condition() {
            guard DispatchTime.now().uptimeNanoseconds < deadline else {
                XCTFail("timed out waiting for asynchronous fixture callback")
                return
            }
            try await Task.sleep(nanoseconds: 5_000_000)
        }
    }
}

private func XCTAssertThrowsErrorAsync(
    _ expression: @autoclosure () async throws -> Void,
    _ handler: (Error) -> Void
) async {
    do {
        try await expression()
        XCTFail("expected expression to throw")
    } catch {
        handler(error)
    }
}
