// STTServiceTests.swift
// HAKI — Unit tests for STTService and VoiceEngine STT wiring
//
// Phase 1 Task 7.2
// Requirements: 3.4, 3.6

#if canImport(XCTest)
import XCTest
@testable import HAKIAudio
@testable import HAKIIPC

// MARK: - Helpers

/// A mock IPC client that plays back pre-loaded ServerMessage sequences.
private final class MockIPCClient: IPCClientProtocol, @unchecked Sendable {

    public private(set) var isConnected: Bool
    private var inboundContinuation: AsyncStream<ServerMessage>.Continuation?
    private let lock = NSLock()

    public lazy var inbound: AsyncStream<ServerMessage> = {
        AsyncStream { [weak self] continuation in
            self?.lock.withLock { self?.inboundContinuation = continuation }
        }
    }()

    init(connected: Bool = true) {
        self.isConnected = connected
        _ = inbound          // materialise the stream / continuation eagerly
    }

    func connect() async throws { isConnected = true }
    func disconnect() async     { isConnected = false; inboundContinuation?.finish() }

    /// Captured outbound messages.
    private(set) var sentMessages: [ClientMessage] = []

    func send(_ message: ClientMessage) async throws {
        lock.withLock { sentMessages.append(message) }
    }

    /// Inject messages that the client will receive from the "server".
    func inject(_ message: ServerMessage) {
        lock.withLock { _ = inboundContinuation?.yield(message) }
    }

    func finishInbound() {
        inboundContinuation?.finish()
    }
}

/// Build a realistic-looking `AudioFrame`.
private func makeFrame(rmsAmplitude: Float = 0.05, timestamp: Date = Date()) -> AudioFrame {
    let sampleCount = 320   // 20 ms at 16 kHz
    var samples = [Int16](repeating: 0, count: sampleCount)
    // Fill with a sine-like pattern scaled to the requested amplitude.
    let amplitude = Int16(rmsAmplitude * Float(Int16.max))
    for i in 0..<sampleCount {
        samples[i] = (i % 2 == 0) ? amplitude : -amplitude
    }
    return AudioFrame(samples: samples, timestamp: timestamp)
}

// MARK: - STTServiceTests

final class STTServiceTests: XCTestCase {

    // MARK: Audio features extraction

    func test_extractAudioFeatures_emptyFrames_returnsZeroDuration() {
        let client = MockIPCClient()
        let stt = STTService(ipcClient: client)
        let features = stt.extractAudioFeatures(from: [])
        XCTAssertEqual(features.durationMs, 0)
    }

    func test_extractAudioFeatures_duration_is20msPerFrame() {
        let client = MockIPCClient()
        let stt = STTService(ipcClient: client)
        let frames = (0..<5).map { _ in makeFrame() }
        let features = stt.extractAudioFeatures(from: frames)
        XCTAssertEqual(features.durationMs, 100)  // 5 × 20 ms
    }

    func test_extractAudioFeatures_energyDb_isNegativeForNonSilentFrames() {
        let client = MockIPCClient()
        let stt = STTService(ipcClient: client)
        let frames = [makeFrame(rmsAmplitude: 0.1)]
        let features = stt.extractAudioFeatures(from: frames)
        // Non-silent frames have energy > -96 dBFS.
        XCTAssertGreaterThan(features.energyDb, -96)
    }

    func test_extractAudioFeatures_pitchHz_isPositiveForAltSignal() {
        let client = MockIPCClient()
        let stt = STTService(ipcClient: client)
        // Alternating +/-amplitude produces many zero crossings → positive pitch.
        let frames = [makeFrame(rmsAmplitude: 0.1)]
        let features = stt.extractAudioFeatures(from: frames)
        XCTAssertGreaterThan(features.pitchHz, 0)
    }

    // MARK: Transcribe — IPC not connected

    func test_transcribe_whenNotConnected_emitsNoSpeechDetected() async {
        let client = MockIPCClient(connected: false)
        let stt = STTService(ipcClient: client)
        let frames = [makeFrame()]
        var events: [STTEvent] = []
        for await event in stt.transcribe(frames: frames) {
            events.append(event)
        }
        XCTAssertEqual(events.count, 1)
        if case .noSpeechDetected = events[0] { /* pass */ } else {
            XCTFail("Expected .noSpeechDetected, got \(events[0])")
        }
    }

    // MARK: Transcribe — empty frame buffer

    func test_transcribe_emptyFrames_emitsNoSpeechDetected() async {
        let client = MockIPCClient(connected: true)
        let stt = STTService(ipcClient: client)
        var events: [STTEvent] = []
        for await event in stt.transcribe(frames: []) {
            events.append(event)
        }
        XCTAssertEqual(events.count, 1)
        if case .noSpeechDetected = events[0] { /* pass */ } else {
            XCTFail("Expected .noSpeechDetected, got \(events[0])")
        }
    }

    // MARK: Transcribe — partial then final transcript

    func test_transcribe_partialThenFinal_emitsPartialThenFinal() async {
        let client = MockIPCClient(connected: true)
        let stt = STTService(ipcClient: client)
        let frames = [makeFrame()]

        // Inject server responses after a brief yield.
        Task {
            try? await Task.sleep(nanoseconds: 1_000_000)  // 1 ms
            client.inject(.partialTranscript(HAKIPartialTranscript(text: "hel", isFinal: false, sequenceNum: 1)))
            client.inject(.partialTranscript(HAKIPartialTranscript(text: "hello", isFinal: true, sequenceNum: 2)))
        }

        var events: [STTEvent] = []
        for await event in stt.transcribe(frames: frames) {
            events.append(event)
        }

        XCTAssertEqual(events.count, 2)
        if case .partial(let t) = events[0] { XCTAssertEqual(t, "hel") }
        else { XCTFail("Expected .partial, got \(events[0])") }

        if case .final(let t, _) = events[1] { XCTAssertEqual(t, "hello") }
        else { XCTFail("Expected .final, got \(events[1])") }
    }

    // MARK: Transcribe — empty final transcript → noSpeechDetected (Req 3.6)

    func test_transcribe_emptyFinalTranscript_emitsNoSpeechDetected() async {
        let client = MockIPCClient(connected: true)
        let stt = STTService(ipcClient: client)
        let frames = [makeFrame()]

        Task {
            try? await Task.sleep(nanoseconds: 1_000_000)
            // Core returns an empty string as the final transcript.
            client.inject(.partialTranscript(HAKIPartialTranscript(text: "", isFinal: true, sequenceNum: 1)))
        }

        var events: [STTEvent] = []
        for await event in stt.transcribe(frames: frames) {
            events.append(event)
        }

        XCTAssertEqual(events.count, 1)
        if case .noSpeechDetected = events[0] { /* pass */ } else {
            XCTFail("Expected .noSpeechDetected for empty transcript, got \(events[0])")
        }
    }

    // MARK: Transcribe — whitespace-only final transcript → noSpeechDetected (Req 3.6)

    func test_transcribe_whitespaceOnlyFinalTranscript_emitsNoSpeechDetected() async {
        let client = MockIPCClient(connected: true)
        let stt = STTService(ipcClient: client)
        let frames = [makeFrame()]

        Task {
            try? await Task.sleep(nanoseconds: 1_000_000)
            client.inject(.partialTranscript(HAKIPartialTranscript(text: "   ", isFinal: true, sequenceNum: 1)))
        }

        var events: [STTEvent] = []
        for await event in stt.transcribe(frames: frames) {
            events.append(event)
        }

        XCTAssertEqual(events.count, 1)
        if case .noSpeechDetected = events[0] { /* pass */ } else {
            XCTFail("Expected .noSpeechDetected for whitespace-only transcript, got \(events[0])")
        }
    }

    // MARK: Transcribe — Core error → noSpeechDetected

    func test_transcribe_coreError_emitsNoSpeechDetected() async {
        let client = MockIPCClient(connected: true)
        let stt = STTService(ipcClient: client)
        let frames = [makeFrame()]

        Task {
            try? await Task.sleep(nanoseconds: 1_000_000)
            client.inject(.error("STT backend failed"))
        }

        var events: [STTEvent] = []
        for await event in stt.transcribe(frames: frames) {
            events.append(event)
        }

        XCTAssertEqual(events.count, 1)
        if case .noSpeechDetected = events[0] { /* pass */ } else {
            XCTFail("Expected .noSpeechDetected on Core error, got \(events[0])")
        }
    }

    // MARK: Transcribe — frames are streamed to IPC

    func test_transcribe_sendsFramesToIPC() async {
        let client = MockIPCClient(connected: true)
        let stt = STTService(ipcClient: client)
        let frames = (0..<3).map { _ in makeFrame() }

        Task {
            try? await Task.sleep(nanoseconds: 5_000_000)
            client.inject(.partialTranscript(HAKIPartialTranscript(text: "test", isFinal: true, sequenceNum: 1)))
        }

        // Drain the stream.
        for await _ in stt.transcribe(frames: frames) {}

        // Should have sent 3 audio frames + 1 endOfSpeech control event.
        let audioFrameCount = client.sentMessages.filter {
            if case .audioFrame = $0 { return true }
            return false
        }.count
        let controlEventCount = client.sentMessages.filter {
            if case .controlEvent(let ce) = $0, ce.eventType == .endOfSpeech { return true }
            return false
        }.count

        XCTAssertEqual(audioFrameCount, 3, "Expected 3 audio frames sent to IPC")
        XCTAssertEqual(controlEventCount, 1, "Expected 1 endOfSpeech control event sent to IPC")
    }

    // MARK: noSpeechPrompt

    func test_noSpeechPrompt_isNonEmpty() {
        let client = MockIPCClient()
        let stt = STTService(ipcClient: client)
        XCTAssertFalse(stt.noSpeechPrompt.isEmpty)
    }
}

// MARK: - VoiceEngine STT wiring tests

final class VoiceEngineSTTTests: XCTestCase {

    // MARK: finalTranscript event is emitted after endOfSpeech

    func test_voiceEngine_emitsFinalTranscript_afterEndOfSpeech() async throws {
        let mockAudio = MockAudioEngine()
        let mockSTT = MockSTTService()

        let (engine, _, _, _) = VoiceEngineFactory.makeMock(
            audioEngine: mockAudio,
            sttService: mockSTT
        )

        // Prime the STT mock to return a final transcript.
        let dummyFeatures = HAKIAudioFeatures(pitchHz: 220, energyDb: -20, durationMs: 400)
        mockSTT.enqueue(events: [.final("hello HAKI", audioFeatures: dummyFeatures)])

        let stream = try engine.listen()

        // Inject a frame followed by endOfSpeech.
        let frame = AudioFrame(samples: [Int16](repeating: 100, count: 320), timestamp: Date())
        mockAudio.inject(.frame(frame))
        mockAudio.inject(.endOfSpeech)

        var received: [VoiceEvent] = []
        let collectTask = Task {
            for await event in stream {
                received.append(event)
                // Stop once we see the finalTranscript.
                if case .finalTranscript = event { break }
            }
        }

        // Allow the pump to process events.
        try await Task.sleep(nanoseconds: 50_000_000)  // 50 ms
        engine.stopListening()
        collectTask.cancel()
        await collectTask.value

        let finals = received.filter {
            if case .finalTranscript = $0 { return true }
            return false
        }
        XCTAssertEqual(finals.count, 1, "Expected exactly one finalTranscript event")
        if case .finalTranscript(let text, _) = finals[0] {
            XCTAssertEqual(text, "hello HAKI")
        }
    }

    // MARK: noSpeechDetected event is emitted and never dispatched as transcript (Req 3.6)

    func test_voiceEngine_emitsNoSpeechDetected_whenSTTFindsNoSpeech() async throws {
        let mockAudio = MockAudioEngine()
        let mockSTT = MockSTTService()
        let (engine, _, _, _) = VoiceEngineFactory.makeMock(
            audioEngine: mockAudio,
            sttService: mockSTT
        )

        // STT returns noSpeechDetected — no transcript should flow.
        mockSTT.enqueue(events: [.noSpeechDetected])

        let stream = try engine.listen()
        let frame = AudioFrame(samples: [Int16](repeating: 100, count: 320), timestamp: Date())
        mockAudio.inject(.frame(frame))
        mockAudio.inject(.endOfSpeech)

        var received: [VoiceEvent] = []
        let collectTask = Task {
            for await event in stream {
                received.append(event)
                if case .noSpeechDetected = event { break }
            }
        }

        try await Task.sleep(nanoseconds: 50_000_000)
        engine.stopListening()
        collectTask.cancel()
        await collectTask.value

        let noSpeechEvents = received.filter {
            if case .noSpeechDetected = $0 { return true }
            return false
        }
        let finalEvents = received.filter {
            if case .finalTranscript = $0 { return true }
            return false
        }
        XCTAssertEqual(noSpeechEvents.count, 1, "Expected one noSpeechDetected event")
        XCTAssertEqual(finalEvents.count, 0, "No finalTranscript should be emitted on no-speech (Req 3.6)")
    }

    // MARK: bargeIn clears the speech buffer

    func test_voiceEngine_bargeIn_clearsBuffer() async throws {
        let mockAudio = MockAudioEngine()
        let mockSTT = MockSTTService()
        let (engine, _, _, _) = VoiceEngineFactory.makeMock(
            audioEngine: mockAudio,
            sttService: mockSTT
        )

        let stream = try engine.listen()

        // Inject a frame, then a bargeIn (buffer should be discarded).
        let frame = AudioFrame(samples: [Int16](repeating: 100, count: 320), timestamp: Date())
        mockAudio.inject(.frame(frame))
        mockAudio.inject(.bargeIn)

        var received: [VoiceEvent] = []
        let collectTask = Task {
            for await event in stream {
                received.append(event)
                if case .bargeIn = event { break }
            }
        }

        try await Task.sleep(nanoseconds: 50_000_000)
        engine.stopListening()
        collectTask.cancel()
        await collectTask.value

        // STT should NOT have been called — no frames were received.
        XCTAssertEqual(mockSTT.receivedFrameBuffers.count, 0,
                       "STT should not be called after bargeIn discards the buffer")
        let bargeInEvents = received.filter {
            if case .bargeIn = $0 { return true }
            return false
        }
        XCTAssertEqual(bargeInEvents.count, 1)
    }

    // MARK: partialTranscript events flow through

    func test_voiceEngine_emitsPartialTranscripts() async throws {
        let mockAudio = MockAudioEngine()
        let mockSTT = MockSTTService()
        let (engine, _, _, _) = VoiceEngineFactory.makeMock(
            audioEngine: mockAudio,
            sttService: mockSTT
        )

        let dummyFeatures = HAKIAudioFeatures(pitchHz: 180, energyDb: -25, durationMs: 500)
        mockSTT.enqueue(events: [
            .partial("hel"),
            .partial("hello"),
            .final("hello there", audioFeatures: dummyFeatures),
        ])

        let stream = try engine.listen()
        let frame = AudioFrame(samples: [Int16](repeating: 100, count: 320), timestamp: Date())
        mockAudio.inject(.frame(frame))
        mockAudio.inject(.endOfSpeech)

        var received: [VoiceEvent] = []
        let collectTask = Task {
            for await event in stream {
                received.append(event)
                if case .finalTranscript = event { break }
            }
        }

        try await Task.sleep(nanoseconds: 50_000_000)
        engine.stopListening()
        collectTask.cancel()
        await collectTask.value

        let partials = received.compactMap { event -> String? in
            if case .partialTranscript(let t) = event { return t }
            return nil
        }
        XCTAssertEqual(partials, ["hel", "hello"])
    }

    // MARK: noSpeechPrompt is forwarded from STT service

    func test_voiceEngine_noSpeechPromptIsSet() throws {
        let mockAudio = MockAudioEngine()
        let mockSTT = MockSTTService()
        let (engine, _, _, _) = VoiceEngineFactory.makeMock(
            audioEngine: mockAudio,
            sttService: mockSTT
        )
        XCTAssertFalse(engine.noSpeechPrompt.isEmpty)
    }
}
#endif // canImport(XCTest)
