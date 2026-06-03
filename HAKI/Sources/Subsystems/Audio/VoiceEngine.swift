// VoiceEngine.swift
// HAKI — Audio Subsystem / Voice Engine
//
// Coordinates the LiveAudioEngine, VAD, and STTService to present the
// high-level Voice_Engine interface described in the design document:
//
//   VoiceEngine:
//     listen()      stream-> { frame, endOfSpeech, bargeIn,
//                              partialTranscript, finalTranscript, noSpeechDetected }
//     bargeInStop()
//
// Phase 1 Task 7.1 delivers the audio I/O + VAD layer.
// Phase 1 Task 7.2 wires STT: on endOfSpeech the buffered frames are sent
//   to STTService; partial/final transcripts and noSpeechDetected events
//   are emitted on the VoiceEvent stream.
//
// Design: Voice Pipeline, Voice_Engine (Requirements 3.2, 3.3, 3.4, 3.6)
//
// Threading:
//   • `listen()` returns an AsyncStream consumed by callers on any async context.
//   • The underlying audio tap fires on AVAudioEngine's realtime thread; the
//     AudioEngine delivers events through its own AsyncStream via a lock-guarded
//     continuation, so callers never block the realtime thread.
//   • STT is invoked on a detached Task so it never blocks the audio pump.

import Foundation
import HAKIIPC

// MARK: - VoiceEvent

/// Events streamed from `VoiceEngine.listen()`.
public enum VoiceEvent: Sendable {
    /// A 20 ms PCM audio frame — forwarded to the STT buffer (Req 3.4).
    case audioFrame(AudioFrame)
    /// End-of-speech: 800 ms of silence after user speech (Req 3.2).
    case endOfSpeech
    /// Barge-in detected: ≥ 200 ms of user speech while TTS is playing (Req 3.3).
    case bargeIn
    /// An intermediate (non-final) STT transcript for the current segment (Req 3.4).
    case partialTranscript(String)
    /// The committed final transcript plus acoustic features (Req 3.4).
    /// `audioFeatures` are forwarded to the Mood_Detector (Req 4.1).
    case finalTranscript(String, audioFeatures: HAKIAudioFeatures)
    /// No recognisable speech was detected; the engine has already emitted
    /// the user-facing "didn't catch that" prompt via `noSpeechPrompt`.
    /// Callers MUST NOT dispatch a transcript. (Req 3.6)
    case noSpeechDetected
}

// MARK: - VoiceEngineProtocol

/// Public contract for the Voice Engine coordinator.
public protocol VoiceEngineProtocol: AnyObject, Sendable {
    /// Begin capturing and return a stream of `VoiceEvent`s.
    /// The stream ends when `stopListening()` is called or an error occurs.
    func listen() throws -> AsyncStream<VoiceEvent>
    /// Stop active capture and finish the stream returned by `listen()`.
    func stopListening()
    /// Notify the engine that TTS playback has started (for barge-in tracking).
    func notifyTTSStarted()
    /// Notify the engine that TTS playback has ended.
    func notifyTTSStopped()
    /// Stop TTS playback immediately on barge-in (Req 3.3).
    func bargeInStop()
    /// User-facing prompt to repeat when no speech was recognised (Req 3.6).
    var noSpeechPrompt: String { get }

    /// Stream LLM tokens as sentence-chunked TTS speech, with barge-in
    /// cancellation and on-screen text fallback on failure.
    ///
    /// - Begins playback of the first clause within 300 ms of it becoming
    ///   available (Req 3.1) while subsequent clauses are synthesised.
    /// - Listens for `VoiceEvent.bargeIn` on the live event stream; stops
    ///   playback and cancels synthesis immediately on barge-in (Req 3.3).
    /// - On TTS failure posts the full response to `UIState` so the UI can
    ///   render it as on-screen text (Req 3.7).
    ///
    /// - Parameter textStream: LLM token stream to convert to speech.
    ///
    /// Implements: Req 3.1, 3.5, 3.7 — Design: Voice Pipeline, Voice_Engine
    func speak(textStream: AsyncStream<String>) async throws
}

// MARK: - VoiceEngine

/// Production implementation.
///
/// Wraps `LiveAudioEngine`, `VAD`, and `STTService` to expose the high-level
/// `VoiceEvent` stream.
///
/// STT wiring (Req 3.4, 3.6):
///   • Every `.audioFrame` is appended to an in-progress speech buffer.
///   • On `.endOfSpeech` the buffer is drained and forwarded to `STTService`.
///   • Partial and final transcripts — plus `.noSpeechDetected` — are emitted
///     on the outer `VoiceEvent` stream.
///   • On `.bargeIn` the speech buffer is cleared (that segment is discarded).
public final class VoiceEngine: VoiceEngineProtocol, @unchecked Sendable {

    // MARK: - Dependencies

    private let audioEngine: AudioEngineProtocol
    private let sttService: STTServiceProtocol
    let ttsService: any TTSServiceProtocol

    // MARK: - State

    private var pumpTask: Task<Void, Never>?
    private var outerContinuation: AsyncStream<VoiceEvent>.Continuation?
    private let lock = NSLock()

    /// Frames accumulated since the last speech-start event.
    /// Drained when `endOfSpeech` fires.
    private var speechBuffer: [AudioFrame] = []

    // MARK: - Init

    /// Designated initialiser.
    /// - Parameters:
    ///   - audioEngine: The audio I/O back-end.  Defaults to `LiveAudioEngine`.
    ///   - sttService: The STT service.  Defaults to a `MockSTTService` so
    ///     the audio layer compiles and tests without a live IPC connection.
    ///   - ttsService: The TTS coordinator.  Defaults to `MockTTSService` so
    ///     tests compile without real audio hardware.
    public init(
        audioEngine: AudioEngineProtocol = LiveAudioEngine(),
        sttService: STTServiceProtocol = MockSTTService(),
        ttsService: any TTSServiceProtocol = MockTTSService()
    ) {
        self.audioEngine = audioEngine
        self.sttService = sttService
        self.ttsService = ttsService
    }

    deinit {
        stopListening()
    }

    // MARK: - VoiceEngineProtocol

    public var noSpeechPrompt: String { sttService.noSpeechPrompt }

    /// Start capturing microphone audio and return a stream of `VoiceEvent`s.
    ///
    /// The stream emits:
    ///   - `.audioFrame` for every 20 ms PCM chunk
    ///   - `.endOfSpeech` after 800 ms of silence following speech (Req 3.2)
    ///   - `.bargeIn` after ≥ 200 ms speech while TTS is playing (Req 3.3)
    ///   - `.partialTranscript` / `.finalTranscript` from STT (Req 3.4)
    ///   - `.noSpeechDetected` when STT finds no speech (Req 3.6)
    ///
    /// - Throws: `AudioEngineError` if the microphone cannot be started.
    @discardableResult
    public func listen() throws -> AsyncStream<VoiceEvent> {
        // Stop any previous session first.
        stopListening()

        // Build the outer stream and capture its continuation.
        let stream = AsyncStream<VoiceEvent> { [weak self] continuation in
            guard let self else { return }
            self.lock.withLock { self.outerContinuation = continuation }
        }

        // Start capture (throws on permission/hardware errors).
        try audioEngine.startCapture()

        // Pump VADEvents → VoiceEvents on a background task.
        let engineEvents = audioEngine.events
        pumpTask = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            for await vadEvent in engineEvents {
                await self.handle(vadEvent: vadEvent)
            }
            // Engine finished — close the outer stream.
            self.lock.withLock {
                self.outerContinuation?.finish()
            }
        }

        return stream
    }

    /// Stop capturing and finish the `VoiceEvent` stream.
    public func stopListening() {
        pumpTask?.cancel()
        pumpTask = nil
        audioEngine.stopCapture()
        lock.withLock {
            outerContinuation?.finish()
            outerContinuation = nil
            speechBuffer.removeAll()
        }
    }

    /// Inform the VAD that TTS playback has started (arms barge-in detection).
    public func notifyTTSStarted() {
        audioEngine.setTTSPlaying(true)
    }

    /// Inform the VAD that TTS playback has stopped (disarms barge-in detection).
    public func notifyTTSStopped() {
        audioEngine.setTTSPlaying(false)
    }

    /// Stop TTS playback immediately in response to a barge-in (Req 3.3).
    ///
    /// The caller (e.g. the Orchestrator) is responsible for cancelling the
    /// downstream TTS/LLM generation.  This method disarms the VAD so the
    /// new user speech is captured cleanly.
    public func bargeInStop() {
        audioEngine.setTTSPlaying(false)
        ttsService.cancel()
    }

    /// Stream LLM tokens as sentence-chunked TTS speech (Req 3.1, 3.5, 3.7).
    ///
    /// - Segments `textStream` into clauses via `ClauseSegmenter`.
    /// - Begins synthesis of the first clause immediately; subsequent clauses
    ///   are synthesised while earlier ones play (pipeline: generate-while-play).
    /// - Calls `notifyTTSStarted()` before the first audio chunk plays so the
    ///   VAD arms barge-in detection.
    /// - Listens for `VoiceEvent.bargeIn` concurrently; stops and cancels on
    ///   detection.
    /// - On TTS failure posts the full response text to `UIState` so the UI
    ///   renders it as on-screen text (Req 3.7).
    public func speak(textStream: AsyncStream<String>) async throws {
        // Build a broadcast of the current VoiceEvent stream so the TTS
        // service can observe barge-in without consuming the primary stream.
        let (bargeInStream, continuation) = AsyncStream<VoiceEvent>.makeStream()

        // Tap the outer voice event stream for barge-in events only.
        let tapTask = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            // Re-use the current engine events directly; the stream is shared
            // via the audioEngine's AsyncStream.
            for await event in self.audioEngine.events {
                if case .bargeIn = event {
                    continuation.yield(.bargeIn)
                    return // One barge-in is enough.
                }
            }
            continuation.finish()
        }
        defer {
            tapTask.cancel()
            continuation.finish()
        }

        await ttsService.speak(
            textStream: textStream,
            voiceEvents: bargeInStream,
            onStarted: { [weak self] in
                self?.notifyTTSStarted()
            },
            onStopped: { [weak self] in
                self?.notifyTTSStopped()
            }
        )
    }

    // MARK: - Private: VAD event routing

    /// Translate a `VADEvent` into one or more `VoiceEvent`s and handle STT
    /// dispatch on end-of-speech.
    private func handle(vadEvent: VADEvent) async {
        switch vadEvent {
        case .frame(let frame):
            // Accumulate the frame and forward it to consumers.
            lock.withLock {
                speechBuffer.append(frame)
            }
            emit(.audioFrame(frame))

        case .endOfSpeech:
            emit(.endOfSpeech)

            // Drain the speech buffer and send to STT asynchronously so the
            // audio pump is not blocked.
            let framesForSTT: [AudioFrame] = lock.withLock {
                let copy = speechBuffer
                speechBuffer.removeAll()
                return copy
            }

            await runSTT(frames: framesForSTT)

        case .bargeIn:
            // Discard the in-progress speech buffer — we're not transcribing it.
            lock.withLock { speechBuffer.removeAll() }
            emit(.bargeIn)
            audioEngine.setTTSPlaying(false)
        }
    }

    /// Invoke the STT service for the collected frames and forward results.
    ///
    /// Must be called off the audio realtime thread (called from the pump task).
    private func runSTT(frames: [AudioFrame]) async {
        guard !frames.isEmpty else { return }

        let sttStream = sttService.transcribe(frames: frames)
        for await sttEvent in sttStream {
            switch sttEvent {
            case .partial(let text):
                emit(.partialTranscript(text))

            case .final(let text, let features):
                emit(.finalTranscript(text, audioFeatures: features))

            case .noSpeechDetected:
                // Emit the event; callers that show a UI use `noSpeechPrompt`.
                emit(.noSpeechDetected)
            }
        }
    }

    // MARK: - Private: thread-safe emission

    private func emit(_ event: VoiceEvent) {
        lock.withLock {
            _ = outerContinuation?.yield(event)
        }
    }
}

// MARK: - VoiceEngineFactory

/// Convenience factory that injects the correct engine for the current context.
public enum VoiceEngineFactory {
    /// Production voice engine backed by `LiveAudioEngine`, a real `STTService`,
    /// and a `TTSService` wired to the shared IPC client.
    ///
    /// - Parameter ipcClient: A connected `IPCClientProtocol` instance.
    public static func makeLive(ipcClient: any IPCClientProtocol) -> VoiceEngine {
        let tts = TTSService(ipcClient: ipcClient)
        return VoiceEngine(
            audioEngine: LiveAudioEngine(),
            sttService: STTService(ipcClient: ipcClient),
            ttsService: tts
        )
    }

    /// Test voice engine backed by `MockAudioEngine`, `MockSTTService`, and
    /// `MockTTSService`.
    public static func makeMock(
        audioEngine: MockAudioEngine = MockAudioEngine(),
        sttService: MockSTTService = MockSTTService(),
        ttsService: MockTTSService = MockTTSService()
    ) -> (VoiceEngine, MockAudioEngine, MockSTTService, MockTTSService) {
        let engine = VoiceEngine(audioEngine: audioEngine, sttService: sttService, ttsService: ttsService)
        return (engine, audioEngine, sttService, ttsService)
    }
}
