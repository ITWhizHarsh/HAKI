// STTService.swift
// HAKI — Audio Subsystem / Speech-to-Text Service
//
// Accepts a buffer of AudioFrames collected during a speech segment,
// streams them to the Python Core via the IPC channel, and converts the
// streamed PartialTranscript responses back into STTEvents.
//
// Design: Voice Pipeline, Voice_Engine (Requirements 3.4, 3.6)
// Phase 1 Task 7.2
//
// Responsibilities:
//   • Compute HAKIAudioFeatures from the frame buffer before forwarding.
//   • Stream each frame as a ClientMessage.audioFrame upstream.
//   • Signal end-of-speech with a ControlEvent.endOfSpeech message.
//   • Collect ServerMessage.partialTranscript responses:
//       – non-final → emit .partial
//       – isFinal && non-empty → emit .final
//       – isFinal && empty (no recognisable speech) → emit .noSpeechDetected
//         and publish the "didn't catch that" user prompt.
//   • Never dispatch an empty transcript downstream (Req 3.6).
//
// Threading:
//   All public methods are async and safe to call from any task context.
//   The inbound IPC stream is consumed on a background Task spawned inside
//   `transcribe(frames:)`.

import Foundation
import HAKIIPC

// MARK: - STTEvent

/// Events emitted by the STT service for a single speech segment.
public enum STTEvent: Sendable {
    /// An intermediate (non-final) transcript — good for live UI updates.
    case partial(String)
    /// The committed, final transcript plus acoustic features for the
    /// Mood_Detector (Req 4.1).
    case final(String, audioFeatures: HAKIAudioFeatures)
    /// The Core's STT returned an empty transcript — no recognisable speech.
    /// The caller must prompt the user to repeat; nothing is dispatched (Req 3.6).
    case noSpeechDetected
}

// MARK: - STTServiceProtocol

/// Public contract for the STT coordinator.
public protocol STTServiceProtocol: AnyObject, Sendable {
    /// Transcribe a complete speech segment.
    ///
    /// - Parameter frames: The `AudioFrame`s collected between speech-start
    ///   and end-of-speech (800 ms silence, Req 3.2).
    /// - Returns: An `AsyncStream<STTEvent>` that emits `.partial` events
    ///   followed by exactly one `.final` or `.noSpeechDetected` event.
    func transcribe(frames: [AudioFrame]) -> AsyncStream<STTEvent>

    /// The user-facing "didn't catch that" prompt text (Req 3.6).
    var noSpeechPrompt: String { get }
}

// MARK: - STTService

/// Production implementation.
///
/// Streams frames to the Python Core over the IPC channel and converts
/// `ServerMessage.partialTranscript` responses to `STTEvent`s.
///
/// If the IPC client is not connected, `transcribe` falls back to
/// `.noSpeechDetected` so the caller always receives a well-formed response.
public final class STTService: STTServiceProtocol, @unchecked Sendable {

    // MARK: - Configuration

    public let noSpeechPrompt: String = "I didn't catch that — could you repeat?"

    /// Sample rate used by the audio pipeline.
    private static let sampleRate: Double = 16_000

    // MARK: - Dependencies

    private let ipcClient: IPCClientProtocol

    // MARK: - State

    /// Monotonically-increasing sequence counter for IPC messages.
    private var sequenceCounter: UInt32 = 0
    private let counterLock = NSLock()

    // MARK: - Init

    public init(ipcClient: IPCClientProtocol) {
        self.ipcClient = ipcClient
    }

    // MARK: - STTServiceProtocol

    public func transcribe(frames: [AudioFrame]) -> AsyncStream<STTEvent> {
        let features = extractAudioFeatures(from: frames)

        return AsyncStream<STTEvent> { continuation in
            Task.detached(priority: .userInitiated) { [weak self] in
                guard let self else {
                    continuation.yield(.noSpeechDetected)
                    continuation.finish()
                    return
                }

                // Guard: nothing to transcribe — treat as no speech.
                guard !frames.isEmpty else {
                    continuation.yield(.noSpeechDetected)
                    continuation.finish()
                    return
                }

                // Guard: IPC must be connected.
                guard self.ipcClient.isConnected else {
                    continuation.yield(.noSpeechDetected)
                    continuation.finish()
                    return
                }

                do {
                    // 1. Stream each frame upstream.
                    try await self.sendFrames(frames)
                    // 2. Signal end-of-speech.
                    try await self.sendEndOfSpeech()
                } catch {
                    print("[STTService] IPC send error: \(error)")
                    continuation.yield(.noSpeechDetected)
                    continuation.finish()
                    return
                }

                // 3. Collect server responses until a final transcript arrives.
                for await serverMessage in self.ipcClient.inbound {
                    switch serverMessage {
                    case .partialTranscript(let pt):
                        if pt.isFinal {
                            let trimmed = pt.text.trimmingCharacters(in: .whitespacesAndNewlines)
                            if trimmed.isEmpty {
                                // Empty final → no recognisable speech (Req 3.6).
                                continuation.yield(.noSpeechDetected)
                            } else {
                                continuation.yield(.final(trimmed, audioFeatures: features))
                            }
                            continuation.finish()
                            return
                        } else {
                            let trimmed = pt.text.trimmingCharacters(in: .whitespacesAndNewlines)
                            if !trimmed.isEmpty {
                                continuation.yield(.partial(trimmed))
                            }
                        }

                    case .error(let msg):
                        print("[STTService] Core STT error: \(msg)")
                        continuation.yield(.noSpeechDetected)
                        continuation.finish()
                        return

                    case .controlEvent(let ce) where ce.eventType == .cancel:
                        // Turn was cancelled mid-transcription.
                        continuation.finish()
                        return

                    default:
                        // LLM tokens, TTS chunks — not relevant during STT.
                        break
                    }
                }

                // Stream ended without a final transcript.
                continuation.yield(.noSpeechDetected)
                continuation.finish()
            }
        }
    }

    // MARK: - Private helpers

    /// Stream each frame to the Core as `ClientMessage.audioFrame` messages.
    private func sendFrames(_ frames: [AudioFrame]) async throws {
        for frame in frames {
            let seq = nextSequence()
            let ipcFrame = HAKIAudioFrame(
                samples: frameSamplesToData(frame.samples),
                timestampMs: UInt64(frame.timestamp.timeIntervalSince1970 * 1_000),
                sequenceNum: seq,
                sampleRate: UInt32(STTService.sampleRate),
                channels: 1
            )
            try await ipcClient.send(.audioFrame(ipcFrame))
        }
    }

    /// Send an `endOfSpeech` control event so the Core knows the segment is done.
    private func sendEndOfSpeech() async throws {
        let event = HAKIControlEvent(eventType: .endOfSpeech, sequenceNum: nextSequence())
        try await ipcClient.send(.controlEvent(event))
    }

    /// Compute `HAKIAudioFeatures` from the buffered frames.
    ///
    /// - `rmsEnergy` normalised to [0,1] then converted to dBFS
    /// - `pitchHz` via zero-crossing rate (same approach as `VAD.zeroCrossingRate`)
    /// - `durationMs` from frame count × 20 ms
    func extractAudioFeatures(from frames: [AudioFrame]) -> HAKIAudioFeatures {
        guard !frames.isEmpty else {
            return HAKIAudioFeatures(pitchHz: 0, energyDb: -96, durationMs: 0)
        }

        let allSamples = frames.flatMap { $0.samples }
        let durationMs = UInt32(frames.count * 20)  // 20 ms per frame

        // RMS energy → dBFS
        let rms = rmsEnergy(allSamples)
        let energyDb: Float = rms > 0 ? (20.0 * log10(rms)) : -96.0

        // Pitch via zero-crossing rate
        let pitchHz = zeroCrossingRate(allSamples, sampleRate: STTService.sampleRate)

        return HAKIAudioFeatures(pitchHz: pitchHz, energyDb: energyDb, durationMs: durationMs)
    }

    /// RMS amplitude normalised to [0, 1].
    private func rmsEnergy(_ samples: [Int16]) -> Float {
        guard !samples.isEmpty else { return 0 }
        let sumOfSquares = samples.reduce(Float(0)) { acc, s in
            let f = Float(s) / Float(Int16.max)
            return acc + f * f
        }
        return sqrt(sumOfSquares / Float(samples.count))
    }

    /// Pitch estimate via zero-crossing rate (Hz).
    private func zeroCrossingRate(_ samples: [Int16], sampleRate: Double) -> Float {
        guard samples.count > 1 else { return 0 }
        var crossings = 0
        for i in 1..<samples.count {
            if (samples[i - 1] >= 0) != (samples[i] >= 0) {
                crossings += 1
            }
        }
        let durationSeconds = Double(samples.count) / sampleRate
        return Float(Double(crossings) / 2.0 / durationSeconds)
    }

    /// Convert `[Int16]` to little-endian `Data`.
    private func frameSamplesToData(_ samples: [Int16]) -> Data {
        var data = Data(count: samples.count * MemoryLayout<Int16>.size)
        data.withUnsafeMutableBytes { ptr in
            guard let base = ptr.baseAddress?.assumingMemoryBound(to: Int16.self) else { return }
            for (i, sample) in samples.enumerated() {
                base[i] = sample.littleEndian
            }
        }
        return data
    }

    /// Thread-safe monotonically-increasing sequence number.
    private func nextSequence() -> UInt32 {
        counterLock.lock()
        defer { counterLock.unlock() }
        sequenceCounter &+= 1
        return sequenceCounter
    }
}

// MARK: - MockSTTService

/// A test double for `STTService`.
///
/// Callers inject scripted `STTEvent`s via `enqueue(events:)` before
/// calling `transcribe(frames:)`.  Captures every frame list passed to it
/// for assertion in tests.
public final class MockSTTService: STTServiceProtocol, @unchecked Sendable {

    public let noSpeechPrompt: String = "I didn't catch that — could you repeat?"

    /// Scripted response sequences — one array per `transcribe` call.
    private var responseQueue: [[STTEvent]] = []
    private let lock = NSLock()

    /// Every frame buffer passed to `transcribe` is recorded here.
    public private(set) var receivedFrameBuffers: [[AudioFrame]] = []

    public init() {}

    /// Enqueue a sequence of `STTEvent`s to be returned by the next
    /// `transcribe` call.
    public func enqueue(events: [STTEvent]) {
        lock.withLock { responseQueue.append(events) }
    }

    public func transcribe(frames: [AudioFrame]) -> AsyncStream<STTEvent> {
        let events: [STTEvent] = lock.withLock {
            receivedFrameBuffers.append(frames)
            guard !responseQueue.isEmpty else { return [.noSpeechDetected] }
            return responseQueue.removeFirst()
        }
        return AsyncStream<STTEvent> { continuation in
            for event in events {
                continuation.yield(event)
            }
            continuation.finish()
        }
    }
}
