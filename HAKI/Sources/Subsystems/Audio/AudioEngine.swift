// AudioEngine.swift
// HAKI — Audio Subsystem
//
// Owns the AVAudioEngine microphone tap, realtime VAD (Voice Activity Detection),
// end-of-speech detection (800 ms silence), and barge-in detection (≥200 ms
// continuous speech during playback).
//
// Acoustic Echo Cancellation (AEC) is applied on the mic path so TTS playback
// does not feed back into the STT input.
//
// Threading model:
//   • The audio tap callback runs on AVAudioEngine's realtime thread.
//   • VAD and energy calculations are performed inline on that thread.
//   • Detected events (endOfSpeech, bargeIn, frame) are dispatched to an
//     async stream consumed by the Voice_Engine coordinator.
//
// Implements: Req 3.2 (800 ms end-of-speech), 3.3 (200 ms barge-in)
// Phase 1 Task 7.1 will provide the full implementation.

import AVFoundation

// MARK: - AudioFrame

/// A single 20 ms PCM audio frame captured from the microphone.
public struct AudioFrame: Sendable {
    /// 20 ms of 16-bit PCM at 16 kHz (320 samples).
    public let samples: [Int16]
    /// Capture timestamp.
    public let timestamp: Date

    public init(samples: [Int16], timestamp: Date = Date()) {
        self.samples = samples
        self.timestamp = timestamp
    }
}

// MARK: - VADEvent

/// Events emitted by the Voice Activity Detector.
public enum VADEvent: Sendable {
    /// A new 20 ms audio frame is available for streaming to STT.
    case frame(AudioFrame)
    /// 800 ms of continuous silence detected — user has finished speaking.
    case endOfSpeech
    /// ≥ 200 ms of continuous speech detected while TTS is playing — barge-in.
    case bargeIn
}

// MARK: - AudioEngineProtocol

/// Abstract interface for the audio I/O subsystem.
/// Concrete implementation is `LiveAudioEngine`; a stub is `MockAudioEngine`.
public protocol AudioEngineProtocol: AnyObject, Sendable {
    /// Start capturing microphone input and emitting `VADEvent`s.
    func startCapture() throws
    /// Stop microphone capture.
    func stopCapture()
    /// Stream of `VADEvent`s produced during active capture.
    var events: AsyncStream<VADEvent> { get }
}

// MARK: - LiveAudioEngine (placeholder)

/// Production implementation — Phase 1 Task 7.1.
public final class LiveAudioEngine: AudioEngineProtocol, @unchecked Sendable {

    // MARK: - Constants

    /// Sample rate used throughout the voice pipeline.
    public static let sampleRate: Double = 16_000
    /// Frame duration in seconds (20 ms).
    public static let frameDuration: Double = 0.020
    /// Samples per frame.
    public static let samplesPerFrame: Int = Int(sampleRate * frameDuration) // 320

    // MARK: - State

    private let engine = AVAudioEngine()
    private var continuation: AsyncStream<VADEvent>.Continuation?

    // MARK: - AudioEngineProtocol

    public lazy var events: AsyncStream<VADEvent> = {
        AsyncStream { [weak self] continuation in
            self?.continuation = continuation
        }
    }()

    public func startCapture() throws {
        // TODO: Phase 1 Task 7.1 — install AVAudioEngine tap, AEC, VAD loop
        throw AudioEngineError.notImplemented
    }

    public func stopCapture() {
        engine.stop()
        continuation?.finish()
    }
}

// MARK: - AudioEngineError

public enum AudioEngineError: Error {
    case notImplemented
    case permissionDenied
    case hardwareUnavailable
}
