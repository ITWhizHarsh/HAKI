// VAD.swift
// HAKI — Audio Subsystem / Voice Activity Detector
//
// A lightweight, on-thread energy-based VAD that:
//   • detects end-of-speech after 800 ms of continuous silence (Req 3.2)
//   • detects barge-in after ≥ 200 ms of continuous speech during TTS playback (Req 3.3)
//
// Full implementation: Phase 1 Task 7.1
// This file contains the public interface and stub.

import Foundation

// MARK: - VADState

/// Internal state machine for the Voice Activity Detector.
public enum VADState: Equatable {
    /// No speech activity.
    case idle
    /// User is speaking (started at `startedAt`).
    case speaking(startedAt: Date)
    /// Silence detected after speech; silence began at `since`.
    case silenceAfterSpeech(since: Date)
}

// MARK: - VAD

/// Energy-based Voice Activity Detector.
///
/// Call `process(frame:)` for every 20 ms audio frame.
/// After 800 ms of silence following speech, `endOfSpeechHandler` is called.
/// After 200 ms of speech during TTS playback, `bargeInHandler` is called.
public final class VAD {

    // MARK: - Configuration

    /// RMS energy threshold below which a frame is considered silent.
    public var silenceThreshold: Float = 0.01
    /// Duration of continuous silence required to trigger end-of-speech (Req 3.2).
    public var endOfSpeechDuration: TimeInterval = 0.800
    /// Duration of continuous speech required to trigger barge-in (Req 3.3).
    public var bargeInDuration: TimeInterval = 0.200

    // MARK: - Handlers

    public var endOfSpeechHandler: (() -> Void)?
    public var bargeInHandler: (() -> Void)?

    // MARK: - State

    private var state: VADState = .idle
    private var isTTSPlaying: Bool = false

    // MARK: - Public API

    public init() {}

    /// Inform the VAD that TTS playback has started or stopped.
    public func setTTSPlaying(_ playing: Bool) {
        isTTSPlaying = playing
    }

    /// Process a single 20 ms audio frame.
    /// - Parameter frame: Captured PCM frame.
    public func process(frame: AudioFrame) {
        let energy = rmsEnergy(frame.samples)
        let isSpeech = energy >= silenceThreshold
        let now = frame.timestamp

        switch state {
        case .idle:
            if isSpeech {
                state = .speaking(startedAt: now)
                // Check barge-in: will be evaluated on subsequent frames.
            }

        case .speaking(let startedAt):
            if isSpeech {
                // Continued speech.
                if isTTSPlaying, now.timeIntervalSince(startedAt) >= bargeInDuration {
                    bargeInHandler?()
                    state = .idle
                }
            } else {
                // Silence began.
                state = .silenceAfterSpeech(since: now)
            }

        case .silenceAfterSpeech(let since):
            if isSpeech {
                // Speech resumed; restart.
                state = .speaking(startedAt: now)
            } else if now.timeIntervalSince(since) >= endOfSpeechDuration {
                endOfSpeechHandler?()
                state = .idle
            }
        }
    }

    // MARK: - Private helpers

    private func rmsEnergy(_ samples: [Int16]) -> Float {
        guard !samples.isEmpty else { return 0 }
        let sumOfSquares = samples.reduce(Float(0)) { acc, s in
            let f = Float(s) / Float(Int16.max)
            return acc + f * f
        }
        return sqrt(sumOfSquares / Float(samples.count))
    }
}
