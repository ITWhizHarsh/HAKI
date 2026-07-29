// VAD.swift
// HAKI — Audio Subsystem / Voice Activity Detector
//
// A lightweight, on-thread energy-based VAD that:
//   • detects end-of-speech after 800 ms of continuous silence (Req 3.2)
//   • detects barge-in after ≥ 200 ms of continuous speech during TTS playback (Req 3.3)
//
// Design: Voice Pipeline — "VAD lives on the Swift realtime audio thread so
//   end-of-speech and barge-in detection never wait on IPC (3.2, 3.3)."
//
// Implementation: Phase 1 Task 7.1

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
/// Call `process(frame:)` for every 20 ms audio frame **on the realtime thread**.
///
/// Thresholds:
/// - `silenceThreshold`: RMS amplitude below which a frame is silent.
/// - `endOfSpeechDuration`: 800 ms of continuous silence after speech → end-of-speech (Req 3.2).
/// - `bargeInDuration`: 200 ms of continuous speech while TTS plays → barge-in (Req 3.3).
///
/// Thread safety: All methods must be called from the same (audio realtime) thread.
/// Callbacks (`endOfSpeechHandler`, `bargeInHandler`) are invoked on that same thread.
public final class VAD {

    // MARK: - Configuration

    /// RMS amplitude threshold. Frames with energy below this are classified silent.
    /// Default 0.01 (~–40 dBFS normalised) works well in normal indoor environments.
    public var silenceThreshold: Float = 0.01

    /// Continuous silence required to declare end-of-speech (Req 3.2).
    public var endOfSpeechDuration: TimeInterval = 0.800

    /// Continuous speech required to trigger barge-in during TTS playback (Req 3.3).
    public var bargeInDuration: TimeInterval = 0.200

    // MARK: - Handlers (called on the audio realtime thread)

    /// Called once after `endOfSpeechDuration` of continuous silence following speech.
    public var endOfSpeechHandler: (() -> Void)?

    /// Called once after `bargeInDuration` of continuous speech while TTS is playing.
    public var bargeInHandler: (() -> Void)?

    // MARK: - State

    private(set) public var state: VADState = .idle

    /// Whether the TTS playback is currently active.
    /// Set this from the audio/playback thread before calling `process(frame:)`.
    private var isTTSPlaying: Bool = false

    // MARK: - Public API

    public init() {}

    /// Inform the VAD that TTS playback has started or stopped.
    ///
    /// - Parameter playing: `true` when TTS output is active.
    public func setTTSPlaying(_ playing: Bool) {
        isTTSPlaying = playing
        // When TTS stops, any in-progress speech that was a barge-in candidate
        // should NOT retrigger; the state machine handles this naturally because
        // bargeInHandler only fires while isTTSPlaying is true.
    }

    /// Reset the VAD to idle. Call this when starting a new recording session.
    public func reset() {
        state = .idle
    }

    /// Process a single 20 ms audio frame on the realtime thread.
    ///
    /// - Parameter frame: Captured PCM frame at 16 kHz mono.
    public func process(frame: AudioFrame) {
        let energy = rmsEnergy(frame.samples)
        let isSpeech = energy >= silenceThreshold
        let now = frame.timestamp

        switch state {
        case .idle:
            if isSpeech {
                state = .speaking(startedAt: now)
                // Barge-in evaluation begins on the next frame.
            }

        case .speaking(let startedAt):
            if isSpeech {
                // Speech continues — check barge-in condition.
                if isTTSPlaying, now.timeIntervalSince(startedAt) >= bargeInDuration {
                    bargeInHandler?()
                    // Return to idle; next speech segment will start fresh.
                    state = .idle
                }
                // else: still accumulating speech duration
            } else {
                // Silence began — transition to post-speech silence tracking.
                state = .silenceAfterSpeech(since: now)
            }

        case .silenceAfterSpeech(let since):
            if isSpeech {
                // Speech resumed; restart speech timer (not a new barge-in from scratch).
                state = .speaking(startedAt: now)
            } else if now.timeIntervalSince(since) >= endOfSpeechDuration {
                // Silence has held for ≥ 800 ms — declare end-of-speech.
                endOfSpeechHandler?()
                state = .idle
            }
            // else: still in silence window, waiting
        }
    }

    // MARK: - Private helpers

    /// Compute the RMS amplitude of an Int16 PCM frame, normalised to [0, 1].
    func rmsEnergy(_ samples: [Int16]) -> Float {
        guard !samples.isEmpty else { return 0 }
        let sumOfSquares = samples.reduce(Float(0)) { acc, s in
            let f = Float(s) / Float(Int16.max)
            return acc + f * f
        }
        return sqrt(sumOfSquares / Float(samples.count))
    }

    /// Compute simple pitch estimation via zero-crossing rate (Hz).
    /// Used to populate `HAKIAudioFeatures.pitchHz` (Req 4.1).
    func zeroCrossingRate(_ samples: [Int16], sampleRate: Double) -> Float {
        guard samples.count > 1 else { return 0 }
        var crossings = 0
        for i in 1..<samples.count {
            if (samples[i - 1] >= 0) != (samples[i] >= 0) {
                crossings += 1
            }
        }
        // ZCR in Hz: (crossings / 2) cycles over (count / sampleRate) seconds
        let durationSeconds = Double(samples.count) / sampleRate
        return Float(Double(crossings) / 2.0 / durationSeconds)
    }
}
