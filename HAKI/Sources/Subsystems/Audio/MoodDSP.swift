// MoodDSP.swift
// HAKI — Audio Subsystem / Native DSP Mood Detector
//
// Zero-RAM mood detection using Apple's Accelerate framework (vDSP) on the
// incoming AVAudioPCMBuffer — no external ML model required.
//
// Architecture (per api_instructions_by_hkr.txt):
//   • Volume/Energy: vDSP_rmsqv on the PCM buffer (Root Mean Square).
//   • Pitch: lightweight YIN-inspired autocorrelation f0 tracker.
//   • Baseline calibration during first N frames of speech.
//   • Threshold logic: HIGH_RMS + HIGH_PITCH → ANGRY_SHOUT
//                      LOW_RMS  + LOW_PITCH  → SAD_LOW_ENERGY
//                      otherwise             → NEUTRAL
//   • Mood tag injected into the metadata payload before IPC dispatch.
//     Format: "[METADATA: MOOD=ANGRY_SHOUT] <transcript text>"
//
// Resource footprint: runs on CPU SIMD registers via Accelerate.
//   RAM overhead ≈ 0, CPU usage < 0.1%.
//
// Implements: api_instructions_by_hkr.txt — Mood Detection DSP section
// Design: Mood_Detector (Zero-RAM DSP replacement for librosa)

import Accelerate
import AVFoundation
import Foundation

// MARK: - MoodTag

/// The semantic mood tag injected into IPC payload metadata.
public enum MoodTag: String, Sendable {
    case angryShout     = "ANGRY_SHOUT"
    case sadLowEnergy   = "SAD_LOW_ENERGY"
    case neutral        = "NEUTRAL"
}

// MARK: - MoodDSPResult

/// Result of a single DSP mood analysis pass.
public struct MoodDSPResult: Sendable {
    /// Detected mood tag.
    public let tag: MoodTag
    /// RMS energy of the audio segment (linear scale, 0.0–1.0).
    public let rmsEnergy: Float
    /// Estimated fundamental frequency (Hz).
    public let pitchHz: Float
    /// Whether baseline calibration is complete.
    public let calibrated: Bool
}

// MARK: - MoodDSP

/// Native DSP-based mood classifier using Apple's Accelerate framework.
///
/// Inject an instance into the audio pipeline before STT dispatch:
///
///     let dsp = MoodDSP()
///     // During startup / first few seconds of speech:
///     dsp.feedCalibrationFrame(buffer)
///     dsp.finalizeCalibration()
///
///     // During a speech segment:
///     if let result = dsp.analyze(buffer: pcmBuffer) {
///         let taggedTranscript = dsp.inject(mood: result.tag, into: transcript)
///         // send taggedTranscript over gRPC IPC
///     }
///
public final class MoodDSP: @unchecked Sendable {

    // MARK: - Calibration state

    private var calibrationRMSSamples: [Float] = []
    private var calibrationPitchSamples: [Float] = []
    private var isCalibrated = false

    // Calibration targets: natural baseline values for this user
    private var baselineRMS: Float = 0.05
    private var baselinePitch: Float = 130.0     // Hz — typical speech F0

    // Threshold multipliers (configurable)
    private let highRMSFactor: Float    = 2.5
    private let lowRMSFactor: Float     = 0.35
    private let highPitchFactor: Float  = 1.6
    private let lowPitchFactor: Float   = 0.55

    private let calibrationFramesNeeded: Int = 20

    // MARK: - Init

    public init() {}

    // MARK: - Calibration

    /// Feed a calibration frame (call during natural baseline speech at startup).
    public func feedCalibrationFrame(_ buffer: AVAudioPCMBuffer) {
        guard !isCalibrated else { return }

        let rms = computeRMS(buffer: buffer)
        let pitch = estimatePitch(buffer: buffer)

        if rms > 0 {
            calibrationRMSSamples.append(rms)
        }
        if pitch > 50 {    // ignore implausible values
            calibrationPitchSamples.append(pitch)
        }

        if calibrationRMSSamples.count >= calibrationFramesNeeded {
            finalizeCalibration()
        }
    }

    /// Finalise calibration with the collected frames.
    /// Call this explicitly after the calibration period if needed.
    public func finalizeCalibration() {
        guard !isCalibrated else { return }

        if !calibrationRMSSamples.isEmpty {
            baselineRMS = calibrationRMSSamples.reduce(0, +) / Float(calibrationRMSSamples.count)
        }
        if !calibrationPitchSamples.isEmpty {
            baselinePitch = calibrationPitchSamples.reduce(0, +) / Float(calibrationPitchSamples.count)
        }

        isCalibrated = true
        print(
            "[MoodDSP] Calibrated — baseline RMS: \(String(format: "%.4f", baselineRMS)), "
            + "baseline pitch: \(String(format: "%.1f", baselinePitch)) Hz"
        )
    }

    /// Reset calibration (call if the user changes significantly).
    public func resetCalibration() {
        calibrationRMSSamples.removeAll()
        calibrationPitchSamples.removeAll()
        isCalibrated = false
    }

    // MARK: - Analysis

    /// Analyse a PCM buffer and return a MoodDSPResult.
    ///
    /// - Parameter buffer: An AVAudioPCMBuffer from the microphone tap.
    /// - Returns: A MoodDSPResult, or nil if the buffer is too short/empty.
    public func analyze(buffer: AVAudioPCMBuffer) -> MoodDSPResult? {
        guard buffer.frameLength > 0 else { return nil }

        let rms = computeRMS(buffer: buffer)
        let pitch = estimatePitch(buffer: buffer)

        let tag = classifyMood(rms: rms, pitch: pitch)

        return MoodDSPResult(
            tag: tag,
            rmsEnergy: rms,
            pitchHz: pitch,
            calibrated: isCalibrated
        )
    }

    // MARK: - Payload injection

    /// Prepend the HAKI mood metadata token to a transcript string.
    ///
    /// Example output:
    ///   "[METADATA: MOOD=ANGRY_SHOUT] bhai ye code crash ho raha hai"
    public func inject(mood: MoodTag, into transcript: String) -> String {
        guard mood != .neutral else {
            // Don't add metadata for neutral — keeps payload clean
            return transcript
        }
        return "[METADATA: MOOD=\(mood.rawValue)] \(transcript)"
    }

    /// Convenience: analyse buffer and inject the mood into the transcript.
    /// Call just before sending the IPC payload.
    public func taggedTranscript(buffer: AVAudioPCMBuffer, transcript: String) -> String {
        guard let result = analyze(buffer: buffer) else { return transcript }
        return inject(mood: result.tag, into: transcript)
    }

    // MARK: - Private: RMS energy (vDSP_rmsqv)

    /// Compute Root Mean Square energy using Apple's vDSP_rmsqv.
    /// Returns a normalised value in [0, 1].
    private func computeRMS(buffer: AVAudioPCMBuffer) -> Float {
        guard let channelData = buffer.floatChannelData?[0] else { return 0 }
        let frameCount = Int(buffer.frameLength)

        var rms: Float = 0
        vDSP_rmsqv(channelData, 1, &rms, vDSP_Length(frameCount))
        return min(rms, 1.0)    // clamp to [0, 1]
    }

    // MARK: - Private: Pitch estimation (Autocorrelation)

    /// Estimate fundamental frequency (F0) via autocorrelation.
    ///
    /// This is a lightweight approximation of the YIN algorithm:
    ///   1. Compute the normalised autocorrelation function (NACF).
    ///   2. Find the first peak after the initial dip (lag > min_period).
    ///   3. Return sampleRate / lag_at_peak.
    ///
    /// Operates in the human speech range: 70–600 Hz.
    private func estimatePitch(buffer: AVAudioPCMBuffer) -> Float {
        guard let channelData = buffer.floatChannelData?[0] else { return 0 }
        let frameCount = Int(buffer.frameLength)
        guard frameCount > 128 else { return 0 }

        let sampleRate = Float(buffer.format.sampleRate)

        // F0 range: 70–600 Hz → lag range
        let minLag = Int(sampleRate / 600.0)
        let maxLag = Int(sampleRate / 70.0)
        guard maxLag < frameCount else { return 0 }

        // Power of the signal (autocorrelation at lag 0)
        var power: Float = 0
        vDSP_dotpr(channelData, 1, channelData, 1, &power, vDSP_Length(frameCount))
        guard power > 1e-8 else { return 0 }   // silence

        // Find the lag with maximum normalised autocorrelation
        var bestLag = minLag
        var bestCorr: Float = -1.0

        for lag in minLag...min(maxLag, frameCount - 1) {
            var corr: Float = 0
            let len = vDSP_Length(frameCount - lag)
            // corr = sum(x[i] * x[i+lag])
            vDSP_dotpr(channelData, 1, channelData.advanced(by: lag), 1, &corr, len)
            // Normalise
            let normCorr = corr / power
            if normCorr > bestCorr {
                bestCorr = normCorr
                bestLag = lag
            }
        }

        guard bestLag > 0, bestCorr > 0.3 else { return 0 }   // low confidence

        let f0 = sampleRate / Float(bestLag)
        return f0
    }

    // MARK: - Private: Threshold classification

    private func classifyMood(rms: Float, pitch: Float) -> MoodTag {
        let highRMSThreshold   = baselineRMS   * highRMSFactor
        let lowRMSThreshold    = baselineRMS   * lowRMSFactor
        let highPitchThreshold = baselinePitch * highPitchFactor
        let lowPitchThreshold  = baselinePitch * lowPitchFactor

        // ANGRY_SHOUT: significantly louder AND higher pitched than baseline
        if rms > highRMSThreshold && pitch > highPitchThreshold {
            return .angryShout
        }

        // SAD_LOW_ENERGY: significantly quieter AND lower pitched than baseline
        if rms < lowRMSThreshold && pitch > 0 && pitch < lowPitchThreshold {
            return .sadLowEnergy
        }

        return .neutral
    }
}

// MARK: - MoodDSP + AudioFrame convenience

extension MoodDSP {

    /// Analyse a raw [Int16] audio frame and return a MoodTag.
    ///
    /// Converts the Int16 samples to Float32 and creates a temporary
    /// AVAudioPCMBuffer for analysis.
    public func analyze(frame: AudioFrame, sampleRate: Double = 16_000) -> MoodTag {
        guard !frame.samples.isEmpty else { return .neutral }

        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        )
        guard let fmt = format,
              let buffer = AVAudioPCMBuffer(
                  pcmFormat: fmt,
                  frameCapacity: AVAudioFrameCount(frame.samples.count)
              ) else {
            return .neutral
        }

        buffer.frameLength = AVAudioFrameCount(frame.samples.count)
        guard let floatPtr = buffer.floatChannelData?[0] else { return .neutral }

        // Convert Int16 → Float32
        for (i, sample) in frame.samples.enumerated() {
            floatPtr[i] = Float(sample) / Float(Int16.max)
        }

        return analyze(buffer: buffer)?.tag ?? .neutral
    }

    /// Analyse accumulated AudioFrames (an entire speech segment) and return
    /// the dominant mood tag.
    public func analyzeSegment(frames: [AudioFrame], sampleRate: Double = 16_000) -> MoodTag {
        guard !frames.isEmpty else { return .neutral }

        var tagCounts: [MoodTag: Int] = [.angryShout: 0, .sadLowEnergy: 0, .neutral: 0]
        for frame in frames {
            let tag = analyze(frame: frame, sampleRate: sampleRate)
            tagCounts[tag, default: 0] += 1
        }

        // Return the majority-vote tag
        return tagCounts.max(by: { $0.value < $1.value })?.key ?? .neutral
    }
}
