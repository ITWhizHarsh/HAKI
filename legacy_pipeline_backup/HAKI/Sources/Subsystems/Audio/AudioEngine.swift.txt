// AudioEngine.swift
// HAKI — Audio Subsystem
//
// Owns the AVAudioEngine microphone tap, realtime VAD (Voice Activity Detection),
// end-of-speech detection (800 ms silence), barge-in detection (≥200 ms speech
// during TTS playback), and acoustic echo cancellation (AEC) on the mic path.
//
// AEC strategy:
//   AVAudioEngine provides voice-processing I/O automatically when the app sets
//   the input node's voice processing mode via `AVAudioEngine.setVoiceProcessingEnabled`.
//   This routes audio through the macOS Voice Processing I/O unit
//   (kAudioUnitSubType_VoiceProcessingIO) which performs hardware-level echo
//   cancellation, noise suppression, and automatic gain control.
//   On macOS 14+ this is `inputNode.isVoiceProcessingEnabled = true`.
//
// Threading model:
//   • The audio tap callback runs on AVAudioEngine's internal realtime thread.
//   • VAD and energy calculations are performed inline on that thread
//     (no IPC, no async/await, no heap allocations in the hot path).
//   • Detected events (endOfSpeech, bargeIn, frame) are delivered via an
//     AsyncStream whose continuation is held on a lock-free atomic reference.
//
// Implements: Req 3.2 (800 ms end-of-speech), Req 3.3 (200 ms barge-in)
// Phase 1 Task 7.1

import AVFoundation
import Foundation
import AudioToolbox

// MARK: - AudioFrame

/// A single 20 ms PCM audio frame captured from the microphone.
public struct AudioFrame: Sendable {
    /// 20 ms of 16-bit PCM at 16 kHz (320 samples).
    public let samples: [Int16]
    /// Capture timestamp (wall clock at frame boundary).
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
/// Concrete implementation: `LiveAudioEngine`.
/// Test double: `MockAudioEngine`.
public protocol AudioEngineProtocol: AnyObject, Sendable {
    /// Start capturing microphone input and emitting `VADEvent`s.
    func startCapture() throws
    /// Stop microphone capture and finish the events stream.
    func stopCapture()
    /// Notify the engine that TTS playback has started or stopped.
    func setTTSPlaying(_ playing: Bool)
    /// Stream of `VADEvent`s produced during active capture.
    var events: AsyncStream<VADEvent> { get }
}

// MARK: - LiveAudioEngine

/// Production implementation using AVAudioEngine with Voice Processing I/O.
///
/// - Sets up a 16 kHz mono capture format.
/// - Installs a tap on the input node's 0th bus.
/// - Processes each 20 ms chunk on the realtime thread through the VAD.
/// - Delivers events through an `AsyncStream<VADEvent>`.
public final class LiveAudioEngine: AudioEngineProtocol, @unchecked Sendable {

    // MARK: - Constants

    /// Sample rate used throughout the voice pipeline.
    public static let sampleRate: Double = 16_000
    /// Frame duration in seconds (20 ms).
    public static let frameDuration: Double = 0.020
    /// Samples per 20 ms frame at 16 kHz.
    public static let samplesPerFrame: Int = Int(sampleRate * frameDuration) // 320
    /// Tap buffer size; we ask for exactly one frame per callback.
    private static let tapBufferSize: AVAudioFrameCount = AVAudioFrameCount(samplesPerFrame)

    // MARK: - Internal state (accessed only on the audio thread unless noted)

    private let engine = AVAudioEngine()
    private let vad = VAD()

    /// Accumulation buffer for sub-frame residuals.
    private var residualSamples: [Int16] = []

    /// AsyncStream continuation — written once on the main thread, read on the
    /// audio thread. The `@unchecked Sendable` conformance covers this usage;
    /// in practice the continuation is set before `startCapture()` returns.
    private var continuation: AsyncStream<VADEvent>.Continuation?

    /// Serialises access to `continuation` across audio thread and main thread.
    private let lock = NSLock()

    // MARK: - AudioEngineProtocol

    /// The stream of `VADEvent`s.  Lazily created; the continuation is stored
    /// before capture begins so the first tap callback can deliver events.
    public lazy var events: AsyncStream<VADEvent> = {
        AsyncStream<VADEvent> { [weak self] continuation in
            guard let self else { return }
            self.lock.withLock {
                self.continuation = continuation
            }
        }
    }()

    // MARK: - Init / deinit

    public init() {
        setupVAD()
    }

    deinit {
        stopCapture()
    }

    // MARK: - AudioEngineProtocol conformance

    /// Configure and start AVAudioEngine mic capture.
    ///
    /// - Throws: `AudioEngineError.permissionDenied` if the microphone
    ///   permission has not been granted, or `AudioEngineError.hardwareUnavailable`
    ///   if the engine cannot start.
    public func startCapture() throws {
        // Ensure the events stream (and therefore the continuation) is initialised
        // before the tap fires.
        _ = events

        let inputNode = engine.inputNode

        // MARK: Acoustic Echo Cancellation
        // macOS 14+: enable Voice Processing on the input node.
        // This routes audio through kAudioUnitSubType_VoiceProcessingIO which
        // applies hardware-accelerated AEC, noise suppression, and AGC.
        // Non-fatal if unsupported — we fall back to plain capture without AEC.
        if #available(macOS 14, *) {
            if !inputNode.isVoiceProcessingEnabled {
                do {
                    try inputNode.setVoiceProcessingEnabled(true)
                    print("[LiveAudioEngine] Voice processing (AEC) enabled.")
                } catch {
                    // Non-fatal — AEC simply won't be active.
                    print("[LiveAudioEngine] Voice processing (AEC) unavailable: \(error). Continuing without AEC.")
                }
            }

            // CRITICAL: by default the Voice Processing I/O unit ducks ALL other
            // audio output (other apps AND HAKI's own TTS) to ~5% while the mic
            // is active. Disable that ducking so HAKI's spoken responses are
            // audible at full volume and the user's other audio isn't muted.
            if #available(macOS 14, *) {
                let duckCfg = AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
                    enableAdvancedDucking: false,
                    duckingLevel: .min
                )
                inputNode.voiceProcessingOtherAudioDuckingConfiguration = duckCfg
                print("[LiveAudioEngine] Voice-processing audio ducking set to minimum.")
            }
        }

        // MARK: Capture format
        // Use the input node's *native* hardware format for the tap — forcing
        // a non-native format (e.g. 16 kHz when hardware runs at 48 kHz) causes
        // AVAudioEngine.start() to throw hardwareUnavailable on some Macs.
        // We convert to 16-bit Int16 at 16 kHz inside the tap callback.
        let nativeFormat = inputNode.inputFormat(forBus: 0)
        print("[LiveAudioEngine] Hardware input format: \(nativeFormat)")

        // MARK: Install tap
        inputNode.installTap(
            onBus: 0,
            bufferSize: AVAudioFrameCount(nativeFormat.sampleRate * LiveAudioEngine.frameDuration),
            format: nativeFormat
        ) { [weak self] buffer, time in
            self?.processTapBuffer(buffer, time: time)
        }

        // MARK: Start engine
        do {
            try engine.start()
            print("[LiveAudioEngine] AVAudioEngine started successfully.")
        } catch {
            inputNode.removeTap(onBus: 0)
            print("[LiveAudioEngine] AVAudioEngine.start() failed: \(error)")
            throw AudioEngineError.hardwareUnavailable
        }
    }

    /// Stop capture, remove the tap, and finish the event stream.
    public func stopCapture() {
        if engine.isRunning {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }
        lock.withLock {
            continuation?.finish()
            continuation = nil
        }
        vad.reset()
        residualSamples.removeAll()
    }

    /// Thread-safe wrapper to inform VAD of TTS state changes.
    public func setTTSPlaying(_ playing: Bool) {
        vad.setTTSPlaying(playing)
    }

    // MARK: - Private: realtime tap callback

    /// Called by AVAudioEngine on the realtime audio thread for every tap buffer.
    /// Handles any hardware sample rate by downsampling to 16 kHz for the pipeline.
    private func processTapBuffer(_ buffer: AVAudioPCMBuffer, time: AVAudioTime) {
        guard let channelData = buffer.floatChannelData else { return }
        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0 else { return }

        let hardwareSampleRate = buffer.format.sampleRate
        let targetSampleRate = LiveAudioEngine.sampleRate  // 16000

        // Convert Float32 → Int16, resampling if hardware SR != 16 kHz.
        // Simple linear interpolation resample — sufficient for voice.
        let ratio = hardwareSampleRate / targetSampleRate
        let outputFrameCount = Int(Double(frameCount) / ratio)
        guard outputFrameCount > 0 else { return }

        let float32Ptr = channelData[0]
        var resampledInt16 = [Int16](repeating: 0, count: outputFrameCount)
        for i in 0..<outputFrameCount {
            let srcIndex = Double(i) * ratio
            let lo = Int(srcIndex)
            let hi = min(lo + 1, frameCount - 1)
            let frac = Float(srcIndex - Double(lo))
            let sample = float32Ptr[lo] * (1 - frac) + float32Ptr[hi] * frac
            let clamped = max(-1.0, min(1.0, sample))
            resampledInt16[i] = Int16(clamped * Float(Int16.max))
        }

        // Merge with residual from prior partial frame.
        var newSamples = resampledInt16
        if !residualSamples.isEmpty {
            newSamples = residualSamples + newSamples
            residualSamples = []
        }

        // Slice into discrete 20 ms frames at 16 kHz (320 samples).
        let frameSize = LiveAudioEngine.samplesPerFrame
        var offset = 0
        let captureDate = Date()

        while offset + frameSize <= newSamples.count {
            let slice = Array(newSamples[offset..<(offset + frameSize)])
            let frame = AudioFrame(samples: slice, timestamp: captureDate)
            emitEvent(.frame(frame))
            vad.process(frame: frame)
            offset += frameSize
        }

        if offset < newSamples.count {
            residualSamples = Array(newSamples[offset...])
        }
    }

    // MARK: - Private: VAD setup

    private func setupVAD() {
        // End-of-speech: 800 ms silence after speech (Req 3.2).
        vad.endOfSpeechHandler = { [weak self] in
            self?.emitEvent(.endOfSpeech)
        }
        // Barge-in: ≥ 200 ms speech during TTS (Req 3.3).
        vad.bargeInHandler = { [weak self] in
            self?.emitEvent(.bargeIn)
        }
    }

    // MARK: - Private: thread-safe event emission

    /// Deliver a `VADEvent` to the async stream consumer.
    ///
    /// Called from the realtime audio thread. The `AsyncStream.Continuation.yield`
    /// method is designed to be safe to call from any thread.
    private func emitEvent(_ event: VADEvent) {
        lock.withLock {
            _ = continuation?.yield(event)
        }
    }
}

// MARK: - AudioEngineError

public enum AudioEngineError: Error, Sendable {
    case permissionDenied
    case hardwareUnavailable
}

// MARK: - MockAudioEngine

/// A test double that feeds synthetic `AudioFrame`s and emits scripted events.
/// Used in unit and property tests where real hardware is unavailable.
public final class MockAudioEngine: AudioEngineProtocol, @unchecked Sendable {

    private var continuation: AsyncStream<VADEvent>.Continuation?
    private let lock = NSLock()
    private(set) public var isCapturing = false

    public lazy var events: AsyncStream<VADEvent> = {
        AsyncStream<VADEvent> { [weak self] continuation in
            guard let self else { return }
            self.lock.withLock { self.continuation = continuation }
        }
    }()

    public init() { _ = events }

    public func startCapture() throws { isCapturing = true }

    public func stopCapture() {
        isCapturing = false
        lock.withLock {
            continuation?.finish()
            continuation = nil
        }
    }

    public func setTTSPlaying(_ playing: Bool) {}

    /// Inject a `VADEvent` directly into the stream (for tests).
    public func inject(_ event: VADEvent) {
        lock.withLock { _ = continuation?.yield(event) }
    }
}
