// VoiceAudioController.swift
// HAKI — strict VoiceProcessingIO capture lifecycle for the local voice runtime.
//
// This controller deliberately does not fall back to plain microphone capture.
// Frames are accepted only after VoiceProcessingIO is both enabled and verified.

import AVFoundation
import Foundation

public enum VoiceAudioRecoveryAction: String, Sendable, Equatable {
    case enableMicrophonePermission
    case reconnectMicrophone
    case retryCapture
    case restartAudioService

    public var userInstruction: String {
        switch self {
        case .enableMicrophonePermission:
            return "Enable HAKI in System Settings → Privacy & Security → Microphone, then retry voice."
        case .reconnectMicrophone:
            return "Connect or select a working microphone input, then retry voice."
        case .retryCapture:
            return "Retry voice capture. If this continues, reconnect the microphone and try again."
        case .restartAudioService:
            return "Restart the audio service or reconnect the microphone, then retry voice."
        }
    }
}

public enum VoiceAudioUnavailableReason: String, Error, Sendable, Equatable {
    case microphonePermission
    case inputRoute
    case voiceProcessingUnavailable
    case tapInstallation
    case engineStart
    case mediaServicesReset

    public var recoveryAction: VoiceAudioRecoveryAction {
        switch self {
        case .microphonePermission:
            return .enableMicrophonePermission
        case .inputRoute:
            return .reconnectMicrophone
        case .voiceProcessingUnavailable, .tapInstallation, .engineStart:
            return .retryCapture
        case .mediaServicesReset:
            return .restartAudioService
        }
    }

    public var userMessage: String {
        switch self {
        case .microphonePermission:
            return "Voice capture is unavailable because microphone permission was not granted. \(recoveryAction.userInstruction)"
        case .inputRoute:
            return "Voice capture is unavailable because no usable microphone route is selected. \(recoveryAction.userInstruction)"
        case .voiceProcessingUnavailable:
            return "Voice capture is unavailable because acoustic echo cancellation could not be enabled. \(recoveryAction.userInstruction)"
        case .tapInstallation:
            return "Voice capture is unavailable because HAKI could not install the microphone tap. \(recoveryAction.userInstruction)"
        case .engineStart:
            return "Voice capture is unavailable because the audio engine could not start. \(recoveryAction.userInstruction)"
        case .mediaServicesReset:
            return "Voice capture is unavailable because the macOS audio service was reset. \(recoveryAction.userInstruction)"
        }
    }
}

public enum VoiceAudioControllerError: Error, Sendable, Equatable {
    case invalidState
    case unavailable(VoiceAudioUnavailableReason)
}

public enum VoiceAudioControllerState: Sendable, Equatable {
    case idle
    case configuring
    case capturing
    /// Playback uses the same engine and leaves the input tap installed.
    case playing
    case stopping
    case retrying(attempt: Int, delayMilliseconds: Int)
    case unavailable(VoiceAudioUnavailableReason)
}

public enum VoiceAudioDiagnosticStage: String, Sendable, Equatable {
    case voiceProcessing = "voice_processing"
}

/// A privacy-safe native diagnostic that can be bridged to the session
/// diagnostics store. It deliberately carries no samples or transcript text.
public struct VoiceAudioDiagnosticEvent: Sendable, Equatable {
    public let stage: VoiceAudioDiagnosticStage
    public let reason: VoiceAudioUnavailableReason
    public let recoveryAction: VoiceAudioRecoveryAction
    public let capturedAtMonotonicNs: UInt64

    public init(
        stage: VoiceAudioDiagnosticStage = .voiceProcessing,
        reason: VoiceAudioUnavailableReason,
        recoveryAction: VoiceAudioRecoveryAction,
        capturedAtMonotonicNs: UInt64
    ) {
        self.stage = stage
        self.reason = reason
        self.recoveryAction = recoveryAction
        self.capturedAtMonotonicNs = capturedAtMonotonicNs
    }
}

/// A copied native input buffer. The live runtime copies AVAudioPCMBuffer data
/// before it returns from the AVFoundation tap callback, allowing normalization
/// to happen on the controller's serial lifecycle queue.
public struct VoiceAudioInputBuffer: Sendable {
    /// Interleaved Float32 samples in the source route's native format.
    public let interleavedSamples: [Float]
    public let sourceSampleRateHz: Double
    public let channelCount: UInt8
    public let capturedAtMonotonicNs: UInt64

    public init(
        interleavedSamples: [Float],
        sourceSampleRateHz: Double,
        channelCount: UInt8,
        capturedAtMonotonicNs: UInt64 = DispatchTime.now().uptimeNanoseconds
    ) {
        self.interleavedSamples = interleavedSamples
        self.sourceSampleRateHz = sourceSampleRateHz
        self.channelCount = channelCount
        self.capturedAtMonotonicNs = capturedAtMonotonicNs
    }
}

/// Normalized microphone data for local ASR/VAD and the local-only audio ring.
/// It is never a transcript/control socket payload.
public struct VoiceAudioFrame: Sendable, Equatable {
    public let sessionID: UUID
    /// Strictly increasing for the entire capture session, including recovery.
    public let sequence: UInt64
    /// Strictly increasing monotonic capture time in nanoseconds.
    public let capturedAtMonotonicNs: UInt64
    public let sampleRateHz: Int
    public let channels: UInt8
    public let pcmS16LE: Data

    public init(
        sessionID: UUID,
        sequence: UInt64,
        capturedAtMonotonicNs: UInt64,
        sampleRateHz: Int,
        channels: UInt8,
        pcmS16LE: Data
    ) {
        self.sessionID = sessionID
        self.sequence = sequence
        self.capturedAtMonotonicNs = capturedAtMonotonicNs
        self.sampleRateHz = sampleRateHz
        self.channels = channels
        self.pcmS16LE = pcmS16LE
    }
}

public struct VoiceAudioInputFormat: Sendable, Equatable {
    public let sampleRateHz: Double
    public let channelCount: UInt8

    public init(sampleRateHz: Double, channelCount: UInt8) {
        self.sampleRateHz = sampleRateHz
        self.channelCount = channelCount
    }
}

/// Testable boundary around the one AVAudioEngine owned by a voice session.
/// Production uses `SystemVoiceAudioRuntime`; focused tests use a deterministic
/// implementation to exercise start order and error recovery without hardware.
public protocol VoiceAudioRuntime: AnyObject, Sendable {
    var microphoneAuthorization: AVAuthorizationStatus { get }
    var inputFormat: VoiceAudioInputFormat { get }
    var isVoiceProcessingEnabled: Bool { get }

    func setVoiceProcessingEnabled(_ enabled: Bool) throws
    func installInputTap(
        bufferSize: AVAudioFrameCount,
        handler: @escaping @Sendable (VoiceAudioInputBuffer) -> Void
    ) throws
    func removeInputTap()
    func startEngine() throws
    func stopEngine()
    func setConfigurationChangeHandler(_ handler: (@Sendable () -> Void)?)
}

public typealias VoiceAudioRetryScheduler = @Sendable (
    _ delay: TimeInterval,
    _ action: @escaping @Sendable () -> Void
) -> Void

/// A serial dispatch-bound controller. AVFoundation lifecycle calls and capture
/// sequencing run on one queue; the AVFoundation tap only copies native samples
/// and enqueues them, so downstream ASR/VAD consumers never block that callback.
public final class VoiceAudioController: @unchecked Sendable {
    public static let normalizedSampleRateHz = 16_000
    public static let resetRetryDelays: [TimeInterval] = [0.25, 0.5, 1.0]

    public let sessionID: UUID
    public let frames: AsyncStream<VoiceAudioFrame>
    public let stateUpdates: AsyncStream<VoiceAudioControllerState>
    public let diagnostics: AsyncStream<VoiceAudioDiagnosticEvent>

    /// The session-owned engine when the production runtime is in use. Playback
    /// task 4.2 attaches its player node to this exact engine.
    public var audioEngine: AVAudioEngine? {
        (runtime as? SystemVoiceAudioRuntime)?.engine
    }

    public var state: VoiceAudioControllerState {
        onLifecycleQueue { stateStorage }
    }

    private let runtime: any VoiceAudioRuntime
    private let lifecycleQueue = DispatchQueue(label: "com.haki.voice.audio.lifecycle", qos: .userInitiated)
    private let lifecycleQueueKey = DispatchSpecificKey<UInt8>()
    private let retryScheduler: VoiceAudioRetryScheduler

    private let frameContinuation: AsyncStream<VoiceAudioFrame>.Continuation
    private let stateContinuation: AsyncStream<VoiceAudioControllerState>.Continuation
    private let diagnosticContinuation: AsyncStream<VoiceAudioDiagnosticEvent>.Continuation

    private var stateStorage: VoiceAudioControllerState = .idle
    private var nextSequence: UInt64 = 0
    private var lastCapturedAtMonotonicNs: UInt64 = 0
    private var retryAttempt = 0
    private var recoveryGeneration: UInt64 = 0

    public convenience init(sessionID: UUID = UUID()) {
        self.init(
            sessionID: sessionID,
            runtime: SystemVoiceAudioRuntime(),
            retryScheduler: Self.defaultRetryScheduler
        )
    }

    public convenience init(sessionID: UUID = UUID(), runtime: any VoiceAudioRuntime) {
        self.init(
            sessionID: sessionID,
            runtime: runtime,
            retryScheduler: Self.defaultRetryScheduler
        )
    }

    public init(
        sessionID: UUID = UUID(),
        runtime: any VoiceAudioRuntime,
        retryScheduler: @escaping VoiceAudioRetryScheduler
    ) {
        self.sessionID = sessionID
        self.runtime = runtime
        self.retryScheduler = retryScheduler

        let frameStream = AsyncStream<VoiceAudioFrame>.makeStream(bufferingPolicy: .bufferingNewest(64))
        frames = frameStream.stream
        frameContinuation = frameStream.continuation

        let stateStream = AsyncStream<VoiceAudioControllerState>.makeStream(bufferingPolicy: .bufferingNewest(32))
        stateUpdates = stateStream.stream
        stateContinuation = stateStream.continuation

        let diagnosticStream = AsyncStream<VoiceAudioDiagnosticEvent>.makeStream(bufferingPolicy: .bufferingNewest(32))
        diagnostics = diagnosticStream.stream
        diagnosticContinuation = diagnosticStream.continuation

        lifecycleQueue.setSpecific(key: lifecycleQueueKey, value: 1)
        runtime.setConfigurationChangeHandler { [weak self] in
            self?.handleAudioRouteChange()
        }
    }

    deinit {
        runtime.setConfigurationChangeHandler(nil)
        onLifecycleQueue {
            recoveryGeneration &+= 1
            stopEngineAndRemoveTapLocked()
        }
        frameContinuation.finish()
        stateContinuation.finish()
        diagnosticContinuation.finish()
    }

    /// Starts native capture with the required enable → verify → tap → engine
    /// order. There is no non-VoiceProcessingIO fallback.
    public func startCapture() throws {
        try onLifecycleQueue {
            guard canStartFromCurrentStateLocked() else {
                throw VoiceAudioControllerError.invalidState
            }
            recoveryGeneration &+= 1
            retryAttempt = 0
            do {
                try configureAndStartCaptureLocked()
            } catch let reason as VoiceAudioUnavailableReason {
                transitionToUnavailableLocked(reason)
                throw VoiceAudioControllerError.unavailable(reason)
            }
        }
    }

    /// Explicit user retry after an unavailable state. It cancels any pending
    /// automatic reset recovery and repeats the strict start order once.
    public func retryCapture() throws {
        try onLifecycleQueue {
            guard case .unavailable = stateStorage else {
                throw VoiceAudioControllerError.invalidState
            }
            recoveryGeneration &+= 1
            retryAttempt = 0
            do {
                try configureAndStartCaptureLocked()
            } catch let reason as VoiceAudioUnavailableReason {
                transitionToUnavailableLocked(reason)
                throw VoiceAudioControllerError.unavailable(reason)
            }
        }
    }

    public func stopCapture() {
        onLifecycleQueue {
            guard stateStorage != .idle else { return }
            recoveryGeneration &+= 1
            transitionLocked(.stopping)
            stopEngineAndRemoveTapLocked()
            retryAttempt = 0
            transitionLocked(.idle)
        }
    }

    /// Called by the renderer immediately before it schedules output on this
    /// controller's `audioEngine`. It changes UI/lifecycle state only: capture
    /// remains installed and continues receiving VoiceProcessingIO frames.
    public func playbackDidStart() throws {
        try onLifecycleQueue {
            guard case .capturing = stateStorage else {
                throw VoiceAudioControllerError.invalidState
            }
            guard runtime.isVoiceProcessingEnabled else {
                transitionToUnavailableLocked(.voiceProcessingUnavailable)
                throw VoiceAudioControllerError.unavailable(.voiceProcessingUnavailable)
            }
            transitionLocked(.playing)
        }
    }

    public func playbackDidStop() {
        onLifecycleQueue {
            guard case .playing = stateStorage else { return }
            transitionLocked(.capturing)
        }
    }

    /// Route/configuration changes are handled serially. The controller stops
    /// the old tap, rebuilds VoiceProcessingIO, and retries at most three times.
    public func handleAudioRouteChange() {
        lifecycleQueue.async { [weak self] in
            self?.beginBoundedRecoveryLocked(reason: .inputRoute)
        }
    }

    /// Call from media-services-reset/interruption integration. It uses the
    /// same bounded reset policy and leaves capture unavailable after exhaustion.
    public func handleMediaServicesReset() {
        lifecycleQueue.async { [weak self] in
            self?.beginBoundedRecoveryLocked(reason: .mediaServicesReset)
        }
    }

    private func canStartFromCurrentStateLocked() -> Bool {
        switch stateStorage {
        case .idle, .unavailable:
            return true
        case .configuring, .capturing, .playing, .stopping, .retrying:
            return false
        }
    }

    private func configureAndStartCaptureLocked() throws {
        transitionLocked(.configuring)

        guard runtime.microphoneAuthorization == .authorized else {
            throw VoiceAudioUnavailableReason.microphonePermission
        }

        let inputFormat = runtime.inputFormat
        guard inputFormat.sampleRateHz.isFinite,
              inputFormat.sampleRateHz > 0,
              inputFormat.channelCount > 0 else {
            throw VoiceAudioUnavailableReason.inputRoute
        }

        do {
            // This must precede tap installation. Do not weaken it to a
            // best-effort setting: AEC is a session precondition.
            try runtime.setVoiceProcessingEnabled(true)
        } catch {
            throw VoiceAudioUnavailableReason.voiceProcessingUnavailable
        }
        guard runtime.isVoiceProcessingEnabled else {
            throw VoiceAudioUnavailableReason.voiceProcessingUnavailable
        }

        let nativeTapFrames = max(1, Int((inputFormat.sampleRateHz * 0.020).rounded()))
        do {
            try runtime.installInputTap(bufferSize: AVAudioFrameCount(nativeTapFrames)) { [weak self] input in
                self?.receiveTapInput(input)
            }
        } catch {
            throw VoiceAudioUnavailableReason.tapInstallation
        }

        do {
            try runtime.startEngine()
        } catch {
            runtime.removeInputTap()
            throw VoiceAudioUnavailableReason.engineStart
        }

        transitionLocked(.capturing)
    }

    /// The AVFoundation callback invokes this after copying its buffer. The
    /// serial queue preserves the arrival order before sequence assignment.
    private func receiveTapInput(_ input: VoiceAudioInputBuffer) {
        lifecycleQueue.async { [weak self] in
            self?.acceptTapInputLocked(input)
        }
    }

    private func acceptTapInputLocked(_ input: VoiceAudioInputBuffer) {
        guard case .capturing = stateStorage else {
            guard case .playing = stateStorage else { return }
            acceptFrameWhilePlayingLocked(input)
            return
        }
        acceptNormalizedFrameLocked(input)
    }

    private func acceptFrameWhilePlayingLocked(_ input: VoiceAudioInputBuffer) {
        // Full duplex is intentional. The input tap remains active while the
        // same engine renders TTS, allowing VoiceProcessingIO to supply AEC.
        acceptNormalizedFrameLocked(input)
    }

    private func acceptNormalizedFrameLocked(_ input: VoiceAudioInputBuffer) {
        // Verification is repeated at the final gate. A route reset or an
        // external API change can disable voice processing after startup; in
        // that case not one frame may reach ASR/VAD/ring consumers.
        guard runtime.isVoiceProcessingEnabled else {
            transitionToUnavailableLocked(.voiceProcessingUnavailable)
            return
        }
        guard let samples = normalizeTo16kMono(input) else { return }

        nextSequence &+= 1
        let timestamp: UInt64
        if input.capturedAtMonotonicNs > lastCapturedAtMonotonicNs {
            timestamp = input.capturedAtMonotonicNs
        } else {
            timestamp = lastCapturedAtMonotonicNs &+ 1
        }
        lastCapturedAtMonotonicNs = timestamp

        let frame = VoiceAudioFrame(
            sessionID: sessionID,
            sequence: nextSequence,
            capturedAtMonotonicNs: timestamp,
            sampleRateHz: Self.normalizedSampleRateHz,
            channels: 1,
            pcmS16LE: pcmS16LE(from: samples)
        )
        _ = frameContinuation.yield(frame)
    }

    private func beginBoundedRecoveryLocked(reason: VoiceAudioUnavailableReason) {
        guard isCaptureActiveLocked else { return }
        recoveryGeneration &+= 1
        retryAttempt = 0
        stopEngineAndRemoveTapLocked()
        scheduleNextRecoveryAttemptLocked(reason: reason, generation: recoveryGeneration)
    }

    private var isCaptureActiveLocked: Bool {
        switch stateStorage {
        case .configuring, .capturing, .playing, .retrying:
            return true
        case .idle, .stopping, .unavailable:
            return false
        }
    }

    private func scheduleNextRecoveryAttemptLocked(
        reason: VoiceAudioUnavailableReason,
        generation: UInt64
    ) {
        guard retryAttempt < Self.resetRetryDelays.count else {
            transitionToUnavailableLocked(reason)
            return
        }

        let delay = Self.resetRetryDelays[retryAttempt]
        retryAttempt += 1
        transitionLocked(.retrying(
            attempt: retryAttempt,
            delayMilliseconds: Int((delay * 1_000).rounded())
        ))

        retryScheduler(delay) { [weak self] in
            self?.lifecycleQueue.async {
                guard let self, self.recoveryGeneration == generation else { return }
                do {
                    try self.configureAndStartCaptureLocked()
                    self.retryAttempt = 0
                } catch let nextReason as VoiceAudioUnavailableReason {
                    self.stopEngineAndRemoveTapLocked()
                    self.scheduleNextRecoveryAttemptLocked(reason: nextReason, generation: generation)
                } catch {
                    self.stopEngineAndRemoveTapLocked()
                    self.scheduleNextRecoveryAttemptLocked(reason: reason, generation: generation)
                }
            }
        }
    }

    private func transitionToUnavailableLocked(_ reason: VoiceAudioUnavailableReason) {
        recoveryGeneration &+= 1
        stopEngineAndRemoveTapLocked()
        transitionLocked(.unavailable(reason))
        _ = diagnosticContinuation.yield(VoiceAudioDiagnosticEvent(
            reason: reason,
            recoveryAction: reason.recoveryAction,
            capturedAtMonotonicNs: DispatchTime.now().uptimeNanoseconds
        ))
    }

    private func transitionLocked(_ state: VoiceAudioControllerState) {
        stateStorage = state
        _ = stateContinuation.yield(state)
    }

    private func stopEngineAndRemoveTapLocked() {
        runtime.removeInputTap()
        runtime.stopEngine()
    }

    private func normalizeTo16kMono(_ input: VoiceAudioInputBuffer) -> [Int16]? {
        let channels = Int(input.channelCount)
        guard channels > 0,
              input.sourceSampleRateHz.isFinite,
              input.sourceSampleRateHz > 0 else {
            return nil
        }

        let sourceFrameCount = input.interleavedSamples.count / channels
        guard sourceFrameCount > 0 else { return nil }

        let targetCount = max(
            1,
            Int((Double(sourceFrameCount) * Double(Self.normalizedSampleRateHz) / input.sourceSampleRateHz).rounded(.down))
        )

        func monoSample(at frameIndex: Int) -> Float {
            let offset = frameIndex * channels
            var sum: Float = 0
            for channel in 0..<channels {
                sum += input.interleavedSamples[offset + channel]
            }
            return sum / Float(channels)
        }

        let sourcePerTarget = input.sourceSampleRateHz / Double(Self.normalizedSampleRateHz)
        var result = [Int16](repeating: 0, count: targetCount)
        for targetIndex in 0..<targetCount {
            let position = Double(targetIndex) * sourcePerTarget
            let lower = min(Int(position), sourceFrameCount - 1)
            let upper = min(lower + 1, sourceFrameCount - 1)
            let fraction = Float(position - Double(lower))
            let sample = monoSample(at: lower) * (1 - fraction) + monoSample(at: upper) * fraction
            let clamped = min(1, max(-1, sample))
            result[targetIndex] = Int16((clamped * Float(Int16.max)).rounded())
        }
        return result
    }

    private func pcmS16LE(from samples: [Int16]) -> Data {
        let littleEndianSamples = samples.map { $0.littleEndian }
        return littleEndianSamples.withUnsafeBytes { Data($0) }
    }

    private func onLifecycleQueue<T>(_ operation: () throws -> T) rethrows -> T {
        if DispatchQueue.getSpecific(key: lifecycleQueueKey) != nil {
            return try operation()
        }
        return try lifecycleQueue.sync(execute: operation)
    }

    private static let defaultRetryScheduler: VoiceAudioRetryScheduler = { delay, action in
        DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + delay, execute: action)
    }
}

/// Production AVFoundation adapter. It creates exactly one AVAudioEngine per
/// controller/session, supplies both input and future player-node playback, and
/// reports engine configuration changes back to the controller for recovery.
private final class SystemVoiceAudioRuntime: VoiceAudioRuntime, @unchecked Sendable {
    let engine = AVAudioEngine()

    private let lock = NSLock()
    private var configurationChangeHandler: (@Sendable () -> Void)?
    private var configurationObserver: NSObjectProtocol?

    init() {
        configurationObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: engine,
            queue: nil
        ) { [weak self] _ in
            guard let self else { return }
            let handler = self.lock.withLock { self.configurationChangeHandler }
            handler?()
        }
    }

    deinit {
        if let configurationObserver {
            NotificationCenter.default.removeObserver(configurationObserver)
        }
    }

    var microphoneAuthorization: AVAuthorizationStatus {
        AVCaptureDevice.authorizationStatus(for: .audio)
    }

    var inputFormat: VoiceAudioInputFormat {
        let format = engine.inputNode.inputFormat(forBus: 0)
        return VoiceAudioInputFormat(
            sampleRateHz: format.sampleRate,
            channelCount: UInt8(clamping: Int(format.channelCount))
        )
    }

    var isVoiceProcessingEnabled: Bool {
        engine.inputNode.isVoiceProcessingEnabled
    }

    func setVoiceProcessingEnabled(_ enabled: Bool) throws {
        try engine.inputNode.setVoiceProcessingEnabled(enabled)
    }

    func installInputTap(
        bufferSize: AVAudioFrameCount,
        handler: @escaping @Sendable (VoiceAudioInputBuffer) -> Void
    ) throws {
        let inputNode = engine.inputNode
        let format = inputNode.inputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0, format.channelCount <= AVAudioChannelCount(UInt8.max) else {
            throw VoiceAudioUnavailableReason.inputRoute
        }

        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: nil) { buffer, _ in
            guard let channelData = buffer.floatChannelData else { return }
            let frameCount = Int(buffer.frameLength)
            let channelCount = Int(buffer.format.channelCount)
            guard frameCount > 0, channelCount > 0 else { return }

            var copiedSamples = [Float](repeating: 0, count: frameCount * channelCount)
            for channel in 0..<channelCount {
                let source = channelData[channel]
                for frame in 0..<frameCount {
                    copiedSamples[frame * channelCount + channel] = source[frame]
                }
            }

            handler(VoiceAudioInputBuffer(
                interleavedSamples: copiedSamples,
                sourceSampleRateHz: buffer.format.sampleRate,
                channelCount: UInt8(clamping: channelCount),
                capturedAtMonotonicNs: DispatchTime.now().uptimeNanoseconds
            ))
        }
    }

    func removeInputTap() {
        engine.inputNode.removeTap(onBus: 0)
    }

    func startEngine() throws {
        try engine.start()
    }

    func stopEngine() {
        engine.stop()
    }

    func setConfigurationChangeHandler(_ handler: (@Sendable () -> Void)?) {
        lock.withLock {
            configurationChangeHandler = handler
        }
    }
}
