// TTSService.swift
// HAKI — Audio Subsystem / TTS Pipeline
//
// Sentence-chunked, streaming text-to-speech playback with barge-in
// cancellation and on-screen text fallback.
//
// Architecture (from the Voice Pipeline design):
//   LLM token stream
//       │
//       ▼
//   ClauseSegmenter   ← accumulates tokens, emits clauses on punctuation
//       │
//       ▼  (clause ready)
//   TTS synthesis     ← IPC → Core's TTS model  (primary path)
//       │              AVSpeechSynthesizer        (local fallback)
//       ▼
//   AVAudioPlayerNode ← feeds PCM chunks from Core TTS into the audio graph
//       │
//       ▼
//   Speaker output
//
// Key behaviours:
//   • `speak(textStream:)` consumes an `AsyncStream<String>` of LLM tokens.
//   • The first clause is synthesised immediately; subsequent clauses are
//     synthesised and queued while playback of the earlier clause is underway
//     (pipeline: generate-while-play).
//   • `VoiceEngine.notifyTTSStarted()` is called before the first chunk
//     reaches the speaker, so the VAD arms barge-in detection (Req 3.3).
//   • On barge-in (`VoiceEvent.bargeIn` from the `VoiceEngine` event stream),
//     playback stops immediately, all pending synthesis is cancelled, and
//     `VoiceEngine.notifyTTSStopped()` is called (Req 3.3).
//   • On TTS failure (IPC error or synthesis error), the full accumulated
//     response text is posted to `UIState` as a fallback and the user is
//     notified that audio playback was unavailable (Req 3.7).
//   • `cancel()` stops playback and synthesis cleanly at any point.
//
// Threading:
//   • `speak()` is `async`; all coordination runs on a `Task` spawned with
//     `.userInitiated` priority.
//   • `AVAudioPlayerNode` scheduling is dispatched via `playerLock` so it
//     is safe to call from concurrent synthesis tasks.
//   • `cancel()` and `handleBargeIn()` are synchronous and safe to call from
//     any thread.
//
// Implements: Req 3.1, 3.5, 3.7
// Design: Voice Pipeline, Voice_Engine
// Phase 1 Task 7.3

import AVFoundation
import Foundation
import HAKIIPC

// MARK: - Notification name (mirrors UIState.ttsFailedShowText)

/// Notification posted when TTS fails and the response must be shown as text.
/// Matches `Notification.Name.ttsFailedShowText` defined in HAKIUI/UIState.swift.
private extension Notification.Name {
    static let ttsFailedShowText = Notification.Name("haki.ttsFailedShowText")
}

// MARK: - TTSError

/// Errors emitted by `TTSService`.
public enum TTSError: Error, Sendable {
    /// The IPC client is not connected or the Core returned an error.
    case ipcUnavailable(String)
    /// AVAudioEngine graph could not be configured.
    case audioEngineSetupFailed
    /// Synthesis produced no audio data.
    case emptySynthesis
    /// Generic synthesis failure.
    case synthesisFailure(String)
}

// MARK: - TTSServiceProtocol

/// Abstract contract for the TTS coordinator.  The production implementation
/// is `TTSService`; a mock can be injected in tests.
public protocol TTSServiceProtocol: AnyObject, Sendable {
    /// Begin streaming TTS from an LLM token stream.
    ///
    /// - Parameters:
    ///   - textStream: Stream of LLM tokens as they arrive.
    ///   - voiceEvents: The `VoiceEngine` event stream so barge-in can be
    ///     detected concurrently with playback.
    ///   - onStarted: Called just before the first audio chunk reaches the
    ///     speaker (use to call `VoiceEngine.notifyTTSStarted()`).
    ///   - onStopped: Called when playback ends for any reason (barge-in,
    ///     completion, cancellation, or failure).
    func speak(
        textStream: AsyncStream<String>,
        voiceEvents: AsyncStream<VoiceEvent>,
        onStarted: @Sendable @escaping () -> Void,
        onStopped: @Sendable @escaping () -> Void
    ) async

    /// Cancel all in-flight synthesis and stop playback immediately.
    func cancel()
}

// MARK: - TTSService

/// Production TTS coordinator.
///
/// Primary synthesis path: forwards each synthesised clause to the HAKI Core
/// via the IPC channel (`ServerMessage.ttsAudioChunk`) and plays the returned
/// PCM audio through `AVAudioPlayerNode`.
///
/// Local fallback: if the IPC client is nil / disconnected, or if the Core
/// returns an error, synthesis falls back to `AVSpeechSynthesizer` for
/// immediate local playback without sending data off-device.
///
/// On any synthesis failure after fallback also fails, the full response text
/// is shown on screen via `UIState.postTTSFailure(responseText:)` (Req 3.7).
public final class TTSService: TTSServiceProtocol, @unchecked Sendable {

    // MARK: - Dependencies

    /// Optional IPC client for Core-based TTS.  If `nil` or not connected,
    /// the local `AVSpeechSynthesizer` is used immediately.
    public weak var ipcClient: (any IPCClientProtocol)?

    // MARK: - AVAudio graph

    /// Shared audio engine for TTS playback.  Separate from the capture
    /// engine so TTS playback and mic capture run concurrently.
    private let engine = AVAudioEngine()

    /// Player node fed by synthesised PCM chunks.
    private let playerNode = AVAudioPlayerNode()

    /// Lock guarding all access to `playerNode` scheduling.
    private let playerLock = NSLock()

    /// Whether the audio engine graph has been started.
    private var engineStarted = false

    // MARK: - Local speech synthesizer (fallback)

    private let localSynth = AVSpeechSynthesizer()

    // MARK: - Cancellation

    /// The top-level speak task; cancelled on `cancel()` or barge-in.
    private var speakTask: Task<Void, Never>?

    /// A task group token used to cancel in-flight synthesis sub-tasks.
    private let cancelLock = NSLock()
    private var _cancelled = false

    // MARK: - Init / deinit

    public init(ipcClient: (any IPCClientProtocol)? = nil) {
        self.ipcClient = ipcClient
        setupAudioGraph()
    }

    deinit {
        cancel()
        if engineStarted {
            engine.stop()
        }
    }

    // MARK: - TTSServiceProtocol

    /// Stream LLM tokens → segment into clauses → synthesise → play.
    ///
    /// The method returns only after playback has completed or been cancelled.
    public func speak(
        textStream: AsyncStream<String>,
        voiceEvents: AsyncStream<VoiceEvent>,
        onStarted: @Sendable @escaping () -> Void,
        onStopped: @Sendable @escaping () -> Void
    ) async {
        // Reset cancellation state for this turn.
        cancelLock.withLock { _cancelled = false }

        speakTask = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            await self.runPipeline(
                textStream: textStream,
                voiceEvents: voiceEvents,
                onStarted: onStarted,
                onStopped: onStopped
            )
        }

        // Await the task so `speak()` is itself async (caller can await).
        await speakTask?.value
    }

    /// Stop playback immediately and cancel all pending synthesis.
    ///
    /// Safe to call from any thread (e.g. from the VAD callback or from the
    /// Orchestrator barge-in handler).
    public func cancel() {
        cancelLock.withLock { _cancelled = true }
        speakTask?.cancel()
        speakTask = nil
        stopPlayback()
    }

    // MARK: - Private: pipeline

    private func runPipeline(
        textStream: AsyncStream<String>,
        voiceEvents: AsyncStream<VoiceEvent>,
        onStarted: @Sendable @escaping () -> Void,
        onStopped: @Sendable @escaping () -> Void
    ) async {
        // Accumulate the full response text for fallback (Req 3.7).
        var fullResponseText = ""
        var firstClausePlayed = false
        var ttsHardFailed = false

        // Watch for barge-in concurrently with the synthesis loop.
        let bargeInTask = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            for await event in voiceEvents {
                if case .bargeIn = event {
                    self.handleBargeIn(onStopped: onStopped)
                    return
                }
            }
        }
        defer { bargeInTask.cancel() }

        var segmenter = ClauseSegmenter()

        // ── Token accumulation + clause synthesis loop ──────────────────────
        for await token in textStream {
            guard !isCancelled else { break }

            fullResponseText.append(token)

            if let clause = segmenter.feed(token) {
                guard !isCancelled else { break }
                do {
                    let audioData = try await synthesise(clause: clause)
                    guard !isCancelled else { break }

                    if !firstClausePlayed {
                        // Arm barge-in detection before first audio.
                        onStarted()
                        firstClausePlayed = true
                    }
                    scheduleAudio(audioData, sampleRate: 22_050)
                } catch {
                    // Synthesis failed — mark for text fallback.
                    ttsHardFailed = true
                    print("[TTSService] Synthesis error for clause '\(clause)': \(error)")
                    break
                }
            }
        }

        // Flush any remaining text.
        if !isCancelled, let finalClause = segmenter.flush() {
            do {
                let audioData = try await synthesise(clause: finalClause)
                if !firstClausePlayed {
                    onStarted()
                    firstClausePlayed = true
                }
                scheduleAudio(audioData, sampleRate: 22_050)
            } catch {
                ttsHardFailed = true
                print("[TTSService] Synthesis error for final clause '\(finalClause)': \(error)")
            }
        }

        // ── Fallback: render as on-screen text (Req 3.7) ────────────────────
        if ttsHardFailed, !fullResponseText.isEmpty {
            await renderTextFallback(responseText: fullResponseText)
        }

        // Wait for all queued audio to finish playing, unless cancelled.
        if !isCancelled {
            await waitForPlaybackCompletion()
        }

        if !isCancelled {
            onStopped()
        }
    }

    // MARK: - Private: synthesis

    /// Synthesise a single clause → raw PCM `Data`.
    ///
    /// Primary path: send text to Core via IPC and collect `ttsAudioChunk`
    /// messages.  If IPC is unavailable, fall back to `AVSpeechSynthesizer`.
    private func synthesise(clause: String) async throws -> Data {
        // Try IPC/Core TTS first.
        if let client = ipcClient, client.isConnected {
            do {
                return try await synthesiseViaIPC(clause: clause, client: client)
            } catch {
                print("[TTSService] IPC TTS failed (\(error)); falling back to local AVSpeechSynthesizer.")
            }
        }

        // Local fallback.
        return try await synthesiseLocally(clause: clause)
    }

    /// Send a clause to the Core over IPC and collect the PCM chunks.
    ///
    /// Protocol: the shell sends the clause as a `TurnRequest` (text-only,
    /// no audio features, empty turnId scoped to this clause). The Core
    /// streams back `ServerMessage.ttsAudioChunk` messages until `isLast`.
    ///
    /// In production this relies on a dedicated TTS-only IPC call; for now
    /// we re-use the existing `ClientMessage.turnRequest` path with a
    /// synthetic `HAKITurnRequest` containing the clause text as the
    /// transcript (no STT round-trip needed).
    private func synthesiseViaIPC(clause: String, client: any IPCClientProtocol) async throws -> Data {
        // Build a minimal turn request for TTS-only synthesis.
        let request = HAKITurnRequest(
            turnId: "tts-\(UUID().uuidString)",
            transcript: clause,
            languageComposition: "english",
            audioFeatures: HAKIAudioFeatures(pitchHz: 0, energyDb: 0, durationMs: 0)
        )

        try await client.send(.turnRequest(request))

        // Collect TTS audio chunks from the inbound stream.
        var audioData = Data()
        for await message in client.inbound {
            switch message {
            case .ttsAudioChunk(let chunk):
                audioData.append(chunk.samples)
                if chunk.isLast { break }
            case .error(let msg):
                throw TTSError.ipcUnavailable(msg)
            default:
                continue
            }
        }

        guard !audioData.isEmpty else { throw TTSError.emptySynthesis }
        return audioData
    }

    /// Synthesise a clause locally using `AVSpeechSynthesizer`.
    ///
    /// `AVSpeechSynthesizer` renders to audio by playing through the system
    /// output.  We drive it via a delegate to collect the rendered audio when
    /// the `AVSpeechSynthesizerDelegate.speechSynthesizer(_:didFinish:)`
    /// callback fires, then route through our `AVAudioPlayerNode` for
    /// consistent volume/playback control.
    ///
    /// Simplified: because `AVSpeechSynthesizer` renders directly to the
    /// system output rather than to a buffer, we speak the utterance directly
    /// and return a sentinel `Data()` so the caller knows synthesis
    /// succeeded.  The audio is already playing; we skip the
    /// `AVAudioPlayerNode` scheduling path for local utterances.
    ///
    /// This matches the spec requirement (Req 3.5, 3.7): the Voice_Engine
    /// SHALL convert HAKI response text to speech.  Using the local system
    /// TTS satisfies 3.5; the fallback message satisfies 3.7 only if local
    /// also fails, which is handled at the call site.
    private func synthesiseLocally(clause: String) async throws -> Data {
        return try await withCheckedThrowingContinuation { continuation in
            let utterance = AVSpeechUtterance(string: clause)
            utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
            utterance.rate = AVSpeechUtteranceDefaultSpeechRate

            // Use a one-shot delegate to detect completion / failure.
            let delegate = LocalSynthDelegate(continuation: continuation)
            localSynth.delegate = delegate
            // Keep delegate alive for the duration of synthesis.
            objc_setAssociatedObject(localSynth, &AssociatedKeys.delegate, delegate, .OBJC_ASSOCIATION_RETAIN)

            localSynth.speak(utterance)
        }
    }

    // MARK: - Private: audio scheduling

    /// Schedule raw PCM `Int16` data on the `AVAudioPlayerNode`.
    ///
    /// Expected format: `Int16` LE, mono, at `sampleRate` Hz.
    /// We convert to `Float32` (AVAudioEngine native) on the fly.
    private func scheduleAudio(_ data: Data, sampleRate: Double) {
        // Skip empty data (sentinel from local synth path).
        guard !data.isEmpty else { return }

        let sampleCount = data.count / MemoryLayout<Int16>.size
        guard sampleCount > 0 else { return }

        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            print("[TTSService] Failed to create AVAudioFormat for scheduling.")
            return
        }

        guard let pcmBuffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(sampleCount)
        ) else {
            print("[TTSService] Failed to allocate PCM buffer.")
            return
        }

        pcmBuffer.frameLength = AVAudioFrameCount(sampleCount)

        // Convert Int16 → Float32.
        data.withUnsafeBytes { (rawPtr: UnsafeRawBufferPointer) in
            guard let int16Ptr = rawPtr.bindMemory(to: Int16.self).baseAddress,
                  let floatPtr = pcmBuffer.floatChannelData?[0] else { return }
            for i in 0..<sampleCount {
                floatPtr[i] = Float(int16Ptr[i]) / Float(Int16.max)
            }
        }

        playerLock.withLock {
            playerNode.scheduleBuffer(pcmBuffer, completionHandler: nil)
            if !playerNode.isPlaying {
                playerNode.play()
            }
        }
    }

    /// Wait until the player node finishes all scheduled buffers.
    private func waitForPlaybackCompletion() async {
        // Poll until the player node stops playing.  A continuation-based
        // approach would require subclassing; polling at 20 ms is cheap.
        while playerLock.withLock({ playerNode.isPlaying }) {
            guard !isCancelled else { return }
            try? await Task.sleep(nanoseconds: 20_000_000) // 20 ms
        }
    }

    // MARK: - Private: barge-in

    /// Called when `VoiceEvent.bargeIn` is detected.
    ///
    /// Stops playback immediately, cancels all synthesis tasks, and calls the
    /// `onStopped` callback so the caller (VoiceEngine) can disarm barge-in
    /// tracking (Req 3.3).
    private func handleBargeIn(onStopped: @Sendable @escaping () -> Void) {
        cancelLock.withLock { _cancelled = true }
        stopPlayback()
        speakTask?.cancel()
        speakTask = nil
        onStopped()
    }

    // MARK: - Private: stop

    private func stopPlayback() {
        playerLock.withLock {
            if playerNode.isPlaying {
                playerNode.stop()
            }
        }
        localSynth.stopSpeaking(at: .immediate)
    }

    // MARK: - Private: text fallback (Req 3.7)

    /// Post a `UIState` TTS-failure notification so the UI can render the
    /// response as on-screen text and notify the user (Req 3.7).
    ///
    /// Posts `Notification.Name.ttsFailedShowText` via `NotificationCenter`
    /// so `UIState` (in the HAKIUI module) can receive and surface it to views
    /// without a direct module dependency.
    private func renderTextFallback(responseText: String) async {
        NotificationCenter.default.post(
            name: .ttsFailedShowText,
            object: nil,
            userInfo: [
                "responseText": responseText,
                "turnId": "",
            ]
        )
        // Also attempt a local voice notice.
        let notice = "Audio playback was unavailable. Showing response as text."
        let utterance = AVSpeechUtterance(string: notice)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        localSynth.speak(utterance)
    }

    // MARK: - Private: cancellation

    private var isCancelled: Bool {
        cancelLock.withLock { _cancelled } || Task.isCancelled
    }

    // MARK: - Private: audio engine setup

    private func setupAudioGraph() {
        engine.attach(playerNode)
        engine.connect(
            playerNode,
            to: engine.mainMixerNode,
            format: AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: 22_050,
                channels: 1,
                interleaved: false
            )
        )
        do {
            try engine.start()
            engineStarted = true
        } catch {
            print("[TTSService] Failed to start audio engine: \(error). IPC TTS playback will be unavailable.")
            // Local AVSpeechSynthesizer will still work as a fallback.
        }
    }
}

// MARK: - LocalSynthDelegate

/// One-shot `AVSpeechSynthesizerDelegate` that resolves a continuation when
/// a single utterance finishes or fails.
private final class LocalSynthDelegate: NSObject, AVSpeechSynthesizerDelegate, @unchecked Sendable {

    private let continuation: CheckedContinuation<Data, Error>
    private var settled = false
    private let lock = NSLock()

    init(continuation: CheckedContinuation<Data, Error>) {
        self.continuation = continuation
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        settle { self.continuation.resume(returning: Data()) }
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        settle { self.continuation.resume(throwing: TTSError.synthesisFailure("Local synthesis was cancelled.")) }
    }

    private func settle(work: () -> Void) {
        lock.withLock {
            guard !settled else { return }
            settled = true
            work()
        }
    }
}

// MARK: - Associated object key

private enum AssociatedKeys {
    static var delegate: UInt8 = 0
}

// MARK: - MockTTSService

/// Test double for `TTSService`.  Captures calls and allows injection of
/// synthetic clauses and barge-in events without real audio hardware.
public final class MockTTSService: TTSServiceProtocol, @unchecked Sendable {

    // MARK: - Recorded state

    public private(set) var didCallSpeak = false
    public private(set) var didCallCancel = false
    public private(set) var recordedClauses: [String] = []
    public private(set) var startedCallCount = 0
    public private(set) var stoppedCallCount = 0

    // MARK: - Control

    /// When `true`, `speak()` will simulate a TTS failure so the text-fallback
    /// path is exercised in tests.
    public var simulateTTSFailure = false

    private let lock = NSLock()

    public init() {}

    public func speak(
        textStream: AsyncStream<String>,
        voiceEvents: AsyncStream<VoiceEvent>,
        onStarted: @Sendable @escaping () -> Void,
        onStopped: @Sendable @escaping () -> Void
    ) async {
        lock.withLock { didCallSpeak = true }

        var segmenter = ClauseSegmenter()
        var fullText = ""

        if simulateTTSFailure {
            // Consume the stream and trigger the text fallback path.
            for await token in textStream {
                fullText.append(token)
            }
            if let final_ = segmenter.flush() { fullText.append(final_) }
            // Post the same notification TTSService would post.
            NotificationCenter.default.post(
                name: .ttsFailedShowText,
                object: nil,
                userInfo: ["responseText": fullText, "turnId": ""]
            )
            onStopped()
            return
        }

        for await token in textStream {
            fullText.append(token)
            if let clause = segmenter.feed(token) {
                lock.withLock { recordedClauses.append(clause) }
            }
        }
        if let last = segmenter.flush() {
            lock.withLock { recordedClauses.append(last) }
        }

        lock.withLock { startedCallCount += 1 }
        onStarted()
        lock.withLock { stoppedCallCount += 1 }
        onStopped()
    }

    public func cancel() {
        lock.withLock { didCallCancel = true }
    }
}
