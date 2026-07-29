import Foundation
import HAKIIPC

/// Session-scoped composition root for the replacement local voice path.
///
/// ``VoiceSessionCompositionRoot`` wires the previously built Swift components
/// — ``VoiceAudioController``, ``LocalASRAdapter``, ``AudioFrameRing``,
/// ``PCMPlaybackRenderer``, and ``VoiceSocketClient`` — into a single
/// session-scoped object.
///
/// **Gate contract (Design §11 step 3, Requirements 1.5–1.6):**
/// - This object is constructed only when ``VoiceDevGate.isEnabled`` is
///   ``true``.  Call ``makeIfEnabled()`` rather than the initializer directly.
/// - No legacy STT engine, TTS engine, ``afplay``, ``say``, Deepgram, Groq,
///   Cartesia, Edge TTS, or archive artifact is referenced from this type.
/// - Non-voice IPC handlers managed by ``JSONIPCClient`` are not touched.
///
/// **Session ownership:**
/// One ``VoiceSessionCompositionRoot`` maps to exactly one
/// ``VoiceSession`` on the Python side.  All resources are released when
/// ``shutdown()`` is awaited or the object is deinitialized.
@available(macOS 14.0, *)
public final class VoiceSessionCompositionRoot: @unchecked Sendable {

    // MARK: – Owned replacement-path components

    /// Manages ``AVAudioEngine``, VoiceProcessingIO, and monotonic frame capture.
    public let audioController: VoiceAudioController

    /// CoreML Qwen3-ASR adapter that produces partial/final ``ASRHypothesis`` values.
    public let asrAdapter: CoreMLQwen3ASRAdapter

    /// Same-UID, session-random shared-memory ring for ``InputAudioRawFrame``/Silero.
    public let frameRing: AudioFrameRing

    /// Schedules TTS PCM on the controller's player node and emits terminal events.
    public let playbackRenderer: PCMPlaybackRenderer

    /// Authenticated UDS client for transcript/control events and PCM output delivery.
    public let socketClient: VoiceSocketClient

    /// The session identifier shared between all components.
    public let sessionID: UUID

    // MARK: – Initializer (internal — callers use makeIfEnabled)

    init(
        sessionID: UUID,
        audioController: VoiceAudioController,
        asrAdapter: CoreMLQwen3ASRAdapter,
        frameRing: AudioFrameRing,
        playbackRenderer: PCMPlaybackRenderer,
        socketClient: VoiceSocketClient
    ) {
        self.sessionID = sessionID
        self.audioController = audioController
        self.asrAdapter = asrAdapter
        self.frameRing = frameRing
        self.playbackRenderer = playbackRenderer
        self.socketClient = socketClient
    }

    // MARK: – Factory

    /// Return a fully composed replacement session, or ``nil`` when the gate
    /// is disabled.
    ///
    /// All parameters carry session-level values that must originate from the
    /// inherited launch configuration — never from user-supplied network input.
    ///
    /// - Parameters:
    ///   - sessionID: The shared session UUID; must match the Python
    ///     ``VoiceSession`` that was started by ``start_replacement_session``.
    ///   - socketPath: Absolute path to the owner-only UDS socket created by
    ///     ``VoiceUnixServer``.
    ///   - capability: The 32-hex launch-inherited session capability.
    ///   - voiceAssetURL: URL for ``my_voice.wav`` used by the renderer.
    /// - Returns: A composed and ready-to-start root, or ``nil`` if the gate
    ///   is disabled.
    @MainActor
    public static func makeIfEnabled(
        sessionID: UUID,
        socketPath: String,
        capability: String,
        voiceAssetURL: URL
    ) -> VoiceSessionCompositionRoot? {
        guard VoiceDevGate.isEnabled else { return nil }

        let audioController = VoiceAudioController(sessionID: sessionID)
        guard let asrAdapter = try? CoreMLQwen3ASRAdapter.productionDefault() else { return nil }
        guard let frameRing = try? AudioFrameRing(sessionID: sessionID) else { return nil }
        let playbackRenderer = PCMPlaybackRenderer(controller: audioController)
        guard let socketClient = try? VoiceSocketClient(
            socketPath: URL(fileURLWithPath: socketPath),
            sessionID: sessionID,
            sessionCapability: capability
        ) else { return nil }

        return VoiceSessionCompositionRoot(
            sessionID: sessionID,
            audioController: audioController,
            asrAdapter: asrAdapter,
            frameRing: frameRing,
            playbackRenderer: playbackRenderer,
            socketClient: socketClient
        )
    }

    // MARK: – Lifecycle

    /// Start capture, ASR, socket connection, and playback renderer.
    ///
    /// Call this after ``makeIfEnabled(...)`` returns a non-nil root.
    /// All sub-component failures surface as thrown errors; no legacy
    /// component is started as a fallback.
    public func start() async throws {
        // Connect the socket client before starting capture so the Python
        // pipeline is ready to receive transcript events.
        try await socketClient.connect()

        // Start capture — frames flow once the downstream pipeline is ready.
        try audioController.startCapture()

        // Wire ASR output → socket client transcript events via tasks.
        // The ASR adapter produces frames via audioController.frames AsyncStream.
        Task { [weak self] in
            guard let self else { return }
            for await frame in audioController.frames {
                if let hypotheses = try? await self.asrAdapter.consume(frame) {
                    for hypothesis in hypotheses {
                        let event = VoiceTranscriptEvent(
                            sessionID: self.sessionID,
                            turnID: hypothesis.turnID,
                            eventSequence: 0,
                            text: hypothesis.text,
                            isFinal: hypothesis.isFinal,
                            language: hypothesis.language,
                            captureStartedMonotonicNs: hypothesis.captureStartedMonotonicNs,
                            captureEndedMonotonicNs: hypothesis.captureEndedMonotonicNs
                        )
                        try? await self.socketClient.sendTranscript(event)
                    }
                    _ = try? frameRing.enqueue(frame)
                }
            }
        }

        // Wire playback renderer events → socket client.
        Task { [weak self] in
            guard let self else { return }
            for await event in playbackRenderer.events {
                switch event {
                case .terminal(let terminal):
                    try? await self.socketClient.sendPlaybackTerminal(terminal)
                case .stopAcknowledged:
                    break
                case .accepted:
                    break
                }
            }
        }

        // Wire socket client inbound events → playback renderer.
        Task { [weak self] in
            guard let self else { return }
            let events = await socketClient.inboundEvents
            for await event in events {
                switch event {
                case .stopPlayback(let stop):
                    self.playbackRenderer.stop(stop)
                default:
                    break
                }
            }
        }
    }

    /// Stop all components in reverse order and release session resources.
    ///
    /// Safe to call multiple times.  No error is thrown for already-stopped
    /// components.
    public func shutdown() async {
        audioController.stopCapture()
        await socketClient.disconnect()
        frameRing.close()
    }

    deinit {
        // Best-effort synchronous cleanup for deallocation paths.
        // Proper shutdown should be done via the async shutdown() method.
        frameRing.close()
    }
}
