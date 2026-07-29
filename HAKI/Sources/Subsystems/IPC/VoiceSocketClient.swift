// VoiceSocketClient.swift
// Strict v1 text/control UDS client for the local realtime voice runtime.
//
// This client is deliberately separate from the legacy JSON IPC transport. It
// authenticates with the launch-inherited session capability, sends only
// normalized transcript/control records to Python, and accepts PCM only from
// Python in the documented length-prefixed output frame.

import Foundation
import Network

public enum VoiceTranscriptLanguage: String, Sendable, Codable, CaseIterable {
    case hi
    case en
    case hinglish
}

public enum VoicePlaybackTerminalKind: String, Sendable, Codable, CaseIterable {
    case confirmed = "PLAYBACK_CONFIRMED"
    case cancelled = "PLAYBACK_CANCELLED"
    case failed = "PLAYBACK_FAILED"
}

public enum VoiceSocketConnectionState: Sendable, Equatable {
    case idle
    case connecting
    case connected
    case reconnecting(attempt: Int)
    case reconnectExhausted
    case failed(reason: String)
}

public enum VoiceSocketClientError: Error, Sendable, Equatable {
    case invalidSessionCapability
    case invalidMessage(String)
    case notConnected
    case invalidTranscriptSequence
    case turnAlreadyFinalized
    case turnDiscarded
    case duplicatePlaybackTerminal
    case invalidPCMSequence
    case connectionClosed
}

/// A normalized ASR event. Its wire representation contains text only; this
/// type intentionally has no audio/sample/PCM property.
public struct VoiceTranscriptEvent: Sendable, Equatable {
    public let eventID: UUID
    public let sessionID: UUID
    public let turnID: UUID
    public let eventSequence: UInt64
    public let text: String
    public let isFinal: Bool
    public let language: VoiceTranscriptLanguage
    public let captureStartedMonotonicNs: UInt64
    public let captureEndedMonotonicNs: UInt64

    public init(
        eventID: UUID = UUID(),
        sessionID: UUID,
        turnID: UUID,
        eventSequence: UInt64,
        text: String,
        isFinal: Bool,
        language: VoiceTranscriptLanguage,
        captureStartedMonotonicNs: UInt64,
        captureEndedMonotonicNs: UInt64
    ) {
        self.eventID = eventID
        self.sessionID = sessionID
        self.turnID = turnID
        self.eventSequence = eventSequence
        self.text = text
        self.isFinal = isFinal
        self.language = language
        self.captureStartedMonotonicNs = captureStartedMonotonicNs
        self.captureEndedMonotonicNs = captureEndedMonotonicNs
    }
}

public struct VoiceCaptureInterrupted: Sendable, Equatable {
    public let eventID: UUID
    public let sessionID: UUID
    public let turnID: UUID
    public let reason: String

    public init(eventID: UUID = UUID(), sessionID: UUID, turnID: UUID, reason: String) {
        self.eventID = eventID
        self.sessionID = sessionID
        self.turnID = turnID
        self.reason = reason
    }
}

public struct VoiceEventAcknowledgement: Sendable, Equatable {
    public enum Status: String, Sendable, Equatable {
        case accepted
        case discarded
    }

    public let eventID: UUID
    public let status: Status
    public let reason: String?
}

/// Metadata for Python-to-Swift TTS PCM. The PCM bytes follow the JSONL header
/// in a four-byte, big-endian length frame and are never used for microphone
/// transport.
public struct VoicePCMChunk: Sendable, Equatable {
    public let sessionID: UUID
    public let turnID: UUID
    public let sentenceID: UUID
    public let sequence: UInt64
    public let sampleRateHz: Int
    public let channels: UInt8
    public let byteLength: Int
}

public struct VoiceStopPlayback: Sendable, Equatable {
    public let sessionID: UUID
    public let turnID: UUID
    public let generation: UInt64
}

public struct VoicePlaybackTerminal: Sendable, Equatable {
    public let kind: VoicePlaybackTerminalKind
    public let eventID: UUID
    public let sessionID: UUID
    public let turnID: UUID
    public let sentenceID: UUID
    public let errorClass: String?

    public init(
        kind: VoicePlaybackTerminalKind,
        eventID: UUID = UUID(),
        sessionID: UUID,
        turnID: UUID,
        sentenceID: UUID,
        errorClass: String? = nil
    ) {
        self.kind = kind
        self.eventID = eventID
        self.sessionID = sessionID
        self.turnID = turnID
        self.sentenceID = sentenceID
        self.errorClass = errorClass
    }
}

public enum VoiceSocketInboundEvent: Sendable, Equatable {
    case eventAcknowledgement(VoiceEventAcknowledgement)
    case pcmChunk(VoicePCMChunk)
    case stopPlayback(VoiceStopPlayback)
    case stopPlaybackAcknowledged(VoiceStopPlayback)
}

/// Small transport seam used by protocol tests. Production transport is an
/// `NWConnection` to a Unix-domain socket; fixtures use an in-memory peer.
public protocol VoiceSocketTransport: AnyObject, Sendable {
    func connect(path: String) async throws
    func send(_ data: Data) async throws
    func receive() async throws -> Data?
    func close()
}

public typealias VoiceSocketTransportFactory = @Sendable () -> any VoiceSocketTransport
public typealias VoicePCMChunkHandler = @Sendable (VoicePCMChunk, Data) async throws -> Void
public typealias VoiceStopPlaybackHandler = @Sendable (VoiceStopPlayback) async -> Void

/// Authenticated, reconnecting v1 client for the dedicated voice UDS.
///
/// Transcript turn state is intentionally never replayed after a transport
/// loss. A pending final has no delivery proof, so it is discarded with its
/// incomplete turn before reconnection; the user can begin a new turn once the
/// socket recovers.
public actor VoiceSocketClient {
    public static let protocolVersion = 1
    public static let reconnectDelaysNanoseconds: [UInt64] = [100_000_000, 250_000_000, 500_000_000, 1_000_000_000]

    public let socketPath: URL
    public let sessionID: UUID
    public let stateUpdates: AsyncStream<VoiceSocketConnectionState>
    public let inboundEvents: AsyncStream<VoiceSocketInboundEvent>
    public let discardedTurns: AsyncStream<UUID>

    public private(set) var state: VoiceSocketConnectionState = .idle

    private let sessionCapability: String
    private let transportFactory: VoiceSocketTransportFactory
    private let onPCMChunk: VoicePCMChunkHandler
    private let onStopPlayback: VoiceStopPlaybackHandler
    private let reconnectDelays: [UInt64]

    private let stateContinuation: AsyncStream<VoiceSocketConnectionState>.Continuation
    private let inboundContinuation: AsyncStream<VoiceSocketInboundEvent>.Continuation
    private let discardedTurnContinuation: AsyncStream<UUID>.Continuation

    private var currentTransport: (any VoiceSocketTransport)?
    private var receiveTask: Task<Void, Never>?
    private var reconnectTask: Task<Void, Never>?
    private var intentionallyDisconnected = false
    private var reconnectAttempt = 0

    private var receiveBuffer = Data()
    private var pendingPCMChunk: VoicePCMChunk?
    private var transcriptTurns: [UUID: TranscriptTurnState] = [:]
    private var pendingAcknowledgements: [UUID: PendingAcknowledgement] = [:]
    private var pcmSequences: [SentenceKey: UInt64] = [:]
    private var playbackTerminals = Set<SentenceKey>()
    private var handledStops = Set<PlaybackGenerationKey>()

    public init(
        socketPath: URL,
        sessionID: UUID,
        sessionCapability: String,
        transportFactory: @escaping VoiceSocketTransportFactory = { NetworkVoiceSocketTransport() },
        reconnectDelaysNanoseconds: [UInt64] = VoiceSocketClient.reconnectDelaysNanoseconds,
        onPCMChunk: @escaping VoicePCMChunkHandler = { _, _ in },
        onStopPlayback: @escaping VoiceStopPlaybackHandler = { _ in }
    ) throws {
        guard Self.isValidSessionCapability(sessionCapability) else {
            throw VoiceSocketClientError.invalidSessionCapability
        }
        guard !reconnectDelaysNanoseconds.isEmpty,
              reconnectDelaysNanoseconds.allSatisfy({ $0 > 0 }) else {
            throw VoiceSocketClientError.invalidMessage("invalid_reconnect_delays")
        }

        self.socketPath = socketPath
        self.sessionID = sessionID
        self.sessionCapability = sessionCapability
        self.transportFactory = transportFactory
        self.reconnectDelays = reconnectDelaysNanoseconds
        self.onPCMChunk = onPCMChunk
        self.onStopPlayback = onStopPlayback

        let stateStream = AsyncStream<VoiceSocketConnectionState>.makeStream(bufferingPolicy: .bufferingNewest(32))
        stateUpdates = stateStream.stream
        stateContinuation = stateStream.continuation

        let inboundStream = AsyncStream<VoiceSocketInboundEvent>.makeStream(bufferingPolicy: .bufferingNewest(128))
        inboundEvents = inboundStream.stream
        inboundContinuation = inboundStream.continuation

        let discardedStream = AsyncStream<UUID>.makeStream(bufferingPolicy: .bufferingNewest(32))
        discardedTurns = discardedStream.stream
        discardedTurnContinuation = discardedStream.continuation
    }

    deinit {
        receiveTask?.cancel()
        reconnectTask?.cancel()
        currentTransport?.close()
        stateContinuation.finish()
        inboundContinuation.finish()
        discardedTurnContinuation.finish()
    }

    /// Opens the UDS and sends the capability preface before any protocol
    /// record. The server deliberately does not echo this secret.
    public func connect() async throws {
        intentionallyDisconnected = false
        reconnectTask?.cancel()
        reconnectTask = nil
        reconnectAttempt = 0
        try await establishConnection()
    }

    /// Ends the current client lifecycle. Any unfinished transcript is
    /// terminally discarded and will never be sent on a later connection.
    public func disconnect() {
        intentionallyDisconnected = true
        reconnectTask?.cancel()
        reconnectTask = nil
        receiveTask?.cancel()
        receiveTask = nil
        discardUnfinishedTurns()
        currentTransport?.close()
        currentTransport = nil
        receiveBuffer.removeAll(keepingCapacity: false)
        pendingPCMChunk = nil
        transition(to: .idle)
    }

    /// Encodes a strictly ordered, text-only transcript event. The caller
    /// supplies the per-turn event sequence; this client rejects gaps,
    /// duplicates, post-final events, stale sessions, and empty text locally.
    public func sendTranscript(_ event: VoiceTranscriptEvent) async throws {
        guard event.sessionID == sessionID else {
            throw VoiceSocketClientError.invalidMessage("stale_session")
        }
        guard !event.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              event.captureEndedMonotonicNs >= event.captureStartedMonotonicNs else {
            throw VoiceSocketClientError.invalidMessage("invalid_transcript")
        }

        switch transcriptTurns[event.turnID] {
        case .none:
            transcriptTurns[event.turnID] = event.isFinal
                ? .finalAwaitingAcknowledgement(eventID: event.eventID, sequence: event.eventSequence)
                : .active(lastSequence: event.eventSequence)
        case .active(let lastSequence):
            guard event.eventSequence == lastSequence + 1 else {
                throw VoiceSocketClientError.invalidTranscriptSequence
            }
            transcriptTurns[event.turnID] = event.isFinal
                ? .finalAwaitingAcknowledgement(eventID: event.eventID, sequence: event.eventSequence)
                : .active(lastSequence: event.eventSequence)
        case .finalAwaitingAcknowledgement, .completed:
            throw VoiceSocketClientError.turnAlreadyFinalized
        case .discarded:
            throw VoiceSocketClientError.turnDiscarded
        }

        pendingAcknowledgements[event.eventID] = .transcript(turnID: event.turnID, isFinal: event.isFinal)
        do {
            try await sendJSONObject(VoiceSocketWire.transcriptJSONObject(event))
        } catch {
            // The event remains pending only long enough for the connection
            // failure path to discard the turn. It is never replayed.
            throw error
        }
    }

    /// Invalidates a native capture turn after interruption, route loss, or a
    /// media-services reset. The turn becomes terminal locally before sending.
    public func sendCaptureInterrupted(_ event: VoiceCaptureInterrupted) async throws {
        guard event.sessionID == sessionID,
              !event.reason.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw VoiceSocketClientError.invalidMessage("invalid_capture_interrupted")
        }
        discardTurn(event.turnID)
        pendingAcknowledgements[event.eventID] = .captureInterrupted(turnID: event.turnID)
        try await sendJSONObject(VoiceSocketWire.captureInterruptedJSONObject(event))
    }

    /// Sends a renderer terminal event once and only once for a sentence. A
    /// failed write does not reopen this guard: delivery is ambiguous and a
    /// second terminal message could corrupt the Python playback ledger.
    public func sendPlaybackTerminal(_ terminal: VoicePlaybackTerminal) async throws {
        guard terminal.sessionID == sessionID else {
            throw VoiceSocketClientError.invalidMessage("stale_session")
        }
        if terminal.kind == .failed,
           terminal.errorClass?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false {
            throw VoiceSocketClientError.invalidMessage("invalid_playback_error_class")
        }

        let key = SentenceKey(turnID: terminal.turnID, sentenceID: terminal.sentenceID)
        guard playbackTerminals.insert(key).inserted else {
            throw VoiceSocketClientError.duplicatePlaybackTerminal
        }
        pendingAcknowledgements[terminal.eventID] = .playbackTerminal
        try await sendJSONObject(VoiceSocketWire.playbackTerminalJSONObject(terminal))
    }

    public func sendPlaybackConfirmed(turnID: UUID, sentenceID: UUID) async throws {
        try await sendPlaybackTerminal(VoicePlaybackTerminal(
            kind: .confirmed,
            sessionID: sessionID,
            turnID: turnID,
            sentenceID: sentenceID
        ))
    }

    public func sendPlaybackCancelled(turnID: UUID, sentenceID: UUID) async throws {
        try await sendPlaybackTerminal(VoicePlaybackTerminal(
            kind: .cancelled,
            sessionID: sessionID,
            turnID: turnID,
            sentenceID: sentenceID
        ))
    }

    public func sendPlaybackFailed(
        turnID: UUID,
        sentenceID: UUID,
        errorClass: String
    ) async throws {
        try await sendPlaybackTerminal(VoicePlaybackTerminal(
            kind: .failed,
            sessionID: sessionID,
            turnID: turnID,
            sentenceID: sentenceID,
            errorClass: errorClass
        ))
    }

    private func establishConnection() async throws {
        guard currentTransport == nil else { return }
        transition(to: .connecting)
        let transport = transportFactory()
        do {
            try await transport.connect(path: socketPath.path)
            try await transport.send(Data("VOICE_AUTH \(sessionCapability)\n".utf8))
        } catch {
            transport.close()
            transition(to: .failed(reason: "connection_unavailable"))
            throw error
        }

        currentTransport = transport
        receiveBuffer.removeAll(keepingCapacity: true)
        pendingPCMChunk = nil
        reconnectAttempt = 0
        transition(to: .connected)
        startReceiveLoop(for: transport)
    }

    private func startReceiveLoop(for transport: any VoiceSocketTransport) {
        receiveTask?.cancel()
        receiveTask = Task { [weak self, transport] in
            do {
                while !Task.isCancelled {
                    guard let data = try await transport.receive() else { break }
                    await self?.receive(data, from: transport)
                }
            } catch is CancellationError {
                return
            } catch {
                // Connection loss is normalized below; do not surface socket
                // paths, capabilities, or transcript content.
            }
            await self?.connectionLost(from: transport)
        }
    }

    private func receive(_ data: Data, from transport: any VoiceSocketTransport) async {
        guard isCurrentTransport(transport) else { return }
        receiveBuffer.append(data)
        do {
            try await processReceiveBuffer()
        } catch {
            await connectionLost(from: transport)
        }
    }

    private func processReceiveBuffer() async throws {
        while true {
            if let pcmChunk = pendingPCMChunk {
                guard receiveBuffer.count >= 4 else { return }
                let length = receiveBuffer.prefix(4).reduce(UInt32(0)) { partial, byte in
                    (partial << 8) | UInt32(byte)
                }
                guard length == UInt32(pcmChunk.byteLength) else {
                    throw VoiceSocketClientError.invalidMessage("invalid_binary_length")
                }
                let payloadEnd = 4 + pcmChunk.byteLength
                guard receiveBuffer.count >= payloadEnd else { return }
                let payload = receiveBuffer.subdata(in: 4..<payloadEnd)
                receiveBuffer.removeSubrange(0..<payloadEnd)
                pendingPCMChunk = nil
                try await acceptPCMChunk(pcmChunk, payload: payload)
                continue
            }

            guard let newline = receiveBuffer.firstIndex(of: 0x0A) else {
                if receiveBuffer.count > VoiceSocketWire.maxJSONLineBytes {
                    throw VoiceSocketClientError.invalidMessage("json_line_too_large")
                }
                return
            }
            guard newline <= VoiceSocketWire.maxJSONLineBytes else {
                throw VoiceSocketClientError.invalidMessage("json_line_too_large")
            }
            let line = receiveBuffer.subdata(in: 0..<newline)
            receiveBuffer.removeSubrange(0...newline)
            guard !line.isEmpty else { continue }

            switch try VoiceSocketWire.decodeInbound(line) {
            case .eventAcknowledgement(let acknowledgement):
                acceptAcknowledgement(acknowledgement)
                inboundContinuation.yield(.eventAcknowledgement(acknowledgement))
            case .pcmChunk(let chunk):
                guard chunk.sessionID == sessionID else {
                    throw VoiceSocketClientError.invalidMessage("stale_session")
                }
                pendingPCMChunk = chunk
            case .stopPlayback(let stop):
                guard stop.sessionID == sessionID else {
                    throw VoiceSocketClientError.invalidMessage("stale_session")
                }
                await acceptStopPlayback(stop)
            case .stopPlaybackAcknowledged(let stop):
                inboundContinuation.yield(.stopPlaybackAcknowledged(stop))
            }
        }
    }

    private func acceptPCMChunk(_ chunk: VoicePCMChunk, payload: Data) async throws {
        let key = SentenceKey(turnID: chunk.turnID, sentenceID: chunk.sentenceID)
        if let previous = pcmSequences[key], chunk.sequence != previous + 1 {
            throw VoiceSocketClientError.invalidPCMSequence
        }
        pcmSequences[key] = chunk.sequence
        try await onPCMChunk(chunk, payload)
        try await sendJSONObject(VoiceSocketWire.pcmAcceptedJSONObject(chunk))
        inboundContinuation.yield(.pcmChunk(chunk))
    }

    private func acceptStopPlayback(_ stop: VoiceStopPlayback) async {
        let key = PlaybackGenerationKey(turnID: stop.turnID, generation: stop.generation)
        let firstDelivery = handledStops.insert(key).inserted
        if firstDelivery {
            inboundContinuation.yield(.stopPlayback(stop))
            await onStopPlayback(stop)
        }
        do {
            // Repeated controls receive an acknowledgement but never repeat the
            // renderer stop operation, which makes STOP_PLAYBACK idempotent.
            try await sendJSONObject(VoiceSocketWire.stopPlaybackJSONObject(stop, type: "STOP_PLAYBACK_ACK"))
        } catch {
            // The receive loop will observe the closed connection; it must not
            // retry a stopped generation on the next connection.
        }
    }

    private func acceptAcknowledgement(_ acknowledgement: VoiceEventAcknowledgement) {
        guard let pending = pendingAcknowledgements.removeValue(forKey: acknowledgement.eventID) else {
            return
        }
        switch pending {
        case .transcript(let turnID, let isFinal):
            if acknowledgement.status == .discarded {
                discardTurn(turnID)
            } else if isFinal {
                transcriptTurns[turnID] = .completed
            }
        case .captureInterrupted, .playbackTerminal:
            break
        }
    }

    private func sendJSONObject(_ object: [String: Any]) async throws {
        let data = try VoiceSocketWire.encodeJSONObject(object)
        guard let transport = currentTransport else {
            throw VoiceSocketClientError.notConnected
        }
        do {
            try await transport.send(data)
        } catch {
            await connectionLost(from: transport)
            throw VoiceSocketClientError.connectionClosed
        }
    }

    private func connectionLost(from transport: any VoiceSocketTransport) async {
        guard isCurrentTransport(transport) else { return }
        currentTransport = nil
        transport.close()
        receiveBuffer.removeAll(keepingCapacity: false)
        pendingPCMChunk = nil
        discardUnfinishedTurns()
        guard !intentionallyDisconnected else {
            transition(to: .idle)
            return
        }
        scheduleReconnect()
    }

    private func scheduleReconnect() {
        guard reconnectTask == nil else { return }
        guard reconnectAttempt < reconnectDelays.count else {
            transition(to: .reconnectExhausted)
            return
        }

        let delay = reconnectDelays[reconnectAttempt]
        reconnectAttempt += 1
        transition(to: .reconnecting(attempt: reconnectAttempt))
        reconnectTask = Task { [weak self] in
            do {
                try await Task.sleep(nanoseconds: delay)
            } catch {
                return
            }
            await self?.attemptReconnect()
        }
    }

    private func attemptReconnect() async {
        reconnectTask = nil
        guard !intentionallyDisconnected, currentTransport == nil else { return }
        do {
            try await establishConnection()
        } catch {
            scheduleReconnect()
        }
    }

    private func discardUnfinishedTurns() {
        for turnID in transcriptTurns.keys {
            guard case .completed = transcriptTurns[turnID] else {
                discardTurn(turnID)
                continue
            }
        }
    }

    private func discardTurn(_ turnID: UUID) {
        guard transcriptTurns[turnID] != .discarded else { return }
        transcriptTurns[turnID] = .discarded
        pendingAcknowledgements = pendingAcknowledgements.filter { _, pending in
            switch pending {
            case .transcript(let pendingTurnID, _), .captureInterrupted(let pendingTurnID):
                return pendingTurnID != turnID
            case .playbackTerminal:
                return true
            }
        }
        discardedTurnContinuation.yield(turnID)
    }

    private func isCurrentTransport(_ transport: any VoiceSocketTransport) -> Bool {
        guard let currentTransport else { return false }
        return ObjectIdentifier(currentTransport) == ObjectIdentifier(transport)
    }

    private func transition(to newState: VoiceSocketConnectionState) {
        state = newState
        stateContinuation.yield(newState)
    }

    private static func isValidSessionCapability(_ candidate: String) -> Bool {
        guard candidate.count == 32 else { return false }
        return candidate.unicodeScalars.allSatisfy { scalar in
            switch scalar.value {
            case 48...57, 97...102: return true
            default: return false
            }
        }
    }
}

private enum TranscriptTurnState: Equatable {
    case active(lastSequence: UInt64)
    case finalAwaitingAcknowledgement(eventID: UUID, sequence: UInt64)
    case completed
    case discarded
}

private enum PendingAcknowledgement: Equatable {
    case transcript(turnID: UUID, isFinal: Bool)
    case captureInterrupted(turnID: UUID)
    case playbackTerminal
}

private struct SentenceKey: Hashable {
    let turnID: UUID
    let sentenceID: UUID
}

private struct PlaybackGenerationKey: Hashable {
    let turnID: UUID
    let generation: UInt64
}

/// Production Network-framework transport for `VoiceSocketClient`.
public final class NetworkVoiceSocketTransport: VoiceSocketTransport, @unchecked Sendable {
    private let queue = DispatchQueue(label: "com.haki.voice.socket", qos: .userInitiated)
    private var connection: NWConnection?

    public init() {}

    public func connect(path: String) async throws {
        let connection = NWConnection(
            to: .unix(path: path),
            using: NWParameters(tls: nil, tcp: NWProtocolTCP.Options())
        )
        self.connection = connection
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let completion = NetworkConnectionCompletion(continuation)
            connection.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    completion.resumeSuccess()
                case .failed:
                    completion.resumeFailure(VoiceSocketClientError.connectionClosed)
                case .cancelled:
                    completion.resumeFailure(VoiceSocketClientError.connectionClosed)
                default:
                    break
                }
            }
            connection.start(queue: queue)
        }
    }

    public func send(_ data: Data) async throws {
        guard let connection else { throw VoiceSocketClientError.notConnected }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            connection.send(content: data, completion: .contentProcessed { error in
                if error == nil {
                    continuation.resume()
                } else {
                    continuation.resume(throwing: VoiceSocketClientError.connectionClosed)
                }
            })
        }
    }

    public func receive() async throws -> Data? {
        guard let connection else { throw VoiceSocketClientError.notConnected }
        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Data?, Error>) in
            connection.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { data, _, complete, error in
                if error != nil {
                    continuation.resume(throwing: VoiceSocketClientError.connectionClosed)
                } else if let data, !data.isEmpty {
                    continuation.resume(returning: data)
                } else if complete {
                    continuation.resume(returning: nil)
                } else {
                    continuation.resume(throwing: VoiceSocketClientError.connectionClosed)
                }
            }
        }
    }

    public func close() {
        connection?.cancel()
        connection = nil
    }
}

private final class NetworkConnectionCompletion: @unchecked Sendable {
    private let lock = NSLock()
    private var didResume = false
    private let continuation: CheckedContinuation<Void, Error>

    init(_ continuation: CheckedContinuation<Void, Error>) {
        self.continuation = continuation
    }

    func resumeSuccess() {
        lock.lock()
        defer { lock.unlock() }
        guard !didResume else { return }
        didResume = true
        continuation.resume()
    }

    func resumeFailure(_ error: Error) {
        lock.lock()
        defer { lock.unlock() }
        guard !didResume else { return }
        didResume = true
        continuation.resume(throwing: error)
    }
}

private enum VoiceSocketWire {
    static let maxJSONLineBytes = 64 * 1024
    private static let microphoneFieldTokens = [
        "audio", "microphone", "mic", "sample", "pcm", "waveform",
        "wave", "buffer", "base64", "binary", "bytes",
    ]

    enum Inbound {
        case eventAcknowledgement(VoiceEventAcknowledgement)
        case pcmChunk(VoicePCMChunk)
        case stopPlayback(VoiceStopPlayback)
        case stopPlaybackAcknowledged(VoiceStopPlayback)
    }

    static func transcriptJSONObject(_ event: VoiceTranscriptEvent) -> [String: Any] {
        [
            "version": VoiceSocketClient.protocolVersion,
            "type": "TRANSCRIPT_EVENT",
            "event_id": canonical(event.eventID),
            "session_id": canonical(event.sessionID),
            "turn_id": canonical(event.turnID),
            "event_seq": event.eventSequence,
            "text": event.text,
            "is_final": event.isFinal,
            "language": event.language.rawValue,
            "capture_started_monotonic_ns": event.captureStartedMonotonicNs,
            "capture_ended_monotonic_ns": event.captureEndedMonotonicNs,
        ]
    }

    static func captureInterruptedJSONObject(_ event: VoiceCaptureInterrupted) -> [String: Any] {
        [
            "version": VoiceSocketClient.protocolVersion,
            "type": "CAPTURE_INTERRUPTED",
            "event_id": canonical(event.eventID),
            "session_id": canonical(event.sessionID),
            "turn_id": canonical(event.turnID),
            "reason": event.reason,
        ]
    }

    static func pcmAcceptedJSONObject(_ chunk: VoicePCMChunk) -> [String: Any] {
        [
            "version": VoiceSocketClient.protocolVersion,
            "type": "PCM_ACCEPTED",
            "session_id": canonical(chunk.sessionID),
            "turn_id": canonical(chunk.turnID),
            "sentence_id": canonical(chunk.sentenceID),
            "sequence": chunk.sequence,
        ]
    }

    static func stopPlaybackJSONObject(_ stop: VoiceStopPlayback, type: String) -> [String: Any] {
        [
            "version": VoiceSocketClient.protocolVersion,
            "type": type,
            "session_id": canonical(stop.sessionID),
            "turn_id": canonical(stop.turnID),
            "generation": stop.generation,
        ]
    }

    static func playbackTerminalJSONObject(_ terminal: VoicePlaybackTerminal) -> [String: Any] {
        var object: [String: Any] = [
            "version": VoiceSocketClient.protocolVersion,
            "type": terminal.kind.rawValue,
            "event_id": canonical(terminal.eventID),
            "session_id": canonical(terminal.sessionID),
            "turn_id": canonical(terminal.turnID),
            "sentence_id": canonical(terminal.sentenceID),
        ]
        if terminal.kind == .failed {
            object["error_class"] = terminal.errorClass ?? ""
        }
        return object
    }

    static func encodeJSONObject(_ object: [String: Any]) throws -> Data {
        try validateJSONObject(object)
        var data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        data.append(0x0A)
        return data
    }

    static func decodeInbound(_ line: Data) throws -> Inbound {
        guard line.count <= maxJSONLineBytes else {
            throw VoiceSocketClientError.invalidMessage("json_line_too_large")
        }
        let object = try decodeJSONObject(line)
        try validateJSONObject(object)
        guard let type = object["type"] as? String else {
            throw VoiceSocketClientError.invalidMessage("invalid_message_type")
        }

        switch type {
        case "EVENT_ACK":
            let statusRaw = try string(object, "status")
            guard let status = VoiceEventAcknowledgement.Status(rawValue: statusRaw) else {
                throw VoiceSocketClientError.invalidMessage("invalid_ack_status")
            }
            return .eventAcknowledgement(VoiceEventAcknowledgement(
                eventID: try uuid(object, "event_id"),
                status: status,
                reason: object["reason"] as? String
            ))
        case "PCM_CHUNK":
            return .pcmChunk(VoicePCMChunk(
                sessionID: try uuid(object, "session_id"),
                turnID: try uuid(object, "turn_id"),
                sentenceID: try uuid(object, "sentence_id"),
                sequence: try unsigned(object, "sequence"),
                sampleRateHz: try integer(object, "sample_rate_hz"),
                channels: UInt8(try integer(object, "channels")),
                byteLength: try integer(object, "byte_length")
            ))
        case "STOP_PLAYBACK", "STOP_PLAYBACK_ACK":
            let stop = VoiceStopPlayback(
                sessionID: try uuid(object, "session_id"),
                turnID: try uuid(object, "turn_id"),
                generation: try unsigned(object, "generation")
            )
            return type == "STOP_PLAYBACK" ? .stopPlayback(stop) : .stopPlaybackAcknowledged(stop)
        default:
            throw VoiceSocketClientError.invalidMessage("incoming_message_type_invalid")
        }
    }

    private static func decodeJSONObject(_ data: Data) throws -> [String: Any] {
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw VoiceSocketClientError.invalidMessage("message_must_be_object")
        }
        return object
    }

    private static func validateJSONObject(_ object: [String: Any]) throws {
        guard integerValue(object["version"]) == VoiceSocketClient.protocolVersion else {
            throw VoiceSocketClientError.invalidMessage("protocol_version_incompatible")
        }
        guard let type = object["type"] as? String else {
            throw VoiceSocketClientError.invalidMessage("invalid_message_type")
        }

        switch type {
        case "TRANSCRIPT_EVENT":
            try validateFields(object, required: [
                "version", "type", "event_id", "session_id", "turn_id", "event_seq", "text",
                "is_final", "language", "capture_started_monotonic_ns", "capture_ended_monotonic_ns",
            ])
            _ = try uuid(object, "event_id")
            _ = try uuid(object, "session_id")
            _ = try uuid(object, "turn_id")
            _ = try unsigned(object, "event_seq")
            guard let text = object["text"] as? String, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  object["is_final"] is Bool,
                  let language = object["language"] as? String, VoiceTranscriptLanguage(rawValue: language) != nil else {
                throw VoiceSocketClientError.invalidMessage("invalid_transcript")
            }
            let started = try unsigned(object, "capture_started_monotonic_ns")
            let ended = try unsigned(object, "capture_ended_monotonic_ns")
            guard ended >= started else { throw VoiceSocketClientError.invalidMessage("invalid_capture_timestamps") }
        case "EVENT_ACK":
            try validateFields(object, required: ["version", "type", "event_id", "status"], optional: ["reason"])
            _ = try uuid(object, "event_id")
            guard let status = object["status"] as? String, VoiceEventAcknowledgement.Status(rawValue: status) != nil else {
                throw VoiceSocketClientError.invalidMessage("invalid_ack_status")
            }
            if let reason = object["reason"], !(reason is String) {
                throw VoiceSocketClientError.invalidMessage("invalid_ack_reason")
            }
        case "CAPTURE_INTERRUPTED":
            try validateFields(object, required: ["version", "type", "event_id", "session_id", "turn_id", "reason"])
            _ = try uuid(object, "event_id")
            _ = try uuid(object, "session_id")
            _ = try uuid(object, "turn_id")
            guard let reason = object["reason"] as? String, !reason.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw VoiceSocketClientError.invalidMessage("invalid_control_reason")
            }
        case "PCM_CHUNK":
            try validateFields(object, required: [
                "version", "type", "session_id", "turn_id", "sentence_id", "sequence",
                "sample_rate_hz", "channels", "format", "byte_length",
            ])
            _ = try uuid(object, "session_id")
            _ = try uuid(object, "turn_id")
            _ = try uuid(object, "sentence_id")
            _ = try unsigned(object, "sequence")
            let sampleRate = try integer(object, "sample_rate_hz")
            let channels = try integer(object, "channels")
            let byteLength = try integer(object, "byte_length")
            guard (8_000...192_000).contains(sampleRate),
                  (1...2).contains(channels),
                  object["format"] as? String == "s16le",
                  (1...(4 * 1024 * 1024)).contains(byteLength),
                  byteLength.isMultiple(of: 2) else {
                throw VoiceSocketClientError.invalidMessage("invalid_pcm_metadata")
            }
        case "PCM_ACCEPTED":
            try validateFields(object, required: ["version", "type", "session_id", "turn_id", "sentence_id", "sequence"])
            _ = try uuid(object, "session_id")
            _ = try uuid(object, "turn_id")
            _ = try uuid(object, "sentence_id")
            _ = try unsigned(object, "sequence")
        case "STOP_PLAYBACK", "STOP_PLAYBACK_ACK":
            try validateFields(object, required: ["version", "type", "session_id", "turn_id", "generation"])
            _ = try uuid(object, "session_id")
            _ = try uuid(object, "turn_id")
            _ = try unsigned(object, "generation")
        case "PLAYBACK_CONFIRMED", "PLAYBACK_CANCELLED", "PLAYBACK_FAILED":
            var fields: Set<String> = ["version", "type", "event_id", "session_id", "turn_id", "sentence_id"]
            if type == "PLAYBACK_FAILED" { fields.insert("error_class") }
            try validateFields(object, required: fields)
            _ = try uuid(object, "event_id")
            _ = try uuid(object, "session_id")
            _ = try uuid(object, "turn_id")
            _ = try uuid(object, "sentence_id")
            if type == "PLAYBACK_FAILED" {
                guard let errorClass = object["error_class"] as? String,
                      !errorClass.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    throw VoiceSocketClientError.invalidMessage("invalid_playback_error_class")
                }
            }
        default:
            throw VoiceSocketClientError.invalidMessage("unknown_message_type")
        }
    }

    private static func validateFields(
        _ object: [String: Any],
        required: Set<String>,
        optional: Set<String> = []
    ) throws {
        let unknown = Set(object.keys).subtracting(required).subtracting(optional)
        if !unknown.isEmpty {
            if unknown.contains(where: isMicrophonePayloadField) {
                throw VoiceSocketClientError.invalidMessage("microphone_payload_forbidden")
            }
            throw VoiceSocketClientError.invalidMessage("unknown_field")
        }
        guard required.isSubset(of: object.keys) else {
            throw VoiceSocketClientError.invalidMessage("missing_required_field")
        }
    }

    private static func isMicrophonePayloadField(_ key: String) -> Bool {
        let normalized = key.lowercased().replacingOccurrences(of: "_", with: "").replacingOccurrences(of: "-", with: "")
        return microphoneFieldTokens.contains { normalized.contains($0) }
    }

    private static func uuid(_ object: [String: Any], _ key: String) throws -> UUID {
        guard let raw = object[key] as? String,
              let value = UUID(uuidString: raw),
              canonical(value) == raw else {
            throw VoiceSocketClientError.invalidMessage("malformed_id:\(key)")
        }
        return value
    }

    private static func string(_ object: [String: Any], _ key: String) throws -> String {
        guard let value = object[key] as? String else {
            throw VoiceSocketClientError.invalidMessage("invalid_\(key)")
        }
        return value
    }

    private static func integer(_ object: [String: Any], _ key: String) throws -> Int {
        guard let value = integerValue(object[key]) else {
            throw VoiceSocketClientError.invalidMessage("invalid_\(key)")
        }
        return value
    }

    private static func unsigned(_ object: [String: Any], _ key: String) throws -> UInt64 {
        guard let value = integerValue(object[key]), value >= 0 else {
            throw VoiceSocketClientError.invalidMessage("invalid_\(key)")
        }
        return UInt64(value)
    }

    private static func integerValue(_ value: Any?) -> Int? {
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID() else {
            return nil
        }
        let decimal = number.decimalValue
        guard NSDecimalNumber(decimal: decimal).doubleValue.rounded() == NSDecimalNumber(decimal: decimal).doubleValue else {
            return nil
        }
        let int = number.intValue
        guard NSNumber(value: int) == number else { return nil }
        return int
    }

    private static func canonical(_ uuid: UUID) -> String {
        uuid.uuidString.lowercased()
    }
}
