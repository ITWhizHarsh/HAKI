// JSONIPCClient.swift
// HAKI — IPC Subsystem
//
// A line-delimited JSON transport over a UNIX domain socket using
// Apple's Network framework (NWConnection / NWEndpoint.unix).
//
// This is the Phase 0 IPC implementation.  It avoids the grpc-swift
// dependency while delivering the same bidirectional streaming API
// defined by IPCClientProtocol.  When grpc-swift becomes available in a
// future task the generated client can be dropped in behind the same
// protocol without changing any call site.
//
// Message format (each direction):
//   One JSON object per line, terminated by "\n".
//   ClientMessage: { "type": "...", "payload": {...} }
//   ServerMessage: { "type": "...", "payload": {...} }
//
// Reconnect behaviour:
//   If the connection drops while isConnected == true, the client waits
//   1 second and retries up to 5 times before giving up.
//
// Design: Architecture, Security Considerations (local IPC only).
// Requirements: 3.1

import Foundation
import Network

// MARK: - Codable message envelopes

/// Generic JSON envelope sent upstream to the Core.
private struct JSONClientEnvelope: Encodable {
    let type: String
    let payload: AnyEncodable
}

/// Generic JSON envelope received downstream from the Core.
private struct JSONServerEnvelope: Decodable {
    let type: String
    let payload: AnyCodable
}

// MARK: - JSONIPCClient

/// Production IPC client — line-delimited JSON over a UNIX domain socket.
///
/// Satisfies `IPCClientProtocol` so it can be substituted for the future
/// grpc-swift client without changing any call site.
public final class JSONIPCClient: IPCClientProtocol, @unchecked Sendable {

    // MARK: Configuration

    /// UNIX domain socket path matching `CoreProcessManager.socketPath`.
    public let socketPath: URL

    // MARK: State

    public private(set) var isConnected: Bool = false

    private var connection: NWConnection?
    private var inboundContinuation: AsyncStream<ServerMessage>.Continuation?
    private let queue = DispatchQueue(label: "haki.ipc.json", qos: .userInteractive)

    /// Reconnect state
    private static let maxReconnectAttempts = 5
    private static let reconnectDelaySeconds: Double = 1.0
    private var reconnectAttempts = 0
    private var intentionallyStopped = false

    // MARK: AsyncStream

    public lazy var inbound: AsyncStream<ServerMessage> = {
        AsyncStream { [weak self] continuation in
            self?.inboundContinuation = continuation
        }
    }()

    // MARK: Init

    public init(socketPath: URL) {
        self.socketPath = socketPath
    }

    // MARK: IPCClientProtocol

    public func connect() async throws {
        intentionallyStopped = false
        reconnectAttempts = 0
        try await openConnection()
    }

    public func disconnect() async {
        intentionallyStopped = true
        teardown()
    }

    public func send(_ message: ClientMessage) async throws {
        guard isConnected, let conn = connection else {
            throw IPCError.notConnected
        }

        let envelope = try encode(message)
        let data = envelope + Data([UInt8(ascii: "\n")])

        return try await withCheckedThrowingContinuation { continuation in
            conn.send(
                content: data,
                completion: .contentProcessed { error in
                    if let error = error {
                        continuation.resume(throwing: IPCError.protocolError(error.localizedDescription))
                    } else {
                        continuation.resume()
                    }
                }
            )
        }
    }

    // MARK: - Private: Connection lifecycle

    private func openConnection() async throws {
        let unixConn = NWConnection(
            to: NWEndpoint.unix(path: socketPath.path),
            using: NWParameters()
        )

        connection = unixConn

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            // Use a SendableBox to pass the 'resumed' flag across the concurrent
            // boundary without triggering Swift 6 concurrency warnings.
            let flag = SendableFlag()
            unixConn.stateUpdateHandler = { [weak self] state in
                guard let self else { return }
                switch state {
                case .ready:
                    if flag.checkAndSet() {
                        self.isConnected = true
                        self.reconnectAttempts = 0
                        continuation.resume()
                        // Start the inbound pump
                        self.receiveNextMessage()
                    }
                case .failed(let error):
                    if flag.checkAndSet() {
                        continuation.resume(throwing: IPCError.protocolError(error.localizedDescription))
                    }
                case .cancelled:
                    if flag.checkAndSet() {
                        continuation.resume(throwing: IPCError.socketUnavailable(self.socketPath))
                    }
                default:
                    break
                }
            }
            unixConn.start(queue: queue)
        }
    }

    private func teardown() {
        isConnected = false
        connection?.cancel()
        connection = nil
        if intentionallyStopped {
            inboundContinuation?.finish()
            inboundContinuation = nil
        }
    }

    private func handleConnectionDrop() {
        isConnected = false
        guard !intentionallyStopped else { return }

        reconnectAttempts += 1
        guard reconnectAttempts <= Self.maxReconnectAttempts else {
            print("[JSONIPCClient] Max reconnect attempts reached — giving up.")
            inboundContinuation?.finish()
            return
        }

        print("[JSONIPCClient] Connection dropped — reconnecting in \(Self.reconnectDelaySeconds)s (attempt \(reconnectAttempts)/\(Self.maxReconnectAttempts))…")
        Task {
            try? await Task.sleep(nanoseconds: UInt64(Self.reconnectDelaySeconds * 1_000_000_000))
            guard !self.intentionallyStopped else { return }
            try? await self.openConnection()
        }
    }

    // MARK: - Private: Receive loop

    private func receiveNextMessage() {
        connection?.receive(
            minimumIncompleteLength: 1,
            maximumLength: 65_536
        ) { [weak self] content, _, isComplete, error in
            guard let self else { return }

            if let error = error {
                print("[JSONIPCClient] Receive error: \(error)")
                self.handleConnectionDrop()
                return
            }

            if let data = content, !data.isEmpty {
                self.processIncomingData(data)
            }

            if isComplete {
                self.handleConnectionDrop()
            } else {
                self.receiveNextMessage()
            }
        }
    }

    /// Accumulated bytes for incomplete lines
    private var receiveBuffer = Data()

    private func processIncomingData(_ data: Data) {
        receiveBuffer.append(data)

        // Process all complete newline-terminated lines
        while let newlineRange = receiveBuffer.range(of: Data([UInt8(ascii: "\n")])) {
            let lineData = receiveBuffer[receiveBuffer.startIndex..<newlineRange.lowerBound]
            receiveBuffer.removeSubrange(receiveBuffer.startIndex...newlineRange.lowerBound)

            guard !lineData.isEmpty else { continue }
            decode(lineData)
        }
    }

    // MARK: - Private: Encode / Decode

    private func encode(_ message: ClientMessage) throws -> Data {
        let dict: [String: Any]
        switch message {
        case .audioFrame(let frame):
            dict = [
                "type": "AUDIO_FRAME",
                "payload": [
                    "samples": frame.samples.base64EncodedString(),
                    "timestamp_ms": frame.timestampMs,
                    "sequence_num": frame.sequenceNum,
                    "sample_rate": frame.sampleRate,
                    "channels": frame.channels,
                ] as [String: Any],
            ]
        case .partialTranscript(let pt):
            dict = [
                "type": "PARTIAL_TRANSCRIPT",
                "payload": [
                    "text": pt.text,
                    "is_final": pt.isFinal,
                    "sequence_num": pt.sequenceNum,
                ] as [String: Any],
            ]
        case .turnRequest(let tr):
            dict = [
                "type": "TURN_REQUEST",
                "payload": [
                    "turn_id": tr.turnId,
                    "transcript": tr.transcript,
                    "language_composition": tr.languageComposition,
                    "audio_features": [
                        "pitch_hz": tr.audioFeatures.pitchHz,
                        "energy_db": tr.audioFeatures.energyDb,
                        "duration_ms": tr.audioFeatures.durationMs,
                    ] as [String: Any],
                ] as [String: Any],
            ]
        case .controlEvent(let ce):
            dict = [
                "type": "CONTROL_EVENT",
                "payload": [
                    "event_type": ce.eventType.rawStringValue,
                    "sequence_num": ce.sequenceNum,
                ] as [String: Any],
            ]
        }
        return try JSONSerialization.data(withJSONObject: dict, options: [])
    }

    private func decode(_ data: Data) {
        guard
            let obj = try? JSONSerialization.jsonObject(with: data),
            let dict = obj as? [String: Any],
            let typeStr = dict["type"] as? String
        else {
            print("[JSONIPCClient] Failed to decode server message: \(String(data: data, encoding: .utf8) ?? "<binary>")")
            return
        }

        let payload = dict["payload"] as? [String: Any] ?? [:]
        let serverMsg: ServerMessage

        switch typeStr {
        case "HEARTBEAT":
            serverMsg = .controlEvent(
                HAKIControlEvent(eventType: .heartbeat, sequenceNum: 0)
            )
        case "PARTIAL_TRANSCRIPT":
            let text = payload["text"] as? String ?? ""
            let isFinal = payload["is_final"] as? Bool ?? false
            let seqNum = payload["sequence_num"] as? UInt32 ?? 0
            serverMsg = .partialTranscript(
                HAKIPartialTranscript(text: text, isFinal: isFinal, sequenceNum: seqNum)
            )
        case "LLM_TOKEN":
            let text = payload["text"] as? String ?? ""
            let seqNum = payload["sequence_num"] as? UInt32 ?? 0
            let isLast = payload["is_last"] as? Bool ?? false
            serverMsg = .llmToken(
                HAKILLMToken(text: text, sequenceNum: seqNum, isLast: isLast)
            )
        case "TTS_AUDIO_CHUNK":
            let b64 = payload["samples"] as? String ?? ""
            let samples = Data(base64Encoded: b64) ?? Data()
            let seqNum = payload["sequence_num"] as? UInt32 ?? 0
            let isLast = payload["is_last"] as? Bool ?? false
            let sampleRate = payload["sample_rate"] as? UInt32 ?? 22_050
            serverMsg = .ttsAudioChunk(
                HAKITTSAudioChunk(samples: samples, sequenceNum: seqNum, isLast: isLast, sampleRate: sampleRate)
            )
        case "CONTROL_EVENT":
            let eventTypeStr = payload["event_type"] as? String ?? ""
            let seqNum = payload["sequence_num"] as? UInt32 ?? 0
            let eventType = HAKIControlEvent.EventType(rawStringValue: eventTypeStr) ?? .heartbeat
            serverMsg = .controlEvent(
                HAKIControlEvent(eventType: eventType, sequenceNum: seqNum)
            )
        case "ERROR":
            let msg = payload["message"] as? String ?? "unknown error"
            serverMsg = .error(msg)
        default:
            print("[JSONIPCClient] Unknown server message type: \(typeStr)")
            return
        }

        inboundContinuation?.yield(serverMsg)
    }
}

// MARK: - Concurrency helpers

/// A thread-safe, Sendable flag that can be atomically checked-and-set once.
private final class SendableFlag: @unchecked Sendable {
    private var _value = false
    private let lock = NSLock()

    /// Returns `true` on the first call; `false` on every subsequent call.
    func checkAndSet() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !_value else { return false }
        _value = true
        return true
    }
}

// MARK: - HAKIControlEvent.EventType raw string helpers

private extension HAKIControlEvent.EventType {
    var rawStringValue: String {
        switch self {
        case .cancel: return "CANCEL"
        case .bargeIn: return "BARGE_IN"
        case .endOfSpeech: return "END_OF_SPEECH"
        case .heartbeat: return "HEARTBEAT"
        }
    }

    init?(rawStringValue: String) {
        switch rawStringValue {
        case "CANCEL": self = .cancel
        case "BARGE_IN": self = .bargeIn
        case "END_OF_SPEECH": self = .endOfSpeech
        case "HEARTBEAT": self = .heartbeat
        default: return nil
        }
    }
}

// MARK: - Helpers for JSON serialisation

/// Type-erased Encodable wrapper.
private struct AnyEncodable: Encodable {
    private let _encode: (Encoder) throws -> Void
    init<T: Encodable>(_ value: T) { _encode = value.encode }
    func encode(to encoder: Encoder) throws { try _encode(encoder) }
}

/// Type-erased Codable wrapper for decoding arbitrary JSON.
private struct AnyCodable: Codable {
    let value: Any
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let dict = try? container.decode([String: AnyCodable].self) {
            value = dict.mapValues { $0.value }
        } else if let arr = try? container.decode([AnyCodable].self) {
            value = arr.map { $0.value }
        } else if let str = try? container.decode(String.self) {
            value = str
        } else if let int = try? container.decode(Int.self) {
            value = int
        } else if let dbl = try? container.decode(Double.self) {
            value = dbl
        } else if let bool = try? container.decode(Bool.self) {
            value = bool
        } else {
            value = NSNull()
        }
    }
    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch value {
        case let str as String: try container.encode(str)
        case let int as Int: try container.encode(int)
        case let dbl as Double: try container.encode(dbl)
        case let bool as Bool: try container.encode(bool)
        default: try container.encodeNil()
        }
    }
}
