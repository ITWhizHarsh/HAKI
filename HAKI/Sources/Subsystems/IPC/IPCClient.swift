// IPCClient.swift
// HAKI — IPC Subsystem
//
// Swift-side message types and client stub for the bidirectional streaming
// channel between the HAKI Shell (Body) and the HAKI Core (Mind).
//
// Transport: gRPC (preferred) or JSON-RPC over a UNIX domain socket.
//   Socket path: ~/Library/Application Support/HAKI/haki_core.sock
//
// Proto contract: proto/haki_ipc.proto  (package haki, service HAKICore)
//
// This file hand-mirrors the proto message types as native Swift structs so
// the rest of the shell can compile and reason about IPC messages today.
// When `protoc-gen-grpc-swift` and `swift-protobuf` are available the
// generated Swift types will replace these hand-written ones (see
// proto/README.md for regeneration instructions).
//
// Full gRPC wiring:  Phase 0 Task 1.4
// Implements:        Req 3.1 (streaming transport), Design: Process & Threading Model

import Foundation

// MARK: - Primitive streaming units

/// A 20 ms PCM audio frame captured by AVAudioEngine.
/// Mirrors proto: `message AudioFrame`
public struct HAKIAudioFrame: Sendable {
    /// Raw PCM Int16 LE samples (20 ms at the configured sample rate).
    public let samples: Data
    /// Monotonic wall-clock timestamp at capture (milliseconds).
    public let timestampMs: UInt64
    /// Monotonically increasing sequence number, per turn.
    public let sequenceNum: UInt32
    /// Sample rate in Hz (typically 16 000 for STT input).
    public let sampleRate: UInt32
    /// Channel count (1 = mono).
    public let channels: UInt32

    public init(
        samples: Data,
        timestampMs: UInt64,
        sequenceNum: UInt32,
        sampleRate: UInt32 = 16_000,
        channels: UInt32 = 1
    ) {
        self.samples = samples
        self.timestampMs = timestampMs
        self.sequenceNum = sequenceNum
        self.sampleRate = sampleRate
        self.channels = channels
    }
}

/// Incremental STT output.
/// Mirrors proto: `message PartialTranscript`
public struct HAKIPartialTranscript: Sendable {
    public let text: String
    /// `true` when this is the committed, final transcript for the turn.
    public let isFinal: Bool
    public let sequenceNum: UInt32

    public init(text: String, isFinal: Bool, sequenceNum: UInt32) {
        self.text = text
        self.isFinal = isFinal
        self.sequenceNum = sequenceNum
    }
}

/// Acoustic features extracted by the VAD / audio analyser.
/// Forwarded to Core for Mood_Detector classification (Req 4.1).
/// Mirrors proto: `message AudioFeatures`
public struct HAKIAudioFeatures: Sendable {
    /// Fundamental frequency in Hz.
    public let pitchHz: Float
    /// RMS energy in dBFS.
    public let energyDb: Float
    /// Speech duration (ms) used for feature extraction.
    public let durationMs: UInt32

    public init(pitchHz: Float, energyDb: Float, durationMs: UInt32) {
        self.pitchHz = pitchHz
        self.energyDb = energyDb
        self.durationMs = durationMs
    }
}

/// A single LLM output token streamed as soon as the model produces it.
/// Fine-grained so TTS sentence-chunking begins immediately (Req 3.1).
/// Mirrors proto: `message LLMToken`
public struct HAKILLMToken: Sendable {
    public let text: String
    public let sequenceNum: UInt32
    /// `true` on the final token of the turn.
    public let isLast: Bool

    public init(text: String, sequenceNum: UInt32, isLast: Bool) {
        self.text = text
        self.sequenceNum = sequenceNum
        self.isLast = isLast
    }
}

/// A fine-grained TTS audio chunk. Chunked at clause/sentence boundaries
/// so playback begins within 300 ms of the first words (Req 3.1).
/// Mirrors proto: `message TTSAudioChunk`
public struct HAKITTSAudioChunk: Sendable {
    /// Raw PCM Int16 LE samples.
    public let samples: Data
    public let sequenceNum: UInt32
    /// `true` on the final chunk of the turn.
    public let isLast: Bool
    /// Sample rate in Hz (typically 22 050 or 24 000 for TTS output).
    public let sampleRate: UInt32

    public init(samples: Data, sequenceNum: UInt32, isLast: Bool, sampleRate: UInt32 = 22_050) {
        self.samples = samples
        self.sequenceNum = sequenceNum
        self.isLast = isLast
        self.sampleRate = sampleRate
    }
}

/// Control / lifecycle signals.
/// Mirrors proto: `message ControlEvent`
public struct HAKIControlEvent: Sendable {
    public enum EventType: Sendable {
        /// Abort the current turn immediately.
        case cancel
        /// User started speaking mid-response — stop TTS (Req 3.3).
        case bargeIn
        /// VAD detected 800 ms silence — end of user speech (Req 3.2).
        case endOfSpeech
        /// Keep-alive ping on an idle stream.
        case heartbeat
    }
    public let eventType: EventType
    public let sequenceNum: UInt32

    public init(eventType: EventType, sequenceNum: UInt32) {
        self.eventType = eventType
        self.sequenceNum = sequenceNum
    }
}

// MARK: - Turn-level messages

/// Complete input to a new conversational turn.
/// Mirrors proto: `message TurnRequest`
public struct HAKITurnRequest: Sendable {
    /// UUID scoped to this session.
    public let turnId: String
    /// Committed STT text for the turn.
    public let transcript: String
    /// Language composition: "hindi" | "english" | "hinglish" | "unknown"
    public let languageComposition: String
    /// Acoustic features for Mood_Detector.
    public let audioFeatures: HAKIAudioFeatures

    public init(
        turnId: String,
        transcript: String,
        languageComposition: String,
        audioFeatures: HAKIAudioFeatures
    ) {
        self.turnId = turnId
        self.transcript = transcript
        self.languageComposition = languageComposition
        self.audioFeatures = audioFeatures
    }
}

// MARK: - Stream envelope types

/// All messages the Swift shell sends on the bidirectional stream (upstream).
/// Mirrors proto: `message ClientMessage` (oneof payload)
public enum ClientMessage: Sendable {
    case audioFrame(HAKIAudioFrame)
    case partialTranscript(HAKIPartialTranscript)
    case turnRequest(HAKITurnRequest)
    case controlEvent(HAKIControlEvent)
}

/// All messages the Python Core sends on the bidirectional stream (downstream).
/// Mirrors proto: `message ServerMessage` (oneof payload)
public enum ServerMessage: Sendable {
    case partialTranscript(HAKIPartialTranscript)
    case llmToken(HAKILLMToken)
    case ttsAudioChunk(HAKITTSAudioChunk)
    case controlEvent(HAKIControlEvent)
    case error(String)
}

// MARK: - IPCClientProtocol

/// The contract for the Swift-side gRPC streaming client.
/// Full implementation wires to the generated grpc-swift stubs in Task 1.4.
public protocol IPCClientProtocol: AnyObject, Sendable {
    /// Open the streaming channel to the Core.
    func connect() async throws
    /// Close the channel gracefully.
    func disconnect() async
    /// Send a message upstream to the Core.
    func send(_ message: ClientMessage) async throws
    /// Async stream of messages received from the Core.
    var inbound: AsyncStream<ServerMessage> { get }
    /// `true` when the channel is open and healthy.
    var isConnected: Bool { get }
}

// MARK: - IPCClient (stub — wired to generated stubs in Task 1.4)

/// Production gRPC/JSON-RPC client over a UNIX domain socket.
/// Phase 0: defines the full message API; actual socket I/O wired in Task 1.4.
///
/// Socket path must match `CoreProcessManager.socketPath`.
public final class IPCClient: IPCClientProtocol, @unchecked Sendable {

    // MARK: Configuration

    /// UNIX domain socket path — e.g. `~/Library/Application Support/HAKI/haki_core.sock`
    public let socketPath: URL

    // MARK: State

    public private(set) var isConnected: Bool = false
    private var inboundContinuation: AsyncStream<ServerMessage>.Continuation?

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
        // TODO (Task 1.4): create grpc-swift channel, set up bidirectional
        // streaming call to HAKICore/StreamTurn, start inbound pump task.
        throw IPCError.notImplemented
    }

    public func disconnect() async {
        isConnected = false
        inboundContinuation?.finish()
    }

    public func send(_ message: ClientMessage) async throws {
        guard isConnected else { throw IPCError.notConnected }
        // TODO (Task 1.4): serialise ClientMessage to proto bytes and write
        // to the gRPC stream.
    }
}

// MARK: - IPCError

public enum IPCError: Error, Sendable {
    case notImplemented
    case notConnected
    case socketUnavailable(URL)
    case protocolError(String)
}
