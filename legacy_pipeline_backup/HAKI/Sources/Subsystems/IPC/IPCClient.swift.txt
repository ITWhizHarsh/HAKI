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
        /// Core has started speaking a response (Python-side TTS playback began).
        /// The shell arms barge-in detection so HAKI's own voice is not treated
        /// as a new user turn.
        case speakingStarted
        /// Core has finished speaking a response (Python-side TTS playback ended).
        case speakingStopped
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

// MARK: - Calendar proposal (Scheduler — Req 11.1, 11.2)

/// A calendar event proposal sent from Core to Shell for user confirmation.
///
/// The user can confirm, reject, or edit the proposal via the UI panel.
/// Mirrors the Python `CalendarProposal` dict sent as `PROPOSAL` IPC message.
public struct HAKICalendarProposal: Sendable {
    /// Unique proposal ID (stable across confirm/reject/edit actions).
    public let proposalId: String
    /// Event title extracted from the actionable item.
    public var title: String
    /// Date string ("YYYY-MM-DD"), `nil` if not extracted.
    public var date: String?
    /// Time string ("HH:mm"), `nil` if not extracted.
    public var time: String?
    /// Optional location string.
    public var location: String?
    /// Event description.
    public var description: String
    /// `true` when the date or time is missing and user clarification is needed.
    public let needsClarification: Bool
    /// Current lifecycle status: "proposed" | "confirmed" | "rejected" | "failed"
    public var status: String

    public init(
        proposalId: String,
        title: String,
        date: String?,
        time: String?,
        location: String?,
        description: String,
        needsClarification: Bool,
        status: String = "proposed"
    ) {
        self.proposalId = proposalId
        self.title = title
        self.date = date
        self.time = time
        self.location = location
        self.description = description
        self.needsClarification = needsClarification
        self.status = status
    }
}

// MARK: - Reminder notification (Scheduler — Req 12.6)

/// An in-app reminder surfaced by the Scheduler via the REMINDER IPC message.
///
/// Mirrors the Python `Reminder` + `Task` dict combined for display.
public struct HAKIReminderNotification: Sendable {
    /// Unique reminder ID.
    public let reminderId: String
    /// The ID of the task this reminder belongs to.
    public let taskId: String
    /// Task title (shown in the notification).
    public let taskTitle: String
    /// Severity label (e.g. "EXAM", "BIRTHDAY", "DEFAULT").
    public let severity: String
    /// Formatted fire-at timestamp (ISO-8601).
    public let fireAt: String
    /// `true` for the birthday day-of prompt (Req 12.5).
    public let isBirthdayDayOf: Bool

    public init(
        reminderId: String,
        taskId: String,
        taskTitle: String,
        severity: String,
        fireAt: String,
        isBirthdayDayOf: Bool = false
    ) {
        self.reminderId = reminderId
        self.taskId = taskId
        self.taskTitle = taskTitle
        self.severity = severity
        self.fireAt = fireAt
        self.isBirthdayDayOf = isBirthdayDayOf
    }
}

// MARK: - Automation progress (Automation_Library — Req 17.5)

/// A step-by-step progress event from a running automation.
///
/// Mirrors the Python `AUTOMATION_PROGRESS` IPC message.
/// Surfaced in the automation-progress panel in the SwiftUI menu-bar UI.
public struct HAKIAutomationProgress: Sendable {
    /// Name of the automation being run.
    public let automationName: String
    /// Label of the current step (intent string or step ID).
    public let step: String
    /// Step status: "started" | "completed" | "failed" | "plan_complete" | "not_found"
    public let status: String
    /// Optional human-readable message for this status.
    public let message: String

    public init(
        automationName: String,
        step: String,
        status: String,
        message: String = ""
    ) {
        self.automationName = automationName
        self.step = step
        self.status = status
        self.message = message
    }
}

/// An image generated or edited by the Image_Studio, delivered from Core to Shell.
///
/// The Python Core generates the image, saves it to disk (Req 15.4), and sends
/// this message so the Swift UI can display it inline in the chat/image panel.
/// ``savedPath`` is set when the save succeeded (Req 15.4); when it is nil
/// the save failed but the image is retained in-session (Req 15.5).
///
/// Mirrors proto: `message ImageResponse`
public struct HAKIImageResponse: Sendable {
    /// Unique image ID within the current session.
    public let imageId: String
    /// Short label, e.g. "Image 3".
    public let displayLabel: String
    /// Raw image bytes (PNG).  May be empty when ``savedPath`` is set and the
    /// Shell prefers to load from disk.
    public let imageData: Data
    /// Absolute path on disk where the image was saved, or nil on save failure.
    public let savedPath: String?
    /// Human-readable confirmation or failure message for the user (Req 15.4, 15.5, 15.6).
    public let message: String
    /// true when generation/editing succeeded; false when it failed (Req 15.6).
    public let success: Bool

    public init(
        imageId: String,
        displayLabel: String,
        imageData: Data,
        savedPath: String?,
        message: String,
        success: Bool
    ) {
        self.imageId = imageId
        self.displayLabel = displayLabel
        self.imageData = imageData
        self.savedPath = savedPath
        self.message = message
        self.success = success
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
    /// An image generated or edited by the Image_Studio (Req 15.1, 15.2, 15.3).
    case imageResponse(HAKIImageResponse)
    /// A calendar event proposal for user confirmation (Req 11.1).
    case proposalReceived(HAKICalendarProposal)
    /// An in-app reminder notification (Req 12.6).
    case reminderFired(HAKIReminderNotification)
    /// Step-by-step automation progress event (Req 17.5).
    case automationProgress(HAKIAutomationProgress)
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
