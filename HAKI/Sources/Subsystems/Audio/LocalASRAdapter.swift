// LocalASRAdapter.swift
// HAKI — local-only CoreML Qwen3 ASR and transcript-generation contract.
//
// The normal voice path never selects a network recognizer. This adapter accepts
// only VoiceAudioController's normalized 16 kHz mono frames, normalizes text
// before it reaches the transcript socket, and converts empty finals into a
// repeat prompt rather than an LLM turn.

import CoreML
import Foundation
import HAKIIPC

public enum LocalASRError: Error, Sendable, Equatable {
    case invalidFrameFormat
    case turnNotStarted
    case turnCancelled
    case duplicateTurn
    case localModelUnavailable
    case unsupportedLocalModelOutput
}

public struct ASRHypothesis: Sendable, Equatable {
    public let turnID: UUID
    public let text: String
    public let isFinal: Bool
    public let language: VoiceTranscriptLanguage
    public let captureStartedMonotonicNs: UInt64
    public let captureEndedMonotonicNs: UInt64

    public init(
        turnID: UUID,
        text: String,
        isFinal: Bool,
        language: VoiceTranscriptLanguage,
        captureStartedMonotonicNs: UInt64,
        captureEndedMonotonicNs: UInt64
    ) {
        self.turnID = turnID
        self.text = text
        self.isFinal = isFinal
        self.language = language
        self.captureStartedMonotonicNs = captureStartedMonotonicNs
        self.captureEndedMonotonicNs = captureEndedMonotonicNs
    }
}

/// Content-safe ASR failure metadata. It intentionally excludes the recognized
/// text and every audio/PCM field.
public struct ASREmptyFinalDiagnostic: Sendable, Equatable {
    public let turnID: UUID
    public let captureStartedMonotonicNs: UInt64
    public let captureEndedMonotonicNs: UInt64
    public let stage = "asr"
    public let reason = "empty_final"
    public let repeatPrompt: String
}

public enum ASRFinalization: Sendable, Equatable {
    case transcript([ASRHypothesis])
    case repeatPrompt(ASREmptyFinalDiagnostic)
}

/// Common local ASR contract. No implementation has a cloud endpoint or a raw
/// microphone socket transport; frame bytes remain process/local-ring memory.
public protocol LocalASRAdapter: Sendable {
    func startTurn(_ id: UUID) async throws
    func consume(_ frame: VoiceAudioFrame) async throws -> [ASRHypothesis]
    func finalize(turnID: UUID) async throws -> ASRFinalization
    func cancel(turnID: UUID) async
}

public enum VoiceASRBackend: String, Sendable, Equatable {
    case coreMLQwen3Local = "qwen3_asr_coreml"
}

/// Production ASR selection is deliberately a closed, local-only enum.
public struct VoiceASRConfiguration: Sendable, Equatable {
    public let backend: VoiceASRBackend
    public let modelID: String
    public let modelURL: URL
    public let sampleRateHz: Int
    public let vocabularyVersion: String

    /// Explicit local configuration seam for app composition and deterministic
    /// fixtures. The closed backend enum prevents cloud/legacy ASR selection.
    public init(
        backend: VoiceASRBackend = .coreMLQwen3Local,
        modelID: String,
        modelURL: URL,
        sampleRateHz: Int,
        vocabularyVersion: String
    ) {
        self.backend = backend
        self.modelID = modelID
        self.modelURL = modelURL
        self.sampleRateHz = sampleRateHz
        self.vocabularyVersion = vocabularyVersion
    }

    public init(localAssets: VoiceLocalAssetConfiguration) throws {
        let artifact = try localAssets.coreMLQwen3ASRArtifact()
        self.init(
            modelID: artifact.modelID,
            modelURL: artifact.artifactURL,
            sampleRateHz: artifact.sampleRateHz,
            vocabularyVersion: artifact.vocabularyVersion
        )
    }
}

/// Inference seam for the CoreML package. Fixture implementations exercise the
/// transcript contract without a model download; production loads only the
/// verified local `mlmodelc` artifact named by `VoiceASRConfiguration`.
public protocol CoreMLQwen3Inference: Sendable {
    func partialTranscript(turnID: UUID, pcmS16LE: Data, sampleRateHz: Int) async throws -> String?
    func finalTranscript(turnID: UUID, pcmS16LE: Data, sampleRateHz: Int) async throws -> String?
    func resetPartial(turnID: UUID) async
}

/// Dynamic bridge for a provisioned Qwen3 CoreML artifact. The artifact must
/// expose a Float32 waveform input and a String transcript output; names can be
/// supplied with local CoreML metadata keys `haki.asr.audio_input` and
/// `haki.asr.text_output`. Unsupported artifact schemas fail locally and never
/// fall back to a remote recognizer.
public final class CoreMLQwen3ModelInference: CoreMLQwen3Inference, @unchecked Sendable {
    private let model: MLModel
    private let inputName: String
    private let outputName: String

    public init(configuration: VoiceASRConfiguration) throws {
        guard configuration.backend == .coreMLQwen3Local,
              configuration.modelID == VoiceLocalAssetConfiguration.coreMLASRModelID,
              configuration.sampleRateHz == CoreMLQwen3ASRAdapter.requiredSampleRateHz,
              FileManager.default.fileExists(atPath: configuration.modelURL.path) else {
            throw LocalASRError.localModelUnavailable
        }
        do {
            let modelConfiguration = MLModelConfiguration()
            modelConfiguration.computeUnits = .all
            model = try MLModel(contentsOf: configuration.modelURL, configuration: modelConfiguration)
        } catch {
            throw LocalASRError.localModelUnavailable
        }
        let metadata = model.modelDescription.metadata
        inputName = (metadata[MLModelMetadataKey(rawValue: "haki.asr.audio_input")] as? String)
            ?? model.modelDescription.inputDescriptionsByName.keys.sorted().first
            ?? "audio"
        outputName = (metadata[MLModelMetadataKey(rawValue: "haki.asr.text_output")] as? String)
            ?? model.modelDescription.outputDescriptionsByName.keys.sorted().first
            ?? "text"
    }

    public func partialTranscript(turnID: UUID, pcmS16LE: Data, sampleRateHz: Int) async throws -> String? {
        // Qwen3 artifacts commonly produce partials only when their streaming
        // model declares it. A nil result means "no new partial", not fallback.
        try predict(pcmS16LE: pcmS16LE, sampleRateHz: sampleRateHz)
    }

    public func finalTranscript(turnID: UUID, pcmS16LE: Data, sampleRateHz: Int) async throws -> String? {
        try predict(pcmS16LE: pcmS16LE, sampleRateHz: sampleRateHz)
    }

    public func resetPartial(turnID: UUID) async {}

    private func predict(pcmS16LE: Data, sampleRateHz: Int) throws -> String? {
        guard sampleRateHz == CoreMLQwen3ASRAdapter.requiredSampleRateHz,
              pcmS16LE.count.isMultiple(of: 2),
              !pcmS16LE.isEmpty else {
            throw LocalASRError.invalidFrameFormat
        }
        let sampleCount = pcmS16LE.count / 2
        guard let waveform = try? MLMultiArray(
            shape: [NSNumber(value: sampleCount)],
            dataType: .float32
        ) else {
            throw LocalASRError.localModelUnavailable
        }
        pcmS16LE.withUnsafeBytes { source in
            let bytes = source.bindMemory(to: UInt8.self)
            for index in 0..<sampleCount {
                let byteOffset = index * 2
                let raw = UInt16(bytes[byteOffset]) | (UInt16(bytes[byteOffset + 1]) << 8)
                waveform[index] = NSNumber(value: Float(Int16(bitPattern: raw)) / Float(Int16.max))
            }
        }
        do {
            let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [inputName: waveform]))
            return output.featureValue(for: outputName)?.stringValue
        } catch {
            throw LocalASRError.unsupportedLocalModelOutput
        }
    }
}

/// Production-default local adapter. It maintains turn-local PCM and partial
/// state on one actor, resetting only the partial on a capture sequence gap.
public actor CoreMLQwen3ASRAdapter: LocalASRAdapter {
    public static let requiredSampleRateHz = 16_000
    public static let emptyFinalRepeatPrompt = "I didn't catch that — could you repeat?"

    private let configuration: VoiceASRConfiguration
    private let inference: any CoreMLQwen3Inference
    private var turns: [UUID: TurnState] = [:]
    private var activeTurnID: UUID?
    private var cancelledTurns = Set<UUID>()

    public init(configuration: VoiceASRConfiguration, inference: any CoreMLQwen3Inference) {
        self.configuration = configuration
        self.inference = inference
    }

    public static func productionDefault(
        localAssets: VoiceLocalAssetConfiguration = VoiceLocalAssetConfiguration()
    ) throws -> CoreMLQwen3ASRAdapter {
        let configuration = try VoiceASRConfiguration(localAssets: localAssets)
        let inference = try CoreMLQwen3ModelInference(configuration: configuration)
        return CoreMLQwen3ASRAdapter(configuration: configuration, inference: inference)
    }

    public func startTurn(_ id: UUID) async throws {
        guard turns[id] == nil, activeTurnID == nil else { throw LocalASRError.duplicateTurn }
        cancelledTurns.remove(id)
        turns[id] = TurnState()
        activeTurnID = id
    }

    public func consume(_ frame: VoiceAudioFrame) async throws -> [ASRHypothesis] {
        guard frame.sampleRateHz == Self.requiredSampleRateHz,
              frame.channels == 1,
              !frame.pcmS16LE.isEmpty,
              frame.pcmS16LE.count.isMultiple(of: 2) else {
            throw LocalASRError.invalidFrameFormat
        }
        guard let turnID = activeTurnID,
              !cancelledTurns.contains(turnID),
              var state = turns[turnID] else {
            throw cancelledTurns.isEmpty ? LocalASRError.turnNotStarted : LocalASRError.turnCancelled
        }

        if let previous = state.lastSequence, frame.sequence != previous + 1 {
            state.lastPartial = nil
            state.pcm.removeAll(keepingCapacity: true)
            state.captureStartedMonotonicNs = frame.capturedAtMonotonicNs
            await inference.resetPartial(turnID: turnID)
        }
        state.lastSequence = frame.sequence
        state.captureStartedMonotonicNs = state.captureStartedMonotonicNs ?? frame.capturedAtMonotonicNs
        state.captureEndedMonotonicNs = frame.capturedAtMonotonicNs
        state.pcm.append(frame.pcmS16LE)
        turns[turnID] = state

        let rawPartial = try await inference.partialTranscript(
            turnID: turnID,
            pcmS16LE: state.pcm,
            sampleRateHz: configuration.sampleRateHz
        )
        guard var current = turns[turnID] else { throw LocalASRError.turnCancelled }
        let normalized = Self.normalize(rawPartial ?? "")
        guard !normalized.isEmpty, normalized != current.lastPartial,
              let started = current.captureStartedMonotonicNs,
              let ended = current.captureEndedMonotonicNs else {
            return []
        }
        current.lastPartial = normalized
        turns[turnID] = current
        return [ASRHypothesis(
            turnID: turnID,
            text: normalized,
            isFinal: false,
            language: Self.classifyLanguage(normalized),
            captureStartedMonotonicNs: started,
            captureEndedMonotonicNs: ended
        )]
    }

    public func finalize(turnID: UUID) async throws -> ASRFinalization {
        guard !cancelledTurns.contains(turnID) else { throw LocalASRError.turnCancelled }
        guard let state = turns.removeValue(forKey: turnID),
              let started = state.captureStartedMonotonicNs,
              let ended = state.captureEndedMonotonicNs else {
            throw LocalASRError.turnNotStarted
        }
        activeTurnID = nil
        let rawFinal = try await inference.finalTranscript(
            turnID: turnID,
            pcmS16LE: state.pcm,
            sampleRateHz: configuration.sampleRateHz
        )
        let normalized = Self.normalize(rawFinal ?? "")
        guard !normalized.isEmpty else {
            return .repeatPrompt(ASREmptyFinalDiagnostic(
                turnID: turnID,
                captureStartedMonotonicNs: started,
                captureEndedMonotonicNs: ended,
                repeatPrompt: Self.emptyFinalRepeatPrompt
            ))
        }

        let language = Self.classifyLanguage(normalized)
        var ordered: [ASRHypothesis] = []
        // A final after no emitted partial receives a final pre-commit partial,
        // preserving the protocol's partial-before-final ordering contract.
        if state.lastPartial == nil {
            ordered.append(ASRHypothesis(
                turnID: turnID,
                text: normalized,
                isFinal: false,
                language: language,
                captureStartedMonotonicNs: started,
                captureEndedMonotonicNs: ended
            ))
        }
        ordered.append(ASRHypothesis(
            turnID: turnID,
            text: normalized,
            isFinal: true,
            language: language,
            captureStartedMonotonicNs: started,
            captureEndedMonotonicNs: ended
        ))
        return .transcript(ordered)
    }

    public func cancel(turnID: UUID) async {
        turns.removeValue(forKey: turnID)
        if activeTurnID == turnID { activeTurnID = nil }
        cancelledTurns.insert(turnID)
        await inference.resetPartial(turnID: turnID)
    }

    public nonisolated static func normalize(_ raw: String) -> String {
        let nfc = raw.precomposedStringWithCanonicalMapping
        let withoutControls = String(String.UnicodeScalarView(
            nfc.unicodeScalars.filter { !CharacterSet.controlCharacters.contains($0) }
        ))
        return withoutControls
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    public nonisolated static func classifyLanguage(_ normalizedText: String) -> VoiceTranscriptLanguage {
        let scalars = normalizedText.unicodeScalars
        let devanagariCount = scalars.filter { (0x0900...0x097F).contains($0.value) }.count
        let latinCount = scalars.filter {
            (65...90).contains($0.value) || (97...122).contains($0.value)
        }.count
        if devanagariCount > 0 && latinCount > 0 { return .hinglish }
        if devanagariCount > 0 { return .hi }

        let romanHindiTokens: Set<String> = [
            "aaj", "acha", "accha", "batao", "hai", "hain", "haan", "kaise", "kal",
            "kar", "karna", "karo", "kya", "mera", "mujhe", "nahi", "nahin", "theek", "yaar"
        ]
        let tokens = normalizedText.lowercased().split { !$0.isLetter }.map(String.init)
        return latinCount > 0 && tokens.contains(where: romanHindiTokens.contains) ? .hinglish : .en
    }

    private struct TurnState {
        var pcm = Data()
        var lastSequence: UInt64?
        var captureStartedMonotonicNs: UInt64?
        var captureEndedMonotonicNs: UInt64?
        var lastPartial: String?
    }
}

/// Makes transcript-event sequencing explicit and supplies the single boolean
/// integration gate a pipeline uses before allocating an LLM turn.
public enum LocalASRTranscriptOutcome: Sendable, Equatable {
    case transcript([VoiceTranscriptEvent])
    case repeatPrompt(ASREmptyFinalDiagnostic)

    public var createsLLMTurn: Bool {
        if case .transcript(let events) = self {
            return events.contains(where: \.isFinal)
        }
        return false
    }
}

public actor LocalASRTranscriptGenerator {
    private let sessionID: UUID
    private let adapter: any LocalASRAdapter
    private var nextEventSequence: [UUID: UInt64] = [:]

    public init(sessionID: UUID, adapter: any LocalASRAdapter) {
        self.sessionID = sessionID
        self.adapter = adapter
    }

    public func startTurn(_ turnID: UUID) async throws {
        try await adapter.startTurn(turnID)
        nextEventSequence[turnID] = 0
    }

    public func consume(_ frame: VoiceAudioFrame) async throws -> [VoiceTranscriptEvent] {
        try await events(from: adapter.consume(frame))
    }

    public func finalize(turnID: UUID) async throws -> LocalASRTranscriptOutcome {
        switch try await adapter.finalize(turnID: turnID) {
        case .transcript(let hypotheses):
            let events = try await events(from: hypotheses)
            return .transcript(events)
        case .repeatPrompt(let diagnostic):
            nextEventSequence.removeValue(forKey: turnID)
            return .repeatPrompt(diagnostic)
        }
    }

    public func cancel(turnID: UUID) async {
        await adapter.cancel(turnID: turnID)
        nextEventSequence.removeValue(forKey: turnID)
    }

    private func events(from hypotheses: [ASRHypothesis]) async throws -> [VoiceTranscriptEvent] {
        try hypotheses.map { hypothesis in
            let sequence = nextEventSequence[hypothesis.turnID] ?? 0
            nextEventSequence[hypothesis.turnID] = sequence + 1
            return VoiceTranscriptEvent(
                sessionID: sessionID,
                turnID: hypothesis.turnID,
                eventSequence: sequence,
                text: hypothesis.text,
                isFinal: hypothesis.isFinal,
                language: hypothesis.language,
                captureStartedMonotonicNs: hypothesis.captureStartedMonotonicNs,
                captureEndedMonotonicNs: hypothesis.captureEndedMonotonicNs
            )
        }
    }
}
