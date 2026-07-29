// LocalASRAdapterTests.swift
// Focused fixtures for realtime-local-voice-agent task 4.3.
// Validates: Requirements 3.1–3.3, 3.7

#if canImport(XCTest)
import Foundation
import XCTest
@testable import HAKIAudio
@testable import HAKIIPC

private actor ScriptedQwen3Inference: CoreMLQwen3Inference {
    private var partials: [String?]
    private var finals: [String?]
    private(set) var resetTurns: [UUID] = []

    init(partials: [String?], finals: [String?]) {
        self.partials = partials
        self.finals = finals
    }

    func partialTranscript(turnID: UUID, pcmS16LE: Data, sampleRateHz: Int) async throws -> String? {
        partials.isEmpty ? nil : partials.removeFirst()
    }

    func finalTranscript(turnID: UUID, pcmS16LE: Data, sampleRateHz: Int) async throws -> String? {
        finals.isEmpty ? nil : finals.removeFirst()
    }

    func resetPartial(turnID: UUID) async {
        resetTurns.append(turnID)
    }
}

final class LocalASRAdapterTests: XCTestCase {
    func testHindiEnglishAndHinglishFixturesEmitNormalizedOrderedHypotheses() async throws {
        try await assertTranscript(
            partial: "  नमस्ते\n",
            final: "नमस्ते  दुनिया\u{0000}",
            expectedLanguage: .hi
        )
        try await assertTranscript(
            partial: "hello",
            final: "hello   world",
            expectedLanguage: .en
        )
        try await assertTranscript(
            partial: "Kal meeting",
            final: "Kal meeting  reschedule kar do",
            expectedLanguage: .hinglish
        )
    }

    func testFrameGapResetsOnlyPartialStateAndCancelledTurnCannotFinalize() async throws {
        let inference = ScriptedQwen3Inference(partials: ["first", "after gap"], finals: ["final"])
        let adapter = makeAdapter(inference: inference)
        let turnID = UUID()
        let sessionID = UUID()
        try await adapter.startTurn(turnID)
        _ = try await adapter.consume(frame(sessionID: sessionID, sequence: 1))
        let afterGap = try await adapter.consume(frame(sessionID: sessionID, sequence: 3))
        XCTAssertEqual(afterGap.first?.text, "after gap")
        XCTAssertEqual(await inference.resetTurns, [turnID])

        await adapter.cancel(turnID: turnID)
        do {
            _ = try await adapter.finalize(turnID: turnID)
            XCTFail("cancelled turn must not produce a final transcript")
        } catch {
            XCTAssertEqual(error as? LocalASRError, .turnCancelled)
        }
    }

    func testEmptyFinalEmitsDiagnosticRepeatOutcomeAndNeverCreatesLLMTurn() async throws {
        let inference = ScriptedQwen3Inference(partials: [nil], finals: [" \n\u{0001} "])
        let generator = LocalASRTranscriptGenerator(sessionID: UUID(), adapter: makeAdapter(inference: inference))
        let turnID = UUID()
        try await generator.startTurn(turnID)
        _ = try await generator.consume(frame(sessionID: UUID(), sequence: 1))

        let outcome = try await generator.finalize(turnID: turnID)
        guard case .repeatPrompt(let diagnostic) = outcome else {
            return XCTFail("empty final must request a repeat")
        }
        XCTAssertEqual(diagnostic.stage, "asr")
        XCTAssertEqual(diagnostic.reason, "empty_final")
        XCTAssertFalse(outcome.createsLLMTurn)
    }

    private func assertTranscript(
        partial: String,
        final: String,
        expectedLanguage: VoiceTranscriptLanguage
    ) async throws {
        let inference = ScriptedQwen3Inference(partials: [partial], finals: [final])
        let adapter = makeAdapter(inference: inference)
        let generator = LocalASRTranscriptGenerator(sessionID: UUID(), adapter: adapter)
        let turnID = UUID()
        try await generator.startTurn(turnID)
        let partials = try await generator.consume(frame(sessionID: UUID(), sequence: 1))
        let outcome = try await generator.finalize(turnID: turnID)
        guard case .transcript(let finalEvents) = outcome else {
            return XCTFail("expected transcript")
        }
        let events = partials + finalEvents
        XCTAssertEqual(events.map(\.isFinal), [false, true])
        XCTAssertEqual(events.map(\.language), [expectedLanguage, expectedLanguage])
        XCTAssertLessThan(events[0].eventSequence, events[1].eventSequence)
        XCTAssertFalse(events[1].text.contains("\n"))
    }

    private func makeAdapter(inference: any CoreMLQwen3Inference) -> CoreMLQwen3ASRAdapter {
        CoreMLQwen3ASRAdapter(
            configuration: VoiceASRConfigurationFixture.configuration,
            inference: inference
        )
    }

    private func frame(sessionID: UUID, sequence: UInt64) -> VoiceAudioFrame {
        VoiceAudioFrame(
            sessionID: sessionID,
            sequence: sequence,
            capturedAtMonotonicNs: sequence * 20_000_000,
            sampleRateHz: 16_000,
            channels: 1,
            pcmS16LE: Data([0, 0, 1, 0])
        )
    }
}

private enum VoiceASRConfigurationFixture {
    static let configuration = VoiceASRConfiguration(
        backend: .coreMLQwen3Local,
        modelID: VoiceLocalAssetConfiguration.coreMLASRModelID,
        modelURL: URL(fileURLWithPath: "/tmp/fixture-Qwen3ASR.mlmodelc"),
        sampleRateHz: 16_000,
        vocabularyVersion: "fixture"
    )
}
#endif
