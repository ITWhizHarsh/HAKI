// VoiceSocketProtocolFixtures.swift
// Exact, content-safe v1 fixtures mirrored from Core/core/ipc/voice_protocol.py.

import Foundation

@testable import HAKIIPC

enum VoiceSocketProtocolFixtures {
    static let sessionID = UUID(uuidString: "00000000-0000-4000-8000-000000000001")!
    static let turnID = UUID(uuidString: "00000000-0000-4000-8000-000000000002")!
    static let partialEventID = UUID(uuidString: "00000000-0000-4000-8000-000000000003")!
    static let finalEventID = UUID(uuidString: "00000000-0000-4000-8000-000000000004")!
    static let sentenceID = UUID(uuidString: "00000000-0000-4000-8000-000000000005")!
    static let confirmedEventID = UUID(uuidString: "00000000-0000-4000-8000-000000000006")!
    static let capability = "0123456789abcdef0123456789abcdef"

    static func transcriptPartial(sequence: UInt64 = 17) -> VoiceTranscriptEvent {
        VoiceTranscriptEvent(
            eventID: partialEventID,
            sessionID: sessionID,
            turnID: turnID,
            eventSequence: sequence,
            text: "Kal meeting",
            isFinal: false,
            language: .hinglish,
            captureStartedMonotonicNs: 123,
            captureEndedMonotonicNs: 456
        )
    }

    static func transcriptFinal(sequence: UInt64 = 18) -> VoiceTranscriptEvent {
        VoiceTranscriptEvent(
            eventID: finalEventID,
            sessionID: sessionID,
            turnID: turnID,
            eventSequence: sequence,
            text: "Kal meeting reschedule kar do",
            isFinal: true,
            language: .hinglish,
            captureStartedMonotonicNs: 123,
            captureEndedMonotonicNs: 789
        )
    }

    static func pcmMetadata(sequence: UInt64 = 0, byteLength: Int = 4) -> [String: Any] {
        [
            "version": 1,
            "type": "PCM_CHUNK",
            "session_id": sessionID.uuidString.lowercased(),
            "turn_id": turnID.uuidString.lowercased(),
            "sentence_id": sentenceID.uuidString.lowercased(),
            "sequence": sequence,
            "sample_rate_hz": 24_000,
            "channels": 1,
            "format": "s16le",
            "byte_length": byteLength,
        ]
    }

    static func stopPlayback(generation: UInt64 = 4) -> [String: Any] {
        [
            "version": 1,
            "type": "STOP_PLAYBACK",
            "session_id": sessionID.uuidString.lowercased(),
            "turn_id": turnID.uuidString.lowercased(),
            "generation": generation,
        ]
    }

    static func jsonLine(_ object: [String: Any]) throws -> Data {
        var data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        data.append(0x0A)
        return data
    }

    static func pcmFrame(_ metadata: [String: Any], payload: Data) throws -> Data {
        var frame = try jsonLine(metadata)
        let length = UInt32(payload.count).bigEndian
        withUnsafeBytes(of: length) { frame.append(contentsOf: $0) }
        frame.append(payload)
        return frame
    }
}
