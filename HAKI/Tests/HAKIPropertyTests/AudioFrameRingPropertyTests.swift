// AudioFrameRingPropertyTests.swift
// Feature: realtime-local-voice-agent
// Property: bounded descriptor order after non-final frame eviction
// Validates: Requirements 2.3, 2.5–2.6, 4.2

#if canImport(XCTest)
import Foundation
import SwiftCheck
import XCTest
@testable import HAKIAudio

final class AudioFrameRingPropertyTests: XCTestCase {
    func testDescriptorsRemainOrderedAfterAnyOldestNonFinalEviction() {
        property("Feature: realtime-local-voice-agent, ordered ring descriptors preserve survivors") <- forAll(
            Gen<Int>.choose((1, 10_000))
        ) { seed in
            guard let ring = try? AudioFrameRing(sessionID: UUID(), capacity: 2, slotByteCapacity: 16) else {
                return false
            }
            defer { ring.close() }
            let sessionID = ring.descriptor.sessionID
            guard
                (try? ring.enqueue(self.frame(sessionID, sequence: UInt64(seed)), isFinal: false)) != nil,
                (try? ring.enqueue(self.frame(sessionID, sequence: UInt64(seed + 1)), isFinal: true)) != nil,
                case .accepted(_, let droppedSequence)? = try? ring.enqueue(self.frame(sessionID, sequence: UInt64(seed + 2)), isFinal: true),
                let first = try? ring.dequeue(),
                let second = try? ring.dequeue()
            else {
                return false
            }
            return droppedSequence == UInt64(seed)
                && [first?.descriptor.sequence, second?.descriptor.sequence] == [UInt64(seed + 1), UInt64(seed + 2)]
        }
    }

    private func frame(_ sessionID: UUID, sequence: UInt64) -> VoiceAudioFrame {
        VoiceAudioFrame(
            sessionID: sessionID,
            sequence: sequence,
            capturedAtMonotonicNs: sequence,
            sampleRateHz: 16_000,
            channels: 1,
            pcmS16LE: Data([0, 0, 1, 0])
        )
    }
}
#endif
