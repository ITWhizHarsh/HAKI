// AudioFrameRingTests.swift
// Focused coverage for realtime-local-voice-agent task 4.2.
// Validates: Requirements 2.3, 2.5–2.6, 4.2

#if canImport(XCTest)
import Darwin
import Foundation
import XCTest
@testable import HAKIAudio

final class AudioFrameRingTests: XCTestCase {
    func testSameUIDDescriptorUsesOwnerOnlyModeAndRejectsWrongUser() throws {
        let ring = try AudioFrameRing(sessionID: UUID(), capacity: 2, slotByteCapacity: 16)
        defer { ring.close() }

        XCTAssertEqual(ring.descriptor.fileSystemMode, 0o600)
        XCTAssertEqual(ring.descriptor.ownerUID, UInt32(getuid()))
        XCTAssertFalse(ring.descriptor.sharedMemoryName.dropFirst().contains("/"))
        XCTAssertThrowsError(try AudioFrameRing.openSameUser(
            ring.descriptor,
            sessionCapability: ring.descriptor.sessionCapability,
            currentUID: ring.descriptor.ownerUID &+ 1
        )) { error in
            XCTAssertEqual(error as? AudioFrameRingError, .accessDenied)
        }

        let sameUser = try AudioFrameRing.openSameUser(
            ring.descriptor,
            sessionCapability: ring.descriptor.sessionCapability
        )
        sameUser.close()
    }

    func testRingDropsOnlyOldestNonFinalFrameAndPreservesDescriptorOrder() throws {
        let sessionID = UUID()
        let ring = try AudioFrameRing(sessionID: sessionID, capacity: 2, slotByteCapacity: 16)
        defer { ring.close() }

        _ = try ring.enqueue(frame(sessionID, sequence: 1), isFinal: false)
        _ = try ring.enqueue(frame(sessionID, sequence: 2), isFinal: true)
        let result = try ring.enqueue(frame(sessionID, sequence: 3), isFinal: true)
        guard case .accepted(let descriptor, let droppedSequence) = result else {
            return XCTFail("expected non-final replacement")
        }
        XCTAssertEqual(droppedSequence, 1)
        XCTAssertEqual(descriptor.sequence, 3)

        let first = try XCTUnwrap(ring.dequeue())
        let second = try XCTUnwrap(ring.dequeue())
        XCTAssertEqual([first.descriptor.sequence, second.descriptor.sequence], [2, 3])
        XCTAssertEqual(first.pcmS16LE, frame(sessionID, sequence: 2).pcmS16LE)
    }

    func testRingRefusesOverwriteWhenAllResidentFramesAreFinalAndZeroizesOnClose() throws {
        let sessionID = UUID()
        let ring = try AudioFrameRing(sessionID: sessionID, capacity: 1, slotByteCapacity: 16)
        let descriptor = ring.descriptor
        _ = try ring.enqueue(frame(sessionID, sequence: 1), isFinal: true)
        XCTAssertEqual(try ring.enqueue(frame(sessionID, sequence: 2), isFinal: false), .rejectedAllFramesFinal)

        // Descriptors and metadata have no raw microphone byte/text field.
        let encoded = try JSONEncoder().encode(descriptor)
        let text = String(decoding: encoded, as: UTF8.self).lowercased()
        XCTAssertFalse(text.contains("pcm"))
        XCTAssertFalse(text.contains("sample"))

        ring.close()
        XCTAssertTrue(ring.lastCloseZeroizedMemory)
        XCTAssertThrowsError(try AudioFrameRing.openSameUser(
            descriptor,
            sessionCapability: descriptor.sessionCapability
        ))
    }

    private func frame(_ sessionID: UUID, sequence: UInt64) -> VoiceAudioFrame {
        VoiceAudioFrame(
            sessionID: sessionID,
            sequence: sequence,
            capturedAtMonotonicNs: sequence * 100,
            sampleRateHz: 16_000,
            channels: 1,
            pcmS16LE: Data([UInt8(sequence), 0, UInt8(sequence), 0])
        )
    }
}
#endif
