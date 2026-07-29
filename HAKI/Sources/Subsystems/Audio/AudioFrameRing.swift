// AudioFrameRing.swift
// HAKI — same-UID ephemeral shared-memory microphone frame transport.
//
// This ring is intentionally not a socket or a diagnostic facility. Its only
// payload is bounded 16 kHz PCM for the local Pipecat/Silero reader. Descriptors
// never contain sample bytes and are suitable only for same-UID local mapping.

import Darwin
import Foundation

// Darwin imports variadic `shm_open` as unavailable to Swift. Keep its narrow
// fixed-arity POSIX declaration private to this local-memory implementation.
@_silgen_name("shm_open")
private func hakiRawShmOpen(_ name: UnsafePointer<CChar>, _ flags: Int32, _ mode: mode_t) -> Int32

private func hakiShmOpen(_ name: String, flags: Int32, mode: mode_t) -> Int32 {
    name.withCString { hakiRawShmOpen($0, flags, mode) }
}

public enum AudioFrameRingError: Error, Sendable, Equatable {
    case invalidCapacity
    case invalidSlotCapacity
    case frameTooLarge
    case closed
    case accessDenied
    case invalidDescriptor
    case unavailable
    case allFramesFinal
}

/// Capability-bearing metadata for a locally inherited shared-memory mapping.
/// It must not be serialized on the transcript/control socket.
public struct AudioFrameRingDescriptor: Codable, Sendable, Equatable {
    public let sharedMemoryName: String
    public let sessionID: UUID
    public let sessionCapability: String
    public let ownerUID: UInt32
    public let fileSystemMode: UInt16
    public let capacity: Int
    public let slotByteCapacity: Int

    public init(
        sharedMemoryName: String,
        sessionID: UUID,
        sessionCapability: String,
        ownerUID: UInt32,
        fileSystemMode: UInt16,
        capacity: Int,
        slotByteCapacity: Int
    ) {
        self.sharedMemoryName = sharedMemoryName
        self.sessionID = sessionID
        self.sessionCapability = sessionCapability
        self.ownerUID = ownerUID
        self.fileSystemMode = fileSystemMode
        self.capacity = capacity
        self.slotByteCapacity = slotByteCapacity
    }
}

/// Metadata referencing PCM inside `AudioFrameRing`. It deliberately contains
/// no microphone sample field, encoded payload, or diagnostic content.
public struct AudioFrameRingFrameDescriptor: Codable, Sendable, Equatable {
    public let sessionID: UUID
    public let slotIndex: Int
    public let sequence: UInt64
    public let capturedAtMonotonicNs: UInt64
    public let sampleRateHz: Int
    public let channels: UInt8
    public let byteLength: Int
    public let isFinal: Bool
}

public struct AudioFrameRingFrame: Sendable, Equatable {
    public let descriptor: AudioFrameRingFrameDescriptor
    /// PCM exists only while read from the local shared-memory slot.
    public let pcmS16LE: Data
}

public enum AudioFrameRingEnqueueResult: Sendable, Equatable {
    case accepted(descriptor: AudioFrameRingFrameDescriptor, droppedSequence: UInt64?)
    /// The ring is full of final frames. The caller must retain control/final
    /// state instead of overwriting a final frame or reordering descriptors.
    case rejectedAllFramesFinal
}

/// Metadata-only accounting. This is safe for privacy-preserving diagnostics.
public struct AudioFrameRingDiagnostic: Sendable, Equatable {
    public enum Kind: String, Sendable, Equatable {
        case oldestNonFinalDropped
        case fullOfFinalFrames
        case zeroizedAndUnlinked
    }

    public let kind: Kind
    public let sequence: UInt64?

    public init(kind: Kind, sequence: UInt64? = nil) {
        self.kind = kind
        self.sequence = sequence
    }
}

/// A bounded, session-random POSIX shared-memory ring. The Swift writer uses a
/// mutex to preserve descriptor order; a same-UID consumer receives descriptor
/// offsets and maps the object temporarily. It is never a UDS payload.
public final class AudioFrameRing: @unchecked Sendable {
    public static let requiredMode: UInt16 = 0o600
    public static let normalizedSampleRateHz = 16_000

    private static let headerBytes = 64
    private static let slotHeaderBytes = 32
    private static let magic = Array("HAKIRING".utf8)
    private static let version: UInt32 = 1

    public let descriptor: AudioFrameRingDescriptor
    public private(set) var lastCloseZeroizedMemory = false

    private let lock = NSLock()
    private let diagnosticsContinuation: AsyncStream<AudioFrameRingDiagnostic>.Continuation
    public let diagnostics: AsyncStream<AudioFrameRingDiagnostic>

    private var fileDescriptor: Int32
    private var mapping: UnsafeMutableRawPointer?
    private let mappingLength: Int
    private let slotStride: Int
    private var queuedSlots: [Int] = []
    private var occupiedSlots = Set<Int>()
    private var isClosed = false

    /// Creates a new random session mapping. The name and capability each carry
    /// 128 bits of randomness and the object is owner read/write only.
    public init(
        sessionID: UUID,
        capacity: Int = 128,
        slotByteCapacity: Int = 4_096,
        ownerUID: UInt32 = UInt32(getuid())
    ) throws {
        guard capacity > 0 else { throw AudioFrameRingError.invalidCapacity }
        guard slotByteCapacity > 0 else { throw AudioFrameRingError.invalidSlotCapacity }

        let name = "/haki-voice-ring-\(UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: ""))"
        let capability = UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
        let stride = Self.aligned(Self.slotHeaderBytes + slotByteCapacity)
        let length = Self.headerBytes + (capacity * stride)

        let fd = hakiShmOpen(name, flags: O_CREAT | O_EXCL | O_RDWR, mode: mode_t(Self.requiredMode))
        guard fd >= 0 else { throw AudioFrameRingError.unavailable }
        guard fchmod(fd, mode_t(Self.requiredMode)) == 0 else {
            Darwin.close(fd)
            shm_unlink(name)
            throw AudioFrameRingError.unavailable
        }
        guard ftruncate(fd, off_t(length)) == 0 else {
            Darwin.close(fd)
            shm_unlink(name)
            throw AudioFrameRingError.unavailable
        }
        guard let map = Self.map(fd: fd, length: length) else {
            Darwin.close(fd)
            shm_unlink(name)
            throw AudioFrameRingError.unavailable
        }

        self.descriptor = AudioFrameRingDescriptor(
            sharedMemoryName: name,
            sessionID: sessionID,
            sessionCapability: capability,
            ownerUID: ownerUID,
            fileSystemMode: Self.requiredMode,
            capacity: capacity,
            slotByteCapacity: slotByteCapacity
        )
        self.fileDescriptor = fd
        self.mapping = map
        self.mappingLength = length
        self.slotStride = stride

        let stream = AsyncStream<AudioFrameRingDiagnostic>.makeStream(bufferingPolicy: .bufferingNewest(32))
        diagnostics = stream.stream
        diagnosticsContinuation = stream.continuation
        initializeHeader()
    }

    private init(
        descriptor: AudioFrameRingDescriptor,
        fileDescriptor: Int32,
        mapping: UnsafeMutableRawPointer,
        mappingLength: Int,
        slotStride: Int
    ) {
        self.descriptor = descriptor
        self.fileDescriptor = fileDescriptor
        self.mapping = mapping
        self.mappingLength = mappingLength
        self.slotStride = slotStride
        let stream = AsyncStream<AudioFrameRingDiagnostic>.makeStream(bufferingPolicy: .bufferingNewest(32))
        diagnostics = stream.stream
        diagnosticsContinuation = stream.continuation
    }

    deinit {
        close()
    }

    /// Opens a mapping only for its owning UID and expected session capability.
    /// Callers should pass this descriptor through inherited local session
    /// configuration, never through the transcript/control protocol.
    public static func openSameUser(
        _ descriptor: AudioFrameRingDescriptor,
        sessionCapability: String,
        currentUID: UInt32 = UInt32(getuid())
    ) throws -> AudioFrameRing {
        guard descriptor.ownerUID == currentUID,
              descriptor.fileSystemMode == requiredMode,
              descriptor.sessionCapability == sessionCapability,
              descriptor.capacity > 0,
              descriptor.slotByteCapacity > 0,
              descriptor.sharedMemoryName.hasPrefix("/haki-voice-ring-") else {
            throw AudioFrameRingError.accessDenied
        }

        let fd = hakiShmOpen(descriptor.sharedMemoryName, flags: O_RDWR, mode: 0)
        guard fd >= 0 else { throw AudioFrameRingError.unavailable }
        var attributes = stat()
        guard fstat(fd, &attributes) == 0,
              UInt32(attributes.st_uid) == descriptor.ownerUID,
              UInt16(attributes.st_mode & 0o777) == requiredMode else {
            Darwin.close(fd)
            throw AudioFrameRingError.accessDenied
        }

        let expectedStride = aligned(slotHeaderBytes + descriptor.slotByteCapacity)
        let expectedLength = headerBytes + descriptor.capacity * expectedStride
        guard Int(attributes.st_size) == expectedLength,
              let map = map(fd: fd, length: expectedLength) else {
            Darwin.close(fd)
            throw AudioFrameRingError.invalidDescriptor
        }
        guard headerIsValid(map, descriptor: descriptor) else {
            munmap(map, expectedLength)
            Darwin.close(fd)
            throw AudioFrameRingError.invalidDescriptor
        }
        return AudioFrameRing(
            descriptor: descriptor,
            fileDescriptor: fd,
            mapping: map,
            mappingLength: expectedLength,
            slotStride: expectedStride
        )
    }

    /// Stores one normalized microphone frame. At capacity, only the oldest
    /// queued non-final frame may be removed; remaining descriptors stay in
    /// their original order and sequence gaps are explicit.
    public func enqueue(_ frame: VoiceAudioFrame, isFinal: Bool = false) throws -> AudioFrameRingEnqueueResult {
        try withLock {
            guard !isClosed, let mapping else { throw AudioFrameRingError.closed }
            guard frame.sampleRateHz == Self.normalizedSampleRateHz, frame.channels == 1 else {
                throw AudioFrameRingError.invalidDescriptor
            }
            guard frame.pcmS16LE.count <= descriptor.slotByteCapacity else {
                throw AudioFrameRingError.frameTooLarge
            }

            var droppedSequence: UInt64?
            var slot: Int
            if occupiedSlots.count == descriptor.capacity {
                guard let dropIndex = queuedSlots.firstIndex(where: { !readSlotDescriptor(at: $0, mapping: mapping).isFinal }) else {
                    diagnosticsContinuation.yield(AudioFrameRingDiagnostic(kind: .fullOfFinalFrames))
                    return .rejectedAllFramesFinal
                }
                slot = queuedSlots.remove(at: dropIndex)
                let dropped = readSlotDescriptor(at: slot, mapping: mapping)
                droppedSequence = dropped.sequence
                occupiedSlots.remove(slot)
                zeroSlot(at: slot, mapping: mapping)
                diagnosticsContinuation.yield(AudioFrameRingDiagnostic(kind: .oldestNonFinalDropped, sequence: dropped.sequence))
            } else {
                guard let free = (0..<descriptor.capacity).first(where: { !occupiedSlots.contains($0) }) else {
                    throw AudioFrameRingError.unavailable
                }
                slot = free
            }

            write(frame, isFinal: isFinal, at: slot, mapping: mapping)
            occupiedSlots.insert(slot)
            queuedSlots.append(slot)
            let frameDescriptor = readSlotDescriptor(at: slot, mapping: mapping)
            return .accepted(descriptor: frameDescriptor, droppedSequence: droppedSequence)
        }
    }

    /// Returns and releases the oldest descriptor in FIFO order. A local reader
    /// must copy its PCM and release promptly; the cleared slot is reusable.
    public func dequeue() throws -> AudioFrameRingFrame? {
        try withLock {
            guard !isClosed, let mapping else { throw AudioFrameRingError.closed }
            guard let slot = queuedSlots.first else { return nil }
            queuedSlots.removeFirst()
            let frameDescriptor = readSlotDescriptor(at: slot, mapping: mapping)
            let payloadOffset = slotOffset(slot) + Self.slotHeaderBytes
            let pcm = Data(bytes: mapping.advanced(by: payloadOffset), count: frameDescriptor.byteLength)
            zeroSlot(at: slot, mapping: mapping)
            occupiedSlots.remove(slot)
            return AudioFrameRingFrame(descriptor: frameDescriptor, pcmS16LE: pcm)
        }
    }

    /// Securely clears every mapped byte before detaching and unlinks the POSIX
    /// object. Existing same-UID mappings observe zeroed payload bytes.
    public func close() {
        lock.lock()
        defer { lock.unlock() }
        guard !isClosed else { return }
        isClosed = true
        if let mapping {
            memset(mapping, 0, mappingLength)
            lastCloseZeroizedMemory = UnsafeRawBufferPointer(start: mapping, count: mappingLength).allSatisfy { $0 == 0 }
            _ = msync(mapping, mappingLength, MS_SYNC)
            _ = munmap(mapping, mappingLength)
            self.mapping = nil
        }
        if fileDescriptor >= 0 {
            _ = Darwin.close(fileDescriptor)
            fileDescriptor = -1
        }
        _ = shm_unlink(descriptor.sharedMemoryName)
        queuedSlots.removeAll(keepingCapacity: false)
        occupiedSlots.removeAll(keepingCapacity: false)
        diagnosticsContinuation.yield(AudioFrameRingDiagnostic(kind: .zeroizedAndUnlinked))
        diagnosticsContinuation.finish()
    }

    private func initializeHeader() {
        guard let mapping else { return }
        memset(mapping, 0, mappingLength)
        Self.magic.withUnsafeBytes { source in
            memcpy(mapping, source.baseAddress!, Self.magic.count)
        }
        writeUInt32(Self.version, at: 8, mapping: mapping)
        writeUInt32(descriptor.ownerUID, at: 12, mapping: mapping)
        writeUInt32(UInt32(descriptor.capacity), at: 16, mapping: mapping)
        writeUInt32(UInt32(descriptor.slotByteCapacity), at: 20, mapping: mapping)
    }

    private func write(_ frame: VoiceAudioFrame, isFinal: Bool, at slot: Int, mapping: UnsafeMutableRawPointer) {
        zeroSlot(at: slot, mapping: mapping)
        let offset = slotOffset(slot)
        writeUInt64(frame.sequence, at: offset, mapping: mapping)
        writeUInt64(frame.capturedAtMonotonicNs, at: offset + 8, mapping: mapping)
        writeUInt32(UInt32(frame.pcmS16LE.count), at: offset + 16, mapping: mapping)
        writeUInt32(UInt32(frame.sampleRateHz), at: offset + 20, mapping: mapping)
        mapping.storeBytes(of: frame.channels, toByteOffset: offset + 24, as: UInt8.self)
        mapping.storeBytes(of: isFinal ? UInt8(1) : UInt8(0), toByteOffset: offset + 25, as: UInt8.self)
        frame.pcmS16LE.withUnsafeBytes { source in
            guard let base = source.baseAddress else { return }
            memcpy(mapping.advanced(by: offset + Self.slotHeaderBytes), base, frame.pcmS16LE.count)
        }
    }

    private func readSlotDescriptor(at slot: Int, mapping: UnsafeMutableRawPointer) -> AudioFrameRingFrameDescriptor {
        let offset = slotOffset(slot)
        return AudioFrameRingFrameDescriptor(
            sessionID: descriptor.sessionID,
            slotIndex: slot,
            sequence: readUInt64(at: offset, mapping: mapping),
            capturedAtMonotonicNs: readUInt64(at: offset + 8, mapping: mapping),
            sampleRateHz: Int(Self.readUInt32(at: offset + 20, mapping: mapping)),
            channels: mapping.load(fromByteOffset: offset + 24, as: UInt8.self),
            byteLength: Int(Self.readUInt32(at: offset + 16, mapping: mapping)),
            isFinal: mapping.load(fromByteOffset: offset + 25, as: UInt8.self) == 1
        )
    }

    private func zeroSlot(at slot: Int, mapping: UnsafeMutableRawPointer) {
        memset(mapping.advanced(by: slotOffset(slot)), 0, slotStride)
    }

    private func slotOffset(_ slot: Int) -> Int {
        Self.headerBytes + slot * slotStride
    }

    private func withLock<T>(_ operation: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try operation()
    }

    private static func map(fd: Int32, length: Int) -> UnsafeMutableRawPointer? {
        let value = mmap(nil, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        return value == MAP_FAILED ? nil : value
    }

    private static func headerIsValid(_ mapping: UnsafeMutableRawPointer, descriptor: AudioFrameRingDescriptor) -> Bool {
        let magic = Array(UnsafeRawBufferPointer(start: mapping, count: Self.magic.count))
        return magic == Self.magic
            && readUInt32(at: 8, mapping: mapping) == Self.version
            && readUInt32(at: 12, mapping: mapping) == descriptor.ownerUID
            && readUInt32(at: 16, mapping: mapping) == UInt32(descriptor.capacity)
            && readUInt32(at: 20, mapping: mapping) == UInt32(descriptor.slotByteCapacity)
    }

    private static func aligned(_ value: Int) -> Int {
        (value + 7) & ~7
    }

    private func writeUInt32(_ value: UInt32, at offset: Int, mapping: UnsafeMutableRawPointer) {
        Self.writeUInt32(value, at: offset, mapping: mapping)
    }

    private static func writeUInt32(_ value: UInt32, at offset: Int, mapping: UnsafeMutableRawPointer) {
        mapping.storeBytes(of: value.littleEndian, toByteOffset: offset, as: UInt32.self)
    }

    private func writeUInt64(_ value: UInt64, at offset: Int, mapping: UnsafeMutableRawPointer) {
        mapping.storeBytes(of: value.littleEndian, toByteOffset: offset, as: UInt64.self)
    }

    private static func readUInt32(at offset: Int, mapping: UnsafeMutableRawPointer) -> UInt32 {
        UInt32(littleEndian: mapping.load(fromByteOffset: offset, as: UInt32.self))
    }

    private func readUInt64(at offset: Int, mapping: UnsafeMutableRawPointer) -> UInt64 {
        UInt64(littleEndian: mapping.load(fromByteOffset: offset, as: UInt64.self))
    }
}
