// PCMPlaybackRenderer.swift
// HAKI — same-engine native PCM renderer for local XTTS output.
//
// The renderer owns no AVAudioEngine. It attaches one player node to the
// VoiceAudioController engine so VoiceProcessingIO capture remains active while
// TTS plays. It reports accepted, terminal, and stop-acknowledgement events
// without ever handling microphone payloads.

import AVFoundation
import Foundation
import HAKIIPC

public enum PCMPlaybackRendererError: Error, Sendable, Equatable {
    case staleSession
    case invalidPCM
    case sentenceAlreadyTerminal
    case generationMismatch
    case missingSharedEngine
    case playbackUnavailable
}

public struct PCMPlaybackEnqueueAcknowledgement: Sendable, Equatable {
    public let sessionID: UUID
    public let turnID: UUID
    public let sentenceID: UUID
    public let sequence: UInt64
}

public enum PCMPlaybackRendererEvent: Sendable, Equatable {
    case accepted(PCMPlaybackEnqueueAcknowledgement)
    case terminal(VoicePlaybackTerminal)
    case stopAcknowledged(VoiceStopPlayback)
}

/// Narrow lifecycle seam shared by production VoiceAudioController and focused
/// tests. It changes controller state but never stops microphone capture.
public protocol PCMPlaybackLifecycle: AnyObject, Sendable {
    func playbackDidStart() throws
    func playbackDidStop()
}

extension VoiceAudioController: PCMPlaybackLifecycle {}

/// Player-node seam. The production implementation attaches to the supplied
/// controller engine; tests complete queued buffers deterministically.
public protocol PCMPlaybackNode: AnyObject, Sendable {
    func attach(to engine: AVAudioEngine?) throws
    func schedule(
        pcmS16LE: Data,
        sampleRateHz: Int,
        channels: UInt8,
        completion: @escaping @Sendable (Result<Void, Error>) -> Void
    ) throws
    func play() throws
    func stop()
}

public final class SystemPCMPlaybackNode: PCMPlaybackNode, @unchecked Sendable {
    private let player = AVAudioPlayerNode()
    private var attachedEngine: AVAudioEngine?

    public init() {}

    public func attach(to engine: AVAudioEngine?) throws {
        guard let engine else { throw PCMPlaybackRendererError.missingSharedEngine }
        if attachedEngine === engine { return }
        guard attachedEngine == nil else { throw PCMPlaybackRendererError.playbackUnavailable }
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: nil)
        attachedEngine = engine
    }

    public func schedule(
        pcmS16LE: Data,
        sampleRateHz: Int,
        channels: UInt8,
        completion: @escaping @Sendable (Result<Void, Error>) -> Void
    ) throws {
        guard sampleRateHz > 0, channels > 0, pcmS16LE.count.isMultiple(of: 2) else {
            throw PCMPlaybackRendererError.invalidPCM
        }
        let sampleCount = pcmS16LE.count / MemoryLayout<Int16>.size
        guard sampleCount > 0, sampleCount % Int(channels) == 0,
              let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(sampleRateHz),
                channels: AVAudioChannelCount(channels),
                interleaved: false
              ),
              let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: AVAudioFrameCount(sampleCount / Int(channels))
              ),
              let output = buffer.floatChannelData else {
            throw PCMPlaybackRendererError.invalidPCM
        }

        let frames = sampleCount / Int(channels)
        buffer.frameLength = AVAudioFrameCount(frames)
        pcmS16LE.withUnsafeBytes { source in
            let bytes = source.bindMemory(to: UInt8.self)
            for frame in 0..<frames {
                for channel in 0..<Int(channels) {
                    let index = (frame * Int(channels) + channel) * 2
                    let raw = UInt16(bytes[index]) | (UInt16(bytes[index + 1]) << 8)
                    output[channel][frame] = Float(Int16(bitPattern: raw)) / Float(Int16.max)
                }
            }
        }

        player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { _ in
            completion(.success(()))
        }
    }

    public func play() throws {
        guard attachedEngine != nil else { throw PCMPlaybackRendererError.missingSharedEngine }
        if !player.isPlaying { player.play() }
    }

    public func stop() {
        player.stop()
    }
}

/// Serial native playback queue. A sentence receives exactly one terminal
/// outcome: confirmation only after its final buffer played, cancellation after
/// stop, or failure after an attach/schedule/play error.
public final class PCMPlaybackRenderer: @unchecked Sendable {
    public let events: AsyncStream<PCMPlaybackRendererEvent>

    private let sessionID: UUID
    private let engineProvider: @Sendable () -> AVAudioEngine?
    private weak var lifecycle: (any PCMPlaybackLifecycle)?
    private let player: any PCMPlaybackNode
    private let queue = DispatchQueue(label: "com.haki.voice.pcm-renderer", qos: .userInitiated)
    private let queueKey = DispatchSpecificKey<UInt8>()
    private let eventContinuation: AsyncStream<PCMPlaybackRendererEvent>.Continuation
    private let terminalSink: @Sendable (VoicePlaybackTerminal) -> Void

    private var isAttached = false
    private var sentenceOrder: [SentenceKey] = []
    private var sentences: [SentenceKey: SentenceState] = [:]
    private var activeKey: SentenceKey?
    private var inFlightKey: SentenceKey?
    private var stoppedGenerations = Set<GenerationKey>()

    public convenience init(
        controller: VoiceAudioController,
        terminalSink: @escaping @Sendable (VoicePlaybackTerminal) -> Void = { _ in }
    ) {
        self.init(
            sessionID: controller.sessionID,
            engineProvider: { controller.audioEngine },
            lifecycle: controller,
            player: SystemPCMPlaybackNode(),
            terminalSink: terminalSink
        )
    }

    public init(
        sessionID: UUID,
        engineProvider: @escaping @Sendable () -> AVAudioEngine?,
        lifecycle: any PCMPlaybackLifecycle,
        player: any PCMPlaybackNode,
        terminalSink: @escaping @Sendable (VoicePlaybackTerminal) -> Void = { _ in }
    ) {
        self.sessionID = sessionID
        self.engineProvider = engineProvider
        self.lifecycle = lifecycle
        self.player = player
        self.terminalSink = terminalSink
        let stream = AsyncStream<PCMPlaybackRendererEvent>.makeStream(bufferingPolicy: .bufferingNewest(128))
        events = stream.stream
        eventContinuation = stream.continuation
        queue.setSpecific(key: queueKey, value: 1)
    }

    deinit {
        onQueue {
            player.stop()
            lifecycle?.playbackDidStop()
        }
        eventContinuation.finish()
    }

    /// Adds a TTS-only PCM chunk to the renderer. The acknowledgement event is
    /// published immediately after the chunk enters the bounded native queue;
    /// VoiceSocketClient then emits `PCM_ACCEPTED` to Python.
    @discardableResult
    public func enqueue(
        _ chunk: VoicePCMChunk,
        pcmS16LE: Data,
        generation: UInt64
    ) throws -> PCMPlaybackEnqueueAcknowledgement {
        try onQueue {
            guard chunk.sessionID == sessionID else { throw PCMPlaybackRendererError.staleSession }
            guard chunk.byteLength == pcmS16LE.count,
                  !pcmS16LE.isEmpty,
                  pcmS16LE.count.isMultiple(of: 2),
                  chunk.sampleRateHz > 0,
                  chunk.channels > 0,
                  (pcmS16LE.count / 2).isMultiple(of: Int(chunk.channels)) else {
                throw PCMPlaybackRendererError.invalidPCM
            }
            let key = SentenceKey(turnID: chunk.turnID, sentenceID: chunk.sentenceID)
            let generationKey = GenerationKey(turnID: chunk.turnID, generation: generation)
            if stoppedGenerations.contains(generationKey) {
                throw PCMPlaybackRendererError.sentenceAlreadyTerminal
            }

            if var state = sentences[key] {
                guard state.generation == generation, !state.terminalSent else {
                    throw state.generation == generation
                        ? PCMPlaybackRendererError.sentenceAlreadyTerminal
                        : PCMPlaybackRendererError.generationMismatch
                }
                state.buffers.append(PCMBuffer(chunk: chunk, data: pcmS16LE))
                sentences[key] = state
            } else {
                sentences[key] = SentenceState(generation: generation, buffers: [PCMBuffer(chunk: chunk, data: pcmS16LE)])
                sentenceOrder.append(key)
            }

            let acknowledgement = PCMPlaybackEnqueueAcknowledgement(
                sessionID: chunk.sessionID,
                turnID: chunk.turnID,
                sentenceID: chunk.sentenceID,
                sequence: chunk.sequence
            )
            eventContinuation.yield(.accepted(acknowledgement))
            pumpLocked()
            return acknowledgement
        }
    }

    /// Marks that no more PCM will arrive for a sentence. Confirmation waits
    /// for every previously scheduled sample to reach player-node completion.
    public func finishSentence(turnID: UUID, sentenceID: UUID, generation: UInt64) throws {
        try onQueue {
            let key = SentenceKey(turnID: turnID, sentenceID: sentenceID)
            guard var state = sentences[key], !state.terminalSent else {
                throw PCMPlaybackRendererError.sentenceAlreadyTerminal
            }
            guard state.generation == generation else { throw PCMPlaybackRendererError.generationMismatch }
            state.inputFinished = true
            sentences[key] = state
            pumpLocked()
        }
    }

    /// High-priority and idempotent cancellation. Repeated controls acknowledge
    /// immediately but execute node stop/cancellation only on the first call.
    public func stop(_ request: VoiceStopPlayback) {
        onQueue {
            guard request.sessionID == sessionID else { return }
            let generationKey = GenerationKey(turnID: request.turnID, generation: request.generation)
            if stoppedGenerations.insert(generationKey).inserted {
                let matching = sentenceOrder.filter { key in
                    guard let state = sentences[key] else { return false }
                    return key.turnID == request.turnID && state.generation == request.generation
                }
                let activeMatches = activeKey.map { matching.contains($0) } ?? false
                if activeMatches { player.stop() }
                for key in matching {
                    emitTerminalLocked(key: key, kind: .cancelled, errorClass: nil)
                }
                if activeMatches {
                    activeKey = nil
                    inFlightKey = nil
                }
                pumpLocked()
            }
            eventContinuation.yield(.stopAcknowledged(request))
        }
    }

    private func pumpLocked() {
        guard inFlightKey == nil else { return }
        while let next = sentenceOrder.first {
            guard var state = sentences[next] else {
                sentenceOrder.removeFirst()
                continue
            }
            activeKey = next
            if state.buffers.isEmpty {
                if state.inputFinished {
                    emitTerminalLocked(key: next, kind: .failed, errorClass: "empty_sentence_audio")
                    continue
                }
                return
            }

            let buffer = state.buffers.removeFirst()
            sentences[next] = state
            do {
                if !isAttached {
                    try player.attach(to: engineProvider())
                    isAttached = true
                }
                try lifecycle?.playbackDidStart()
                try player.schedule(
                    pcmS16LE: buffer.data,
                    sampleRateHz: buffer.chunk.sampleRateHz,
                    channels: buffer.chunk.channels
                ) { [weak self] result in
                    self?.queue.async { self?.bufferCompletedLocked(key: next, result: result) }
                }
                inFlightKey = next
                try player.play()
                return
            } catch {
                emitTerminalLocked(key: next, kind: .failed, errorClass: "native_pcm_schedule_failed")
            }
        }
        activeKey = nil
        lifecycle?.playbackDidStop()
    }

    private func bufferCompletedLocked(key: SentenceKey, result: Result<Void, Error>) {
        guard inFlightKey == key else { return }
        inFlightKey = nil
        guard let state = sentences[key], !state.terminalSent else {
            activeKey = nil
            pumpLocked()
            return
        }
        switch result {
        case .failure:
            emitTerminalLocked(key: key, kind: .failed, errorClass: "native_pcm_playback_failed")
        case .success:
            if state.buffers.isEmpty, state.inputFinished {
                emitTerminalLocked(key: key, kind: .confirmed, errorClass: nil)
            }
        }
        activeKey = nil
        pumpLocked()
    }

    private func emitTerminalLocked(
        key: SentenceKey,
        kind: VoicePlaybackTerminalKind,
        errorClass: String?
    ) {
        guard var state = sentences[key], !state.terminalSent else { return }
        state.terminalSent = true
        sentences[key] = state
        sentenceOrder.removeAll { $0 == key }
        let terminal = VoicePlaybackTerminal(
            kind: kind,
            sessionID: sessionID,
            turnID: key.turnID,
            sentenceID: key.sentenceID,
            errorClass: errorClass
        )
        eventContinuation.yield(.terminal(terminal))
        terminalSink(terminal)
    }

    private func onQueue<T>(_ operation: () throws -> T) rethrows -> T {
        if DispatchQueue.getSpecific(key: queueKey) != nil {
            return try operation()
        }
        return try queue.sync(execute: operation)
    }

    private struct SentenceKey: Hashable {
        let turnID: UUID
        let sentenceID: UUID
    }

    private struct GenerationKey: Hashable {
        let turnID: UUID
        let generation: UInt64
    }

    private struct PCMBuffer {
        let chunk: VoicePCMChunk
        let data: Data
    }

    private struct SentenceState {
        let generation: UInt64
        var buffers: [PCMBuffer]
        var inputFinished = false
        var terminalSent = false
    }
}
