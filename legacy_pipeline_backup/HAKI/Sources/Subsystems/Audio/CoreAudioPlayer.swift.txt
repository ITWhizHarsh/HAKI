// CoreAudioPlayer.swift
// HAKI — Audio Subsystem
//
// Plays TTS audio chunks received from the Python Core via IPC.
// Uses AVAudioEngine with AVAudioPlayerNode for streaming playback.

import AVFoundation
import Foundation
import HAKIIPC

/// Plays TTS audio chunks from the Core in real-time.
public final class CoreAudioPlayer: @unchecked Sendable {
    
    // MARK: - AVAudio graph
    
    private let engine = AVAudioEngine()
    private let playerNode = AVAudioPlayerNode()
    private let playerLock = NSLock()
    private var engineStarted = false
    private var currentSampleRate: Double = 0
    
    // MARK: - Init / deinit
    
    public init() {
        engine.attach(playerNode)
        // Connect at the default hardware sample rate; we'll reconnect if the
        // Core sends a different rate. Starting the engine here ensures we own
        // the hardware output before TTSService or anything else can grab it.
        if let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 22_050,
            channels: 1,
            interleaved: false
        ) {
            engine.connect(playerNode, to: engine.mainMixerNode, format: format)
        }
        engine.prepare()
        do {
            try engine.start()
            engineStarted = true
            print("[CoreAudioPlayer] ✅ Audio engine started at init")
        } catch {
            print("[CoreAudioPlayer] ⚠️ Failed to start audio engine at init: \(error)")
        }
    }
    
    deinit {
        stop()
        if engineStarted {
            engine.stop()
        }
    }
    
    // MARK: - Public API
    
    /// Play a TTS audio chunk received from the Core.
    public func playChunk(_ chunk: HAKITTSAudioChunk) {
        print("[CoreAudioPlayer] 📥 Received chunk: \(chunk.samples.count) bytes, rate: \(chunk.sampleRate) Hz, isLast: \(chunk.isLast)")
        
        guard !chunk.samples.isEmpty else {
            print("[CoreAudioPlayer] ⚠️ Empty audio chunk, skipping")
            return
        }
        
        let sampleRate = Double(chunk.sampleRate)
        scheduleAudio(chunk.samples, sampleRate: sampleRate)
        
        if chunk.isLast {
            print("[CoreAudioPlayer] ✅ Final TTS chunk received")
        }
    }
    
    /// Stop playback immediately.
    public func stop() {
        playerLock.withLock {
            if playerNode.isPlaying {
                playerNode.stop()
            }
        }
    }
    
    // MARK: - Private: audio scheduling
    
    /// Schedule raw PCM Int16 LE data on the AVAudioPlayerNode.
    private func scheduleAudio(_ data: Data, sampleRate: Double) {
        let sampleCount = data.count / MemoryLayout<Int16>.size
        guard sampleCount > 0 else { return }
        
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            print("[CoreAudioPlayer] Failed to create AVAudioFormat")
            return
        }
        
        // Reconnect if sample rate changed between chunks.
        playerLock.withLock {
            if currentSampleRate != sampleRate {
                let wasPlaying = playerNode.isPlaying
                if wasPlaying { playerNode.pause() }
                engine.disconnectNodeOutput(playerNode)
                engine.connect(playerNode, to: engine.mainMixerNode, format: format)
                currentSampleRate = sampleRate
                if wasPlaying { playerNode.play() }
            }
        }
        
        guard let pcmBuffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(sampleCount)
        ) else {
            print("[CoreAudioPlayer] Failed to allocate PCM buffer")
            return
        }
        
        pcmBuffer.frameLength = AVAudioFrameCount(sampleCount)
        
        // Convert Int16 → Float32 safely
        data.withUnsafeBytes { (rawPtr: UnsafeRawBufferPointer) in
            guard let int16Ptr = rawPtr.bindMemory(to: Int16.self).baseAddress,
                  let floatPtr = pcmBuffer.floatChannelData?[0] else { return }
            for i in 0..<sampleCount {
                floatPtr[i] = Float(int16Ptr[i]) / Float(Int16.max)
            }
        }
        
        playerLock.withLock {
            playerNode.scheduleBuffer(pcmBuffer, completionHandler: nil)
            // Start playback. engineStarted is true from init; even if init
            // failed, attempting play() is harmless (it no-ops if engine is off).
            if !playerNode.isPlaying {
                playerNode.play()
                print("[CoreAudioPlayer] 🔊 Started TTS playback (\(sampleCount) samples @ \(Int(sampleRate)) Hz)")
            }
        }
    }
}
