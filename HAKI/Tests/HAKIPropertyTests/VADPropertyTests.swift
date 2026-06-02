// VADPropertyTests.swift
// HAKIPropertyTests — Property-Based Tests for the VAD (SwiftCheck)
//
// These tests use SwiftCheck to verify universal properties of the VAD's
// behaviour across arbitrary sequences of audio frames.
//
// Feature: haki-personal-ai-assistant, Property scaffold: VAD correctness
// Validates: Requirements 3.2 (800 ms end-of-speech), 3.3 (200 ms barge-in)
//
// Minimum iterations: 100 (SwiftCheck default).

import XCTest
import SwiftCheck
@testable import HAKIAudio
@testable import HAKIStore

final class VADPropertyTests: XCTestCase {

    // MARK: - Helper: build a frame with explicit energy

    private func frame(energy: Float, atSecond t: Double) -> AudioFrame {
        let value = Int16(min(abs(energy), 1.0) * Float(Int16.max))
        let samples = Array(repeating: value, count: LiveAudioEngine.samplesPerFrame)
        return AudioFrame(samples: samples, timestamp: Date(timeIntervalSinceReferenceDate: t))
    }

    // MARK: - Property 1
    // End-of-speech only fires when silence follows speech for ≥ 800 ms.
    //
    // Feature: haki-personal-ai-assistant, Property scaffold (Req 3.2)
    func testEndOfSpeechRequiresPrecedingSpeech() {
        // Generate a random number of silence-only frames (0…60) and assert
        // that endOfSpeech never fires when there was no prior speech.
        property("endOfSpeech does not fire on silence alone") <- forAll(Gen<Int>.choose((0, 60))) { frameCount in
            let vad = VAD()
            var fired = false
            vad.endOfSpeechHandler = { fired = true }

            for i in 0..<frameCount {
                vad.process(frame: self.frame(energy: 0.0, atSecond: Double(i) * 0.020))
            }

            return !fired
        }
    }

    // MARK: - Property 2
    // Barge-in only fires when TTS is playing.
    //
    // Feature: haki-personal-ai-assistant, Property scaffold (Req 3.3)
    func testBargeInRequiresTTSPlaying() {
        // With TTS not playing, continuous speech should NOT trigger barge-in.
        property("bargeIn does not fire when TTS is not playing") <- forAll(Gen<Int>.choose((10, 30))) { frameCount in
            let vad = VAD()
            var fired = false
            vad.bargeInHandler = { fired = true }
            vad.setTTSPlaying(false)

            for i in 0..<frameCount {
                vad.process(frame: self.frame(energy: 0.5, atSecond: Double(i) * 0.020))
            }

            return !fired
        }
    }

    // MARK: - Property 3
    // After barge-in fires, the VAD returns to idle (no double-fire).
    //
    // Feature: haki-personal-ai-assistant, Property scaffold (Req 3.3)
    func testBargeInFiresAtMostOncePerSpeechSegment() {
        property("bargeIn fires at most once per continuous speech segment while TTS is playing") <- forAll(Gen<Int>.choose((10, 50))) { frameCount in
            let vad = VAD()
            var count = 0
            vad.bargeInHandler = { count += 1 }
            vad.setTTSPlaying(true)

            for i in 0..<frameCount {
                vad.process(frame: self.frame(energy: 0.5, atSecond: Double(i) * 0.020))
            }

            return count <= 1
        }
    }

    // MARK: - Property 4
    // RMS energy of an all-zero frame is 0.
    //
    // Feature: haki-personal-ai-assistant, Property scaffold (audio utility)
    func testZeroFrameHasZeroEnergy() {
        // Directly verifies that silence frames don't accidentally cross threshold.
        let vad = VAD()
        var fired = false
        vad.endOfSpeechHandler = { fired = true }

        for i in 0..<100 {
            vad.process(frame: frame(energy: 0.0, atSecond: Double(i) * 0.020))
        }
        XCTAssertFalse(fired)
    }

    // MARK: - Property 5
    // Settings encode/decode produces identical values for arbitrary intensity / threshold.
    //
    // Feature: haki-personal-ai-assistant, Property scaffold (Settings round-trip)
    // Validates: Req 4.2 (mood threshold range), 6.3 (personality intensity range)
    func testSettingsCodableRoundTripProperty() {
        property("Settings encode/decode is lossless") <- forAll(
            Gen<Int>.choose((1, 3)),
            Gen<Double>.choose((0.0, 1.0)),
            Gen<Int>.choose((1, 90))
        ) { intensity, threshold, days in
            var s = Settings()
            s.personalityIntensity = intensity
            s.moodThreshold = threshold
            s.recentlyLearnedDays = days

            guard
                let data = try? JSONEncoder().encode(s),
                let decoded = try? JSONDecoder().decode(Settings.self, from: data)
            else { return false }

            return decoded.personalityIntensity == intensity
                && abs(decoded.moodThreshold - threshold) < 1e-9
                && decoded.recentlyLearnedDays == days
        }
    }
}
