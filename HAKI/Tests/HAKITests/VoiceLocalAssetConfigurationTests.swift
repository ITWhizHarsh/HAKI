// VoiceLocalAssetConfigurationTests.swift
// Local voice provisioning checks — Requirements 3.1, 6.2, 7.1–7.2, 9.6

#if canImport(XCTest)
import AVFoundation
import CryptoKit
import Foundation
import XCTest
@testable import HAKIAudio

final class VoiceLocalAssetConfigurationTests: XCTestCase {
    private var root: URL!
    private var configuration: VoiceLocalAssetConfiguration!
    private var asrURL: URL!
    private var llmURL: URL!
    private var voiceURL: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("haki-voice-assets-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        asrURL = root.appendingPathComponent("asr/Qwen3ASR.mlmodelc")
        llmURL = root.appendingPathComponent("llm/Qwen3-4B-Instruct-4bit")
        voiceURL = root.appendingPathComponent("my_voice.wav")
        try FileManager.default.createDirectory(at: asrURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: llmURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("verified-asr".utf8).write(to: asrURL)
        try Data("verified-llm".utf8).write(to: llmURL)
        try Data("RIFF\0\0\0\0WAVE".utf8).write(to: voiceURL)
        try writeManifest()
        configuration = VoiceLocalAssetConfiguration(modelDirectory: root)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    func testVerifiedLocalArtifactsAndReadableVoiceAssetAreReady() {
        let availability = configuration.availability(microphoneAuthorization: .authorized)
        XCTAssertTrue(availability.isReady)
        XCTAssertTrue(availability.issues.isEmpty)
    }

    func testMissingVoiceAssetIsActionableAndDoesNotSelectFallback() throws {
        try FileManager.default.removeItem(at: voiceURL)
        let availability = configuration.availability(microphoneAuthorization: .authorized)
        let issue = try XCTUnwrap(availability.issues.first { $0.assetID == "my_voice.wav" })
        XCTAssertEqual(issue.code, "voice_asset_missing")
        XCTAssertTrue(issue.action.localizedCaseInsensitiveContains("readable"))
        XCTAssertFalse(availability.actionableSummary.localizedCaseInsensitiveContains("fallback"))
    }

    func testHashMismatchAndUnreadableAssetAreRejected() throws {
        try Data("tampered".utf8).write(to: llmURL)
        var availability = configuration.availability(microphoneAuthorization: .authorized)
        XCTAssertTrue(availability.issues.contains { $0.code == "artifact_hash_mismatch" })

        try Data("verified-llm".utf8).write(to: llmURL)
        try writeManifest()
        try FileManager.default.setAttributes([.posixPermissions: 0], ofItemAtPath: voiceURL.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: voiceURL.path) }
        availability = configuration.availability(microphoneAuthorization: .authorized)
        XCTAssertTrue(availability.issues.contains { $0.code == "voice_asset_unreadable" })
    }

    func testMicrophonePermissionIsReportedAtStartup() {
        let availability = configuration.availability(microphoneAuthorization: .denied)
        XCTAssertTrue(availability.issues.contains { $0.code == "microphone_permission_unavailable" })
        XCTAssertTrue(availability.actionableSummary.localizedCaseInsensitiveContains("microphone"))
    }

    private func writeManifest() throws {
        let asrHash = try VoiceLocalAssetConfiguration.artifactSHA256(at: asrURL)
        let llmHash = try VoiceLocalAssetConfiguration.artifactSHA256(at: llmURL)
        let manifest: [String: Any] = [
            "schema_version": 1,
            "artifacts": [
                [
                    "artifact_id": "qwen3_asr_coreml",
                    "model_id": VoiceLocalAssetConfiguration.coreMLASRModelID,
                    "artifact_path": "asr/Qwen3ASR.mlmodelc",
                    "sha256": asrHash,
                    "version": "2025.1",
                    "sample_rate_hz": 16_000,
                    "vocabulary_version": "2025.1",
                ],
                [
                    "artifact_id": "qwen3_4b_instruct_4bit",
                    "model_id": VoiceLocalAssetConfiguration.qwenLLMModelID,
                    "artifact_path": "llm/Qwen3-4B-Instruct-4bit",
                    "sha256": llmHash,
                    "version": "2025.1",
                ],
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
        try data.write(to: root.appendingPathComponent(VoiceLocalAssetConfiguration.manifestFilename))
    }
}
#endif
