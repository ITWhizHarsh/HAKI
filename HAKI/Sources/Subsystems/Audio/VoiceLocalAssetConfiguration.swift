// VoiceLocalAssetConfiguration.swift
// HAKI — local model/voice-asset provisioning contract.
//
// This configuration only verifies artifacts provisioned by the user. It does
// not contact a provider, download or convert a model, or select any fallback
// speech/LLM implementation.

import AVFoundation
import CryptoKit
import Foundation

public enum VoiceLocalArtifactID: String, CaseIterable, Sendable {
    case coreMLQwen3ASR = "qwen3_asr_coreml"
    case qwen34BInstruct4Bit = "qwen3_4b_instruct_4bit"
}

public struct VoiceModelArtifactManifest: Codable, Sendable {
    public let artifactID: String
    public let modelID: String
    public let artifactPath: String
    public let sha256: String
    public let version: String
    public let sampleRateHz: Int?
    public let vocabularyVersion: String?

    enum CodingKeys: String, CodingKey {
        case artifactID = "artifact_id"
        case modelID = "model_id"
        case artifactPath = "artifact_path"
        case sha256
        case version
        case sampleRateHz = "sample_rate_hz"
        case vocabularyVersion = "vocabulary_version"
    }
}

public struct VoiceModelManifest: Codable, Sendable {
    public let schemaVersion: Int
    public let artifacts: [VoiceModelArtifactManifest]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case artifacts
    }
}

/// Verified local configuration consumed by the production CoreML Qwen3 ASR
/// adapter. This contains manifest metadata and a local filesystem URL only;
/// it never represents a network model source or fallback provider.
public struct VoiceLocalASRArtifact: Sendable, Equatable {
    public let modelID: String
    public let artifactURL: URL
    public let sha256: String
    public let version: String
    public let sampleRateHz: Int
    public let vocabularyVersion: String
}

public enum VoiceLocalASRArtifactError: Error, Sendable, Equatable {
    case manifestMissing
    case manifestInvalid
    case artifactMissing
    case artifactInvalid
    case artifactHashMismatch
    case artifactUnreadable
}

public struct VoiceAssetAvailabilityIssue: Sendable, Equatable {
    public let assetID: String
    public let code: String
    public let path: URL
    public let message: String
    public let action: String
}

public struct VoiceAssetAvailability: Sendable {
    public let issues: [VoiceAssetAvailabilityIssue]

    public var isReady: Bool { issues.isEmpty }

    public var actionableSummary: String {
        isReady
            ? "Local Qwen3 ASR, Qwen3-4B-Instruct-4bit, and XTTS voice assets are available."
            : issues.map(\.action).joined(separator: " ")
    }
}

/// Non-secret local paths and identifiers used by the Swift voice subsystem.
/// Expected content hashes come from the user-local JSON manifest and are never
/// embedded in application code or source control.
public struct VoiceLocalAssetConfiguration: Sendable {
    public static let manifestFilename = "voice-model-manifest.json"
    public static let coreMLASRModelID = "Qwen/Qwen3-ASR-CoreML"
    public static let qwenLLMModelID = "Qwen/Qwen3-4B-Instruct-4bit"

    public let modelDirectory: URL
    public let manifestURL: URL
    public let voiceAssetURL: URL

    public init(
        modelDirectory: URL = Self.defaultModelDirectory(),
        manifestURL: URL? = nil,
        voiceAssetURL: URL? = nil
    ) {
        self.modelDirectory = modelDirectory
        self.manifestURL = manifestURL ?? modelDirectory.appendingPathComponent(Self.manifestFilename)
        self.voiceAssetURL = voiceAssetURL ?? modelDirectory.appendingPathComponent("my_voice.wav")
    }

    public static func defaultModelDirectory(fileManager: FileManager = .default) -> URL {
        let applicationSupport = fileManager.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        return applicationSupport
            .appendingPathComponent("HAKI", isDirectory: true)
            .appendingPathComponent("models", isDirectory: true)
    }

    /// Validates only local files. Invoke from a utility queue at startup so a
    /// large artifact hash cannot block the main/UI or audio threads.
    public func availability(
        microphoneAuthorization: AVAuthorizationStatus = AVCaptureDevice.authorizationStatus(for: .audio)
    ) -> VoiceAssetAvailability {
        var issues: [VoiceAssetAvailabilityIssue] = []
        validateMicrophoneAuthorization(microphoneAuthorization, issues: &issues)
        validateVoiceAsset(issues: &issues)

        guard let manifest = loadManifest(issues: &issues) else {
            return VoiceAssetAvailability(issues: issues)
        }
        validateRequiredArtifacts(manifest, issues: &issues)
        return VoiceAssetAvailability(issues: issues)
    }

    /// Returns the verified, locally provisioned CoreML Qwen3 ASR artifact for
    /// the native adapter. This narrow method deliberately does not check for
    /// unrelated voice assets, download files, or offer cloud/legacy fallback.
    public func coreMLQwen3ASRArtifact() throws -> VoiceLocalASRArtifact {
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw VoiceLocalASRArtifactError.manifestMissing
        }
        let manifest: VoiceModelManifest
        do {
            try Self.requireReadableFile(manifestURL)
            manifest = try JSONDecoder().decode(VoiceModelManifest.self, from: Data(contentsOf: manifestURL))
        } catch {
            throw VoiceLocalASRArtifactError.manifestInvalid
        }
        guard manifest.schemaVersion == 1,
              let artifact = manifest.artifacts.first(where: { $0.artifactID == VoiceLocalArtifactID.coreMLQwen3ASR.rawValue }),
              artifact.modelID == Self.coreMLASRModelID,
              artifact.sampleRateHz == 16_000,
              artifact.vocabularyVersion?.isEmpty == false,
              isSHA256(artifact.sha256),
              !artifact.version.isEmpty,
              !artifact.artifactPath.isEmpty,
              !artifact.artifactPath.hasPrefix("/"),
              !URL(fileURLWithPath: artifact.artifactPath).pathComponents.contains("..") else {
            throw VoiceLocalASRArtifactError.artifactInvalid
        }

        let artifactURL = modelDirectory.appendingPathComponent(artifact.artifactPath).standardizedFileURL
        guard artifactURL.path == modelDirectory.standardizedFileURL.path ||
                artifactURL.path.hasPrefix(modelDirectory.standardizedFileURL.path + "/") else {
            throw VoiceLocalASRArtifactError.artifactInvalid
        }
        guard FileManager.default.fileExists(atPath: artifactURL.path) else {
            throw VoiceLocalASRArtifactError.artifactMissing
        }
        do {
            try Self.requireReadableFile(artifactURL)
            guard try Self.artifactSHA256(at: artifactURL) == artifact.sha256 else {
                throw VoiceLocalASRArtifactError.artifactHashMismatch
            }
        } catch let error as VoiceLocalASRArtifactError {
            throw error
        } catch {
            throw VoiceLocalASRArtifactError.artifactUnreadable
        }

        return VoiceLocalASRArtifact(
            modelID: artifact.modelID,
            artifactURL: artifactURL,
            sha256: artifact.sha256,
            version: artifact.version,
            sampleRateHz: 16_000,
            vocabularyVersion: artifact.vocabularyVersion!
        )
    }

    /// Deterministic content hash for the same file/directory format validated
    /// by Core. This is used by explicit provisioning tooling, never during a
    /// voice turn to replace a mismatched expected hash.
    public static func artifactSHA256(at url: URL) throws -> String {
        var hasher = SHA256()
        let manager = FileManager.default
        var isDirectory: ObjCBool = false
        guard manager.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            throw ConfigurationError.missing(url)
        }

        if isDirectory.boolValue {
            guard let enumerator = manager.enumerator(
                at: url,
                includingPropertiesForKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey],
                options: [.skipsHiddenFiles]
            ) else {
                throw ConfigurationError.unreadable(url)
            }
            var entries: [(url: URL, isDirectory: Bool)] = []
            for case let entry as URL in enumerator {
                let values = try entry.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey])
                if values.isSymbolicLink == true {
                    throw ConfigurationError.invalidManifest("symbolic link in artifact")
                }
                if values.isDirectory == true {
                    entries.append((entry, true))
                } else if values.isRegularFile == true {
                    try requireReadableFile(entry)
                    entries.append((entry, false))
                } else {
                    throw ConfigurationError.invalidManifest("unsupported artifact file type")
                }
            }
            for entry in entries.sorted(by: { $0.url.path < $1.url.path }) {
                let relative = entry.url.path.replacingOccurrences(of: url.path + "/", with: "")
                if entry.isDirectory {
                    hasher.update(data: Data("D\0\(relative)\0".utf8))
                } else {
                    hasher.update(data: Data("F\0\(relative)\0".utf8))
                    try update(&hasher, withFile: entry.url)
                }
            }
        } else {
            try requireReadableFile(url)
            try update(&hasher, withFile: url)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private func validateMicrophoneAuthorization(
        _ authorization: AVAuthorizationStatus,
        issues: inout [VoiceAssetAvailabilityIssue]
    ) {
        guard authorization == .authorized else {
            issues.append(VoiceAssetAvailabilityIssue(
                assetID: "microphone_permission",
                code: "microphone_permission_unavailable",
                path: modelDirectory,
                message: "Microphone permission is not granted for local voice capture.",
                action: "Enable HAKI in System Settings → Privacy & Security → Microphone, then retry voice."
            ))
            return
        }
    }

    private func validateVoiceAsset(issues: inout [VoiceAssetAvailabilityIssue]) {
        guard FileManager.default.fileExists(atPath: voiceAssetURL.path) else {
            issues.append(VoiceAssetAvailabilityIssue(
                assetID: "my_voice.wav",
                code: "voice_asset_missing",
                path: voiceAssetURL,
                message: "The XTTS conditioning file my_voice.wav is missing.",
                action: "Place a readable user-supplied my_voice.wav at \(voiceAssetURL.path)."
            ))
            return
        }
        do {
            try Self.requireReadableFile(voiceAssetURL)
        } catch {
            issues.append(VoiceAssetAvailabilityIssue(
                assetID: "my_voice.wav",
                code: "voice_asset_unreadable",
                path: voiceAssetURL,
                message: error.localizedDescription,
                action: "Grant the current user read access to my_voice.wav and retry voice."
            ))
        }
    }

    private func loadManifest(issues: inout [VoiceAssetAvailabilityIssue]) -> VoiceModelManifest? {
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            issues.append(VoiceAssetAvailabilityIssue(
                assetID: "voice_model_manifest",
                code: "manifest_missing",
                path: manifestURL,
                message: "The local voice model manifest is missing.",
                action: "Provision local Qwen3 artifacts and create \(Self.manifestFilename) before starting voice."
            ))
            return nil
        }
        do {
            try Self.requireReadableFile(manifestURL)
            let manifest = try JSONDecoder().decode(VoiceModelManifest.self, from: Data(contentsOf: manifestURL))
            guard manifest.schemaVersion == 1 else {
                throw ConfigurationError.invalidManifest("unsupported schema \(manifest.schemaVersion)")
            }
            return manifest
        } catch {
            issues.append(VoiceAssetAvailabilityIssue(
                assetID: "voice_model_manifest",
                code: "manifest_invalid",
                path: manifestURL,
                message: error.localizedDescription,
                action: "Repair or reprovision the non-secret local voice manifest before starting voice."
            ))
            return nil
        }
    }

    private func validateRequiredArtifacts(
        _ manifest: VoiceModelManifest,
        issues: inout [VoiceAssetAvailabilityIssue]
    ) {
        validateArtifact(
            id: .coreMLQwen3ASR,
            modelID: Self.coreMLASRModelID,
            manifest: manifest,
            requiresASRMetadata: true,
            issues: &issues
        )
        validateArtifact(
            id: .qwen34BInstruct4Bit,
            modelID: Self.qwenLLMModelID,
            manifest: manifest,
            requiresASRMetadata: false,
            issues: &issues
        )
    }

    private func validateArtifact(
        id: VoiceLocalArtifactID,
        modelID: String,
        manifest: VoiceModelManifest,
        requiresASRMetadata: Bool,
        issues: inout [VoiceAssetAvailabilityIssue]
    ) {
        guard let artifact = manifest.artifacts.first(where: { $0.artifactID == id.rawValue }) else {
            issues.append(VoiceAssetAvailabilityIssue(
                assetID: id.rawValue,
                code: "artifact_manifest_missing",
                path: modelDirectory,
                message: "The manifest does not declare \(id.rawValue).",
                action: "Add a verified \(id.rawValue) entry to \(Self.manifestFilename)."
            ))
            return
        }
        guard artifact.modelID == modelID,
              isSHA256(artifact.sha256),
              !artifact.version.isEmpty,
              !artifact.artifactPath.isEmpty,
              !URL(fileURLWithPath: artifact.artifactPath).pathComponents.contains(".."),
              !artifact.artifactPath.hasPrefix("/") else {
            issues.append(VoiceAssetAvailabilityIssue(
                assetID: id.rawValue,
                code: "artifact_manifest_invalid",
                path: modelDirectory,
                message: "The \(id.rawValue) manifest entry has invalid local metadata.",
                action: "Reprovision \(id.rawValue) and its non-secret manifest entry."
            ))
            return
        }
        if requiresASRMetadata && (artifact.sampleRateHz != 16_000 || artifact.vocabularyVersion?.isEmpty != false) {
            issues.append(VoiceAssetAvailabilityIssue(
                assetID: id.rawValue,
                code: "asr_manifest_invalid",
                path: modelDirectory.appendingPathComponent(artifact.artifactPath),
                message: "CoreML Qwen3 ASR must declare a 16000 Hz sample rate and vocabulary version.",
                action: "Repair the CoreML Qwen3 ASR manifest metadata before starting voice."
            ))
            return
        }

        let artifactURL = modelDirectory.appendingPathComponent(artifact.artifactPath).standardizedFileURL
        guard artifactURL.path == modelDirectory.standardizedFileURL.path ||
                artifactURL.path.hasPrefix(modelDirectory.standardizedFileURL.path + "/") else {
            issues.append(VoiceAssetAvailabilityIssue(
                assetID: id.rawValue,
                code: "artifact_path_invalid",
                path: artifactURL,
                message: "The artifact path escapes the user-local model directory.",
                action: "Use a relative artifact path inside the local HAKI models directory."
            ))
            return
        }
        guard FileManager.default.fileExists(atPath: artifactURL.path) else {
            issues.append(VoiceAssetAvailabilityIssue(
                assetID: id.rawValue,
                code: "artifact_missing",
                path: artifactURL,
                message: "Required local artifact \(id.rawValue) is missing.",
                action: "Provision \(id.rawValue) locally before starting voice."
            ))
            return
        }

        do {
            let actualHash = try Self.artifactSHA256(at: artifactURL)
            guard actualHash == artifact.sha256 else {
                issues.append(VoiceAssetAvailabilityIssue(
                    assetID: id.rawValue,
                    code: "artifact_hash_mismatch",
                    path: artifactURL,
                    message: "The local artifact hash differs from the provisioned manifest.",
                    action: "Reprovision \(id.rawValue); do not replace its expected hash during a voice turn."
                ))
                return
            }
        } catch {
            issues.append(VoiceAssetAvailabilityIssue(
                assetID: id.rawValue,
                code: "artifact_unreadable",
                path: artifactURL,
                message: error.localizedDescription,
                action: "Grant the current user read access to \(id.rawValue) and retry voice."
            ))
        }
    }

    private static func regularFiles(in directory: URL) throws -> [URL] {
        var files: [URL] = []
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey],
            options: [.skipsHiddenFiles]
        ) else {
            throw ConfigurationError.unreadable(directory)
        }
        for case let url as URL in enumerator {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            if values.isSymbolicLink == true { throw ConfigurationError.invalidManifest("symbolic link in artifact") }
            if values.isRegularFile == true {
                try requireReadableFile(url)
                files.append(url)
            }
        }
        return files.sorted { $0.path < $1.path }
    }

    private static func requireReadableFile(_ url: URL) throws {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard attributes[.type] as? FileAttributeType == .typeRegular,
              let permissions = attributes[.posixPermissions] as? NSNumber,
              permissions.intValue & 0o444 != 0,
              FileManager.default.isReadableFile(atPath: url.path) else {
            throw ConfigurationError.unreadable(url)
        }
    }

    private static func update(_ hasher: inout SHA256, withFile url: URL) throws {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        while let chunk = try handle.read(upToCount: 1_048_576), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
    }

    private func isSHA256(_ value: String) -> Bool {
        value.count == 64 && value.allSatisfy { $0.isHexDigit && !$0.isUppercase }
    }

    private enum ConfigurationError: LocalizedError {
        case missing(URL)
        case unreadable(URL)
        case invalidManifest(String)

        var errorDescription: String? {
            switch self {
            case .missing(let url): return "Missing local artifact at \(url.path)"
            case .unreadable(let url): return "Local artifact is unreadable: \(url.path)"
            case .invalidManifest(let message): return "Invalid local voice manifest: \(message)"
            }
        }
    }
}
