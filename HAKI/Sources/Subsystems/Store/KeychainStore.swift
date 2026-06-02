// KeychainStore.swift
// HAKI — Store Subsystem / Keychain
//
// Stores OAuth tokens and API keys in the macOS Keychain.
// The AppStore holds only the `keyRef` string; the actual secret never
// appears in the SQLite database, logs, vault notes, or plan state.
//
// Full implementation: Phase 0 Task 2.2.
// Implements: Req 20.2 (secret handling), Design: Security Considerations

import Foundation
import Security

// MARK: - KeychainStore

/// Keychain manager for secure secret storage.
/// 
/// Provides secure storage for:
/// - OAuth tokens for communication accounts (WhatsApp, email)
/// - API keys for external model providers (STT, LLM, TTS, image, embeddings)
/// 
/// Security properties:
/// - Secrets are stored in macOS Keychain with `kSecAttrAccessibleWhenUnlockedThisDeviceOnly`
/// - Only keyRef identifiers are returned to callers (secrets never exposed)
/// - Secrets are never logged or exposed in error messages
/// - Keychain items are scoped to the app (com.haki.app service)
public final class KeychainStore: @unchecked Sendable {

    public static let shared = KeychainStore()

    /// Service identifier for Keychain items
    private let serviceIdentifier = "com.haki.app"

    private init() {}

    // MARK: - KeyRef Generation

    /// Generates a unique keyRef identifier for storing secrets.
    /// The keyRef is a UUID that serves as the handle stored in the AppStore.
    public func generateKeyRef() -> String {
        return "haki_\(UUID().uuidString)"
    }

    /// Generates a keyRef with a descriptive prefix for organization.
    /// - Parameter prefix: Descriptive prefix (e.g., "api_key", "oauth_token")
    /// - Returns: A unique keyRef with the given prefix
    public func generateKeyRef(prefix: String) -> String {
        let sanitizedPrefix = prefix
            .lowercased()
            .replacingOccurrences(of: " ", with: "_")
            .prefix(20)
        return "\(sanitizedPrefix)_\(UUID().uuidString)"
    }

    // MARK: - Secret Storage API

    /// Store a secret value under a generated keyRef in the Keychain.
    /// Returns the keyRef that should be stored in the AppStore.
    /// - Parameter secret: The secret value to store
    /// - Returns: The keyRef string to store in the database
    @discardableResult
    public func store(secret: String) throws -> String {
        let keyRef = generateKeyRef()
        return try store(secret: secret, keyRef: keyRef)
    }

    /// Store a secret value under a specific keyRef in the Keychain.
    /// Returns the keyRef that should be stored in the AppStore.
    /// - Parameters:
    ///   - secret: The secret value to store
    ///   - keyRef: The key reference to use (will be created if doesn't exist)
    /// - Returns: The keyRef string
    @discardableResult
    public func store(secret: String, keyRef: String) throws -> String {
        let data = Data(secret.utf8)
        let query: [String: Any] = [
            kSecClass as String:       kSecClassGenericPassword,
            kSecAttrAccount as String: keyRef,
            kSecAttrService as String: serviceIdentifier,
            kSecValueData as String:   data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]

        // Delete any existing entry before adding (supports rotation/update)
        SecItemDelete(query as CFDictionary)

        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw KeychainError.writeFailed(status)
        }
        return keyRef
    }

    /// Store an API key for a model provider capability.
    /// - Parameters:
    ///   - apiKey: The API key to store
    ///   - capability: The capability this key is for (stt, llm, tts, mood, image, embeddings)
    ///   - providerName: Optional provider name for disambiguation
    /// - Returns: The keyRef string to store in the database
    @discardableResult
    public func storeAPIKey(_ apiKey: String, capability: ModelCapability, providerName: String? = nil) throws -> String {
        let prefix = "api_key_\(capability.rawValue)"
        let keyRef: String
        if let provider = providerName {
            keyRef = generateKeyRef(prefix: "\(prefix)_\(provider)")
        } else {
            keyRef = generateKeyRef(prefix: prefix)
        }
        return try store(secret: apiKey, keyRef: keyRef)
    }

    /// Store an OAuth token for a communication account.
    /// - Parameters:
    ///   - token: The OAuth token to store
    ///   - accountType: The communication account type (whatsapp, email)
    ///   - accountId: Optional account identifier for multiple accounts
    /// - Returns: The keyRef string to store in the database
    @discardableResult
    public func storeOAuthToken(_ token: String, accountType: CommunicationAccount, accountId: String? = nil) throws -> String {
        let prefix = "oauth_\(accountType.rawValue)"
        let keyRef: String
        if let id = accountId {
            keyRef = generateKeyRef(prefix: "\(prefix)_\(id)")
        } else {
            keyRef = generateKeyRef(prefix: prefix)
        }
        return try store(secret: token, keyRef: keyRef)
    }

    /// Retrieve a secret value by keyRef.
    /// - Parameter keyRef: The key reference to look up
    /// - Returns: The secret value
    /// - Throws: KeychainError.notFound if the keyRef doesn't exist
    public func retrieve(keyRef: String) throws -> String {
        let query: [String: Any] = [
            kSecClass as String:       kSecClassGenericPassword,
            kSecAttrAccount as String: keyRef,
            kSecAttrService as String: serviceIdentifier,
            kSecReturnData as String:  true,
            kSecMatchLimit as String:  kSecMatchLimitOne
        ]

        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess else {
            if status == errSecItemNotFound {
                throw KeychainError.notFound(keyRef)
            }
            throw KeychainError.readFailed(status)
        }
        guard let data = item as? Data else {
            throw KeychainError.invalidData(keyRef)
        }
        return String(decoding: data, as: UTF8.self)
    }

    /// Update (rotate) an existing secret.
    /// - Parameters:
    ///   - newSecret: The new secret value
    ///   - keyRef: The existing keyRef to update
    /// - Throws: KeychainError.notFound if the keyRef doesn't exist
    @discardableResult
    public func update(secret newSecret: String, keyRef: String) throws -> String {
        // Check if the keyRef exists first
        let checkQuery: [String: Any] = [
            kSecClass as String:       kSecClassGenericPassword,
            kSecAttrAccount as String: keyRef,
            kSecAttrService as String: serviceIdentifier
        ]
        
        let checkStatus = SecItemCopyMatching(checkQuery as CFDictionary, nil)
        if checkStatus == errSecItemNotFound {
            throw KeychainError.notFound(keyRef)
        }
        
        // Use store which handles both insert and update via delete + add
        return try store(secret: newSecret, keyRef: keyRef)
    }

    /// Delete the secret associated with keyRef.
    /// - Parameter keyRef: The key reference to delete
    /// - Throws: KeychainError.deleteFailed if deletion fails (not found is ignored)
    public func delete(keyRef: String) throws {
        let query: [String: Any] = [
            kSecClass as String:       kSecClassGenericPassword,
            kSecAttrAccount as String: keyRef,
            kSecAttrService as String: serviceIdentifier
        ]
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.deleteFailed(status)
        }
    }

    /// Check if a keyRef exists in the Keychain.
    /// - Parameter keyRef: The key reference to check
    /// - Returns: True if the keyRef exists
    public func exists(keyRef: String) -> Bool {
        let query: [String: Any] = [
            kSecClass as String:       kSecClassGenericPassword,
            kSecAttrAccount as String: keyRef,
            kSecAttrService as String: serviceIdentifier,
            kSecReturnData as String:  false
        ]
        let status = SecItemCopyMatching(query as CFDictionary, nil)
        return status == errSecSuccess
    }

    /// Delete all secrets for a specific capability.
    /// Useful when switching providers or removing a capability configuration.
    /// - Parameter capability: The capability to clean up
    public func deleteAll(forCapability capability: ModelCapability) throws {
        let prefix = "api_key_\(capability.rawValue)"
        try deleteAll(withPrefix: prefix)
    }

    /// Delete all secrets for a specific communication account type.
    /// - Parameter accountType: The account type to clean up
    public func deleteAll(forAccountType accountType: CommunicationAccount) throws {
        let prefix = "oauth_\(accountType.rawValue)"
        try deleteAll(withPrefix: prefix)
    }

    /// Delete all Keychain items created by this app.
    /// Use with caution - this removes all stored secrets.
    public func deleteAll() throws {
        let query: [String: Any] = [
            kSecClass as String:       kSecClassGenericPassword,
            kSecAttrService as String: serviceIdentifier
        ]
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.deleteFailed(status)
        }
    }

    // MARK: - Private Helpers

    /// Delete all items with a specific prefix in the account name.
    private func deleteAll(withPrefix prefix: String) throws {
        // Query all items for our service
        let query: [String: Any] = [
            kSecClass as String:       kSecClassGenericPassword,
            kSecAttrService as String: serviceIdentifier,
            kSecReturnAttributes as String: true,
            kSecMatchLimit as String:  kSecMatchLimitAll
        ]

        var items: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &items)
        
        guard status == errSecSuccess else {
            if status == errSecItemNotFound {
                return // Nothing to delete
            }
            throw KeychainError.readFailed(status)
        }
        
        guard let itemArray = items as? [[String: Any]] else {
            return
        }
        
        // Delete items matching the prefix
        for item in itemArray {
            guard let account = item[kSecAttrAccount as String] as? String else {
                continue
            }
            if account.hasPrefix(prefix) {
                try delete(keyRef: account)
            }
        }
    }
}

// MARK: - Supporting Types

/// Model capability types that can have API keys
public enum ModelCapability: String, Codable, Sendable, CaseIterable {
    case stt       // Speech-to-text
    case llm       // Large language model
    case tts       // Text-to-speech
    case mood      // Mood detection
    case image     // Image generation
    case embeddings // Text embeddings
}

/// Communication account types for OAuth tokens
public enum CommunicationAccount: String, Codable, Sendable, CaseIterable {
    case whatsapp
    case email
    case slack
    case teams
}

// MARK: - KeychainError

/// Errors that can occur during Keychain operations
public enum KeychainError: Error, LocalizedError {
    case writeFailed(OSStatus)
    case readFailed(OSStatus)
    case notFound(String)
    case deleteFailed(OSStatus)
    case invalidData(String)

    public var errorDescription: String? {
        switch self {
        case .writeFailed(let status):
            return "Failed to write to Keychain (status: \(status))"
        case .readFailed(let status):
            return "Failed to read from Keychain (status: \(status))"
        case .notFound:
            return "Secret not found in Keychain"
        case .deleteFailed(let status):
            return "Failed to delete from Keychain (status: \(status))"
        case .invalidData:
            return "Invalid data stored in Keychain"
        }
    }
}
