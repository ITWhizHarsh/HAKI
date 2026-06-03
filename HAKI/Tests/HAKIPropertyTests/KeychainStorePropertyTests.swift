// KeychainStorePropertyTests.swift
// HAKIPropertyTests — Property-Based Tests for KeychainStore (SwiftCheck)
//
// These tests use SwiftCheck to verify universal properties of the KeychainStore's
// behaviour across arbitrary secret values and key references.
//
// Feature: haki-personal-ai-assistant, Property 20.2: Secret handling
// Validates: Requirements 20.2 (secret handling - keyRef pattern)
//
// Minimum iterations: 100 (SwiftCheck default).

#if canImport(XCTest)
import XCTest
import SwiftCheck
@testable import HAKI
@testable import HAKIStore

final class KeychainStorePropertyTests: XCTestCase {

    private var store: KeychainStore!
    private let testPrefix = "property_test_"

    override func setUp() {
        super.setUp()
        store = KeychainStore()
    }

    override func tearDown() {
        cleanupTestEntries()
        super.tearDown()
    }

    // MARK: - Property 1
    // Any secret value can be stored and retrieved identically.
    //
    // Feature: haki-personal-ai-assistant, Property 20.2a: Store/retrieve round trip
    // Validates: Req 20.2 (secrets stored in Keychain, keyRef returned)
    func testStoreRetrieveRoundTrip() {
        property("stored secret can be retrieved with the same keyRef") <- forAll { (secret: String) in
            let keyRef = "\(self.testPrefix)\(UUID().uuidString)"
            
            do {
                let returnedKeyRef = try self.store.store(secret: secret, keyRef: keyRef)
                let retrieved = try self.store.retrieve(keyRef: returnedKeyRef)
                return retrieved == secret
            } catch {
                return false
            }
        }
    }

    // MARK: - Property 2
    // After updating a secret, the new value is retrieved.
    //
    // Feature: haki-personal-ai-assistant, Property 20.2b: Secret rotation
    // Validates: Req 20.2 (secret rotation/update)
    func testSecretUpdate() {
        property("updating a secret replaces the old value") <- forAll(
            Gen<String>.from(elements: ["old_secret_1", "old_value_X", "previous"]),
            Gen<String>.from(elements: ["new_secret_1", "new_value_Y", "updated"])
        ) { oldSecret, newSecret in
            let keyRef = "\(self.testPrefix)\(UUID().uuidString)"
            
            do {
                // Store original
                _ = try self.store.store(secret: oldSecret, keyRef: keyRef)
                
                // Update
                _ = try self.store.update(secret: newSecret, keyRef: keyRef)
                
                // Verify new value
                let retrieved = try self.store.retrieve(keyRef: keyRef)
                return retrieved == newSecret && retrieved != oldSecret
            } catch {
                return false
            }
        }
    }

    // MARK: - Property 3
    // Deleting a secret makes it inaccessible.
    //
    // Feature: haki-personal-ai-assistant, Property 20.2c: Secret deletion
    // Validates: Req 20.2 (secret deletion)
    func testSecretDeletion() {
        property("deleting a secret makes it inaccessible") <- forAll { (secret: String) in
            let keyRef = "\(self.testPrefix)\(UUID().uuidString)"
            
            do {
                // Store
                _ = try self.store.store(secret: secret, keyRef: keyRef)
                
                // Verify exists
                guard self.store.exists(keyRef: keyRef) else { return false }
                
                // Delete
                try self.store.delete(keyRef: keyRef)
                
                // Verify gone
                return !self.store.exists(keyRef: keyRef)
            } catch {
                return false
            }
        }
    }

    // MARK: - Property 4
    // KeyRef format is consistent (always has the expected prefix).
    //
    // Feature: haki-personal-ai-assistant, Property 20.2d: KeyRef format
    // Validates: Req 20.2 (keyRef pattern)
    func testKeyRefFormat() {
        property("generated keyRef always has haki_ prefix") <- forAll { (_: String) in
            let keyRef = self.store.generateKeyRef()
            return keyRef.hasPrefix("haki_")
        }

        property("keyRef with prefix always has the provided prefix") <- forAll(
            Gen<String>.from(elements: ["api_key", "oauth_token", "my_secret", "test"])
        ) { prefix in
            let keyRef = self.store.generateKeyRef(prefix: prefix)
            let expectedPrefix = prefix.lowercased().replacingOccurrences(of: " ", with: "_")
            return keyRef.hasPrefix("\(expectedPrefix)_")
        }
    }

    // MARK: - Property 5
    // API keys for all capabilities can be stored and retrieved.
    //
    // Feature: haki-personal-ai-assistant, Property 20.2e: API key capabilities
    // Validates: Req 20.2 (API keys for STT, LLM, TTS, mood, image, embeddings)
    func testAPIKeyForAllCapabilities() {
        property("API key can be stored and retrieved for any capability") <- forAll(
            Gen<String>.from(elements: ModelCapability.allCases.map { $0.rawValue }),
            Gen<String>.suchThat { !$0.isEmpty }
        ) { capabilityName, apiKey in
            guard let capability = ModelCapability(rawValue: capabilityName) else {
                return false
            }
            
            do {
                let keyRef = try self.store.storeAPIKey(apiKey, capability: capability)
                let retrieved = try self.store.retrieve(keyRef: keyRef)
                return retrieved == apiKey && keyRef.hasPrefix("api_key_\(capabilityName)_")
            } catch {
                return false
            }
        }
    }

    // MARK: - Property 6
    // OAuth tokens for all account types can be stored and retrieved.
    //
    // Feature: haki-personal-ai-assistant, Property 20.2f: OAuth token account types
    // Validates: Req 20.2 (OAuth tokens for WhatsApp, email, etc.)
    func testOAuthTokenForAllAccountTypes() {
        property("OAuth token can be stored and retrieved for any account type") <- forAll(
            Gen<String>.from(elements: CommunicationAccount.allCases.map { $0.rawValue }),
            Gen<String>.suchThat { !$0.isEmpty }
        ) { accountTypeName, token in
            guard let accountType = CommunicationAccount(rawValue: accountTypeName) else {
                return false
            }
            
            do {
                let keyRef = try self.store.storeOAuthToken(token, accountType: accountType)
                let retrieved = try self.store.retrieve(keyRef: keyRef)
                return retrieved == token && keyRef.hasPrefix("oauth_\(accountTypeName)_")
            } catch {
                return false
            }
        }
    }

    // MARK: - Property 7
    // Existence check correctly identifies stored vs non-stored keys.
    //
    // Feature: haki-personal-ai-assistant, Property 20.2g: Existence check
    // Validates: Req 20.2 (secret existence verification)
    func testExistsCorrectlyIdentifiesStoredSecrets() {
        property("exists returns true for stored keys, false for non-stored") <- forAll { (secret: String) in
            let keyRef = "\(self.testPrefix)\(UUID().uuidString)"
            let nonExistentKey = "\(self.testPrefix)nonexistent_\(UUID().uuidString)"
            
            do {
                // Store the secret
                _ = try self.store.store(secret: secret, keyRef: keyRef)
                
                // Check existence
                let storedExists = self.store.exists(keyRef: keyRef)
                let notStoredExists = self.store.exists(keyRef: nonExistentKey)
                
                return storedExists && !notStoredExists
            } catch {
                return false
            }
        }
    }

    // MARK: - Property 8
    // Unicode secrets are handled correctly.
    //
    // Feature: haki-personal-ai-assistant, Property 20.2h: Unicode support
    // Validates: Req 20.2 (international secret values)
    func testUnicodeSecrets() {
        property("Unicode secrets can be stored and retrieved") <- forAll(
            Gen<String>.from(elements: [
                "api_key_日本語",
                "token_हिंदी",
                "secret_🔐",
                "key_émojis_🎉",
                "oauth_中文_key"
            ])
        ) { secret in
            let keyRef = "\(self.testPrefix)\(UUID().uuidString)"
            
            do {
                _ = try self.store.store(secret: secret, keyRef: keyRef)
                let retrieved = try self.store.retrieve(keyRef: keyRef)
                return retrieved == secret
            } catch {
                return false
            }
        }
    }

    // MARK: - Property 9
    // Special characters in secrets are preserved.
    //
    // Feature: haki-personal-ai-assistant, Property 20.2i: Special characters
    // Validates: Req 20.2 (complex secret values)
    func testSpecialCharactersInSecrets() {
        property("Special characters in secrets are preserved") <- forAll(
            Gen<String>.from(elements: [
                "sk-abc123!@#$%^&*()",
                "token+with/special\\chars",
                "key=with+plus&pipe|",
                "secret\"with\"quotes",
                "key\nwith\tnewlines"
            ])
        ) { secret in
            let keyRef = "\(self.testPrefix)\(UUID().uuidString)"
            
            do {
                _ = try self.store.store(secret: secret, keyRef: keyRef)
                let retrieved = try self.store.retrieve(keyRef: keyRef)
                return retrieved == secret
            } catch {
                return false
            }
        }
    }

    // MARK: - Property 10
    // Empty secrets are handled (edge case).
    //
    // Feature: haki-personal-ai-assistant, Property 20.2j: Empty secret edge case
    // Validates: Req 20.2 (boundary condition)
    func testEmptySecret() {
        property("empty string secrets can be stored and retrieved") <- forAll { (_: String) in
            let keyRef = "\(self.testPrefix)\(UUID().uuidString)"
            let emptySecret = ""
            
            do {
                _ = try self.store.store(secret: emptySecret, keyRef: keyRef)
                let retrieved = try self.store.retrieve(keyRef: keyRef)
                return retrieved == emptySecret
            } catch {
                return false
            }
        }
    }

    // MARK: - Helpers

    private func cleanupTestEntries() {
        let query: [String: Any] = [
            kSecClass as String:       kSecClassGenericPassword,
            kSecAttrService as String: "com.haki.app",
            kSecReturnAttributes as String: true,
            kSecMatchLimit as String:  kSecMatchLimitAll
        ]

        var items: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &items)
        
        guard status == errSecSuccess, let itemArray = items as? [[String: Any]] else {
            return
        }

        for item in itemArray {
            guard let account = item[kSecAttrAccount as String] as? String else {
                continue
            }
            if account.hasPrefix(testPrefix) {
                let deleteQuery: [String: Any] = [
                    kSecClass as String:       kSecClassGenericPassword,
                    kSecAttrAccount as String: account,
                    kSecAttrService as String: "com.haki.app"
                ]
                SecItemDelete(deleteQuery as CFDictionary)
            }
        }
    }
}
#endif // canImport(XCTest)