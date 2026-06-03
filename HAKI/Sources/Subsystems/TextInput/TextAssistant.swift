// TextAssistant.swift
// HAKI — TextInput Subsystem
//
// Implements the Text_Assistant component described in the design:
//
//   TextAssistant:
//     onInput(field, text) -> inlineCorrection? (confidence>=threshold)    (16.1)
//     suggestCompletion(field, context) -> singleSuggestion                (16.2, 16.3)
//     accept(suggestion) / dismiss(suggestion)                             (16.4, 16.5)
//     enabled: Bool  # when off, no detection/prep at all                  (16.6)
//
// Task 19.1 adds:
//   - SpellCheckProvider protocol + NativeSpellCheckProvider (NSSpellChecker)
//   - InlineCorrectionResult enum (disabled / corrected / belowThreshold / noCorrection)
//   - observeField(_:) / stopObserving(_:) via AXObserver / kAXValueChangedNotification
//   - onInput(field:text:) -> InlineCorrectionResult (confidence-gated inline write)
//
// Design reference: Text_Assistant (design.md)
// Implements: Req 16.1, 16.2, 16.3, 16.4, 16.5, 16.6

import Foundation
import AppKit
import ApplicationServices

// MARK: - SpellCheckResult

/// The result returned by a ``SpellCheckProvider``.
public struct SpellCheckResult: Sendable {
    /// The suggested corrected text (may equal the input when no correction is needed).
    public let corrected: String
    /// Confidence in [0.0, 1.0].  0.0 means no correction was found.
    public let confidence: Double

    public init(corrected: String, confidence: Double) {
        self.corrected = corrected
        self.confidence = min(max(confidence, 0.0), 1.0)
    }
}

// MARK: - SpellCheckProvider

/// Abstraction over a spelling-correction back-end.
///
/// `TextAssistant` uses this protocol so tests can inject a deterministic stub
/// and the production path can use `NativeSpellCheckProvider`.
public protocol SpellCheckProvider: Sendable {
    /// Check `text` for spelling errors.
    ///
    /// - Returns: A `SpellCheckResult` whose `corrected` field is the suggested
    ///   replacement and whose `confidence` is in [0.0, 1.0].  Return
    ///   `confidence == 0.0` when no correction is available.
    func check(_ text: String) async -> SpellCheckResult
}

// MARK: - NativeSpellCheckProvider

/// `SpellCheckProvider` backed by `NSSpellChecker`.
///
/// Correction logic:
///   • Locate the first misspelled word in `text`.
///   • If `NSSpellChecker` can suggest a replacement, return the corrected
///     text with confidence **0.9**.
///   • If no misspelling is found or no replacement can be suggested, return
///     the original text with confidence **0.0**.
///
/// `NSSpellChecker` must be accessed on the main thread; all work is
/// dispatched to `@MainActor` accordingly.
public final class NativeSpellCheckProvider: SpellCheckProvider, @unchecked Sendable {

    public init() {}

    public func check(_ text: String) async -> SpellCheckResult {
        return await MainActor.run {
            let checker = NSSpellChecker.shared

            // Use nil to let NSSpellChecker auto-detect the language when
            // automaticallyIdentifiesLanguages is enabled; otherwise call the
            // no-argument language() method to get the current language string.
            let lang: String? = checker.automaticallyIdentifiesLanguages ? nil : checker.language()

            let misspelledRange = checker.checkSpelling(
                of: text,
                startingAt: 0,
                language: lang,
                wrap: false,
                inSpellDocumentWithTag: 0,
                wordCount: nil
            )

            // NSNotFound → text is correctly spelled.
            guard misspelledRange.location != NSNotFound, misspelledRange.length > 0 else {
                return SpellCheckResult(corrected: text, confidence: 0.0)
            }

            // Ask for the best replacement for the misspelled word.
            guard let suggestion = checker.correction(
                forWordRange: misspelledRange,
                in: text,
                language: checker.language(),
                inSpellDocumentWithTag: 0
            ) else {
                return SpellCheckResult(corrected: text, confidence: 0.0)
            }

            let corrected = (text as NSString).replacingCharacters(in: misspelledRange, with: suggestion)
            // Fixed confidence for real NSSpellChecker corrections (per spec).
            return SpellCheckResult(corrected: corrected, confidence: 0.9)
        }
    }
}

// MARK: - InlineCorrectionResult

/// The outcome returned by ``TextAssistant/onInput(field:text:)``.  (Req 16.1)
public enum InlineCorrectionResult: Equatable {
    /// The assistant is disabled; no work was performed.  (Req 16.6)
    case disabled
    /// A correction was applied inline at or above the configured threshold.
    case corrected(original: String, corrected: String, confidence: Double)
    /// A correction candidate existed but its confidence was below the threshold.
    /// The field was **not** modified.
    case belowThreshold(confidence: Double)
    /// The text was already correctly spelled; no correction was needed.
    case noCorrection

    public static func == (lhs: InlineCorrectionResult, rhs: InlineCorrectionResult) -> Bool {
        switch (lhs, rhs) {
        case (.disabled, .disabled), (.noCorrection, .noCorrection):
            return true
        case let (.corrected(lo, lc, lconf), .corrected(ro, rc, rconf)):
            return lo == ro && lc == rc && lconf == rconf
        case let (.belowThreshold(lc), .belowThreshold(rc)):
            return lc == rc
        default:
            return false
        }
    }
}

// MARK: - CompletionProvider

/// Provides a single context-aware completion for an input field.  (Req 16.2, 16.3)
public protocol CompletionProvider: Sendable {
    /// Return a completion string to append/offer, or `nil` when none is useful.
    func complete(fieldText: String, memoryContext: String) async -> String?
}

// MARK: - MemoryBackedCompletionProvider

/// Stub `CompletionProvider` that always returns `nil`.
///
/// Replace with a real Memory_Brain-backed implementation once the IPC
/// channel is available.
public struct MemoryBackedCompletionProvider: CompletionProvider {
    public let memorySnippet: String

    public init(memorySnippet: String = "") {
        self.memorySnippet = memorySnippet
    }

    public func complete(fieldText: String, memoryContext: String) async -> String? {
        return nil
    }
}

// MARK: - CompletionSuggestion

/// A single in-flight completion suggestion.  (Req 16.2–16.5)
public struct CompletionSuggestion: Equatable, Sendable {
    public let text: String
    /// Stable identifier for the AX field (derived from its AX attributes).
    public let fieldId: String
    /// Verbatim field text at generation time — used as the dismissal key.
    public let inputState: String

    public init(text: String, fieldId: String, inputState: String) {
        self.text = text
        self.fieldId = fieldId
        self.inputState = inputState
    }
}

// MARK: - DismissedKey

/// Composite key that uniquely identifies a (field, input-state) pair for
/// dismissal deduplication.  (Req 16.5)
public struct DismissedKey: Hashable, Sendable {
    public let fieldId: String
    public let inputState: String

    public init(fieldId: String, inputState: String) {
        self.fieldId = fieldId
        self.inputState = inputState
    }
}

// MARK: - InlineCorrection (legacy helper used by the completion path)

/// Represents a spelling correction candidate with its confidence.
public struct InlineCorrection: Equatable, Sendable {
    public let correctedText: String
    public let confidence: Double

    public init(correctedText: String, confidence: Double) {
        self.correctedText = correctedText
        self.confidence = confidence
    }
}

// MARK: - TextAssistant

/// Swift `actor` that implements the Text_Assistant component.
///
/// ### Task 19.1 — AX field observation + confidence-gated inline correction
/// The actor wraps an `AXObserver` registry (see ``observeField(_:)`` /
/// ``stopObserving(_:)``) that fires ``onInput(field:text:)`` whenever the
/// value of a registered text field changes.  Corrections are applied inline
/// only when the `SpellCheckProvider`'s confidence is at or above
/// ``correctionThreshold``.  (Req 16.1)
///
/// ### Disabled inertness  (Req 16.6)
/// When ``enabled`` is `false`, all entry points return immediately without
/// touching the AX tree, calling any provider, or doing any background work.
///
/// Design reference: Text_Assistant (design.md)
/// Implements: Req 16.1, 16.2, 16.3, 16.4, 16.5, 16.6
public actor TextAssistant {

    // MARK: - Public state

    /// Master on/off switch.  (Req 16.6)
    public private(set) var enabled: Bool

    /// Single active completion suggestion.  (Req 16.3)
    public private(set) var currentSuggestion: CompletionSuggestion?

    /// (fieldId, inputState) pairs for which suggestions have been dismissed.
    public private(set) var dismissedSuggestions: Set<DismissedKey>

    // MARK: - Configuration

    /// Inline-correction confidence threshold in [0.0, 1.0].  (Req 16.1)
    ///
    /// A correction is applied only when the provider's confidence is
    /// **≥ this value**.  Values outside [0.0, 1.0] are clamped on write.
    public var correctionThreshold: Double {
        didSet { correctionThreshold = min(max(correctionThreshold, 0.0), 1.0) }
    }

    // MARK: - Private state

    private let spellChecker: any SpellCheckProvider
    private let completionProvider: any CompletionProvider
    private var pendingCompletionTask: Task<Void, Never>?

    /// AXObserver registry: one observer per owning PID.
    ///
    /// Stored as raw pointers because `AXObserver` is a Core Foundation object
    /// and cannot be stored directly in an actor without additional bridging.
    /// We keep `Unmanaged` references and balance retain/release manually.
    private var axObservers: [pid_t: AXObserver] = [:]

    // MARK: - Init

    /// Creates a `TextAssistant`.
    ///
    /// - Parameters:
    ///   - enabled:             Initial enabled state.  Defaults to `true`.
    ///   - correctionThreshold: Confidence threshold for inline corrections.
    ///                          Defaults to `0.6`.  Clamped to [0.0, 1.0].
    ///   - spellChecker:        Provider for spelling corrections.
    ///                          Defaults to `NativeSpellCheckProvider`.
    ///   - completionProvider:  Provider for context-aware completions.
    ///                          Defaults to `MemoryBackedCompletionProvider`.
    public init(
        enabled: Bool = true,
        correctionThreshold: Double = 0.6,
        spellChecker: any SpellCheckProvider = NativeSpellCheckProvider(),
        completionProvider: any CompletionProvider = MemoryBackedCompletionProvider()
    ) {
        self.enabled = enabled
        self.correctionThreshold = min(max(correctionThreshold, 0.0), 1.0)
        self.spellChecker = spellChecker
        self.completionProvider = completionProvider
        self.dismissedSuggestions = []
        self.currentSuggestion = nil
    }

    // MARK: - Enable / disable

    /// Enable or disable the Text_Assistant.
    ///
    /// Setting to `false` cancels any pending completion task and clears the
    /// active suggestion.  (Req 16.6)
    public func setEnabled(_ value: Bool) {
        enabled = value
        if !value {
            pendingCompletionTask?.cancel()
            pendingCompletionTask = nil
            currentSuggestion = nil
        }
    }

    // MARK: - Task 19.1 — AX field observation (Req 16.1, 16.6)

    /// Register an `AXUIElement` input field for value-change observation.
    ///
    /// Subscribes to `kAXValueChangedNotification` via an `AXObserver` scoped
    /// to the element's owning process.  When a notification fires,
    /// ``onInput(field:text:)`` is called automatically.
    ///
    /// When `enabled == false`, this method is a no-op.  (Req 16.6)
    ///
    /// - Note: Must be called on the main thread because `CFRunLoopAddSource`
    ///   adds a source to the main run loop.
    public func observeField(_ element: AXUIElement) {
        guard enabled else { return }

        var pid: pid_t = 0
        guard AXUIElementGetPid(element, &pid) == .success else { return }

        // Only one observer per PID.
        guard axObservers[pid] == nil else { return }

        // The AX callback: fired on the main thread whenever the field value
        // changes.  We capture `self` as an unretained pointer so the callback
        // can call back into the actor without creating a retain cycle.
        var newObserver: AXObserver?
        let createStatus = AXObserverCreate(pid, { _, element, _, refcon in
            guard let refcon else { return }
            let assistant = Unmanaged<AnyObject>.fromOpaque(refcon).takeUnretainedValue()
                as! TextAssistant  // safe: we stored self
            var valueRef: CFTypeRef?
            guard AXUIElementCopyAttributeValue(element,
                                                kAXValueAttribute as CFString,
                                                &valueRef) == .success,
                  let text = valueRef as? String else { return }
            Task { _ = await assistant.onInput(field: element, text: text) }
        }, &newObserver)

        guard createStatus == .success, let observer = newObserver else { return }

        let selfPtr = Unmanaged.passUnretained(self as AnyObject).toOpaque()
        AXObserverAddNotification(observer, element, kAXValueChangedNotification as CFString, selfPtr)
        CFRunLoopAddSource(CFRunLoopGetMain(), AXObserverGetRunLoopSource(observer), .defaultMode)

        axObservers[pid] = observer
    }

    /// Unregister an `AXUIElement` previously registered with ``observeField(_:)``.
    ///
    /// Removes the `kAXValueChangedNotification` subscription and removes the
    /// run-loop source.
    ///
    /// - Parameter element: The element to stop observing.
    public func stopObserving(_ element: AXUIElement) {
        var pid: pid_t = 0
        guard AXUIElementGetPid(element, &pid) == .success else { return }
        guard let observer = axObservers[pid] else { return }

        AXObserverRemoveNotification(observer, element, kAXValueChangedNotification as CFString)
        CFRunLoopRemoveSource(CFRunLoopGetMain(), AXObserverGetRunLoopSource(observer), .defaultMode)
        axObservers.removeValue(forKey: pid)
    }

    // MARK: - Task 19.1 — Confidence-gated inline correction (Req 16.1)

    /// Process a text-change event for the given field.
    ///
    /// Behaviour:
    ///   1. Returns `.disabled` immediately when `enabled == false`.  (Req 16.6)
    ///   2. Asks the `SpellCheckProvider` for a correction and confidence.
    ///   3. `confidence == 0.0` → `.noCorrection` (text was already correct).
    ///   4. `confidence >= correctionThreshold` → writes the correction into the
    ///      AX field via `kAXValueAttribute` and returns `.corrected(...)`.
    ///   5. `confidence < correctionThreshold` → returns `.belowThreshold(...)`;
    ///      the field is **not** modified.
    ///
    /// Also cancels any pending completion and schedules a new one (Req 16.2).
    ///
    /// - Parameters:
    ///   - field: The `AXUIElement` representing the input field.
    ///   - text:  The current text content of the field after the change.
    /// - Returns: An `InlineCorrectionResult` describing what happened.
    @discardableResult
    public func onInput(field: AXUIElement, text: String) async -> InlineCorrectionResult {
        guard enabled else { return .disabled }

        // ── Spell-check ──────────────────────────────────────────────────────
        let checkResult = await spellChecker.check(text)

        let correctionResult: InlineCorrectionResult
        if checkResult.confidence <= 0.0 {
            correctionResult = .noCorrection
        } else if checkResult.confidence >= correctionThreshold {
            // Apply correction inline via the Accessibility API.  (Req 16.1)
            AXUIElementSetAttributeValue(field,
                                         kAXValueAttribute as CFString,
                                         checkResult.corrected as CFTypeRef)
            correctionResult = .corrected(
                original: text,
                corrected: checkResult.corrected,
                confidence: checkResult.confidence
            )
        } else {
            correctionResult = .belowThreshold(confidence: checkResult.confidence)
        }

        // ── Schedule completion (Req 16.2) ───────────────────────────────────
        await scheduleCompletion(for: field, text: text)

        return correctionResult
    }

    // MARK: - Task 19.2 — Focus-without-typing path (Req 16.2)

    /// Called when a supported field gains focus without the user typing.
    public func onFocusGained(field: AXUIElement, currentText: String) async {
        guard enabled else { return }
        await scheduleCompletion(for: field, text: currentText)
    }

    // MARK: - Completion scheduling (Req 16.2, 16.3, 16.5, 16.6)

    /// Schedule a context-aware completion after 500 ms of idle.  (Req 16.2)
    public func scheduleCompletion(for field: AXUIElement, text: String) async {
        guard enabled else { return }

        pendingCompletionTask?.cancel()

        let fieldId = axFieldIdentifier(field)
        let inputState = text

        let task = Task { [weak self] in
            guard let self else { return }
            do {
                try await Task.sleep(nanoseconds: 500_000_000)
            } catch {
                return  // Cancelled.
            }
            guard await self.enabled else { return }
            let dismissKey = DismissedKey(fieldId: fieldId, inputState: inputState)
            guard await !self.isDismissed(key: dismissKey) else { return }

            let completion = await self.completionProvider.complete(
                fieldText: inputState,
                memoryContext: ""
            )
            guard let text = completion else { return }
            await self.setCurrentSuggestion(
                CompletionSuggestion(text: text, fieldId: fieldId, inputState: inputState)
            )
        }
        pendingCompletionTask = task
    }

    // MARK: - Task 19.2 — Accept suggestion (Req 16.4)

    /// Insert the current suggestion at the cursor position.  (Req 16.4)
    public func acceptSuggestion(in field: AXUIElement) async {
        guard let suggestion = currentSuggestion else { return }
        insertSuggestion(suggestion, into: field)
        currentSuggestion = nil
    }

    // MARK: - Task 19.2 — Dismiss suggestion (Req 16.5)

    /// Record the current suggestion as dismissed and clear it.  (Req 16.5)
    public func dismissSuggestion() {
        guard let suggestion = currentSuggestion else { return }
        let key = DismissedKey(fieldId: suggestion.fieldId, inputState: suggestion.inputState)
        dismissedSuggestions.insert(key)
        currentSuggestion = nil
    }

    // MARK: - Private helpers

    private func setCurrentSuggestion(_ suggestion: CompletionSuggestion) {
        currentSuggestion = suggestion
    }

    private func isDismissed(key: DismissedKey) -> Bool {
        dismissedSuggestions.contains(key)
    }

    private func axFieldIdentifier(_ field: AXUIElement) -> String {
        var descRef: CFTypeRef?
        let desc: String
        if AXUIElementCopyAttributeValue(field, kAXDescriptionAttribute as CFString, &descRef) == .success,
           let d = descRef as? String, !d.isEmpty {
            desc = d
        } else {
            desc = ""
        }

        var windowTitle = ""
        var parentRef: CFTypeRef?
        if AXUIElementCopyAttributeValue(field, kAXWindowAttribute as CFString, &parentRef) == .success,
           let win = parentRef {
            var titleRef: CFTypeRef?
            if AXUIElementCopyAttributeValue(win as! AXUIElement,
                                              kAXTitleAttribute as CFString,
                                              &titleRef) == .success,
               let t = titleRef as? String {
                windowTitle = t
            }
        }

        let combined = "\(windowTitle)/\(desc)"
        if combined != "/" { return combined }
        // Pointer-based fallback: use CFTypeID as a best-effort stable
        // identifier for the lifetime of this AX object.
        let typeID = CFGetTypeID(field)
        return String(format: "ax-field-type-%lu", typeID)
    }

    private func insertSuggestion(_ suggestion: CompletionSuggestion, into field: AXUIElement) {
        var valueRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(field, kAXValueAttribute as CFString, &valueRef) == .success,
              let currentText = valueRef as? String else { return }

        var rangeRef: CFTypeRef?
        var insertionOffset = (currentText as NSString).length

        if AXUIElementCopyAttributeValue(field, kAXSelectedTextRangeAttribute as CFString, &rangeRef) == .success,
           let rangeCF = rangeRef {
            var nsRange = NSRange()
            if AXValueGetValue(rangeCF as! AXValue, AXValueType.cfRange, &nsRange) {
                insertionOffset = nsRange.location
            }
        }

        let ns = currentText as NSString
        let before = ns.substring(to: min(insertionOffset, ns.length))
        let after  = ns.substring(from: min(insertionOffset, ns.length))
        let newText = before + suggestion.text + after

        AXUIElementSetAttributeValue(field, kAXValueAttribute as CFString, newText as CFString)

        let newCursor = insertionOffset + (suggestion.text as NSString).length
        var newCFRange = CFRange(location: newCursor, length: 0)
        if let axRange = AXValueCreate(AXValueType.cfRange, &newCFRange) {
            AXUIElementSetAttributeValue(field, kAXSelectedTextRangeAttribute as CFString, axRange)
        }
    }
}
