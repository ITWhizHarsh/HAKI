// TextAssistantTests.swift
// HAKI — Unit tests for TextAssistant (Task 19.1)
//
// Covers:
//  - onInput: disabled → .disabled immediately (Req 16.6)
//  - onInput: confidence >= threshold → .corrected (Req 16.1)
//  - onInput: confidence < threshold → .belowThreshold, field NOT modified (Req 16.1)
//  - onInput: confidence 0.0 → .noCorrection (Req 16.1)
//  - correctionThreshold: default 0.6
//  - correctionThreshold: clamped to [0.0, 1.0]
//  - SpellCheckResult: confidence clamped to [0.0, 1.0]
//  - NativeSpellCheckProvider: correctly spelled text → confidence 0.0
//  - NativeSpellCheckProvider: misspelled word → confidence 0.0 or 0.9 (env-dependent)
//
// Phase 3 Task 19.1
// Requirements: 16.1, 16.6

#if canImport(XCTest)
import XCTest
@testable import HAKITextInput
import ApplicationServices

// MARK: - Stub SpellCheckProvider

/// Returns a preconfigured ``SpellCheckResult`` for any input.
private struct StubSpellChecker: SpellCheckProvider {
    let result: SpellCheckResult
    func check(_ text: String) async -> SpellCheckResult { result }
}

// MARK: - Synthetic AXUIElement

/// Use the systemwide AX element as a handle in tests — we never actually
/// write to a live field in unit tests (the write silently fails without
/// Accessibility permission, which is fine for testing the return values).
private func syntheticElement() -> AXUIElement {
    AXUIElementCreateSystemWide()
}

// MARK: - TextAssistantTests

final class TextAssistantTests: XCTestCase {

    // MARK: enabled guard (Req 16.6)

    func test_onInput_whenDisabled_returnsDisabled() async {
        let assistant = TextAssistant(
            enabled: false,
            spellChecker: StubSpellChecker(result: SpellCheckResult(corrected: "hello", confidence: 0.9))
        )
        let result = await assistant.onInput(field: syntheticElement(), text: "helo")
        XCTAssertEqual(result, .disabled,
                       "onInput must return .disabled when enabled is false (Req 16.6)")
    }

    func test_enabledByDefault() async {
        let assistant = TextAssistant()
        let isEnabled = await assistant.enabled
        XCTAssertTrue(isEnabled)
    }

    // MARK: correctionThreshold default (Req 16.1)

    func test_correctionThreshold_default_0_6() async {
        let assistant = TextAssistant()
        let threshold = await assistant.correctionThreshold
        XCTAssertEqual(threshold, 0.6, accuracy: 1e-9,
                       "Default correctionThreshold must be 0.6")
    }

    func test_correctionThreshold_clampedBelow() async {
        let assistant = TextAssistant(correctionThreshold: -0.5)
        let threshold = await assistant.correctionThreshold
        XCTAssertEqual(threshold, 0.0, accuracy: 1e-9,
                       "correctionThreshold below 0 should be clamped to 0.0")
    }

    func test_correctionThreshold_clampedAbove() async {
        let assistant = TextAssistant(correctionThreshold: 1.5)
        let threshold = await assistant.correctionThreshold
        XCTAssertEqual(threshold, 1.0, accuracy: 1e-9,
                       "correctionThreshold above 1 should be clamped to 1.0")
    }

    func test_correctionThreshold_exactBoundaries() async {
        let a0 = TextAssistant(correctionThreshold: 0.0)
        XCTAssertEqual(await a0.correctionThreshold, 0.0, accuracy: 1e-9)
        let a1 = TextAssistant(correctionThreshold: 1.0)
        XCTAssertEqual(await a1.correctionThreshold, 1.0, accuracy: 1e-9)
    }

    // MARK: confidence at threshold → .corrected (Req 16.1)

    func test_onInput_confidenceAtThreshold_returnsCorrected() async {
        let threshold = 0.6
        let assistant = TextAssistant(
            correctionThreshold: threshold,
            spellChecker: StubSpellChecker(result: SpellCheckResult(corrected: "hello", confidence: threshold))
        )
        let result = await assistant.onInput(field: syntheticElement(), text: "helo")
        if case .corrected(let original, let corrected, let conf) = result {
            XCTAssertEqual(original, "helo")
            XCTAssertEqual(corrected, "hello")
            XCTAssertEqual(conf, threshold, accuracy: 1e-9)
        } else {
            XCTFail("Expected .corrected at threshold, got \(result)")
        }
    }

    func test_onInput_confidenceAboveThreshold_returnsCorrected() async {
        let assistant = TextAssistant(
            correctionThreshold: 0.6,
            spellChecker: StubSpellChecker(result: SpellCheckResult(corrected: "world", confidence: 0.9))
        )
        let result = await assistant.onInput(field: syntheticElement(), text: "wrold")
        if case .corrected(let original, let corrected, _) = result {
            XCTAssertEqual(original, "wrold")
            XCTAssertEqual(corrected, "world")
        } else {
            XCTFail("Expected .corrected when confidence > threshold, got \(result)")
        }
    }

    // MARK: confidence below threshold → .belowThreshold (Req 16.1)

    func test_onInput_confidenceBelowThreshold_returnsBelowThreshold() async {
        let assistant = TextAssistant(
            correctionThreshold: 0.6,
            spellChecker: StubSpellChecker(result: SpellCheckResult(corrected: "hello", confidence: 0.4))
        )
        let result = await assistant.onInput(field: syntheticElement(), text: "helo")
        if case .belowThreshold(let conf) = result {
            XCTAssertEqual(conf, 0.4, accuracy: 1e-9)
        } else {
            XCTFail("Expected .belowThreshold when confidence < threshold, got \(result)")
        }
    }

    func test_onInput_confidenceJustBelowThreshold_returnsBelowThreshold() async {
        let threshold = 0.6
        let justBelow = threshold - 1e-9
        let assistant = TextAssistant(
            correctionThreshold: threshold,
            spellChecker: StubSpellChecker(result: SpellCheckResult(corrected: "hello", confidence: justBelow))
        )
        let result = await assistant.onInput(field: syntheticElement(), text: "helo")
        if case .belowThreshold(_) = result { /* expected */ } else {
            XCTFail("Expected .belowThreshold just below threshold, got \(result)")
        }
    }

    // MARK: no correction → .noCorrection (Req 16.1)

    func test_onInput_confidence0_returnsNoCorrection() async {
        let assistant = TextAssistant(
            correctionThreshold: 0.6,
            spellChecker: StubSpellChecker(result: SpellCheckResult(corrected: "hello", confidence: 0.0))
        )
        let result = await assistant.onInput(field: syntheticElement(), text: "hello")
        XCTAssertEqual(result, .noCorrection,
                       "Expected .noCorrection when provider returns confidence 0.0")
    }

    // MARK: threshold = 0.0 → any positive confidence corrects

    func test_threshold0_anyPositiveConfidence_corrects() async {
        let assistant = TextAssistant(
            correctionThreshold: 0.0,
            spellChecker: StubSpellChecker(result: SpellCheckResult(corrected: "fixed", confidence: 0.01))
        )
        let result = await assistant.onInput(field: syntheticElement(), text: "fixd")
        if case .corrected(_, let corrected, _) = result {
            XCTAssertEqual(corrected, "fixed")
        } else {
            XCTFail("Expected .corrected with threshold 0.0 and confidence 0.01, got \(result)")
        }
    }

    // MARK: threshold = 1.0 → only confidence exactly 1.0 corrects

    func test_threshold1_confidence099_returnsBelowThreshold() async {
        let assistant = TextAssistant(
            correctionThreshold: 1.0,
            spellChecker: StubSpellChecker(result: SpellCheckResult(corrected: "fixed", confidence: 0.99))
        )
        let result = await assistant.onInput(field: syntheticElement(), text: "fixd")
        if case .belowThreshold(_) = result { /* expected */ } else {
            XCTFail("Expected .belowThreshold with threshold 1.0 and confidence 0.99, got \(result)")
        }
    }

    func test_threshold1_confidence1_corrects() async {
        let assistant = TextAssistant(
            correctionThreshold: 1.0,
            spellChecker: StubSpellChecker(result: SpellCheckResult(corrected: "fixed", confidence: 1.0))
        )
        let result = await assistant.onInput(field: syntheticElement(), text: "fixd")
        if case .corrected(_, let corrected, _) = result {
            XCTAssertEqual(corrected, "fixed")
        } else {
            XCTFail("Expected .corrected with threshold 1.0 and confidence 1.0, got \(result)")
        }
    }

    // MARK: re-enable after disable (Req 16.6)

    func test_disabledThenReenabled_processesInput() async {
        let assistant = TextAssistant(
            enabled: true,
            spellChecker: StubSpellChecker(result: SpellCheckResult(corrected: "hello", confidence: 0.9))
        )
        await assistant.setEnabled(false)
        let r1 = await assistant.onInput(field: syntheticElement(), text: "helo")
        XCTAssertEqual(r1, .disabled)

        await assistant.setEnabled(true)
        let r2 = await assistant.onInput(field: syntheticElement(), text: "helo")
        if case .corrected(_, _, _) = r2 { /* expected */ } else {
            XCTFail("Expected .corrected after re-enabling, got \(r2)")
        }
    }

    // MARK: SpellCheckResult confidence clamping

    func test_spellCheckResult_clampsBelowZero() {
        let r = SpellCheckResult(corrected: "x", confidence: -1.0)
        XCTAssertEqual(r.confidence, 0.0, accuracy: 1e-9)
    }

    func test_spellCheckResult_clampsAboveOne() {
        let r = SpellCheckResult(corrected: "x", confidence: 2.0)
        XCTAssertEqual(r.confidence, 1.0, accuracy: 1e-9)
    }

    func test_spellCheckResult_preservesValidConfidence() {
        let r = SpellCheckResult(corrected: "x", confidence: 0.75)
        XCTAssertEqual(r.confidence, 0.75, accuracy: 1e-9)
    }

    // MARK: NativeSpellCheckProvider

    func test_nativeProvider_correctlySpelled_returnsZeroConfidence() async {
        let provider = NativeSpellCheckProvider()
        let result = await provider.check("hello world")
        XCTAssertEqual(result.confidence, 0.0, accuracy: 1e-9,
                       "Correctly spelled text should return confidence 0.0")
        XCTAssertEqual(result.corrected, "hello world")
    }

    func test_nativeProvider_confidence_isBinaryZeroOrNinety() async {
        let provider = NativeSpellCheckProvider()
        // "teh" is a common misspelling; the result depends on system spell-check state.
        let result = await provider.check("teh quick brown fox")
        XCTAssertTrue(
            result.confidence == 0.0 || result.confidence == 0.9,
            "NativeSpellCheckProvider confidence must be 0.0 or 0.9, got \(result.confidence)"
        )
    }
}
#endif // canImport(XCTest)
