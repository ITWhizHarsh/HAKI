// TTSServiceTests.swift
// HAKITests — Unit Tests for Task 7.3
//
// Tests for:
//   • ClauseSegmenter — token-to-clause segmentation logic
//   • TTSService / MockTTSService — streaming TTS pipeline
//   • VoiceEngine.speak(textStream:) wiring
//
// Requirements: 3.1, 3.5, 3.7
// Phase 1 Task 7.3

#if canImport(XCTest)
import XCTest
@testable import HAKIAudio

final class ClauseSegmenterTests: XCTestCase {

    // MARK: - Basic segmentation

    func testHardTerminatorBreaksAtMinimumLength() {
        var seg = ClauseSegmenter(minimumLength: 5)
        // Feed up to the point where "Hello." is exactly 6 chars
        let result = seg.feed("Hello.")
        XCTAssertNotNil(result, "A hard terminator should emit a clause once minimumLength is met.")
        XCTAssertEqual(result, "Hello.")
    }

    func testHardTerminatorDoesNotBreakBelowMinimumLength() {
        var seg = ClauseSegmenter(minimumLength: 20)
        // "Hi." is only 3 chars — below the 20-char minimum
        let result = seg.feed("Hi.")
        XCTAssertNil(result, "Should not break below minimumLength even with a hard terminator.")
    }

    func testSentenceBreakOnPeriod() {
        var seg = ClauseSegmenter(minimumLength: 10)
        // Accumulate enough characters
        _ = seg.feed("Hello world")   // 11 chars, no punctuation → nil
        let result = seg.feed(".")    // period after ≥10 chars → emit
        XCTAssertNotNil(result)
        XCTAssertTrue(result?.hasSuffix(".") == true)
    }

    func testSentenceBreakOnExclamation() {
        var seg = ClauseSegmenter(minimumLength: 10)
        _ = seg.feed("Great news")
        let result = seg.feed("!")
        XCTAssertNotNil(result)
        XCTAssertTrue(result?.hasSuffix("!") == true)
    }

    func testSentenceBreakOnQuestion() {
        var seg = ClauseSegmenter(minimumLength: 10)
        _ = seg.feed("How are you")
        let result = seg.feed("?")
        XCTAssertNotNil(result)
        XCTAssertTrue(result?.hasSuffix("?") == true)
    }

    func testSoftTerminatorBreaksAtMinimumLength() {
        var seg = ClauseSegmenter(minimumLength: 5)
        _ = seg.feed("Hello")  // exactly 5 chars
        let result = seg.feed(",") // soft terminator at minimumLength → emit
        XCTAssertNotNil(result, "Soft terminator (comma) should break at or after minimumLength.")
    }

    func testSoftTerminatorDoesNotBreakBelowMinimumLength() {
        var seg = ClauseSegmenter(minimumLength: 20)
        // "Hi," is only 3 chars
        let result = seg.feed("Hi,")
        XCTAssertNil(result, "Soft terminator should not break before minimumLength.")
    }

    func testFlushEmitsRemainder() {
        var seg = ClauseSegmenter(minimumLength: 50)
        _ = seg.feed("This sentence has no punctuation at all")
        let result = seg.flush()
        XCTAssertNotNil(result, "flush() should emit remaining buffer even without punctuation.")
        XCTAssertEqual(result, "This sentence has no punctuation at all")
    }

    func testFlushReturnsNilOnEmptyBuffer() {
        var seg = ClauseSegmenter()
        let result = seg.flush()
        XCTAssertNil(result, "flush() should return nil when the buffer is empty.")
    }

    func testMultipleClausesFromTokens() {
        var seg = ClauseSegmenter(minimumLength: 5)
        var clauses: [String] = []

        let tokens = ["Hello", " world", ".", " Goodbye", " world", "!"]
        for token in tokens {
            if let clause = seg.feed(token) {
                clauses.append(clause)
            }
        }
        if let final_ = seg.flush() {
            clauses.append(final_)
        }

        XCTAssertEqual(clauses.count, 2, "Should produce exactly two clauses.")
        XCTAssertTrue(clauses[0].hasSuffix("."))
        XCTAssertTrue(clauses[1].hasSuffix("!"))
    }

    func testTailCarriesOverAfterBreak() {
        var seg = ClauseSegmenter(minimumLength: 5)
        // Feed a token that contains a break mid-token
        _ = seg.feed("Hello. ")
        // Now feed more — the tail "Hello. " should have been trimmed to ""
        // then the next token starts a new clause
        _ = seg.feed("World")
        let result = seg.flush()
        XCTAssertEqual(result, "World", "Text after the break should be in the next clause.")
    }

    func testResetClearsBuffer() {
        var seg = ClauseSegmenter(minimumLength: 50)
        _ = seg.feed("Some accumulated text")
        seg.reset()
        let result = seg.flush()
        XCTAssertNil(result, "After reset(), flush() should return nil.")
    }

    func testDefaultMinimumLengthIs20() {
        let seg = ClauseSegmenter()
        XCTAssertEqual(seg.minimumLength, 20)
    }

    // MARK: - Edge cases

    func testEmptyTokenDoesNotCrash() {
        var seg = ClauseSegmenter()
        let result = seg.feed("")
        XCTAssertNil(result, "Empty token should produce no clause.")
    }

    func testSingleCharacterMinimumIsRespected() {
        var seg = ClauseSegmenter(minimumLength: 1)
        let result = seg.feed(".")
        XCTAssertNotNil(result, "With minimumLength=1, a period alone should emit a clause.")
    }

    func testMinimumLengthClampedToOne() {
        let seg = ClauseSegmenter(minimumLength: 0)
        XCTAssertEqual(seg.minimumLength, 1, "minimumLength should be clamped to at least 1.")
    }
}

// MARK: - MockTTSService Tests

final class MockTTSServiceTests: XCTestCase {

    func testSpeakConsumesStreamAndInvokesCallbacks() async {
        let mock = MockTTSService()
        let tokens = ["Hello", " world", ".", " How", " are", " you", "?"]
        let stream = AsyncStream<String> { cont in
            for t in tokens { cont.yield(t) }
            cont.finish()
        }
        let (bargeInStream, cont) = AsyncStream<VoiceEvent>.makeStream()
        cont.finish()

        var startedCalled = false
        var stoppedCalled = false

        await mock.speak(
            textStream: stream,
            voiceEvents: bargeInStream,
            onStarted: { startedCalled = true },
            onStopped: { stoppedCalled = true }
        )

        XCTAssertTrue(mock.didCallSpeak)
        XCTAssertTrue(startedCalled, "onStarted should be called")
        XCTAssertTrue(stoppedCalled, "onStopped should be called")
        XCTAssertFalse(mock.recordedClauses.isEmpty, "At least one clause should have been recorded")
    }

    func testSimulateTTSFailurePostsNotification() async {
        let mock = MockTTSService()
        mock.simulateTTSFailure = true

        let expectation = expectation(description: "ttsFailedShowText notification received")
        let observer = NotificationCenter.default.addObserver(
            forName: Notification.Name("haki.ttsFailedShowText"),
            object: nil,
            queue: .main
        ) { notification in
            let text = notification.userInfo?["responseText"] as? String ?? ""
            XCTAssertFalse(text.isEmpty, "Response text in notification should not be empty.")
            expectation.fulfill()
        }
        defer { NotificationCenter.default.removeObserver(observer) }

        let tokens = ["TTS", " should", " fail", " here", " because", " we", " said", " so"]
        let stream = AsyncStream<String> { cont in
            for t in tokens { cont.yield(t) }
            cont.finish()
        }
        let (bargeInStream, bCont) = AsyncStream<VoiceEvent>.makeStream()
        bCont.finish()

        await mock.speak(
            textStream: stream,
            voiceEvents: bargeInStream,
            onStarted: {},
            onStopped: {}
        )

        await fulfillment(of: [expectation], timeout: 2.0)
    }

    func testCancelSetsCancelledFlag() {
        let mock = MockTTSService()
        mock.cancel()
        XCTAssertTrue(mock.didCallCancel)
    }
}

// MARK: - VoiceEngine speak() wiring tests

final class VoiceEngineSpeakTests: XCTestCase {

    func testSpeakCallsTTSServiceAndNotifiesVAD() async throws {
        let mockAudio = MockAudioEngine()
        let mockSTT = MockSTTService()
        let mockTTS = MockTTSService()
        let engine = VoiceEngine(
            audioEngine: mockAudio,
            sttService: mockSTT,
            ttsService: mockTTS
        )

        // Start listening so the engine has an event stream ready.
        let _ = try engine.listen()

        let tokens = ["Hello", " there", ",", " this", " is", " a", " test", " of", " TTS", "!"]
        let tokenStream = AsyncStream<String> { cont in
            for t in tokens { cont.yield(t) }
            cont.finish()
        }

        try await engine.speak(textStream: tokenStream)

        XCTAssertTrue(mockTTS.didCallSpeak, "speak() on VoiceEngine should delegate to TTSService.")
    }

    func testBargeInStopCancelsTTSService() {
        let mockAudio = MockAudioEngine()
        let mockSTT = MockSTTService()
        let mockTTS = MockTTSService()
        let engine = VoiceEngine(
            audioEngine: mockAudio,
            sttService: mockSTT,
            ttsService: mockTTS
        )

        engine.bargeInStop()
        XCTAssertTrue(mockTTS.didCallCancel, "bargeInStop() should cancel the TTS service.")
    }

    func testTTSFailureSendsNotification() async throws {
        let mockAudio = MockAudioEngine()
        let mockSTT = MockSTTService()
        let mockTTS = MockTTSService()
        mockTTS.simulateTTSFailure = true

        let engine = VoiceEngine(
            audioEngine: mockAudio,
            sttService: mockSTT,
            ttsService: mockTTS
        )

        let _ = try engine.listen()

        let expectation = expectation(description: "TTS failure notification")
        let observer = NotificationCenter.default.addObserver(
            forName: Notification.Name("haki.ttsFailedShowText"),
            object: nil,
            queue: .main
        ) { _ in
            expectation.fulfill()
        }
        defer { NotificationCenter.default.removeObserver(observer) }

        let tokens = ["This", " should", " fail", " gracefully", " and", " show", " text"]
        let stream = AsyncStream<String> { cont in
            for t in tokens { cont.yield(t) }
            cont.finish()
        }

        try await engine.speak(textStream: stream)
        await fulfillment(of: [expectation], timeout: 2.0)
    }
}
#endif // canImport(XCTest)
