// ClauseSegmenter.swift
// HAKI — Audio Subsystem / TTS Pipeline
//
// Segments a stream of LLM tokens into speakable clauses so TTS synthesis
// can begin on the first clause while subsequent clauses are still generating
// (the "pop-sentences" pattern from the design's Voice Pipeline section).
//
// Segmentation rules (in priority order):
//   1. Hard terminators — `.`, `!`, `?` — always break a clause (after
//      reaching the minimum clause length).
//   2. Soft terminators — `,`, `;`, `:`, `—` (em-dash), `–` (en-dash) —
//      break a clause only after `minimumLength` characters have accumulated.
//   3. Flush — when the stream ends, any remaining buffer is emitted as a
//      final clause regardless of length or punctuation.
//
// Minimum clause length default: 20 characters.
// This avoids submitting single-word fragments like "Hi," to TTS, which
// wastes round-trip overhead and produces unnatural prosody.
//
// Thread safety: `ClauseSegmenter` is a value type (`struct`). Callers own
// the mutable copy; no locking is required.
//
// Design: Voice Pipeline ("sentence/chunk-wise TTS"). Requirements: 3.1
// Phase 1 Task 7.3

import Foundation

// MARK: - ClauseSegmenter

/// Stateful segmenter that consumes one LLM token at a time and emits
/// complete clauses ready for TTS.
///
/// Usage:
/// ```swift
/// var segmenter = ClauseSegmenter()
/// for await token in llmTokenStream {
///     if let clause = segmenter.feed(token) {
///         await tts.synthesize(clause)
///     }
/// }
/// if let final = segmenter.flush() {
///     await tts.synthesize(final)
/// }
/// ```
public struct ClauseSegmenter {

    // MARK: - Configuration

    /// Minimum buffer length (characters) before a soft-terminator can break
    /// a clause.  Hard terminators (`.`, `!`, `?`) always break immediately
    /// (subject to the same minimum so single characters like "Mr." don't
    /// split too early).
    public var minimumLength: Int

    // MARK: - State

    /// Accumulated characters since the last emitted clause.
    private var buffer: String = ""

    // MARK: - Constants

    /// Terminators that always trigger a break (after `minimumLength`).
    private static let hardTerminators: Set<Character> = [".", "!", "?"]

    /// Terminators that trigger a break only after `minimumLength`.
    private static let softTerminators: Set<Character> = [",", ";", ":", "—", "–"]

    // MARK: - Init

    /// Create a segmenter.
    /// - Parameter minimumLength: Minimum number of characters that must
    ///   accumulate before any terminator can trigger a clause break.
    ///   Default: 20.
    public init(minimumLength: Int = 20) {
        self.minimumLength = max(1, minimumLength)
    }

    // MARK: - Public API

    /// Feed a single LLM token into the segmenter.
    ///
    /// - Parameter token: One or more characters produced by the LLM.
    /// - Returns: A complete clause if a break point was detected; `nil` if
    ///   more tokens are needed.
    ///
    /// A token may contain a break character in the middle (e.g. `"end. Start"`).
    /// In that case the function returns the clause up to and including the
    /// break character, and the tail is carried over into the next token's
    /// accumulation. Only one clause is returned per call; feed again
    /// immediately if you need to drain multiple break points in one token.
    public mutating func feed(_ token: String) -> String? {
        buffer.append(token)
        return checkBreak()
    }

    /// Emit whatever remains in the buffer, regardless of length or
    /// punctuation.  Call this when the LLM stream ends (final token seen).
    ///
    /// - Returns: The remaining text if the buffer is non-empty; `nil` if
    ///   there is nothing left.
    public mutating func flush() -> String? {
        let trimmed = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
        buffer = ""
        return trimmed.isEmpty ? nil : trimmed
    }

    /// Reset all state (for reuse across turns).
    public mutating func reset() {
        buffer = ""
    }

    // MARK: - Private helpers

    /// Scan the buffer for the first eligible break point.
    ///
    /// Returns the clause (from buffer start through the break character) and
    /// stores the tail back into `buffer`.  Returns `nil` if no eligible
    /// break is found yet.
    private mutating func checkBreak() -> String? {
        guard buffer.count >= minimumLength else { return nil }

        for (idx, char) in buffer.enumerated() {
            let reachedMinimum = idx >= minimumLength - 1  // 0-based

            if Self.hardTerminators.contains(char), reachedMinimum {
                return extractClause(upThrough: idx)
            }
            if Self.softTerminators.contains(char), reachedMinimum {
                return extractClause(upThrough: idx)
            }
        }

        return nil
    }

    /// Extract buffer[0...idx] as the emitted clause, storing the rest back.
    ///
    /// The emitted string includes the break character so prosody (e.g.
    /// a period) is passed to TTS. Leading/trailing whitespace is trimmed
    /// from the emitted clause; the tail is left-trimmed to remove any space
    /// immediately after the punctuation.
    private mutating func extractClause(upThrough endIndex: Int) -> String {
        let bufferChars = Array(buffer)
        let clauseChars = Array(bufferChars[0...endIndex])
        let tailChars   = endIndex + 1 < bufferChars.count
            ? Array(bufferChars[(endIndex + 1)...])
            : []

        let clause = String(clauseChars).trimmingCharacters(in: .whitespacesAndNewlines)
        // Drop leading whitespace from the tail.
        buffer = String(tailChars).trimmingCharacters(in: .init(charactersIn: " \t"))
        return clause
    }
}
