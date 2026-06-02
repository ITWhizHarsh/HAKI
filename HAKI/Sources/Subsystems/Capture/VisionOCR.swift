// VisionOCR.swift
// HAKI — Capture Subsystem
//
// Wraps the Vision framework's VNRecognizeTextRequest to perform OCR on
// a CGImage captured via ScreenCaptureKit.
//
// Full implementation: Phase 3 Task 18.1
// Implements: Req 1.3, 1.4 (OCR fallback)

import Foundation
import Vision

// MARK: - VisionOCR

/// Performs on-device OCR using Apple's Vision framework.
public struct VisionOCR: Sendable {

    /// Minimum confidence threshold for a recognised text observation.
    public var minimumConfidence: Float = 0.5

    public init() {}

    /// Extract text from a CGImage, returning `nil` if nothing was recognised.
    public func recogniseText(in image: CGImage) async -> String? {
        return await withCheckedContinuation { continuation in
            let request = VNRecognizeTextRequest { request, error in
                guard error == nil,
                      let observations = request.results as? [VNRecognizedTextObservation]
                else {
                    continuation.resume(returning: nil)
                    return
                }

                let text = observations
                    .compactMap { $0.topCandidates(1).first }
                    .filter { $0.confidence >= self.minimumConfidence }
                    .map { $0.string }
                    .joined(separator: "\n")

                continuation.resume(returning: text.isEmpty ? nil : text)
            }

            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = true
            // Prefer Hindi and English scripts (Req 5)
            request.recognitionLanguages = ["hi-IN", "en-US"]

            let handler = VNImageRequestHandler(cgImage: image, options: [:])
            do {
                try handler.perform([request])
            } catch {
                continuation.resume(returning: nil)
            }
        }
    }
}
