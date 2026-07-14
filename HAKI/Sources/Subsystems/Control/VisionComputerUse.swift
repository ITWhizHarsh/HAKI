// VisionComputerUse.swift
// HAKI — Control Subsystem
//
// Implements the Vision computer-use fallback for the Mac_Controller.
//
// When no AX element or CDP selector can target a required UI element,
// this subsystem:
//   1. Captures the screen (or a sub-region) via ScreenCaptureKit.
//   2. Runs Vision framework OCR + element detection to locate text that
//      best matches a human-readable element description.
//   3. Synthesizes a CGEvent mouse click at the centre of the located rect.
//
// This fallback is always classified as CONSEQUENTIAL — it synthesises
// physical input events — so the Safety_Gate (on the Core/Python side)
// MUST have requested and received explicit user confirmation before this
// code is invoked.
//
// Security containment:
//   - Only operates on the local screen.
//   - Never transmits screen content off-device.
//   - Requires the Accessibility permission (for CGEvent posting).
//   - Requires the Screen Recording permission (for ScreenCaptureKit capture).
//
// Full implementation: Phase 4 Task 24.2.
// Implements: Req 21.6, 21.12 (vision computer-use loop)

import Foundation
import CoreGraphics
import ScreenCaptureKit
import Vision

// MARK: - VisionElementResult

/// The result of an attempt to locate a UI element on screen using Vision OCR.
public enum VisionElementResult: Sendable, Equatable {
    /// An element was found; the associated rect is in screen coordinates (points).
    case found(CGRect)
    /// No element matching the description could be located.
    case notFound(String)
    /// The screen-capture permission required to locate elements was denied.
    case capturePermissionDenied
}

// MARK: - VisionComputerUseResult

/// The result of a computer-use actuation attempt.
public enum VisionComputerUseResult: Sendable, Equatable {
    /// The operation completed successfully.
    case success
    /// The described element could not be located on screen (Req 21.12).
    case elementNotFound(String)
    /// A required macOS permission was not granted.
    case permissionDenied(String)
    /// A lower-level failure occurred.
    case failed(String)

    public static func == (lhs: VisionComputerUseResult, rhs: VisionComputerUseResult) -> Bool {
        switch (lhs, rhs) {
        case (.success, .success): return true
        case (.elementNotFound(let a), .elementNotFound(let b)): return a == b
        case (.permissionDenied(let a), .permissionDenied(let b)): return a == b
        case (.failed(let a), .failed(let b)): return a == b
        default: return false
        }
    }
}

// MARK: - VisionComputerUseFallback

/// Vision-based computer-use fallback for the Mac_Controller.
///
/// Used when no AX element or CDP selector is available to target a UI
/// element.  Captures the screen via ScreenCaptureKit, runs Vision OCR
/// to locate matching text, then synthesises mouse events via CGEvent.
///
/// This class is always a CONSEQUENTIAL actuator — every use of it
/// synthesises physical input events.  The caller (ExecutionEngine / 
/// Safety_Gate) MUST obtain explicit user confirmation before invoking
/// ``findAndClick(description:)``.
///
/// ## Thread safety
/// All async methods are safe to call from any Swift concurrency context.
/// CGEvent posting is done synchronously on the calling task but does not
/// require the main actor.
///
/// ## Security containment
/// - Screen content captured here is used only locally for OCR matching.
/// - No captured image data is persisted or transmitted off-device.
/// - Screen Recording permission is required; declines gracefully if denied.
public final class VisionComputerUseFallback: @unchecked Sendable {

    // MARK: - Configuration

    /// Minimum OCR confidence to consider a text observation a candidate match.
    public var minimumOCRConfidence: Float = 0.5

    // MARK: - Init

    public init() {}

    // MARK: - Public API

    /// Capture the screen (or a sub-region) and use Vision OCR to locate
    /// an element that best matches `description`.
    ///
    /// The element-matching heuristic performs a case-insensitive substring
    /// search of the OCR observations against `description`.  The best
    /// match is defined as the first observation whose text contains the
    /// description string, converted back to screen-coordinate space.
    ///
    /// - Parameters:
    ///   - description: A human-readable description of the target element
    ///     (e.g. ``"Submit button"`` or ``"Search field"``).
    ///   - region: An optional ``CGRect`` in screen coordinates to restrict
    ///     the capture area.  Pass ``nil`` to capture the primary display.
    /// - Returns: A ``VisionElementResult`` describing the outcome.
    ///
    /// Req 21.12: when the element cannot be found, return `.notFound`.
    public func findElement(
        description: String,
        in region: CGRect? = nil
    ) async -> VisionElementResult {
        // 1. Capture screen (or sub-region) via ScreenCaptureKit.
        guard let image = await captureScreen(region: region) else {
            // Permission denied or capture API failure.
            return .capturePermissionDenied
        }

        // 2. Run Vision OCR to get bounding boxes for each text observation.
        let observations = await recogniseTextObservations(in: image)
        if observations.isEmpty {
            return .notFound(description)
        }

        // 3. Find best matching observation.
        let imageWidth  = CGFloat(image.width)
        let imageHeight = CGFloat(image.height)

        let lowerDescription = description.lowercased()

        for obs in observations {
            guard let candidate = obs.topCandidates(1).first,
                  candidate.confidence >= minimumOCRConfidence
            else { continue }

            if candidate.string.lowercased().contains(lowerDescription) {
                // Convert normalised (0…1) Vision rect to screen points.
                let screenRect = visionRectToScreen(
                    obs.boundingBox,
                    imageWidth: imageWidth,
                    imageHeight: imageHeight,
                    captureRegion: region
                )
                return .found(screenRect)
            }
        }

        return .notFound(description)
    }

    /// Synthesise a left mouse click at the centre of `rect` using CGEvent.
    ///
    /// Posts a ``mouseDown`` immediately followed by a ``mouseUp`` event at
    /// the centre of `rect` in global screen coordinates.
    ///
    /// Requires the macOS Accessibility permission so CGEvent posting is
    /// permitted.  If posting fails, returns ``.failed``.
    ///
    /// - Parameter rect: The bounding rect of the target element in global
    ///   screen coordinates (points).
    /// - Returns: A ``VisionComputerUseResult``.
    public func clickElement(at rect: CGRect) async -> VisionComputerUseResult {
        let center = CGPoint(x: rect.midX, y: rect.midY)
        return await synthesiseClick(at: center)
    }

    /// Locate `description` on screen and click it.
    ///
    /// Combines ``findElement(description:in:)`` and
    /// ``clickElement(at:)`` in a single call.  If the element cannot be
    /// located, returns ``.elementNotFound`` and stops — the caller
    /// (ExecutionEngine) must stop dependent steps and inform the user
    /// which step could not be completed (Req 21.12).
    ///
    /// **This method is always CONSEQUENTIAL.**  It MUST only be called
    /// after the Safety_Gate has obtained explicit user confirmation.
    ///
    /// - Parameter description: Human-readable description of the target
    ///   element.
    /// - Returns: A ``VisionComputerUseResult``.
    ///
    /// Req 21.12: element not locatable → stop dependents, inform which step.
    public func findAndClick(description: String) async -> VisionComputerUseResult {
        let findResult = await findElement(description: description, in: nil)

        switch findResult {
        case .capturePermissionDenied:
            return .permissionDenied(
                "Screen Recording permission is required to locate on-screen elements. "
                + "Grant it in System Settings → Privacy & Security → Screen Recording."
            )

        case .notFound(let desc):
            // Req 21.12: stop dependents, inform which step failed.
            return .elementNotFound(desc)

        case .found(let rect):
            return await clickElement(at: rect)
        }
    }

    // MARK: - Private: Screen capture

    /// Capture the primary display (or the given sub-region) as a ``CGImage``.
    ///
    /// Uses ScreenCaptureKit (macOS 12.3+).  Returns ``nil`` when Screen
    /// Recording permission is denied or the capture fails.
    ///
    /// Security: captured data is used only for local OCR matching and
    /// is never transmitted off-device.
    private func captureScreen(region: CGRect?) async -> CGImage? {
        guard let scContent = try? await SCShareableContent.excludingDesktopWindows(
            false,
            onScreenWindowsOnly: false
        ) else { return nil }

        // Find the primary display to capture.
        guard let display = scContent.displays.first else { return nil }

        let filter = SCContentFilter(display: display, excludingWindows: [])

        let config = SCStreamConfiguration()
        // Use the full display dimensions by default.
        let displayWidth  = Int(display.frame.width)
        let displayHeight = Int(display.frame.height)
        config.width  = displayWidth  > 0 ? displayWidth  : 1920
        config.height = displayHeight > 0 ? displayHeight : 1080
        config.pixelFormat = kCVPixelFormatType_32BGRA
        config.showsCursor = false

        guard let fullImage = try? await SCScreenshotManager.captureImage(
            contentFilter: filter,
            configuration: config
        ) else { return nil }

        // If a sub-region was requested, crop the image.
        if let region = region {
            return cropImage(fullImage, to: region, displayFrame: display.frame)
        }

        return fullImage
    }

    /// Crop `image` to a sub-region expressed in screen coordinates.
    private func cropImage(
        _ image: CGImage,
        to region: CGRect,
        displayFrame: CGRect
    ) -> CGImage? {
        // Scale the screen-coordinate region to the pixel dimensions of the image.
        let scaleX = CGFloat(image.width)  / displayFrame.width
        let scaleY = CGFloat(image.height) / displayFrame.height

        let pixelRegion = CGRect(
            x: (region.minX - displayFrame.minX) * scaleX,
            y: (region.minY - displayFrame.minY) * scaleY,
            width: region.width  * scaleX,
            height: region.height * scaleY
        ).integral

        return image.cropping(to: pixelRegion)
    }

    // MARK: - Private: Vision OCR

    /// Run Vision OCR on `image` and return all text observations.
    private func recogniseTextObservations(
        in image: CGImage
    ) async -> [VNRecognizedTextObservation] {
        return await withCheckedContinuation { continuation in
            let request = VNRecognizeTextRequest { request, error in
                guard error == nil,
                      let results = request.results as? [VNRecognizedTextObservation]
                else {
                    continuation.resume(returning: [])
                    return
                }
                continuation.resume(returning: results)
            }

            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = true
            // Support both Hindi and English UI labels (Req 5).
            request.recognitionLanguages = ["hi-IN", "en-US"]

            let handler = VNImageRequestHandler(cgImage: image, options: [:])
            do {
                try handler.perform([request])
            } catch {
                continuation.resume(returning: [])
            }
        }
    }

    // MARK: - Private: Coordinate conversion

    /// Convert a Vision normalised bounding box (origin at bottom-left,
    /// y increasing upward, unit scale) to screen points (origin at
    /// top-left, y increasing downward, display-pixel scale).
    ///
    /// - Parameters:
    ///   - visionRect: The ``CGRect`` in Vision normalised space.
    ///   - imageWidth:  Width of the captured image in pixels.
    ///   - imageHeight: Height of the captured image in pixels.
    ///   - captureRegion: The capture region in screen coordinates, or ``nil``
    ///     for a full-display capture.
    /// - Returns: The rect in global screen coordinates (points).
    private func visionRectToScreen(
        _ visionRect: CGRect,
        imageWidth: CGFloat,
        imageHeight: CGFloat,
        captureRegion: CGRect?
    ) -> CGRect {
        // Vision coordinate system has (0,0) at bottom-left;
        // macOS screen coordinates have (0,0) at bottom-left of the primary
        // display.  ScreenCaptureKit images have (0,0) at top-left in pixel
        // space.  We need to flip the Vision y-axis to get screen coordinates.

        let origin = captureRegion?.origin ?? .zero
        let totalWidth  = captureRegion?.width  ?? imageWidth
        let totalHeight = captureRegion?.height ?? imageHeight

        let screenX = origin.x + visionRect.minX * totalWidth
        // Flip y: Vision y=0 is at the bottom of the image.
        let screenY = origin.y + (1.0 - visionRect.maxY) * totalHeight
        let screenW = visionRect.width  * totalWidth
        let screenH = visionRect.height * totalHeight

        return CGRect(x: screenX, y: screenY, width: screenW, height: screenH)
    }

    // MARK: - Private: CGEvent mouse synthesis

    /// Post a left mouseDown + mouseUp pair at `point` using CGEvent.
    ///
    /// Requires Accessibility permission.  Falls back to reporting failure
    /// if CGEvent creation returns nil (which typically indicates that the
    /// Accessibility permission is not granted).
    private func synthesiseClick(at point: CGPoint) async -> VisionComputerUseResult {
        // CGEvent posting is synchronous and does not require the main actor.
        guard
            let mouseDown = CGEvent(
                mouseEventSource: nil,
                mouseType: .leftMouseDown,
                mouseCursorPosition: point,
                mouseButton: .left
            ),
            let mouseUp = CGEvent(
                mouseEventSource: nil,
                mouseType: .leftMouseUp,
                mouseCursorPosition: point,
                mouseButton: .left
            )
        else {
            // CGEvent construction failing usually means Accessibility permission denied.
            return .permissionDenied(
                "Accessibility permission is required to synthesise mouse clicks. "
                + "Grant it in System Settings → Privacy & Security → Accessibility."
            )
        }

        mouseDown.post(tap: .cghidEventTap)
        // Brief delay between down and up to ensure the target app registers the click.
        try? await Task.sleep(nanoseconds: 50_000_000) // 50 ms
        mouseUp.post(tap: .cghidEventTap)

        return .success
    }
}

// MARK: - MacController extension: vision-use integration

extension MacController {

    /// The shared ``VisionComputerUseFallback`` instance used when no AX or CDP
    /// selector can reach a required element.
    ///
    /// Creating a fresh instance is cheap; the lazy property is used here for
    /// convenience so callers do not need to instantiate it themselves.
    public var visionFallback: VisionComputerUseFallback {
        VisionComputerUseFallback()
    }

    /// Locate a UI element by description and click it using the Vision
    /// computer-use fallback.
    ///
    /// This convenience method wires together:
    ///   1. ``VisionComputerUseFallback.findAndClick(description:)``
    ///   2. A mapping from the Vision result to the Mac_Controller's
    ///      ``ActuatorResult`` type (defined in ``MacController.swift``).
    ///
    /// The caller MUST have ensured the Safety_Gate confirmed this
    /// CONSEQUENTIAL action before invoking it.
    ///
    /// - Parameter description: Human-readable element description.
    /// - Returns: An ``ActuatorResult``.
    ///
    /// Req 21.12: element not locatable → stop dependents, inform which step.
    public func activateWithVision(description: String) async -> ActuatorResult {
        let result = await visionFallback.findAndClick(description: description)
        switch result {
        case .success:
            return .success
        case .elementNotFound(let desc):
            return .failed(MacControllerError.elementNotFound(desc))
        case .permissionDenied(let msg):
            return .permissionDenied(msg)
        case .failed(let msg):
            return .failed(MacControllerError.visionClickFailed(msg))
        }
    }
}

// MARK: - MacControllerError additions

extension MacControllerError {
    /// The Vision OCR loop could not locate a required on-screen element (Req 21.12).
    static func elementNotFound(_ description: String) -> MacControllerError {
        .axActionFailed("Element not found: \(description)")
    }

    /// The Vision OCR loop located the element but CGEvent click synthesis failed.
    static func visionClickFailed(_ detail: String) -> MacControllerError {
        .axActionFailed("Vision click failed: \(detail)")
    }
}
