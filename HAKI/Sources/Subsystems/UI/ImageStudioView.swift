// ImageStudioView.swift
// HAKI — UI Subsystem / Image_Studio
//
// SwiftUI views for displaying and browsing session images generated or
// edited by the Image_Studio (Req 15.1, 15.2, 15.3, 15.4, 15.5, 15.6).
//
// Components:
//  - `ImageStudioPanel`  — the full scrollable gallery of session images.
//  - `ImageStudioCard`   — a single image cell with label, save status, and
//                          reveal-in-Finder button.
//  - `ImageStudioInline` — a compact single-image view for embedding inside
//                          the chat panel alongside a text response.
//
// Routing to the UI:
//  The IPC inbound handler calls `UIState.postImageResponse(_:)` whenever a
//  `ServerMessage.imageResponse` arrives.  `UIState.shared.sessionImages`
//  is `@Published` so views that observe `UIState` automatically refresh.
//
// Implements: Req 15.1 (display generated image), 15.4 (confirm save),
//             15.5 (inform of save failure), 15.6 (inform of generation failure).
// Design: Image_Studio.

import SwiftUI
import HAKIIPC

// MARK: - ImageStudioPanel

/// Scrollable gallery of all session images, newest at the top.
///
/// Embed in the HAKI menu-bar popover or a dedicated panel window.
/// Observes `UIState.sessionImages` so updates arrive automatically.
public struct ImageStudioPanel: View {

    @ObservedObject private var uiState: UIState

    public init(uiState: UIState) {
        self.uiState = uiState
    }

    /// Convenience initializer that uses `UIState.shared` on the main actor.
    @MainActor
    public init() {
        self.uiState = UIState.shared
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header row
            HStack {
                Label("Images", systemImage: "photo.on.rectangle.angled")
                    .font(.headline)
                Spacer()
                if !uiState.sessionImages.isEmpty {
                    Button(role: .destructive) {
                        uiState.clearSessionImages()
                    } label: {
                        Image(systemName: "trash")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Clear session images")
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)

            Divider()

            if uiState.sessionImages.isEmpty {
                emptyState
            } else {
                imageGallery
            }
        }
        .frame(minWidth: 300, maxWidth: .infinity)
    }

    // MARK: Private

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "photo.badge.plus")
                .font(.largeTitle)
                .foregroundColor(.secondary)
            Text("No images yet")
                .foregroundColor(.secondary)
            Text("Ask HAKI to generate an image.")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    private var imageGallery: some View {
        ScrollView {
            LazyVStack(spacing: 12) {
                    // Newest first — copy to Array first so ForEach gets a RandomAccessCollection
                    let reversed = Array(uiState.sessionImages.reversed())
                    ForEach(reversed, id: \.imageId) { response in
                        ImageStudioCard(response: response)
                    }
                }
            .padding()
        }
    }
}

// MARK: - ImageStudioCard

/// A single session-image cell showing the image, its label, save status,
/// and a button to reveal the saved file in Finder.
///
/// - When `response.success` is false, shows an error badge (Req 15.6).
/// - When `response.savedPath` is nil, shows a "not saved" badge (Req 15.5).
/// - When `response.savedPath` is set, shows a "Saved" badge (Req 15.4).
public struct ImageStudioCard: View {

    public let response: HAKIImageResponse

    public init(response: HAKIImageResponse) {
        self.response = response
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // Image or failure placeholder
            imageView
                .frame(maxWidth: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 8))

            // Label row
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(response.displayLabel)
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Text(response.message)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                }
                Spacer()
                saveStatusBadge
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(NSColor.controlBackgroundColor))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color(NSColor.separatorColor), lineWidth: 0.5)
        )
    }

    // MARK: Private

    @ViewBuilder
    private var imageView: some View {
        if !response.imageData.isEmpty,
           let nsImage = NSImage(data: response.imageData) {
            Image(nsImage: nsImage)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(maxHeight: 240)
        } else if let path = response.savedPath,
                  let nsImage = NSImage(contentsOfFile: path) {
            Image(nsImage: nsImage)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(maxHeight: 240)
        } else {
            // Failure or no data
            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(NSColor.unemphasizedSelectedContentBackgroundColor))
                    .frame(height: 120)
                if response.success {
                    ProgressView()
                } else {
                    Label("Could not generate image", systemImage: "exclamationmark.triangle")
                        .foregroundColor(.red)
                        .font(.caption)
                }
            }
        }
    }

    @ViewBuilder
    private var saveStatusBadge: some View {
        if !response.success {
            // Generation failed (Req 15.6)
            Label("Failed", systemImage: "xmark.circle.fill")
                .labelStyle(.iconOnly)
                .foregroundColor(.red)
                .help(response.message)
        } else if let savedPath = response.savedPath {
            // Saved successfully — tap to reveal in Finder (Req 15.4)
            Button {
                NSWorkspace.shared.selectFile(savedPath, inFileViewerRootedAtPath: "")
            } label: {
                Label("Saved", systemImage: "checkmark.circle.fill")
                    .labelStyle(.titleAndIcon)
                    .font(.caption)
                    .foregroundColor(.green)
            }
            .buttonStyle(.plain)
            .help("Reveal in Finder: \(savedPath)")
        } else {
            // Save failed but image in session (Req 15.5)
            Label("Not saved", systemImage: "exclamationmark.circle.fill")
                .labelStyle(.titleAndIcon)
                .font(.caption)
                .foregroundColor(.orange)
                .help("Image could not be saved to folder but is available in this session.")
        }
    }
}

// MARK: - ImageStudioInline

/// Compact single-image view for embedding inside the chat message list.
///
/// Shows the image thumbnail at a reduced size with the display label below.
/// Tapping the image opens the full-size preview via Quick Look.
public struct ImageStudioInline: View {

    public let response: HAKIImageResponse
    @State private var showFullScreen = false

    public init(response: HAKIImageResponse) {
        self.response = response
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            imageThumbnail
                .onTapGesture {
                    // Reveal the saved file in Finder (closest Quick Look equivalent
                    // without requiring QuickLookUI framework in SwiftPM context).
                    if let path = response.savedPath {
                        NSWorkspace.shared.selectFile(path, inFileViewerRootedAtPath: "")
                    }
                    showFullScreen = true
                }
                .help("Tap to view in Finder")

            HStack(spacing: 4) {
                Text(response.displayLabel)
                    .font(.caption)
                    .foregroundColor(.secondary)
                if response.savedPath != nil {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.caption2)
                        .foregroundColor(.green)
                } else if !response.success {
                    Image(systemName: "xmark.circle.fill")
                        .font(.caption2)
                        .foregroundColor(.red)
                } else {
                    Image(systemName: "exclamationmark.circle.fill")
                        .font(.caption2)
                        .foregroundColor(.orange)
                }
            }
        }
    }

    // MARK: Private

    private var savedURL: URL? {
        guard let path = response.savedPath else { return nil }
        return URL(fileURLWithPath: path)
    }

    @ViewBuilder
    private var imageThumbnail: some View {
        if !response.imageData.isEmpty,
           let nsImage = NSImage(data: response.imageData) {
            Image(nsImage: nsImage)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(maxWidth: 200, maxHeight: 150)
                .clipShape(RoundedRectangle(cornerRadius: 6))
        } else if let path = response.savedPath,
                  let nsImage = NSImage(contentsOfFile: path) {
            Image(nsImage: nsImage)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(maxWidth: 200, maxHeight: 150)
                .clipShape(RoundedRectangle(cornerRadius: 6))
        } else {
            RoundedRectangle(cornerRadius: 6)
                .fill(Color.secondary.opacity(0.2))
                .frame(width: 200, height: 120)
                .overlay(
                    Image(systemName: "photo")
                        .foregroundColor(.secondary)
                )
        }
    }
}

// MARK: - Previews
// Note: #Preview macro requires Xcode — use SwiftUI_Previews via PreviewProvider for SPM builds.
