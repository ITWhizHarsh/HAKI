// AutomationProgressView.swift
// HAKI — UI Subsystem / Automation_Library
//
// SwiftUI views for displaying step-by-step automation progress (Req 17.5).
//
// Components:
//  - `AutomationProgressPanel`  — the full progress panel for a running automation.
//  - `AutomationStepRow`        — a single step row with status icon and label.
//
// Routing to the UI:
//  The IPC inbound handler calls `UIState.postAutomationProgress(_:)` whenever
//  a `ServerMessage.automationProgress` arrives.  `UIState.automationProgressEvents`
//  is `@Published` so views observing `UIState` refresh automatically.
//
// Implements: Req 17.5 (report currently executing step), 17.6 (cancel support),
//             17.7 (failure propagation report).
// Design: Automation_Library + Execution_Engine, Intent Routing.

import SwiftUI
import HAKIIPC

// MARK: - AutomationProgressPanel

/// Full progress panel for the currently-running (or most-recently-run) automation.
///
/// Shows each step event as a row, colour-coded by status.
/// Observes `UIState.automationProgressEvents` so updates arrive automatically.
public struct AutomationProgressPanel: View {

    @ObservedObject private var uiState: UIState
    /// Optional callback invoked when the user taps "Cancel".
    public var onCancel: (() -> Void)?

    public init(uiState: UIState, onCancel: (() -> Void)? = nil) {
        self.uiState = uiState
        self.onCancel = onCancel
    }

    @MainActor
    public init(onCancel: (() -> Void)? = nil) {
        self.uiState = UIState.shared
        self.onCancel = onCancel
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                Label("Automation", systemImage: "gearshape.2.fill")
                    .font(.headline)
                Spacer()
                if uiState.currentAutomationName != nil {
                    Button("Cancel") {
                        onCancel?()
                    }
                    .font(.caption)
                    .buttonStyle(.plain)
                    .foregroundColor(.red)
                    .help("Cancel the running automation")
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)

            if let name = uiState.currentAutomationName ?? uiState.automationProgressEvents.first?.automationName {
                Text(name)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.horizontal)
                    .padding(.bottom, 4)
            }

            Divider()

            if uiState.automationProgressEvents.isEmpty {
                emptyState
            } else {
                stepList
            }
        }
        .frame(minWidth: 280, maxWidth: .infinity)
    }

    // MARK: Private

    private var emptyState: some View {
        HStack {
            Spacer()
            VStack(spacing: 6) {
                Image(systemName: "gearshape")
                    .font(.title2)
                    .foregroundColor(.secondary)
                Text("No automation running")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            Spacer()
        }
        .padding(.vertical, 16)
    }

    private var stepList: some View {
        ScrollView {
            LazyVStack(spacing: 4) {
                ForEach(Array(uiState.automationProgressEvents.enumerated()), id: \.offset) { _, event in
                    AutomationStepRow(event: event)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 6)
        }
        .frame(maxHeight: 200)
    }
}

// MARK: - AutomationStepRow

/// A single automation step row with status icon and label.
public struct AutomationStepRow: View {

    public let event: HAKIAutomationProgress

    public init(event: HAKIAutomationProgress) {
        self.event = event
    }

    public var body: some View {
        HStack(spacing: 8) {
            statusIcon
                .frame(width: 18)

            VStack(alignment: .leading, spacing: 1) {
                Text(stepLabel)
                    .font(.caption)
                    .fontWeight(event.status == "started" ? .semibold : .regular)
                    .foregroundColor(stepLabelColor)
                if !event.message.isEmpty {
                    Text(event.message)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                }
            }

            Spacer()
        }
        .padding(.vertical, 3)
    }

    // MARK: Private

    private var stepLabel: String {
        switch event.status {
        case "plan_complete": return "✓ Automation complete"
        case "not_found":     return "✕ \(event.automationName) not found"
        default:              return event.step == "(none)" ? event.automationName : event.step
        }
    }

    private var stepLabelColor: Color {
        switch event.status {
        case "completed", "plan_complete": return .primary
        case "failed", "not_found":       return .red
        case "started":                   return .accentColor
        default:                          return .secondary
        }
    }

    @ViewBuilder
    private var statusIcon: some View {
        switch event.status {
        case "started":
            ProgressView()
                .scaleEffect(0.6)
                .progressViewStyle(.circular)
        case "completed":
            Image(systemName: "checkmark.circle.fill")
                .foregroundColor(.green)
                .font(.caption)
        case "plan_complete":
            Image(systemName: "checkmark.seal.fill")
                .foregroundColor(.green)
                .font(.caption)
        case "failed", "not_found":
            Image(systemName: "xmark.circle.fill")
                .foregroundColor(.red)
                .font(.caption)
        default:
            Image(systemName: "circle")
                .foregroundColor(.secondary)
                .font(.caption)
        }
    }
}
