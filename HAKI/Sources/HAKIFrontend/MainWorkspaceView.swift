// MainWorkspaceView.swift
// HAKIFrontend — Primary NavigationSplitView workspace

import SwiftUI

// MARK: - Conversation Model

/// The role of a participant in a conversation turn.
enum ConversationRole: Hashable {
    case user
    case assistant
}

/// A single turn in the conversation timeline.
struct ConversationEntry: Identifiable, Hashable {
    let id: UUID
    let role: ConversationRole
    let text: String
    let timestamp: Date

    init(id: UUID = UUID(), role: ConversationRole, text: String, timestamp: Date) {
        self.id = id
        self.role = role
        self.text = text
        self.timestamp = timestamp
    }

    /// Mock data spanning two distinct calendar days for preview and testing purposes.
    static var mockData: [ConversationEntry] {
        let calendar = Calendar.current
        // Day 1: yesterday
        let yesterday = calendar.date(byAdding: .day, value: -1, to: Date()) ?? Date()
        let yesterdayMorning = calendar.date(bySettingHour: 9, minute: 0, second: 0, of: yesterday) ?? yesterday
        let yesterdayAfternoon = calendar.date(bySettingHour: 14, minute: 30, second: 0, of: yesterday) ?? yesterday
        // Day 2: today
        let todayMorning = calendar.date(bySettingHour: 8, minute: 15, second: 0, of: Date()) ?? Date()

        return [
            ConversationEntry(
                role: .user,
                text: "Hey HAKI, what's on my schedule for today?",
                timestamp: yesterdayMorning
            ),
            ConversationEntry(
                role: .assistant,
                text: "Good morning! You have a team standup at 10 AM and a design review at 3 PM.",
                timestamp: yesterdayAfternoon
            ),
            ConversationEntry(
                role: .user,
                text: "Remind me to send the quarterly report.",
                timestamp: todayMorning
            )
        ]
    }
}

// MARK: - Conversation Row View

/// Extracted row view to help the compiler type-check the List body.
private struct ConversationRowView: View {
    let entry: ConversationEntry
    let onTap: () -> Void

    var body: some View {
        HStack {
            Text(String(entry.text.prefix(40)))
                .lineLimit(1)
            Spacer()
            Text(timeString)
                .foregroundStyle(.secondary)
                .font(.caption)
        }
        .contentShape(Rectangle())
        .onTapGesture(perform: onTap)
    }

    private var timeString: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm"
        return formatter.string(from: entry.timestamp)
    }
}

// MARK: - Main Workspace View

struct MainWorkspaceView: View {
    @Environment(HAKIStateModel.self) private var stateModel
    @State private var conversations: [ConversationEntry] = ConversationEntry.mockData
    @State private var selectedEntry: ConversationEntry?
    @State private var commandText: String = ""
    @State private var columnVisibility = NavigationSplitViewVisibility.all

    // MARK: - Date Formatter

    private static let timeFormatter: DateFormatter = {
        let fmt = DateFormatter()
        fmt.dateFormat = "HH:mm"
        return fmt
    }()

    private static let isoDateFormatter: ISO8601DateFormatter = {
        let fmt = ISO8601DateFormatter()
        fmt.formatOptions = [.withFullDate]
        return fmt
    }()

    // MARK: - Grouping

    /// Conversations grouped by calendar day (year/month/day), sorted most-recent first.
    var groupedConversations: [(key: DateComponents, value: [ConversationEntry])] {
        let grouped = Dictionary(grouping: conversations) { entry in
            Calendar.current.dateComponents([.year, .month, .day], from: entry.timestamp)
        }
        return grouped.sorted { a, b in
            if (a.key.year ?? 0) != (b.key.year ?? 0) {
                return (a.key.year ?? 0) > (b.key.year ?? 0)
            }
            if (a.key.month ?? 0) != (b.key.month ?? 0) {
                return (a.key.month ?? 0) > (b.key.month ?? 0)
            }
            return (a.key.day ?? 0) > (b.key.day ?? 0)
        }
    }

    // MARK: - Body

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            sidebarContent
        } detail: {
            detailContent
        }
        .background(.ultraThinMaterial)
        .preferredColorScheme(nil)
        .overlay(alignment: .top) {
            if stateModel.currentState == .error {
                errorBanner
            }
        }
        .toolbar {
            ToolbarItem(placement: .navigation) {
                Button {
                    withAnimation {
                        columnVisibility = columnVisibility == .all ? .detailOnly : .all
                    }
                } label: {
                    Image(systemName: "sidebar.left")
                }
                .accessibilityLabel("Toggle conversation history sidebar")
                .keyboardShortcut("s", modifiers: .command)
            }
        }
    }

    // MARK: - Sidebar

    @ViewBuilder
    private var sidebarContent: some View {
        if conversations.isEmpty {
            Text("No conversations yet")
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            List(selection: $selectedEntry) {
                ForEach(groupedConversations, id: \.key) { group in
                    Section(header: Text(sectionTitle(for: group.key))) {
                        ForEach(group.value) { entry in
                            ConversationRowView(
                                entry: entry,
                                onTap: { selectedEntry = entry }
                            )
                        }
                    }
                }
            }
        }
    }

    /// Derive an ISO 8601 date string (e.g. "2024-06-01") from `DateComponents`.
    private func sectionTitle(for components: DateComponents) -> String {
        guard let date = Calendar.current.date(from: components) else {
            return "\(components.year ?? 0)-\(components.month ?? 0)-\(components.day ?? 0)"
        }
        return Self.isoDateFormatter.string(from: date)
    }

    // MARK: - Detail

    @ViewBuilder
    private var detailContent: some View {
        VStack(spacing: 0) {
            // Task 5.4: JARVISParticleView
            JARVISParticleView(audioLevel: Binding(
                get: { stateModel.audioLevel },
                set: { _ in }
            ))
            .frame(height: 240)
            .frame(maxWidth: .infinity)
            .accessibilityLabel("HAKI status: \(stateModel.currentState)")
            .accessibilityHidden(false)

            // Task 5.4: Conversation timeline
            ScrollView {
                LazyVStack(spacing: 12) {
                    ForEach(conversations) { entry in
                        messageBubble(for: entry)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
            }

            // Task 5.5: floating command bar (stub)
            commandBar
        }
    }

    @ViewBuilder
    private func messageBubble(for entry: ConversationEntry) -> some View {
        let isUser = entry.role == .user
        HStack {
            if isUser { Spacer(minLength: 60) }
            Text(entry.text)
                .foregroundStyle(isUser ? Color.primary : stateModel.currentState.accentColor)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
            if !isUser { Spacer(minLength: 60) }
        }
        .frame(maxWidth: .infinity, alignment: isUser ? .trailing : .leading)
    }

    @State private var isDropTargeted: Bool = false

    @ViewBuilder
    private var commandBar: some View {
        ZStack {
            HStack(spacing: 12) {
                // Text input
                TextField("Ask HAKI…", text: $commandText)
                    .textFieldStyle(.plain)

                // File attachment button
                Button {
                    // File attachment action (handled via onDrop)
                } label: {
                    Image(systemName: "paperclip")
                        .foregroundStyle(isDropTargeted ? Color.accentColor : Color.secondary)
                }
                .buttonStyle(.plain)
                .onDrop(of: [.fileURL], isTargeted: $isDropTargeted) { providers in
                    // Handle dropped file URLs
                    _ = providers
                    return true
                }
                .accessibilityLabel("Attach file")

                // Live audio waveform
                WaveformView(level: stateModel.audioLevel)

                // Microphone toggle button
                Button {
                    if stateModel.currentState == .listening {
                        stateModel.currentState = .idle
                    } else {
                        stateModel.currentState = .listening
                    }
                } label: {
                    Image(systemName: stateModel.currentState == .listening ? "mic.fill" : "mic")
                        .foregroundStyle(stateModel.currentState == .listening ? .green : .secondary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Toggle microphone")
                .accessibilityHint("Activates or deactivates voice listening mode")
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
        }
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20))
        .shadow(color: .black.opacity(0.15), radius: 8, y: 4)
        .padding(.horizontal, 16)
        .padding(.bottom, 16)
    }

    // MARK: - Error Banner (task 5.6 stub)

    @ViewBuilder
    private var errorBanner: some View {
        ZStack {
            Color.red.opacity(0.15)
            Text("HAKI encountered an error")
                .foregroundStyle(.red)
                .padding(.vertical, 8)
        }
        .frame(maxWidth: .infinity)
        .fixedSize(horizontal: false, vertical: true)
    }
}

// MARK: - WaveformView

/// A live audio waveform visualiser consisting of capsule-shaped bars
/// whose heights scale proportionally with `audioLevel` (0.0 – 1.0).
struct WaveformView: View {
    let level: Float

    /// Per-bar height multipliers to create a natural waveform silhouette.
    private let multipliers: [CGFloat] = [0.4, 0.7, 1.0, 0.7, 0.4, 0.6, 0.85]

    private let minBarHeight: CGFloat = 4
    private let maxBarHeight: CGFloat = 24

    var body: some View {
        HStack(alignment: .center, spacing: 3) {
            ForEach(multipliers.indices, id: \.self) { index in
                Capsule()
                    .fill(Color.accentColor.opacity(level > 0 ? 0.85 : 0.3))
                    .frame(
                        width: 3,
                        height: barHeight(for: index)
                    )
                    .animation(.easeOut(duration: 0.08), value: level)
            }
        }
        .accessibilityHidden(true)
    }

    private func barHeight(for index: Int) -> CGFloat {
        let multiplier = multipliers[index]
        let scaledLevel = CGFloat(level)
        let height = minBarHeight + (maxBarHeight - minBarHeight) * scaledLevel * multiplier
        return max(minBarHeight, height)
    }
}
