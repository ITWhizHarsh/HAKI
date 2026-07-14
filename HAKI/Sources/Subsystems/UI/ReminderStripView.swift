// ReminderStripView.swift
// HAKI — UI Subsystem / Scheduler
//
// SwiftUI views for in-app reminder notifications (Req 12.6).
//
// Components:
//  - `ReminderStrip`  — compact vertical list of recent reminder banners.
//  - `ReminderBanner` — a single reminder banner with severity colour coding.
//
// Routing to the UI:
//  The IPC inbound handler calls `UIState.postReminder(_:)` whenever a
//  `ServerMessage.reminderFired` arrives.  `UIState.recentReminders` is
//  `@Published` so views observing `UIState` refresh automatically.
//
// System notification channel:
//  The Scheduler in Core is responsible for issuing the system UNUserNotification
//  (Req 12.6 — dual-channel: VOICE + NOTIFICATION).  This file handles the
//  in-app visual representation only; the `NotificationManager` (OSActions)
//  handles the system notification.
//
// Implements: Req 12.6 (in-app notification channel), 12.9 (partial failure OK).
// Design: Scheduler, Intent Routing.

import SwiftUI
import HAKIIPC

// MARK: - ReminderStrip

/// Compact vertical list of recent reminder notifications.
///
/// Embed in the HAKI menu-bar popover alongside the chat panel.
/// Observes `UIState.recentReminders` so updates arrive automatically.
public struct ReminderStrip: View {

    @ObservedObject private var uiState: UIState

    public init(uiState: UIState) {
        self.uiState = uiState
    }

    @MainActor
    public init() {
        self.uiState = UIState.shared
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                Label("Reminders", systemImage: "bell.badge.fill")
                    .font(.headline)
                Spacer()
                if !uiState.recentReminders.isEmpty {
                    Button(role: .destructive) {
                        uiState.clearReminders()
                    } label: {
                        Image(systemName: "xmark.circle")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Dismiss all reminders")
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)

            Divider()

            if uiState.recentReminders.isEmpty {
                emptyState
            } else {
                reminderList
            }
        }
        .frame(minWidth: 280, maxWidth: .infinity)
    }

    // MARK: Private

    private var emptyState: some View {
        HStack {
            Spacer()
            Text("No recent reminders")
                .font(.caption)
                .foregroundColor(.secondary)
            Spacer()
        }
        .padding(.vertical, 12)
    }

    private var reminderList: some View {
        ScrollView {
            LazyVStack(spacing: 6) {
                ForEach(uiState.recentReminders, id: \.reminderId) { reminder in
                    ReminderBanner(reminder: reminder)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 6)
        }
        .frame(maxHeight: 200)
    }
}

// MARK: - ReminderBanner

/// A single reminder notification banner, colour-coded by severity.
public struct ReminderBanner: View {

    public let reminder: HAKIReminderNotification

    public init(reminder: HAKIReminderNotification) {
        self.reminder = reminder
    }

    public var body: some View {
        HStack(spacing: 10) {
            // Severity icon
            Image(systemName: severityIcon)
                .foregroundColor(severityColor)
                .font(.title3)
                .frame(width: 24)

            VStack(alignment: .leading, spacing: 2) {
                Text(bannerTitle)
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(.primary)

                Text(reminder.taskTitle)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(1)

                if !reminder.fireAt.isEmpty {
                    Text(reminder.fireAt)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }

            Spacer()
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(severityColor.opacity(0.08))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(severityColor.opacity(0.3), lineWidth: 0.5)
        )
    }

    // MARK: Private helpers

    private var bannerTitle: String {
        if reminder.isBirthdayDayOf {
            return "🎂 Birthday today!"
        }
        switch reminder.severity.uppercased() {
        case "EXAM":    return "📚 Exam reminder"
        case "ASSIGNMENT": return "📝 Assignment due"
        case "BIRTHDAY":   return "🎁 Birthday coming up"
        default:        return "⏰ Reminder"
        }
    }

    private var severityIcon: String {
        if reminder.isBirthdayDayOf { return "gift.fill" }
        switch reminder.severity.uppercased() {
        case "EXAM":       return "graduationcap.fill"
        case "ASSIGNMENT": return "doc.text.fill"
        case "BIRTHDAY":   return "birthday.cake.fill"
        default:           return "bell.fill"
        }
    }

    private var severityColor: Color {
        switch reminder.severity.uppercased() {
        case "EXAM":       return .red
        case "ASSIGNMENT": return .orange
        case "BIRTHDAY":   return .pink
        default:           return .blue
        }
    }
}
