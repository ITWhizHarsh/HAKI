// NotificationManager.swift
// HAKI — OSActions Subsystem
//
// Posts macOS User Notifications (via `UserNotifications` framework) as
// the secondary channel for reminders (Req 12.6).
//
// Full implementation: Phase 2 (Task 12 / Scheduler).

import Foundation
import UserNotifications

// MARK: - NotificationManager

public final class NotificationManager: @unchecked Sendable {

    public static let shared = NotificationManager()
    private init() {}

    // MARK: - Authorisation

    public func requestAuthorisation() async throws {
        let center = UNUserNotificationCenter.current()
        let granted = try await center.requestAuthorization(options: [.alert, .sound, .badge])
        guard granted else {
            throw NotificationError.permissionDenied
        }
    }

    // MARK: - Schedule

    /// Schedule a notification at a specific date (Req 12.6).
    public func schedule(
        identifier: String,
        title: String,
        body: String,
        at date: Date
    ) throws {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body  = body
        content.sound = .default

        let components = Calendar.current.dateComponents(
            [.year, .month, .day, .hour, .minute, .second],
            from: date
        )
        let trigger = UNCalendarNotificationTrigger(dateMatching: components, repeats: false)
        let request = UNNotificationRequest(identifier: identifier, content: content, trigger: trigger)

        UNUserNotificationCenter.current().add(request) { error in
            if let error {
                print("[NotificationManager] Failed to schedule '\(identifier)': \(error)")
            }
        }
    }

    /// Cancel a previously scheduled notification.
    public func cancel(identifier: String) {
        UNUserNotificationCenter.current().removePendingNotificationRequests(
            withIdentifiers: [identifier]
        )
    }
}

// MARK: - NotificationError

public enum NotificationError: Error {
    case permissionDenied
}
