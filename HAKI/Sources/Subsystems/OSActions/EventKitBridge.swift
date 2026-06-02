// EventKitBridge.swift
// HAKI — OSActions Subsystem
//
// Wraps EventKit for calendar event creation and reminder scheduling.
//
// Full implementation: Phase 2–3 (Tasks 11, 12).
// Implements: Req 11 (Calendar Automation), Req 12 (Severity Reminders)

import EventKit
import Foundation

// MARK: - EventKitBridge

public final class EventKitBridge: @unchecked Sendable {

    // MARK: - State

    private let store = EKEventStore()

    // MARK: - Authorisation

    /// Request access to calendars and reminders.
    public func requestAccess() async throws {
        if #available(macOS 14.0, *) {
            try await store.requestFullAccessToEvents()
            try await store.requestFullAccessToReminders()
        } else {
            // Fallback for older macOS
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                store.requestAccess(to: .event) { granted, error in
                    if let error { continuation.resume(throwing: error); return }
                    guard granted else { continuation.resume(throwing: EventKitError.accessDenied); return }
                    continuation.resume()
                }
            }
        }
    }

    // MARK: - Calendar events (Req 11)

    /// Create a new calendar event.
    ///
    /// Returns the event identifier on success. Throws on failure without
    /// creating a partial event (Req 11.7).
    public func createEvent(
        title: String,
        startDate: Date,
        endDate: Date,
        location: String? = nil,
        notes: String? = nil
    ) throws -> String {
        let event = EKEvent(eventStore: store)
        event.title = title
        event.startDate = startDate
        event.endDate = endDate
        event.location = location
        event.notes = notes
        event.calendar = store.defaultCalendarForNewEvents

        try store.save(event, span: .thisEvent)
        return event.eventIdentifier
    }

    // MARK: - Reminders (Req 12)

    /// Create a reminder firing at `dueDate`.
    public func createReminder(
        title: String,
        dueDate: Date,
        notes: String? = nil
    ) throws -> String {
        let reminder = EKReminder(eventStore: store)
        reminder.title = title
        reminder.dueDateComponents = Calendar.current.dateComponents(
            [.year, .month, .day, .hour, .minute],
            from: dueDate
        )
        reminder.notes = notes
        reminder.calendar = store.defaultCalendarForNewReminders()

        try store.save(reminder, commit: true)
        return reminder.calendarItemIdentifier
    }
}

// MARK: - EventKitError

public enum EventKitError: Error {
    case accessDenied
    case saveFailed(String)
}
