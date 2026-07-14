// Scheduler.swift
// HAKI — Scheduler Subsystem
//
// The Scheduler manages the lifecycle of `CalendarProposal` objects:
//   1. Building a proposal from an `ActionableItem` (Req 11.1)
//   2. Requiring explicit User confirmation before creating any event (Req 11.2)
//   3. Applying User edits to extracted values (Req 11.5)
//   4. Validating date + time before EventKit creation (Req 11.6)
//   5. Delegating creation to `EventKitBridgeProtocol` (Req 11.3)
//   6. Handling EventKit failures atomically (Req 11.7)
//   7. Rejecting proposals cleanly (Req 11.4)
//
// Implements: Req 11.1 – 11.7

import Foundation

// MARK: - EventKitBridgeProtocol

/// Protocol that abstracts the EventKit calendar-creation call.
///
/// In production the `EventKitBridge` concrete class satisfies this protocol.
/// In tests a `MockEventKitBridge` is injected so no calendar permissions are
/// required.
///
/// - Requirement 11.3: Event creation is performed through this interface.
public protocol EventKitBridgeProtocol: Sendable {

    /// Create a calendar event.
    ///
    /// - Parameters:
    ///   - title:     Event title.
    ///   - startDate: Event start date/time.
    ///   - endDate:   Event end date/time (typically start + 1 hour).
    ///   - location:  Optional location string.
    ///   - notes:     Optional notes/description.
    /// - Returns: The EventKit event identifier.
    /// - Throws: Any error from EventKit (no partial event is created on throw).
    func createEvent(
        title: String,
        startDate: Date,
        endDate: Date,
        location: String?,
        notes: String?
    ) throws -> String
}

// MARK: - SchedulerProtocol

/// The public interface of the Scheduler actor.
public protocol SchedulerProtocol: Sendable {

    /// Build a `CalendarProposal` from the given `ActionableItem`.
    ///
    /// The proposal is stored internally and returned in `PROPOSED` state.
    /// Requirement 11.1 — production must complete within 5 s of identification.
    ///
    /// - Parameter actionable: The source actionable item.
    /// - Returns: The new `CalendarProposal` in `.proposed` status.
    func proposeEvent(from actionable: ActionableItem) async -> CalendarProposal

    /// Confirm a proposal, optionally applying User edits, and create the event.
    ///
    /// - Parameters:
    ///   - proposal: The proposal to confirm (must be in `.proposed` state).
    ///   - edits:    Optional struct of values that override the extracted ones
    ///               (Req 11.5). Only non-nil fields in `edits` override.
    /// - Returns: The updated `CalendarProposal` in `.confirmed` or `.failed`
    ///            state, or throws a `SchedulerError`.
    @discardableResult
    func confirmEvent(_ proposal: CalendarProposal, edits: ProposalEdits?) async throws -> CalendarProposal

    /// Reject a proposal — marks it `.rejected` without creating anything.
    ///
    /// - Parameter proposal: The proposal to reject (must be in `.proposed` state).
    /// - Returns: The updated proposal in `.rejected` state.
    /// - Requirement 11.4
    @discardableResult
    func rejectEvent(_ proposal: CalendarProposal) async throws -> CalendarProposal
}

// MARK: - ProposalEdits

/// A bag of optional overrides applied to a `CalendarProposal` before creation.
///
/// Any field left `nil` means "keep the extracted value".
///
/// - Requirement 11.5: User edits override extracted details.
public struct ProposalEdits: Sendable {
    public var title: String?
    public var date: String?
    public var time: String?
    public var location: String?
    public var description: String?

    public init(
        title: String? = nil,
        date: String? = nil,
        time: String? = nil,
        location: String? = nil,
        description: String? = nil
    ) {
        self.title = title
        self.date = date
        self.time = time
        self.location = location
        self.description = description
    }
}

// MARK: - SchedulerError

/// Errors surfaced by the `Scheduler`.
public enum SchedulerError: Error, LocalizedError, Sendable {

    /// No proposal with the given ID exists in the Scheduler's store.
    case proposalNotFound(String)

    /// The proposal has already been confirmed and cannot be re-confirmed.
    case alreadyConfirmed(String)

    /// The proposal has already been rejected and cannot be acted on.
    case alreadyRejected(String)

    /// The date or time fields are missing or do not form a valid calendar date.
    ///
    /// - Requirement 11.6: Blocks creation; the remaining confirmed details are
    ///   retained in the proposal for correction.
    case invalidDateTime(String)

    /// EventKit threw an error during event creation.
    ///
    /// Carries the full `CalendarProposal` so the User can retry.
    ///
    /// - Requirement 11.7: No partial event is created on failure.
    case creationFailed(CalendarProposal, underlying: Error)

    /// Calendar access has been denied by the User or the OS.
    case accessDenied

    // MARK: LocalizedError

    public var errorDescription: String? {
        switch self {
        case .proposalNotFound(let id):
            return "No calendar proposal found with ID '\(id)'."
        case .alreadyConfirmed(let id):
            return "Proposal '\(id)' has already been confirmed."
        case .alreadyRejected(let id):
            return "Proposal '\(id)' has already been rejected."
        case .invalidDateTime(let reason):
            return "Invalid date or time: \(reason). Please correct the date and time."
        case .creationFailed(let proposal, let underlying):
            return "Failed to create calendar event '\(proposal.title)': \(underlying.localizedDescription)."
        case .accessDenied:
            return "Calendar access has been denied. Please grant access in System Settings."
        }
    }
}

// MARK: - Scheduler

/// Actor that manages the lifecycle of calendar event proposals.
///
/// The Scheduler builds `CalendarProposal` objects, stores them keyed by ID,
/// and gates EventKit creation behind explicit User confirmation.
///
/// Thread-safety is provided by the Swift `actor` isolation.
///
/// - Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7
public actor Scheduler: SchedulerProtocol {

    // MARK: - State

    /// In-flight proposals keyed by their ID.
    private var proposals: [String: CalendarProposal] = [:]

    /// The EventKit (or mock) bridge used for calendar creation.
    private let bridge: EventKitBridgeProtocol

    // MARK: - Init

    /// Creates a `Scheduler` using the supplied `EventKitBridgeProtocol`.
    ///
    /// - Parameter bridge: The EventKit bridge; defaults to the production
    ///   `DefaultEventKitBridgeAdapter` when not injected.
    public init(bridge: EventKitBridgeProtocol = DefaultEventKitBridgeAdapter()) {
        self.bridge = bridge
    }

    // MARK: - SchedulerProtocol

    /// Build a `CalendarProposal` from an `ActionableItem` and store it.
    ///
    /// - Requirement 11.1: Carries extracted date, time, and description;
    ///   caller should invoke this within 5 s of identification.
    public func proposeEvent(from actionable: ActionableItem) async -> CalendarProposal {
        let proposal = CalendarProposal(from: actionable)
        proposals[proposal.id] = proposal
        return proposal
    }

    /// Confirm a proposal and create the calendar event.
    ///
    /// Processing order:
    /// 1. Fetch the stored proposal (throws `.proposalNotFound` if missing).
    /// 2. Validate the proposal is still PROPOSED (throws if already confirmed/rejected).
    /// 3. Apply any User edits (Req 11.5).
    /// 4. Validate that the resulting date + time form a valid `Date` (Req 11.6).
    ///    On validation failure: retain the edited proposal as `.proposed`, throw
    ///    `.invalidDateTime`.
    /// 5. Create the event via `bridge.createEvent(...)` (Req 11.3).
    ///    On bridge throw: mark proposal `.failed`, retain full details, throw
    ///    `.creationFailed` (Req 11.7).
    /// 6. Mark proposal `.confirmed` and return it.
    @discardableResult
    public func confirmEvent(_ proposal: CalendarProposal, edits: ProposalEdits? = nil) async throws -> CalendarProposal {

        // 1. Look up the stored proposal.
        guard var stored = proposals[proposal.id] else {
            throw SchedulerError.proposalNotFound(proposal.id)
        }

        // 2. Status gate — Req 11.2.
        switch stored.status {
        case .confirmed:
            throw SchedulerError.alreadyConfirmed(stored.id)
        case .rejected:
            throw SchedulerError.alreadyRejected(stored.id)
        case .failed, .proposed:
            break // Retry from .failed is permitted (Req 11.7).
        }

        // 3. Apply User edits — Req 11.5.
        if let edits {
            if let newTitle = edits.title       { stored.title = newTitle }
            if let newDate  = edits.date        { stored.date  = newDate  }
            if let newTime  = edits.time        { stored.time  = newTime  }
            if let newLoc   = edits.location    { stored.location = newLoc }
            if let newDesc  = edits.description { stored.description = newDesc }
        }

        // Persist the edits immediately (even before validation) so that the
        // updated fields survive a validation failure for correction.
        proposals[stored.id] = stored

        // 4. Validate date + time — Req 11.6.
        guard let startDate = buildDate(dateString: stored.date, timeString: stored.time) else {
            // Retain updated proposal as `.proposed`; throw validation error.
            throw SchedulerError.invalidDateTime(
                "date='\(stored.date ?? "nil")', time='\(stored.time ?? "nil")'."
            )
        }

        let endDate = startDate.addingTimeInterval(3600) // default 1-hour duration

        // 5. Delegate to EventKit — Req 11.3, 11.7.
        do {
            let eventId = try bridge.createEvent(
                title: stored.title,
                startDate: startDate,
                endDate: endDate,
                location: stored.location,
                notes: stored.description.isEmpty ? nil : stored.description
            )
            // 6. Mark confirmed.
            stored.status = .confirmed
            stored.eventIdentifier = eventId
            proposals[stored.id] = stored
            return stored
        } catch {
            // On any EventKit error: mark failed, retain full details (Req 11.7).
            stored.status = .failed
            proposals[stored.id] = stored
            throw SchedulerError.creationFailed(stored, underlying: error)
        }
    }

    /// Reject a proposal — discards it without creating any event.
    ///
    /// - Requirement 11.4
    @discardableResult
    public func rejectEvent(_ proposal: CalendarProposal) async throws -> CalendarProposal {

        guard var stored = proposals[proposal.id] else {
            throw SchedulerError.proposalNotFound(proposal.id)
        }

        switch stored.status {
        case .confirmed:
            throw SchedulerError.alreadyConfirmed(stored.id)
        case .rejected:
            // Idempotent — return existing rejected proposal.
            return stored
        case .proposed, .failed:
            stored.status = .rejected
            proposals[stored.id] = stored
            return stored
        }
    }

    // MARK: - Helpers

    /// Parses a `date` string ("YYYY-MM-DD") and `time` string ("HH:mm") into a
    /// `Date`, returning `nil` if either is absent or cannot be parsed.
    ///
    /// - Requirement 11.6: Both date AND time must be present and valid.
    private func buildDate(dateString: String?, timeString: String?) -> Date? {
        guard let dateString, let timeString,
              !dateString.isEmpty, !timeString.isEmpty else {
            return nil
        }
        let combined = "\(dateString) \(timeString)"
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd HH:mm"
        return formatter.date(from: combined)
    }
}

// MARK: - DefaultEventKitBridgeAdapter

/// Thin adapter that satisfies `EventKitBridgeProtocol` using the production
/// `EventKitBridge`.
///
/// Imported via `HAKIOSActions`; exists here so `HAKIScheduler` does not need
/// to import `HAKIOSActions` in the core actor code — only this adapter does.
import HAKIOSActions

/// Production adapter wrapping `EventKitBridge`.
public final class DefaultEventKitBridgeAdapter: EventKitBridgeProtocol, @unchecked Sendable {

    private let bridge = EventKitBridge()

    public init() {}

    public func createEvent(
        title: String,
        startDate: Date,
        endDate: Date,
        location: String?,
        notes: String?
    ) throws -> String {
        try bridge.createEvent(
            title: title,
            startDate: startDate,
            endDate: endDate,
            location: location,
            notes: notes
        )
    }
}
