// CalendarProposal.swift
// HAKI — Scheduler Subsystem
//
// Data models for the Scheduler:
//   • ActionableItem  — a structured item extracted from communications or
//     screen content that may imply a calendar event, task, or reminder.
//   • CalendarProposal — a draft calendar event proposed to the User before
//     creation, carrying extracted or edited details and a lifecycle status.
//
// Implements: Req 11.1 (proposal model), Req 11.2 (status lifecycle)

import Foundation

// MARK: - ActionableItem

/// A structured item extracted from an inbound message, email, or screen
/// content that implies a calendar event, task, or reminder.
///
/// - Requirement 11.1: The Scheduler produces a `CalendarProposal` from an
///   `ActionableItem` within 5 s of identification.
/// - Requirement 10.3: The Comms_Reader extracts date, time, location, and
///   description where present.
public struct ActionableItem: Sendable, Equatable {

    // MARK: - ActionableType

    /// The implied output type of this actionable item.
    public enum ActionableType: String, Sendable, Equatable, Codable {
        /// Implies a calendar event.
        case event
        /// Implies a task in the Task_Tracker.
        case task
        /// Implies a standalone reminder.
        case reminder
    }

    // MARK: - Properties

    /// Stable unique identifier.
    public let id: String

    /// The communication account or source (e.g. "WhatsApp", "Gmail").
    public let sourceAccount: String

    /// The identifier of the originating message or email.
    public let sourceMessageId: String

    /// The implied type of this item.
    public let type: ActionableType

    /// Extracted date string (e.g. "2025-08-15"), `nil` if absent.
    ///
    /// A `nil` date triggers the `needsClarification` flag (Req 10.4).
    public let date: String?

    /// Extracted time string (e.g. "14:30"), `nil` if absent.
    ///
    /// A `nil` time triggers the `needsClarification` flag (Req 10.4).
    public let time: String?

    /// Extracted location, `nil` if absent.
    public let location: String?

    /// Extracted description or body text of the item.
    public let description: String

    /// `true` when the item lacks an explicit date or time and requires
    /// User clarification before the Scheduler can propose an event.
    ///
    /// - Requirement 10.4
    public let needsClarification: Bool

    // MARK: - Initialiser

    public init(
        id: String = UUID().uuidString,
        sourceAccount: String,
        sourceMessageId: String,
        type: ActionableType = .event,
        date: String? = nil,
        time: String? = nil,
        location: String? = nil,
        description: String,
        needsClarification: Bool? = nil
    ) {
        self.id = id
        self.sourceAccount = sourceAccount
        self.sourceMessageId = sourceMessageId
        self.type = type
        self.date = date
        self.time = time
        self.location = location
        self.description = description
        // Derive needsClarification automatically unless overridden.
        self.needsClarification = needsClarification ?? (date == nil || time == nil)
    }
}

// MARK: - CalendarProposalStatus

/// Lifecycle status of a `CalendarProposal`.
///
/// - Requirement 11.2: Creation is gated — only PROPOSED proposals may be
///   confirmed; confirming advances to CONFIRMED or triggers EventKit creation.
public enum CalendarProposalStatus: String, Sendable, Equatable, Codable {
    /// Initial state: the proposal has been shown to the User but not yet
    /// confirmed or rejected.
    case proposed

    /// The User confirmed the proposal; the calendar event was created
    /// successfully.
    case confirmed

    /// The User rejected the proposal; no event will be created.
    case rejected

    /// The User confirmed but EventKit creation failed; the proposal details
    /// are retained for retry (Req 11.7).
    case failed
}

// MARK: - CalendarProposal

/// A draft calendar event awaiting User confirmation.
///
/// The Scheduler builds a `CalendarProposal` from an `ActionableItem`, presents
/// it to the User, and creates the EventKit event only after the User confirms.
///
/// - Requirement 11.1: Carries extracted date, time, and description.
/// - Requirement 11.2: Status gate — creation only proceeds from `proposed`.
/// - Requirement 11.5: Supports user-provided edits that override extracted
///   values before creation.
/// - Requirement 11.7: On failure the proposal is retained in `failed` state
///   so the User can retry.
public struct CalendarProposal: Sendable, Equatable {

    // MARK: - Properties

    /// Stable unique identifier.
    public let id: String

    /// Event title (may be edited by the User — Req 11.5).
    public var title: String

    /// Extracted or edited date string (e.g. "2025-08-15").
    public var date: String?

    /// Extracted or edited time string (e.g. "14:30").
    public var time: String?

    /// Optional location string.
    public var location: String?

    /// Event description or notes.
    public var description: String

    /// The `ActionableItem.id` that triggered this proposal.
    public let sourceActionableId: String

    /// Current lifecycle status.
    public var status: CalendarProposalStatus

    /// Identifier of the created EventKit event, populated on successful
    /// confirmation.
    public var eventIdentifier: String?

    // MARK: - Initialisers

    /// Creates a new proposal in the `.proposed` state from an `ActionableItem`.
    ///
    /// - Parameter actionable: The source item; its date, time, location, and
    ///   description are carried into the proposal.
    public init(from actionable: ActionableItem) {
        self.id = UUID().uuidString
        self.title = actionable.description
        self.date = actionable.date
        self.time = actionable.time
        self.location = actionable.location
        self.description = actionable.description
        self.sourceActionableId = actionable.id
        self.status = .proposed
        self.eventIdentifier = nil
    }

    /// Full member-wise initialiser for testing or reconstruction.
    public init(
        id: String = UUID().uuidString,
        title: String,
        date: String? = nil,
        time: String? = nil,
        location: String? = nil,
        description: String,
        sourceActionableId: String,
        status: CalendarProposalStatus = .proposed,
        eventIdentifier: String? = nil
    ) {
        self.id = id
        self.title = title
        self.date = date
        self.time = time
        self.location = location
        self.description = description
        self.sourceActionableId = sourceActionableId
        self.status = status
        self.eventIdentifier = eventIdentifier
    }
}
