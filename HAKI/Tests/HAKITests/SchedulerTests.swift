// SchedulerTests.swift
// HAKI — Unit Tests for the Scheduler subsystem
//
// Covers:
//   1. testProposeEventCarriesExtractedDetails      (Req 11.1)
//   2. testConfirmEventCreatesInCalendar            (Req 11.3)
//   3. testRejectEventDiscards                      (Req 11.4)
//   4. testEditedDetailsOverrideExtracted           (Req 11.5)
//   5. testMissingDateBlocksCreation                (Req 11.6)
//   6. testInvalidTimeBlocksCreation                (Req 11.6)
//   7. testCreationFailureAtomicity                 (Req 11.7)
//   8. testProposalRequiresConfirmationBeforeCreation (Req 11.2)
//
// Additional guard tests:
//   - Re-confirm a confirmed proposal → alreadyConfirmed
//   - Reject a confirmed proposal     → alreadyConfirmed
//   - Confirm/reject unknown ID       → proposalNotFound
//   - Reject a proposal twice         → idempotent
//
// Phase 5 Task 28
// Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7

#if canImport(XCTest)
import XCTest
@testable import HAKIScheduler

// MARK: - MockEventKitBridge

/// Stub that satisfies `EventKitBridgeProtocol` without real EventKit access.
///
/// Test scenarios configure it via:
///   • `shouldThrow`   — forces a throw on the next `createEvent` call.
///   • `stubbedId`     — the identifier returned on success.
///   • `callCount`     — records how many times `createEvent` was called.
///   • `lastTitle` etc — captures the arguments of the most recent call.
final class MockEventKitBridge: EventKitBridgeProtocol, @unchecked Sendable {

    // MARK: - Configuration
    var shouldThrow: Bool = false
    var stubbedId: String = "mock-event-id"

    // MARK: - Capture
    var callCount: Int = 0
    var lastTitle: String?
    var lastStartDate: Date?
    var lastEndDate: Date?
    var lastLocation: String?
    var lastNotes: String?

    // MARK: - EventKitBridgeProtocol

    func createEvent(
        title: String,
        startDate: Date,
        endDate: Date,
        location: String?,
        notes: String?
    ) throws -> String {
        callCount += 1
        lastTitle = title
        lastStartDate = startDate
        lastEndDate = endDate
        lastLocation = location
        lastNotes = notes

        if shouldThrow {
            throw MockBridgeError.simulatedFailure
        }
        return stubbedId
    }
}

enum MockBridgeError: Error, LocalizedError {
    case simulatedFailure

    var errorDescription: String? { "Simulated EventKit failure." }
}

// MARK: - Helpers

private func makeActionable(
    id: String = UUID().uuidString,
    date: String? = "2025-12-01",
    time: String? = "10:00",
    location: String? = nil,
    description: String = "Team standup"
) -> ActionableItem {
    ActionableItem(
        id: id,
        sourceAccount: "WhatsApp",
        sourceMessageId: "msg-\(id)",
        type: .event,
        date: date,
        time: time,
        location: location,
        description: description
    )
}

// MARK: - SchedulerTests

final class SchedulerTests: XCTestCase {

    // MARK: - 1. Propose Event Carries Extracted Details (Req 11.1)

    /// `proposeEvent(from:)` must return a `CalendarProposal` in `.proposed`
    /// status that carries the extracted date, time, and description from the
    /// `ActionableItem`.
    ///
    /// Requirement 11.1
    func testProposeEventCarriesExtractedDetails() async {
        let bridge = MockEventKitBridge()
        let scheduler = Scheduler(bridge: bridge)

        let actionable = makeActionable(
            date: "2025-11-10",
            time: "15:30",
            location: "Room 42",
            description: "Mid-semester exam"
        )
        let proposal = await scheduler.proposeEvent(from: actionable)

        XCTAssertEqual(proposal.status, .proposed,
                       "A freshly proposed event must have PROPOSED status (Req 11.1)")
        XCTAssertEqual(proposal.date, "2025-11-10",
                       "Proposal must carry the extracted date")
        XCTAssertEqual(proposal.time, "15:30",
                       "Proposal must carry the extracted time")
        XCTAssertEqual(proposal.location, "Room 42",
                       "Proposal must carry the extracted location")
        XCTAssertEqual(proposal.description, "Mid-semester exam",
                       "Proposal must carry the extracted description")
        XCTAssertEqual(proposal.sourceActionableId, actionable.id,
                       "Proposal must reference the source actionable ID")
        XCTAssertNil(proposal.eventIdentifier,
                     "Event identifier must be nil until confirmed")
    }

    // MARK: - 2. Confirm Event Creates in Calendar (Req 11.3)

    /// After calling `confirmEvent`, the bridge `createEvent` must be called
    /// exactly once and the returned proposal must be in `.confirmed` status
    /// with a non-nil `eventIdentifier`.
    ///
    /// Requirement 11.3
    func testConfirmEventCreatesInCalendar() async throws {
        let bridge = MockEventKitBridge()
        bridge.stubbedId = "ek-confirmed-001"
        let scheduler = Scheduler(bridge: bridge)

        let actionable = makeActionable(date: "2025-09-05", time: "09:00")
        let proposal = await scheduler.proposeEvent(from: actionable)

        let confirmed = try await scheduler.confirmEvent(proposal, edits: nil)

        XCTAssertEqual(confirmed.status, .confirmed,
                       "Confirmed proposal must have CONFIRMED status (Req 11.3)")
        XCTAssertEqual(confirmed.eventIdentifier, "ek-confirmed-001",
                       "eventIdentifier must be set to the value returned by EventKit")
        XCTAssertEqual(bridge.callCount, 1,
                       "EventKit bridge createEvent must be called exactly once")
    }

    // MARK: - 3. Reject Event Discards Proposal (Req 11.4)

    /// `rejectEvent` must set the proposal status to `.rejected` without calling
    /// the EventKit bridge.
    ///
    /// Requirement 11.4
    func testRejectEventDiscards() async throws {
        let bridge = MockEventKitBridge()
        let scheduler = Scheduler(bridge: bridge)

        let actionable = makeActionable()
        let proposal = await scheduler.proposeEvent(from: actionable)

        let rejected = try await scheduler.rejectEvent(proposal)

        XCTAssertEqual(rejected.status, .rejected,
                       "Rejected proposal must have REJECTED status (Req 11.4)")
        XCTAssertEqual(bridge.callCount, 0,
                       "EventKit bridge must NOT be called when a proposal is rejected (Req 11.4)")
        XCTAssertNil(rejected.eventIdentifier,
                     "No event identifier should be set for a rejected proposal")
    }

    // MARK: - 4. Edited Details Override Extracted Details (Req 11.5)

    /// When `confirmEvent` is called with `edits`, the bridge must receive
    /// the edited values, not the original extracted values.
    ///
    /// Requirement 11.5
    func testEditedDetailsOverrideExtracted() async throws {
        let bridge = MockEventKitBridge()
        let scheduler = Scheduler(bridge: bridge)

        let actionable = makeActionable(
            date: "2025-10-01",
            time: "08:00",
            description: "Original description"
        )
        let proposal = await scheduler.proposeEvent(from: actionable)

        let edits = ProposalEdits(
            title: "Updated Title",
            date: "2025-10-15",
            time: "10:30",
            location: "Conference Hall",
            description: "Updated description"
        )
        let confirmed = try await scheduler.confirmEvent(proposal, edits: edits)

        // The bridge must have received the edited values.
        XCTAssertEqual(bridge.lastTitle, "Updated Title",
                       "Bridge must receive the edited title (Req 11.5)")
        // startDate should reflect the edited date + time.
        let expectedComponents = DateComponents(year: 2025, month: 10, day: 15, hour: 10, minute: 30)
        let expectedDate = Calendar.current.date(from: expectedComponents)!
        XCTAssertEqual(bridge.lastStartDate, expectedDate,
                       "Bridge must receive start date built from the edited date/time (Req 11.5)")
        XCTAssertEqual(bridge.lastLocation, "Conference Hall",
                       "Bridge must receive the edited location (Req 11.5)")
        XCTAssertEqual(bridge.lastNotes, "Updated description",
                       "Bridge must receive the edited description as notes (Req 11.5)")

        XCTAssertEqual(confirmed.title, "Updated Title",
                       "Returned proposal must carry the edited title")
        XCTAssertEqual(confirmed.date, "2025-10-15",
                       "Returned proposal must carry the edited date")
    }

    // MARK: - 5. Missing Date Blocks Creation (Req 11.6)

    /// If the confirmed proposal has no date, `confirmEvent` must throw
    /// `.invalidDateTime` without calling the bridge, and the proposal
    /// details must be retained for correction.
    ///
    /// Requirement 11.6
    func testMissingDateBlocksCreation() async throws {
        let bridge = MockEventKitBridge()
        let scheduler = Scheduler(bridge: bridge)

        // Actionable with no date.
        let actionable = makeActionable(date: nil, time: "14:00")
        let proposal = await scheduler.proposeEvent(from: actionable)

        do {
            _ = try await scheduler.confirmEvent(proposal, edits: nil)
            XCTFail("confirmEvent should throw .invalidDateTime when date is missing (Req 11.6)")
        } catch SchedulerError.invalidDateTime {
            // Expected path.
        } catch {
            XCTFail("Expected .invalidDateTime but got \(error)")
        }

        // Bridge must NOT have been called.
        XCTAssertEqual(bridge.callCount, 0,
                       "EventKit bridge must NOT be called when date is missing (Req 11.6)")
    }

    // MARK: - 6. Invalid / Missing Time Blocks Creation (Req 11.6)

    /// If the confirmed proposal has no time, `confirmEvent` must throw
    /// `.invalidDateTime` without calling the bridge.
    ///
    /// Requirement 11.6
    func testInvalidTimeBlocksCreation() async throws {
        let bridge = MockEventKitBridge()
        let scheduler = Scheduler(bridge: bridge)

        // Actionable with a valid date but no time.
        let actionable = makeActionable(date: "2025-12-01", time: nil)
        let proposal = await scheduler.proposeEvent(from: actionable)

        do {
            _ = try await scheduler.confirmEvent(proposal, edits: nil)
            XCTFail("confirmEvent should throw .invalidDateTime when time is missing (Req 11.6)")
        } catch SchedulerError.invalidDateTime {
            // Expected path.
        } catch {
            XCTFail("Expected .invalidDateTime but got \(error)")
        }

        XCTAssertEqual(bridge.callCount, 0,
                       "EventKit bridge must NOT be called when time is missing (Req 11.6)")
    }

    // MARK: - 7. Creation Failure Atomicity (Req 11.7)

    /// When EventKit throws, `confirmEvent` must:
    ///   • throw `.creationFailed` carrying the full proposal
    ///   • leave the proposal in `.failed` state (details intact for retry)
    ///   • NOT create a partial event
    ///
    /// Requirement 11.7
    func testCreationFailureAtomicity() async throws {
        let bridge = MockEventKitBridge()
        bridge.shouldThrow = true
        let scheduler = Scheduler(bridge: bridge)

        let actionable = makeActionable(
            date: "2025-07-04",
            time: "12:00",
            description: "Independence Day BBQ"
        )
        let proposal = await scheduler.proposeEvent(from: actionable)

        do {
            _ = try await scheduler.confirmEvent(proposal, edits: nil)
            XCTFail("confirmEvent must throw .creationFailed when EventKit throws (Req 11.7)")
        } catch SchedulerError.creationFailed(let retainedProposal, let underlying) {
            // The retained proposal must carry the original details.
            XCTAssertEqual(retainedProposal.title, "Independence Day BBQ",
                           "Full proposal details must be retained on failure (Req 11.7)")
            XCTAssertEqual(retainedProposal.date, "2025-07-04",
                           "Date must be retained after creation failure")
            XCTAssertEqual(retainedProposal.time, "12:00",
                           "Time must be retained after creation failure")
            XCTAssertEqual(retainedProposal.status, .failed,
                           "Proposal status must be FAILED after EventKit error (Req 11.7)")
            XCTAssertTrue(underlying is MockBridgeError,
                          "Underlying error must be the one thrown by the bridge")
        } catch {
            XCTFail("Expected .creationFailed but got \(error)")
        }

        // Bridge was called (it threw), proving atomicity: no partial event.
        XCTAssertEqual(bridge.callCount, 1,
                       "Bridge must have been called exactly once (it threw)")
    }

    // MARK: - 8. Proposal Requires Confirmation Before Creation (Req 11.2)

    /// A proposal in `.proposed` state must not become a calendar event until
    /// `confirmEvent` is explicitly called.  Simply creating a proposal via
    /// `proposeEvent` must NOT trigger EventKit.
    ///
    /// Requirement 11.2
    func testProposalRequiresConfirmationBeforeCreation() async {
        let bridge = MockEventKitBridge()
        let scheduler = Scheduler(bridge: bridge)

        let actionable = makeActionable()
        let proposal = await scheduler.proposeEvent(from: actionable)

        // At this point no confirm has been called.
        XCTAssertEqual(proposal.status, .proposed,
                       "Proposal must remain PROPOSED until confirmed (Req 11.2)")
        XCTAssertEqual(bridge.callCount, 0,
                       "EventKit bridge must NOT be called before confirmation (Req 11.2)")
        XCTAssertNil(proposal.eventIdentifier,
                     "No event identifier should exist before confirmation")
    }

    // MARK: - Additional Guard Tests

    // MARK: Re-confirm an already-confirmed proposal → alreadyConfirmed

    func testConfirmAlreadyConfirmedProposal_throwsAlreadyConfirmed() async throws {
        let bridge = MockEventKitBridge()
        let scheduler = Scheduler(bridge: bridge)

        let proposal = await scheduler.proposeEvent(from: makeActionable())
        _ = try await scheduler.confirmEvent(proposal, edits: nil)

        do {
            _ = try await scheduler.confirmEvent(proposal, edits: nil)
            XCTFail("Should throw .alreadyConfirmed")
        } catch SchedulerError.alreadyConfirmed {
            // Expected
        } catch {
            XCTFail("Expected .alreadyConfirmed but got \(error)")
        }
    }

    // MARK: Confirm a proposal by unknown ID → proposalNotFound

    func testConfirmUnknownProposal_throwsProposalNotFound() async throws {
        let bridge = MockEventKitBridge()
        let scheduler = Scheduler(bridge: bridge)

        let ghost = CalendarProposal(
            id: "does-not-exist",
            title: "Ghost",
            date: "2025-01-01",
            time: "09:00",
            description: "Ghost event",
            sourceActionableId: "src-0"
        )
        do {
            _ = try await scheduler.confirmEvent(ghost, edits: nil)
            XCTFail("Should throw .proposalNotFound")
        } catch SchedulerError.proposalNotFound {
            // Expected
        } catch {
            XCTFail("Expected .proposalNotFound but got \(error)")
        }
    }

    // MARK: Reject a proposal twice → idempotent (second call returns .rejected)

    func testRejectProposalTwice_isIdempotent() async throws {
        let bridge = MockEventKitBridge()
        let scheduler = Scheduler(bridge: bridge)

        let proposal = await scheduler.proposeEvent(from: makeActionable())
        _ = try await scheduler.rejectEvent(proposal)
        let secondReject = try await scheduler.rejectEvent(proposal)

        XCTAssertEqual(secondReject.status, .rejected,
                       "Rejecting an already-rejected proposal should be idempotent")
        XCTAssertEqual(bridge.callCount, 0)
    }

    // MARK: Retry after .failed state succeeds when bridge no longer throws

    func testRetryAfterFailureSucceeds() async throws {
        let bridge = MockEventKitBridge()
        bridge.shouldThrow = true
        let scheduler = Scheduler(bridge: bridge)

        let proposal = await scheduler.proposeEvent(from: makeActionable(date: "2025-06-01", time: "08:00"))

        // First attempt fails.
        let failedProposal: CalendarProposal
        do {
            _ = try await scheduler.confirmEvent(proposal, edits: nil)
            XCTFail("Should have failed")
            return
        } catch SchedulerError.creationFailed(let p, _) {
            failedProposal = p
        }

        // Fix the bridge and retry.
        bridge.shouldThrow = false
        let confirmed = try await scheduler.confirmEvent(failedProposal, edits: nil)

        XCTAssertEqual(confirmed.status, .confirmed,
                       "Retry from .failed state must succeed when bridge no longer throws")
        XCTAssertEqual(bridge.callCount, 2,
                       "Bridge should have been called twice (once fail, once success)")
    }

    // MARK: Partial edits — only title changed, date/time from extraction

    func testPartialEdits_onlyTitleChanged() async throws {
        let bridge = MockEventKitBridge()
        let scheduler = Scheduler(bridge: bridge)

        let actionable = makeActionable(date: "2025-03-10", time: "11:00", description: "Lecture")
        let proposal = await scheduler.proposeEvent(from: actionable)

        let edits = ProposalEdits(title: "CS301 Lecture")
        let confirmed = try await scheduler.confirmEvent(proposal, edits: edits)

        XCTAssertEqual(confirmed.title, "CS301 Lecture",
                       "Edited title should be used")
        XCTAssertEqual(confirmed.date, "2025-03-10",
                       "Un-edited date should remain from extraction")
        XCTAssertEqual(confirmed.time, "11:00",
                       "Un-edited time should remain from extraction")
    }
}
#endif // canImport(XCTest)
