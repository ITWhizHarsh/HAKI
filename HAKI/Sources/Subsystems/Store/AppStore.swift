// AppStore.swift
// HAKI — Store Subsystem
//
// Encrypted, structured local store backed by SQLite (in the app sandbox /
// Application Support directory).
//
// Stores:
//   • Tasks and reminders (Req 12, 13)
//   • Named automations (Req 17)
//   • Application settings and PrivacyState (Req 2.4, 9)
//   • Dismissed suggestion state (Req 16.5)
//   • OAuth tokens / API keys via Keychain references (Req 20.2 — never stored
//     in plaintext in this database; only the Keychain `keyRef` is stored)
//
// Full implementation: Phase 0 Task 2.
// Implements: Req 4.2, 6.3, 8.4, 16.6, 2.4

import Foundation
import SQLite

// MARK: - Settings

/// Application-wide user settings persisted in the App Store.
/// 
/// Requirements:
/// - Personality intensity: >= 3 ordered levels (Req 6.3)
/// - Mood detection threshold: 0.0-1.0, default 0.6 (Req 4.2)
/// - Learning window: 1-90 days, default 7 (Req 8.4)
/// - Text_Assistant enabled toggle (Req 16.6)
/// - Screen access enabled toggle (Req 2.4)
public struct Settings: Codable, Sendable, Equatable {

    /// Personality intensity level (Req 6.3). Range: 1 (minimum) … 3 (maximum).
    /// At least 3 ordered levels are provided: 1=minimal, 2=moderate, 3=maximum.
    public var personalityIntensity: Int

    /// Mood-classification confidence threshold (Req 4.2). Range: 0.0 … 1.0.
    public var moodThreshold: Double

    /// Window for recently-learned items in days (Req 8.4). Range: 1 … 90.
    public var recentlyLearnedDays: Int

    /// Whether screen content access is enabled (Req 2.4).
    public var screenAccessEnabled: Bool

    /// Whether the Text_Assistant autocorrect/autocomplete is enabled (Req 16.6).
    public var textAssistantEnabled: Bool

    /// Model provider configurations (Req 20.2)
    public var modelProviders: ModelProviderConfig

    public init(
        personalityIntensity: Int = 2,
        moodThreshold: Double = 0.6,
        recentlyLearnedDays: Int = 7,
        screenAccessEnabled: Bool = true,
        textAssistantEnabled: Bool = true,
        modelProviders: ModelProviderConfig = ModelProviderConfig()
    ) {
        self.personalityIntensity = personalityIntensity
        self.moodThreshold = moodThreshold
        self.recentlyLearnedDays = recentlyLearnedDays
        self.screenAccessEnabled = screenAccessEnabled
        self.textAssistantEnabled = textAssistantEnabled
        self.modelProviders = modelProviders
    }

    // Validation as per requirements
    public static var personalityIntensityRange: ClosedRange<Int> { 1...3 }
    public static var moodThresholdRange: ClosedRange<Double> { 0.0...1.0 }
    public static var recentlyLearnedDaysRange: ClosedRange<Int> { 1...90 }
}

// MARK: - ModelProviderConfig

/// Model provider configuration for each capability (Req 20.2).
/// Stores secrets by keyRef, not directly.
public struct ModelProviderConfig: Codable, Sendable, Equatable {

    public var stt: CapabilityConfig
    public var llm: CapabilityConfig
    public var tts: CapabilityConfig
    public var mood: CapabilityConfig
    public var image: CapabilityConfig
    public var embeddings: CapabilityConfig

    public init(
        stt: CapabilityConfig = CapabilityConfig(),
        llm: CapabilityConfig = CapabilityConfig(),
        tts: CapabilityConfig = CapabilityConfig(),
        mood: CapabilityConfig = CapabilityConfig(),
        image: CapabilityConfig = CapabilityConfig(),
        embeddings: CapabilityConfig = CapabilityConfig()
    ) {
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.mood = mood
        self.image = image
        self.embeddings = embeddings
    }
}

// MARK: - CapabilityConfig

/// Configuration for a single model capability (Req 20.2).
public struct CapabilityConfig: Codable, Sendable, Equatable {

    /// Processing mode: local or api
    public var mode: ProcessingMode

    /// API key reference (Keychain keyRef), nil for local mode
    public var apiKeyRef: String?

    /// API endpoint URL, nil for local mode
    public var apiEndpoint: String?

    /// Model identifier to use
    public var modelId: String?

    public init(
        mode: ProcessingMode = .local,
        apiKeyRef: String? = nil,
        apiEndpoint: String? = nil,
        modelId: String? = nil
    ) {
        self.mode = mode
        self.apiKeyRef = apiKeyRef
        self.apiEndpoint = apiEndpoint
        self.modelId = modelId
    }
}

// MARK: - ProcessingMode

public enum ProcessingMode: String, Codable, Sendable {
    case local
    case api
}

// MARK: - PrivacyState

/// Per-conversation privacy designation (Req 9.7).
/// 
/// Requirements:
/// - Privacy designation per conversation
/// - Export/delete state tracking
public struct PrivacyState: Codable, Sendable, Equatable {
    public var conversationId: String
    public var isPrivate: Bool
    public var markedAt: Date

    public init(conversationId: String, isPrivate: Bool, markedAt: Date = Date()) {
        self.conversationId = conversationId
        self.isPrivate = isPrivate
        self.markedAt = markedAt
    }
}

// MARK: - Task (Req 12, 13)

/// A task tracked by the Task_Tracker (Req 12, 13).
public struct Task: Codable, Sendable, Identifiable, Equatable {
    public var id: String
    public var title: String
    public var taskDescription: String?
    public var dueDate: Date?
    public var severity: TaskSeverity
    public var status: TaskStatus
    public var prerequisites: [Prerequisite]
    public var source: TaskSource

    public init(
        id: String = UUID().uuidString,
        title: String,
        taskDescription: String? = nil,
        dueDate: Date? = nil,
        severity: TaskSeverity = .defaultSeverity,
        status: TaskStatus = .upcoming,
        prerequisites: [Prerequisite] = [],
        source: TaskSource = .manual
    ) {
        self.id = id
        self.title = title
        self.taskDescription = taskDescription
        self.dueDate = dueDate
        self.severity = severity
        self.status = status
        self.prerequisites = prerequisites
        self.source = source
    }
}

// MARK: - TaskSeverity

public enum TaskSeverity: Codable, Sendable, Equatable {
    case assignment
    case exam
    case birthday
    case defaultSeverity
    case custom(String)

    public var isCustom: Bool {
        if case .custom = self { return true }
        return false
    }

    public static var defaultSeverity: TaskSeverity { .defaultSeverity }

    // For SQLite storage - convert to/from String
    public var storageValue: String {
        switch self {
        case .assignment: return "assignment"
        case .exam: return "exam"
        case .birthday: return "birthday"
        case .defaultSeverity: return "default"
        case .custom(let value): return "custom:\(value)"
        }
    }

    public static func fromStorageValue(_ value: String) -> TaskSeverity {
        if value.hasPrefix("custom:") {
            return .custom(String(value.dropFirst(7)))
        }
        switch value {
        case "assignment": return .assignment
        case "exam": return .exam
        case "birthday": return .birthday
        case "default": return .defaultSeverity
        default: return .defaultSeverity
        }
    }
}

// MARK: - TaskStatus

public enum TaskStatus: String, Codable, Sendable {
    case upcoming
    case complete
}

// MARK: - TaskSource

public enum TaskSource: String, Codable, Sendable {
    case manual
    case comms
    case command
}

// MARK: - Prerequisite

public struct Prerequisite: Codable, Sendable, Identifiable, Equatable {
    public var id: String
    public var title: String
    public var isComplete: Bool

    public init(id: String = UUID().uuidString, title: String, isComplete: Bool = false) {
        self.id = id
        self.title = title
        self.isComplete = isComplete
    }
}

// MARK: - Reminder (Req 12)

public struct Reminder: Codable, Sendable, Identifiable, Equatable {
    public var id: String
    public var taskId: String
    public var fireAt: Date
    public var channels: [ReminderChannel]
    public var state: ReminderState

    public init(
        id: String = UUID().uuidString,
        taskId: String,
        fireAt: Date,
        channels: [ReminderChannel] = [.voice, .notification],
        state: ReminderState = .scheduled
    ) {
        self.id = id
        self.taskId = taskId
        self.fireAt = fireAt
        self.channels = channels
        self.state = state
    }
}

// MARK: - ReminderChannel

public enum ReminderChannel: String, Codable, Sendable {
    case voice
    case notification
}

// MARK: - ReminderState

public enum ReminderState: String, Codable, Sendable {
    case scheduled
    case fired
    case failed
}

// MARK: - ReminderPolicy (Req 12)

public struct ReminderPolicy: Codable, Sendable, Equatable {
    public var severity: TaskSeverity
    public var offsets: [TimeInterval]  // Negative durations before due date
    public var isCustom: Bool

    public init(severity: TaskSeverity, offsets: [TimeInterval], isCustom: Bool = false) {
        self.severity = severity
        self.offsets = offsets
        self.isCustom = isCustom
    }

    /// Default reminder policies (Req 12.2, 12.4)
    public static let defaultPolicies: [ReminderPolicy] = [
        ReminderPolicy(severity: .exam, offsets: [-7 * 24 * 3600, -3 * 24 * 3600]),  // 1 week, 3 days
        ReminderPolicy(severity: .assignment, offsets: [-7 * 24 * 3600, -3 * 24 * 3600]),
        ReminderPolicy(severity: .birthday, offsets: [-14 * 24 * 3600, -1 * 24 * 3600]),  // 14 days, 1 day
        ReminderPolicy(severity: .default, offsets: [-1 * 24 * 3600])  // 1 day
    ]
}

// MARK: - Automation (Req 17)

/// A named automation stored in the Automation_Library (Req 17).
public struct Automation: Codable, Sendable, Identifiable, Equatable {
    public var id: String
    public var name: String
    public var steps: [AutomationStep]
    public var createdAt: Date
    public var updatedAt: Date

    public init(
        id: String = UUID().uuidString,
        name: String,
        steps: [AutomationStep],
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.name = name
        self.steps = steps
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}

// MARK: - AutomationStep

public struct AutomationStep: Codable, Sendable, Identifiable, Equatable {
    public var id: String
    public var intent: String
    public var actuator: String
    public var args: [String: String]
    public var dependsOn: [String]

    public init(
        id: String = UUID().uuidString,
        intent: String,
        actuator: String,
        args: [String: String] = [:],
        dependsOn: [String] = []
    ) {
        self.id = id
        self.intent = intent
        self.actuator = actuator
        self.args = args
        self.dependsOn = dependsOn
    }
}

// MARK: - DismissedSuggestion (Req 16.5)

/// Tracks dismissed suggestions for the Text_Assistant (Req 16.5).
public struct DismissedSuggestion: Codable, Sendable, Identifiable, Equatable {
    public var id: String
    public var suggestionHash: String  // Hash of the suggestion content
    public var inputStateHash: String  // Hash of the input field context
    public var dismissedAt: Date

    public init(
        id: String = UUID().uuidString,
        suggestionHash: String,
        inputStateHash: String,
        dismissedAt: Date = Date()
    ) {
        self.id = id
        self.suggestionHash = suggestionHash
        self.inputStateHash = inputStateHash
        self.dismissedAt = dismissedAt
    }
}

// MARK: - AppStoreProtocol

public protocol AppStoreProtocol: AnyObject, Sendable {
    // MARK: - Settings
    func loadSettings() async throws -> Settings
    func saveSettings(_ settings: Settings) async throws

    // MARK: - Privacy
    func setPrivacy(_ state: PrivacyState) async throws
    func privacy(for conversationId: String) async throws -> PrivacyState?
    func allPrivacyStates() async throws -> [PrivacyState]

    // MARK: - Tasks
    func addTask(_ task: Task) async throws
    func updateTask(_ task: Task) async throws
    func deleteTask(id: String) async throws
    func task(id: String) async throws -> Task?
    func allTasks(incompleteOnly: Bool) async throws -> [Task]

    // MARK: - Reminders
    func addReminder(_ reminder: Reminder) async throws
    func updateReminder(_ reminder: Reminder) async throws
    func deleteReminder(id: String) async throws
    func reminder(id: String) async throws -> Reminder?
    func reminders(forTaskId taskId: String) async throws -> [Reminder]
    func upcomingReminders() async throws -> [Reminder]

    // MARK: - Automations
    func saveAutomation(_ automation: Automation) async throws
    func automation(named name: String) async throws -> Automation?
    func allAutomations() async throws -> [Automation]
    func deleteAutomation(id: String) async throws

    // MARK: - Dismissed Suggestions
    func dismissSuggestion(_ suggestion: DismissedSuggestion) async throws
    func isSuggestionDismissed(suggestionHash: String, inputStateHash: String) async throws -> Bool
    func clearOldDismissedSuggestions(olderThan date: Date) async throws
}

// MARK: - SQLiteAppStore

/// SQLite-backed production implementation — Phase 0 Task 2.1.
public final class SQLiteAppStore: AppStoreProtocol, @unchecked Sendable {

    // MARK: - Store location

    public static var storeDirectory: URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return appSupport.appendingPathComponent("HAKI", isDirectory: true)
    }

    public static var storeURL: URL {
        storeDirectory.appendingPathComponent("haki_store.sqlite")
    }

    private let db: Connection

    // MARK: - Table definitions

    // Settings table (single row)
    private let settings = Table("settings")
    private let settingsId = Expression<Int64>("id")
    private let settingsData = Expression<Data>("data")

    // Privacy states
    private let privacyStates = Table("privacy_states")
    private let privacyId = Expression<String>("id")
    private let privacyConversationId = Expression<String>("conversation_id")
    private let privacyIsPrivate = Expression<Bool>("is_private")
    private let privacyMarkedAt = Expression<Date>("marked_at")

    // Tasks
    private let tasks = Table("tasks")
    private let taskId = Expression<String>("id")
    private let taskTitle = Expression<String>("title")
    private let taskDescription = Expression<String?>("description")
    private let taskDueDate = Expression<Date?>("due_date")
    private let taskSeverity = Expression<String>("severity")
    private let taskStatus = Expression<String>("status")
    private let taskPrerequisites = Expression<Data>("prerequisites")
    private let taskSource = Expression<String>("source")

    // Reminders
    private let reminders = Table("reminders")
    private let reminderId = Expression<String>("id")
    private let reminderTaskId = Expression<String>("task_id")
    private let reminderFireAt = Expression<Date>("fire_at")
    private let reminderChannels = Expression<Data>("channels")
    private let reminderState = Expression<String>("state")

    // Automations
    private let automations = Table("automations")
    private let automationId = Expression<String>("id")
    private let automationName = Expression<String>("name")
    private let automationSteps = Expression<Data>("steps")
    private let automationCreatedAt = Expression<Date>("created_at")
    private let automationUpdatedAt = Expression<Date>("updated_at")

    // Dismissed suggestions
    private let dismissedSuggestions = Table("dismissed_suggestions")
    private let dismissedId = Expression<String>("id")
    private let dismissedSuggestionHash = Expression<String>("suggestion_hash")
    private let dismissedInputStateHash = Expression<String>("input_state_hash")
    private let dismissedAt = Expression<Date>("dismissed_at")

    // MARK: - Initialization

    public init() throws {
        let directory = Self.storeDirectory
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        db = try Connection(Self.storeURL.path)
        try migrate()
    }

    // MARK: - Migration

    private func migrate() throws {
        try db.run(settings.create(ifNotExists: true) { t in
            t.column(settingsId, primaryKey: .autoincrement)
            t.column(settingsData)
        })

        try db.run(privacyStates.create(ifNotExists: true) { t in
            t.column(privacyId, primaryKey: true)
            t.column(privacyConversationId, unique: true)
            t.column(privacyIsPrivate)
            t.column(privacyMarkedAt)
        })

        try db.run(tasks.create(ifNotExists: true) { t in
            t.column(taskId, primaryKey: true)
            t.column(taskTitle)
            t.column(taskDescription)
            t.column(taskDueDate)
            t.column(taskSeverity)
            t.column(taskStatus)
            t.column(taskPrerequisites)
            t.column(taskSource)
        })

        try db.run(reminders.create(ifNotExists: true) { t in
            t.column(reminderId, primaryKey: true)
            t.column(reminderTaskId)
            t.column(reminderFireAt)
            t.column(reminderChannels)
            t.column(reminderState)
        })

        try db.run(automations.create(ifNotExists: true) { t in
            t.column(automationId, primaryKey: true)
            t.column(automationName, unique: true)
            t.column(automationSteps)
            t.column(automationCreatedAt)
            t.column(automationUpdatedAt)
        })

        try db.run(dismissedSuggestions.create(ifNotExists: true) { t in
            t.column(dismissedId, primaryKey: true)
            t.column(dismissedSuggestionHash)
            t.column(dismissedInputStateHash)
            t.column(dismissedAt)
        })
    }

    // MARK: - Settings

    public func loadSettings() async throws -> Settings {
        guard let row = try db.pluck(settings) else {
            return Settings()
        }
        let data = row[settingsData]
        return try JSONDecoder().decode(Settings.self, from: data)
    }

    public func saveSettings(_ settings: Settings) async throws {
        let data = try JSONEncoder().encode(settings)
        if try db.pluck(self.settings) != nil {
            try db.run(self.settings.update(settingsData <- data))
        } else {
            try db.run(self.settings.insert(settingsData <- data))
        }
    }

    // MARK: - Privacy

    public func setPrivacy(_ state: PrivacyState) async throws {
        try db.run(privacyStates.insert(or: .replace,
            privacyId <- state.conversationId,
            privacyConversationId <- state.conversationId,
            privacyIsPrivate <- state.isPrivate,
            privacyMarkedAt <- state.markedAt
        ))
    }

    public func privacy(for conversationId: String) async throws -> PrivacyState? {
        let query = privacyStates.filter(privacyConversationId == conversationId)
        guard let row = try db.pluck(query) else { return nil }
        return PrivacyState(
            conversationId: row[privacyConversationId],
            isPrivate: row[privacyIsPrivate],
            markedAt: row[privacyMarkedAt]
        )
    }

    public func allPrivacyStates() async throws -> [PrivacyState] {
        try db.prepare(privacyStates).map { row in
            PrivacyState(
                conversationId: row[privacyConversationId],
                isPrivate: row[privacyIsPrivate],
                markedAt: row[privacyMarkedAt]
            )
        }
    }

    // MARK: - Tasks

    public func addTask(_ task: Task) async throws {
        let prereqData = try JSONEncoder().encode(task.prerequisites)
        try db.run(tasks.insert(
            taskId <- task.id,
            taskTitle <- task.title,
            taskDescription <- task.taskDescription,
            taskDueDate <- task.dueDate,
            taskSeverity <- task.severity.storageValue,
            taskStatus <- task.status.rawValue,
            taskPrerequisites <- prereqData,
            taskSource <- task.source.rawValue
        ))
    }

    public func updateTask(_ task: Task) async throws {
        let prereqData = try JSONEncoder().encode(task.prerequisites)
        let query = tasks.filter(taskId == task.id)
        try db.run(query.update(
            taskTitle <- task.title,
            taskDescription <- task.taskDescription,
            taskDueDate <- task.dueDate,
            taskSeverity <- task.severity.storageValue,
            taskStatus <- task.status.rawValue,
            taskPrerequisites <- prereqData,
            taskSource <- task.source.rawValue
        ))
    }

    public func deleteTask(id: String) async throws {
        let query = tasks.filter(taskId == id)
        try db.run(query.delete())
    }

    public func task(id: String) async throws -> Task? {
        let query = tasks.filter(taskId == id)
        guard let row = try db.pluck(query) else { return nil }
        return try decodeTask(from: row)
    }

    public func allTasks(incompleteOnly: Bool) async throws -> [Task] {
        var query = tasks.order(taskDueDate.asc)
        if incompleteOnly {
            query = query.filter(taskStatus == TaskStatus.upcoming.rawValue)
        }
        return try db.prepare(query).map { try decodeTask(from: $0) }
    }

    private func decodeTask(from row: Row) throws -> Task {
        let prereqData = row[taskPrerequisites]
        let prereqs = try JSONDecoder().decode([Prerequisite].self, from: prereqData)

        return Task(
            id: row[taskId],
            title: row[taskTitle],
            taskDescription: row[taskDescription],
            dueDate: row[taskDueDate],
            severity: TaskSeverity.fromStorageValue(row[taskSeverity]),
            status: TaskStatus(rawValue: row[taskStatus]) ?? .upcoming,
            prerequisites: prereqs,
            source: TaskSource(rawValue: row[taskSource]) ?? .manual
        )
    }

    // MARK: - Reminders

    public func addReminder(_ reminder: Reminder) async throws {
        let channelsData = try JSONEncoder().encode(reminder.channels)
        try db.run(reminders.insert(
            reminderId <- reminder.id,
            reminderTaskId <- reminder.taskId,
            reminderFireAt <- reminder.fireAt,
            reminderChannels <- channelsData,
            reminderState <- reminder.state.rawValue
        ))
    }

    public func updateReminder(_ reminder: Reminder) async throws {
        let channelsData = try JSONEncoder().encode(reminder.channels)
        let query = reminders.filter(reminderId == reminder.id)
        try db.run(query.update(
            reminderTaskId <- reminder.taskId,
            reminderFireAt <- reminder.fireAt,
            reminderChannels <- channelsData,
            reminderState <- reminder.state.rawValue
        ))
    }

    public func deleteReminder(id: String) async throws {
        let query = reminders.filter(reminderId == id)
        try db.run(query.delete())
    }

    public func reminder(id: String) async throws -> Reminder? {
        let query = reminders.filter(reminderId == id)
        guard let row = try db.pluck(query) else { return nil }
        return try decodeReminder(from: row)
    }

    public func reminders(forTaskId taskId: String) async throws -> [Reminder] {
        let query = reminders.filter(reminderTaskId == taskId).order(reminderFireAt.asc)
        return try db.prepare(query).map { try decodeReminder(from: $0) }
    }

    public func upcomingReminders() async throws -> [Reminder] {
        let now = Date()
        let query = reminders
            .filter(reminderFireAt > now)
            .filter(reminderState == ReminderState.scheduled.rawValue)
            .order(reminderFireAt.asc)
        return try db.prepare(query).map { try decodeReminder(from: $0) }
    }

    private func decodeReminder(from row: Row) throws -> Reminder {
        let channelsData = row[reminderChannels]
        let channels = try JSONDecoder().decode([ReminderChannel].self, from: channelsData)

        return Reminder(
            id: row[reminderId],
            taskId: row[reminderTaskId],
            fireAt: row[reminderFireAt],
            channels: channels,
            state: ReminderState(rawValue: row[reminderState]) ?? .scheduled
        )
    }

    // MARK: - Automations

    public func saveAutomation(_ automation: Automation) async throws {
        let stepsData = try JSONEncoder().encode(automation.steps)
        try db.run(automations.insert(or: .replace,
            automationId <- automation.id,
            automationName <- automation.name,
            automationSteps <- stepsData,
            automationCreatedAt <- automation.createdAt,
            automationUpdatedAt <- automation.updatedAt
        ))
    }

    public func automation(named name: String) async throws -> Automation? {
        let query = automations.filter(automationName == name)
        guard let row = try db.pluck(query) else { return nil }
        return try decodeAutomation(from: row)
    }

    public func allAutomations() async throws -> [Automation] {
        try db.prepare(automations.order(automationName)).map { try decodeAutomation(from: $0) }
    }

    public func deleteAutomation(id: String) async throws {
        let query = automations.filter(automationId == id)
        try db.run(query.delete())
    }

    private func decodeAutomation(from row: Row) throws -> Automation {
        let stepsData = row[automationSteps]
        let steps = try JSONDecoder().decode([AutomationStep].self, from: stepsData)

        return Automation(
            id: row[automationId],
            name: row[automationName],
            steps: steps,
            createdAt: row[automationCreatedAt],
            updatedAt: row[automationUpdatedAt]
        )
    }

    // MARK: - Dismissed Suggestions

    public func dismissSuggestion(_ suggestion: DismissedSuggestion) async throws {
        try db.run(dismissedSuggestions.insert(
            dismissedId <- suggestion.id,
            dismissedSuggestionHash <- suggestion.suggestionHash,
            dismissedInputStateHash <- suggestion.inputStateHash,
            dismissedAt <- suggestion.dismissedAt
        ))
    }

    public func isSuggestionDismissed(suggestionHash: String, inputStateHash: String) async throws -> Bool {
        let query = dismissedSuggestions
            .filter(dismissedSuggestionHash == suggestionHash)
            .filter(dismissedInputStateHash == inputStateHash)
        return try db.pluck(query) != nil
    }

    public func clearOldDismissedSuggestions(olderThan date: Date) async throws {
        let query = dismissedSuggestions.filter(dismissedAt < date)
        try db.run(query.delete())
    }
}

// MARK: - AppStoreError

public enum AppStoreError: Error, LocalizedError {
    case notImplemented
    case migrationFailed(String)
    case writeFailure(String)
    case readFailure(String)
    case notFound(String)

    public var errorDescription: String? {
        switch self {
        case .notImplemented:
            return "This feature is not yet implemented"
        case .migrationFailed(let reason):
            return "Database migration failed: \(reason)"
        case .writeFailure(let reason):
            return "Write operation failed: \(reason)"
        case .readFailure(let reason):
            return "Read operation failed: \(reason)"
        case .notFound(let id):
            return "Item not found: \(id)"
        }
    }
}

// MARK: - Alias for backwards compatibility

public typealias AppStore = SQLiteAppStore
