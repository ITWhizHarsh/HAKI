# Requirements Document

## Introduction

HAKI (Heuristic Augmented Knowledge Interface) is an all-in-one personal AI assistant for macOS, inspired by JARVIS. The name draws on the Japanese concept of willpower and ambition ("Haki") and the Hindu concept of the Third Eye (foresight and deep contextual understanding). HAKI is intended to be a constant companion for a student: it reads what is on screen aloud, holds natural zero-latency voice conversations in Hinglish, senses mood from voice, remembers everything through a persistent knowledge brain, automates calendar and communication tasks, generates and edits images by voice, and runs reusable named automations for academic work.

This document captures the full product vision as a set of structured, testable requirements. It deliberately separates *what* HAKI must do from *how* it will be built; technology choices referenced by the user (Obsidian, RAG, the "Chroma" image model, streaming TTS, offline vs. API models) are recorded as context and constraints but the concrete architecture is deferred to the design phase.

Because the scope is large, requirements are grouped by capability area. Many capabilities are interdependent (for example, memory underpins automation, personality, and learning), and those dependencies are noted where relevant.

## Glossary

- **HAKI**: Heuristic Augmented Knowledge Interface. The complete personal AI assistant system, comprising all subsystems below. Used as the system name in requirements that span multiple subsystems.
- **Screen_Reader**: The subsystem that captures on-screen content and renders it as speech.
- **Voice_Engine**: The subsystem that handles speech-to-text capture of user speech and text-to-speech playback of HAKI responses, including low-latency streaming playback.
- **Mood_Detector**: The subsystem that infers the user's emotional state from vocal characteristics of captured speech.
- **Language_Engine**: The subsystem that understands and produces Hindi, English, and Hinglish (mixed Hindi-English).
- **Persona_Engine**: The subsystem that shapes HAKI's responses with a consistent personality, including emotion and wit.
- **Memory_Brain**: The persistent knowledge store and retrieval subsystem. Stores and retrieves information the user shares or HAKI observes, modeled on an Obsidian-style note vault with retrieval-augmented generation (RAG).
- **Learning_Engine**: The subsystem that extracts durable knowledge from conversations and writes it into the Memory_Brain so HAKI improves over time.
- **Comms_Reader**: The subsystem that reads the user's WhatsApp messages and email to detect actionable items.
- **Scheduler**: The subsystem that creates, stores, and manages calendar events and tasks, and issues reminders.
- **Task_Tracker**: The subsystem that maintains the list of upcoming tasks and tracks their completion status and prerequisites.
- **Clock**: The subsystem that provides HAKI with current date and time and timezone awareness.
- **Image_Studio**: The subsystem that generates and edits images from voice or text instructions.
- **Text_Assistant**: The subsystem that provides autocorrection and context-aware autocompletion for user typing.
- **Automation_Library**: The subsystem that stores and runs reusable named automations (referred to by the user as "#Hkr prompts").
- **Actionable_Item**: A message, email, or screen content that implies a calendar event, task, or reminder (for example an assignment deadline, exam date, or birthday).
- **Severity**: A classification of a task or event that determines reminder timing and frequency (for example exam/assignment versus birthday).
- **User**: The single primary human owner of the HAKI installation (a student).
- **Confirmation**: An explicit user approval (by voice or UI) required before HAKI performs a side-effecting action such as creating a calendar event.
- **Wake_Invocation**: The user addressing HAKI to begin or direct an interaction (for example "HAKI, ...").
- **Streaming_Playback**: Beginning text-to-speech playback of the earliest available words of a response before the full response has been generated, to minimize perceived latency.
- **Mac_Controller**: The subsystem that performs agentic control of the macOS environment in response to ad-hoc natural-language commands, including launching and operating applications, navigating and operating websites, filling forms, and activating on-screen user-interface elements. Distinct from the Automation_Library, which runs only pre-defined named automations.
- **Dialogue_Manager**: The subsystem that conducts interactive two-way conversation, detecting when a User request is ambiguous or underspecified and asking the User clarifying questions before and during task execution.
- **Command_Plan**: An ordered sequence of one or more steps, spanning one or more applications or websites, that the Mac_Controller generates from a single ad-hoc natural-language command and executes to fulfill the command's intent.
- **Consequential_Action**: An action performed by HAKI that is destructive, high-impact, or hard to reverse, including deleting files or data, sending a message or email, placing a call, making a purchase or payment, modifying system settings, or any other irreversible action.
- **Reversible_Action**: An action performed by HAKI that is non-destructive and easily reversible, including opening an application, opening browser tabs, and reading on-screen or web content.
- **Default_Browser**: The web browser configured as the macOS default for opening web links; for the User, the Default_Browser is Arc.

## Requirements

### Requirement 1: Screen Reading (Read-Aloud)

**User Story:** As a student, I want HAKI to read aloud whatever is on my screen, so that I can consume PDFs, articles, and AI responses by listening instead of reading.

#### Acceptance Criteria

1. WHEN the User issues a read-aloud Wake_Invocation, THE Screen_Reader SHALL capture the textual content of the frontmost application's focused window and pass that content to the Voice_Engine for playback within 3 seconds of the Wake_Invocation for captured content of 10,000 characters or fewer.
2. WHERE a PDF document is the focused content, THE Screen_Reader SHALL extract the document text in reading order before playback.
3. IF extracting selectable text from a focused PDF document yields no text or returns an error, THEN THE Screen_Reader SHALL apply optical character recognition to the PDF content before playback.
4. WHERE on-screen content is presented as an image without selectable text, THE Screen_Reader SHALL apply optical character recognition to extract readable text before playback.
5. WHILE read-aloud playback is active, THE Screen_Reader SHALL process User playback commands in the order received, pausing playback on a pause command, resuming playback from the paused position on a resume command, and ending playback on a stop command.
6. IF the Screen_Reader cannot extract any text from the focused content after attempting text extraction and optical character recognition, THEN THE Screen_Reader SHALL NOT begin playback and SHALL inform the User that no readable text was found.
7. WHEN the User requests read-aloud of a specific application's content by name, THE Screen_Reader SHALL capture content from the named application's focused window.
8. IF the User issues more than one playback command within the same 200-millisecond window, THEN THE Screen_Reader SHALL apply a stop command before any pause command and a pause command before any resume command.
9. IF the User requests read-aloud of a named application that is not running or cannot be found, THEN THE Screen_Reader SHALL NOT begin playback and SHALL inform the User that the named application is unavailable.

### Requirement 2: Screen Content Access and Permissions

**User Story:** As a macOS user, I want HAKI to access screen content only with my granted permission, so that I retain control over my privacy.

#### Acceptance Criteria

1. WHEN HAKI first attempts an operation that requires screen capture and the required macOS screen-recording or accessibility permission has not previously been requested, THE HAKI SHALL present the corresponding macOS permission request and direct the User to grant the permission in macOS System Settings.
2. IF the User attempts a capability that requires a macOS permission that is not granted, THEN THE HAKI SHALL decline the capability and inform the User, within 2 seconds of the attempt, which permission is missing, which capability is unavailable without that permission, and how to grant the missing permission in macOS System Settings.
3. WHILE both the screen-recording and accessibility permissions are granted, THE HAKI SHALL present no missing-permission messages to the User.
4. THE HAKI SHALL provide the User a control that enables or disables screen content access, and SHALL keep that control accessible to the User whenever HAKI is running.
5. WHILE screen content access is disabled by the User, THE Screen_Reader SHALL decline screen-reading requests and inform the User that screen content access is disabled.
6. IF the User denies a macOS permission that HAKI has requested, THEN THE HAKI SHALL preserve the User's existing settings, identify each capability that remains unavailable without that permission, and direct the User to grant the permission in macOS System Settings.
7. WHEN a previously granted screen-recording or accessibility permission is revoked while HAKI is running, THE HAKI SHALL, within 5 seconds of detecting the revocation, disable each capability that depends on the revoked permission and inform the User which capabilities have become unavailable and how to restore the permission in macOS System Settings.

### Requirement 3: Zero-Latency Conversational Voice

**User Story:** As a user, I want talking to HAKI to feel like talking to a real person, so that conversations are natural and free of awkward delays.

#### Acceptance Criteria

1. WHEN HAKI begins generating a spoken response, THE Voice_Engine SHALL start Streaming_Playback of the earliest available words within 300 milliseconds of those words becoming available and before the full response text has been generated.
2. WHEN the User finishes speaking a request, detected as a continuous silence of at least 800 milliseconds following User speech, THE Voice_Engine SHALL begin audible response playback within 1.5 seconds measured from the end of the User's speech.
3. WHILE HAKI is playing a spoken response, IF the User speaks continuously for 200 milliseconds or longer, THEN THE Voice_Engine SHALL stop playback within 200 milliseconds and capture the new User speech.
4. THE Voice_Engine SHALL convert captured User speech to text for processing by HAKI.
5. THE Voice_Engine SHALL convert HAKI response text to speech for playback to the User.
6. IF the Voice_Engine cannot convert captured User speech to text because no recognizable speech is detected, THEN THE Voice_Engine SHALL inform the User that the request was not understood and prompt the User to repeat the request, without processing a transcribed request.
7. IF the Voice_Engine cannot convert HAKI response text to speech, THEN THE Voice_Engine SHALL present the response to the User as on-screen text and inform the User that audible playback was unavailable.

### Requirement 4: Mood Detection from Voice

**User Story:** As a user, I want HAKI to sense my mood from how I speak, so that it responds in a way that fits how I feel.

#### Acceptance Criteria

1. WHEN the User speaks a request comprising at least 1 second of captured speech, THE Mood_Detector SHALL classify a single primary mood for that request, selected from the set {angry, sad, happy, neutral}, from vocal characteristics including pitch and volume, and SHALL assign that classification a confidence value in the range 0.0 to 1.0 inclusive.
2. THE Mood_Detector SHALL apply a configurable confidence threshold in the range 0.0 to 1.0 inclusive, with a default value of 0.6.
3. WHEN the Mood_Detector classifies the User's mood as angry with a confidence value at or above the configured confidence threshold, THE Persona_Engine SHALL produce a calming response.
4. WHEN the Mood_Detector classifies the User's mood as sad with a confidence value at or above the configured confidence threshold, THE Persona_Engine SHALL produce an encouraging response.
5. WHEN the Mood_Detector classifies the User's mood as a mood other than angry or sad with a confidence value at or above the configured confidence threshold, THE Persona_Engine SHALL produce a neutral response.
6. IF the Mood_Detector does not classify a primary mood with a confidence value at or above the configured confidence threshold, THEN THE Persona_Engine SHALL produce a neutral response.
7. IF the captured User speech for a request is shorter than 1 second, THEN THE Mood_Detector SHALL report the request as unclassifiable and SHALL NOT assign a primary mood for that request.
8. THE Mood_Detector SHALL provide to the Persona_Engine, for each processed request, either the classified primary mood with its confidence value or an unclassifiable indication.

### Requirement 5: Multilingual and Hinglish Understanding

**User Story:** As a Hindi-English bilingual user, I want HAKI to understand and reply in Hinglish, so that I can speak naturally in my own mixed language.

#### Acceptance Criteria

1. WHEN the User speaks or types a request in Hindi, English, or a Hinglish mix, THE Language_Engine SHALL accept and process the request without prompting the User to select a language and SHALL NOT reject or defer the request based on its language composition.
2. WHERE the User's request contains at least one Hindi-origin word and at least one English-origin word, THE Language_Engine SHALL produce HAKI's response as a mix containing at least one Hindi-origin word and at least one English-origin word and SHALL NOT produce the response entirely in Hindi.
3. WHERE the User's request is expressed entirely in Hindi or entirely in English, THE Language_Engine SHALL produce HAKI's response in the same single language as the request.
4. WHEN the Voice_Engine plays a Hinglish response, THE Voice_Engine SHALL pronounce each Hindi-origin word using Hindi pronunciation and each English-origin word using English pronunciation.
5. IF the Language_Engine cannot interpret the request or cannot determine the language composition of the request, THEN THE Language_Engine SHALL NOT process the request and SHALL inform the User that the request was not understood and prompt the User to repeat or rephrase the request.

### Requirement 6: Personality and Wit

**User Story:** As a user, I want HAKI to have its own personality with emotion and wit, so that interactions feel engaging rather than robotic.

#### Acceptance Criteria

1. THE Persona_Engine SHALL apply a consistent HAKI personality identity to every response presented to the User, regardless of the personality intensity setting.
2. WHEN HAKI responds to the User and the Mood_Detector has provided a detected mood and the Memory_Brain has provided relevant context, THE Persona_Engine SHALL incorporate that detected mood and that relevant context into the response tone.
3. THE Persona_Engine SHALL provide the User a control to set personality expression intensity to one of at least three discrete, ordered levels spanning from a defined minimum level to a defined maximum level.
4. WHILE the User has set personality intensity to its minimum level, THE Persona_Engine SHALL produce concise responses, prioritizing conciseness over personality identity expression when the two conflict.
5. IF the detected mood from the Mood_Detector, the relevant context from the Memory_Brain, or both are unavailable when HAKI responds, THEN THE Persona_Engine SHALL produce the response applying the HAKI personality identity at the current intensity level using whichever of the detected mood and relevant context is available, and SHALL present the response without delaying it for the unavailable input.

### Requirement 7: Persistent Core Memory

**User Story:** As a user, I want HAKI to remember everything I share, so that it has full context about my life and work over time.

#### Acceptance Criteria

1. WHEN the User shares information HAKI is instructed to remember, THE Memory_Brain SHALL store that information as a retrievable note and SHALL confirm the store to the User only after the write to persistent storage completes successfully.
2. IF writing a note to persistent storage fails, THEN THE Memory_Brain SHALL NOT confirm the store, SHALL NOT retain a partial note, and SHALL inform the User that the information could not be saved.
3. WHEN HAKI processes a User request, THE Memory_Brain SHALL retrieve stored notes whose content contains at least one term or topic present in that request and provide them as context for the response within 2 seconds of the request, and SHALL NOT include notes whose content contains no terms or topics matching the request.
4. WHEN HAKI starts, THE Memory_Brain SHALL initialize its persistent storage infrastructure even when no notes have been stored.
5. THE Memory_Brain SHALL persist stored notes across HAKI restarts and across device restarts.
6. WHEN the User requests deletion of a specific stored memory, THE Memory_Brain SHALL remove that memory and confirm the removal to the User.
7. WHEN the User asks what HAKI knows about a topic, THE Memory_Brain SHALL return, within 2 seconds, the stored notes whose content contains at least one term or topic matching the specified topic, and SHALL NOT return notes whose content contains no terms or topics matching the specified topic.
8. THE Memory_Brain SHALL store notes in a structured, file-based vault compatible with an Obsidian-style note format.

### Requirement 8: Autonomous Learning

**User Story:** As a user, I want HAKI to learn from every conversation on its own, so that it gets smarter and more useful over time without me managing it.

#### Acceptance Criteria

1. WHEN a conversation between the User and HAKI concludes, defined as the earlier of the User explicitly ending the conversation or a continuous period of 300 seconds elapsing with no User speech or text input following the most recent exchange, THE Learning_Engine SHALL extract from that conversation its durable items, defined as facts and preferences the User states as applicable beyond the current conversation, and write each extracted item into the Memory_Brain.
2. WHEN the Learning_Engine extracts an item that asserts a value for a fact or preference different from the value recorded in an existing stored note for that same fact or preference, THE Learning_Engine SHALL store the extracted item as a new note and mark the conflicting prior note as superseded.
3. THE Learning_Engine SHALL make each learned item that is not marked as superseded retrievable by the Memory_Brain in subsequent conversations.
4. WHEN the User requests the record of recently learned items, THE Learning_Engine SHALL provide a record of items learned during a recent period that is User-configurable to any value from 1 to 90 days and that defaults to 7 days.
5. WHERE the User marks a learned item as incorrect, THE Learning_Engine SHALL remove that item from the Memory_Brain and confirm the removal to the User.
6. IF the Learning_Engine cannot extract durable items from a concluded conversation, THEN THE Learning_Engine SHALL leave all existing stored notes unchanged and SHALL record the conversation's learning as incomplete in the learned-items record provided to the User.
7. IF writing an extracted item into the Memory_Brain fails, THEN THE Learning_Engine SHALL NOT retain a partially written note for that item, SHALL leave all previously stored notes unchanged, and SHALL record that item's learning as incomplete in the learned-items record provided to the User.

### Requirement 9: Privacy Boundaries for Memory and Learning

**User Story:** As a user, I want control over what HAKI remembers, so that sensitive information is not stored without my awareness.

#### Acceptance Criteria

1. WHERE the User has designated a conversation as private using the privacy-designation control, THE Learning_Engine SHALL NOT write items from that conversation into the Memory_Brain.
2. THE Memory_Brain SHALL store all notes locally on the User's device.
3. WHEN the User requests export of all stored memory, THE Memory_Brain SHALL produce a single file containing every stored note, save it to a User-accessible location, and confirm completion to the User.
4. THE Memory_Brain SHALL remove stored notes only when the User explicitly requests deletion.
5. WHEN the User requests deletion of all stored memory, THE Memory_Brain SHALL remove all stored notes and confirm completion to the User.
6. IF the Memory_Brain cannot confirm completion of a requested deletion, THEN THE Memory_Brain SHALL NOT remove the stored notes and SHALL inform the User that the deletion did not proceed.
7. THE HAKI SHALL provide the User a control to designate a conversation as private, and that control SHALL be accessible to the User before and during any conversation.
8. IF the Memory_Brain cannot complete a requested export, THEN THE Memory_Brain SHALL NOT produce a partial export file and SHALL inform the User that the export did not complete.

### Requirement 10: Communication Reading

**User Story:** As a student, I want HAKI to read my WhatsApp messages and email, so that it can find tasks and events I need to act on.

#### Acceptance Criteria

1. WHERE the User has granted access to a connected WhatsApp account, WHEN a new incoming WhatsApp message arrives, THE Comms_Reader SHALL read the message and identify any Actionable_Items it contains within 60 seconds of the message's arrival.
2. WHERE the User has granted access to a connected email account, WHEN a new incoming email arrives, THE Comms_Reader SHALL read the email and identify any Actionable_Items it contains within 60 seconds of the email's arrival.
3. WHEN the Comms_Reader identifies an Actionable_Item, THE Comms_Reader SHALL extract each associated detail present in the message, including date, time, location, and description.
4. IF an Actionable_Item lacks an explicit date or an explicit time, THEN THE Comms_Reader SHALL flag the Actionable_Item as requiring User clarification and SHALL present the flagged Actionable_Item to the User.
5. THE Comms_Reader SHALL provide the User a control to grant or revoke access to each connected communication account.
6. IF the Comms_Reader fails to read messages from a connected account, THEN THE Comms_Reader SHALL retry reading that account up to 3 additional times at intervals of 30 seconds.
7. IF the Comms_Reader fails to read messages from a connected account after exhausting all retry attempts, THEN THE Comms_Reader SHALL notify the User of the failure and SHALL indicate which account could not be read.
8. WHEN the User revokes access to a connected communication account, THE Comms_Reader SHALL stop reading messages from that account within 5 seconds of the revocation.
9. WHEN the User grants access to a communication account, THE Comms_Reader SHALL begin reading incoming messages from that account.

### Requirement 11: Calendar Automation with Confirmation

**User Story:** As a student, I want HAKI to create calendar events from my messages but ask me first, so that my calendar stays accurate and under my control.

#### Acceptance Criteria

1. WHEN the Comms_Reader identifies an Actionable_Item that implies a calendar event, THE Scheduler SHALL propose to the User, within 5 seconds of the identification, a calendar event containing the extracted date, time, and description.
2. THE Scheduler SHALL require explicit User Confirmation, given by voice or UI, before creating any calendar event, and SHALL NOT create a proposed calendar event until the User confirms it.
3. WHEN the User confirms a proposed calendar event, THE Scheduler SHALL create the event in the User's calendar with the confirmed details and SHALL confirm the creation to the User.
4. IF the User rejects a proposed calendar event, THEN THE Scheduler SHALL discard the proposal and SHALL NOT create the event.
5. WHEN the User confirms a proposed calendar event with edited details, THE Scheduler SHALL create the event using the User's edited details in place of the extracted details and SHALL confirm the creation to the User.
6. IF the User confirms a proposed calendar event with edited details that omit the date or time or that specify a date and time which is not a valid calendar value, THEN THE Scheduler SHALL NOT create the event and SHALL prompt the User to correct the invalid details while retaining the remaining confirmed details.
7. IF creating a confirmed calendar event fails, THEN THE Scheduler SHALL NOT create a partial event, SHALL inform the User that the event could not be created, and SHALL retain the confirmed details so that the User can retry creation.

### Requirement 12: Severity-Based Reminders

**User Story:** As a student, I want reminders timed by how important and time-sensitive a task is, so that I am prepared for deadlines and never miss an important date.

#### Acceptance Criteria

1. WHEN the Scheduler creates a task, THE Scheduler SHALL assign the task a Severity classification.
2. WHERE a task is classified as an assignment submission or exam, THE Scheduler SHALL issue a reminder 1 week before the task due date and a reminder 3 days before the task due date.
3. WHERE a task is classified as a Severity other than assignment submission, exam, or birthday, THE Scheduler SHALL issue a reminder according to the default reminder schedule for that Severity.
4. WHERE an event is classified as a birthday, THE Scheduler SHALL issue a reminder 14 days before the birthday to arrange a gift and a reminder 1 day before the birthday to send wishes.
5. WHERE an event is classified as a birthday, THE Scheduler SHALL prompt the User on the day of the birthday to confirm whether the User sent wishes.
6. THE Scheduler SHALL issue each reminder through the Voice_Engine and through an on-screen notification.
7. WHERE the User has configured valid custom reminder timing for a Severity, including the assignment-submission and exam Severities, THE Scheduler SHALL use the User's configured timing instead of the default timing for that Severity.
8. IF the User's configured custom reminder timing for a Severity is invalid or incomplete, THEN THE Scheduler SHALL use the default timing for that Severity.
9. IF the Scheduler fails to issue one of multiple reminders for a task due to a technical failure, THEN THE Scheduler SHALL issue the remaining reminders for that task and notify the User of the reminder that could not be issued.
10. IF a task is created with a due date that falls within a scheduled reminder window such that one or more reminder times have already elapsed, THEN THE Scheduler SHALL issue an immediate reminder for each already-elapsed reminder time and SHALL schedule any remaining future reminders normally.
11. IF the Severity of a task cannot be determined at creation time, THEN THE Scheduler SHALL assign the task a default Severity and notify the User that the default Severity was applied.

### Requirement 13: Task Tracking and Completion

**User Story:** As a student, I want HAKI to keep a list of my tasks and check whether they are done, so that nothing falls through the cracks.

#### Acceptance Criteria

1. THE Task_Tracker SHALL maintain a list of all upcoming and scheduled tasks with their due dates and Severity.
2. WHEN the User asks for upcoming tasks, THE Task_Tracker SHALL return the list of incomplete tasks ordered by due date within 2 seconds of the request.
3. WHEN a task's due date passes, THE Task_Tracker SHALL ask the User whether the task was completed within 60 seconds of the due date passing.
4. WHEN the User reports a task as completed or HAKI detects a task as completed, THE Task_Tracker SHALL mark that task as completed.
5. WHEN a task is marked as completed in the Task_Tracker, THE Task_Tracker SHALL stop further reminders for that task.
6. WHERE a task records prerequisite requirements, THE Task_Tracker SHALL track the completion status of each prerequisite and report which prerequisites remain incomplete when the User requests the task's status.
7. IF adding a task to the Task_Tracker's persistent list fails, THEN THE Task_Tracker SHALL NOT add a partial task entry, SHALL inform the User that the task could not be saved, and SHALL retain the task details so the User can retry.

### Requirement 14: Temporal Awareness

**User Story:** As a user, I want HAKI to always know the current time, so that it schedules and reminds me accurately.

#### Acceptance Criteria

1. THE Clock SHALL provide the current date, time, and timezone to all subsystems that request it.
2. WHEN the Scheduler computes a reminder time, THE Scheduler SHALL use the current date and time from the Clock.
3. WHEN the User asks for the current date or time, THE HAKI SHALL respond with the current date and time from the Clock within 1 second of the request.
4. WHEN the User's device timezone changes, THE Clock SHALL provide the updated timezone to all subsystems that request it within 5 seconds of detecting the change.
5. IF the Clock cannot obtain the current date, time, or timezone from the system, THEN THE Clock SHALL inform each subsystem that requested it that the time is currently unavailable, and THE HAKI SHALL inform the User that time-dependent features are temporarily unavailable.

### Requirement 15: Image Generation and Voice-Driven Editing

**User Story:** As a user, I want to create and edit images by speaking, so that I can produce visuals hands-free.

#### Acceptance Criteria

1. WHEN the User describes an image to create, THE Image_Studio SHALL generate an image matching the User's described subject, style, and composition, display it in HAKI's UI, and confirm generation to the User.
2. WHEN the User describes an edit to a displayed image by voice and does not specify a prior image, THE Image_Studio SHALL apply the described edit to the most recently displayed image and present the revised image in HAKI's UI.
3. WHEN the User describes an edit and explicitly references a prior displayed image, THE Image_Studio SHALL apply the described edit to that specified prior image and present the revised image in HAKI's UI.
4. WHEN the Image_Studio generates or edits an image, THE Image_Studio SHALL save that image to a designated User-accessible save location and confirm the save to the User.
5. IF saving a generated or edited image to the designated location fails, THEN THE Image_Studio SHALL inform the User that the image could not be saved and SHALL retain the image for display in HAKI's UI for the duration of the current session.
6. IF the Image_Studio cannot produce an image for a request, THEN THE Image_Studio SHALL inform the User that the image could not be produced and state the reason.

### Requirement 16: Smart Text Input

**User Story:** As a user, I want HAKI to fix my typing and suggest completions from my context, so that I type faster and with fewer errors.

#### Acceptance Criteria

1. WHEN the User types text in a supported input field, THE Text_Assistant SHALL correct a detected spelling error inline without a separate confirm step, and only when its confidence in the correction is at or above its configured threshold.
2. WHEN the User types text in a supported input field, THE Text_Assistant SHALL suggest a single context-aware completion drawn from Memory_Brain context and recent User input.
3. WHILE the User has paused typing for at least 500 milliseconds or focused a supported input field without typing, THE Text_Assistant SHALL offer a context-aware completion for that field.
4. WHEN the User accepts a suggested completion, THE Text_Assistant SHALL insert the accepted completion at the current cursor position in the input field.
5. WHEN the User dismisses or rejects a suggested completion, THE Text_Assistant SHALL NOT insert the completion and SHALL NOT suggest the same completion again for the same input state.
6. WHERE the User has disabled the Text_Assistant, THE Text_Assistant SHALL NOT modify or suggest changes to User input and SHALL NOT perform background error detection or completion preparation.

### Requirement 17: Named Custom Automations (#Hkr Prompts)

**User Story:** As a student, I want to invoke reusable named automations by voice, so that I can trigger complex multi-step tasks just by referring to them.

#### Acceptance Criteria

1. THE Automation_Library SHALL store named automations defined by the User, each comprising a name and an ordered sequence of steps.
2. WHEN the User invokes a stored automation by its exact name, THE Automation_Library SHALL run the automation's steps in their defined order, and MAY run steps that do not depend on each other in parallel while preserving that overall order.
3. WHEN the User defines a new named automation, THE Automation_Library SHALL store the automation for later invocation by name.
4. IF the User invokes an automation by a name that does not exactly match any stored automation, THEN THE Automation_Library SHALL inform the User that no matching automation was found and SHALL suggest the nearest stored automation name if one exists.
5. WHILE an automation is running, THE Automation_Library SHALL report the name of the currently executing step to the User and SHALL allow the User to cancel the automation.
6. WHEN the User cancels a running automation, THE Automation_Library SHALL stop executing any steps that have not yet started, SHALL allow any step currently in progress to complete or interrupt it within 5 seconds, and SHALL report to the User which steps were completed and which were not.
7. IF a step in a running automation fails, THEN THE Automation_Library SHALL stop executing subsequent dependent steps, SHALL inform the User which step failed and the reason, and SHALL report which steps completed successfully before the failure.

### Requirement 18: Question-Paper Analysis Automation

**User Story:** As a student, I want a named automation that analyzes question papers against my course material, so that I know which chapters to prioritize for study.

#### Acceptance Criteria

1. WHEN the User invokes the question-paper analysis automation, THE Automation_Library SHALL require at least one question paper to be provided and SHALL NOT begin analysis without it.
2. WHEN the question-paper analysis automation processes a set of question papers, THE Automation_Library SHALL identify topics that appear in two or more of the provided papers and present those topics as recurring topics.
3. WHERE course slides or course content are available, THE question-paper analysis automation SHALL cross-reference the identified recurring topics against that course content and annotate each recurring topic with the matching course content reference.
4. THE question-paper analysis automation SHALL be considered complete only when the Automation_Library has successfully presented a prioritized list of chapters or topics to study to the User, ordered from most to least frequently recurring.
5. IF fewer than all provided question papers can be processed, THEN THE Automation_Library SHALL complete the analysis using the successfully processed papers, inform the User which papers could not be processed and why, and present the prioritized list based on the processed papers.
6. IF the Automation_Library cannot present the prioritized list, THEN THE Automation_Library SHALL inform the User that the analysis did not complete and state the reason.

### Requirement 19: Document Humanization Automation

**User Story:** As a student, I want a named automation that humanizes my LaTeX papers, so that long documents read naturally without manual rewriting.

#### Acceptance Criteria

1. WHEN the User invokes the document-humanization automation with a LaTeX document, THE Automation_Library SHALL read the LaTeX source and separate prose content from LaTeX markup.
2. IF the LaTeX source cannot be parsed or contains no separable prose content, THEN THE Automation_Library SHALL NOT begin segment processing and SHALL inform the User that the document could not be prepared for humanization, stating the reason.
3. WHEN the document-humanization automation divides prose content into segments, each segment SHALL contain between 800 and 1200 words, except for a final segment that may contain fewer words if the remaining prose is less than 800 words.
4. WHEN the document-humanization automation processes each segment, THE Automation_Library SHALL rewrite the segment prose into a more human-sounding form while preserving the segment's meaning.
5. WHEN humanization of the segments completes, THE Automation_Library SHALL write the humanized prose segments back into the LaTeX document while preserving the original LaTeX markup, and SHALL consider the automation successfully complete only when every humanized segment has been written back.
6. IF writing the humanized prose to the LaTeX document fails for any segment, THEN THE Automation_Library SHALL report the automation as not successfully completed, SHALL warn the User which segments could not be saved, and SHALL NOT overwrite any successfully saved segments with unsaved content.

### Requirement 20: Deployment Model and Data Locality

**User Story:** As a privacy-conscious user, I want clarity and control over whether HAKI runs locally or via external APIs, so that I can manage the privacy and capability trade-off.

#### Acceptance Criteria

1. THE HAKI SHALL run on macOS.
2. THE HAKI SHALL provide the User a setting to select, per model-backed capability, whether processing uses a local model or an external API.
3. WHEN the User changes the processing mode for a capability, THE HAKI SHALL apply the new mode before the next invocation of that capability.
4. WHERE a capability is configured to use a local model, THE HAKI SHALL process that capability's data on the User's device.
5. WHERE a capability is configured to use an external API, THE HAKI SHALL inform the User that data for that capability is sent to an external service before the first use of that capability under that configuration.
6. IF a configured external API is unavailable, THEN THE HAKI SHALL inform the User that the capability is temporarily unavailable and SHALL NOT silently fall back to a different processing mode.
7. IF a configured local model fails to load, THEN THE HAKI SHALL inform the User that the local model could not be loaded, identify the affected capability, and SHALL NOT process that capability's data until the User selects a working processing mode.

### Requirement 21: General Mac Control (Agentic Computer Use)

**User Story:** As a student, I want HAKI to control my Mac by natural-language command so it can open and operate any application, send messages, place calls, and operate websites on my behalf, so that I can accomplish ad-hoc tasks hands-free without saving them as automations.

#### Acceptance Criteria

1. WHEN the User issues an ad-hoc natural-language command to operate the Mac, THE Mac_Controller SHALL generate a Command_Plan whose ordered steps fulfill the command's intent and SHALL execute the Command_Plan, without requiring the command to match a stored named automation.
2. WHEN the User commands HAKI to open a named application, THE Mac_Controller SHALL launch the named application and bring its window to the front within 5 seconds of the command for an application installed on the device.
3. WHEN the User commands HAKI to send a message to a named contact through a named application, THE Mac_Controller SHALL open the named application, select the named contact's conversation, enter the User-specified message text, and send the message.
4. WHEN the User commands HAKI to place a call to a named contact through a named application, THE Mac_Controller SHALL open the named application, select the named contact, and initiate the call.
5. WHEN the User commands HAKI to open web search results in browser tabs, THE Mac_Controller SHALL open each result in a separate tab in the Default_Browser.
6. WHEN the User commands HAKI to operate a website, THE Mac_Controller SHALL navigate to the website and perform the commanded operations, including activating on-screen user-interface elements and entering values into form fields.
7. WHEN the Mac_Controller executes a Command_Plan and the command references a fact that is stored in the Memory_Brain, THE Mac_Controller SHALL retrieve that fact from the Memory_Brain and use it in the relevant step instead of prompting the User to provide it.
8. WHEN the Mac_Controller completes execution of a Command_Plan, THE Mac_Controller SHALL report to the User which steps were completed.
9. IF the application named in a command is not installed on the device, THEN THE Mac_Controller SHALL NOT execute the dependent steps and SHALL inform the User that the named application is not installed.
10. IF the application named in a command is installed but closed, THEN THE Mac_Controller SHALL open the named application before performing the dependent steps.
11. IF the contact named in a command cannot be found in the target application, THEN THE Mac_Controller SHALL NOT send a message or place a call to a different contact and SHALL inform the User that the named contact could not be found.
12. IF a required on-screen user-interface element cannot be located while executing a step, THEN THE Mac_Controller SHALL stop executing the dependent steps and SHALL inform the User which step could not be completed and that the required element could not be found.
13. IF a website is unreachable while executing a step, THEN THE Mac_Controller SHALL stop executing the dependent steps that require that website and SHALL inform the User that the website could not be reached.
14. IF a step in a Command_Plan fails midway through execution, THEN THE Mac_Controller SHALL stop executing subsequent dependent steps, SHALL report to the User which steps completed before the failure, and SHALL state which step failed and the reason.
15. IF a macOS permission required for the Mac_Controller to control applications or the system, including the macOS automation or accessibility permission, has not been granted, THEN THE Mac_Controller SHALL NOT execute the control steps of the Command_Plan and SHALL inform the User which permission is missing, which steps are unavailable without it, and how to grant the missing permission in macOS System Settings.
16. IF the contact named in a command matches more than one contact in the target application, THEN THE Mac_Controller SHALL NOT select any of the matching contacts, SHALL prompt the User to identify which single contact is intended, and SHALL NOT send a message or place a call until the User identifies a single contact.

### Requirement 22: Safety Confirmation for Consequential Actions

**User Story:** As a user who has given HAKI broad control of my Mac, I want HAKI to ask me before doing anything destructive or hard to reverse, so that I stay in control and avoid unintended consequences.

#### Acceptance Criteria

1. WHEN a step in a Command_Plan is a Consequential_Action, THE Mac_Controller SHALL request explicit User Confirmation describing the action before performing that step, and SHALL NOT perform the step until the User confirms it.
2. WHEN the User confirms a requested Consequential_Action, THE Mac_Controller SHALL perform that action and continue executing the remaining steps of the Command_Plan.
3. IF the User rejects a requested Consequential_Action, THEN THE Mac_Controller SHALL NOT perform that action.
4. WHERE a step in a Command_Plan is a Reversible_Action, THE Mac_Controller SHALL perform that step without requesting User Confirmation.
5. WHEN a Consequential_Action requiring Confirmation is reached after one or more steps of the Command_Plan have already been performed, THE Mac_Controller SHALL pause execution at that step, request Confirmation, and resume execution only after the User confirms.
6. IF the User rejects a Consequential_Action that occurs mid-Command_Plan, THEN THE Mac_Controller SHALL stop executing subsequent dependent steps and SHALL report to the User which steps were completed and which step was not performed.
7. IF the Mac_Controller cannot determine whether a step is a Consequential_Action or a Reversible_Action, THEN THE Mac_Controller SHALL treat that step as a Consequential_Action and request User Confirmation before performing it.
8. IF the User does not respond to a Confirmation request for a Consequential_Action, THEN THE Mac_Controller SHALL NOT perform that action and SHALL NOT perform subsequent dependent steps until the User responds.

### Requirement 23: Interactive Clarifying Dialogue (Two-Way Conversation)

**User Story:** As a user, I want HAKI to ask me clarifying questions instead of guessing, both before and during a task, so that conversations feel like genuine two-way communication and HAKI acts on accurate intent.

#### Acceptance Criteria

1. WHEN the User issues a request that is ambiguous or underspecified for the information HAKI needs to act, THE Dialogue_Manager SHALL ask the User one or more clarifying questions and SHALL NOT begin executing the request until the missing information is resolved.
2. WHEN the Dialogue_Manager needs information to clarify a request, THE Dialogue_Manager SHALL retrieve any of that information available in the Memory_Brain and SHALL ask the User only for the information that is not available in the Memory_Brain.
3. WHEN a new decision point or new information requiring User input is reached while a task is executing, THE Dialogue_Manager SHALL pause the in-progress task, ask the User the additional clarifying question or questions, incorporate the User's answer, and resume the paused task from the point at which it was paused.
4. WHEN the Dialogue_Manager has obtained sufficient information to act on the request, THE Dialogue_Manager SHALL proceed with the request without asking further clarifying questions for already-resolved information.
5. WHEN the User answers a clarifying question, THE Dialogue_Manager SHALL incorporate the answer into the request before continuing.
6. IF the User declines to answer a clarifying question for a step that has a reasonable default value, THEN THE Dialogue_Manager SHALL proceed using the default value and SHALL inform the User which default value was applied.
7. IF the User declines to answer a clarifying question for a step that has no reasonable default value, THEN THE Dialogue_Manager SHALL abandon that step, SHALL inform the User that the step was not performed, and SHALL continue with the remaining steps that do not depend on the abandoned step.
8. WHERE a request requires the User to choose among multiple candidate options that HAKI has gathered, such as products to purchase, THE Dialogue_Manager SHALL present the candidate options to the User and SHALL NOT select a single option on the User's behalf without the User's choice.
9. WHILE a task is paused awaiting the User's answer to a clarifying question, THE Dialogue_Manager SHALL retain the in-progress task state so that the task resumes from the paused point once the User answers.
