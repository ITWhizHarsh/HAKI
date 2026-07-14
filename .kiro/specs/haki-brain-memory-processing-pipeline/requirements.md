# Requirements Document

## Introduction

This feature adds an automated "Architect's Hybrid Pipeline" to HAKI's Obsidian vault at
`/Users/harshkumarroy/Downloads/HKR/HAKI/HAKI_Brain`. The pipeline runs in the background,
independent of the live conversational turn, and transforms raw inputs and conversation logs
into durable, interlinked memory notes using a two-pass approach designed to protect the 8GB
unified memory budget on an M2 Mac.

The pipeline is organized around two processing passes:

1. **Fast Pass** — every Source_File is first processed by deterministic Python rules (regex,
   fast string matching, lightweight NLP) that extract entities and facts without any LLM call.
   If the Fast Pass yields at least one extractable entity or fact, Wiki_Links and
   Provenance_Links are generated in milliseconds at zero GPU cost.
2. **Heavy Pass** — only invoked as a fallback when the Fast Pass produces nothing. The Heavy
   Pass uses ChromaDB semantic search to retrieve the most relevant existing Memory_Note(s),
   then feeds both the old note and the new file content to Bonsai-8B (via Ollama) with a
   strict prompt that instructs the model to update facts and draw an Evolutionary_Link from
   the old Memory_Note to the new Memory_Note.

This engineering approach is inspired by the Zero-LLM-Call Entity Extraction principles from
Garry Tan's `gbrain` project, built from scratch rather than adopted wholesale.

Two independent processing flows are defined:

1. **Raw file processing** — files dropped into `raw/` are processed by the hybrid pipeline,
   Memory_Notes are written to `wiki/`, and the source file is physically moved to `processed/`
   after successful completion, preventing any duplicate compute.
2. **Conversation processing** — daily logs in `conversations/` (named `YYYY-MM-DD.md`) are
   processed chronologically once per day by the same hybrid pipeline. Conversation files are
   never moved; they remain in `conversations/` to preserve chat history.

All processing is best-effort and MUST NOT corrupt the vault. The vault-path bug is handled by
a separate spec (`haki-brain-persistent-memory-fix`) and is out of scope here.

## Glossary

- **Vault**: The Obsidian directory tree rooted at the Vault_Root
  `/Users/harshkumarroy/Downloads/HKR/HAKI/HAKI_Brain`, containing the subfolders `raw/`,
  `processed/`, `wiki/`, and `conversations/`.
- **Vault_Root**: The canonical absolute path `/Users/harshkumarroy/Downloads/HKR/HAKI/HAKI_Brain`
  where the Vault resides on disk; read at runtime from `HAKI_OBSIDIAN_VAULT` in the `.env` file.
- **Raw_Folder**: The `raw/` subfolder inside the Vault where the owner drops files to be remembered.
- **Processed_Folder**: The `processed/` subfolder inside the Vault where Raw_Folder Source_Files
  are physically moved after successful processing.
- **Wiki_Folder**: The `wiki/` subfolder inside the Vault where all generated Memory_Notes and
  graph connections are saved exclusively.
- **Conversations_Folder**: The `conversations/` subfolder inside the Vault containing one log
  file per day, named `YYYY-MM-DD.md`.
- **Memory_Note**: A markdown note in Wiki_Folder representing one durable memory extracted
  by the Pipeline, produced by either the Fast_Pass or the Heavy_Pass.
- **Source_File**: An input file (a Raw_Folder file or a conversation log) from which one or
  more Memory_Notes are extracted.
- **Wiki_Link**: An Obsidian-style link of the form `[[note-name]]` connecting two notes.
- **Provenance_Link**: A Wiki_Link from a Memory_Note to the Source_File it was extracted from.
- **Evolutionary_Link**: A directed Wiki_Link from an existing Memory_Note to a newly created
  Memory_Note, generated exclusively during the Heavy_Pass, representing the chronological
  evolution of a fact.
- **Fast_Pass**: The first processing stage that applies deterministic Python rules (regex,
  fast string matching, lightweight NLP) to a Source_File's contents to extract entities and
  facts without making any LLM call.
- **Heavy_Pass**: The fallback processing stage triggered when the Fast_Pass yields no
  extractable content; uses ChromaDB_Index for semantic retrieval and Bonsai_LLM for
  extraction and evolutionary linking.
- **Bonsai_LLM**: The locally hosted Bonsai-8B language model accessed via Ollama, used
  exclusively during the Heavy_Pass for memory extraction and evolutionary linking.
- **Local_LLM**: Synonym for Bonsai_LLM; specifically the Bonsai-8B model running via Ollama.
- **ChromaDB_Index**: The ChromaDB vector database index used during the Heavy_Pass to perform
  semantic search against Wiki_Folder Memory_Notes and retrieve the most relevant existing note(s).
- **Pipeline**: The background system that orchestrates the Fast_Pass, Heavy_Pass, provenance
  linking, evolutionary linking, scheduling, and file lifecycle management.
- **Raw_Scheduler**: The scheduler that triggers Raw_Folder processing on the Raw_Interval.
- **Conversation_Scheduler**: The scheduler that triggers Conversations_Folder processing once
  per day at the Conversation_Run_Time.
- **Raw_Interval**: The configurable interval between Raw_Folder processing runs (default 30 minutes).
- **Conversation_Run_Time**: The configurable time of day at which conversation processing runs.
- **Live_Turn**: An in-progress interactive conversational exchange between the owner and HAKI.

## Requirements

### Requirement 1: Fast Pass Extraction

**User Story:** As the vault owner, I want files processed instantly using deterministic rules before any LLM is involved, so that the 8GB unified memory budget on my M2 Mac is preserved for files that truly need it.

#### Acceptance Criteria

1. WHEN the Pipeline begins processing a Source_File, THE Pipeline SHALL first execute the Fast_Pass before loading or invoking the Bonsai_LLM.
2. WHEN the Fast_Pass is executed, THE Pipeline SHALL apply deterministic Python rules (regex, fast string matching, and Python-based NLP operations) to the Source_File contents to extract entities and facts without making any LLM call.
3. WHEN the Fast_Pass yields at least one extractable entity or fact from the Source_File, THE Pipeline SHALL generate Wiki_Links and Provenance_Links for the extracted content without invoking the Bonsai_LLM.
4. WHEN the Fast_Pass successfully extracts entities or facts from a Source_File, THE Pipeline SHALL create one Memory_Note per extracted entity or fact in Wiki_Folder.
5. IF the Fast_Pass yields no extractable entity or fact from a Source_File, THEN THE Pipeline SHALL mark the Source_File as Fast_Pass_Failed and proceed to the Heavy_Pass.
6. IF the Fast_Pass succeeds for a Source_File, THEN THE Pipeline SHALL complete processing of that Source_File without loading the Bonsai_LLM.
7. IF the Fast_Pass raises a processing error (e.g., unreadable file, Python exception) for a Source_File, THEN THE Pipeline SHALL mark the Source_File as Fast_Pass_Error, record a diagnostic message, and proceed to the Heavy_Pass.

### Requirement 2: Heavy Pass Extraction

**User Story:** As the vault owner, I want a fallback LLM pass for files that resist deterministic extraction, so that complex or unstructured files still produce durable memories that connect to my existing knowledge graph.

#### Acceptance Criteria

1. WHEN a Source_File is marked as Fast_Pass_Failed or Fast_Pass_Error, THE Pipeline SHALL execute the Heavy_Pass for that Source_File.
2. WHEN the Heavy_Pass is triggered for a Source_File, THE Pipeline SHALL use the ChromaDB_Index to perform a semantic search against Wiki_Folder and retrieve the top 3 most semantically similar existing Memory_Notes for that Source_File.
3. IF the ChromaDB_Index returns at least one existing Memory_Note for a Source_File, THEN THE Pipeline SHALL provide both the retrieved Memory_Note content and the Source_File content to the Bonsai_LLM in a single prompt during the Heavy_Pass.
4. IF the ChromaDB_Index returns at least one existing Memory_Note for a Source_File, THEN THE Pipeline SHALL include in the prompt an instruction directing the Bonsai_LLM to merge the facts from the retrieved Memory_Note with the new information from the Source_File and to draw a Wiki_Link from the old Memory_Note to the new Memory_Note.
5. WHEN the Bonsai_LLM produces updated memory content during the Heavy_Pass, THE Pipeline SHALL create a new Memory_Note in Wiki_Folder containing the updated content.
6. IF the ChromaDB_Index returns no existing Memory_Note for a Source_File during the Heavy_Pass, THEN THE Pipeline SHALL invoke the Bonsai_LLM with only the Source_File content to create a new Memory_Note without an Evolutionary_Link.
7. IF the Bonsai_LLM does not respond within 30 seconds when the Heavy_Pass begins for a Source_File, THEN THE Pipeline SHALL abort the Heavy_Pass for that Source_File, record a diagnostic message identifying the Source_File name and the failure reason, and leave the Source_File in its origin folder for a future run.
8. IF the Bonsai_LLM returns empty or unparseable content during the Heavy_Pass, THEN THE Pipeline SHALL abort the Heavy_Pass for that Source_File, record a diagnostic message, and leave the Source_File in its origin folder for a future run.

### Requirement 3: Provenance Linking

**User Story:** As the vault owner, I want each generated memory note to link back to its source file, so that I can trace where every memory originated regardless of which pass produced it.

#### Acceptance Criteria

1. WHEN the Pipeline creates a Memory_Note via the Fast_Pass, THE Pipeline SHALL embed a Provenance_Link in the Memory_Note's body that references the Source_File using its Vault-relative path in Wiki_Link format.
2. WHEN the Pipeline creates a Memory_Note via the Heavy_Pass, THE Pipeline SHALL embed a Provenance_Link in the new Memory_Note's body that references the Source_File using its Vault-relative path in Wiki_Link format.
3. WHEN the Pipeline adds a Provenance_Link, THE Pipeline SHALL verify that the Source_File path referenced by the link resolves to an existing file within the Vault.
4. IF the Source_File path does not resolve to an existing file within the Vault, THEN THE Pipeline SHALL not create the Memory_Note and SHALL record a diagnostic message.
5. WHERE a Source_File is a Markdown-formatted file, THE Pipeline SHALL append a Wiki_Link to each Memory_Note extracted from it at the end of the Source_File's body.

### Requirement 4: Source File Movement and No-Reprocessing

**User Story:** As the vault owner, I want processed raw files physically moved to `processed/` after successful processing, so that the same file is never processed twice and no duplicate compute occurs.

#### Acceptance Criteria

1. WHEN the Pipeline successfully completes processing a Source_File in Raw_Folder, THE Pipeline SHALL physically move the Source_File from Raw_Folder to Processed_Folder.
2. WHEN the Raw_Scheduler triggers a run, THE Pipeline SHALL process only Source_Files present in Raw_Folder at the time the run begins, excluding any files added after the run has started.
3. IF a Source_File with the same filename already exists in Processed_Folder, THEN THE Pipeline SHALL move the Source_File to Processed_Folder under a unique filename that retains the original filename stem and original file extension.
4. IF processing of a Source_File fails before its Memory_Notes and Provenance_Links have been successfully written to Wiki_Folder, THEN THE Pipeline SHALL leave the Source_File in Raw_Folder without moving it to Processed_Folder.
5. THE Pipeline SHALL move a Source_File to Processed_Folder only after both its Memory_Notes and Provenance_Links have been successfully written to Wiki_Folder.
6. IF the filesystem move of a Source_File to Processed_Folder fails after its Memory_Notes and Provenance_Links have been successfully written, THEN THE Pipeline SHALL retain the Source_File in Raw_Folder, record a diagnostic message, and not mark the file as processed.

### Requirement 5: Evolutionary Memory Graph

**User Story:** As the vault owner, I want my memory graph to capture how facts evolve over time, so that I can trace the chronological history of any piece of knowledge from its origin to its most current state.

#### Acceptance Criteria

1. WHEN the Heavy_Pass creates a new Memory_Note that supersedes or contradicts facts from an existing Memory_Note, as determined by Bonsai-8B, THE Pipeline SHALL add an Evolutionary_Link from the old Memory_Note to the new Memory_Note.
2. THE Pipeline SHALL write all Evolutionary_Links exclusively in Wiki_Folder.
3. WHEN the Pipeline adds an Evolutionary_Link, THE Pipeline SHALL format it as a valid Wiki_Link of the form `[[old-memory-note-name]]` appended to the end of the new Memory_Note's body, annotated with the processing run date in YYYY-MM-DD format.
4. IF the target Memory_Note referenced by an Evolutionary_Link does not exist in Wiki_Folder, THEN THE Pipeline SHALL skip writing that Evolutionary_Link and record a diagnostic message.
5. IF the ChromaDB_Index returns an empty result set during the Heavy_Pass, THEN THE Pipeline SHALL create the new Memory_Note without an Evolutionary_Link.

### Requirement 6: Conversation Processing

**User Story:** As the vault owner, I want my daily conversation logs processed into memories using the same hybrid pipeline, so that what I discuss with HAKI becomes durable, linked knowledge while my chat history remains intact in `conversations/`.

#### Acceptance Criteria

1. WHEN the Conversation_Scheduler triggers a run, THE Pipeline SHALL process each conversation log in Conversations_Folder whose filename date falls on or before the previous calendar day, as determined by local system time, and that has not yet been marked as processed.
2. WHEN the Pipeline processes a conversation log, THE Pipeline SHALL apply the Fast_Pass using the same extraction and linking rules as raw file processing.
3. THE Pipeline SHALL leave each processed conversation log in Conversations_Folder under its original `YYYY-MM-DD.md` name and SHALL NOT move conversation logs to Processed_Folder.
4. WHEN the Conversation_Scheduler triggers a run, THE Pipeline SHALL NOT process the conversation log whose filename date matches the current calendar day, as determined by local system time.
5. IF a conversation log has been marked as processed in a prior run, THEN THE Pipeline SHALL NOT process that log again; a log is marked as processed upon successful completion of either the Fast_Pass or the Heavy_Pass for that log.
6. WHEN the Pipeline processes conversation logs in a single Conversation_Scheduler run, THE Pipeline SHALL process the logs in chronological order from the oldest unprocessed log to the most recent unprocessed log.
7. IF the Fast_Pass fails for a conversation log, THEN THE Pipeline SHALL apply the Heavy_Pass using the same extraction and linking rules as raw file processing.
8. IF both the Fast_Pass and the Heavy_Pass fail for a conversation log, THEN THE Pipeline SHALL leave that log unmarked as processed so that it is retried in the next Conversation_Scheduler run.

### Requirement 7: Hardware Constraint

**User Story:** As the vault owner, I want the pipeline to respect the 8GB unified memory limit on my M2 Mac, so that background processing never causes memory pressure that disrupts active work or live conversation.

#### Acceptance Criteria

1. IF the Pipeline is about to load the Bonsai_LLM for a Source_File and the Fast_Pass for that Source_File has not yet completed, THEN THE Pipeline SHALL not load the Bonsai_LLM and SHALL complete the Fast_Pass first.
2. WHILE a Heavy_Pass execution is in progress, THE Pipeline SHALL load and invoke the Bonsai_LLM for at most one Source_File at a time.
3. THE Pipeline SHALL NOT batch-process multiple Source_Files through the Bonsai_LLM simultaneously.
4. WHEN the Heavy_Pass for a Source_File is complete, THE Pipeline SHALL release all Bonsai_LLM model resources before loading the Bonsai_LLM for the next Source_File.
5. THE Pipeline SHALL process Source_Files sequentially within a single scheduler run and SHALL NOT process Source_Files concurrently.
6. IF the system's available memory drops below a configurable low-memory threshold during a Heavy_Pass, THEN THE Pipeline SHALL suspend the current Heavy_Pass, release all Bonsai_LLM model resources, and defer the remaining Heavy_Pass processing to the next scheduled run.

### Requirement 8: Path Configuration

**User Story:** As the vault owner, I want the pipeline to use the correct absolute vault path read from configuration, so that memory notes are always written to the right location on disk.

#### Acceptance Criteria

1. THE Pipeline SHALL read the Vault_Root path exclusively from the `HAKI_OBSIDIAN_VAULT` environment variable defined in the `.env` file.
2. THE Pipeline SHALL use the value of `HAKI_OBSIDIAN_VAULT` as the Vault_Root for all Vault file read and write operations.
3. IF `HAKI_OBSIDIAN_VAULT` is not set, is empty, or does not contain an absolute path beginning with `/` at pipeline startup, THEN THE Pipeline SHALL abort startup, record a diagnostic message, and perform no Vault modifications.
4. THE `.env` file SHALL set `HAKI_OBSIDIAN_VAULT` to the absolute path `/Users/harshkumarroy/Downloads/HKR/HAKI/HAKI_Brain` to accurately point to the Vault_Root.
5. WHEN the Pipeline starts, THE Pipeline SHALL verify that the directory at the `HAKI_OBSIDIAN_VAULT` path exists and is accessible for both read and write operations; if not, THE Pipeline SHALL abort startup and record a diagnostic message.

### Requirement 9: Scheduling

**User Story:** As the vault owner, I want raw and conversation processing on independent schedules that do not disrupt live conversation, so that background work stays out of my way.

#### Acceptance Criteria

1. THE Raw_Scheduler SHALL trigger Raw_Folder processing at each Raw_Interval, with a default Raw_Interval of 30 minutes and a configurable range of 1 to 1440 minutes.
2. THE Conversation_Scheduler SHALL trigger Conversations_Folder processing once per day at the Conversation_Run_Time, where Conversation_Run_Time is specified in HH:MM 24-hour format using the system's local timezone, with a default of 02:00.
3. THE Pipeline SHALL read Raw_Interval and Conversation_Run_Time from configuration.
4. THE Raw_Scheduler and the Conversation_Scheduler SHALL operate independently such that a failure or active run in one scheduler does not delay or prevent the scheduled execution of the other.
5. WHILE a Live_Turn is in progress, THE Pipeline SHALL defer all Vault-modifying operations until the Live_Turn completes.
6. IF a scheduled run begins while a previous run of the same scheduler is still active, THEN THE Pipeline SHALL skip the new trigger and wait for the next scheduled interval.
7. IF Raw_Interval or Conversation_Run_Time is absent from configuration or contains an out-of-range value, THEN THE Pipeline SHALL use the respective default value and log an error indicating which configuration key is invalid.
8. IF Vault-modifying operations have been deferred due to an active Live_Turn for more than 10 minutes, THEN THE Pipeline SHALL abort the deferred operations and log a warning indicating the deferral timeout was exceeded.

### Requirement 10: Best-Effort Execution and Vault Integrity

**User Story:** As the vault owner, I want processing to be safe and best-effort, so that a failure never corrupts or loses my vault contents.

#### Acceptance Criteria

1. IF an error occurs while processing a single Source_File, THEN THE Pipeline SHALL record a diagnostic message including the Source_File identifier and error description, and continue processing the remaining Source_Files in the same run.
2. IF the Pipeline cannot complete a write to a Memory_Note, THEN THE Pipeline SHALL remove any partially written content from the failed write attempt and leave all previously existing Vault files unmodified.
3. THE Pipeline SHALL move a Source_File to Processed_Folder only after both its Memory_Notes and Provenance_Links have been successfully written to Wiki_Folder.
4. WHEN the Pipeline finishes a run, THE Pipeline SHALL append to the run diagnostic log a summary including the count of Source_Files processed, Memory_Notes created, Fast_Pass successes, Heavy_Pass invocations, and errors encountered.
5. IF the filesystem move of a Source_File to Processed_Folder fails after successful writes, THEN THE Pipeline SHALL record a diagnostic message and leave the Source_File in its origin folder.

### Requirement 11: Link Integrity

**User Story:** As the vault owner, I want all generated links to be valid Obsidian wiki links, so that my Obsidian graph never contains broken or dangling links produced by the Pipeline.

#### Acceptance Criteria

1. THE Pipeline SHALL format every generated link as `[[target]]` where target is the exact name of the destination note as it appears in the Vault.
2. WHEN the Pipeline adds a link to a Vault note, THE Pipeline SHALL verify the link target matches the exact name of an existing note in the Vault before writing the link.
3. IF adding any generated Wiki_Link, Provenance_Link, or Evolutionary_Link would create a duplicate of an existing link of the same type targeting the same destination note within the same Vault note, THEN THE Pipeline SHALL skip that link and leave the destination note unchanged.
4. THE Pipeline SHALL validate all generated Wiki_Links, Provenance_Links, and Evolutionary_Links before writing them to any Vault note.
5. IF a generated link's target does not match the exact name of any existing note in the Vault, THEN THE Pipeline SHALL discard that link without writing it and leave the destination note's existing content unchanged.

## Open Decisions and Assumptions

These items are intentionally unresolved and should be revisited during design.

1. **Memory_Note naming and deduplication**: The naming scheme for Memory_Notes (e.g.,
   slug-based, timestamp-based, concept-based) and the strategy for handling near-duplicate
   memories across multiple runs are to be determined in design.

2. **Fast Pass NLP library choice**: The specific lightweight NLP library used in the Fast_Pass
   (e.g., spaCy, NLTK, or a custom regex engine) is to be selected during design. The chosen
   library MUST operate without GPU and MUST fit within the 8GB unified memory constraint.

3. **ChromaDB embedding model selection**: The embedding model used to populate and query the
   ChromaDB_Index (e.g., `all-MiniLM-L6-v2`, `nomic-embed-text`, or another local model) is
   to be chosen during design. The model MUST run locally and respect the hardware constraint.
