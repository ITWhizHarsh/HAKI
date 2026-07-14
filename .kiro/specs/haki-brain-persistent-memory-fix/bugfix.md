# Bugfix Requirements Document

## Introduction

The user reports that HAKI's persistent memory is not working: the project-local
brain folder at `HAKI/HAKI_Brain/` only contains `Welcome.md` and the `.obsidian/`
config folder, with no chat or conversation files being written.

Investigation shows the persistent-memory subsystem (`HAKIBrain.log_conversation`)
is functioning correctly and is writing daily conversation logs and embeddings on
every turn. However, it is writing them to a different vault than the one the user
is inspecting. The vault path is read from the `HAKI_OBSIDIAN_VAULT` environment
variable in `Core/.env`, which is currently set to
`/Users/harshkumarroy/Obsidian/HAKI_Brain` (an external Obsidian vault outside the
project). Conversation files do exist there
(`conversations/2026-06-18.md`, `2026-06-20.md`, etc.).

The project-local folder `HAKI/HAKI_Brain/` is never opened by the running service
because it is not the configured vault, so it stays empty apart from the manually
created `Welcome.md` and `.obsidian/` config. From the user's point of view this
looks identical to "persistent memory is broken", because the brain folder they
keep open in Obsidian never gets new files.

The fix is a configuration / path-resolution issue: the running HAKI service must
persist conversations into the project-local `HAKI/HAKI_Brain/` vault (the one the
user actually has open), so that the user can see chat logs accumulating there
turn by turn.

## Bug Analysis

### Current Behavior (Defect)

When the user runs the HAKI core service and has a conversation, then opens the
project-local `HAKI/HAKI_Brain/` folder in Obsidian, no conversation files appear
there. The `conversations/`, `raw/`, `processed/`, and `wiki/` subfolders are not
created in the project-local vault either.

1.1 WHEN the HAKI core service starts with the default project layout
    (project root at `HAKI/`, brain folder at `HAKI/HAKI_Brain/`) and
    `HAKI_OBSIDIAN_VAULT` is unset or points somewhere other than
    `HAKI/HAKI_Brain/` THEN the system initialises `HAKIBrain` against the
    wrong vault path and never creates `raw/`, `processed/`, `wiki/`, or
    `conversations/` inside `HAKI/HAKI_Brain/`.

1.2 WHEN the user completes a conversation turn while the service is running
    against the wrong vault path THEN `HAKIBrain.log_conversation` writes the
    daily `conversations/YYYY-MM-DD.md` file into the wrong vault, so the
    project-local `HAKI/HAKI_Brain/` folder remains empty (only the
    pre-existing `Welcome.md` and `.obsidian/` are present).

1.3 WHEN the user inspects the project-local `HAKI/HAKI_Brain/` folder after
    multiple conversations THEN no chat or conversation files are visible
    there, giving the false impression that persistent memory is broken.

### Expected Behavior (Correct)

2.1 WHEN the HAKI core service starts with the default project layout THEN the
    system SHALL resolve the brain vault path to the project-local
    `HAKI/HAKI_Brain/` folder by default (so it works out-of-the-box without
    requiring the user to set `HAKI_OBSIDIAN_VAULT`), and SHALL still honour
    `HAKI_OBSIDIAN_VAULT` when the user explicitly sets it to override the
    default.

2.2 WHEN the HAKI core service starts and the resolved brain folder exists but
    is missing the `raw/`, `processed/`, `wiki/`, and `conversations/`
    subfolders THEN the system SHALL create those subfolders inside the
    resolved brain folder during `HAKIBrain.init()`.

2.3 WHEN the user completes a conversation turn THEN the system SHALL append
    that exchange to `conversations/YYYY-MM-DD.md` inside the project-local
    `HAKI/HAKI_Brain/` folder (or the explicitly configured vault) so the file
    becomes visible to the user immediately in Obsidian.

2.4 WHEN the bug-fix is deployed and the user runs the service with the
    default configuration THEN the user SHALL see new conversation files
    accumulating inside `HAKI/HAKI_Brain/conversations/` as they chat,
    confirming persistent memory is working.

### Unchanged Behavior (Regression Prevention)

3.1 WHEN the user has explicitly set `HAKI_OBSIDIAN_VAULT` to a custom path
    THEN the system SHALL CONTINUE TO use that custom path as the vault
    location (the explicit override always wins over the new default).

3.2 WHEN `HAKIBrain.log_conversation` is called THEN the system SHALL CONTINUE
    TO append to the daily `conversations/YYYY-MM-DD.md` file, embed the
    exchange into ChromaDB, and remain best-effort (a failure SHALL CONTINUE
    TO never break the conversational turn).

3.3 WHEN existing conversation files are already present in the configured
    vault THEN the system SHALL CONTINUE TO append to them and SHALL NOT
    overwrite or delete previously stored conversation history.

3.4 WHEN `HAKIBrain.remember_fact`, `ingest_pending`, `search`,
    `search_and_format`, and `load_recent_history` are invoked THEN the
    system SHALL CONTINUE TO behave exactly as before against the resolved
    vault path.

3.5 WHEN the orchestrator finishes a turn and fires the
    `log_conversation` task THEN the system SHALL CONTINUE TO do so as a
    fire-and-forget background task that does not delay the spoken response.

## Bug Condition

```pascal
FUNCTION isBugCondition(X)
  INPUT: X = (project_root, env_HAKI_OBSIDIAN_VAULT)
  OUTPUT: boolean

  // Bug triggers when env var is unset/empty or points outside the
  // project-local HAKI_Brain folder, so conversations land somewhere
  // other than {project_root}/HAKI_Brain.
  RETURN env_HAKI_OBSIDIAN_VAULT IS NULL
      OR env_HAKI_OBSIDIAN_VAULT = ""
      OR realpath(env_HAKI_OBSIDIAN_VAULT) != realpath(project_root + "/HAKI_Brain")
END FUNCTION
```

```pascal
// Property: Fix Checking — conversations land in project-local HAKI_Brain
FOR ALL X WHERE isBugCondition(X) DO
  start_service(X)
  run_one_conversation_turn(user="hello", assistant="hi")
  ASSERT exists(X.project_root + "/HAKI_Brain/conversations/" + today() + ".md")
  ASSERT file_contains(that_file, "hello") AND file_contains(that_file, "hi")
END FOR
```

```pascal
// Property: Preservation Checking — existing behavior unchanged
FOR ALL X WHERE NOT isBugCondition(X) DO
  ASSERT F(X) = F'(X)   // explicit valid HAKI_OBSIDIAN_VAULT keeps working identically
END FOR
```
