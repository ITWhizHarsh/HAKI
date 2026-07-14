# HAKI Brain Persistent Memory Fix — Bugfix Design

## Overview

HAKI's persistent-memory subsystem is functionally healthy: `HAKIBrain.log_conversation`
writes daily conversation files and embeddings on every turn. The defect is purely a
vault-path resolution issue in `Core/haki_core_service.py`. The default vault path
falls back to `~/Obsidian/HAKI_Brain` when `HAKI_OBSIDIAN_VAULT` is unset, which is
external to the project. The user has the project-local folder
`HAKI/HAKI_Brain/` open in Obsidian and never sees new files there, making it look
like persistent memory is broken.

The fix changes the default-resolution logic so that, absent an explicit override,
`HAKIBrain` is instantiated against the project-local `HAKI/HAKI_Brain/` folder.
Explicit `HAKI_OBSIDIAN_VAULT` overrides continue to win. Initialisation continues
to create the `raw/`, `processed/`, `wiki/`, and `conversations/` subfolders.

## Glossary

- **Bug_Condition (C)**: `HAKI_OBSIDIAN_VAULT` is unset/empty, or set to a path that
  does not resolve to `{project_root}/HAKI_Brain`. Under C, conversations are
  persisted somewhere other than the project-local brain folder.
- **Property (P)**: After one conversation turn, `{project_root}/HAKI_Brain/conversations/{today}.md`
  exists and contains the user/assistant exchange.
- **Preservation**: When `HAKI_OBSIDIAN_VAULT` is explicitly set to the
  project-local brain folder, behavior is identical between original and fixed code.
  All `HAKIBrain` operations (`log_conversation`, `remember_fact`, `ingest_pending`,
  `search`, `search_and_format`, `load_recent_history`) behave identically.
- **project_root**: The HAKI project root — the directory containing `Core/` and
  `HAKI_Brain/`. At runtime resolvable from `Core/haki_core_service.py` as
  `Path(__file__).resolve().parent.parent`.
- **resolve_vault_path**: New helper that returns the configured vault path,
  preferring `HAKI_OBSIDIAN_VAULT` when set and non-empty, otherwise the
  project-local default.
- **HAKIBrain**: Persistent-memory subsystem in `Core/core/memory/haki_brain.py`.

## Bug Details

### Bug Condition

The bug manifests at service startup when `HAKI_OBSIDIAN_VAULT` is unset, empty,
or points to a directory other than the project-local `HAKI_Brain/`. The startup
code in `haki_core_service.py` falls back to `~/Obsidian/HAKI_Brain`, so
`HAKIBrain` opens a vault outside the project. All subsequent `log_conversation`
writes succeed but land in the wrong vault. The project-local folder the user has
open in Obsidian never receives `conversations/`, `raw/`, `processed/`, or
`wiki/` subfolders or daily logs.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input = (project_root, env_HAKI_OBSIDIAN_VAULT)
  OUTPUT: boolean

  RETURN env_HAKI_OBSIDIAN_VAULT IS NULL
      OR env_HAKI_OBSIDIAN_VAULT = ""
      OR realpath(env_HAKI_OBSIDIAN_VAULT) != realpath(project_root + "/HAKI_Brain")
END FUNCTION
```

### Examples

- `HAKI_OBSIDIAN_VAULT` unset: `obsidian_root` resolves to `~/Obsidian/HAKI_Brain`.
  Expected: `{project_root}/HAKI_Brain/conversations/2026-MM-DD.md` exists after a
  turn. Actual: file appears under `~/Obsidian/HAKI_Brain/conversations/`, project
  folder remains empty (only `Welcome.md` and `.obsidian/`).
- `HAKI_OBSIDIAN_VAULT=/Users/harshkumarroy/Obsidian/HAKI_Brain` (current `.env`):
  Expected: project-local logs accumulate. Actual: logs land in the external vault.
- `HAKI_OBSIDIAN_VAULT=""` (empty string): same as unset; falls back to the wrong
  default.
- Edge case — `HAKI_OBSIDIAN_VAULT` explicitly equal to `{project_root}/HAKI_Brain`:
  no bug; behavior must be preserved exactly.

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Explicit `HAKI_OBSIDIAN_VAULT` override continues to win — when the user sets it
  to a non-empty path, that exact path is used (Req 3.1).
- `HAKIBrain.log_conversation` semantics are unchanged: appends to daily
  `conversations/YYYY-MM-DD.md`, embeds into ChromaDB, and remains best-effort
  (failures never break a turn) (Req 3.2).
- Existing conversation files are appended to, never overwritten or deleted
  (Req 3.3).
- `remember_fact`, `ingest_pending`, `search`, `search_and_format`, and
  `load_recent_history` behave identically against the resolved vault path
  (Req 3.4).
- Orchestrator continues to fire `log_conversation` as a fire-and-forget background
  task that does not delay the spoken response (Req 3.5).

**Scope:**
The change is localised to vault-path resolution in `Core/haki_core_service.py`
(plus the value of `HAKI_OBSIDIAN_VAULT` in `Core/.env`). Any input that does not
involve vault-path resolution is unaffected. This includes:
- All public methods on `HAKIBrain` (their behavior depends only on the resolved
  path, which is identical when `¬C(X)` holds).
- All non-memory subsystems (orchestrator routing, LLM, STT/TTS, IPC server).
- Filesystem operations on already-existing conversation files.

## Hypothesized Root Cause

Based on the requirements and a read of `Core/haki_core_service.py:170-200` and
`Core/.env`, the cause is unambiguous:

1. **Wrong default vault path**: `haki_core_service.py` falls back to
   `Path.home() / "Obsidian" / "HAKI_Brain"` when `HAKI_OBSIDIAN_VAULT` is unset.
   This default is external to the project and unrelated to the folder the user
   has open in Obsidian.

2. **Stale `.env` override pointing externally**: `Core/.env` currently sets
   `HAKI_OBSIDIAN_VAULT=/Users/harshkumarroy/Obsidian/HAKI_Brain`, which makes
   even the explicit-config path land outside the project for this user.

3. **No project-local default**: The startup code does not derive a default from
   the project layout (`{project_root}/HAKI_Brain`), so out-of-the-box the system
   never targets the project folder.

4. **No empty-string handling**: `os.environ.get(..., default)` returns `""` when
   the variable is set but empty, so an empty value would still bypass the
   default rather than falling through to it.

## Correctness Properties

Property 1: Bug Condition - Conversations Land In Project-Local HAKI_Brain

_For any_ startup input where the bug condition holds (HAKI_OBSIDIAN_VAULT is
unset, empty, or points outside `{project_root}/HAKI_Brain`), the fixed service
SHALL initialise `HAKIBrain` against `{project_root}/HAKI_Brain`, create the
`raw/`, `processed/`, `wiki/`, and `conversations/` subfolders there, and after
one conversation turn the file
`{project_root}/HAKI_Brain/conversations/{today}.md` SHALL exist and contain
both the user message and the assistant reply.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**

Property 2: Preservation - Explicit Override And Memory Operations Unchanged

_For any_ startup input where the bug condition does NOT hold (HAKI_OBSIDIAN_VAULT
is explicitly set to `{project_root}/HAKI_Brain`), the fixed code SHALL produce
the same resolved vault path and the same behavior as the original code for all
`HAKIBrain` operations (`log_conversation`, `remember_fact`, `ingest_pending`,
`search`, `search_and_format`, `load_recent_history`), including append-only
semantics on existing daily files and best-effort failure handling.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

## Fix Implementation

### Changes Required

Assuming the root cause is correct, the fix is small and confined to startup
configuration.

**File**: `Core/haki_core_service.py`

**Function**: vault-path resolution block immediately before `HAKIBrain(...)`
instantiation (currently around lines 183-189).

**Specific Changes**:

1. **Compute project-local default**: Derive
   `project_root = Path(__file__).resolve().parent.parent` and
   `default_vault = project_root / "HAKI_Brain"`.

2. **Honour explicit override, ignore empty**: Read
   `env_value = os.environ.get("HAKI_OBSIDIAN_VAULT", "").strip()`. If
   `env_value` is non-empty, use `Path(env_value)`; otherwise use
   `default_vault`. Empty string falls through to the default.

3. **Optional helper**: Extract the resolution into a small private helper
   (`_resolve_haki_brain_vault()`) so it is independently testable.

4. **Log resolved path with source**: Update the existing log line to indicate
   whether the path came from `HAKI_OBSIDIAN_VAULT` or the project-local default,
   to make future misconfigurations easier to diagnose.

5. **Update `Core/.env`**: Either unset `HAKI_OBSIDIAN_VAULT` (preferred — relies
   on the new project-local default) or change it to point at the project-local
   `HAKI_Brain/`. Document the new default in `.env.example`.

No changes to `HAKIBrain` itself, the orchestrator, or any consumers — they all
keep working with whatever path is passed in. `HAKIBrain.init()` already creates
the `raw/`, `processed/`, `wiki/`, and `conversations/` subfolders, satisfying
Req 2.2 with no code change there.

## Testing Strategy

### Validation Approach

Two-phase: first surface a counterexample on the unfixed code (logs land outside
the project folder), then verify the fix routes them correctly and preserves
behavior when an explicit override is set.

### Exploratory Bug Condition Checking

**Goal**: Surface a counterexample on UNFIXED code and confirm the root cause is
vault-path resolution.

**Test Plan**: Use `pytest` with `monkeypatch` to (a) point `Path.home()` at a
temporary directory and (b) clear `HAKI_OBSIDIAN_VAULT`. Drive the resolution
block (or, in a thinner unit, a refactored `_resolve_haki_brain_vault` helper)
and assert the resolved path equals `{project_root}/HAKI_Brain`. On unfixed
code, the assertion fails because the path resolves to
`{tmp_home}/Obsidian/HAKI_Brain`.

**Test Cases**:
1. **Unset env var**: `monkeypatch.delenv("HAKI_OBSIDIAN_VAULT")`; expect
   resolved path to be project-local (will fail on unfixed code).
2. **Empty env var**: `monkeypatch.setenv("HAKI_OBSIDIAN_VAULT", "")`; expect
   project-local default (will fail on unfixed code; `os.environ.get` returns
   `""` and is used as-is).
3. **External path env var**: set to `/tmp/some-other-vault`; on unfixed code
   that path is honoured, on fixed code that path is also honoured (this case
   is preservation, not a bug case — it confirms the override semantics survive
   the refactor).
4. **End-to-end turn (integration)**: start the service with
   `HAKI_OBSIDIAN_VAULT` unset and `cwd` in a temp project layout, run one
   conversation turn, assert
   `{tmp_project}/HAKI_Brain/conversations/{today}.md` exists and contains both
   sides of the exchange. Will fail on unfixed code.

**Expected Counterexamples**:
- Unfixed: resolved path is `~/Obsidian/HAKI_Brain`, not `{project_root}/HAKI_Brain`.
- Possible causes: missing project-local default, no empty-string handling,
  external `.env` override.

### Fix Checking

**Goal**: For all inputs where the bug condition holds, the fixed resolution
produces the project-local path and conversations land there.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  resolved := _resolve_haki_brain_vault_fixed(input.project_root, input.env)
  ASSERT realpath(resolved) = realpath(input.project_root + "/HAKI_Brain")

  start_service_with(resolved)
  run_one_conversation_turn(user="hello", assistant="hi")
  ASSERT exists(resolved + "/conversations/" + today() + ".md")
  ASSERT file_contains(that_file, "hello") AND file_contains(that_file, "hi")
END FOR
```

### Preservation Checking

**Goal**: For all inputs where the bug condition does NOT hold (explicit env var
pointing at the project-local brain folder, or any explicit non-empty path),
fixed behavior matches original.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT _resolve_haki_brain_vault_fixed(input.project_root, input.env)
       = _resolve_haki_brain_vault_original(input.project_root, input.env)

  // and downstream HAKIBrain operations are byte-equivalent
  ASSERT log_conversation_fixed(...)   = log_conversation_original(...)
  ASSERT remember_fact_fixed(...)      = remember_fact_original(...)
  ASSERT search_fixed(...)             = search_original(...)
END FOR
```

**Testing Approach**: Property-based testing (Hypothesis) is recommended for
preservation. Generators produce arbitrary explicit `HAKI_OBSIDIAN_VAULT` values
(non-empty strings that resolve to existing temp directories) and arbitrary
sequences of `HAKIBrain` operations; the property asserts the resolved path and
final on-disk state match between original and fixed. PBT is preferred because
it covers many path shapes (trailing slashes, symlinks, relative paths) and
operation sequences with little code.

**Test Plan**: Capture original behavior on the unfixed code first (resolved
path, files written, content of daily file) for explicit overrides, then assert
the fixed code produces identical artefacts.

**Test Cases**:
1. **Explicit override preservation**: With `HAKI_OBSIDIAN_VAULT` set to a temp
   path, resolved path and `log_conversation` output are byte-identical between
   original and fixed.
2. **Append semantics preservation**: Pre-seed a daily conversations file, run
   a turn, assert append (no overwrite/truncation) on both versions.
3. **Other HAKIBrain ops preservation**: `remember_fact`, `ingest_pending`,
   `search`, `search_and_format`, `load_recent_history` produce identical
   results on the same vault state.
4. **Fire-and-forget preservation**: Orchestrator turn latency does not
   regress; `log_conversation` failure does not raise into the turn path.

### Unit Tests

- `_resolve_haki_brain_vault` returns project-local default when env var is
  unset, empty, or whitespace-only.
- `_resolve_haki_brain_vault` returns `Path(env)` when env var is non-empty.
- `HAKIBrain.init()` creates `raw/`, `processed/`, `wiki/`, `conversations/`
  inside the resolved path.

### Property-Based Tests

- For arbitrary non-empty env values, fixed resolved path equals original
  resolved path (override preservation).
- For arbitrary user/assistant message pairs, after a turn the daily file
  contains both messages (fix property).
- For arbitrary sequences of `HAKIBrain` ops on a vault state and a fixed
  resolved path, the post-state under fixed code equals post-state under
  original code (preservation).

### Integration Tests

- Full service startup with `HAKI_OBSIDIAN_VAULT` unset, in a temp project
  layout: a single conversation turn produces
  `{project_root}/HAKI_Brain/conversations/{today}.md` containing the exchange.
- Service startup with `HAKI_OBSIDIAN_VAULT` explicitly set to a temp path:
  conversations land at the override path, project-local folder is untouched.
- Restart the service mid-day with existing daily file present: subsequent turn
  appends rather than overwrites.
