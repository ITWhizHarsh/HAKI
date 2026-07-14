# Implementation Plan

- [ ] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Conversations Land In Project-Local HAKI_Brain
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples showing that under the bug condition, conversations are NOT written to `{project_root}/HAKI_Brain/conversations/{today}.md`
  - **Scoped PBT Approach**: For deterministic vault-resolution paths, scope the property to concrete failing cases:
    - `HAKI_OBSIDIAN_VAULT` unset (deleted via `monkeypatch.delenv`)
    - `HAKI_OBSIDIAN_VAULT` set to empty string `""`
    - `HAKI_OBSIDIAN_VAULT` set to an external path (e.g. `{tmp_home}/Obsidian/HAKI_Brain`) that does NOT equal `{project_root}/HAKI_Brain`
  - Implement using `pytest` + `monkeypatch` + Hypothesis:
    - Redirect `Path.home()` to a temp directory
    - Use a temp project layout with a `HAKI_Brain/` folder under `project_root`
    - For each input satisfying `isBugCondition` (env unset / empty / external), invoke the vault-resolution block (or `_resolve_haki_brain_vault` once it exists) and exercise one full conversation turn through `HAKIBrain.log_conversation`
    - Assert: `realpath(resolved_vault) == realpath(project_root / "HAKI_Brain")` AND `{project_root}/HAKI_Brain/conversations/{today}.md` exists AND contains both the user message and the assistant reply
  - Bug Condition (from design): `env_HAKI_OBSIDIAN_VAULT IS NULL OR "" OR realpath(env) != realpath(project_root + "/HAKI_Brain")`
  - Expected Behavior assertion (from design Property 1): resolved vault is project-local; daily conversations file exists under it after one turn
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS - resolved path will be `~/Obsidian/HAKI_Brain` (or the external `.env` value), and the project-local `conversations/{today}.md` will not exist
  - Document counterexamples found (e.g. "with env unset, resolved path = `/tmp_home/Obsidian/HAKI_Brain` instead of `{project_root}/HAKI_Brain`")
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4_

- [ ] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Explicit Override And Memory Operations Unchanged
  - **IMPORTANT**: Follow observation-first methodology - run UNFIXED code first and capture actual behavior, then encode it as property assertions
  - Non-bug condition (from design): `HAKI_OBSIDIAN_VAULT` is explicitly set to a non-empty path (override case where `¬isBugCondition` may hold)
  - Observations to capture on UNFIXED code with explicit override pointing at a temp vault:
    - Resolved vault path equals `Path(env_value)` exactly
    - `log_conversation(user, assistant)` writes to `{vault}/conversations/{today}.md` and appends rather than overwrites when the file exists
    - `remember_fact`, `ingest_pending`, `search`, `search_and_format`, `load_recent_history` produce specific outputs / on-disk state for a given vault state
    - Orchestrator dispatches `log_conversation` as fire-and-forget (turn latency unaffected; raised exceptions in `log_conversation` do not propagate into the turn)
  - Property-based tests (Hypothesis) to write:
    - For arbitrary non-empty `HAKI_OBSIDIAN_VAULT` values resolving to existing temp dirs, `_resolve_haki_brain_vault_fixed(env) == _resolve_haki_brain_vault_original(env)` (once fix exists; until then assert against captured observations)
    - For arbitrary `(user, assistant)` message pairs, after one turn the daily file contains both messages and pre-existing content is preserved (append-only)
    - For arbitrary sequences of `HAKIBrain` operations against a fixed vault path, post-state matches the captured baseline
  - Verify these tests PASS on UNFIXED code (this is the baseline to preserve)
  - **EXPECTED OUTCOME**: Tests PASS on unfixed code
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 3. Fix vault-path resolution to default to project-local HAKI_Brain

  - [ ] 3.1 Implement the fix in `Core/haki_core_service.py`
    - Add private helper `_resolve_haki_brain_vault()` that:
      - Computes `project_root = Path(__file__).resolve().parent.parent`
      - Computes `default_vault = project_root / "HAKI_Brain"`
      - Reads `env_value = os.environ.get("HAKI_OBSIDIAN_VAULT", "").strip()`
      - Returns `Path(env_value)` when `env_value` is non-empty, else returns `default_vault`
    - Replace the existing fallback (`Path.home() / "Obsidian" / "HAKI_Brain"`) at the `HAKIBrain(...)` instantiation site with a call to the helper
    - Update the startup log line to record both the resolved path and whether it came from `HAKI_OBSIDIAN_VAULT` or the project-local default
    - Update `Core/.env`: remove the external `HAKI_OBSIDIAN_VAULT=/Users/harshkumarroy/Obsidian/HAKI_Brain` (or repoint to `{project_root}/HAKI_Brain`); document the new project-local default in `Core/.env.example`
    - No changes to `HAKIBrain` itself, the orchestrator, or any other consumers — `HAKIBrain.init()` already creates `raw/`, `processed/`, `wiki/`, `conversations/`
    - _Bug_Condition: `isBugCondition(input)` where `env_HAKI_OBSIDIAN_VAULT` is null/empty or resolves outside `{project_root}/HAKI_Brain`_
    - _Expected_Behavior: resolved vault is `{project_root}/HAKI_Brain`; one turn produces `conversations/{today}.md` with both sides of the exchange (Property 1 in design)_
    - _Preservation: explicit non-empty `HAKI_OBSIDIAN_VAULT` overrides continue to win; all `HAKIBrain` operations (`log_conversation`, `remember_fact`, `ingest_pending`, `search`, `search_and_format`, `load_recent_history`) behave identically; append-only semantics preserved; orchestrator fire-and-forget preserved (Property 2 in design)_
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 3.5_

  - [ ] 3.2 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Conversations Land In Project-Local HAKI_Brain
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - Run the bug condition exploration test from step 1 against the fixed code
    - **EXPECTED OUTCOME**: Test PASSES — resolved path equals `{project_root}/HAKI_Brain` for env-unset / empty / external-path cases, and `conversations/{today}.md` exists under the project-local folder with both messages
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ] 3.3 Verify preservation tests still pass
    - **Property 2: Preservation** - Explicit Override And Memory Operations Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2 against the fixed code
    - **EXPECTED OUTCOME**: Tests PASS — explicit override resolution unchanged; `log_conversation` append semantics unchanged; `remember_fact`, `ingest_pending`, `search`, `search_and_format`, `load_recent_history` all unchanged; orchestrator fire-and-forget behavior unchanged
    - Confirm no regressions in any non-memory subsystem
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 4. Checkpoint - Ensure all tests pass
  - Run the full test suite (exploration, preservation, unit, integration)
  - Confirm Property 1 (bug fixed) and Property 2 (preservation) both hold
  - Manual sanity check: start the service with `HAKI_OBSIDIAN_VAULT` unset, run one turn, confirm `HAKI/HAKI_Brain/conversations/{today}.md` appears in Obsidian
  - Ask the user if any questions arise
