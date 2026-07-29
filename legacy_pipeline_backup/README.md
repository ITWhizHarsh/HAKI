# Legacy Voice Pipeline Static Archive

This directory is a repository-tracked, inert reference archive of the legacy HAKI cloud and turn-based voice pipeline. It is not a package, launch target, configuration source, dependency source, service, compatibility mode, or runtime fallback.

## Contents

- `inventory.jsonl` has one deterministic record per archived source artifact. Each record maps its original repository-relative path and category to an archive-relative `.txt` copy, with SHA-256 digests for both source and archive content.
- Each archived source is stored beneath this directory at its original repository-relative path with an additional `.txt` suffix. The copies are static text and have non-executable `0644` permissions.
- Supported configuration files are parsed before archival. Credential values are replaced with `__REDACTED_LEGACY_SECRET__`; keys, non-secret values, and relative paths are retained.

## Regeneration and verification

Regenerate the archive only through the approved deterministic migrator from the repository root:

```sh
python3.11 tools/archive_legacy_voice.py --repo-root . --manifest legacy_voice_manifest.yaml
```

Verify the existing archive without writing it:

```sh
python3.11 tools/archive_legacy_voice.py --repo-root . --manifest legacy_voice_manifest.yaml --check
```

Do not import, execute, package, launch, add to `sys.path`, or use any archived item as a fallback. The archive deliberately contains no importable modules, package markers, launch scripts, or symlinks.
