#!/usr/bin/env python3
"""Create a deterministic, inert archive of legacy HAKI voice artifacts.

The archiver is deliberately dependency-free.  It discovers only repository
files allowed by ``legacy_voice_manifest.yaml``, classifies every matching file
once, sanitizes supported configuration formats, and writes text-only copies
below ``legacy_pipeline_backup``.  It never imports or executes archived code.

The manifest is JSON-compatible YAML so it can be parsed with the standard
library.  A small, fail-closed YAML reader is available for ordinary mapping
configuration files; complex YAML requires conversion to a supported safe form
before archival.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import re
import stat
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
ARCHIVE_DIRECTORY = "legacy_pipeline_backup"
INVENTORY_FILENAME = "inventory.jsonl"
REDACTION_PLACEHOLDER = "__REDACTED_LEGACY_SECRET__"
CATEGORIES = frozenset(
    {
        "script",
        "handler",
        "configuration",
        "dependency_declaration",
        "startup_path",
        "routing_rule",
    }
)

# The mandated suffix patterns are intentionally stricter than a generic
# ``secret`` substring so harmless keys such as ``monkey`` are not redacted.
_SECRET_KEY_TOKENS = frozenset(
    {"key", "token", "secret", "password", "credential", "credentials", "passwd"}
)
_PROVIDER_SECRET_KEYS = frozenset(
    {
        "deepgramapikey",
        "groqapikey",
        "cartesiaapikey",
        "edgeapikey",
        "geminiapikey",
        "openaiapikey",
        "anthropickey",
        "azureopenaiapikey",
    }
)
_ASSIGNMENT_RE = re.compile(
    r"^(?P<prefix>\s*(?:export\s+)?)(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)(?P<sep>\s*=\s*)(?P<value>.*)$"
)
_TOML_ASSIGNMENT_RE = re.compile(
    r"^(?P<prefix>\s*)(?P<key>(?:[A-Za-z0-9_.-]+|\"[^\"]+\"|'[^']+'))(?P<sep>\s*=\s*)(?P<value>.*)$"
)
_YAML_MAPPING_RE = re.compile(
    r"^(?P<indent> *)(?P<key>(?:\"[^\"]+\"|'[^']+'|[^:#][^:]*?))(?P<sep>\s*:\s*)(?P<value>.*)$"
)
# Recognizes common real credential shapes when parsing fails.  The sensitive
# key detector below catches provider values even when their token shape is
# unfamiliar.
_PROBABLE_SECRET_VALUE_RE = re.compile(
    r"(?:\b(?:sk|rk|pk|AIza)[-_A-Za-z0-9]{16,}\b|\b[A-Za-z0-9+/]{32,}={0,2}\b)"
)


class ArchiveError(RuntimeError):
    """A source cannot be safely archived without risking secret exposure."""


@dataclass(frozen=True)
class DiscoveredArtifact:
    """A manifest-matched legacy source file and its single category."""

    original_path: str
    category: str


@dataclass(frozen=True)
class SanitizedContent:
    """Text-safe archive content and an explicit redaction result."""

    text: str
    sanitized: bool
    status: str


def sha256_bytes(data: bytes) -> str:
    """Return a lower-case SHA-256 digest for raw source or archive bytes."""

    return hashlib.sha256(data).hexdigest()


def _normalise_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def is_secret_key(key: str) -> bool:
    """Return whether *key* names a credential value that must be redacted."""

    normalised = _normalise_key(key)
    if normalised in _PROVIDER_SECRET_KEYS:
        return True
    return any(normalised.endswith(token) for token in _SECRET_KEY_TOKENS)


def _split_unquoted_comment(value: str) -> tuple[str, str]:
    """Split a shell/YAML/TOML value from a trailing ``#`` comment safely."""

    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if char == "#" and quote is None:
            return value[:index].rstrip(), value[index:]
    return value.rstrip(), ""


def _ensure_text(path: Path, raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArchiveError(f"{path}: legacy artifacts must be UTF-8 text") from exc


def _redact_object(value: Any) -> tuple[Any, bool]:
    """Recursively redact dictionary values while retaining every key/path."""

    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        changed = False
        for key, child in value.items():
            if not isinstance(key, str):
                raise ArchiveError("configuration objects must use string keys")
            if is_secret_key(key):
                redacted[key] = REDACTION_PLACEHOLDER
                changed = True
            else:
                redacted_child, child_changed = _redact_object(child)
                redacted[key] = redacted_child
                changed = changed or child_changed
        return redacted, changed
    if isinstance(value, list):
        redacted_items: list[Any] = []
        changed = False
        for child in value:
            redacted_child, child_changed = _redact_object(child)
            redacted_items.append(redacted_child)
            changed = changed or child_changed
        return redacted_items, changed
    return value, False


def _sanitize_env(text: str, path: Path) -> SanitizedContent:
    lines: list[str] = []
    changed = False
    for number, line in enumerate(text.splitlines(keepends=True), start=1):
        newline = "\n" if line.endswith("\n") else ""
        body = line[:-1] if newline else line
        if not body.strip() or body.lstrip().startswith("#"):
            lines.append(line)
            continue
        match = _ASSIGNMENT_RE.match(body)
        if not match:
            raise ArchiveError(f"{path}:{number}: unsupported line-oriented setting")
        if is_secret_key(match.group("key")):
            _, comment = _split_unquoted_comment(match.group("value"))
            suffix = f" {comment}" if comment else ""
            lines.append(
                f"{match.group('prefix')}{match.group('key')}{match.group('sep')}"
                f"{REDACTION_PLACEHOLDER}{suffix}{newline}"
            )
            changed = True
        else:
            lines.append(line)
    return SanitizedContent("".join(lines), changed, "redacted" if changed else "not_required")


def _sanitize_json(text: str, path: Path) -> SanitizedContent:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ArchiveError(f"{path}: invalid JSON configuration") from exc
    redacted, changed = _redact_object(parsed)
    return SanitizedContent(
        json.dumps(redacted, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        changed,
        "redacted" if changed else "not_required",
    )


def _validate_restricted_yaml_line(path: Path, number: int, body: str) -> None:
    stripped = body.strip()
    if not stripped or stripped.startswith("#") or stripped in {"---", "..."}:
        return
    if "\t" in body[: len(body) - len(body.lstrip(" \t"))]:
        raise ArchiveError(f"{path}:{number}: tab indentation is not safely supported")
    if stripped.startswith(("&", "*", "!")) or " <<:" in stripped:
        raise ArchiveError(f"{path}:{number}: YAML aliases/tags/merges are not safely supported")
    if _YAML_MAPPING_RE.match(body) or stripped.startswith("- ") or stripped == "-":
        return
    raise ArchiveError(f"{path}:{number}: unsupported YAML construct")


def _sanitize_yaml(text: str, path: Path) -> SanitizedContent:
    """Sanitize a conservative YAML mapping/list subset without PyYAML.

    The output retains source layout and comments.  Unsupported constructs fail
    closed rather than risking an undetected credential value.
    """

    lines: list[str] = []
    changed = False
    for number, line in enumerate(text.splitlines(keepends=True), start=1):
        newline = "\n" if line.endswith("\n") else ""
        body = line[:-1] if newline else line
        _validate_restricted_yaml_line(path, number, body)
        match = _YAML_MAPPING_RE.match(body)
        if match is None or not is_secret_key(match.group("key").strip(" '\"")):
            lines.append(line)
            continue
        value, comment = _split_unquoted_comment(match.group("value"))
        if value.strip() in {"", "|", ">", "|-", ">-", "|+", ">+"}:
            raise ArchiveError(
                f"{path}:{number}: multi-line or implicit YAML credential cannot be safely redacted"
            )
        suffix = f" {comment}" if comment else ""
        lines.append(
            f"{match.group('indent')}{match.group('key')}{match.group('sep')}"
            f"{REDACTION_PLACEHOLDER}{suffix}{newline}"
        )
        changed = True
    return SanitizedContent("".join(lines), changed, "redacted" if changed else "not_required")


def _sanitize_toml(text: str, path: Path) -> SanitizedContent:
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ArchiveError(f"{path}: invalid TOML configuration") from exc

    lines: list[str] = []
    changed = False
    for number, line in enumerate(text.splitlines(keepends=True), start=1):
        newline = "\n" if line.endswith("\n") else ""
        body = line[:-1] if newline else line
        match = _TOML_ASSIGNMENT_RE.match(body)
        if match is None or not is_secret_key(match.group("key").strip(" '\"")):
            lines.append(line)
            continue
        value, comment = _split_unquoted_comment(match.group("value"))
        if value.lstrip().startswith(('"""', "'''")):
            raise ArchiveError(f"{path}:{number}: multi-line TOML credential cannot be safely redacted")
        suffix = f" {comment}" if comment else ""
        lines.append(
            f"{match.group('prefix')}{match.group('key')}{match.group('sep')}"
            f'"{REDACTION_PLACEHOLDER}"{suffix}{newline}'
        )
        changed = True

    result = "".join(lines)
    try:
        tomllib.loads(result)
    except tomllib.TOMLDecodeError as exc:
        raise ArchiveError(f"{path}: sanitized TOML is not valid") from exc
    return SanitizedContent(result, changed, "redacted" if changed else "not_required")


def _sanitize_plist(raw: bytes, path: Path) -> SanitizedContent:
    try:
        parsed = plistlib.loads(raw)
    except (plistlib.InvalidFileException, ValueError) as exc:
        raise ArchiveError(f"{path}: invalid plist configuration") from exc
    redacted, changed = _redact_object(parsed)
    return SanitizedContent(
        plistlib.dumps(redacted, fmt=plistlib.FMT_XML, sort_keys=True).decode("utf-8"),
        changed,
        "redacted" if changed else "not_required",
    )


def _contains_probable_secret(text: str) -> bool:
    for line in text.splitlines():
        match = _ASSIGNMENT_RE.match(line.strip()) or _YAML_MAPPING_RE.match(line)
        if match and is_secret_key(match.group("key").strip(" '\"")):
            value = match.group("value").strip()
            if value and REDACTION_PLACEHOLDER not in value:
                return True
    return bool(_PROBABLE_SECRET_VALUE_RE.search(text))


def sanitize_configuration(path: Path, raw: bytes) -> SanitizedContent:
    """Parse and redact a supported configuration artifact, failing closed."""

    text = _ensure_text(path, raw)
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name.startswith(".env") or suffix in {".env", ".ini", ".cfg", ".conf"}:
        return _sanitize_env(text, path)
    if suffix == ".json":
        return _sanitize_json(text, path)
    if suffix in {".yaml", ".yml"}:
        return _sanitize_yaml(text, path)
    if suffix == ".toml":
        return _sanitize_toml(text, path)
    if suffix == ".plist":
        return _sanitize_plist(raw, path)
    if _contains_probable_secret(text):
        raise ArchiveError(f"{path}: probable secret in unsupported configuration format")
    raise ArchiveError(f"{path}: unsupported configuration format")


def _load_manifest(path: Path) -> dict[str, Any]:
    """Load JSON-compatible YAML manifest without adding a package dependency."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveError(
            f"{path}: manifest must be JSON-compatible YAML for dependency-free execution"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ArchiveError(f"{path}: unsupported manifest schema")
    rules = payload.get("rules")
    precedence = payload.get("category_precedence")
    exclusions = payload.get("exclude_path_globs", [])
    if not isinstance(rules, dict) or not isinstance(precedence, list):
        raise ArchiveError(f"{path}: rules and category_precedence are required")
    if not isinstance(exclusions, list) or not all(isinstance(value, str) for value in exclusions):
        raise ArchiveError(f"{path}: exclude_path_globs must be a string list")
    if set(precedence) != set(rules) or not set(precedence).issubset(CATEGORIES):
        raise ArchiveError(f"{path}: rules must define each supported category exactly once")
    for category, rule in rules.items():
        if not isinstance(rule, dict):
            raise ArchiveError(f"{path}: {category} rule must be an object")
        for field in ("path_globs", "content_patterns"):
            values = rule.get(field, [])
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise ArchiveError(f"{path}: {category}.{field} must be a string list")
    return payload


def _glob_matches(path: str, pattern: str) -> bool:
    """Match POSIX manifest globs consistently for root and nested paths."""

    candidate = PurePosixPath(path)
    return (
        fnmatchcase(path, pattern)
        or candidate.match(pattern)
        or (pattern.startswith("**/") and _glob_matches(path, pattern[3:]))
    )


def _rule_matches(path: str, text: str, rule: Mapping[str, Any]) -> bool:
    globs = tuple(rule.get("path_globs", []))
    patterns = tuple(rule.get("content_patterns", []))
    path_match = bool(globs) and any(_glob_matches(path, pattern) for pattern in globs)
    content_match = bool(patterns) and any(
        re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) is not None
        for pattern in patterns
    )
    if bool(rule.get("require_content_match", False)):
        return path_match and content_match
    # Scoped rules identify their declared paths.  Content patterns are
    # evidence/validation for those paths, not a repository-wide classifier.
    if globs:
        return path_match
    return content_match


def _tracked_paths(repo_root: Path, include_untracked: bool) -> list[str]:
    """Return sorted regular repository files, defaulting to Git-tracked files."""

    if not include_untracked:
        try:
            output = subprocess.check_output(
                ["git", "-C", str(repo_root), "ls-files", "-z"], stderr=subprocess.DEVNULL
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ArchiveError(
                "repository is not Git-tracked; pass --include-untracked only for a controlled fixture"
            ) from exc
        return sorted(part.decode("utf-8") for part in output.split(b"\0") if part)

    excluded = {".git", ".hypothesis", ".pytest_cache", ".venv", "__pycache__", ARCHIVE_DIRECTORY}
    paths: list[str] = []
    for candidate in repo_root.rglob("*"):
        relative = candidate.relative_to(repo_root)
        if any(part in excluded for part in relative.parts):
            continue
        if candidate.is_symlink():
            raise ArchiveError(f"{relative.as_posix()}: symlinked artifacts are not permitted")
        if candidate.is_file():
            paths.append(relative.as_posix())
    return sorted(paths)


def discover_artifacts(
    repo_root: Path, manifest: Mapping[str, Any], *, include_untracked: bool = False
) -> list[DiscoveredArtifact]:
    """Discover all manifest-matched legacy artifacts in deterministic order."""

    artifacts: list[DiscoveredArtifact] = []
    exclusions = tuple(manifest.get("exclude_path_globs", []))
    for relative_path in _tracked_paths(repo_root, include_untracked):
        if (
            relative_path == INVENTORY_FILENAME
            or relative_path.startswith(f"{ARCHIVE_DIRECTORY}/")
            or any(_glob_matches(relative_path, pattern) for pattern in exclusions)
        ):
            continue
        source = repo_root / relative_path
        explicit_path_match = any(
            any(_glob_matches(relative_path, pattern) for pattern in rule.get("path_globs", []))
            for rule in manifest["rules"].values()
        )
        if source.is_symlink():
            if explicit_path_match:
                raise ArchiveError(f"{relative_path}: symlinked artifacts are not permitted")
            continue
        mode = source.stat().st_mode
        if not stat.S_ISREG(mode):
            if explicit_path_match:
                raise ArchiveError(f"{relative_path}: only regular files may be archived")
            continue
        source_bytes = source.read_bytes()
        try:
            text = source_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            explicit_path_match = any(
                any(_glob_matches(relative_path, pattern) for pattern in rule.get("path_globs", []))
                for rule in manifest["rules"].values()
            )
            if explicit_path_match:
                raise ArchiveError(f"{relative_path}: legacy artifacts must be UTF-8 text") from exc
            # Non-text repository data cannot satisfy a content rule and is
            # outside the manifest's explicit archive set.
            continue
        matches = [
            category
            for category in manifest["category_precedence"]
            if _rule_matches(relative_path, text, manifest["rules"][category])
        ]
        if len(matches) > 1:
            # Precedence makes the result deterministic while allowing concise
            # rules.  Manifest reviewers can see every possible category.
            category = matches[0]
        elif matches:
            category = matches[0]
        else:
            continue
        artifacts.append(DiscoveredArtifact(relative_path, category))
    return artifacts


def _archive_relative_path(original_path: str) -> str:
    return f"{ARCHIVE_DIRECTORY}/{original_path}.txt"


def build_inventory(
    repo_root: Path,
    manifest: Mapping[str, Any],
    *,
    archive_directory: str = ARCHIVE_DIRECTORY,
    include_untracked: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Build deterministic records and in-memory archive files without writing."""

    if archive_directory != ARCHIVE_DIRECTORY:
        raise ArchiveError("archive directory must remain legacy_pipeline_backup")
    records: list[dict[str, Any]] = []
    archived_files: dict[str, bytes] = {}
    for artifact in discover_artifacts(repo_root, manifest, include_untracked=include_untracked):
        source = repo_root / artifact.original_path
        source_bytes = source.read_bytes()
        if artifact.category == "configuration":
            sanitized = sanitize_configuration(source, source_bytes)
        else:
            sanitized = SanitizedContent(
                _ensure_text(source, source_bytes), False, "not_applicable"
            )
        archive_path = _archive_relative_path(artifact.original_path)
        archive_bytes = sanitized.text.encode("utf-8")
        archived_files[archive_path] = archive_bytes
        records.append(
            {
                "archive_path": archive_path,
                "archive_sha256": sha256_bytes(archive_bytes),
                "category": artifact.category,
                "discovered_by": "legacy_voice_manifest.yaml",
                "original_path": artifact.original_path,
                "sanitized": sanitized.sanitized,
                "sanitizer_status": sanitized.status,
                "schema_version": SCHEMA_VERSION,
                "source_sha256": sha256_bytes(source_bytes),
            }
        )
    return records, archived_files


def _inventory_text(records: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        for record in sorted(records, key=lambda record: str(record["original_path"]))
    )


def _write_archive(repo_root: Path, records: Sequence[Mapping[str, Any]], files: Mapping[str, bytes]) -> None:
    archive_root = repo_root / ARCHIVE_DIRECTORY
    archive_root.mkdir(mode=0o755, exist_ok=True)
    expected_paths = set(files)
    for existing in sorted(archive_root.rglob("*.txt")):
        relative_path = existing.relative_to(repo_root).as_posix()
        if relative_path in expected_paths:
            continue
        if existing.is_symlink() or not existing.is_file():
            raise ArchiveError(f"{relative_path}: refusing to remove non-regular archive entry")
        existing.unlink()
    for archive_path, content in sorted(files.items()):
        destination = repo_root / archive_path
        destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        if destination.exists() and destination.is_symlink():
            raise ArchiveError(f"{archive_path}: refusing to overwrite symlink")
        destination.write_bytes(content)
        destination.chmod(0o644)
    inventory = archive_root / INVENTORY_FILENAME
    inventory.write_text(_inventory_text(records), encoding="utf-8")
    inventory.chmod(0o644)


def _check_archive(repo_root: Path, records: Sequence[Mapping[str, Any]], files: Mapping[str, bytes]) -> None:
    inventory_path = repo_root / ARCHIVE_DIRECTORY / INVENTORY_FILENAME
    expected_inventory = _inventory_text(records)
    try:
        actual_inventory = inventory_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArchiveError(f"missing inventory: {inventory_path}") from exc
    if actual_inventory != expected_inventory:
        raise ArchiveError("inventory is not deterministic or is out of date")
    for archive_path, expected_bytes in files.items():
        destination = repo_root / archive_path
        if destination.is_symlink() or not destination.is_file():
            raise ArchiveError(f"missing inert archive file: {archive_path}")
        actual_bytes = destination.read_bytes()
        if actual_bytes != expected_bytes:
            raise ArchiveError(f"archive digest mismatch: {archive_path}")
        if destination.suffix != ".txt":
            raise ArchiveError(f"archive content is not inert text: {archive_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=Path("legacy_voice_manifest.yaml"))
    parser.add_argument("--check", action="store_true", help="verify the tracked archive without writing")
    parser.add_argument(
        "--include-untracked",
        action="store_true",
        help="fixture-only mode; scan regular non-ignored files instead of Git-tracked files",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest
    try:
        manifest = _load_manifest(manifest_path)
        records, files = build_inventory(
            repo_root, manifest, include_untracked=args.include_untracked
        )
        if args.check:
            _check_archive(repo_root, records, files)
        else:
            _write_archive(repo_root, records, files)
    except ArchiveError as exc:
        print(f"archive_legacy_voice: error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"artifacts": len(records), "check": args.check, "inventory": f"{ARCHIVE_DIRECTORY}/{INVENTORY_FILENAME}"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
