#!/usr/bin/env python3
"""Fail closed when the static legacy voice archive is coupled to runtime code.

The checker deliberately uses the archive migrator's deterministic inventory
logic before inspecting the repository for references that could import,
discover, execute, bundle, or select ``legacy_pipeline_backup``.  It is safe to
run locally and in CI: it only reads repository files and exits non-zero for
any policy violation.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _load_archiver() -> Any:
    """Load the sibling migrator without relying on the caller's sys.path."""

    module_name = "_haki_archive_legacy_voice"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).with_name("archive_legacy_voice.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load archive migrator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


archive = _load_archiver()
ARCHIVE_DIRECTORY = archive.ARCHIVE_DIRECTORY
INVENTORY_FILENAME = archive.INVENTORY_FILENAME
REDACTION_PLACEHOLDER = archive.REDACTION_PLACEHOLDER

_POLICY_EXCLUDED_PREFIXES = (
    ".git/",
    ".hypothesis/",
    ".pytest_cache/",
    ".venv/",
    "__pycache__/",
    ".kiro/",
    f"{ARCHIVE_DIRECTORY}/",
    "tools/archive_legacy_voice.py",
    "tools/check_voice_archive.py",
)
_IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".hypothesis",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pyright",
    ".build",
    "build",
    "dist",
}
_SOURCE_SUFFIXES = {".py", ".pyi", ".sh", ".bash", ".zsh", ".swift", ".js", ".ts"}
_METADATA_NAMES = {
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "MANIFEST.in",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "Package.swift",
    "Package.resolved",
}
_METADATA_SUFFIXES = {".toml", ".yaml", ".yml", ".json", ".plist", ".cfg", ".ini"}
_DYNAMIC_IMPORT_CALLS = {
    "importlib.import_module",
    "importlib.util.find_spec",
    "pkgutil.get_loader",
    "pkgutil.iter_modules",
    "builtins.__import__",
    "__import__",
}
_EXECUTION_CALL_SUFFIXES = {
    "run",
    "call",
    "check_call",
    "check_output",
    "Popen",
    "system",
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "create_subprocess_exec",
    "create_subprocess_shell",
}
_WRITE_CALL_SUFFIXES = {
    "open",
    "write_text",
    "write_bytes",
    "touch",
    "copy",
    "copy2",
    "copytree",
    "move",
}
_LEGACY_PROVIDER_RE = re.compile(
    r"(?:deepgram|groq|cartesia|edge[-_ ]?tts|kokoro|chattts|\bafplay\b|\bsay\b)",
    re.IGNORECASE,
)
_FALLBACK_IDENTIFIER_RE = re.compile(r"(?:legacy|archive).*(?:fallback|compat|switch|mode)|(?:fallback|compat|switch|mode).*(?:legacy|archive)", re.IGNORECASE)


class PolicyError(RuntimeError):
    """Raised for invalid command-line policy setup."""


def _is_policy_excluded(relative_path: str) -> bool:
    return any(relative_path == prefix or relative_path.startswith(prefix) for prefix in _POLICY_EXCLUDED_PREFIXES)


def _iter_repository_paths(repo_root: Path, *, include_untracked: bool) -> list[str]:
    """Return deterministic file/symlink paths while excluding generated state."""

    if not include_untracked:
        try:
            output = subprocess.check_output(
                ["git", "-C", str(repo_root), "ls-files", "-z"], stderr=subprocess.DEVNULL
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PolicyError(
                "repository is not Git-tracked; pass --include-untracked only for controlled fixtures"
            ) from exc
        return sorted(part.decode("utf-8") for part in output.split(b"\0") if part)

    paths: list[str] = []
    for root, directories, filenames in os.walk(repo_root, followlinks=False):
        directories[:] = sorted(
            directory for directory in directories if directory not in _IGNORED_DIRECTORY_NAMES
        )
        root_path = Path(root)
        for name in sorted(filenames):
            candidate = root_path / name
            paths.append(candidate.relative_to(repo_root).as_posix())
        for name in directories:
            candidate = root_path / name
            if candidate.is_symlink():
                paths.append(candidate.relative_to(repo_root).as_posix())
    return sorted(set(paths))


def _qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _string_value(node: ast.AST, bindings: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                rendered = _string_value(value.value, bindings)
                if rendered is None:
                    return None
                parts.append(rendered)
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _string_value(node.left, bindings)
        right = _string_value(node.right, bindings)
        return left + right if left is not None and right is not None else None
    return None


def _assignment_bindings(tree: ast.AST) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = _string_value(node.value, bindings)
            if value is None:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = _string_value(node.value, bindings) if node.value is not None else None
            if value is not None:
                bindings[node.target.id] = value
    return bindings


def _node_contains_archive_path(node: ast.AST, bindings: Mapping[str, str]) -> bool:
    return any(
        (value := _string_value(candidate, bindings)) is not None and ARCHIVE_DIRECTORY in value
        for candidate in ast.walk(node)
    )


def _is_voice_runtime(relative_path: str) -> bool:
    normalized = relative_path.lower()
    return (
        "/core/voice/" in f"/{normalized}"
        or normalized.startswith("core/tests/voice/")
        or (normalized.startswith("haki/sources/subsystems/") and "voice" in Path(normalized).name)
    )


def _archived_python_modules(records: Iterable[Mapping[str, Any]]) -> set[str]:
    modules: set[str] = set()
    for record in records:
        original_path = str(record.get("original_path", ""))
        if not original_path.endswith(".py") or not original_path.startswith("Core/"):
            continue
        module_path = original_path.removeprefix("Core/").removesuffix(".py")
        if module_path.endswith("/__init__"):
            module_path = module_path[: -len("/__init__")]
        modules.add(module_path.replace("/", "."))
    return modules


def _scan_python(
    relative_path: str, text: str, archived_modules: set[str]
) -> list[str]:
    """Use AST semantics for imports, path mutation, dynamic loading, and execution."""

    errors: list[str] = []
    try:
        tree = ast.parse(text, filename=relative_path)
    except SyntaxError as exc:
        return [f"{relative_path}:{exc.lineno}: Python source cannot be parsed for archive policy"]

    bindings = _assignment_bindings(tree)
    voice_runtime = _is_voice_runtime(relative_path)
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 1)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == ARCHIVE_DIRECTORY or alias.name.startswith(f"{ARCHIVE_DIRECTORY}."):
                    errors.append(f"{relative_path}:{line}: imports the legacy archive")
                if voice_runtime and alias.name in archived_modules:
                    errors.append(f"{relative_path}:{line}: imports an archived legacy module {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == ARCHIVE_DIRECTORY or module.startswith(f"{ARCHIVE_DIRECTORY}."):
                errors.append(f"{relative_path}:{line}: imports the legacy archive")
            if voice_runtime and (module in archived_modules or any(module.startswith(f"{item}.") for item in archived_modules)):
                errors.append(f"{relative_path}:{line}: imports an archived legacy module {module}")
        elif isinstance(node, ast.Name) and _FALLBACK_IDENTIFIER_RE.search(node.id):
            errors.append(f"{relative_path}:{line}: exposes an archive/legacy fallback switch ({node.id})")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if voice_runtime and _LEGACY_PROVIDER_RE.search(value):
                errors.append(f"{relative_path}:{line}: selects a legacy voice provider or playback command")
            if _FALLBACK_IDENTIFIER_RE.search(value):
                errors.append(f"{relative_path}:{line}: exposes an archive/legacy fallback switch")
        elif isinstance(node, ast.Call):
            qualified = _qualified_name(node.func) or "<call>"
            call_suffix = qualified.rsplit(".", 1)[-1]
            has_archive_path = _node_contains_archive_path(node, bindings)
            if qualified in _DYNAMIC_IMPORT_CALLS and has_archive_path:
                errors.append(f"{relative_path}:{line}: dynamically imports or discovers the legacy archive")
            if qualified.startswith("sys.path.") and has_archive_path:
                errors.append(f"{relative_path}:{line}: adds the legacy archive to sys.path")
            if call_suffix in _EXECUTION_CALL_SUFFIXES and has_archive_path:
                errors.append(f"{relative_path}:{line}: executes the legacy archive through {qualified}")
            if call_suffix in _WRITE_CALL_SUFFIXES and has_archive_path:
                errors.append(f"{relative_path}:{line}: generates or writes code targeting the legacy archive")
            if has_archive_path and (
                call_suffix in {"glob", "rglob", "iterdir", "walk", "find_spec"}
                or "discover" in qualified.lower()
            ):
                errors.append(f"{relative_path}:{line}: discovers the legacy archive")
            if has_archive_path and not any(
                message.startswith(f"{relative_path}:{line}:") for message in errors
            ):
                errors.append(f"{relative_path}:{line}: references the legacy archive at runtime")
    return errors


def _scan_text_source(relative_path: str, text: str) -> list[str]:
    """Cover shell/Swift/JS launch and process constructs unavailable to Python AST."""

    errors: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if ARCHIVE_DIRECTORY in line:
            errors.append(f"{relative_path}:{line_number}: references the legacy archive in source or launch code")
        if _FALLBACK_IDENTIFIER_RE.search(line):
            errors.append(f"{relative_path}:{line_number}: exposes an archive/legacy fallback switch")
        if _is_voice_runtime(relative_path) and _LEGACY_PROVIDER_RE.search(line):
            errors.append(f"{relative_path}:{line_number}: selects a legacy voice provider or playback command")
    return errors


def _walk_metadata(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_metadata(child, path + (str(key).lower(),))
    elif isinstance(value, list):
        for child in value:
            yield from _walk_metadata(child, path)
    elif isinstance(value, str):
        yield path, value


def _metadata_reference_is_exclusion(path: tuple[str, ...]) -> bool:
    return any("exclude" in part or part in {"prune", "ignore"} for part in path)


def _scan_metadata(relative_path: str, text: str) -> list[str]:
    errors: list[str] = []
    name = Path(relative_path).name
    if name == "pyproject.toml":
        try:
            parsed = tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            return [f"{relative_path}: invalid build metadata: {exc}"]
        for path, value in _walk_metadata(parsed):
            if ARCHIVE_DIRECTORY in value and not _metadata_reference_is_exclusion(path):
                errors.append(f"{relative_path}: bundles or discovers the legacy archive in package metadata")
        return errors
    if name == "MANIFEST.in":
        for line_number, line in enumerate(text.splitlines(), start=1):
            if ARCHIVE_DIRECTORY not in line:
                continue
            if not re.match(r"\s*(?:prune|exclude)\s+legacy_pipeline_backup(?:/|\s|$)", line):
                errors.append(f"{relative_path}:{line_number}: bundles the legacy archive in a source manifest")
        return errors
    for line_number, line in enumerate(text.splitlines(), start=1):
        if ARCHIVE_DIRECTORY in line and not re.search(r"\b(?:exclude|prune|ignore)\b", line, re.IGNORECASE):
            errors.append(f"{relative_path}:{line_number}: bundles or discovers the legacy archive in metadata")
    return errors


def _scan_symlinks(repo_root: Path, paths: Iterable[str]) -> list[str]:
    errors: list[str] = []
    archive_root = (repo_root / ARCHIVE_DIRECTORY).resolve()
    for relative_path in paths:
        candidate = repo_root / relative_path
        if not candidate.is_symlink():
            continue
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            resolved = candidate
        inside_archive = relative_path.startswith(f"{ARCHIVE_DIRECTORY}/")
        try:
            target_is_archive = resolved.is_relative_to(archive_root)
        except ValueError:
            target_is_archive = False
        if inside_archive or target_is_archive or ARCHIVE_DIRECTORY in os.readlink(candidate):
            errors.append(f"{relative_path}: symlink targets or exists inside the legacy archive")
    return errors


def _check_archive_layout(
    repo_root: Path, records: Sequence[Mapping[str, Any]], files: Mapping[str, bytes]
) -> list[str]:
    """Verify archive content remains inert and redacted beyond digest validation."""

    errors: list[str] = []
    archive_root = repo_root / ARCHIVE_DIRECTORY
    expected_paths = set(files) | {
        f"{ARCHIVE_DIRECTORY}/{INVENTORY_FILENAME}",
        f"{ARCHIVE_DIRECTORY}/README.md",
    }
    if not archive_root.is_dir():
        return [f"missing archive directory: {ARCHIVE_DIRECTORY}"]
    if not (archive_root / "README.md").is_file():
        errors.append(f"{ARCHIVE_DIRECTORY}: missing static archive README")
    for candidate in archive_root.rglob("*"):
        relative_path = candidate.relative_to(repo_root).as_posix()
        if candidate.is_symlink():
            errors.append(f"{relative_path}: archive entries must not be symlinks")
            continue
        if candidate.is_dir():
            continue
        if relative_path not in expected_paths:
            errors.append(f"{relative_path}: untracked or non-inventory archive artifact")
        if candidate.name == "__init__.py" or candidate.suffix not in {".txt", ".jsonl", ".md"}:
            errors.append(f"{relative_path}: archive contains importable or executable content")
        if candidate.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            errors.append(f"{relative_path}: archive content must not be executable")

    for record in records:
        if record.get("category") != "configuration":
            continue
        archive_path = repo_root / str(record["archive_path"])
        original_path = Path(str(record["original_path"]))
        try:
            raw = archive_path.read_bytes()
            sanitized = archive.sanitize_configuration(original_path, raw)
        except (OSError, archive.ArchiveError) as exc:
            errors.append(f"{record['archive_path']}: cannot verify secret redaction: {exc}")
            continue
        if raw.decode("utf-8", errors="replace") != sanitized.text:
            errors.append(f"{record['archive_path']}: unredacted secret residue or non-canonical configuration")
        if REDACTION_PLACEHOLDER not in sanitized.text and bool(record.get("sanitized")):
            errors.append(f"{record['archive_path']}: expected redaction placeholder is missing")
    return errors


def _check_package_exclusion(repo_root: Path) -> list[str]:
    """Require an explicit Python packaging exclusion and inert Swift target paths."""

    errors: list[str] = []
    pyproject = repo_root / "Core/pyproject.toml"
    if pyproject.is_file():
        try:
            parsed = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            errors.append(f"Core/pyproject.toml: cannot inspect package exclusion: {exc}")
        else:
            find = (
                parsed.get("tool", {})
                .get("setuptools", {})
                .get("packages", {})
                .get("find", {})
            )
            excludes = find.get("exclude", []) if isinstance(find, dict) else []
            if not isinstance(excludes, list) or not any(
                isinstance(item, str) and item.startswith(ARCHIVE_DIRECTORY) for item in excludes
            ):
                errors.append("Core/pyproject.toml: must explicitly exclude legacy_pipeline_backup from package discovery")
    package_swift = repo_root / "HAKI/Package.swift"
    if package_swift.is_file() and ARCHIVE_DIRECTORY in package_swift.read_text(encoding="utf-8"):
        errors.append("HAKI/Package.swift: application bundle target references legacy_pipeline_backup")
    return errors


def check_repository(repo_root: Path, *, manifest_path: Path, include_untracked: bool = False) -> list[str]:
    """Return all archive integrity and zero-coupling policy violations."""

    repo_root = repo_root.resolve()
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    files: dict[str, bytes] = {}
    try:
        manifest = archive._load_manifest(manifest_path)
        records, files = archive.build_inventory(
            repo_root, manifest, include_untracked=include_untracked
        )
    except (OSError, archive.ArchiveError) as exc:
        errors.append(f"inventory: {exc}")
    else:
        try:
            archive._check_archive(repo_root, records, files)
        except (OSError, archive.ArchiveError) as exc:
            errors.append(f"inventory: {exc}")

    errors.extend(_check_archive_layout(repo_root, records, files))
    errors.extend(_check_package_exclusion(repo_root))
    try:
        paths = _iter_repository_paths(repo_root, include_untracked=include_untracked)
    except PolicyError as exc:
        return errors + [str(exc)]
    errors.extend(_scan_symlinks(repo_root, paths))
    archived_modules = _archived_python_modules(records)
    for relative_path in paths:
        if _is_policy_excluded(relative_path):
            continue
        candidate = repo_root / relative_path
        if candidate.is_symlink() or not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        suffix = candidate.suffix.lower()
        name = candidate.name
        if suffix in {".py", ".pyi"}:
            errors.extend(_scan_python(relative_path, text, archived_modules))
        elif suffix in _SOURCE_SUFFIXES:
            errors.extend(_scan_text_source(relative_path, text))
        if name in _METADATA_NAMES or suffix in _METADATA_SUFFIXES:
            errors.extend(_scan_metadata(relative_path, text))
    return sorted(set(errors))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=Path("legacy_voice_manifest.yaml"))
    parser.add_argument(
        "--include-untracked",
        action="store_true",
        help="fixture-only mode; inspect regular untracked files rather than Git-tracked files",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest
    violations = check_repository(
        repo_root, manifest_path=manifest_path, include_untracked=args.include_untracked
    )
    if violations:
        for violation in violations:
            print(f"check_voice_archive: error: {violation}", file=sys.stderr)
        return 1
    print(json.dumps({"archive": ARCHIVE_DIRECTORY, "check": "passed"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
