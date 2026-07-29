#!/usr/bin/env python3
"""Statically enforce the dependency boundary of the replacement voice package.

This checker reads Python ASTs only. It keeps ``core.voice`` isolated from
retired STT/TTS/model-provider services and the process-wide orchestrator, so
voice sessions own their pipeline and playback-confirmed context.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable, Sequence


VOICE_PACKAGE_PATH = Path("Core/core/voice")
PERMITTED_CORE_IMPORT_PREFIXES = (
    "core.voice",
    "core.memory.haki_brain",
    "core.automation.screen_agent",
    "core.ipc.voice_protocol",
)
_PROHIBITED_CORE_IMPORT_PREFIXES = {
    "core.model_provider": "imports an excluded model-provider runtime",
    "core.orchestrator": "imports the process-wide Orchestrator runtime",
}
_ARCHIVE_MODULE = "legacy_pipeline_backup"
_DYNAMIC_IMPORTS = {"__import__", "importlib.import_module", "importlib.util.find_spec"}


class VoiceImportBoundaryError(RuntimeError):
    """Raised when the requested voice package cannot be inspected."""


def _is_prefixed(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _source_package(relative_path: Path) -> tuple[str, ...]:
    module_parts = relative_path.with_suffix("").parts
    return module_parts[:-1]


def _resolve_import_module(relative_path: Path, node: ast.ImportFrom) -> str:
    """Resolve an ``ImportFrom`` target relative to the scanned module."""
    if node.level == 0:
        return node.module or ""

    package = _source_package(relative_path)
    keep = len(package) - (node.level - 1)
    if keep < 0:
        return node.module or ""
    base = package[:keep]
    if node.module:
        return ".".join((*base, *node.module.split(".")))
    return ".".join(base)


def _import_violation(module: str) -> str | None:
    for prefix, description in _PROHIBITED_CORE_IMPORT_PREFIXES.items():
        if _is_prefixed(module, prefix):
            return description
    if _is_prefixed(module, _ARCHIVE_MODULE):
        return "imports the static legacy archive"
    if module == "core" or module.startswith("core."):
        if not any(_is_prefixed(module, allowed) for allowed in PERMITTED_CORE_IMPORT_PREFIXES):
            return "imports a non-permitted Core dependency"
    return None


def _qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _literal_string(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def scan_voice_module(relative_path: Path, source: str) -> list[str]:
    """Return import-boundary violations for one ``core.voice`` source file."""
    try:
        tree = ast.parse(source, filename=relative_path.as_posix())
    except SyntaxError as exc:
        return [f"{relative_path}:{exc.lineno}: cannot parse voice module"]

    violations: list[str] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 1)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if violation := _import_violation(alias.name):
                    violations.append(f"{relative_path}:{line}: {violation} ({alias.name})")
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_import_module(relative_path, node)
            if violation := _import_violation(module):
                violations.append(f"{relative_path}:{line}: {violation} ({module})")
        elif isinstance(node, ast.Attribute):
            if node.attr == "_routing_order":
                violations.append(f"{relative_path}:{line}: reuses LLMRouter._routing_order()")
            elif node.attr == "_conversation_history":
                violations.append(
                    f"{relative_path}:{line}: accesses process-wide Orchestrator conversation history"
                )
        elif isinstance(node, ast.Call):
            qualified = _qualified_name(node.func)
            if qualified in _DYNAMIC_IMPORTS and node.args:
                module = _literal_string(node.args[0])
                if module and (violation := _import_violation(module)):
                    violations.append(f"{relative_path}:{line}: dynamically {violation} ({module})")
    return violations


def _iter_voice_modules(package_root: Path) -> Iterable[Path]:
    yield from sorted(path for path in package_root.rglob("*.py") if path.is_file())


def check_voice_package(repo_root: Path) -> list[str]:
    """Return all prohibited imports or legacy-router accesses in ``core.voice``."""
    package_root = repo_root / VOICE_PACKAGE_PATH
    if not package_root.is_dir():
        raise VoiceImportBoundaryError(f"missing voice package: {VOICE_PACKAGE_PATH}")

    violations: list[str] = []
    for source_path in _iter_voice_modules(package_root):
        relative_path = source_path.relative_to(repo_root)
        try:
            source = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            violations.append(f"{relative_path}: cannot read voice module: {exc}")
            continue
        violations.extend(scan_voice_module(relative_path, source))
    return sorted(set(violations))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        violations = check_voice_package(args.repo_root.resolve())
    except VoiceImportBoundaryError as exc:
        print(f"check_voice_import_boundary: error: {exc}")
        return 1

    if violations:
        for violation in violations:
            print(f"check_voice_import_boundary: error: {violation}")
        return 1
    print("check_voice_import_boundary: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
