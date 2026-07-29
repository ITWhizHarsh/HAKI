"""Focused zero-coupling fixtures for the static legacy voice archive policy."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_CHECKER_PATH = Path(__file__).resolve().parents[2] / "tools" / "check_voice_archive.py"
_SPEC = importlib.util.spec_from_file_location("check_voice_archive", _CHECKER_PATH)
assert _SPEC and _SPEC.loader
checker = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = checker
_SPEC.loader.exec_module(checker)

_ARCHIVE = "legacy" + "_pipeline_backup"


def _manifest() -> dict[str, object]:
    categories = [
        "configuration",
        "dependency_declaration",
        "startup_path",
        "handler",
        "routing_rule",
        "script",
    ]
    return {
        "schema_version": 1,
        "category_precedence": categories,
        "rules": {category: {"path_globs": [], "content_patterns": []} for category in categories},
    }


def _write_fixture_repository(tmp_path: Path) -> Path:
    (tmp_path / "legacy_voice_manifest.yaml").write_text(
        json.dumps(_manifest()), encoding="utf-8"
    )
    (tmp_path / "Core").mkdir()
    (tmp_path / "Core/pyproject.toml").write_text(
        """[tool.setuptools.packages.find]
exclude = [\"legacy_pipeline_backup\", \"legacy_pipeline_backup.*\"]
""",
        encoding="utf-8",
    )
    records, files = checker.archive.build_inventory(
        tmp_path, _manifest(), include_untracked=True
    )
    checker.archive._write_archive(tmp_path, records, files)
    (tmp_path / _ARCHIVE / "README.md").write_text("inert fixture archive\n", encoding="utf-8")
    return tmp_path


def _violations(tmp_path: Path) -> list[str]:
    return checker.check_repository(
        tmp_path,
        manifest_path=tmp_path / "legacy_voice_manifest.yaml",
        include_untracked=True,
    )


@pytest.mark.parametrize(
    ("relative_path", "content", "expected"),
    [
        ("runtime.py", f"import {_ARCHIVE}\n", "imports the legacy archive"),
        (
            "runtime.py",
            f"import sys\nsys.path.append('{_ARCHIVE}')\n",
            "adds the legacy archive to sys.path",
        ),
        (
            "runtime.py",
            f"import importlib\nimportlib.import_module('{_ARCHIVE}.handler')\n",
            "dynamically imports or discovers the legacy archive",
        ),
        (
            "runtime.py",
            f"import subprocess\nsubprocess.run(['python', '{_ARCHIVE}/tool.py'])\n",
            "executes the legacy archive",
        ),
        ("start.sh", f"python {_ARCHIVE}/boot.py\n", "references the legacy archive"),
        (
            "pyproject.toml",
            f"[tool.setuptools.package-data]\nhaki = ['{_ARCHIVE}/**']\n",
            "bundles or discovers the legacy archive",
        ),
    ],
)
def test_policy_rejects_runtime_and_build_archive_coupling(
    tmp_path: Path, relative_path: str, content: str, expected: str
) -> None:
    root = _write_fixture_repository(tmp_path)
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    assert any(expected in error for error in _violations(root))


def test_policy_rejects_unredacted_archive_secret_residue(tmp_path: Path) -> None:
    root = _write_fixture_repository(tmp_path)
    source = root / "settings.env"
    source.write_text("API_TOKEN=__REDACTED_LEGACY_SECRET__\n", encoding="utf-8")
    manifest = _manifest()
    manifest["rules"] = {
        **manifest["rules"],
        "configuration": {"path_globs": ["settings.env"], "content_patterns": []},
    }
    (root / "legacy_voice_manifest.yaml").write_text(json.dumps(manifest), encoding="utf-8")
    records, files = checker.archive.build_inventory(root, manifest, include_untracked=True)
    checker.archive._write_archive(root, records, files)
    (root / _ARCHIVE / "README.md").write_text("inert fixture archive\n", encoding="utf-8")
    archive_copy = root / _ARCHIVE / "settings.env.txt"
    archive_copy.write_text("API_TOKEN=live-secret\n", encoding="utf-8")

    assert any("unredacted secret residue" in error for error in _violations(root))


def test_policy_rejects_archive_symlink_targets(tmp_path: Path) -> None:
    root = _write_fixture_repository(tmp_path)
    link = root / "runtime-link"
    link.symlink_to(root / _ARCHIVE / "README.md")

    assert any("symlink targets" in error for error in _violations(root))
