"""Focused fixture tests for deterministic legacy voice archival."""

from __future__ import annotations

import importlib.util
import plistlib
import sys
from pathlib import Path

import pytest


_ARCHIVER_PATH = Path(__file__).resolve().parents[2] / "tools" / "archive_legacy_voice.py"
_SPEC = importlib.util.spec_from_file_location("archive_legacy_voice", _ARCHIVER_PATH)
assert _SPEC and _SPEC.loader
archive = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = archive
_SPEC.loader.exec_module(archive)


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "category_precedence": [
            "configuration",
            "dependency_declaration",
            "startup_path",
            "handler",
            "routing_rule",
            "script",
        ],
        "rules": {
            "configuration": {"path_globs": ["settings.env"], "content_patterns": []},
            "dependency_declaration": {"path_globs": ["requirements.txt"], "content_patterns": []},
            "startup_path": {"path_globs": ["start.sh"], "content_patterns": []},
            "handler": {"path_globs": ["handler.py"], "content_patterns": []},
            "routing_rule": {"path_globs": ["routes.py"], "content_patterns": []},
            "script": {"path_globs": ["legacy.py"], "content_patterns": []},
        },
    }


def test_fixture_inventory_covers_all_categories_and_is_deterministic(tmp_path: Path) -> None:
    fixtures = {
        "settings.env": "HAKI_DEEPGRAM_API_KEY=real-secret\nVOICE_PATH=/tmp/voice.wav\n",
        "requirements.txt": "deepgram-sdk==3.5.0\n",
        "start.sh": "#!/bin/sh\n./legacy.py\n",
        "handler.py": "def handle_audio(): pass\n",
        "routes.py": "ROUTE = 'groq'\n",
        "legacy.py": "print('legacy voice')\n",
    }
    for relative_path, content in fixtures.items():
        (tmp_path / relative_path).write_text(content, encoding="utf-8")

    first_records, first_files = archive.build_inventory(
        tmp_path, _manifest(), include_untracked=True
    )
    second_records, second_files = archive.build_inventory(
        tmp_path, _manifest(), include_untracked=True
    )

    assert first_records == second_records
    assert first_files == second_files
    assert {record["category"] for record in first_records} == {
        "script",
        "handler",
        "configuration",
        "dependency_declaration",
        "startup_path",
        "routing_rule",
    }
    config = next(record for record in first_records if record["original_path"] == "settings.env")
    assert config["sanitized"] is True
    assert config["sanitizer_status"] == "redacted"
    archived = first_files[config["archive_path"]].decode("utf-8")
    assert "HAKI_DEEPGRAM_API_KEY=__REDACTED_LEGACY_SECRET__" in archived
    assert "VOICE_PATH=/tmp/voice.wav" in archived
    assert all(path.endswith(".txt") for path in first_files)


@pytest.mark.parametrize(
    ("filename", "raw", "expected_key"),
    [
        ("settings.env", b"API_TOKEN=secret\nPATH=/voice.wav\n", "API_TOKEN=__REDACTED_LEGACY_SECRET__"),
        ("settings.json", b'{"api_key":"secret","path":"/voice.wav"}', '"api_key": "__REDACTED_LEGACY_SECRET__"'),
        ("settings.yaml", b"api_key: secret\npath: /voice.wav\n", "api_key: __REDACTED_LEGACY_SECRET__"),
        ("settings.toml", b'api_key = "secret"\npath = "/voice.wav"\n', 'api_key = "__REDACTED_LEGACY_SECRET__"'),
    ],
)
def test_supported_text_config_parsers_redact_values(
    tmp_path: Path, filename: str, raw: bytes, expected_key: str
) -> None:
    source = tmp_path / filename
    source.write_bytes(raw)

    sanitized = archive.sanitize_configuration(source, raw)

    assert sanitized.sanitized is True
    assert expected_key in sanitized.text
    assert "/voice.wav" in sanitized.text
    assert "secret" not in sanitized.text


def test_plist_parser_redacts_values_and_preserves_paths(tmp_path: Path) -> None:
    source = tmp_path / "settings.plist"
    raw = plistlib.dumps({"api_token": "secret", "voice_path": "/voice.wav"})
    source.write_bytes(raw)

    sanitized = archive.sanitize_configuration(source, raw)

    assert sanitized.sanitized is True
    result = plistlib.loads(sanitized.text.encode("utf-8"))
    assert result == {
        "api_token": archive.REDACTION_PLACEHOLDER,
        "voice_path": "/voice.wav",
    }


def test_unparseable_probable_yaml_secret_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "settings.yaml"
    raw = b"api_key: |\n  real-secret\n"
    source.write_bytes(raw)

    with pytest.raises(archive.ArchiveError, match="cannot be safely redacted"):
        archive.sanitize_configuration(source, raw)
