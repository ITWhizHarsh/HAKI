"""Property 1: Archive inventory and redaction preservation.

Feature: realtime-local-voice-agent, Property 1: Archive inventory and redaction preservation

For all discovered legacy artifact sets and supported configuration secret values,
migration produces exactly one inventory entry and one inert archived copy per
source path, preserves the artifact category and source SHA-256 mapping, and
replaces every credential value with the approved placeholder without changing
its configuration key.

**Validates: Requirements 1.1, 1.3, 1.4**

Design reference: §1, Property 1; V-ARCHIVE

Covers (100+ Hypothesis-generated cases plus seeded boundary cases):
- Exactly one inventory entry per source path
- Exactly one inert archived copy per source path
- Category and source SHA-256 preserved in inventory
- Archive copy content is inert (no importable Python modules, no executable
  entrypoints): verified by .txt extension and absence of package markers
- Archive path uses .txt extension, not .py/.sh/etc.
- For all supported config formats (.env, JSON, YAML, TOML):
  - Credential values redacted to __REDACTED_LEGACY_SECRET__
  - Configuration keys preserved (not redacted)
  - Paths preserved (not redacted)
  - Non-secret values preserved
- Fail-closed: malformed configs raise ArchiveError without exposing secrets
- Seeded malformed/ambiguous configs verify fail-closed behavior
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Import the archiver module directly (no package dependency)
# ---------------------------------------------------------------------------

_ARCHIVER_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "archive_legacy_voice.py"
)
_SPEC = importlib.util.spec_from_file_location("archive_legacy_voice", _ARCHIVER_PATH)
assert _SPEC and _SPEC.loader, f"Could not find archiver at {_ARCHIVER_PATH}"
archive = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = archive
_SPEC.loader.exec_module(archive)  # type: ignore[union-attr]

REDACTED = archive.REDACTION_PLACEHOLDER  # "__REDACTED_LEGACY_SECRET__"

# ---------------------------------------------------------------------------
# Supported categories
# ---------------------------------------------------------------------------

_ALL_CATEGORIES = [
    "configuration",
    "dependency_declaration",
    "startup_path",
    "handler",
    "routing_rule",
    "script",
]

# ---------------------------------------------------------------------------
# Manifest factory — used throughout the property tests.
# ---------------------------------------------------------------------------

def _manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "category_precedence": list(_ALL_CATEGORIES),
        "rules": {
            "configuration": {
                "path_globs": [
                    "*.env", "*.json", "*.yaml", "*.yml", "*.toml",
                    "*.cfg", "*.conf", ".env*",
                ],
                "content_patterns": [],
            },
            "dependency_declaration": {
                "path_globs": ["requirements.txt", "pyproject.toml", "Package.swift"],
                "content_patterns": [],
            },
            "startup_path": {
                "path_globs": ["start.sh", "start_*.sh", "haki_service.py"],
                "content_patterns": [],
            },
            "handler": {
                "path_globs": ["handler*.py", "stt_engine.py", "tts_engine.py"],
                "content_patterns": [],
            },
            "routing_rule": {
                "path_globs": ["routes*.py", "llm_router.py"],
                "content_patterns": [],
            },
            "script": {
                "path_globs": ["legacy*.py", "legacy*.sh"],
                "content_patterns": [],
            },
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_in_dir(
    root: Path, files: dict[str, bytes]
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Write files to root, run build_inventory, return (records, archived)."""
    for relative, content in files.items():
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    return archive.build_inventory(root, _manifest(), include_untracked=True)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Secret key names — all must be redacted.
# Note: the archiver uses suffix matching on normalized keys.
# 'NONSECRET' normalizes to 'nonsecret' which ends with 'secret' → IS a secret key.
# So we use only keys that end with the actual secret suffixes.
_secret_key_st = st.sampled_from([
    "API_KEY",       # ends with 'key'
    "API_TOKEN",     # ends with 'token'
    "MY_SECRET",     # ends with 'secret'
    "DB_PASSWORD",   # ends with 'password'
    "AUTH_CREDENTIAL",  # ends with 'credential'
    "ACCESS_CREDENTIALS",  # ends with 'credentials'
    "DB_PASSWD",     # ends with 'passwd'
    "DEEPGRAM_API_KEY",
    "GROQ_API_KEY",
    "CARTESIA_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
])

# Non-secret keys — none must end with a secret suffix token.
# Confirmed safe via is_secret_key checks: HOST, BASE_URL, PORT, LOG_LEVEL,
# MAX_RETRIES, ENVIRONMENT, REGION, VOICE_PATH all return False.
_safe_key_st = st.sampled_from([
    "VOICE_PATH",
    "HOST",
    "PORT",
    "REGION",
    "ENVIRONMENT",
    "LOG_LEVEL",
    "MAX_RETRIES",
    "BASE_URL",
])

# Safe (non-secret) values
_safe_value_st = st.one_of(
    st.just("/tmp/voice.wav"),
    st.just("localhost"),
    st.just("8080"),
    st.just("production"),
    st.just("us-east-1"),
    st.just(""),
)

# Short plausible secret values (no special chars that break config parsers)
_secret_value_st = st.one_of(
    st.just("sk-abc123XYZ789realkey"),
    st.just("gsk_realtoken1234567890abcdef"),
    st.just("my_super_secret_password"),
    st.just("hunter2"),
    st.just("deadbeefcafe1234"),
)


# --- Per-format content generators ---

@st.composite
def _env_content_st(draw) -> tuple[bytes, str, str, str]:
    """Return (raw_bytes, secret_key, safe_key, safe_value)."""
    secret_key = draw(_secret_key_st)
    secret_val = draw(_secret_value_st)
    safe_key = draw(_safe_key_st)
    safe_val = draw(_safe_value_st)
    content = f"{secret_key}={secret_val}\n{safe_key}={safe_val}\n"
    return content.encode(), secret_key, safe_key, safe_val


@st.composite
def _json_content_st(draw) -> tuple[bytes, str, str, str]:
    secret_key = draw(_secret_key_st).lower()
    secret_val = draw(_secret_value_st)
    safe_key = draw(_safe_key_st).lower()
    safe_val = draw(_safe_value_st)
    payload = {secret_key: secret_val, safe_key: safe_val}
    return json.dumps(payload).encode(), secret_key, safe_key, safe_val


@st.composite
def _yaml_content_st(draw) -> tuple[bytes, str, str, str]:
    secret_key = draw(_secret_key_st).lower()
    secret_val = draw(_secret_value_st)
    safe_key = draw(_safe_key_st).lower()
    safe_val = draw(_safe_value_st)
    # Values must not contain colon or # to work with the simple YAML sanitizer
    # All sampled values are safe for this already.
    content = f"{secret_key}: {secret_val}\n{safe_key}: {safe_val}\n"
    return content.encode(), secret_key, safe_key, safe_val


@st.composite
def _toml_content_st(draw) -> tuple[bytes, str, str, str]:
    secret_key = draw(_secret_key_st).lower()
    secret_val = draw(_secret_value_st)
    safe_key = draw(_safe_key_st).lower()
    safe_val = draw(_safe_value_st)
    content = f'{secret_key} = "{secret_val}"\n{safe_key} = "{safe_val}"\n'
    return content.encode(), secret_key, safe_key, safe_val

