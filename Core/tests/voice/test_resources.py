"""Availability checks for pre-provisioned local voice assets.

Requirements: 3.1, 6.2, 7.1–7.2, 9.6
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from pathlib import Path

import pytest

from core.voice import resources


def _provisioned_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    """Create small local artifacts and an explicitly generated manifest."""
    model_directory = tmp_path / "models"
    asr_artifact = model_directory / "asr" / "Qwen3ASR.mlmodelc"
    llm_artifact = model_directory / "llm" / "Qwen3-4B-Instruct-4bit"
    voice_asset = model_directory / "my_voice.wav"
    asr_artifact.parent.mkdir(parents=True)
    llm_artifact.parent.mkdir(parents=True)
    asr_artifact.write_bytes(b"verified-coreml-qwen3-asr")
    llm_artifact.write_bytes(b"verified-qwen3-4b-instruct-4bit")
    voice_asset.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

    manifest = resources.build_voice_model_manifest(
        model_directory=model_directory,
        coreml_asr_path=asr_artifact,
        qwen_llm_path=llm_artifact,
        coreml_asr_version="2025.1",
        vocabulary_version="2025.1",
        qwen_llm_version="2025.1",
    )
    manifest_path = model_directory / resources.MODEL_MANIFEST_FILENAME
    resources.write_voice_model_manifest(manifest, manifest_path)
    return model_directory, manifest_path, voice_asset, asr_artifact, llm_artifact


def _check(tmp_path: Path) -> tuple[resources.VoiceStartupHealth, tuple[Path, Path, Path, Path, Path]]:
    fixture = _provisioned_fixture(tmp_path)
    model_directory, manifest_path, voice_asset, _, _ = fixture
    return (
        resources.check_voice_startup_availability(
            model_directory=model_directory,
            manifest_path=manifest_path,
            voice_asset_path=voice_asset,
        ),
        fixture,
    )


def test_present_verified_local_assets_are_ready(tmp_path: Path) -> None:
    """Pre-provisioned local Qwen ASR/LLM and readable XTTS asset pass startup."""
    report, _ = _check(tmp_path)

    assert report.is_ready
    assert report.issues == ()
    assert "available" in report.actionable_summary.lower()


def test_missing_model_or_voice_asset_has_an_actionable_failure(tmp_path: Path) -> None:
    """Missing prerequisites stay unavailable and tell the owner what to provision."""
    _, fixture = _check(tmp_path)
    model_directory, manifest_path, voice_asset, asr_artifact, _ = fixture
    asr_artifact.unlink()
    voice_asset.unlink()

    report = resources.check_voice_startup_availability(
        model_directory=model_directory,
        manifest_path=manifest_path,
        voice_asset_path=voice_asset,
    )

    assert not report.is_ready
    assert {issue.code for issue in report.issues} == {"artifact_missing", "voice_asset_missing"}
    assert all(issue.action for issue in report.issues)
    assert "provision" in report.actionable_summary.lower()


def test_hash_mismatched_model_is_rejected_without_rewriting_manifest(tmp_path: Path) -> None:
    """Changed local model bytes cannot be accepted by regenerating a startup hash."""
    _, fixture = _check(tmp_path)
    model_directory, manifest_path, voice_asset, _, llm_artifact = fixture
    original_manifest = manifest_path.read_text(encoding="utf-8")
    llm_artifact.write_bytes(b"tampered-local-qwen-artifact")

    report = resources.check_voice_startup_availability(
        model_directory=model_directory,
        manifest_path=manifest_path,
        voice_asset_path=voice_asset,
    )

    assert not report.is_ready
    issue = next(issue for issue in report.issues if issue.asset_id == resources.QWEN3_4B_INSTRUCT_ARTIFACT_ID)
    assert issue.code == "artifact_hash_mismatch"
    assert "do not regenerate" in issue.action.lower()
    assert manifest_path.read_text(encoding="utf-8") == original_manifest


def test_unreadable_user_voice_asset_is_reported_before_xtts_initialization(tmp_path: Path) -> None:
    """A user-supplied my_voice.wav needs readable permission bits, not a fallback voice."""
    _, fixture = _check(tmp_path)
    model_directory, manifest_path, voice_asset, _, _ = fixture
    voice_asset.chmod(0)
    try:
        report = resources.check_voice_startup_availability(
            model_directory=model_directory,
            manifest_path=manifest_path,
            voice_asset_path=voice_asset,
        )
    finally:
        voice_asset.chmod(0o600)

    assert not report.is_ready
    issue = next(issue for issue in report.issues if issue.asset_id == "my_voice.wav")
    assert issue.code == "voice_asset_unreadable"
    assert "read access" in issue.action.lower()


@pytest.mark.asyncio
async def test_startup_probe_reports_unavailable_status_off_event_loop(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The asynchronous startup wrapper reports local availability but does not block startup."""
    model_directory = tmp_path / "missing-models"
    logger = logging.getLogger("test.voice.resources")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        task = asyncio.create_task(
            resources.run_startup_voice_health_check(
                logger=logger,
                model_directory=model_directory,
                manifest_path=model_directory / resources.MODEL_MANIFEST_FILENAME,
                voice_asset_path=model_directory / "my_voice.wav",
            )
        )
        await asyncio.sleep(0)
        report = await task

    assert report is not None
    assert not report.is_ready
    assert "Local voice is unavailable" in caplog.text
    assert not model_directory.exists(), "health checks must not provision or create model directories"


def test_resources_module_has_no_legacy_or_cloud_route_selection() -> None:
    """The availability probe cannot import or select a retired/cloud voice provider."""
    source = inspect.getsource(resources).lower()
    for forbidden in ("deepgram", "groq", "cartesia", "edge tts", "legacy_pipeline", "subprocess"):
        assert forbidden not in source
    assert resources.LOCAL_VOICE_COMPONENT_IDS == (
        resources.COREML_QWEN3_ASR_ARTIFACT_ID,
        resources.QWEN3_4B_INSTRUCT_ARTIFACT_ID,
        "xtts_v2",
    )
