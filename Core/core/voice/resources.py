"""Local voice model manifests and non-blocking startup availability checks.

The normal voice path is provisioned entirely on the user's machine.  This
module only inspects files that are already present: it never imports a model
runtime, downloads an artifact, converts a model, or selects a cloud/legacy
provider.  Call :func:`run_startup_voice_health_check` after the IPC server is
listening so hashing large local artifacts cannot delay service readiness.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import stat
import sys
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence, runtime_checkable


MODEL_DIRECTORY_ENV = "HAKI_VOICE_MODEL_DIR"
MODEL_MANIFEST_ENV = "HAKI_VOICE_MODEL_MANIFEST"
VOICE_ASSET_ENV = "HAKI_VOICE_ASSET_PATH"
MODEL_MANIFEST_FILENAME = "voice-model-manifest.json"
MODEL_MANIFEST_SCHEMA_VERSION = 1
HASH_CHUNK_BYTES = 1024 * 1024

COREML_QWEN3_ASR_ARTIFACT_ID = "qwen3_asr_coreml"
QWEN3_4B_INSTRUCT_ARTIFACT_ID = "qwen3_4b_instruct_4bit"
COREML_QWEN3_ASR_MODEL_ID = "Qwen/Qwen3-ASR-CoreML"
QWEN3_4B_INSTRUCT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-4bit"
LOCAL_VOICE_COMPONENT_IDS = (
    COREML_QWEN3_ASR_ARTIFACT_ID,
    QWEN3_4B_INSTRUCT_ARTIFACT_ID,
    "xtts_v2",
)


class ModelManifestError(ValueError):
    """A user-local voice model manifest is malformed or unsafe."""


@dataclass(frozen=True)
class ModelArtifactManifest:
    """A non-secret, locally provisioned model artifact declaration."""

    artifact_id: str
    model_id: str
    artifact_path: str
    sha256: str
    version: str
    sample_rate_hz: int | None = None
    vocabulary_version: str | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ModelManifestError("artifact_id must not be empty")
        if not self.model_id:
            raise ModelManifestError("model_id must not be empty")
        _validate_relative_artifact_path(self.artifact_path)
        if not _is_sha256(self.sha256):
            raise ModelManifestError(
                f"{self.artifact_id}: sha256 must be 64 lowercase hexadecimal characters"
            )
        if not self.version:
            raise ModelManifestError(f"{self.artifact_id}: version must not be empty")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelArtifactManifest":
        allowed = {
            "artifact_id",
            "model_id",
            "artifact_path",
            "sha256",
            "version",
            "sample_rate_hz",
            "vocabulary_version",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ModelManifestError(f"unknown manifest fields: {sorted(unknown)!r}")
        try:
            return cls(
                artifact_id=_required_string(value, "artifact_id"),
                model_id=_required_string(value, "model_id"),
                artifact_path=_required_string(value, "artifact_path"),
                sha256=_required_string(value, "sha256"),
                version=_required_string(value, "version"),
                sample_rate_hz=_optional_int(value, "sample_rate_hz"),
                vocabulary_version=_optional_string(value, "vocabulary_version"),
            )
        except KeyError as exc:
            raise ModelManifestError(f"missing manifest field: {exc.args[0]}") from exc


@dataclass(frozen=True)
class VoiceModelManifest:
    """Versioned local manifest for the required CoreML ASR and MLX LLM."""

    schema_version: int
    artifacts: tuple[ModelArtifactManifest, ...]

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_MANIFEST_SCHEMA_VERSION:
            raise ModelManifestError(
                f"unsupported voice model manifest schema {self.schema_version}; "
                f"expected {MODEL_MANIFEST_SCHEMA_VERSION}"
            )
        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ModelManifestError("voice model manifest contains duplicate artifact_id values")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VoiceModelManifest":
        allowed = {"schema_version", "artifacts"}
        unknown = set(value) - allowed
        if unknown:
            raise ModelManifestError(f"unknown manifest fields: {sorted(unknown)!r}")
        schema_version = value.get("schema_version")
        artifacts = value.get("artifacts")
        if not isinstance(schema_version, int):
            raise ModelManifestError("schema_version must be an integer")
        if not isinstance(artifacts, list):
            raise ModelManifestError("artifacts must be an array")
        return cls(
            schema_version=schema_version,
            artifacts=tuple(ModelArtifactManifest.from_mapping(item) for item in artifacts if isinstance(item, dict)),
        )

    @classmethod
    def load(cls, path: Path) -> "VoiceModelManifest":
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            raise
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelManifestError(f"cannot read model manifest: {exc}") from exc
        if not isinstance(payload, dict):
            raise ModelManifestError("voice model manifest root must be an object")
        return cls.from_mapping(payload)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifacts": [asdict(artifact) for artifact in self.artifacts],
        }

    def artifact(self, artifact_id: str) -> ModelArtifactManifest | None:
        return next((item for item in self.artifacts if item.artifact_id == artifact_id), None)


@dataclass(frozen=True)
class VoiceAvailabilityIssue:
    """One safe, actionable local availability failure."""

    asset_id: str
    code: str
    path: Path
    message: str
    action: str


@dataclass(frozen=True)
class VoiceStartupHealth:
    """Read-only result of inspecting pre-provisioned local voice assets."""

    model_directory: Path
    manifest_path: Path
    voice_asset_path: Path
    issues: tuple[VoiceAvailabilityIssue, ...]

    @property
    def is_ready(self) -> bool:
        return not self.issues

    @property
    def actionable_summary(self) -> str:
        if self.is_ready:
            return "Local Qwen3 ASR, Qwen3-4B-Instruct-4bit, and XTTS voice assets are available."
        return " ".join(issue.action for issue in self.issues)


def default_voice_model_directory(environ: Mapping[str, str] | None = None) -> Path:
    """Return the non-secret, user-local model directory without creating it."""
    values = os.environ if environ is None else environ
    configured = values.get(MODEL_DIRECTORY_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "Library" / "Application Support" / "HAKI" / "models"


def default_voice_manifest_path(
    model_directory: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the model-manifest location without writing or provisioning it."""
    values = os.environ if environ is None else environ
    configured = values.get(MODEL_MANIFEST_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return (model_directory or default_voice_model_directory(values)) / MODEL_MANIFEST_FILENAME


def default_voice_asset_path(
    model_directory: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the user-provided XTTS conditioning asset path without creating it."""
    values = os.environ if environ is None else environ
    configured = values.get(VOICE_ASSET_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return (model_directory or default_voice_model_directory(values)) / "my_voice.wav"


def build_voice_model_manifest(
    *,
    model_directory: Path,
    coreml_asr_path: Path,
    qwen_llm_path: Path,
    coreml_asr_version: str,
    vocabulary_version: str,
    qwen_llm_version: str,
) -> VoiceModelManifest:
    """Build a manifest from already-provisioned artifacts.

    This explicit deployment helper computes hashes once during provisioning.
    Startup validation only reads the resulting manifest; it never regenerates
    hashes in response to a mismatch.
    """
    root = model_directory.resolve()
    asr_relative = _relative_to_root(coreml_asr_path, root)
    llm_relative = _relative_to_root(qwen_llm_path, root)
    return VoiceModelManifest(
        schema_version=MODEL_MANIFEST_SCHEMA_VERSION,
        artifacts=(
            ModelArtifactManifest(
                artifact_id=COREML_QWEN3_ASR_ARTIFACT_ID,
                model_id=COREML_QWEN3_ASR_MODEL_ID,
                artifact_path=asr_relative,
                sha256=compute_artifact_sha256(coreml_asr_path),
                version=coreml_asr_version,
                sample_rate_hz=16_000,
                vocabulary_version=vocabulary_version,
            ),
            ModelArtifactManifest(
                artifact_id=QWEN3_4B_INSTRUCT_ARTIFACT_ID,
                model_id=QWEN3_4B_INSTRUCT_MODEL_ID,
                artifact_path=llm_relative,
                sha256=compute_artifact_sha256(qwen_llm_path),
                version=qwen_llm_version,
            ),
        ),
    )


def write_voice_model_manifest(manifest: VoiceModelManifest, path: Path) -> None:
    """Atomically write an explicitly generated, non-secret manifest."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_mapping(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(temporary_path, 0o600)
    temporary_path.replace(path)


def compute_artifact_sha256(path: Path) -> str:
    """Return a deterministic SHA-256 for a readable file or directory tree."""
    resolved = path.resolve(strict=True)
    if resolved.is_symlink():
        raise ModelManifestError(f"refusing symbolic-link artifact: {path}")

    digest = hashlib.sha256()
    if resolved.is_file():
        _require_readable_file(resolved)
        _update_digest_from_file(digest, resolved)
    elif resolved.is_dir():
        _require_readable_directory(resolved)
        entries = sorted(resolved.rglob("*"), key=lambda item: item.relative_to(resolved).as_posix())
        for entry in entries:
            relative = entry.relative_to(resolved).as_posix().encode("utf-8")
            if entry.is_symlink():
                raise ModelManifestError(f"refusing symbolic link within artifact: {entry}")
            if entry.is_dir():
                _require_readable_directory(entry)
                digest.update(b"D\0" + relative + b"\0")
            elif entry.is_file():
                _require_readable_file(entry)
                digest.update(b"F\0" + relative + b"\0")
                _update_digest_from_file(digest, entry)
            else:
                raise ModelManifestError(f"artifact contains unsupported file type: {entry}")
    else:
        raise ModelManifestError(f"artifact must be a regular file or directory: {path}")
    return digest.hexdigest()


def check_voice_startup_availability(
    *,
    model_directory: Path | None = None,
    manifest_path: Path | None = None,
    voice_asset_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> VoiceStartupHealth:
    """Inspect local voice prerequisites and return actionable failures.

    The check is intentionally side-effect free: it does not create
    directories, initialize MLX/CoreML/XTTS, access the network, download, or
    convert artifacts.  It never makes a fallback route decision.
    """
    root = (model_directory or default_voice_model_directory(environ)).expanduser()
    manifest_file = (manifest_path or default_voice_manifest_path(root, environ)).expanduser()
    voice_file = (voice_asset_path or default_voice_asset_path(root, environ)).expanduser()
    issues: list[VoiceAvailabilityIssue] = []

    _validate_readable_voice_asset(voice_file, issues)
    manifest = _load_manifest_for_health(manifest_file, issues)
    if manifest is not None:
        _validate_required_model_artifacts(manifest, root, issues)

    return VoiceStartupHealth(
        model_directory=root,
        manifest_path=manifest_file,
        voice_asset_path=voice_file,
        issues=tuple(issues),
    )


async def run_startup_voice_health_check(
    *,
    logger: logging.Logger | None = None,
    model_directory: Path | None = None,
    manifest_path: Path | None = None,
    voice_asset_path: Path | None = None,
) -> VoiceStartupHealth | None:
    """Run the local-only check away from the event loop and report its result."""
    health_logger = logger or logging.getLogger(__name__)
    try:
        report = await asyncio.to_thread(
            check_voice_startup_availability,
            model_directory=model_directory,
            manifest_path=manifest_path,
            voice_asset_path=voice_asset_path,
        )
    except Exception as exc:  # noqa: BLE001 - startup status must not stop Core
        health_logger.warning("Local voice availability check failed: %s. Verify local voice provisioning.", exc)
        return None

    if report.is_ready:
        health_logger.info("Local voice availability check passed for %s", report.model_directory)
    else:
        health_logger.warning("Local voice is unavailable: %s", report.actionable_summary)
    return report


def _load_manifest_for_health(
    manifest_path: Path,
    issues: list[VoiceAvailabilityIssue],
) -> VoiceModelManifest | None:
    if not manifest_path.exists():
        issues.append(
            VoiceAvailabilityIssue(
                asset_id="voice_model_manifest",
                code="manifest_missing",
                path=manifest_path,
                message="The local voice model manifest is missing.",
                action=(
                    f"Provision the local Qwen3 ASR and Qwen3-4B-Instruct-4bit artifacts, "
                    f"then create {manifest_path.name}."
                ),
            )
        )
        return None
    try:
        _require_readable_file(manifest_path)
        return VoiceModelManifest.load(manifest_path)
    except (OSError, ModelManifestError) as exc:
        issues.append(
            VoiceAvailabilityIssue(
                asset_id="voice_model_manifest",
                code="manifest_invalid",
                path=manifest_path,
                message=str(exc),
                action=f"Repair or reprovision the non-secret local manifest at {manifest_path}.",
            )
        )
        return None


def _validate_required_model_artifacts(
    manifest: VoiceModelManifest,
    root: Path,
    issues: list[VoiceAvailabilityIssue],
) -> None:
    requirements = (
        (COREML_QWEN3_ASR_ARTIFACT_ID, COREML_QWEN3_ASR_MODEL_ID),
        (QWEN3_4B_INSTRUCT_ARTIFACT_ID, QWEN3_4B_INSTRUCT_MODEL_ID),
    )
    for artifact_id, expected_model_id in requirements:
        artifact = manifest.artifact(artifact_id)
        if artifact is None:
            issues.append(
                VoiceAvailabilityIssue(
                    asset_id=artifact_id,
                    code="artifact_manifest_missing",
                    path=root,
                    message=f"The manifest does not declare {artifact_id}.",
                    action=f"Add a verified {artifact_id} entry to {MODEL_MANIFEST_FILENAME}.",
                )
            )
            continue
        if artifact.model_id != expected_model_id:
            issues.append(
                VoiceAvailabilityIssue(
                    asset_id=artifact_id,
                    code="artifact_model_id_invalid",
                    path=root / artifact.artifact_path,
                    message=(f"Expected model ID {expected_model_id!r}, found {artifact.model_id!r}."),
                    action=f"Reprovision the required local {artifact_id} artifact and manifest entry.",
                )
            )
            continue
        if artifact_id == COREML_QWEN3_ASR_ARTIFACT_ID and (
            artifact.sample_rate_hz != 16_000 or not artifact.vocabulary_version
        ):
            issues.append(
                VoiceAvailabilityIssue(
                    asset_id=artifact_id,
                    code="asr_manifest_invalid",
                    path=root / artifact.artifact_path,
                    message="CoreML Qwen3 ASR must declare 16000 Hz and a vocabulary version.",
                    action="Recreate the CoreML Qwen3 ASR manifest entry with its sample-rate and vocabulary metadata.",
                )
            )
            continue

        try:
            artifact_path = _resolve_manifest_artifact_path(root, artifact.artifact_path)
            if not artifact_path.exists():
                raise FileNotFoundError(artifact_path)
            actual_hash = compute_artifact_sha256(artifact_path)
        except FileNotFoundError:
            issues.append(
                VoiceAvailabilityIssue(
                    asset_id=artifact_id,
                    code="artifact_missing",
                    path=root / artifact.artifact_path,
                    message=f"Required local artifact {artifact_id} is missing.",
                    action=f"Provision {artifact_id} at {root / artifact.artifact_path} without starting a voice turn.",
                )
            )
            continue
        except (OSError, ModelManifestError) as exc:
            issues.append(
                VoiceAvailabilityIssue(
                    asset_id=artifact_id,
                    code="artifact_unreadable",
                    path=root / artifact.artifact_path,
                    message=str(exc),
                    action=f"Grant the current user read access to the local {artifact_id} artifact and retry startup.",
                )
            )
            continue

        if actual_hash != artifact.sha256:
            issues.append(
                VoiceAvailabilityIssue(
                    asset_id=artifact_id,
                    code="artifact_hash_mismatch",
                    path=artifact_path,
                    message="The local artifact hash does not match the provisioned manifest.",
                    action=f"Reprovision {artifact_id}; do not regenerate its manifest hash during a voice turn.",
                )
            )


def _validate_readable_voice_asset(path: Path, issues: list[VoiceAvailabilityIssue]) -> None:
    if not path.exists():
        issues.append(
            VoiceAvailabilityIssue(
                asset_id="my_voice.wav",
                code="voice_asset_missing",
                path=path,
                message="The XTTS conditioning file my_voice.wav is missing.",
                action=f"Place a readable user-supplied my_voice.wav at {path} or set {VOICE_ASSET_ENV}.",
            )
        )
        return
    try:
        _require_readable_file(path)
    except (OSError, ModelManifestError) as exc:
        issues.append(
            VoiceAvailabilityIssue(
                asset_id="my_voice.wav",
                code="voice_asset_unreadable",
                path=path,
                message=str(exc),
                action=f"Grant the current user read access to my_voice.wav at {path} and retry startup.",
            )
        )


def _require_readable_file(path: Path) -> None:
    file_stat = path.stat()
    if not stat.S_ISREG(file_stat.st_mode):
        raise ModelManifestError(f"expected a regular readable file: {path}")
    if file_stat.st_mode & (stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH) == 0:
        raise PermissionError(f"file has no read permission bits: {path}")
    with path.open("rb") as handle:
        handle.read(1)


def _require_readable_directory(path: Path) -> None:
    directory_stat = path.stat()
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise ModelManifestError(f"expected a readable directory: {path}")
    if directory_stat.st_mode & (stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH) == 0:
        raise PermissionError(f"directory has no read permission bits: {path}")
    with os.scandir(path):
        pass


def _update_digest_from_file(digest: Any, path: Path) -> None:
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)


def _resolve_manifest_artifact_path(root: Path, relative_path: str) -> Path:
    _validate_relative_artifact_path(relative_path)
    root_resolved = root.resolve()
    candidate = (root_resolved / relative_path).resolve(strict=False)
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ModelManifestError(f"artifact path escapes local model directory: {relative_path}") from exc
    return candidate


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ModelManifestError(f"artifact path must be inside model directory: {path}") from exc


def _validate_relative_artifact_path(value: str) -> None:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ModelManifestError("artifact_path must be a non-empty relative path inside the model directory")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _required_string(value: Mapping[str, Any], name: str) -> str:
    item = value[name]
    if not isinstance(item, str) or not item:
        raise ModelManifestError(f"{name} must be a non-empty string")
    return item


def _optional_string(value: Mapping[str, Any], name: str) -> str | None:
    item = value.get(name)
    if item is not None and not isinstance(item, str):
        raise ModelManifestError(f"{name} must be a string when provided")
    return item


def _optional_int(value: Mapping[str, Any], name: str) -> int | None:
    item = value.get(name)
    if item is not None and (not isinstance(item, int) or isinstance(item, bool)):
        raise ModelManifestError(f"{name} must be an integer when provided")
    return item


# ---------------------------------------------------------------------------
# Warmed-runtime resource admission
# ---------------------------------------------------------------------------

GIBIBYTE_BYTES = 1024**3
MODEL_RESIDENT_LIMIT_BYTES = int(2.5 * GIBIBYTE_BYTES)
PIPELINE_MEMORY_LIMIT_BYTES = 5 * GIBIBYTE_BYTES


class ResourceMeasurementError(RuntimeError):
    """The local process resident-memory measurement cannot be trusted."""


class ResourceLifecycleError(RuntimeError):
    """A required local voice component could not be warmed or released."""


class ResourceAdmissionState(str, Enum):
    """Whether a new voice turn may enter the warmed local pipeline."""

    ADMITTING = "admitting"
    DRAINING = "draining"


class ProcessResidentMemorySampler(Protocol):
    """Reads one process's resident bytes from a process-specific source."""

    @property
    def method(self) -> str:
        """Return a stable description of the source used for diagnostics."""

    def resident_bytes(self, process_id: int) -> int:
        """Return the current resident-byte count for ``process_id``."""


@dataclass(frozen=True, slots=True)
class MacOSProcessResidentMemorySampler:
    """macOS RSS sampler backed by ``psutil.Process(pid).memory_info().rss``.

    RSS is deliberately sampled once for each unique process ID by
    :class:`VoiceResourceManager`.  This avoids counting a Core Python process
    twice when it also hosts the MLX model, or a process registered under more
    than one pipeline role.
    """

    @property
    def method(self) -> str:
        return "macos_process_rss_psutil"

    def resident_bytes(self, process_id: int) -> int:
        if sys.platform != "darwin":
            raise ResourceMeasurementError("macOS process RSS metrics are unavailable on this platform")
        try:
            import psutil

            resident = psutil.Process(process_id).memory_info().rss
        except Exception as exc:  # pragma: no cover - depends on host process lifecycle
            raise ResourceMeasurementError(
                f"could not read macOS resident memory for process {process_id}: {exc}"
            ) from exc
        if not isinstance(resident, int) or resident < 0:
            raise ResourceMeasurementError(
                f"macOS resident memory for process {process_id} was invalid"
            )
        return resident


@dataclass(frozen=True, slots=True)
class VoiceProcessTopology:
    """Process IDs that constitute the warmed local voice runtime.

    The MLX model can share the Core Python process.  The pipeline calculation
    therefore uses the union of all role lists, including MLX, while the model
    calculation uses only the MLX list.  The two reported metrics are never
    added together, so shared model RSS is not double counted.
    """

    mlx_process_ids: tuple[int, ...]
    core_python_process_ids: tuple[int, ...] = ()
    swift_audio_process_ids: tuple[int, ...] = ()
    asr_worker_process_ids: tuple[int, ...] = ()
    supporting_process_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name, process_ids in (
            ("mlx_process_ids", self.mlx_process_ids),
            ("core_python_process_ids", self.core_python_process_ids),
            ("swift_audio_process_ids", self.swift_audio_process_ids),
            ("asr_worker_process_ids", self.asr_worker_process_ids),
            ("supporting_process_ids", self.supporting_process_ids),
        ):
            if not isinstance(process_ids, tuple):
                raise ValueError(f"{name} must be a tuple of process IDs")
            for process_id in process_ids:
                if not isinstance(process_id, int) or isinstance(process_id, bool) or process_id <= 0:
                    raise ValueError(f"{name} must contain positive integer process IDs")
        if not self.mlx_process_ids:
            raise ValueError("mlx_process_ids must identify the warmed MLX model process")

    @property
    def model_process_ids(self) -> tuple[int, ...]:
        """Unique MLX process IDs in deterministic registration order."""
        return _unique_process_ids(self.mlx_process_ids)

    @property
    def pipeline_process_ids(self) -> tuple[int, ...]:
        """Unique resident-memory union for the local voice pipeline."""
        return _unique_process_ids(
            self.mlx_process_ids
            + self.core_python_process_ids
            + self.swift_audio_process_ids
            + self.asr_worker_process_ids
            + self.supporting_process_ids
        )


@dataclass(frozen=True, slots=True)
class ProcessResidentFootprint:
    """One unique process RSS sample used by a memory measurement."""

    process_id: int
    resident_bytes: int


@dataclass(frozen=True, slots=True)
class VoiceMemoryMeasurement:
    """A single, content-free resource sample for voice admission decisions."""

    model_resident_bytes: int
    pipeline_memory_bytes: int
    process_residents: tuple[ProcessResidentFootprint, ...]
    model_process_ids: tuple[int, ...]
    pipeline_process_ids: tuple[int, ...]
    sampled_monotonic_ns: int
    sampling_method: str

    @property
    def model_limit_reached(self) -> bool:
        return self.model_resident_bytes >= MODEL_RESIDENT_LIMIT_BYTES

    @property
    def pipeline_limit_exceeded(self) -> bool:
        return self.pipeline_memory_bytes > PIPELINE_MEMORY_LIMIT_BYTES

    @property
    def exceeds_budget(self) -> bool:
        return self.model_limit_reached or self.pipeline_limit_exceeded

    @property
    def satisfies_recovery_budget(self) -> bool:
        return (
            self.model_resident_bytes < MODEL_RESIDENT_LIMIT_BYTES
            and self.pipeline_memory_bytes <= PIPELINE_MEMORY_LIMIT_BYTES
        )


@dataclass(frozen=True, slots=True)
class ResourceBudgetDiagnostic:
    """Safe memory-budget diagnostic emitted during rejection or recovery."""

    stage: str
    outcome: str
    state: ResourceAdmissionState
    reason: str
    measurement: VoiceMemoryMeasurement | None
    released_components: tuple[str, ...]
    capture_active: bool
    recovery_outcome: str
    error_class: str | None = None


@dataclass(frozen=True, slots=True)
class ResourceAdmissionDecision:
    """The admission result and safe explanation for one requested voice turn."""

    admitted: bool
    state: ResourceAdmissionState
    measurement: VoiceMemoryMeasurement | None
    reason: str | None = None


@runtime_checkable
class WarmableVoiceComponent(Protocol):
    """A local component that must be prepared before timing starts."""

    async def warm_up(self) -> None:
        """Load or initialize only pre-provisioned local runtime state."""


@runtime_checkable
class IdleReleasableVoiceComponent(WarmableVoiceComponent, Protocol):
    """A component whose unused runtime/cache can be released safely."""

    def is_idle(self) -> bool:
        """Return true only when releasing this component cannot interrupt work."""

    async def release_idle(self) -> None:
        """Release the idle component and any associated idle cache."""


ResourceDiagnosticSink = Callable[[ResourceBudgetDiagnostic], Awaitable[None] | None]
CaptureActivityProvider = Callable[[], bool]


class VoiceResourceManager:
    """Warm local voice components and guard turn admission by resident memory.

    Only idle XTTS and then idle Qwen/cache are eligible for release.  ASR,
    Pipecat, and capture are warmed but never released by this manager, which
    keeps native microphone capture safe even while resource draining occurs.
    """

    def __init__(
        self,
        *,
        process_topology: VoiceProcessTopology,
        memory_sampler: ProcessResidentMemorySampler | None = None,
        asr: WarmableVoiceComponent | None = None,
        qwen: IdleReleasableVoiceComponent | None = None,
        xtts: IdleReleasableVoiceComponent | None = None,
        pipecat: WarmableVoiceComponent | None = None,
        capture_is_active: CaptureActivityProvider | None = None,
        diagnostic_sink: ResourceDiagnosticSink | None = None,
    ) -> None:
        self._process_topology = process_topology
        self._memory_sampler = memory_sampler or MacOSProcessResidentMemorySampler()
        self._asr = asr
        self._qwen = qwen
        self._xtts = xtts
        self._pipecat = pipecat
        self._capture_is_active = capture_is_active or (lambda: False)
        self._diagnostic_sink = diagnostic_sink
        self._state = ResourceAdmissionState.ADMITTING
        self._lock = asyncio.Lock()
        self._diagnostics: list[ResourceBudgetDiagnostic] = []
        self._last_measurement: VoiceMemoryMeasurement | None = None

    @property
    def state(self) -> ResourceAdmissionState:
        return self._state

    @property
    def last_measurement(self) -> VoiceMemoryMeasurement | None:
        return self._last_measurement

    @property
    def diagnostics(self) -> tuple[ResourceBudgetDiagnostic, ...]:
        """Return content-free local diagnostics accumulated by this manager."""
        return tuple(self._diagnostics)

    async def warm_up(self) -> VoiceMemoryMeasurement:
        """Warm ASR, Qwen, XTTS, and Pipecat in required local-stack order.

        The post-warm sample is evaluated before any caller can admit a turn;
        a budget breach enters draining and triggers only safe idle releases.
        """
        for name, component in (
            ("asr", self._asr),
            ("qwen", self._qwen),
            ("xtts", self._xtts),
            ("pipecat", self._pipecat),
        ):
            if component is None:
                continue
            try:
                await component.warm_up()
            except Exception as exc:  # noqa: BLE001 - normalized for voice startup diagnostics
                raise ResourceLifecycleError(f"could not warm local voice component {name}: {exc}") from exc

        async with self._lock:
            measurement, _, _ = await self._refresh_locked()
            return measurement

    def measure_memory(self) -> VoiceMemoryMeasurement:
        """Measure unique process RSS for MLX and the complete local pipeline."""
        process_residents: dict[int, int] = {}
        for process_id in self._process_topology.pipeline_process_ids:
            resident_bytes = self._memory_sampler.resident_bytes(process_id)
            if not isinstance(resident_bytes, int) or isinstance(resident_bytes, bool) or resident_bytes < 0:
                raise ResourceMeasurementError(
                    f"resident memory for process {process_id} must be a non-negative integer"
                )
            process_residents[process_id] = resident_bytes

        model_process_ids = self._process_topology.model_process_ids
        pipeline_process_ids = self._process_topology.pipeline_process_ids
        measurement = VoiceMemoryMeasurement(
            model_resident_bytes=sum(process_residents[process_id] for process_id in model_process_ids),
            pipeline_memory_bytes=sum(process_residents[process_id] for process_id in pipeline_process_ids),
            process_residents=tuple(
                ProcessResidentFootprint(process_id, process_residents[process_id])
                for process_id in pipeline_process_ids
            ),
            model_process_ids=model_process_ids,
            pipeline_process_ids=pipeline_process_ids,
            sampled_monotonic_ns=time.monotonic_ns(),
            sampling_method=getattr(self._memory_sampler, "method", type(self._memory_sampler).__name__),
        )
        self._last_measurement = measurement
        return measurement

    async def refresh(self) -> VoiceMemoryMeasurement | None:
        """Record a background sample and transition draining/recovery state."""
        async with self._lock:
            try:
                measurement, _, _ = await self._refresh_locked()
                return measurement
            except ResourceMeasurementError as exc:
                await self._record_measurement_failure_locked(exc)
                return None

    async def admit_new_turn(self) -> ResourceAdmissionDecision:
        """Return a fresh, diagnostics-backed decision for one new voice turn.

        The request that first observes a threshold breach is rejected even if
        releasing idle resources immediately restores the budget.  This makes
        draining observable and ensures no turn slips through a breach.
        """
        async with self._lock:
            state_before_sample = self._state
            try:
                measurement, breached_before_release, _ = await self._refresh_locked()
            except ResourceMeasurementError as exc:
                await self._record_measurement_failure_locked(exc)
                return ResourceAdmissionDecision(
                    admitted=False,
                    state=self._state,
                    measurement=None,
                    reason="Memory measurement is unavailable; voice turns remain paused until a reliable local sample succeeds.",
                )

            if breached_before_release:
                return ResourceAdmissionDecision(
                    admitted=False,
                    state=self._state,
                    measurement=measurement,
                    reason=_budget_reason(measurement),
                )
            if state_before_sample is ResourceAdmissionState.DRAINING and self._state is ResourceAdmissionState.DRAINING:
                return ResourceAdmissionDecision(
                    admitted=False,
                    state=self._state,
                    measurement=measurement,
                    reason=_recovery_reason(measurement),
                )
            return ResourceAdmissionDecision(
                admitted=self._state is ResourceAdmissionState.ADMITTING,
                state=self._state,
                measurement=measurement,
                reason=None if self._state is ResourceAdmissionState.ADMITTING else _recovery_reason(measurement),
            )

    async def _refresh_locked(self) -> tuple[VoiceMemoryMeasurement, bool, tuple[str, ...]]:
        measurement = self.measure_memory()
        if not measurement.exceeds_budget:
            if self._state is ResourceAdmissionState.DRAINING:
                self._state = ResourceAdmissionState.ADMITTING
                await self._emit_locked(
                    ResourceBudgetDiagnostic(
                        stage="memory_budget",
                        outcome="recovered",
                        state=self._state,
                        reason=_recovery_reason(measurement),
                        measurement=measurement,
                        released_components=(),
                        capture_active=self._capture_is_active(),
                        recovery_outcome="admission_resumed",
                    )
                )
            return measurement, False, ()

        was_draining = self._state is ResourceAdmissionState.DRAINING
        self._state = ResourceAdmissionState.DRAINING
        released_components = await self._release_idle_resources_locked()
        await self._emit_locked(
            ResourceBudgetDiagnostic(
                stage="memory_budget",
                outcome="rejected",
                state=self._state,
                reason=_budget_reason(measurement),
                measurement=measurement,
                released_components=released_components,
                capture_active=self._capture_is_active(),
                recovery_outcome="idle_resources_released" if released_components else "awaiting_idle_resources_or_lower_memory",
            )
        )

        # Re-sample after release so a monitoring call can reopen admission only
        # after both documented recovery thresholds are satisfied.
        if released_components:
            recovered_measurement = self.measure_memory()
            if recovered_measurement.satisfies_recovery_budget:
                self._state = ResourceAdmissionState.ADMITTING
                await self._emit_locked(
                    ResourceBudgetDiagnostic(
                        stage="memory_budget",
                        outcome="recovered",
                        state=self._state,
                        reason=_recovery_reason(recovered_measurement),
                        measurement=recovered_measurement,
                        released_components=released_components,
                        capture_active=self._capture_is_active(),
                        recovery_outcome="admission_resumed",
                    )
                )
            return recovered_measurement, True, released_components
        return measurement, not was_draining or measurement.exceeds_budget, released_components

    async def _release_idle_resources_locked(self) -> tuple[str, ...]:
        """Release only idle XTTS followed by idle Qwen/cache; never capture."""
        released: list[str] = []
        for name, component in (("xtts", self._xtts), ("qwen", self._qwen)):
            if component is None or not component.is_idle():
                continue
            try:
                await component.release_idle()
            except Exception as exc:  # noqa: BLE001 - reject safely with diagnostics
                await self._emit_locked(
                    ResourceBudgetDiagnostic(
                        stage="memory_budget",
                        outcome="rejected",
                        state=self._state,
                        reason=f"Idle {name} release failed while memory draining: {exc}",
                        measurement=self._last_measurement,
                        released_components=tuple(released),
                        capture_active=self._capture_is_active(),
                        recovery_outcome="release_failed",
                        error_class=type(exc).__name__,
                    )
                )
                continue
            released.append(name)
        return tuple(released)

    async def _record_measurement_failure_locked(self, error: ResourceMeasurementError) -> None:
        self._state = ResourceAdmissionState.DRAINING
        await self._emit_locked(
            ResourceBudgetDiagnostic(
                stage="memory_budget",
                outcome="rejected",
                state=self._state,
                reason=f"Unable to obtain process-specific macOS memory metrics: {error}",
                measurement=None,
                released_components=(),
                capture_active=self._capture_is_active(),
                recovery_outcome="awaiting_reliable_measurement",
                error_class=type(error).__name__,
            )
        )

    async def _emit_locked(self, diagnostic: ResourceBudgetDiagnostic) -> None:
        self._diagnostics.append(diagnostic)
        if self._diagnostic_sink is None:
            return
        result = self._diagnostic_sink(diagnostic)
        if inspect.isawaitable(result):
            await result


def _unique_process_ids(process_ids: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(process_ids))


def _budget_reason(measurement: VoiceMemoryMeasurement) -> str:
    reasons: list[str] = []
    if measurement.model_limit_reached:
        reasons.append(
            f"MLX model resident footprint is {measurement.model_resident_bytes} bytes, at or above the {MODEL_RESIDENT_LIMIT_BYTES}-byte limit"
        )
    if measurement.pipeline_limit_exceeded:
        reasons.append(
            f"local voice pipeline memory is {measurement.pipeline_memory_bytes} bytes, above the {PIPELINE_MEMORY_LIMIT_BYTES}-byte limit"
        )
    return "; ".join(reasons)


def _recovery_reason(measurement: VoiceMemoryMeasurement) -> str:
    if measurement.satisfies_recovery_budget:
        return (
            f"MLX resident footprint ({measurement.model_resident_bytes} bytes) is below {MODEL_RESIDENT_LIMIT_BYTES} bytes and "
            f"pipeline memory ({measurement.pipeline_memory_bytes} bytes) is at or below {PIPELINE_MEMORY_LIMIT_BYTES} bytes"
        )
    return (
        f"Waiting for both recovery thresholds: model must be below {MODEL_RESIDENT_LIMIT_BYTES} bytes "
        f"(currently {measurement.model_resident_bytes}) and pipeline must be at or below "
        f"{PIPELINE_MEMORY_LIMIT_BYTES} bytes (currently {measurement.pipeline_memory_bytes})."
    )


# This module intentionally does not restrict ``from resources import *``:
# existing provisioning helpers predate the resource manager and remain public.
