"""Static dependency tests for the replacement ``core.voice`` namespace.

Validates: Requirements 1.5, 4.1, 6.1, 7.1
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from uuid import uuid4

import pytest

import core.voice as voice
from core.voice.interfaces import LocalVoiceLLM, LocalVoiceTTS, VoiceSentence, VoiceTurnPipeline, VoiceTurnRequest


_CHECKER_PATH = Path(__file__).resolve().parents[3] / "tools" / "check_voice_import_boundary.py"
_SPEC = importlib.util.spec_from_file_location("check_voice_import_boundary", _CHECKER_PATH)
assert _SPEC and _SPEC.loader
checker = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = checker
_SPEC.loader.exec_module(checker)


def _voice_source(tmp_path: Path, source: str) -> tuple[Path, Path]:
    module_path = tmp_path / "Core/core/voice/module.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(source, encoding="utf-8")
    return tmp_path, module_path.relative_to(tmp_path)


def test_package_exports_only_typed_voice_ports_and_availability_interfaces() -> None:
    """The public namespace exposes voice-owned contracts, not legacy runtime objects."""
    assert voice.__all__ == [
        "LocalVoiceLLM",
        "LocalVoiceTTS",
        "ModelArtifactManifest",
        "VoiceAvailabilityIssue",
        "VoiceContextMessage",
        "VoiceLanguage",
        "VoiceModelManifest",
        "VoiceSentence",
        "VoiceStartupHealth",
        "VoiceTurnPipeline",
        "VoiceTurnRequest",
    ]
    assert isinstance(VoiceTurnRequest(uuid4(), uuid4(), "hello", "en"), VoiceTurnRequest)
    assert isinstance(VoiceSentence(uuid4(), uuid4(), "Hello.", "en"), VoiceSentence)
    assert VoiceTurnPipeline.__module__ == "core.voice.interfaces"
    assert LocalVoiceLLM.__module__ == "core.voice.interfaces"
    assert LocalVoiceTTS.__module__ == "core.voice.interfaces"


def test_current_voice_package_satisfies_the_static_import_boundary() -> None:
    """The committed namespace remains independent from legacy runtime paths."""
    repo_root = Path(__file__).resolve().parents[3]
    assert checker.check_voice_package(repo_root) == []


def test_capability_and_voice_protocol_dependencies_are_explicitly_permitted(tmp_path: Path) -> None:
    """Future adapters may reach only their authoritative capability/protocol targets."""
    root, relative = _voice_source(
        tmp_path,
        "\n".join(
            (
                "from core.voice.interfaces import VoiceTurnRequest",
                "from core.memory.haki_brain import HAKIBrain",
                "from core.automation.screen_agent import ScreenAgent",
                "from core.ipc.voice_protocol import TranscriptEvent",
            )
        ),
    )

    assert checker.scan_voice_module(relative, (root / relative).read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from core.model_provider.stt_engine import STTEngine", "model-provider runtime"),
        ("from core.model_provider.tts_engine import TTSEngine", "model-provider runtime"),
        ("from core.model_provider.llm_router import LLMRouter", "model-provider runtime"),
        ("from core.orchestrator.orchestrator import Orchestrator", "process-wide Orchestrator"),
        ("router._routing_order('hello', False)", "LLMRouter._routing_order"),
        ("orchestrator._conversation_history.append({'role': 'user'})", "conversation history"),
        ("from core.dialogue.manager import DialogueManager", "non-permitted Core dependency"),
    ],
)
def test_static_checker_rejects_prohibited_router_and_orchestrator_coupling(
    tmp_path: Path, source: str, expected: str
) -> None:
    """The boundary rejects every prohibited voice dependency mechanism."""
    root, relative = _voice_source(tmp_path, source)

    violations = checker.scan_voice_module(relative, (root / relative).read_text(encoding="utf-8"))

    assert any(expected in violation for violation in violations)


def test_static_checker_rejects_dynamic_archive_import(tmp_path: Path) -> None:
    """Archive content cannot be reached through a dynamic import expression."""
    archive_module = "_".join(("legacy", "pipeline", "backup"))
    source = f"import importlib\\nimportlib.import_module('{archive_module}.Core.core.ipc.server')"
    root, relative = _voice_source(tmp_path, source)

    violations = checker.scan_voice_module(relative, (root / relative).read_text(encoding="utf-8"))

    assert any("static archive" in violation for violation in violations)
