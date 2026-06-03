"""
Smoke tests for the HAKI Core package scaffold.

Verifies that:
1. The package and all sub-modules are importable.
2. The Hypothesis + pytest test harness is wired correctly.
3. A minimal property test exercises the framework end-to-end.

Feature: haki-personal-ai-assistant
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ------------------------------------------------------------------
# Import smoke tests — confirm every sub-package is importable
# ------------------------------------------------------------------

def test_core_importable():
    import core  # noqa: F401
    assert core.__version__ == "0.1.0"


def test_orchestrator_importable():
    from core.orchestrator import Orchestrator
    assert Orchestrator is not None


def test_model_provider_importable():
    from core.model_provider import Capability, ModelMode, CapabilityConfig, ModelProvider
    assert set(Capability) == {
        Capability.STT, Capability.LLM, Capability.TTS,
        Capability.MOOD, Capability.IMAGE, Capability.EMBEDDINGS,
    }


def test_memory_importable():
    from core.memory import Note, Chunk, MemoryBrain
    assert Note is not None
    assert Chunk is not None
    assert MemoryBrain is not None


def test_learning_importable():
    from core.learning import LearnedItem, LearningReport, LearningEngine
    assert LearningEngine is not None


def test_planner_importable():
    from core.planner import Step, CommandPlan, Planner
    assert Planner is not None


def test_dialogue_importable():
    from core.dialogue import SlotFillResult, DialogueManager
    assert DialogueManager is not None


def test_ipc_importable():
    from core.ipc import IPCServer
    assert IPCServer is not None


# ------------------------------------------------------------------
# Minimal instantiation smoke tests
# ------------------------------------------------------------------

def test_orchestrator_instantiates():
    from core.orchestrator import Orchestrator
    orch = Orchestrator()
    assert orch is not None


def test_memory_brain_init(tmp_vault):
    from core.memory import MemoryBrain
    brain = MemoryBrain(vault_path=tmp_vault)
    brain.init()
    assert tmp_vault.exists()


def test_planner_produces_plan():
    from core.planner import Planner, CommandPlan
    planner = Planner()
    plan = planner.plan("open Safari")
    assert isinstance(plan, CommandPlan)
    assert len(plan.steps) >= 1
    assert plan.origin_command == "open Safari"


def test_dialogue_manager_assess_no_slots():
    from core.dialogue import DialogueManager
    dm = DialogueManager()
    result = dm.assess("hello", needed_slots=[])
    assert result.sufficient is True
    assert result.missing == []


def test_ipc_server_instantiates():
    from core.ipc import IPCServer
    server = IPCServer(socket_path="/tmp/haki_test.sock")
    assert server is not None


# ------------------------------------------------------------------
# Hypothesis property test — framework smoke
#
# Feature: haki-personal-ai-assistant, Property 0: Hypothesis harness works
#
# This is a meta-property that verifies the test framework itself is
# correctly configured.  It is intentionally trivial — the real
# correctness properties begin at Property 1 (Tasks 7+).
# ------------------------------------------------------------------

@settings(max_examples=100)
@given(st.integers(), st.integers())
def test_hypothesis_framework_smoke(a: int, b: int):
    """
    Feature: haki-personal-ai-assistant, Property 0: Hypothesis harness works.

    Validates that addition is commutative — a trivial property that
    exercises the Hypothesis engine and confirms it is correctly installed
    and integrated with pytest.
    """
    assert a + b == b + a


@settings(max_examples=100)
@given(st.text(min_size=1))
def test_memory_brain_remember_retrieve_roundtrip(body: str):
    """
    Feature: haki-personal-ai-assistant, Property 0b: MemoryBrain store is queryable.

    For any non-empty string *body*, storing it should succeed and the
    note should be retrievable via all_notes().

    This is a lightweight scaffold check; the full round-trip property
    (Property 16) is implemented in Task 13.4.
    """
    import tempfile
    from pathlib import Path
    from core.memory import MemoryBrain
    with tempfile.TemporaryDirectory() as tmp:
        brain = MemoryBrain(vault_path=Path(tmp) / "vault")
        brain.init()

        result = brain.remember(content=body)
        # remember() must return a successful StoreResult (Req 7.1)
        assert result.success is True
        assert result.note_id is not None

        # The note must be persisted on disk (Req 7.5)
        assert result.note_id in {n.id for n in brain.all_notes()}


@settings(max_examples=100)
@given(st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5))
def test_command_plan_ready_steps_never_exceed_total(steps_intents: list[str]):
    """
    Feature: haki-personal-ai-assistant, Property 0c: ready_steps ⊆ all steps.

    For any CommandPlan, the number of ready steps can never exceed the
    total number of steps.
    """
    from core.planner import CommandPlan, Step, StepStatus

    plan = CommandPlan(origin_command="test")
    for intent in steps_intents:
        plan.steps.append(Step(intent=intent))

    ready = plan.ready_steps()
    assert len(ready) <= len(plan.steps)
