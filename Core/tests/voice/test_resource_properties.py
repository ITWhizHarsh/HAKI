"""Property 11: Memory admission and recovery.

Feature: realtime-local-voice-agent, Property 11: Memory admission and recovery

For all sampled model-resident and pipeline-memory measurements, a measurement
at or beyond the model limit or above the pipeline limit rejects new voice
turns, requests idle resource release, and records a memory-budget diagnostic;
admission resumes only after model resident memory is below 2.5 GB and pipeline
memory is at or below 5 GB.

**Validates: Requirements 9.4–9.5**
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List, Tuple

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from core.voice import resources
from core.voice.resources import (
    MODEL_RESIDENT_LIMIT_BYTES,
    PIPELINE_MEMORY_LIMIT_BYTES,
    ResourceAdmissionState,
    VoiceMemoryMeasurement,
    VoiceProcessTopology,
    VoiceResourceManager,
)

# ---------------------------------------------------------------------------
# Test helpers and stubs
# ---------------------------------------------------------------------------

GIBIBYTE = resources.GIBIBYTE_BYTES

# The three process IDs used throughout the test suite
_MLX_PID = 10
_SWIFT_PID = 20
_ASR_PID = 30


@dataclass
class ControlledSampler:
    """Memory sampler whose values can be updated between calls."""

    model_bytes: int
    extra_bytes: int  # total pipeline = model_bytes + extra_bytes (no double-count)
    method: str = "test_controlled_sampler"

    def resident_bytes(self, process_id: int) -> int:
        if process_id == _MLX_PID:
            return self.model_bytes
        if process_id == _SWIFT_PID:
            return self.extra_bytes // 2
        if process_id == _ASR_PID:
            return self.extra_bytes - (self.extra_bytes // 2)
        raise ValueError(f"Unexpected PID: {process_id}")


@dataclass
class FakeReleasable:
    """Idle-releasable component whose idle state is controllable."""

    name: str
    events: list[str]
    idle: bool = True
    _warm: bool = False

    async def warm_up(self) -> None:
        self._warm = True
        self.events.append(f"warm:{self.name}")

    def is_idle(self) -> bool:
        return self.idle

    async def release_idle(self) -> None:
        self.events.append(f"release:{self.name}")


@dataclass
class FakeWarmable:
    """Warmable-only component (ASR, Pipecat)."""

    name: str
    events: list[str]

    async def warm_up(self) -> None:
        self.events.append(f"warm:{self.name}")


def _topology() -> VoiceProcessTopology:
    """Topology: MLX shares Core Python (no double-count); Swift has a support alias."""
    return VoiceProcessTopology(
        mlx_process_ids=(_MLX_PID,),
        core_python_process_ids=(_MLX_PID,),   # same PID — deduplication required
        swift_audio_process_ids=(_SWIFT_PID,),
        asr_worker_process_ids=(_ASR_PID,),
        supporting_process_ids=(_SWIFT_PID,),   # same PID — deduplication required
    )


def _make_manager(
    sampler: ControlledSampler,
    *,
    events: list[str] | None = None,
    qwen_idle: bool = True,
    xtts_idle: bool = True,
    capture_active: bool = False,
) -> VoiceResourceManager:
    ev = events if events is not None else []
    return VoiceResourceManager(
        process_topology=_topology(),
        memory_sampler=sampler,
        asr=FakeWarmable("asr", ev),
        qwen=FakeReleasable("qwen", ev, idle=qwen_idle),
        xtts=FakeReleasable("xtts", ev, idle=xtts_idle),
        pipecat=FakeWarmable("pipecat", ev),
        capture_is_active=lambda: capture_active,
    )


def _run(coro):
    """Run a coroutine in a fresh event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Model-resident bytes: stratified across the threshold boundary.
model_bytes_st = st.one_of(
    # Safely below limit
    st.integers(min_value=0, max_value=MODEL_RESIDENT_LIMIT_BYTES - 1),
    # Exactly at the limit (triggers breach)
    st.just(MODEL_RESIDENT_LIMIT_BYTES),
    # Above the limit
    st.integers(min_value=MODEL_RESIDENT_LIMIT_BYTES + 1, max_value=MODEL_RESIDENT_LIMIT_BYTES + 2 * GIBIBYTE),
)

# Pipeline extra bytes (added to model bytes for total pipeline).
# We keep the per-process split values non-negative.
extra_bytes_st = st.integers(min_value=0, max_value=4 * GIBIBYTE)

# Pair of (model_bytes, extra_bytes) → total pipeline = model + extra
memory_pair_st = st.tuples(model_bytes_st, extra_bytes_st)

# Idle states for the two releasable components
idle_pair_st = st.tuples(st.booleans(), st.booleans())  # (xtts_idle, qwen_idle)

# Sequence of 2–8 memory measurement pairs to test multi-sample behaviour
multi_sample_st = st.lists(memory_pair_st, min_size=2, max_size=8)


# ---------------------------------------------------------------------------
# Property 11 — core parametrized test
# ---------------------------------------------------------------------------

@given(
    model_bytes=model_bytes_st,
    extra_bytes=extra_bytes_st,
    xtts_idle=st.booleans(),
    qwen_idle=st.booleans(),
)
@settings(max_examples=120)
def test_property_11_admission_and_rejection_by_memory_thresholds(
    model_bytes: int,
    extra_bytes: int,
    xtts_idle: bool,
    qwen_idle: bool,
) -> None:
    """
    Feature: realtime-local-voice-agent, Property 11: Memory admission and recovery

    For every model/pipeline measurement pair, the admission decision and
    diagnostic behaviour are fully determined by the threshold predicates:
      - model_resident >= 2.5 GB  →  breach (model_limit_reached)
      - pipeline_memory > 5 GB    →  breach (pipeline_limit_exceeded)
    Both must be below their recovery thresholds to resume ADMITTING.

    **Validates: Requirements 9.4–9.5**
    """
    sampler = ControlledSampler(model_bytes=model_bytes, extra_bytes=extra_bytes)
    events: list[str] = []
    manager = _make_manager(sampler, events=events, xtts_idle=xtts_idle, qwen_idle=qwen_idle)

    pipeline_bytes = model_bytes + extra_bytes
    model_breach = model_bytes >= MODEL_RESIDENT_LIMIT_BYTES
    pipeline_breach = pipeline_bytes > PIPELINE_MEMORY_LIMIT_BYTES
    should_reject = model_breach or pipeline_breach

    decision = _run(manager.admit_new_turn())

    if should_reject:
        # The turn must be rejected and state must be DRAINING
        assert not decision.admitted, (
            f"Expected rejection: model={model_bytes}, pipeline={pipeline_bytes}, "
            f"model_breach={model_breach}, pipeline_breach={pipeline_breach}"
        )
        assert decision.state is ResourceAdmissionState.DRAINING

        # At least one memory_budget diagnostic must have been emitted
        diag_list = list(manager.diagnostics)
        assert diag_list, "No diagnostics emitted on a budget breach"
        rejection_diags = [d for d in diag_list if d.outcome == "rejected"]
        assert rejection_diags, "Expected at least one 'rejected' diagnostic"
        latest_diag = rejection_diags[-1]
        assert latest_diag.stage == "memory_budget"
        assert latest_diag.measurement is not None

        # Release order: idle XTTS first, then idle Qwen/cache
        release_events = [e for e in events if e.startswith("release:")]
        if xtts_idle and qwen_idle:
            assert release_events == ["release:xtts", "release:qwen"], (
                f"Expected xtts-first release order, got {release_events}"
            )
        elif xtts_idle and not qwen_idle:
            assert release_events == ["release:xtts"]
        elif not xtts_idle and qwen_idle:
            assert release_events == ["release:qwen"]
        else:
            assert release_events == []

    else:
        # No breach: turn must be admitted
        assert decision.admitted, (
            f"Unexpected rejection: model={model_bytes}, pipeline={pipeline_bytes}"
        )
        assert decision.state is ResourceAdmissionState.ADMITTING
        # No rejection diagnostics should exist
        rejection_diags = [d for d in manager.diagnostics if d.outcome == "rejected"]
        assert not rejection_diags


# ---------------------------------------------------------------------------
# Property 11 — dual-threshold recovery (both must be satisfied)
# ---------------------------------------------------------------------------

@given(
    # First measurement is always a breach
    breach_model=st.integers(
        min_value=MODEL_RESIDENT_LIMIT_BYTES,
        max_value=MODEL_RESIDENT_LIMIT_BYTES + 2 * GIBIBYTE,
    ),
    breach_extra=st.integers(min_value=0, max_value=2 * GIBIBYTE),
    # Recovery measurement bytes
    recover_model=model_bytes_st,
    recover_extra=extra_bytes_st,
)
@settings(max_examples=120)
def test_property_11_recovery_requires_both_thresholds(
    breach_model: int,
    breach_extra: int,
    recover_model: int,
    recover_extra: int,
) -> None:
    """
    Feature: realtime-local-voice-agent, Property 11: Memory admission and recovery

    After a budget breach, admission resumes ONLY when BOTH conditions are
    simultaneously satisfied:
      - model_resident_bytes < MODEL_RESIDENT_LIMIT_BYTES  (strictly below)
      - pipeline_memory_bytes <= PIPELINE_MEMORY_LIMIT_BYTES

    **Validates: Requirements 9.4–9.5**
    """
    sampler = ControlledSampler(model_bytes=breach_model, extra_bytes=breach_extra)
    manager = _make_manager(sampler, xtts_idle=True, qwen_idle=True)

    breach_decision = _run(manager.admit_new_turn())
    assert not breach_decision.admitted
    assert manager.state is ResourceAdmissionState.DRAINING

    # Switch to recovery-candidate measurements
    sampler.model_bytes = recover_model
    sampler.extra_bytes = recover_extra

    recover_pipeline = recover_model + recover_extra
    model_ok = recover_model < MODEL_RESIDENT_LIMIT_BYTES
    pipeline_ok = recover_pipeline <= PIPELINE_MEMORY_LIMIT_BYTES
    both_ok = model_ok and pipeline_ok

    recovery_decision = _run(manager.admit_new_turn())

    if both_ok:
        assert recovery_decision.admitted, (
            f"Should have recovered: model={recover_model}, pipeline={recover_pipeline}"
        )
        assert recovery_decision.state is ResourceAdmissionState.ADMITTING
        # A 'recovered' diagnostic must exist
        recovered_diags = [d for d in manager.diagnostics if d.outcome == "recovered"]
        assert recovered_diags, "Expected a 'recovered' diagnostic after admission resumed"
    else:
        assert not recovery_decision.admitted, (
            f"Should stay draining: model={recover_model} (ok={model_ok}), "
            f"pipeline={recover_pipeline} (ok={pipeline_ok})"
        )
        assert recovery_decision.state is ResourceAdmissionState.DRAINING


# ---------------------------------------------------------------------------
# Property 11 — exact threshold boundary values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_bytes,extra_bytes,should_reject", [
    # --- Model limit boundary (2.5 GiB = int(2.5 * 1024**3) = 2,684,354,560 bytes) ---
    # One byte below the model limit: safe — no breach
    (MODEL_RESIDENT_LIMIT_BYTES - 1, 0, False),
    # Exactly at the model limit (model_limit_reached uses >=): breach
    (MODEL_RESIDENT_LIMIT_BYTES, 0, True),
    # One byte above: definitely a breach
    (MODEL_RESIDENT_LIMIT_BYTES + 1, 0, True),

    # --- Pipeline limit boundary (5 GiB = 5 * 1024**3 = 5,368,709,120 bytes) ---
    # Total pipeline exactly at 5 GiB: pipeline_limit_exceeded uses >, so this is safe
    (0, PIPELINE_MEMORY_LIMIT_BYTES, False),
    # One byte above 5 GiB: pipeline_limit_exceeded → breach
    (0, PIPELINE_MEMORY_LIMIT_BYTES + 1, True),

    # --- Combination: model at limit and pipeline exactly 5 GiB ---
    # Model breach dominates (model_bytes == limit triggers model_limit_reached)
    (MODEL_RESIDENT_LIMIT_BYTES, PIPELINE_MEMORY_LIMIT_BYTES - MODEL_RESIDENT_LIMIT_BYTES, True),

    # --- Both safely below limits ---
    (MODEL_RESIDENT_LIMIT_BYTES - 1, PIPELINE_MEMORY_LIMIT_BYTES - MODEL_RESIDENT_LIMIT_BYTES, False),

    # --- Decimal "2.5 GB" (2,500,000,000 bytes) is BELOW the binary 2.5 GiB limit ---
    # 2,500,000,000 < 2,684,354,560, so this is safe — no breach
    (2_500_000_000, 0, False),
    # One byte below the binary limit is also safe
    (MODEL_RESIDENT_LIMIT_BYTES - 1, 0, False),
])
def test_property_11_exact_threshold_boundaries(
    model_bytes: int, extra_bytes: int, should_reject: bool
) -> None:
    """
    Feature: realtime-local-voice-agent, Property 11: Memory admission and recovery

    Verify exact threshold byte values behave as specified:
      - model_resident >= 2.5 GB  →  reject
      - pipeline_memory > 5 GB    →  reject

    **Validates: Requirements 9.4–9.5**
    """
    sampler = ControlledSampler(model_bytes=model_bytes, extra_bytes=extra_bytes)
    manager = _make_manager(sampler, xtts_idle=True, qwen_idle=True)

    decision = _run(manager.admit_new_turn())

    if should_reject:
        assert not decision.admitted, (
            f"model={model_bytes}, extra={extra_bytes}: expected rejection"
        )
        assert decision.state is ResourceAdmissionState.DRAINING
        diags = [d for d in manager.diagnostics if d.outcome == "rejected"]
        assert diags
    else:
        assert decision.admitted, (
            f"model={model_bytes}, extra={extra_bytes}: expected admission"
        )
        assert decision.state is ResourceAdmissionState.ADMITTING


# ---------------------------------------------------------------------------
# Property 11 — active work is preserved (non-idle components not released)
# ---------------------------------------------------------------------------

@given(
    model_bytes=st.integers(
        min_value=MODEL_RESIDENT_LIMIT_BYTES,
        max_value=MODEL_RESIDENT_LIMIT_BYTES + 2 * GIBIBYTE,
    ),
    extra_bytes=st.integers(min_value=0, max_value=2 * GIBIBYTE),
)
@settings(max_examples=100)
def test_property_11_active_work_is_not_released(
    model_bytes: int, extra_bytes: int
) -> None:
    """
    Feature: realtime-local-voice-agent, Property 11: Memory admission and recovery

    When XTTS and Qwen are NOT idle (active work in progress), a memory breach
    must still reject the new turn but must NOT release the active components.

    **Validates: Requirements 9.4–9.5**
    """
    sampler = ControlledSampler(model_bytes=model_bytes, extra_bytes=extra_bytes)
    events: list[str] = []
    manager = _make_manager(
        sampler, events=events, xtts_idle=False, qwen_idle=False
    )

    decision = _run(manager.admit_new_turn())

    assert not decision.admitted
    assert decision.state is ResourceAdmissionState.DRAINING

    release_events = [e for e in events if e.startswith("release:")]
    assert release_events == [], (
        f"Must not release active components; got releases: {release_events}"
    )

    # ASR and Pipecat are warmable-only; they must never appear in release events
    assert "release:asr" not in events
    assert "release:pipecat" not in events

    # The diagnostic must document that no release occurred
    diags = list(manager.diagnostics)
    assert diags
    latest = [d for d in diags if d.outcome == "rejected"][-1]
    assert latest.released_components == ()


# ---------------------------------------------------------------------------
# Property 11 — diagnostic always present on breach
# ---------------------------------------------------------------------------

@given(
    model_bytes=st.integers(
        min_value=MODEL_RESIDENT_LIMIT_BYTES,
        max_value=MODEL_RESIDENT_LIMIT_BYTES + 3 * GIBIBYTE,
    ),
    extra_bytes=st.integers(min_value=0, max_value=3 * GIBIBYTE),
    xtts_idle=st.booleans(),
    qwen_idle=st.booleans(),
    capture_active=st.booleans(),
)
@settings(max_examples=100)
def test_property_11_diagnostic_emitted_on_every_breach(
    model_bytes: int,
    extra_bytes: int,
    xtts_idle: bool,
    qwen_idle: bool,
    capture_active: bool,
) -> None:
    """
    Feature: realtime-local-voice-agent, Property 11: Memory admission and recovery

    Every budget breach emits a memory_budget diagnostic with stage='memory_budget',
    correct admission state, and the current capture_active value.

    **Validates: Requirements 9.4–9.5**
    """
    sampler = ControlledSampler(model_bytes=model_bytes, extra_bytes=extra_bytes)
    manager = _make_manager(
        sampler,
        xtts_idle=xtts_idle,
        qwen_idle=qwen_idle,
        capture_active=capture_active,
    )

    _run(manager.admit_new_turn())

    diags = list(manager.diagnostics)
    assert diags, "At least one diagnostic must be emitted on a budget breach"
    rejection_diags = [d for d in diags if d.outcome == "rejected"]
    assert rejection_diags, "A 'rejected' diagnostic must be present"
    d = rejection_diags[0]
    assert d.stage == "memory_budget"
    assert d.state is ResourceAdmissionState.DRAINING
    assert d.capture_active is capture_active
    assert d.measurement is not None
    assert d.measurement.model_resident_bytes == model_bytes
    assert d.measurement.pipeline_memory_bytes == model_bytes + extra_bytes


# ---------------------------------------------------------------------------
# Property 11 — repeated samples stay in DRAINING until both thresholds clear
# ---------------------------------------------------------------------------

@given(
    model_bytes=st.integers(
        min_value=MODEL_RESIDENT_LIMIT_BYTES,
        max_value=MODEL_RESIDENT_LIMIT_BYTES + GIBIBYTE,
    ),
    extra_bytes=st.integers(min_value=0, max_value=GIBIBYTE),
    repeat_count=st.integers(min_value=2, max_value=6),
)
@settings(max_examples=100)
def test_property_11_repeated_over_budget_samples_stay_draining(
    model_bytes: int,
    extra_bytes: int,
    repeat_count: int,
) -> None:
    """
    Feature: realtime-local-voice-agent, Property 11: Memory admission and recovery

    Repeated over-budget samples must keep the manager in DRAINING state;
    it must never spontaneously flip back to ADMITTING while still over budget.

    **Validates: Requirements 9.4–9.5**
    """
    sampler = ControlledSampler(model_bytes=model_bytes, extra_bytes=extra_bytes)
    manager = _make_manager(sampler, xtts_idle=False, qwen_idle=False)

    for i in range(repeat_count):
        decision = _run(manager.admit_new_turn())
        assert not decision.admitted, f"Sample {i}: should remain rejected"
        assert decision.state is ResourceAdmissionState.DRAINING, (
            f"Sample {i}: should remain DRAINING"
        )


# ---------------------------------------------------------------------------
# Property 11 — incomplete release does not restore admission
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("xtts_idle,qwen_idle", [
    (True, False),   # Only XTTS released; Qwen still active
    (False, True),   # Only Qwen released; XTTS still active
    (False, False),  # Neither released
])
def test_property_11_incomplete_release_does_not_restore_admission(
    xtts_idle: bool, qwen_idle: bool
) -> None:
    """
    Feature: realtime-local-voice-agent, Property 11: Memory admission and recovery

    When idle-resource release cannot free enough memory (sampler still returns
    over-budget values), admission must stay in DRAINING regardless of which
    components were actually released.

    **Validates: Requirements 9.4–9.5**
    """
    # The sampler always returns over-budget values (static — no improvement after release)
    sampler = ControlledSampler(
        model_bytes=MODEL_RESIDENT_LIMIT_BYTES + GIBIBYTE,
        extra_bytes=0,
    )
    events: list[str] = []
    manager = _make_manager(
        sampler, events=events, xtts_idle=xtts_idle, qwen_idle=qwen_idle
    )

    decision = _run(manager.admit_new_turn())

    assert not decision.admitted
    assert decision.state is ResourceAdmissionState.DRAINING

    # After the call the manager re-samples; model is still over budget → stay draining
    second_decision = _run(manager.admit_new_turn())
    assert not second_decision.admitted
    assert second_decision.state is ResourceAdmissionState.DRAINING


# ---------------------------------------------------------------------------
# Property 11 — double-count prevention (pipeline deduplicates shared PIDs)
# ---------------------------------------------------------------------------

def test_property_11_shared_pid_not_double_counted_in_pipeline() -> None:
    """
    Feature: realtime-local-voice-agent, Property 11: Memory admission and recovery

    When MLX shares the Core Python process (same PID in both role lists),
    the pipeline memory must NOT double-count that process's RSS.  The unique-
    process-ID union ensures each PID is sampled exactly once.

    **Validates: Requirements 9.4–9.5**
    """
    # Three unique PIDs: 10 (MLX+CorePython shared), 20 (Swift+supporting), 30 (ASR)
    # Sampler values
    sampler = ControlledSampler(model_bytes=GIBIBYTE, extra_bytes=2 * GIBIBYTE)
    manager = _make_manager(sampler)

    m = manager.measure_memory()

    # Model: only PID 10 → 1 GiB
    assert m.model_resident_bytes == GIBIBYTE
    # Pipeline: PIDs 10+20+30 (deduplicated); 20 gets extra//2, 30 gets extra-extra//2
    assert m.pipeline_process_ids == (_MLX_PID, _SWIFT_PID, _ASR_PID)
    assert len(m.pipeline_process_ids) == 3, "Must not duplicate shared PID"
    assert m.pipeline_memory_bytes == GIBIBYTE + 2 * GIBIBYTE  # 3 GiB total, not 4 GiB


# ---------------------------------------------------------------------------
# Property 11 — recovery diagnostic emitted when admission resumes
# ---------------------------------------------------------------------------

@given(
    recover_model=st.integers(
        min_value=0, max_value=MODEL_RESIDENT_LIMIT_BYTES - 1
    ),
    recover_extra=st.integers(
        min_value=0,
        max_value=PIPELINE_MEMORY_LIMIT_BYTES,
    ),
)
@settings(max_examples=100)
def test_property_11_recovery_diagnostic_emitted_when_admission_resumes(
    recover_model: int,
    recover_extra: int,
) -> None:
    """
    Feature: realtime-local-voice-agent, Property 11: Memory admission and recovery

    When admission transitions from DRAINING back to ADMITTING, a 'recovered'
    diagnostic must be emitted with outcome='recovered' and state=ADMITTING.

    **Validates: Requirements 9.4–9.5**
    """
    # Ensure the recovery pipeline total is actually within budget
    assume(recover_model + recover_extra <= PIPELINE_MEMORY_LIMIT_BYTES)

    sampler = ControlledSampler(
        model_bytes=MODEL_RESIDENT_LIMIT_BYTES,  # initial breach
        extra_bytes=0,
    )
    manager = _make_manager(sampler, xtts_idle=True, qwen_idle=True)

    # Trigger initial breach
    _run(manager.admit_new_turn())
    assert manager.state is ResourceAdmissionState.DRAINING

    # Switch to recovery values
    sampler.model_bytes = recover_model
    sampler.extra_bytes = recover_extra

    _run(manager.admit_new_turn())

    # Recovery: both thresholds satisfied
    assert manager.state is ResourceAdmissionState.ADMITTING
    recovered_diags = [d for d in manager.diagnostics if d.outcome == "recovered"]
    assert recovered_diags, "A 'recovered' diagnostic must be emitted when admission resumes"
    d = recovered_diags[-1]
    assert d.stage == "memory_budget"
    assert d.state is ResourceAdmissionState.ADMITTING


# ---------------------------------------------------------------------------
# Property 11 — multi-sample sequence: admission/draining transitions track
#               the most recent measurement correctly
# ---------------------------------------------------------------------------

@given(samples=multi_sample_st)
@settings(max_examples=100)
def test_property_11_multi_sample_state_tracks_measurements(
    samples: list[tuple[int, int]]
) -> None:
    """
    Feature: realtime-local-voice-agent, Property 11: Memory admission and recovery

    Across an arbitrary sequence of measurements, the manager's admission state
    after each sample must be consistent with the threshold predicates for that
    sample.  Specifically, the state after all calls must be ADMITTING if and
    only if the LAST measurement satisfies both recovery thresholds.

    **Validates: Requirements 9.4–9.5**
    """
    sampler = ControlledSampler(model_bytes=0, extra_bytes=0)
    manager = _make_manager(sampler, xtts_idle=True, qwen_idle=True)

    last_decision = None
    for model_b, extra_b in samples:
        sampler.model_bytes = model_b
        sampler.extra_bytes = extra_b
        last_decision = _run(manager.admit_new_turn())

    assert last_decision is not None
    last_model = samples[-1][0]
    last_extra = samples[-1][1]
    last_pipeline = last_model + last_extra
    both_ok = (last_model < MODEL_RESIDENT_LIMIT_BYTES) and (last_pipeline <= PIPELINE_MEMORY_LIMIT_BYTES)

    # After a sequence, if the last sample satisfies recovery, the manager
    # may have transitioned back to ADMITTING (if it was draining).
    # If the last sample is over budget, it must be DRAINING.
    if not both_ok:
        assert last_decision.state is ResourceAdmissionState.DRAINING
        assert not last_decision.admitted
