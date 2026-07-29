"""Focused unit coverage for warmed local voice resource admission.

Requirements: 9.1, 9.2, 9.4–9.5
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from core.voice import resources


@dataclass
class StubSampler:
    samples: dict[int, int]
    method: str = "mock_macos_process_rss"
    calls: list[int] = field(default_factory=list)

    def resident_bytes(self, process_id: int) -> int:
        self.calls.append(process_id)
        return self.samples[process_id]


@dataclass
class FakeComponent:
    name: str
    events: list[str]
    idle: bool = True

    async def warm_up(self) -> None:
        self.events.append(f"warm:{self.name}")

    def is_idle(self) -> bool:
        return self.idle

    async def release_idle(self) -> None:
        self.events.append(f"release:{self.name}")


def _topology() -> resources.VoiceProcessTopology:
    """MLX shares Core Python, and Swift shares a support-process registration."""
    return resources.VoiceProcessTopology(
        mlx_process_ids=(10,),
        core_python_process_ids=(10,),
        swift_audio_process_ids=(20,),
        asr_worker_process_ids=(30,),
        supporting_process_ids=(20,),
    )


def _manager(
    sampler: StubSampler,
    *,
    events: list[str] | None = None,
    qwen_idle: bool = False,
    xtts_idle: bool = False,
    capture_active: bool = False,
) -> resources.VoiceResourceManager:
    events = events if events is not None else []
    return resources.VoiceResourceManager(
        process_topology=_topology(),
        memory_sampler=sampler,
        asr=FakeComponent("asr", events),
        qwen=FakeComponent("qwen", events, idle=qwen_idle),
        xtts=FakeComponent("xtts", events, idle=xtts_idle),
        pipecat=FakeComponent("pipecat", events),
        capture_is_active=lambda: capture_active,
    )


@pytest.mark.asyncio
async def test_warm_up_prepares_all_required_components_before_sampling() -> None:
    events: list[str] = []
    manager = _manager(StubSampler({10: 1, 20: 1, 30: 1}), events=events)

    measurement = await manager.warm_up()

    assert events == ["warm:asr", "warm:qwen", "warm:xtts", "warm:pipecat"]
    assert measurement.model_resident_bytes == 1
    assert manager.state is resources.ResourceAdmissionState.ADMITTING


def test_process_rss_measurement_uses_unique_pipeline_pid_union() -> None:
    sampler = StubSampler(
        {
            10: resources.GIBIBYTE_BYTES,
            20: 2 * resources.GIBIBYTE_BYTES,
            30: 3 * resources.GIBIBYTE_BYTES,
        }
    )
    manager = _manager(sampler)

    measurement = manager.measure_memory()

    assert sampler.calls == [10, 20, 30]
    assert measurement.model_process_ids == (10,)
    assert measurement.pipeline_process_ids == (10, 20, 30)
    assert measurement.model_resident_bytes == resources.GIBIBYTE_BYTES
    assert measurement.pipeline_memory_bytes == 6 * resources.GIBIBYTE_BYTES
    assert measurement.sampling_method == "mock_macos_process_rss"


@pytest.mark.asyncio
async def test_limit_breach_releases_idle_xtts_then_qwen_without_touching_capture() -> None:
    events: list[str] = []
    sampler = StubSampler(
        {
            10: resources.MODEL_RESIDENT_LIMIT_BYTES,
            20: 0,
            30: 0,
        }
    )
    manager = _manager(
        sampler,
        events=events,
        qwen_idle=True,
        xtts_idle=True,
        capture_active=True,
    )

    decision = await manager.admit_new_turn()

    assert not decision.admitted
    assert manager.state is resources.ResourceAdmissionState.DRAINING
    assert events == ["release:xtts", "release:qwen"]
    assert all("release:asr" not in event and "release:pipecat" not in event for event in events)
    diagnostic = manager.diagnostics[-1]
    assert diagnostic.capture_active is True
    assert diagnostic.released_components == ("xtts", "qwen")


@pytest.mark.asyncio
async def test_rejection_diagnostic_reports_both_budget_violations() -> None:
    sampler = StubSampler(
        {
            10: 3 * resources.GIBIBYTE_BYTES,
            20: 2 * resources.GIBIBYTE_BYTES,
            30: resources.GIBIBYTE_BYTES,
        }
    )
    manager = _manager(sampler, capture_active=True)

    decision = await manager.admit_new_turn()

    assert not decision.admitted
    assert decision.reason is not None
    assert "MLX model resident footprint" in decision.reason
    assert "local voice pipeline memory" in decision.reason
    diagnostic = manager.diagnostics[-1]
    assert diagnostic.stage == "memory_budget"
    assert diagnostic.outcome == "rejected"
    assert diagnostic.capture_active is True
    assert diagnostic.measurement is not None
    assert diagnostic.measurement.sampling_method == "mock_macos_process_rss"


@pytest.mark.asyncio
async def test_recovery_requires_model_below_and_pipeline_at_or_below_limits() -> None:
    sampler = StubSampler(
        {
            10: resources.MODEL_RESIDENT_LIMIT_BYTES,
            20: 0,
            30: 0,
        }
    )
    manager = _manager(sampler)

    breach = await manager.admit_new_turn()
    assert not breach.admitted
    assert manager.state is resources.ResourceAdmissionState.DRAINING

    # The model's exact 2.5 GiB limit remains draining even though pipeline is safe.
    still_draining = await manager.admit_new_turn()
    assert not still_draining.admitted
    assert manager.state is resources.ResourceAdmissionState.DRAINING

    # Both recovery predicates are now true: model is strictly below and pipeline equals 5 GiB.
    sampler.samples = {
        10: resources.MODEL_RESIDENT_LIMIT_BYTES - 1,
        20: resources.PIPELINE_MEMORY_LIMIT_BYTES - (resources.MODEL_RESIDENT_LIMIT_BYTES - 1),
        30: 0,
    }
    recovered = await manager.admit_new_turn()

    assert recovered.admitted
    assert recovered.state is resources.ResourceAdmissionState.ADMITTING
    assert recovered.measurement is not None
    assert recovered.measurement.model_resident_bytes == resources.MODEL_RESIDENT_LIMIT_BYTES - 1
    assert recovered.measurement.pipeline_memory_bytes == resources.PIPELINE_MEMORY_LIMIT_BYTES
    assert manager.diagnostics[-1].outcome == "recovered"
