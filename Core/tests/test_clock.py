"""
Unit tests for the Clock subsystem.

Feature: haki-personal-ai-assistant
Requirements: 14.1, 14.3, 14.4, 14.5
"""

from __future__ import annotations

import asyncio
import datetime

import pytest

from core.clock import Clock, ClockResult, ClockUnavailable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fixed_override(tz_name: str = "UTC") -> datetime.datetime:
    """Return a fixed, timezone-aware datetime for 2024-06-01 12:00:00 UTC."""
    tz = datetime.timezone.utc
    return datetime.datetime(2024, 6, 1, 12, 0, 0, tzinfo=tz)


# ---------------------------------------------------------------------------
# Task 3.1 tests — now() behaviour
# ---------------------------------------------------------------------------


def test_now_returns_clock_result():
    """
    Calling now() on a default Clock returns a ClockResult with
    non-None date, time, and timezone fields.

    Requirements: 14.1, 14.3.
    """
    clock = Clock()
    result = clock.now()

    assert isinstance(result, ClockResult), (
        f"Expected ClockResult, got {type(result)}: {result}"
    )
    assert result.date is not None, "ClockResult.date must not be None"
    assert result.time is not None, "ClockResult.time must not be None"
    assert result.timezone is not None, "ClockResult.timezone must not be None"


def test_now_returns_unavailable_when_system_fails():
    """
    If _override_now raises an OSError, now() must return ClockUnavailable
    (never raise).

    Requirement: 14.5.
    """
    def broken_now() -> datetime.datetime:
        raise OSError("simulated system clock failure")

    clock = Clock(_override_now=broken_now)
    result = clock.now()

    assert isinstance(result, ClockUnavailable), (
        f"Expected ClockUnavailable, got {type(result)}: {result}"
    )
    assert "simulated system clock failure" in result.reason


def test_now_override_seam():
    """
    Injecting a fixed datetime via _override_now causes now() to return
    a ClockResult matching that fixed value.

    Requirement: 14.1.
    """
    fixed_dt = datetime.datetime(2024, 6, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)

    clock = Clock(_override_now=lambda: fixed_dt)
    result = clock.now()

    assert isinstance(result, ClockResult)
    assert result.date == datetime.date(2024, 6, 1)
    assert result.time == datetime.time(12, 0, 0)


def test_timezone_field_is_iana_string():
    """
    ClockResult.timezone must be a non-empty string (IANA format sanity check).

    Requirements: 14.1, 14.3.
    """
    clock = Clock()
    result = clock.now()

    assert isinstance(result, ClockResult)
    assert isinstance(result.timezone, str)
    assert len(result.timezone) > 0, "timezone must be a non-empty string"


# ---------------------------------------------------------------------------
# Task 3.2 tests — watch_timezone() behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watch_timezone_yields_on_change():
    """
    When the system timezone changes, watch_timezone() must yield the new
    timezone name.

    Uses _poll_interval_seconds = 0.05 and an _override_timezone seam to
    simulate a change from "UTC" to "Asia/Kolkata".

    Requirement: 14.4.
    """
    call_count = 0
    tz_sequence = ["UTC", "Asia/Kolkata"]

    def override_now() -> datetime.datetime:
        nonlocal call_count
        # First call returns UTC, subsequent calls return Asia/Kolkata.
        tz_name = tz_sequence[min(call_count, len(tz_sequence) - 1)]
        call_count += 1
        # Build a fixed-offset timezone that matches the name we want to
        # report, but we rely on the _iana_name fallback which reads %Z.
        # Simpler: supply a ZoneInfo so _iana_name returns the key directly.
        import zoneinfo
        tz = zoneinfo.ZoneInfo(tz_name)
        return datetime.datetime(2024, 6, 1, 12, 0, 0, tzinfo=tz)

    clock = Clock(_override_now=override_now)
    clock._poll_interval_seconds = 0.05

    collected: list[str] = []

    gen = clock.watch_timezone()
    try:
        # Collect the first yielded value, then cancel.
        yielded = await asyncio.wait_for(gen.__anext__(), timeout=2.0)
        collected.append(yielded)
    except StopAsyncIteration:
        pass
    finally:
        await gen.aclose()

    assert len(collected) == 1, f"Expected exactly one yielded value, got {collected}"
    assert collected[0] == "Asia/Kolkata", (
        f"Expected 'Asia/Kolkata', got '{collected[0]}'"
    )


@pytest.mark.asyncio
async def test_watch_timezone_does_not_yield_when_unchanged():
    """
    When the timezone never changes, watch_timezone() must yield nothing
    for several poll cycles.

    Runs for approximately 3 poll cycles (0.05 s each → ~0.15 s total).

    Requirement: 14.4.
    """
    import zoneinfo

    def stable_now() -> datetime.datetime:
        tz = zoneinfo.ZoneInfo("UTC")
        return datetime.datetime(2024, 6, 1, 12, 0, 0, tzinfo=tz)

    clock = Clock(_override_now=stable_now)
    clock._poll_interval_seconds = 0.05

    collected: list[str] = []

    async def collect_for(seconds: float) -> None:
        gen = clock.watch_timezone()
        try:
            async with asyncio.timeout(seconds):
                async for tz in gen:
                    collected.append(tz)
        except TimeoutError:
            pass
        finally:
            await gen.aclose()

    # Run for 3+ poll cycles (0.05 * 4 = 0.2 s)
    await collect_for(0.2)

    assert collected == [], (
        f"Expected no yields when timezone is stable, got {collected}"
    )
