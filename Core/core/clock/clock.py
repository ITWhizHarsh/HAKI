"""
Clock — current date / time / timezone provider and timezone-change watcher.

Design reference: Clock component.
Requirements: 14.1, 14.2, 14.3, 14.4, 14.5.

Public types
------------
ClockResult
    Successful reading: carries ``date``, ``time``, and ``timezone`` (IANA name).

ClockUnavailable
    Failure reading: carries a human-readable ``reason`` string.

Clock
    Thin, synchronous ``now()`` that never raises (14.1, 14.5), plus an
    async ``watch_timezone()`` generator that yields the new IANA timezone
    name whenever it changes (14.4).

Testing seams
-------------
``_override_now``   — callable injected at construction time; when set,
                      ``now()`` calls it instead of reading the system
                      clock.  If the callable raises, ``ClockUnavailable``
                      is returned.
``_poll_interval_seconds`` — interval used by ``watch_timezone()``
                      (default 1.0).  Lower it in tests to speed up
                      polling without real sleeps.
"""

from __future__ import annotations

import asyncio
import datetime
import zoneinfo
from dataclasses import dataclass
from typing import AsyncGenerator, Callable, Optional


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClockResult:
    """A successful reading of the system clock.

    Attributes
    ----------
    date:
        Current local date.
    time:
        Current local time (no microsecond precision needed, but preserved).
    timezone:
        IANA timezone identifier (e.g. ``"Asia/Kolkata"``), or the
        POSIX abbreviation reported by ``datetime.astimezone()`` when the
        IANA name is not directly resolvable.
    """

    date: datetime.date
    time: datetime.time
    timezone: str


@dataclass(frozen=True)
class ClockUnavailable:
    """Returned when the system clock cannot be read (Requirement 14.5).

    Attributes
    ----------
    reason:
        Human-readable description of why the clock is unavailable.
    """

    reason: str


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------


class Clock:
    """Single source of truth for current date, time, and timezone.

    Parameters
    ----------
    _override_now:
        Optional callable that, when provided, is used *instead* of reading
        the real system clock.  It must return a timezone-aware
        ``datetime.datetime``.  If it raises any exception, ``now()``
        returns ``ClockUnavailable``.
    """

    #: Polling interval for :py:meth:`watch_timezone` in seconds.
    _poll_interval_seconds: float = 1.0

    def __init__(
        self,
        _override_now: Optional[Callable[[], datetime.datetime]] = None,
    ) -> None:
        self._override_now = _override_now

    # ------------------------------------------------------------------
    # Public API — Requirements 14.1, 14.3, 14.5
    # ------------------------------------------------------------------

    def now(self) -> ClockResult | ClockUnavailable:
        """Return the current date, time, and timezone.

        Never raises — any system error is surfaced as
        :class:`ClockUnavailable` (Requirement 14.5).

        Requirements: 14.1, 14.3, 14.5.
        """
        try:
            dt = self._read_datetime()
            tz_name = self._iana_name(dt)
            return ClockResult(
                date=dt.date(),
                time=dt.time(),
                timezone=tz_name,
            )
        except Exception as exc:  # noqa: BLE001
            return ClockUnavailable(reason=str(exc))

    # ------------------------------------------------------------------
    # Public API — Requirement 14.4
    # ------------------------------------------------------------------

    async def watch_timezone(self) -> AsyncGenerator[str, None]:
        """Async generator that yields a new IANA timezone name on every change.

        Polls the system timezone at :attr:`_poll_interval_seconds`
        (default 1 s).  The generator yields the new timezone name within
        ``_poll_interval_seconds`` of a change, satisfying the ≤ 5 s
        propagation budget required by Requirement 14.4.

        The generator is safely cancellable — it obeys ``GeneratorExit``
        and ``asyncio.CancelledError`` without leaking resources.

        Requirement: 14.4.

        Yields
        ------
        str
            IANA (or POSIX fallback) timezone name whenever it changes.
        """
        last_tz: Optional[str] = None

        try:
            while True:
                current_tz = self._current_tz_name()
                if last_tz is None:
                    # Bootstrap: record the initial timezone without yielding.
                    last_tz = current_tz
                elif current_tz != last_tz:
                    last_tz = current_tz
                    yield current_tz

                await asyncio.sleep(self._poll_interval_seconds)
        except (GeneratorExit, asyncio.CancelledError):
            # Clean shutdown — nothing to release.
            return

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_datetime(self) -> datetime.datetime:
        """Return the current timezone-aware datetime.

        Uses the injected ``_override_now`` callable when set; otherwise
        reads from the real system clock.
        """
        if self._override_now is not None:
            return self._override_now()
        # datetime.now().astimezone() is the most portable way to obtain a
        # timezone-aware datetime carrying the system's local timezone.
        return datetime.datetime.now().astimezone()

    @staticmethod
    def _iana_name(dt: datetime.datetime) -> str:
        """Extract the IANA timezone name from a timezone-aware datetime.

        On most modern systems ``zoneinfo`` provides IANA names.  If the
        tzinfo is a plain ``timezone`` (e.g. UTC offset), we fall back to
        the abbreviated name reported by ``strftime("%Z")``.
        """
        tzinfo = dt.tzinfo
        if isinstance(tzinfo, zoneinfo.ZoneInfo):
            return tzinfo.key  # e.g. "Asia/Kolkata"
        if tzinfo is not None:
            # strftime %Z gives POSIX abbreviation ("IST", "UTC", "PST+8" …)
            return dt.strftime("%Z") or str(tzinfo)
        return "UTC"

    def _current_tz_name(self) -> str:
        """Return the current system timezone name without raising.

        Used by the watcher loop; returns "unknown" on error so the loop
        can continue safely.
        """
        try:
            dt = self._read_datetime()
            return self._iana_name(dt)
        except Exception:  # noqa: BLE001
            return "unknown"
