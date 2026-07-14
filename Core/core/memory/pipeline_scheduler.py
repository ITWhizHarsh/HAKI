"""
PipelineScheduler — APScheduler-backed background job runner for the HAKI Brain
memory processing pipeline.

Manages two independent processing schedules:
  - Raw file processing (IntervalTrigger, default every 30 minutes)
  - Conversation processing (CronTrigger, default daily at 02:00)

Entirely separate from core/scheduler/Scheduler, which manages user task reminders.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.memory.haki_brain import HAKIBrain

logger = logging.getLogger(__name__)


class PipelineScheduler:
    """
    Manages two independent APScheduler background jobs:
      - raw_job         : calls HAKIBrain.ingest_pending() every Raw_Interval minutes
      - conversation_job: calls HAKIBrain.process_pending_conversations() at Conversation_Run_Time

    Uses AsyncIOScheduler so jobs run in the existing asyncio event loop without
    spawning additional threads.

    Parameters
    ----------
    haki_brain            : HAKIBrain instance
    raw_interval_minutes  : int, default 30
    conv_run_time         : str "HH:MM" 24h, default "02:00"
    """

    def __init__(
        self,
        haki_brain: "HAKIBrain",
        raw_interval_minutes: int = 30,
        conv_run_time: str = "02:00",
    ) -> None:
        self._brain = haki_brain
        self._raw_interval = raw_interval_minutes
        self._conv_time = conv_run_time
        self._scheduler = None
        self._raw_running: bool = False
        self._conv_running: bool = False

    def start(self) -> None:
        """Start the APScheduler with both jobs. Call once at service startup."""
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.interval import IntervalTrigger
            from apscheduler.triggers.cron import CronTrigger
        except ImportError:
            logger.error(
                "[PipelineScheduler] pip install apscheduler — scheduler not started"
            )
            return

        hour, minute = self._parse_time(self._conv_time)
        self._scheduler = AsyncIOScheduler()

        self._scheduler.add_job(
            self._raw_job,
            trigger=IntervalTrigger(minutes=self._raw_interval),
            id="haki_raw_pipeline",
            name="HAKIBrain raw/ processing",
            max_instances=1,
            coalesce=True,
        )

        self._scheduler.add_job(
            self._conv_job,
            trigger=CronTrigger(hour=hour, minute=minute),
            id="haki_conv_pipeline",
            name="HAKIBrain conversations/ processing",
            max_instances=1,
            coalesce=True,
        )

        self._scheduler.start()
        logger.info(
            "[PipelineScheduler] Started — raw every %d min, conv at %s",
            self._raw_interval,
            self._conv_time,
        )

    def stop(self) -> None:
        """Shut down the AsyncIOScheduler gracefully."""
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            logger.info("[PipelineScheduler] Scheduler shut down")

    # ------------------------------------------------------------------
    # Job wrappers with in-flight guards
    # ------------------------------------------------------------------

    async def _raw_job(self) -> None:
        """Job wrapper for raw file processing with in-flight guard.

        If a previous run of this job is still active, the new trigger is
        skipped immediately (Req 9.4, 9.6).
        """
        if self._raw_running:
            logger.info("[PipelineScheduler] Raw job already running — skipping")
            return
        self._raw_running = True
        try:
            results = await self._brain.ingest_pending()
            logger.info(
                "[PipelineScheduler] Raw run complete: %d file(s)", len(results)
            )
        except Exception as exc:
            logger.error("[PipelineScheduler] Raw job failed: %s", exc)
        finally:
            self._raw_running = False

    async def _conv_job(self) -> None:
        """Job wrapper for conversation processing with in-flight guard.

        If a previous run of this job is still active, the new trigger is
        skipped immediately (Req 9.4, 9.6).
        """
        if self._conv_running:
            logger.info(
                "[PipelineScheduler] Conversation job already running — skipping"
            )
            return
        self._conv_running = True
        try:
            await self._brain.process_pending_conversations()
        except Exception as exc:
            logger.error("[PipelineScheduler] Conversation job failed: %s", exc)
        finally:
            self._conv_running = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_time(time_str: str) -> tuple[int, int]:
        """Parse 'HH:MM' into (hour, minute). Falls back to (2, 0) on error."""
        try:
            parts = time_str.split(":")
            if len(parts) != 2:
                raise ValueError("Expected HH:MM format")
            h, m = int(parts[0]), int(parts[1])
            if not (0 <= h <= 23 and 0 <= m <= 59):
                raise ValueError(f"Time out of range: {h}:{m}")
            return h, m
        except Exception:
            logger.error(
                "[PipelineScheduler] Invalid HAKI_PIPELINE_CONV_RUN_TIME '%s' — using 02:00",
                time_str,
            )
            return 2, 0
