"""
Tests for the Scheduler subsystem.

Covers:
- Task creation with severity assignment (Req 12.1)
- Indeterminate severity defaults + notification (Req 12.11)
- Default reminder offsets per severity (Req 12.2, 12.3, 12.4)
- Custom valid offsets override defaults (Req 12.7)
- Invalid/empty custom offsets fall back to defaults (Req 12.8)
- Elapsed-window reminders are set to fire immediately (Req 12.10)
- Dual-channel issuance (Req 12.6)
- Per-reminder failure isolation — single channel failure (Req 12.9)
- Per-reminder failure isolation — full reminder failure (Req 12.9)
- Birthday day-of prompt is added (Req 12.5)
- Birthday day-of prompt is distinct from offset reminders (Req 12.5)
- ReminderPolicy.is_valid() guards invalid custom policies (Req 12.8)

Design: Scheduler, Data Models (Task & Reminder).
Requirements: 12.1 – 12.11.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, call

import pytest

from core.clock.clock import Clock
from core.scheduler.models import (
    DEFAULT_POLICIES,
    Prereq,
    Reminder,
    ReminderChannel,
    ReminderPolicy,
    ReminderState,
    Severity,
    Task,
    TaskSource,
    TaskStatus,
)
from core.scheduler.scheduler import Scheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TZ = datetime.timezone.utc

def _dt(year=2024, month=6, day=1, hour=0, minute=0, second=0) -> datetime.datetime:
    """Return a UTC-aware datetime for use in tests."""
    return datetime.datetime(year, month, day, hour, minute, second, tzinfo=_TZ)


def _make_clock(now: datetime.datetime) -> Clock:
    """Return a Clock whose now() returns a fixed datetime."""
    return Clock(_override_now=lambda: now)


def _make_scheduler(
    now: datetime.datetime = _dt(),
    voice_adapter=None,
    notification_adapter=None,
    on_failure=None,
    on_notification=None,
) -> Scheduler:
    clock = _make_clock(now)
    return Scheduler(
        clock=clock,
        voice_adapter=voice_adapter,
        notification_adapter=notification_adapter,
        on_failure=on_failure,
        on_notification=on_notification,
    )


# ---------------------------------------------------------------------------
# 29.1 — Task creation with severity assignment (Req 12.1)
# ---------------------------------------------------------------------------


class TestTaskCreation:
    """Requirements: 12.1, 12.11."""

    def test_explicit_severity_is_preserved(self):
        """12.1 — every task has a severity."""
        sched = _make_scheduler()
        for sev in Severity:
            task = sched.create_task("T", severity=sev)
            assert task.severity == sev

    def test_task_has_stable_unique_id(self):
        sched = _make_scheduler()
        t1 = sched.create_task("A", severity=Severity.EXAM)
        t2 = sched.create_task("B", severity=Severity.EXAM)
        assert t1.id != t2.id

    def test_task_default_status_is_upcoming(self):
        sched = _make_scheduler()
        task = sched.create_task("T", severity=Severity.ASSIGNMENT)
        assert task.status == TaskStatus.UPCOMING

    def test_task_stores_title_and_description(self):
        sched = _make_scheduler()
        task = sched.create_task("My task", description="Details", severity=Severity.DEFAULT)
        assert task.title == "My task"
        assert task.description == "Details"

    def test_task_stores_due_date(self):
        sched = _make_scheduler()
        due = _dt(2024, 12, 31)
        task = sched.create_task("T", due_date=due, severity=Severity.EXAM)
        assert task.due_date == due

    def test_task_stores_prerequisites(self):
        sched = _make_scheduler()
        prereqs = [Prereq(title="Read chapter 1"), Prereq(title="Read chapter 2")]
        task = sched.create_task("T", severity=Severity.ASSIGNMENT, prerequisites=prereqs)
        assert len(task.prerequisites) == 2

    def test_task_stores_source(self):
        sched = _make_scheduler()
        task = sched.create_task("T", severity=Severity.DEFAULT, source=TaskSource.COMMS)
        assert task.source == TaskSource.COMMS

    # Indeterminate severity → default + notification (12.11)

    def test_none_severity_becomes_default(self):
        """12.11 — indeterminate severity assigns DEFAULT."""
        sched = _make_scheduler()
        task = sched.create_task("T", severity=None)
        assert task.severity == Severity.DEFAULT

    def test_none_severity_sets_severity_was_defaulted(self):
        """12.11 — flag indicates default was applied."""
        sched = _make_scheduler()
        task = sched.create_task("T", severity=None)
        assert task.severity_was_defaulted is True

    def test_explicit_severity_does_not_set_severity_was_defaulted(self):
        """12.11 — explicit severity does not set the flag."""
        sched = _make_scheduler()
        task = sched.create_task("T", severity=Severity.EXAM)
        assert task.severity_was_defaulted is False

    def test_none_severity_calls_on_notification(self):
        """12.11 — user is notified when default is applied."""
        messages: list[str] = []
        sched = _make_scheduler(on_notification=messages.append)
        sched.create_task("T", severity=None)
        assert len(messages) == 1
        assert "severity" in messages[0].lower()

    def test_explicit_severity_does_not_call_on_notification(self):
        """12.11 — no notification when severity is known."""
        messages: list[str] = []
        sched = _make_scheduler(on_notification=messages.append)
        sched.create_task("T", severity=Severity.BIRTHDAY)
        assert messages == []

    def test_task_to_dict_contains_severity(self):
        sched = _make_scheduler()
        task = sched.create_task("T", severity=Severity.EXAM)
        d = task.to_dict()
        assert d["severity"] == "EXAM"

    def test_task_to_dict_contains_severity_was_defaulted(self):
        sched = _make_scheduler()
        task = sched.create_task("T", severity=None)
        d = task.to_dict()
        assert d["severity_was_defaulted"] is True


# ---------------------------------------------------------------------------
# 29.2 — Reminder-offset computation using the Clock (Req 12.2–12.4, 12.7–12.10)
# ---------------------------------------------------------------------------


class TestReminderOffsets:
    """Requirements: 12.2, 12.3, 12.4, 12.7, 12.8, 12.10."""

    # ---- Default offset correctness ----

    def test_assignment_default_offsets(self):
        """12.2 — ASSIGNMENT: −7d and −3d before due date."""
        due = _dt(2024, 7, 1)
        sched = _make_scheduler(now=_dt(2024, 6, 1))  # well before both offsets
        task = sched.create_task("Midterm Essay", due_date=due, severity=Severity.ASSIGNMENT)
        reminders = sched.compute_reminders(task)
        fire_times = {r.fire_at for r in reminders if not r.is_birthday_day_of}
        expected = {due + datetime.timedelta(days=-7), due + datetime.timedelta(days=-3)}
        assert fire_times == expected

    def test_exam_default_offsets(self):
        """12.2 — EXAM: −7d and −3d before due date."""
        due = _dt(2024, 8, 15)
        sched = _make_scheduler(now=_dt(2024, 7, 1))
        task = sched.create_task("Final Exam", due_date=due, severity=Severity.EXAM)
        reminders = sched.compute_reminders(task)
        fire_times = {r.fire_at for r in reminders if not r.is_birthday_day_of}
        expected = {due + datetime.timedelta(days=-7), due + datetime.timedelta(days=-3)}
        assert fire_times == expected

    def test_birthday_default_offsets(self):
        """12.4 — BIRTHDAY: −14d and −1d before birthday."""
        due = _dt(2024, 9, 20)
        sched = _make_scheduler(now=_dt(2024, 8, 1))
        task = sched.create_task("Friend Birthday", due_date=due, severity=Severity.BIRTHDAY)
        reminders = sched.compute_reminders(task)
        non_dayof = [r for r in reminders if not r.is_birthday_day_of]
        fire_times = {r.fire_at for r in non_dayof}
        expected = {due + datetime.timedelta(days=-14), due + datetime.timedelta(days=-1)}
        assert fire_times == expected

    def test_default_severity_one_day_offset(self):
        """12.3 — DEFAULT severity: single −1d offset."""
        due = _dt(2024, 10, 5)
        sched = _make_scheduler(now=_dt(2024, 9, 1))
        task = sched.create_task("Generic", due_date=due, severity=Severity.DEFAULT)
        reminders = sched.compute_reminders(task)
        assert len(reminders) == 1
        assert reminders[0].fire_at == due + datetime.timedelta(days=-1)

    def test_no_due_date_returns_empty_reminders(self):
        """A task without a due date produces no reminders."""
        sched = _make_scheduler()
        task = sched.create_task("T", severity=Severity.EXAM, due_date=None)
        reminders = sched.compute_reminders(task)
        assert reminders == []

    # ---- Custom policy override (12.7) ----

    def test_valid_custom_policy_overrides_default(self):
        """12.7 — valid custom offsets are used."""
        due = _dt(2024, 12, 1)
        sched = _make_scheduler(now=_dt(2024, 10, 1))
        custom = ReminderPolicy(
            severity=Severity.EXAM,
            offsets=[datetime.timedelta(days=-10), datetime.timedelta(days=-5)],
            custom=True,
        )
        sched._custom_policies[Severity.EXAM] = custom
        task = sched.create_task("Exam", due_date=due, severity=Severity.EXAM)
        reminders = sched.compute_reminders(task)
        fire_times = {r.fire_at for r in reminders}
        expected = {due + datetime.timedelta(days=-10), due + datetime.timedelta(days=-5)}
        assert fire_times == expected

    # ---- Invalid custom policy falls back to default (12.8) ----

    def test_empty_custom_offsets_fall_back_to_default(self):
        """12.8 — empty custom policy → fall back to default."""
        due = _dt(2024, 12, 1)
        sched = _make_scheduler(now=_dt(2024, 10, 1))
        invalid_custom = ReminderPolicy(
            severity=Severity.ASSIGNMENT,
            offsets=[],  # invalid — empty
            custom=True,
        )
        sched._custom_policies[Severity.ASSIGNMENT] = invalid_custom
        task = sched.create_task("Essay", due_date=due, severity=Severity.ASSIGNMENT)
        reminders = sched.compute_reminders(task)
        fire_times = {r.fire_at for r in reminders}
        expected = {due + datetime.timedelta(days=-7), due + datetime.timedelta(days=-3)}
        assert fire_times == expected

    def test_reminder_policy_is_valid_false_for_empty_offsets(self):
        """12.8 — ReminderPolicy.is_valid() is False when offsets is empty."""
        policy = ReminderPolicy(severity=Severity.EXAM, offsets=[])
        assert policy.is_valid() is False

    def test_reminder_policy_is_valid_true_for_non_empty_offsets(self):
        policy = ReminderPolicy(
            severity=Severity.EXAM,
            offsets=[datetime.timedelta(days=-3)],
        )
        assert policy.is_valid() is True

    # ---- Elapsed-window: fire immediately (12.10) ----

    def test_elapsed_offset_fires_at_now(self):
        """12.10 — reminder whose computed fire_at < now is set to now."""
        now = _dt(2024, 7, 5)  # already past -7d and -3d for a due_date of July 1
        due = _dt(2024, 7, 1)  # due in the past relative to now
        sched = _make_scheduler(now=now)
        task = sched.create_task("Late Essay", due_date=due, severity=Severity.ASSIGNMENT)
        reminders = sched.compute_reminders(task)
        non_dayof = [r for r in reminders if not r.is_birthday_day_of]
        for reminder in non_dayof:
            # All computed fire times are in the past → all should be set to now
            assert reminder.fire_at >= now

    def test_partially_elapsed_only_past_offsets_set_to_now(self):
        """12.10 — only the elapsed offset is set to now; future stays normal."""
        due = _dt(2024, 7, 10)
        # now = July 5: the -7d offset (July 3) is elapsed; -3d (July 7) is future
        now = _dt(2024, 7, 5)
        sched = _make_scheduler(now=now)
        task = sched.create_task("Essay", due_date=due, severity=Severity.EXAM)
        reminders = sched.compute_reminders(task)
        non_dayof = [r for r in reminders if not r.is_birthday_day_of]
        fire_times = sorted(r.fire_at for r in non_dayof)
        # -7d = July 3 elapsed → now (July 5)
        # -3d = July 7 future  → July 7
        assert fire_times[0] == now
        assert fire_times[1] == due + datetime.timedelta(days=-3)

    def test_all_future_offsets_not_set_to_now(self):
        """12.10 — future offsets are unchanged."""
        due = _dt(2024, 12, 1)
        now = _dt(2024, 10, 1)  # both -7d and -3d are still in the future
        sched = _make_scheduler(now=now)
        task = sched.create_task("Final Exam", due_date=due, severity=Severity.EXAM)
        reminders = sched.compute_reminders(task)
        non_dayof = [r for r in reminders if not r.is_birthday_day_of]
        for reminder in non_dayof:
            assert reminder.fire_at != now


# ---------------------------------------------------------------------------
# 29.3 — Dual-channel issuance, failure isolation, birthday day-of (12.5, 12.6, 12.9)
# ---------------------------------------------------------------------------


class TestReminderIssuance:
    """Requirements: 12.5, 12.6, 12.9."""

    # ---- Dual-channel (12.6) ----

    def test_both_channels_called_per_reminder(self):
        """12.6 — each reminder fires on VOICE and NOTIFICATION."""
        voice_calls: list[tuple] = []
        notif_calls: list[tuple] = []

        def voice(task, reminder):
            voice_calls.append((task.id, reminder.id))

        def notif(task, reminder):
            notif_calls.append((task.id, reminder.id))

        due = _dt(2024, 12, 1)
        sched = _make_scheduler(
            now=_dt(2024, 10, 1),
            voice_adapter=voice,
            notification_adapter=notif,
        )
        task = sched.create_task("Exam", due_date=due, severity=Severity.EXAM)
        reminders = sched.compute_reminders(task)
        sched.issue_reminders(task, reminders)

        assert len(voice_calls) == len(reminders)
        assert len(notif_calls) == len(reminders)

    def test_reminder_state_is_fired_on_success(self):
        """12.6 — state becomes FIRED when both channels succeed."""
        due = _dt(2024, 12, 1)
        sched = _make_scheduler(
            now=_dt(2024, 10, 1),
            voice_adapter=lambda t, r: None,
            notification_adapter=lambda t, r: None,
        )
        task = sched.create_task("T", due_date=due, severity=Severity.DEFAULT)
        reminders = sched.compute_reminders(task)
        sched.issue_reminders(task, reminders)
        for r in reminders:
            assert r.state == ReminderState.FIRED

    def test_reminder_channels_contain_voice_and_notification(self):
        """12.6 — every computed reminder includes both channels."""
        due = _dt(2024, 12, 1)
        sched = _make_scheduler(now=_dt(2024, 10, 1))
        task = sched.create_task("T", due_date=due, severity=Severity.EXAM)
        reminders = sched.compute_reminders(task)
        for r in reminders:
            assert ReminderChannel.VOICE in r.channels
            assert ReminderChannel.NOTIFICATION in r.channels

    # ---- Per-reminder failure isolation (12.9) ----

    def test_single_voice_failure_does_not_block_notification(self):
        """12.9 — notification still fires when voice fails."""
        notif_calls: list[str] = []
        failures: list[tuple] = []

        def failing_voice(task, reminder):
            raise RuntimeError("Voice unavailable")

        def notif(task, reminder):
            notif_calls.append(reminder.id)

        due = _dt(2024, 12, 1)
        sched = _make_scheduler(
            now=_dt(2024, 10, 1),
            voice_adapter=failing_voice,
            notification_adapter=notif,
            on_failure=lambda t, r, reason: failures.append((r.id, reason)),
        )
        task = sched.create_task("T", due_date=due, severity=Severity.DEFAULT)
        reminders = sched.compute_reminders(task)
        sched.issue_reminders(task, reminders)

        # Notification should still have been called
        assert len(notif_calls) == len(reminders)
        # Failure callback should have been invoked
        assert len(failures) == len(reminders)
        for _, reason in failures:
            assert "VOICE" in reason

    def test_single_notification_failure_does_not_block_voice(self):
        """12.9 — voice still fires when notification fails."""
        voice_calls: list[str] = []
        failures: list[tuple] = []

        def voice(task, reminder):
            voice_calls.append(reminder.id)

        def failing_notif(task, reminder):
            raise RuntimeError("Notification unavailable")

        due = _dt(2024, 12, 1)
        sched = _make_scheduler(
            now=_dt(2024, 10, 1),
            voice_adapter=voice,
            notification_adapter=failing_notif,
            on_failure=lambda t, r, reason: failures.append((r.id, reason)),
        )
        task = sched.create_task("T", due_date=due, severity=Severity.DEFAULT)
        reminders = sched.compute_reminders(task)
        sched.issue_reminders(task, reminders)

        assert len(voice_calls) == len(reminders)
        assert len(failures) == len(reminders)
        for _, reason in failures:
            assert "NOTIFICATION" in reason

    def test_one_failed_reminder_does_not_block_others(self):
        """12.9 — failure of one reminder still issues the remaining reminders."""
        voice_calls: list[str] = []
        call_count = [0]

        def voice(task, reminder):
            call_count[0] += 1
            if call_count[0] == 1:
                # First reminder fails on voice
                raise RuntimeError("First fail")
            voice_calls.append(reminder.id)

        notif_calls: list[str] = []

        def notif(task, reminder):
            notif_calls.append(reminder.id)

        failures: list[str] = []
        due = _dt(2024, 12, 1)
        sched = _make_scheduler(
            now=_dt(2024, 10, 1),
            voice_adapter=voice,
            notification_adapter=notif,
            on_failure=lambda t, r, reason: failures.append(r.id),
        )
        task = sched.create_task("Exam", due_date=due, severity=Severity.EXAM)
        reminders = sched.compute_reminders(task)
        # EXAM has 2 offset-based reminders
        assert len(reminders) == 2
        sched.issue_reminders(task, reminders)

        # Notification was called for BOTH reminders
        assert len(notif_calls) == 2
        # on_failure called once (for the first reminder's voice channel)
        assert len(failures) == 1

    def test_full_reminder_failure_marks_state_failed(self):
        """12.9 — reminder state is FAILED when both channels fail."""
        failures: list[str] = []

        def fail_all(task, reminder):
            raise RuntimeError("All channels down")

        due = _dt(2024, 12, 1)
        sched = _make_scheduler(
            now=_dt(2024, 10, 1),
            voice_adapter=fail_all,
            notification_adapter=fail_all,
            on_failure=lambda t, r, reason: failures.append(r.id),
        )
        task = sched.create_task("T", due_date=due, severity=Severity.DEFAULT)
        reminders = sched.compute_reminders(task)
        sched.issue_reminders(task, reminders)

        for r in reminders:
            assert r.state == ReminderState.FAILED
        assert len(failures) == len(reminders)

    def test_on_failure_called_with_reminder_and_reason(self):
        """12.9 — on_failure receives the Reminder and a non-empty reason."""
        collected: list[tuple] = []

        def fail_voice(task, reminder):
            raise RuntimeError("Voice down")

        due = _dt(2024, 12, 1)
        sched = _make_scheduler(
            now=_dt(2024, 10, 1),
            voice_adapter=fail_voice,
            notification_adapter=lambda t, r: None,
            on_failure=lambda t, r, reason: collected.append((r, reason)),
        )
        task = sched.create_task("T", due_date=due, severity=Severity.DEFAULT)
        reminders = sched.compute_reminders(task)
        sched.issue_reminders(task, reminders)

        assert len(collected) == 1
        reminder_obj, reason = collected[0]
        assert isinstance(reminder_obj, Reminder)
        assert reason  # non-empty reason

    # ---- Birthday day-of prompt (12.5) ----

    def test_birthday_has_day_of_reminder(self):
        """12.5 — BIRTHDAY tasks include an is_birthday_day_of=True reminder."""
        due = _dt(2024, 9, 20)
        sched = _make_scheduler(now=_dt(2024, 8, 1))
        task = sched.create_task("Friend Birthday", due_date=due, severity=Severity.BIRTHDAY)
        reminders = sched.compute_reminders(task)
        day_of_reminders = [r for r in reminders if r.is_birthday_day_of]
        assert len(day_of_reminders) == 1

    def test_birthday_day_of_reminder_fires_on_birthday(self):
        """12.5 — the day-of reminder fires on the birthday itself."""
        due = _dt(2024, 9, 20)
        sched = _make_scheduler(now=_dt(2024, 8, 1))
        task = sched.create_task("Friend Birthday", due_date=due, severity=Severity.BIRTHDAY)
        reminders = sched.compute_reminders(task)
        day_of = next(r for r in reminders if r.is_birthday_day_of)
        assert day_of.fire_at.date() == due.date()

    def test_birthday_day_of_has_both_channels(self):
        """12.5 + 12.6 — day-of prompt issued on both channels."""
        due = _dt(2024, 9, 20)
        sched = _make_scheduler(now=_dt(2024, 8, 1))
        task = sched.create_task("Birthday", due_date=due, severity=Severity.BIRTHDAY)
        reminders = sched.compute_reminders(task)
        day_of = next(r for r in reminders if r.is_birthday_day_of)
        assert ReminderChannel.VOICE in day_of.channels
        assert ReminderChannel.NOTIFICATION in day_of.channels

    def test_non_birthday_tasks_have_no_day_of_reminder(self):
        """12.5 — non-BIRTHDAY tasks do not have a day-of reminder."""
        due = _dt(2024, 12, 1)
        sched = _make_scheduler(now=_dt(2024, 10, 1))
        for sev in (Severity.ASSIGNMENT, Severity.EXAM, Severity.DEFAULT):
            task = sched.create_task("T", due_date=due, severity=sev)
            reminders = sched.compute_reminders(task)
            day_of = [r for r in reminders if r.is_birthday_day_of]
            assert day_of == [], f"Expected no day-of reminder for {sev}"

    def test_birthday_day_of_elapsed_set_to_now(self):
        """12.10 + 12.5 — elapsed birthday day-of prompt fires immediately."""
        past_due = _dt(2024, 1, 1)  # birthday already past
        now = _dt(2024, 9, 1)
        sched = _make_scheduler(now=now)
        task = sched.create_task("Old Friend", due_date=past_due, severity=Severity.BIRTHDAY)
        reminders = sched.compute_reminders(task)
        day_of = next(r for r in reminders if r.is_birthday_day_of)
        assert day_of.fire_at >= now

    def test_issue_birthday_day_of_calls_both_adapters(self):
        """12.5 + 12.6 — issuing the day-of reminder calls voice + notification."""
        voice_calls: list[str] = []
        notif_calls: list[str] = []
        due = _dt(2024, 9, 20)
        sched = _make_scheduler(
            now=_dt(2024, 8, 1),
            voice_adapter=lambda t, r: voice_calls.append(r.id),
            notification_adapter=lambda t, r: notif_calls.append(r.id),
        )
        task = sched.create_task("Friend Birthday", due_date=due, severity=Severity.BIRTHDAY)
        reminders = sched.compute_reminders(task)
        day_of_reminders = [r for r in reminders if r.is_birthday_day_of]
        sched.issue_reminders(task, day_of_reminders)
        assert len(voice_calls) == 1
        assert len(notif_calls) == 1


# ---------------------------------------------------------------------------
# Data model unit tests
# ---------------------------------------------------------------------------


class TestModels:
    def test_task_to_dict_round_trip(self):
        sched = _make_scheduler()
        task = sched.create_task(
            "Test",
            due_date=_dt(2024, 11, 1),
            severity=Severity.EXAM,
            description="Final exam",
        )
        d = task.to_dict()
        assert d["title"] == "Test"
        assert d["severity"] == "EXAM"
        assert d["status"] == "UPCOMING"
        assert "2024-11-01" in d["due_date"]

    def test_reminder_to_dict_round_trip(self):
        reminder = Reminder(
            task_id="task-1",
            fire_at=_dt(2024, 11, 1),
            channels=[ReminderChannel.VOICE, ReminderChannel.NOTIFICATION],
            state=ReminderState.SCHEDULED,
        )
        d = reminder.to_dict()
        assert d["task_id"] == "task-1"
        assert "VOICE" in d["channels"]
        assert "NOTIFICATION" in d["channels"]
        assert d["state"] == "SCHEDULED"
        assert d["is_birthday_day_of"] is False

    def test_default_policies_cover_all_severities(self):
        """All Severity values have a default policy."""
        for sev in Severity:
            assert sev in DEFAULT_POLICIES, f"Missing default policy for {sev}"

    def test_default_policies_are_all_valid(self):
        for sev, policy in DEFAULT_POLICIES.items():
            assert policy.is_valid(), f"Default policy for {sev} is invalid"

    def test_severity_values(self):
        assert set(Severity) == {
            Severity.ASSIGNMENT,
            Severity.EXAM,
            Severity.BIRTHDAY,
            Severity.DEFAULT,
        }
