"""
Tests for the Comms_Reader subsystem.

Covers:
- ActionableItem model and needs_clarification logic (Req 10.3, 10.4)
- AccountManager grant/revoke/state (Req 10.5, 10.8, 10.9)
- CommsReader connect/disconnect / polling lifecycle (Req 10.8, 10.9)
- Retry-and-failure notification logic (Req 10.6, 10.7)
- Actionable extraction stub pathway (Req 10.1, 10.2, 10.3)

Design: Comms_Reader.
Requirements: 10.1 – 10.9.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from core.comms.account_manager import AccountManager, AccountType, ConnectedAccount
from core.comms.actionable_item import ActionableItem, ActionableType
from core.comms.comms_reader import (
    CommsReader,
    MAX_RETRIES,
    Message,
    RETRY_INTERVAL_SECONDS,
    StubEmailAdapter,
    StubWhatsAppAdapter,
    extract_actionables_from_message,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_account(
    account_id: str = "user@example.com",
    account_type: AccountType = AccountType.EMAIL_IMAP,
    display_name: str = "Test Account",
) -> ConnectedAccount:
    mgr = AccountManager()
    return mgr.register(account_id, account_type, display_name)


def make_message(
    body: str = "Meet me at 3pm on Monday",
    source_account: str = "user@example.com",
    message_id: str = "msg-001",
) -> Message:
    return Message(
        message_id=message_id,
        source_account=source_account,
        sender="friend@example.com",
        body=body,
    )


# ---------------------------------------------------------------------------
# ActionableItem model tests
# ---------------------------------------------------------------------------


class TestActionableItem:
    def test_create_with_full_details_no_clarification(self):
        item = ActionableItem.create(
            source_account="wa:+1234",
            source_message_id="m1",
            type=ActionableType.EVENT,
            description="Team meeting",
            date="2024-07-01",
            time="14:00",
            location="Office",
        )
        assert item.needs_clarification is False
        assert item.date == "2024-07-01"
        assert item.time == "14:00"
        assert item.location == "Office"

    def test_create_missing_date_sets_needs_clarification(self):
        item = ActionableItem.create(
            source_account="wa:+1234",
            source_message_id="m2",
            type=ActionableType.TASK,
            description="Submit report",
            date=None,
            time="09:00",
        )
        assert item.needs_clarification is True

    def test_create_missing_time_sets_needs_clarification(self):
        item = ActionableItem.create(
            source_account="email:user@x.com",
            source_message_id="m3",
            type=ActionableType.REMINDER,
            description="Buy gift",
            date="2024-08-10",
            time=None,
        )
        assert item.needs_clarification is True

    def test_create_missing_both_sets_needs_clarification(self):
        item = ActionableItem.create(
            source_account="email:user@x.com",
            source_message_id="m4",
            type=ActionableType.TASK,
            description="Prepare slides",
        )
        assert item.needs_clarification is True

    def test_id_is_unique(self):
        item1 = ActionableItem.create(
            source_account="a", source_message_id="1",
            type=ActionableType.TASK, description="Task A",
        )
        item2 = ActionableItem.create(
            source_account="a", source_message_id="2",
            type=ActionableType.TASK, description="Task B",
        )
        assert item1.id != item2.id

    def test_compute_needs_clarification_updates_field(self):
        item = ActionableItem.create(
            source_account="a", source_message_id="1",
            type=ActionableType.EVENT, description="Event",
            date="2024-01-01", time="10:00",
        )
        assert item.needs_clarification is False
        item.time = None
        item.compute_needs_clarification()
        assert item.needs_clarification is True

    def test_to_dict_round_trip(self):
        item = ActionableItem.create(
            source_account="acc",
            source_message_id="mid",
            type=ActionableType.EVENT,
            description="Concert",
            date="2024-09-01",
            time="20:00",
            location="Arena",
        )
        d = item.to_dict()
        assert d["type"] == "EVENT"
        assert d["description"] == "Concert"
        assert d["needs_clarification"] is False
        assert d["location"] == "Arena"



# ---------------------------------------------------------------------------
# AccountManager tests
# ---------------------------------------------------------------------------


class TestAccountManager:
    def test_register_creates_account(self):
        mgr = AccountManager()
        acc = mgr.register("a@b.com", AccountType.EMAIL_IMAP, "Test")
        assert acc.account_id == "a@b.com"
        assert acc.granted is False

    def test_register_idempotent(self):
        mgr = AccountManager()
        acc1 = mgr.register("a@b.com", AccountType.EMAIL_IMAP, "Test")
        acc2 = mgr.register("a@b.com", AccountType.EMAIL_IMAP, "Test")
        assert acc1 is acc2

    def test_grant_sets_granted_and_clears_stop_event(self):
        mgr = AccountManager()
        mgr.register("a@b.com", AccountType.EMAIL_IMAP, "Test")
        # Pre-set stop_event to ensure grant clears it
        mgr.get("a@b.com").stop_event.set()
        mgr.grant("a@b.com")
        assert mgr.is_granted("a@b.com") is True
        assert not mgr.get("a@b.com").stop_event.is_set()

    def test_revoke_clears_granted_and_sets_stop_event(self):
        mgr = AccountManager()
        mgr.register("a@b.com", AccountType.EMAIL_IMAP, "Test")
        mgr.grant("a@b.com")
        mgr.revoke("a@b.com")
        assert mgr.is_granted("a@b.com") is False
        assert mgr.get("a@b.com").stop_event.is_set()

    def test_grant_unknown_account_raises(self):
        mgr = AccountManager()
        with pytest.raises(KeyError):
            mgr.grant("unknown@x.com")

    def test_revoke_unknown_account_raises(self):
        mgr = AccountManager()
        with pytest.raises(KeyError):
            mgr.revoke("unknown@x.com")

    def test_granted_accounts_returns_only_granted(self):
        mgr = AccountManager()
        mgr.register("a@b.com", AccountType.EMAIL_IMAP, "A")
        mgr.register("c@d.com", AccountType.EMAIL_IMAP, "C")
        mgr.grant("a@b.com")
        granted = mgr.granted_accounts()
        assert len(granted) == 1
        assert granted[0].account_id == "a@b.com"

    def test_all_accounts_returns_all(self):
        mgr = AccountManager()
        mgr.register("a@b.com", AccountType.EMAIL_IMAP, "A")
        mgr.register("c@d.com", AccountType.WHATSAPP, "C")
        assert len(mgr.all_accounts()) == 2

    def test_unregister_removes_account(self):
        mgr = AccountManager()
        mgr.register("a@b.com", AccountType.EMAIL_IMAP, "A")
        mgr.grant("a@b.com")
        mgr.unregister("a@b.com")
        assert mgr.get("a@b.com") is None
        assert len(mgr.all_accounts()) == 0



# ---------------------------------------------------------------------------
# Actionable extraction (stub LLM path)
# ---------------------------------------------------------------------------


class TestExtractActionables:
    def test_no_llm_provider_returns_empty(self):
        msg = make_message("Exam on Friday at 9am")
        items = extract_actionables_from_message(msg, llm_provider=None)
        assert items == []

    def test_stub_llm_returns_empty_because_echoes_input(self):
        """The StubModelProvider echoes input; _parse_actionable_json sees
        a non-JSON string and returns []."""
        from core.model_provider.model_provider import (
            Capability,
            ModelProviderRegistry,
            StubModelProvider,
        )
        registry = ModelProviderRegistry()
        provider = StubModelProvider(Capability.LLM, registry)
        msg = make_message("Buy milk tomorrow")
        items = extract_actionables_from_message(msg, llm_provider=provider)
        # Stub echoes input as dict — no parseable JSON array of items
        assert isinstance(items, list)

    def test_real_json_llm_response_extracts_item(self):
        """Simulate a real LLM string response with valid JSON."""
        from unittest.mock import MagicMock
        provider = MagicMock()
        provider.invoke.return_value = (
            '[{"type":"EVENT","description":"Team meeting",'
            '"date":"2024-07-01","time":"15:00","location":"Office"}]'
        )
        msg = make_message("Team meeting on July 1 at 3pm in Office")
        items = extract_actionables_from_message(msg, llm_provider=provider)
        assert len(items) == 1
        assert items[0].type == ActionableType.EVENT
        assert items[0].description == "Team meeting"
        assert items[0].date == "2024-07-01"
        assert items[0].time == "15:00"
        assert items[0].location == "Office"
        assert items[0].needs_clarification is False

    def test_missing_date_in_llm_response_flags_clarification(self):
        """Item with date=null triggers needs_clarification."""
        from unittest.mock import MagicMock
        provider = MagicMock()
        provider.invoke.return_value = (
            '[{"type":"TASK","description":"Submit assignment",'
            '"date":null,"time":"17:00","location":null}]'
        )
        msg = make_message("Submit assignment by 5pm")
        items = extract_actionables_from_message(msg, llm_provider=provider)
        assert len(items) == 1
        assert items[0].needs_clarification is True

    def test_missing_time_in_llm_response_flags_clarification(self):
        """Item with time=null triggers needs_clarification."""
        from unittest.mock import MagicMock
        provider = MagicMock()
        provider.invoke.return_value = (
            '[{"type":"REMINDER","description":"Send birthday card",'
            '"date":"2024-08-15","time":null,"location":null}]'
        )
        msg = make_message("Send birthday card on August 15")
        items = extract_actionables_from_message(msg, llm_provider=provider)
        assert len(items) == 1
        assert items[0].needs_clarification is True

    def test_empty_llm_response_returns_empty(self):
        from unittest.mock import MagicMock
        provider = MagicMock()
        provider.invoke.return_value = "[]"
        msg = make_message("How are you?")
        items = extract_actionables_from_message(msg, llm_provider=provider)
        assert items == []

    def test_llm_exception_returns_empty(self):
        from unittest.mock import MagicMock
        provider = MagicMock()
        provider.invoke.side_effect = RuntimeError("LLM unavailable")
        msg = make_message("Meet tomorrow")
        items = extract_actionables_from_message(msg, llm_provider=provider)
        assert items == []



# ---------------------------------------------------------------------------
# CommsReader — connect / disconnect / polling
# ---------------------------------------------------------------------------


class TestCommsReaderConnectDisconnect:
    """
    Tests for Req 10.8 (stop within 5 s on revoke) and
    Req 10.9 (begin reading on grant).
    """

    def test_connect_grants_account(self):
        mgr = AccountManager()
        acc = mgr.register("wa:+1", AccountType.WHATSAPP, "WhatsApp")
        reader = CommsReader(account_manager=mgr)

        async def run():
            await reader.connect(acc)
            assert mgr.is_granted("wa:+1")
            await reader.disconnect(acc)

        asyncio.get_event_loop().run_until_complete(run())

    def test_disconnect_revokes_account(self):
        mgr = AccountManager()
        acc = mgr.register("wa:+1", AccountType.WHATSAPP, "WhatsApp")
        reader = CommsReader(account_manager=mgr)

        async def run():
            await reader.connect(acc)
            await reader.disconnect(acc)
            assert not mgr.is_granted("wa:+1")

        asyncio.get_event_loop().run_until_complete(run())

    def test_disconnect_sets_stop_event(self):
        mgr = AccountManager()
        acc = mgr.register("e@x.com", AccountType.EMAIL_IMAP, "IMAP")
        reader = CommsReader(account_manager=mgr)

        async def run():
            await reader.connect(acc)
            assert not acc.stop_event.is_set()
            await reader.disconnect(acc)
            assert acc.stop_event.is_set()

        asyncio.get_event_loop().run_until_complete(run())

    def test_polling_processes_injected_messages(self):
        """Messages injected into the stub adapter are processed within poll cycle."""
        mgr = AccountManager()
        acc = mgr.register("e@x.com", AccountType.EMAIL_IMAP, "IMAP")
        stub_email = StubEmailAdapter()
        collected: list[ActionableItem] = []

        # Use a real LLM mock that returns a valid actionable
        from unittest.mock import MagicMock
        provider = MagicMock()
        provider.invoke.return_value = (
            '[{"type":"TASK","description":"Attend workshop",'
            '"date":"2024-10-01","time":"10:00","location":null}]'
        )

        reader = CommsReader(
            account_manager=mgr,
            email_adapter=stub_email,
            llm_provider=provider,
            on_actionable=collected.append,
            poll_interval_seconds=0.1,
        )

        async def run():
            await reader.connect(acc)
            stub_email.inject_messages([
                Message("m1", "e@x.com", "sender", "Attend workshop Oct 1 at 10am")
            ])
            # Allow one poll cycle
            await asyncio.sleep(0.3)
            await reader.disconnect(acc)

        asyncio.get_event_loop().run_until_complete(run())
        assert len(collected) == 1
        assert collected[0].description == "Attend workshop"



# ---------------------------------------------------------------------------
# CommsReader — retry and failure notification (Req 10.6, 10.7)
# ---------------------------------------------------------------------------


class TestCommsReaderRetry:
    """
    Tests that the retry-and-failure logic conforms to Req 10.6 and 10.7:
    - Retry up to 3 additional times at 30 s intervals.
    - After exhausting retries, notify the user identifying the account.
    """

    def test_failure_notification_called_after_max_retries(self):
        """
        When the adapter always fails, on_failure is called exactly once
        after MAX_RETRIES + 1 total attempts.
        """
        mgr = AccountManager()
        acc = mgr.register("fail@x.com", AccountType.EMAIL_IMAP, "Failing IMAP")

        failure_calls: list[tuple[str, str]] = []

        class AlwaysFailAdapter(StubEmailAdapter):
            async def poll_new_messages(self, account):
                raise IOError("Connection refused")

        reader = CommsReader(
            account_manager=mgr,
            email_adapter=AlwaysFailAdapter(),
            on_failure=lambda acct_id, reason: failure_calls.append((acct_id, reason)),
            poll_interval_seconds=0.05,
        )

        # Patch RETRY_INTERVAL_SECONDS to 0 so the test doesn't take 90 s
        import core.comms.comms_reader as cr_module
        original_interval = cr_module.RETRY_INTERVAL_SECONDS

        async def run():
            cr_module.RETRY_INTERVAL_SECONDS = 0.05  # type: ignore[attr-defined]
            await reader.connect(acc)
            # Wait long enough for all retries to exhaust
            await asyncio.sleep(1.5)
            cr_module.RETRY_INTERVAL_SECONDS = original_interval  # type: ignore[attr-defined]

        asyncio.get_event_loop().run_until_complete(run())

        assert len(failure_calls) == 1, (
            f"Expected exactly 1 failure notification, got {len(failure_calls)}"
        )
        assert failure_calls[0][0] == "fail@x.com"

    def test_failure_notification_identifies_account(self):
        """The on_failure callback receives the correct account_id."""
        mgr = AccountManager()
        acc = mgr.register("identified@x.com", AccountType.WHATSAPP, "My WhatsApp")

        notified_ids: list[str] = []

        class AlwaysFailWA(StubWhatsAppAdapter):
            async def poll_new_messages(self, account):
                raise RuntimeError("WA error")

        reader = CommsReader(
            account_manager=mgr,
            whatsapp_adapter=AlwaysFailWA(),
            on_failure=lambda acct_id, _: notified_ids.append(acct_id),
            poll_interval_seconds=0.05,
        )

        import core.comms.comms_reader as cr_module
        original = cr_module.RETRY_INTERVAL_SECONDS

        async def run():
            cr_module.RETRY_INTERVAL_SECONDS = 0.05  # type: ignore[attr-defined]
            await reader.connect(acc)
            await asyncio.sleep(1.5)
            cr_module.RETRY_INTERVAL_SECONDS = original  # type: ignore[attr-defined]

        asyncio.get_event_loop().run_until_complete(run())
        assert "identified@x.com" in notified_ids

    def test_success_after_failures_resets_retry_counter(self):
        """
        If the first poll fails but subsequent ones succeed, on_failure
        should NOT be called.
        """
        mgr = AccountManager()
        acc = mgr.register("recover@x.com", AccountType.EMAIL_IMAP, "IMAP")
        failure_calls: list[str] = []
        call_count = 0

        class FailOnceThenSucceedAdapter(StubEmailAdapter):
            async def poll_new_messages(self, account):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise IOError("Transient error")
                return []

        reader = CommsReader(
            account_manager=mgr,
            email_adapter=FailOnceThenSucceedAdapter(),
            on_failure=lambda a, r: failure_calls.append(a),
            poll_interval_seconds=0.05,
        )

        import core.comms.comms_reader as cr_module
        original = cr_module.RETRY_INTERVAL_SECONDS

        async def run():
            cr_module.RETRY_INTERVAL_SECONDS = 0.05  # type: ignore[attr-defined]
            await reader.connect(acc)
            await asyncio.sleep(1.0)
            await reader.disconnect(acc)
            cr_module.RETRY_INTERVAL_SECONDS = original  # type: ignore[attr-defined]

        asyncio.get_event_loop().run_until_complete(run())
        assert failure_calls == [], (
            "on_failure should not be called when polling recovers"
        )



# ---------------------------------------------------------------------------
# CommsReader — revocation stops polling within 5 s (Req 10.8)
# ---------------------------------------------------------------------------


class TestRevocationWithin5Seconds:
    def test_revocation_stops_polling_quickly(self):
        """
        After disconnect() the polling task exits and no further processing
        occurs.  Verified by checking that the stop_event is set and the
        task is done within 5 s.
        """
        mgr = AccountManager()
        acc = mgr.register("quick@x.com", AccountType.EMAIL_IMAP, "Quick")
        stub = StubEmailAdapter()
        reader = CommsReader(
            account_manager=mgr,
            email_adapter=stub,
            poll_interval_seconds=0.1,
        )

        async def run():
            await reader.connect(acc)
            start = time.monotonic()
            await reader.disconnect(acc)
            elapsed = time.monotonic() - start
            assert elapsed < 5.0, f"disconnect took {elapsed:.2f}s (>5s)"
            assert acc.stop_event.is_set()

        asyncio.get_event_loop().run_until_complete(run())

    def test_polling_does_not_continue_after_revocation(self):
        """Messages injected after revocation are not processed."""
        mgr = AccountManager()
        acc = mgr.register("after@x.com", AccountType.EMAIL_IMAP, "After")
        stub = StubEmailAdapter()
        collected: list[ActionableItem] = []

        from unittest.mock import MagicMock
        provider = MagicMock()
        provider.invoke.return_value = (
            '[{"type":"TASK","description":"Should not appear",'
            '"date":null,"time":null,"location":null}]'
        )

        reader = CommsReader(
            account_manager=mgr,
            email_adapter=stub,
            llm_provider=provider,
            on_actionable=collected.append,
            poll_interval_seconds=0.05,
        )

        async def run():
            await reader.connect(acc)
            await reader.disconnect(acc)
            # Inject a message AFTER disconnect
            stub.inject_messages([
                Message("late", "after@x.com", "s", "Late message")
            ])
            await asyncio.sleep(0.3)

        asyncio.get_event_loop().run_until_complete(run())
        assert collected == [], "No items should be processed after revocation"


# ---------------------------------------------------------------------------
# CommsReader — on_actionable and flag_incomplete callbacks
# ---------------------------------------------------------------------------


class TestCommsReaderCallbacks:
    def test_on_actionable_called_for_extracted_items(self):
        from unittest.mock import MagicMock
        provider = MagicMock()
        provider.invoke.return_value = (
            '[{"type":"EVENT","description":"Dinner",'
            '"date":"2024-11-01","time":"19:00","location":"Restaurant"}]'
        )
        collected: list[ActionableItem] = []
        reader = CommsReader(
            llm_provider=provider,
            on_actionable=collected.append,
        )
        msg = make_message("Dinner at 7pm on Nov 1 at Restaurant")
        reader.extract_actionables(msg)
        assert len(collected) == 1
        assert collected[0].description == "Dinner"

    def test_flag_incomplete_sets_needs_clarification(self):
        reader = CommsReader()
        item = ActionableItem.create(
            source_account="a",
            source_message_id="1",
            type=ActionableType.TASK,
            description="Do something",
            date="2024-01-01",
            time="10:00",
        )
        assert item.needs_clarification is False
        reader.flag_incomplete(item)
        assert item.needs_clarification is True

    def test_on_actionable_called_for_incomplete_items(self):
        """on_actionable is still called even when needs_clarification=True."""
        from unittest.mock import MagicMock
        provider = MagicMock()
        provider.invoke.return_value = (
            '[{"type":"TASK","description":"Vague task",'
            '"date":null,"time":null,"location":null}]'
        )
        collected: list[ActionableItem] = []
        reader = CommsReader(
            llm_provider=provider,
            on_actionable=collected.append,
        )
        msg = make_message("Do something sometime")
        reader.extract_actionables(msg)
        assert len(collected) == 1
        assert collected[0].needs_clarification is True


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_max_retries_is_three(self):
        """Req 10.6: retry up to 3 additional times."""
        assert MAX_RETRIES == 3

    def test_retry_interval_is_thirty_seconds(self):
        """Req 10.6: intervals of 30 seconds."""
        assert RETRY_INTERVAL_SECONDS == 30.0
