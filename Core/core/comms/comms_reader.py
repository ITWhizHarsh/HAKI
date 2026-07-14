"""
Comms_Reader — reads WhatsApp messages and email, extracts Actionable_Items.

This module owns:
- Adapter abstractions (WhatsAppAdapter, EmailAdapter) with stub/mock
  implementations that can be replaced with real integrations.
- Message polling for each connected account.
- LLM-backed ActionableItem extraction within the 60 s budget (Req 10.1, 10.2).
- Clarification flagging for items missing date or time (Req 10.4).
- Bounded retry logic: up to 3 additional attempts at 30 s intervals (Req 10.6).
- User notification after all retries exhausted, naming the failed account
  (Req 10.7).
- Revocation stops polling within 5 s via asyncio.Event (Req 10.8).

Design: Comms_Reader (integration approach).
Requirements: 10.1 – 10.9.

WhatsApp integration notes
--------------------------
There is no official WhatsApp desktop read API.  The real implementation
would use one of two approaches:

1. **Accessibility (AX) tree scraping**: attach to the WhatsApp desktop app
   process, walk its AXUIElement tree to find chat-list rows and message
   bubbles, and read their text content.  This requires the macOS
   Accessibility permission.

2. **CDP / WhatsApp Web**: drive WhatsApp Web via Chrome DevTools Protocol
   (reusing the Arc CDP adapter already in the project).  Log in once, then
   poll ``document.querySelectorAll`` for unread message elements.

The ``WhatsAppAdapter`` base class below defines the contract; the
``StubWhatsAppAdapter`` provides a testable in-memory implementation.
A real adapter would subclass ``WhatsAppAdapter`` and implement the same
``poll_new_messages()`` / ``last_seen_id`` protocol.

Email integration notes
-----------------------
- **Generic IMAP** accounts: use ``imaplib.IMAP4_SSL`` with IDLE (RFC 2177)
  or periodic UID-SEARCH to detect new mail within 60 s.  The
  ``IMAPEmailAdapter`` below provides a concrete (but thin) implementation
  using standard-library ``imaplib``.
- **Gmail** via the Gmail API (OAuth 2.0): use ``history.list`` or Push
  Notifications for near-realtime delivery.  The ``GmailEmailAdapter`` is
  a stub that can be replaced with a real ``google-api-python-client``
  implementation.
"""

from __future__ import annotations

import asyncio
import imaplib
import json
import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence

from .account_manager import AccountManager, AccountType, ConnectedAccount
from .actionable_item import ActionableItem, ActionableType

if TYPE_CHECKING:
    from core.model_provider.model_provider import ModelProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Message model
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """
    A raw incoming message from any channel.

    Attributes
    ----------
    message_id:
        Channel-specific unique identifier (e.g. IMAP UID or WhatsApp msg id).
    source_account:
        Account identifier this message was received on.
    sender:
        Display name or address of the sender.
    body:
        Plain-text body of the message.
    received_at:
        Unix timestamp when the message arrived (used for 60 s budget).
    """

    message_id: str
    source_account: str
    sender: str
    body: str
    received_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# WhatsApp adapter abstraction
# ---------------------------------------------------------------------------


class WhatsAppAdapter(ABC):
    """
    Abstract adapter for reading WhatsApp messages.

    The real implementation would use AX-tree scraping of the WhatsApp
    desktop app or drive WhatsApp Web via the CDP adapter.  Subclasses
    must implement :meth:`poll_new_messages`.
    """

    @abstractmethod
    async def poll_new_messages(self, account: ConnectedAccount) -> List[Message]:
        """
        Return a list of new messages since the last poll.

        Implementations must be non-blocking and return promptly — the
        caller enforces the 60 s budget externally.
        """

    async def connect(self, account: ConnectedAccount) -> None:
        """
        Optional: establish or refresh the connection before polling.

        Called by CommsReader once when an account is granted.
        Default implementation is a no-op.
        """

    async def disconnect(self, account: ConnectedAccount) -> None:
        """
        Optional: clean up connection resources when access is revoked.

        Default implementation is a no-op.
        """


class StubWhatsAppAdapter(WhatsAppAdapter):
    """
    In-memory stub for WhatsApp.

    Tests inject messages via :meth:`inject_messages`.  Real integrations
    replace this class with one that drives the WhatsApp desktop app AX
    tree or WhatsApp Web over CDP.

    Real integration sketch
    -----------------------
    .. code-block:: python

        class AXWhatsAppAdapter(WhatsAppAdapter):
            \"\"\"
            Reads WhatsApp messages by walking the AXUIElement tree of
            the WhatsApp desktop application.

            Prerequisites: macOS Accessibility permission granted.
            \"\"\"
            async def poll_new_messages(self, account):
                import subprocess, json
                # 1. Locate the WhatsApp process.
                # 2. Walk AXUIElement tree to find 'AXRow' elements in
                #    the chat list.
                # 3. For each chat with unread count > 0, open it and
                #    scrape message bubble text from 'AXStaticText'.
                # 4. Return Message objects for each new message.
                ...
    """

    def __init__(self) -> None:
        self._pending: list[Message] = []

    def inject_messages(self, messages: list[Message]) -> None:
        """Enqueue messages to be returned by the next ``poll_new_messages`` call."""
        self._pending.extend(messages)

    async def poll_new_messages(self, account: ConnectedAccount) -> List[Message]:
        """Drain and return all injected messages."""
        msgs = list(self._pending)
        self._pending.clear()
        return msgs


# ---------------------------------------------------------------------------
# Email adapter abstraction
# ---------------------------------------------------------------------------


class EmailAdapter(ABC):
    """
    Abstract adapter for reading email (IMAP or Gmail API).

    The concrete IMAP adapter polls for new UIDs using IMAP IDLE or
    UID-SEARCH.  The Gmail adapter uses the Gmail REST API.
    """

    @abstractmethod
    async def poll_new_messages(self, account: ConnectedAccount) -> List[Message]:
        """Return new messages since the last poll."""

    async def connect(self, account: ConnectedAccount) -> None:
        """Optional: establish / refresh the connection."""

    async def disconnect(self, account: ConnectedAccount) -> None:
        """Optional: release connection resources."""


class IMAPEmailAdapter(EmailAdapter):
    """
    Concrete IMAP adapter using the standard-library ``imaplib``.

    Connects to the IMAP server, selects INBOX, and fetches messages
    newer than the last-seen UID.  Run in an executor to avoid blocking
    the event loop.

    Configuration is passed via the account's ``account_id``, which is
    expected to encode ``"user@host:port"`` or just ``"user@host"``.
    Credentials must be stored externally (e.g. Keychain).

    For production use, replace the ``_fetch_messages`` body with a
    proper async IMAP implementation (e.g. ``aioimaplib``) that supports
    IMAP IDLE for push-like notification within the 60 s budget (Req 10.2).
    """

    def __init__(
        self,
        host: str,
        port: int = 993,
        username: str = "",
        password: str = "",
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._last_uid: dict[str, int] = {}  # account_id -> last UID seen

    async def poll_new_messages(self, account: ConnectedAccount) -> List[Message]:
        """
        Fetch messages with UID > last_seen_uid for the account.

        Runs the blocking IMAP call in a thread executor to keep the
        event loop free.
        """
        loop = asyncio.get_event_loop()
        try:
            messages = await asyncio.wait_for(
                loop.run_in_executor(None, self._fetch_messages, account),
                timeout=30.0,
            )
            return messages
        except asyncio.TimeoutError:
            logger.warning("IMAP fetch timed out for account: %s", account.account_id)
            return []
        except Exception as exc:
            logger.error("IMAP error for %s: %s", account.account_id, exc)
            raise

    def _fetch_messages(self, account: ConnectedAccount) -> List[Message]:
        """
        Blocking IMAP fetch.  Runs in a thread executor.

        Returns new messages since the last-seen UID.
        """
        messages: List[Message] = []
        try:
            conn = imaplib.IMAP4_SSL(self._host, self._port)
            conn.login(self._username, self._password)
            conn.select("INBOX")

            last_uid = self._last_uid.get(account.account_id, 0)
            # Search for UIDs greater than last_seen
            status, data = conn.uid("SEARCH", None, f"UID {last_uid + 1}:*")
            if status != "OK" or not data[0]:
                conn.logout()
                return messages

            uid_list = data[0].split()
            for uid_bytes in uid_list:
                uid_str = uid_bytes.decode()
                uid_int = int(uid_str)
                if uid_int <= last_uid:
                    continue

                # Fetch the message body
                status2, msg_data = conn.uid("FETCH", uid_str, "(BODY[TEXT])")
                if status2 != "OK":
                    continue
                body_raw = msg_data[0][1] if msg_data and msg_data[0] else b""
                body = body_raw.decode(errors="replace") if isinstance(body_raw, bytes) else str(body_raw)

                messages.append(
                    Message(
                        message_id=uid_str,
                        source_account=account.account_id,
                        sender="",
                        body=body,
                    )
                )
                self._last_uid[account.account_id] = max(
                    self._last_uid.get(account.account_id, 0), uid_int
                )

            conn.logout()
        except imaplib.IMAP4.error as exc:
            logger.error("IMAP4 error for %s: %s", account.account_id, exc)
            raise
        return messages


class StubEmailAdapter(EmailAdapter):
    """
    In-memory stub for email.

    Tests inject messages via :meth:`inject_messages`.
    """

    def __init__(self) -> None:
        self._pending: list[Message] = []

    def inject_messages(self, messages: list[Message]) -> None:
        """Enqueue messages to be returned by the next ``poll_new_messages`` call."""
        self._pending.extend(messages)

    async def poll_new_messages(self, account: ConnectedAccount) -> List[Message]:
        msgs = list(self._pending)
        self._pending.clear()
        return msgs


class GmailEmailAdapter(EmailAdapter):
    """
    Stub for Gmail API (OAuth) integration.

    In production, replace this stub with one that authenticates via
    ``google-auth`` + ``google-api-python-client``, calls
    ``users.messages.list`` with ``q='is:unread'`` and uses
    ``users.history.list`` for incremental fetches.

    Real integration sketch
    -----------------------
    .. code-block:: python

        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials

        class GmailEmailAdapter(EmailAdapter):
            def __init__(self, credentials: Credentials) -> None:
                self._service = build('gmail', 'v1', credentials=credentials)
                self._history_id: dict[str, str] = {}

            async def poll_new_messages(self, account):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._fetch, account)

            def _fetch(self, account):
                # Use history.list for efficient incremental polling
                ...
    """

    def __init__(self) -> None:
        self._pending: list[Message] = []

    def inject_messages(self, messages: list[Message]) -> None:
        """Enqueue messages (for testing or demo purposes)."""
        self._pending.extend(messages)

    async def poll_new_messages(self, account: ConnectedAccount) -> List[Message]:
        msgs = list(self._pending)
        self._pending.clear()
        return msgs


# ---------------------------------------------------------------------------
# LLM-backed actionable extraction
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are an assistant that extracts actionable items from messages.

Given the message below, identify any events, tasks, or reminders mentioned.
For each actionable item, return a JSON array (no markdown, no extra text).
Each element must be an object with:
  - "type": one of "EVENT", "TASK", or "REMINDER"
  - "description": concise description of the item
  - "date": the date string if explicitly mentioned (e.g. "2024-06-15"), or null
  - "time": the time string if explicitly mentioned (e.g. "14:30"), or null
  - "location": the location if mentioned, or null

If no actionable items exist, return an empty array: []

Message:
{body}
"""


def _parse_actionable_json(raw: str) -> list[dict]:
    """
    Extract a JSON array from the LLM's raw response.

    Handles markdown fences and leading/trailing text.
    """
    text = raw.strip()
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return []


def extract_actionables_from_message(
    message: Message,
    llm_provider: Optional["ModelProvider"] = None,
) -> list[ActionableItem]:
    """
    Use the LLM to extract ActionableItems from *message*.

    Falls back to an empty list when no LLM provider is configured or
    the response cannot be parsed.

    Parameters
    ----------
    message:
        The raw incoming message.
    llm_provider:
        Optional ``ModelProvider`` for ``Capability.LLM``.  When ``None``
        no extraction is performed and an empty list is returned.

    Returns
    -------
    list[ActionableItem]
        Extracted actionable items.  Each item has ``needs_clarification``
        set to ``True`` when date or time is missing (Req 10.4).
    """
    if llm_provider is None:
        return []

    prompt = _EXTRACTION_PROMPT.format(body=message.body[:4_000])

    try:
        response = llm_provider.invoke(prompt)
    except Exception as exc:
        logger.warning("LLM extraction failed for message %s: %s", message.message_id, exc)
        return []

    # Provider may return dict (stub) or str (real LLM)
    if isinstance(response, dict):
        raw_text = response.get("input", "")
        if not isinstance(raw_text, str):
            return []
    elif isinstance(response, str):
        raw_text = response
    else:
        return []

    items_data = _parse_actionable_json(raw_text)
    results: list[ActionableItem] = []

    for item_data in items_data:
        if not isinstance(item_data, dict):
            continue
        raw_type = str(item_data.get("type", "TASK")).upper()
        try:
            item_type = ActionableType(raw_type)
        except ValueError:
            item_type = ActionableType.TASK

        description = str(item_data.get("description", "")).strip()
        if not description:
            continue

        date = item_data.get("date") or None
        time_val = item_data.get("time") or None
        location = item_data.get("location") or None

        item = ActionableItem.create(
            source_account=message.source_account,
            source_message_id=message.message_id,
            type=item_type,
            description=description,
            date=str(date) if date else None,
            time=str(time_val) if time_val else None,
            location=str(location) if location else None,
        )
        results.append(item)

    return results


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

#: Maximum number of *additional* retry attempts after the first failure.
MAX_RETRIES: int = 3

#: Seconds to wait between retry attempts.
RETRY_INTERVAL_SECONDS: float = 30.0


# ---------------------------------------------------------------------------
# CommsReader
# ---------------------------------------------------------------------------


class CommsReader:
    """
    Reads WhatsApp messages and email, extracts ActionableItems.

    Parameters
    ----------
    account_manager:
        Owns account registration and grant/revoke state.
    whatsapp_adapter:
        Adapter for WhatsApp message polling.
    email_adapter:
        Adapter for email message polling.
    llm_provider:
        LLM for actionable extraction (Req 10.3).  When ``None`` no
        items are extracted.
    on_actionable:
        Optional callback ``(ActionableItem) -> None`` called for every
        extracted item (and once more if clarification is needed, via
        :meth:`_flag_incomplete`).  Used by higher-level subsystems
        (e.g. the Scheduler) to react to new items.
    on_failure:
        Optional callback ``(account_id: str, reason: str) -> None``
        called after all retries are exhausted (Req 10.7).
    poll_interval_seconds:
        How often each account is polled while connected.  Defaults to
        10 s for responsiveness within the 60 s budget.

    Design
    ------
    Each granted account gets its own asyncio polling task
    (``_start_polling_task``).  The task loops until either:
    - the account's ``stop_event`` is set (revocation within 5 s), or
    - all retry attempts are exhausted and a failure notification is sent.

    On each poll the adapter is called.  If it raises, the retry counter
    for that account is incremented.  On success the counter resets to 0.
    After ``MAX_RETRIES`` consecutive failures the user is notified and
    polling stops for that account (Req 10.6, 10.7).
    """

    def __init__(
        self,
        account_manager: Optional[AccountManager] = None,
        whatsapp_adapter: Optional[WhatsAppAdapter] = None,
        email_adapter: Optional[EmailAdapter] = None,
        llm_provider: Optional["ModelProvider"] = None,
        on_actionable: Optional[Callable[[ActionableItem], None]] = None,
        on_failure: Optional[Callable[[str, str], None]] = None,
        poll_interval_seconds: float = 10.0,
    ) -> None:
        self._account_manager = account_manager or AccountManager()
        self._whatsapp_adapter = whatsapp_adapter or StubWhatsAppAdapter()
        self._email_adapter = email_adapter or StubEmailAdapter()
        self._llm_provider = llm_provider
        self._on_actionable = on_actionable
        self._on_failure = on_failure
        self._poll_interval = poll_interval_seconds

        # Map account_id -> asyncio.Task
        self._polling_tasks: dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------
    # Public: connect / disconnect (Req 10.5, 10.8, 10.9)
    # ------------------------------------------------------------------

    async def connect(self, account: ConnectedAccount) -> None:
        """
        Grant access and begin polling *account* (Req 10.9).

        If a polling task is already running for this account it is
        cancelled before a fresh one is started.
        """
        self._account_manager.grant(account.account_id)

        # Notify the appropriate adapter
        if account.account_type == AccountType.WHATSAPP:
            await self._whatsapp_adapter.connect(account)
        else:
            await self._email_adapter.connect(account)

        await self._start_polling_task(account)
        logger.info("CommsReader: connected account %s", account.account_id)

    async def disconnect(self, account: ConnectedAccount) -> None:
        """
        Revoke access and stop polling *account* within 5 s (Req 10.8).

        Revocation sets the ``stop_event``; the polling task checks it on
        each iteration and exits.
        """
        self._account_manager.revoke(account.account_id)

        # Notify the appropriate adapter
        if account.account_type == AccountType.WHATSAPP:
            await self._whatsapp_adapter.disconnect(account)
        else:
            await self._email_adapter.disconnect(account)

        # Cancel the asyncio task as well (belt-and-suspenders)
        task = self._polling_tasks.pop(account.account_id, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        logger.info("CommsReader: disconnected account %s", account.account_id)

    # ------------------------------------------------------------------
    # Public: extract actionables from a single message (Req 10.3, 10.4)
    # ------------------------------------------------------------------

    def extract_actionables(self, message: Message) -> list[ActionableItem]:
        """
        Extract ActionableItems from *message* using the LLM (Req 10.3).

        Flags items missing date or time for clarification (Req 10.4) and
        invokes :attr:`_on_actionable` for each extracted item.

        Returns the list of extracted items.
        """
        items = extract_actionables_from_message(message, self._llm_provider)
        for item in items:
            if item.needs_clarification:
                self._flag_incomplete(item)
            if self._on_actionable is not None:
                self._on_actionable(item)
        return items

    # ------------------------------------------------------------------
    # Public: flag incomplete item (Req 10.4)
    # ------------------------------------------------------------------

    def flag_incomplete(self, item: ActionableItem) -> None:
        """
        Surface a flagged ActionableItem to the user (Req 10.4).

        This is the public entry point that higher-level components (e.g.
        the Dialogue_Manager) call.  Internally delegates to
        :meth:`_flag_incomplete`.
        """
        item.needs_clarification = True
        self._flag_incomplete(item)

    # ------------------------------------------------------------------
    # Polling internals
    # ------------------------------------------------------------------

    async def _start_polling_task(self, account: ConnectedAccount) -> None:
        """
        Launch an asyncio task that polls *account* continuously.

        Any previous task for this account is cancelled first.
        """
        old_task = self._polling_tasks.pop(account.account_id, None)
        if old_task is not None and not old_task.done():
            old_task.cancel()
            try:
                await asyncio.shield(old_task)
            except (asyncio.CancelledError, Exception):
                pass

        task = asyncio.get_event_loop().create_task(
            self._polling_loop(account),
            name=f"comms_poll_{account.account_id}",
        )
        self._polling_tasks[account.account_id] = task

    async def _polling_loop(self, account: ConnectedAccount) -> None:
        """
        Continuous polling loop for a single account.

        Polls every ``_poll_interval`` seconds until:
        - ``account.stop_event`` is set (revocation), OR
        - all retry attempts are exhausted (triggers failure notification).

        Retry logic (Req 10.6, 10.7):
        - On poll failure the retry counter increments.
        - After ``MAX_RETRIES`` consecutive failures, the user is notified
          and the loop exits.
        - On any successful poll the counter resets to 0.
        """
        retry_count: int = 0
        # Total attempts = 1 (initial) + MAX_RETRIES
        max_total_attempts: int = 1 + MAX_RETRIES

        while not account.stop_event.is_set():
            try:
                # Respect the stop_event even during the sleep interval
                # (ensures revocation stops within 5 s — Req 10.8)
                await asyncio.wait_for(
                    account.stop_event.wait(),
                    timeout=self._poll_interval,
                )
                # stop_event was set — exit
                break
            except asyncio.TimeoutError:
                pass  # Normal path — time to poll

            # Check granted state (re-check after sleep)
            if not self._account_manager.is_granted(account.account_id):
                break

            # Poll the appropriate adapter
            try:
                messages = await self._poll_account(account)
                # Success — reset retry counter
                retry_count = 0
                # Process each message
                for message in messages:
                    try:
                        self.extract_actionables(message)
                    except Exception as exc:
                        logger.warning(
                            "Failed to extract actionables from message %s: %s",
                            message.message_id,
                            exc,
                        )

            except asyncio.CancelledError:
                raise

            except Exception as exc:
                retry_count += 1
                logger.warning(
                    "Poll failure %d/%d for account %s: %s",
                    retry_count,
                    max_total_attempts,
                    account.account_id,
                    exc,
                )

                if retry_count >= max_total_attempts:
                    # All retries exhausted — notify user (Req 10.7)
                    self._notify_failure(account, str(exc))
                    break

                # Wait 30 s before retrying, but still respect stop_event
                # (so revocation can interrupt the wait — Req 10.8)
                try:
                    await asyncio.wait_for(
                        account.stop_event.wait(),
                        timeout=RETRY_INTERVAL_SECONDS,
                    )
                    break  # stop_event fired during wait
                except asyncio.TimeoutError:
                    pass  # Normal — time to retry

    async def _poll_account(self, account: ConnectedAccount) -> list[Message]:
        """
        Dispatch a single poll to the correct adapter.

        Returns the list of new messages.
        Raises on adapter errors so the caller can apply retry logic.
        """
        if account.account_type == AccountType.WHATSAPP:
            return await self._whatsapp_adapter.poll_new_messages(account)
        else:
            return await self._email_adapter.poll_new_messages(account)

    # ------------------------------------------------------------------
    # Flagging and notification helpers
    # ------------------------------------------------------------------

    def _flag_incomplete(self, item: ActionableItem) -> None:
        """
        Log and surface an item that requires user clarification (Req 10.4).

        In the running system this would push a notification or add a
        pending-clarification entry to the Dialogue_Manager.
        """
        logger.info(
            "ActionableItem requires clarification (missing date/time): "
            "account=%s, description=%r",
            item.source_account,
            item.description,
        )

    def _notify_failure(self, account: ConnectedAccount, reason: str) -> None:
        """
        Notify the user that *account* could not be read after all retries
        (Req 10.7).

        Calls :attr:`_on_failure` if provided; otherwise logs the failure.
        The notification identifies the specific account (Req 10.7).
        """
        msg = (
            f"HAKI could not read messages from '{account.display_name}' "
            f"({account.account_id}) after {MAX_RETRIES + 1} attempts. "
            f"Reason: {reason}"
        )
        logger.error(msg)
        if self._on_failure is not None:
            self._on_failure(account.account_id, reason)
