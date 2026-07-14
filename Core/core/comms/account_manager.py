"""
AccountManager — per-account grant/revoke controls for the Comms_Reader.

Tracks which communication accounts (WhatsApp, email) are currently
connected and authorised for message polling.

Design: Comms_Reader (integration approach).
Requirements: 10.5, 10.8, 10.9.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class AccountType(str, Enum):
    """Channel/protocol of a connected account."""

    WHATSAPP = "whatsapp"
    EMAIL_IMAP = "email_imap"
    EMAIL_GMAIL = "email_gmail"


@dataclass
class ConnectedAccount:
    """
    Metadata for a single communication account.

    Attributes
    ----------
    account_id:
        Stable, unique identifier for this account (e.g. email address or
        WhatsApp phone number).
    account_type:
        The channel/protocol type.
    display_name:
        Human-readable label shown in the UI (e.g. "Gmail – user@gmail.com").
    granted:
        ``True`` when the user has granted access to this account.
        Set to ``False`` on revocation (Req 10.5, 10.8, 10.9).
    stop_event:
        An ``asyncio.Event`` that polling tasks watch; it is set on
        revocation so polling loops exit within 5 seconds (Req 10.8).
    """

    account_id: str
    account_type: AccountType
    display_name: str
    granted: bool = False
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)


class AccountManager:
    """
    Registry that stores all registered communication accounts and tracks
    their current grant/revoke state.

    The user interacts with accounts through :meth:`grant` and
    :meth:`revoke`.  The CommsReader's polling tasks watch each account's
    ``stop_event`` to stop within 5 s of revocation (Req 10.8).

    Usage::

        manager = AccountManager()
        account = manager.register(
            account_id="user@gmail.com",
            account_type=AccountType.EMAIL_GMAIL,
            display_name="Gmail – user@gmail.com",
        )
        manager.grant("user@gmail.com")   # Req 10.9 — begin reading
        ...
        manager.revoke("user@gmail.com")  # Req 10.8 — stop within 5 s
    """

    def __init__(self) -> None:
        self._accounts: Dict[str, ConnectedAccount] = {}

    # ------------------------------------------------------------------
    # Account registration
    # ------------------------------------------------------------------

    def register(
        self,
        account_id: str,
        account_type: AccountType,
        display_name: str,
    ) -> ConnectedAccount:
        """
        Register a new communication account.

        If an account with *account_id* already exists it is returned
        unchanged.  The account starts in the revoked (``granted=False``)
        state.
        """
        if account_id not in self._accounts:
            self._accounts[account_id] = ConnectedAccount(
                account_id=account_id,
                account_type=account_type,
                display_name=display_name,
            )
            logger.debug("Registered account: %s (%s)", account_id, account_type.value)
        return self._accounts[account_id]

    def unregister(self, account_id: str) -> None:
        """Remove an account from the registry (also revokes it first)."""
        if account_id in self._accounts:
            self.revoke(account_id)
            del self._accounts[account_id]
            logger.debug("Unregistered account: %s", account_id)

    # ------------------------------------------------------------------
    # Grant / revoke (Req 10.5, 10.8, 10.9)
    # ------------------------------------------------------------------

    def grant(self, account_id: str) -> None:
        """
        Grant access to *account_id*.

        Clears the ``stop_event`` so polling can start and sets
        ``granted=True`` (Req 10.9).

        Raises
        ------
        KeyError
            If *account_id* has not been registered.
        """
        account = self._get_or_raise(account_id)
        # Re-arm the stop_event so the new polling task can watch it
        account.stop_event.clear()
        account.granted = True
        logger.info("Access granted for account: %s", account_id)

    def revoke(self, account_id: str) -> None:
        """
        Revoke access to *account_id*.

        Sets the ``stop_event`` — polling tasks that inspect this event
        will exit within 5 seconds (Req 10.8) — and sets
        ``granted=False`` (Req 10.5).

        Raises
        ------
        KeyError
            If *account_id* has not been registered.
        """
        account = self._get_or_raise(account_id)
        account.granted = False
        # Signal the polling loop to stop (Req 10.8)
        account.stop_event.set()
        logger.info("Access revoked for account: %s", account_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_granted(self, account_id: str) -> bool:
        """Return ``True`` iff *account_id* is registered and granted."""
        account = self._accounts.get(account_id)
        return account is not None and account.granted

    def granted_accounts(self) -> list[ConnectedAccount]:
        """Return all currently granted accounts."""
        return [a for a in self._accounts.values() if a.granted]

    def all_accounts(self) -> list[ConnectedAccount]:
        """Return all registered accounts (granted and revoked)."""
        return list(self._accounts.values())

    def get(self, account_id: str) -> Optional[ConnectedAccount]:
        """Return the account with *account_id*, or ``None``."""
        return self._accounts.get(account_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_raise(self, account_id: str) -> ConnectedAccount:
        if account_id not in self._accounts:
            raise KeyError(f"Unknown account: '{account_id}'. Register it first.")
        return self._accounts[account_id]
