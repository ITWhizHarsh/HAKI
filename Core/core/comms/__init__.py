"""
Comms sub-package — Comms_Reader subsystem.

Reads WhatsApp messages and email, extracts ActionableItems, handles
retries, and provides per-account grant/revoke controls.

Design: Comms_Reader.
Requirements: 10.1 – 10.9.
"""

from .account_manager import AccountManager, AccountType, ConnectedAccount
from .actionable_item import ActionableItem, ActionableType
from .comms_reader import (
    CommsReader,
    EmailAdapter,
    GmailEmailAdapter,
    IMAPEmailAdapter,
    MAX_RETRIES,
    Message,
    RETRY_INTERVAL_SECONDS,
    StubEmailAdapter,
    StubWhatsAppAdapter,
    WhatsAppAdapter,
    extract_actionables_from_message,
)

__all__ = [
    # Account management
    "AccountManager",
    "AccountType",
    "ConnectedAccount",
    # Data models
    "ActionableItem",
    "ActionableType",
    "Message",
    # Core reader
    "CommsReader",
    # Adapters
    "WhatsAppAdapter",
    "StubWhatsAppAdapter",
    "EmailAdapter",
    "IMAPEmailAdapter",
    "StubEmailAdapter",
    "GmailEmailAdapter",
    # Utilities
    "extract_actionables_from_message",
    # Constants
    "MAX_RETRIES",
    "RETRY_INTERVAL_SECONDS",
]
