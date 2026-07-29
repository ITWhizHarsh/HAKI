"""Same-UID UNIX-domain transport for the local voice protocol.

``VoiceUnixServer`` is intentionally independent from :mod:`core.ipc.server`.
It accepts only the versioned text/control protocol from ``voice_protocol``;
microphone PCM is never accepted from a client.  Its only PCM facility is the
explicit Python-to-renderer ``PCM_CHUNK`` output frame.

A client proves possession of the launch-inherited session capability before
sending a protocol record::

    VOICE_AUTH <32-lowercase-hex-capability>\n
The authentication preface is transport metadata, not a protocol message.  No
capability is logged or acknowledged.  A client that loses its connection
before its final transcript event must discard that turn: this server marks the
partial turn terminal and replies ``discarded`` to any late event after a
reconnect.
"""

from __future__ import annotations

import asyncio
import ctypes
import errno
import inspect
import logging
import os
import socket
import stat
import struct
import sys
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from uuid import UUID, uuid4

from .voice_protocol import (
    CAPTURE_INTERRUPTED,
    EVENT_ACK,
    MAX_JSON_LINE_BYTES,
    PCM_ACCEPTED,
    PCM_CHUNK,
    PLAYBACK_TERMINAL_TYPES,
    STOP_PLAYBACK,
    STOP_PLAYBACK_ACK,
    TRANSCRIPT_EVENT,
    PeerSecurityError,
    ValidatedMessage,
    VoiceProtocolError,
    VoiceProtocolSession,
    encode_jsonl,
    encode_pcm_chunk,
    generate_session_capability,
    validate_message,
    validate_owner_only_directory,
    validate_owner_only_socket,
    validate_peer_uid,
    validate_session_capability,
)

logger = logging.getLogger(__name__)

RUNTIME_DIRECTORY_MODE: Final = 0o700
SOCKET_MODE: Final = 0o600
AUTH_PREFIX: Final = b"VOICE_AUTH "
AUTH_LINE_MAX_BYTES: Final = len(AUTH_PREFIX) + 32 + 1
DEFAULT_RECONNECT_DELAYS: Final = (0.1, 0.25, 0.5, 1.0)
DEFAULT_MAX_TRACKED_TURNS: Final = 128

MessageHandler = Callable[[ValidatedMessage], Awaitable[None] | None]
TurnDiscardHandler = Callable[[str, str], Awaitable[None] | None]
PeerUIDProvider = Callable[[socket.socket], int]


class VoiceUnixServerError(RuntimeError):
    """Stable lifecycle error that never includes transcript or capability data."""


@dataclass
class ReconnectBackoff:
    """Bounded client reconnect schedule specified by the voice transport design.

    The Swift client owns retry execution.  Keeping the schedule here gives
    both sides one testable definition and prevents an unbounded retry loop
    from becoming part of server/session state.
    """

    delays: tuple[float, ...] = DEFAULT_RECONNECT_DELAYS
    _attempt: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.delays or any(delay <= 0 for delay in self.delays):
            raise ValueError("reconnect_delays_must_be_positive")

    @property
    def exhausted(self) -> bool:
        """Whether the caller must wait for an explicit user retry."""
        return self._attempt >= len(self.delays)

    @property
    def attempts(self) -> int:
        """Return the number of scheduled retries already consumed."""
        return self._attempt

    def next_delay(self) -> float | None:
        """Return the next bounded delay, or ``None`` after the final retry."""
        if self.exhausted:
            return None
        delay = self.delays[self._attempt]
        self._attempt += 1
        return delay

    def reset(self) -> None:
        """Reset retry state after a successful authenticated connection."""
        self._attempt = 0


def default_voice_runtime_directory() -> Path:
    """Return the session-runtime directory without creating it.

    macOS uses the documented Application Support fallback when an XDG runtime
    directory is unavailable.  The caller creates and validates this final
    directory with owner-only permissions before binding a socket.
    """
    xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime_dir:
        candidate = Path(xdg_runtime_dir)
        if candidate.is_absolute():
            return candidate / "haki" / "voice"
    return Path.home() / "Library" / "Application Support" / "HAKI" / "runtime" / "voice"


def create_owner_only_runtime_directory(path: str | Path, *, owner_uid: int | None = None) -> Path:
    """Create one secure runtime directory and refuse symlinked path components."""
    runtime_dir = Path(path)
    if not runtime_dir.is_absolute():
        raise PeerSecurityError("runtime_directory_not_absolute")

    _reject_symlink_components(runtime_dir)
    try:
        runtime_dir.mkdir(mode=RUNTIME_DIRECTORY_MODE, parents=True, exist_ok=True)
    except OSError as exc:
        raise PeerSecurityError("runtime_directory_unavailable") from exc
    _reject_symlink_components(runtime_dir)

    try:
        status = os.lstat(runtime_dir)
    except OSError as exc:
        raise PeerSecurityError("runtime_directory_unavailable") from exc
    expected_uid = os.getuid() if owner_uid is None else owner_uid
    if not stat.S_ISDIR(status.st_mode):
        raise PeerSecurityError("runtime_directory_not_directory")
    if status.st_uid != expected_uid:
        raise PeerSecurityError("runtime_directory_owner_mismatch")

    # ``mkdir`` is subject to the process umask.  The directory is already
    # owner-checked and not a symlink, so reducing its mode is safe.
    try:
        os.chmod(runtime_dir, RUNTIME_DIRECTORY_MODE)
    except OSError as exc:
        raise PeerSecurityError("runtime_directory_unavailable") from exc
    validate_owner_only_directory(runtime_dir, owner_uid=expected_uid)
    return runtime_dir


def remove_validated_stale_socket(path: str | Path, *, owner_uid: int | None = None) -> None:
    """Unlink only a stale, owner-owned 0600 UNIX socket.

    Regular files, directories, symlinks, foreign-owner sockets, and an active
    listener are all refused.  The device/inode comparison prevents removing a
    replacement path observed between validation and unlinking.
    """
    socket_path = Path(path)
    try:
        before = os.lstat(socket_path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise PeerSecurityError("socket_path_unavailable") from exc

    if stat.S_ISLNK(before.st_mode):
        raise PeerSecurityError("socket_path_symlink")
    if not stat.S_ISSOCK(before.st_mode):
        raise PeerSecurityError("socket_path_not_unix_socket")
    validate_owner_only_socket(socket_path, owner_uid=owner_uid)

    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.1)
        result = probe.connect_ex(os.fspath(socket_path))
    finally:
        probe.close()
    if result in (0, errno.EISCONN):
        raise VoiceUnixServerError("socket_path_in_use")
    if result not in (errno.ECONNREFUSED, errno.ENOENT):
        raise VoiceUnixServerError("socket_path_unavailable")

    try:
        current = os.lstat(socket_path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise PeerSecurityError("socket_path_unavailable") from exc
    if stat.S_ISLNK(current.st_mode):
        raise PeerSecurityError("socket_path_symlink")
    if (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
        raise VoiceUnixServerError("socket_path_changed")
    if not stat.S_ISSOCK(current.st_mode):
        raise PeerSecurityError("socket_path_not_unix_socket")
    validate_owner_only_socket(socket_path, owner_uid=owner_uid)
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise VoiceUnixServerError("stale_socket_unlink_failed") from exc


def peer_uid_from_socket(peer_socket: socket.socket) -> int:
    """Read UDS credentials on supported platforms and fail closed elsewhere."""
    getpeereid = getattr(peer_socket, "getpeereid", None)
    if callable(getpeereid):
        credentials = getpeereid()
        if not isinstance(credentials, tuple) or not credentials:
            raise PeerSecurityError("peer_credentials_unavailable")
        return int(credentials[0])

    if sys.platform == "darwin":
        # CPython's asyncio transport exposes a socket wrapper on macOS, but
        # that wrapper does not consistently expose ``getpeereid``.  macOS
        # provides the same kernel credential primitive through libc.
        try:
            peer_uid = ctypes.c_uint()
            peer_gid = ctypes.c_uint()
            getpeereid = ctypes.CDLL(None, use_errno=True).getpeereid
            getpeereid.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint),
                ctypes.POINTER(ctypes.c_uint),
            ]
            getpeereid.restype = ctypes.c_int
            if getpeereid(peer_socket.fileno(), ctypes.byref(peer_uid), ctypes.byref(peer_gid)) != 0:
                raise OSError(ctypes.get_errno(), "getpeereid failed")
            return int(peer_uid.value)
        except (AttributeError, OSError) as exc:
            raise PeerSecurityError("peer_credentials_unavailable") from exc

    so_peercred = getattr(socket, "SO_PEERCRED", None)
    if so_peercred is not None:
        try:
            raw_credentials = peer_socket.getsockopt(socket.SOL_SOCKET, so_peercred, 12)
            # Linux ucred: pid_t pid; uid_t uid; gid_t gid.
            _, peer_uid, _ = struct.unpack("3i", raw_credentials)
            return peer_uid
        except OSError as exc:
            raise PeerSecurityError("peer_credentials_unavailable") from exc

    raise PeerSecurityError("peer_credentials_unavailable")


class VoiceUnixServer:
    """Dedicated authenticated UDS server for a single local voice session.

    A server owns one random session ID, random capability, and socket path. It
    accepts one authenticated client at a time.  Message callbacks receive only
    schema-validated objects after same-UID, capability, session, and ordering
    checks pass.  Callbacks are intentionally separate from ``JSONIPCServer``
    so non-voice handlers cannot receive or emit voice transport traffic.
    """

    def __init__(
        self,
        *,
        runtime_dir: str | Path | None = None,
        socket_path: str | Path | None = None,
        session_id: str | UUID | None = None,
        capability: str | None = None,
        owner_uid: int | None = None,
        on_message: MessageHandler | None = None,
        on_turn_discarded: TurnDiscardHandler | None = None,
        peer_uid_provider: PeerUIDProvider = peer_uid_from_socket,
        auth_timeout_seconds: float = 5.0,
        max_tracked_turns: int = DEFAULT_MAX_TRACKED_TURNS,
    ) -> None:
        if auth_timeout_seconds <= 0:
            raise ValueError("auth_timeout_seconds_must_be_positive")
        if max_tracked_turns <= 0:
            raise ValueError("max_tracked_turns_must_be_positive")

        self.owner_uid = os.getuid() if owner_uid is None else owner_uid
        self.session_id = str(UUID(str(session_id))) if session_id is not None else str(uuid4())
        self.capability = capability or generate_session_capability()
        validate_session_capability(self.capability, self.capability)

        if socket_path is None:
            self.runtime_dir = Path(runtime_dir) if runtime_dir is not None else default_voice_runtime_directory()
            self.socket_path = self.runtime_dir / f"{self.session_id}.sock"
        else:
            self.socket_path = Path(socket_path)
            self.runtime_dir = self.socket_path.parent
            if runtime_dir is not None and Path(runtime_dir) != self.runtime_dir:
                raise ValueError("runtime_dir_must_match_socket_parent")
        if not self.socket_path.is_absolute():
            raise ValueError("socket_path_must_be_absolute")

        self._protocol_session = VoiceProtocolSession(self.session_id)
        self._on_message = on_message
        self._on_turn_discarded = on_turn_discarded
        self._peer_uid_provider = peer_uid_provider
        self._auth_timeout_seconds = auth_timeout_seconds
        self._max_tracked_turns = max_tracked_turns
        self._server: asyncio.AbstractServer | None = None
        self._active_writer: asyncio.StreamWriter | None = None
        self._client_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._owned_socket_identity: tuple[int, int] | None = None
        self._discarded_turn_ids: set[str] = set()
        self._discarded_turn_order: deque[str] = deque()
        self._active_turn_ids: set[str] = set()
        self.reconnect_backoff = ReconnectBackoff()

    @property
    def is_running(self) -> bool:
        """Whether the server currently owns a listening socket."""
        return self._server is not None

    @property
    def discarded_turn_ids(self) -> frozenset[str]:
        """Return terminally discarded turn IDs without exposing transcript content."""
        return frozenset(self._discarded_turn_ids)

    async def start(self) -> None:
        """Create an owner-only session socket through an atomic AF_UNIX bind."""
        if self._server is not None:
            raise VoiceUnixServerError("server_already_started")

        create_owner_only_runtime_directory(self.runtime_dir, owner_uid=self.owner_uid)
        remove_validated_stale_socket(self.socket_path, owner_uid=self.owner_uid)

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            # Bind is atomic.  Tightening the umask around the synchronous bind
            # ensures the path is never momentarily group/world accessible.
            previous_umask = os.umask(0o177)
            try:
                listener.bind(os.fspath(self.socket_path))
            finally:
                os.umask(previous_umask)
            listener.listen(socket.SOMAXCONN)
            listener.setblocking(False)
            validate_owner_only_socket(self.socket_path, owner_uid=self.owner_uid)
            identity = os.lstat(self.socket_path)
            self._owned_socket_identity = (identity.st_dev, identity.st_ino)
            self._server = await asyncio.start_unix_server(
                self._handle_client,
                sock=listener,
                limit=MAX_JSON_LINE_BYTES,
            )
        except Exception:
            listener.close()
            self._unlink_owned_socket()
            raise

    async def stop(self, grace: float = 5.0) -> None:
        """Close the listener and unlink only the socket this server created."""
        if self._server is not None:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=grace)
            except asyncio.TimeoutError:
                logger.warning("VoiceUnixServer did not close within %ss", grace)
            self._server = None

        writer = self._active_writer
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass
        self._unlink_owned_socket()

    async def serve_forever(self) -> None:
        """Start the server and run it until cancelled or stopped."""
        await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def __aenter__(self) -> "VoiceUnixServer":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    async def send_message(self, message: dict[str, object]) -> None:
        """Send one permitted Python-to-Swift JSONL control record."""
        validated = validate_message(message)
        if validated.message_type not in {EVENT_ACK, STOP_PLAYBACK, STOP_PLAYBACK_ACK}:
            raise VoiceUnixServerError("outbound_message_type_forbidden")
        await self._write_jsonl(validated)

    async def send_pcm_chunk(self, metadata: dict[str, object], payload: bytes) -> None:
        """Send renderer PCM; this is the only binary payload accepted by the transport."""
        validated = validate_message(metadata)
        if validated.message_type != PCM_CHUNK:
            raise VoiceUnixServerError("pcm_chunk_metadata_required")
        frame = encode_pcm_chunk(validated, payload)
        await self._write_bytes(frame)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        authenticated = False
        connection_turn_ids: set[str] = set()
        try:
            peer_socket = writer.get_extra_info("socket")
            if peer_socket is None:
                raise PeerSecurityError("peer_credentials_unavailable")
            validate_peer_uid(self._peer_uid_provider(peer_socket), owner_uid=self.owner_uid)
            await self._authenticate(reader)

            async with self._client_lock:
                if self._active_writer is not None and not self._active_writer.is_closing():
                    raise VoiceUnixServerError("voice_client_already_connected")
                self._active_writer = writer
            authenticated = True
            self.reconnect_backoff.reset()

            while True:
                line = await self._read_protocol_line(reader)
                if not line:
                    break
                await self._accept_incoming(line, connection_turn_ids)
        except (PeerSecurityError, VoiceProtocolError, VoiceUnixServerError, asyncio.LimitOverrunError, ValueError) as exc:
            # Deliberately do not log peer data, socket paths, or capabilities.
            logger.info("VoiceUnixServer rejected connection: %s", _safe_reason(exc))
        except (ConnectionError, OSError, asyncio.IncompleteReadError):
            logger.debug("VoiceUnixServer client disconnected")
        finally:
            if authenticated:
                for turn_id in tuple(connection_turn_ids):
                    await self._discard_turn(turn_id, "disconnect_before_final")
                async with self._client_lock:
                    if self._active_writer is writer:
                        self._active_writer = None
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def _authenticate(self, reader: asyncio.StreamReader) -> None:
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=self._auth_timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise PeerSecurityError("authentication_timeout") from exc
        if not line or len(line) > AUTH_LINE_MAX_BYTES:
            raise PeerSecurityError("invalid_session_capability")
        if not line.endswith(b"\n") or not line.startswith(AUTH_PREFIX):
            raise PeerSecurityError("invalid_session_capability")
        try:
            capability = line[len(AUTH_PREFIX) : -1].decode("ascii")
        except UnicodeDecodeError as exc:
            raise PeerSecurityError("invalid_session_capability") from exc
        validate_session_capability(capability, self.capability)

    async def _read_protocol_line(self, reader: asyncio.StreamReader) -> bytes:
        try:
            line = await reader.readline()
        except ValueError as exc:
            # StreamReader uses ValueError for an over-limit separator search.
            raise VoiceProtocolError("json_line_too_large") from exc
        if len(line) > MAX_JSON_LINE_BYTES:
            raise VoiceProtocolError("json_line_too_large")
        return line

    async def _accept_incoming(self, line: bytes, connection_turn_ids: set[str]) -> None:
        # Validate first so an EVENT_ACK never turns an invalid payload into an
        # accepted one.  The strict schema also blocks microphone field names.
        validated = validate_message_from_line(line)
        message_type = validated.message_type
        data = validated.data

        if message_type == PCM_CHUNK:
            raise VoiceProtocolError("incoming_pcm_forbidden")
        if message_type in {STOP_PLAYBACK, STOP_PLAYBACK_ACK}:
            raise VoiceProtocolError("incoming_control_direction_invalid")

        if message_type == TRANSCRIPT_EVENT:
            event_id = data["event_id"]
            turn_id = data["turn_id"]
            if turn_id in self._discarded_turn_ids:
                await self._send_ack(event_id, "discarded", "turn_discarded")
                return
            if turn_id not in self._active_turn_ids and len(self._active_turn_ids) >= self._max_tracked_turns:
                await self._send_ack(event_id, "discarded", "turn_registry_full")
                return
            try:
                self._protocol_session.accept(validated)
            except VoiceProtocolError as exc:
                await self._send_ack(event_id, "discarded", exc.reason)
                return

            if data["is_final"]:
                self._active_turn_ids.discard(turn_id)
                connection_turn_ids.discard(turn_id)
            else:
                self._active_turn_ids.add(turn_id)
                connection_turn_ids.add(turn_id)
            await self._send_ack(event_id, "accepted")
            await _invoke_callback(self._on_message, validated)
            return

        if message_type == CAPTURE_INTERRUPTED:
            self._protocol_session.accept(validated)
            await self._discard_turn(data["turn_id"], "capture_interrupted")
            await self._send_ack(data["event_id"], "accepted")
            await _invoke_callback(self._on_message, validated)
            return

        if message_type in PLAYBACK_TERMINAL_TYPES:
            self._protocol_session.accept(validated)
            await self._send_ack(data["event_id"], "accepted")
            await _invoke_callback(self._on_message, validated)
            return

        # EVENT_ACK and PCM_ACCEPTED are response/control receipts.  They have
        # no event ID that can be acknowledged without causing an ACK loop.
        if message_type in {EVENT_ACK, PCM_ACCEPTED}:
            self._protocol_session.accept(validated)
            await _invoke_callback(self._on_message, validated)
            return
        raise VoiceProtocolError("incoming_message_type_invalid")

    async def _discard_turn(self, turn_id: str, reason: str) -> None:
        if turn_id in self._discarded_turn_ids:
            return
        self._active_turn_ids.discard(turn_id)
        self._discarded_turn_ids.add(turn_id)
        self._discarded_turn_order.append(turn_id)
        # The registry has a hard cap.  A session normally has one current turn;
        # retaining this small terminal set only covers reconnect races.
        while len(self._discarded_turn_order) > self._max_tracked_turns:
            expired = self._discarded_turn_order.popleft()
            self._discarded_turn_ids.discard(expired)
        await _invoke_callback(self._on_turn_discarded, turn_id, reason)

    async def _send_ack(self, event_id: str, status: str, reason: str | None = None) -> None:
        message: dict[str, object] = {
            "version": 1,
            "type": EVENT_ACK,
            "event_id": event_id,
            "status": status,
        }
        if reason is not None:
            message["reason"] = reason
        await self._write_jsonl(validate_message(message))

    async def _write_jsonl(self, message: ValidatedMessage) -> None:
        await self._write_bytes(encode_jsonl(message))

    async def _write_bytes(self, payload: bytes) -> None:
        async with self._write_lock:
            writer = self._active_writer
            if writer is None or writer.is_closing():
                raise VoiceUnixServerError("voice_client_not_connected")
            writer.write(payload)
            try:
                await writer.drain()
            except (ConnectionError, OSError) as exc:
                raise VoiceUnixServerError("voice_client_disconnected") from exc

    def _unlink_owned_socket(self) -> None:
        identity = self._owned_socket_identity
        self._owned_socket_identity = None
        if identity is None:
            return
        try:
            status = os.lstat(self.socket_path)
        except FileNotFoundError:
            return
        except OSError:
            return
        if stat.S_ISLNK(status.st_mode):
            return
        if (status.st_dev, status.st_ino) != identity or not stat.S_ISSOCK(status.st_mode):
            return
        try:
            os.unlink(self.socket_path)
        except OSError:
            logger.warning("VoiceUnixServer could not remove its socket")


def validate_message_from_line(line: bytes) -> ValidatedMessage:
    """Decode JSONL locally to keep transport framing separate from callbacks."""
    from .voice_protocol import decode_jsonl

    return decode_jsonl(line)


def _reject_symlink_components(path: Path) -> None:
    """Refuse a symlink at the owner-only runtime-directory boundary itself.

    System temporary roots on macOS may contain root-owned compatibility
    symlinks (for example ``/var``).  The protected boundary is the final
    runtime directory: after it is created at 0700, socket children cannot be
    replaced by another user.  The socket path is independently lstat-checked.
    """
    try:
        status = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise PeerSecurityError("runtime_directory_unavailable") from exc
    if stat.S_ISLNK(status.st_mode):
        raise PeerSecurityError("runtime_directory_symlink")


def _safe_reason(error: BaseException) -> str:
    """Return only stable machine reasons in logs, never arbitrary exception text."""
    if isinstance(error, (PeerSecurityError, VoiceProtocolError)):
        return error.reason
    if isinstance(error, VoiceUnixServerError):
        return str(error)
    return "connection_error"


async def _invoke_callback(callback: Callable[..., Awaitable[None] | None] | None, *args: object) -> None:
    """Run a callback without allowing application code to crash the transport."""
    if callback is None:
        return
    try:
        result = callback(*args)
        if inspect.isawaitable(result):
            await result
    except Exception:
        logger.exception("VoiceUnixServer callback failed")


# ---------------------------------------------------------------------------
# Replacement-gate session composition
# ---------------------------------------------------------------------------

@dataclass
class ReplacementSessionHandle:
    """Opaque handle returned by ``start_replacement_session``.

    Holds the live ``VoiceUnixServer``, ``VoiceSession``, and
    ``VoiceSessionPipeline`` for one active replacement voice session.
    The caller is responsible for calling ``shutdown()`` when the session ends.
    """

    server: "VoiceUnixServer"
    session: "object"    # core.voice.session.VoiceSession
    pipeline: "object"   # core.voice.pipeline.VoiceSessionPipeline

    async def shutdown(self, grace: float = 5.0) -> None:
        """Close the pipeline, the session, and the UDS server in order."""
        pipeline_close = getattr(self.pipeline, "close", None)
        if callable(pipeline_close):
            try:
                await pipeline_close()
            except Exception:
                logger.debug("ReplacementSessionHandle: pipeline close raised", exc_info=True)
        await self.server.stop(grace=grace)


async def start_replacement_session(
    session_id: "str | UUID | None" = None,
    *,
    runtime_dir: "str | Path | None" = None,
    socket_path: "str | Path | None" = None,
    capability: "str | None" = None,
    on_turn_discarded: "TurnDiscardHandler | None" = None,
) -> "ReplacementSessionHandle":
    """Compose and start a replacement-path voice session.

    This is the sole composition entry point for the new local voice path.
    It wires ``VoiceSession`` + ``VoiceSessionPipeline`` + ``VoiceUnixServer``
    into a single session-scoped unit behind the internal development gate.

    **Gate contract (Req 1.5–1.6, Design §11 step 3):**
    - The gate must be enabled (``HAKI_VOICE_DEV_REPLACEMENT=1``) before
      calling this function; it raises ``VoiceUnixServerError`` otherwise.
    - Only the new local path is composed.  No legacy voice handler, STT,
      TTS, afplay, edge-tts, or Orchestrator voice route is referenced.
    - Non-voice IPC handlers in ``core.ipc.server`` are not touched.

    Parameters
    ----------
    session_id:
        Optional fixed session UUID.  A random UUID is generated when absent.
    runtime_dir:
        Override for the socket parent directory (default: XDG/App Support).
    socket_path:
        Override for the full socket path (takes precedence over runtime_dir).
    capability:
        Override session capability (random 32-hex when absent).
    on_turn_discarded:
        Callback invoked when a turn is discarded due to disconnect/interrupt.

    Returns
    -------
    ReplacementSessionHandle
        Started server + live session + started pipeline.  Caller must call
        ``handle.shutdown()`` to release resources.
    """
    from core.voice.dev_gate import VOICE_REPLACEMENT_GATE_ENABLED
    if not VOICE_REPLACEMENT_GATE_ENABLED:
        raise VoiceUnixServerError(
            "start_replacement_session requires HAKI_VOICE_DEV_REPLACEMENT=1"
        )

    # Import only replacement-path components; no legacy imports below.
    from uuid import uuid4 as _uuid4
    from core.voice.session import VoiceSession
    from core.voice.pipeline import (
        VoiceSessionPipeline,
        VoiceIngressProcessors,
        PipecatFrameAdapter,
        VoicePipelineSinks,
    )
    from core.voice.asr_bridge import (  # noqa: PLC0415
        AuthenticatedRingSlotReader,
        RingSlotDescriptor,
    )

    class _NullRingSlotReader:
        """Inert ring reader for sessions where the shared-memory ring is not wired."""

        def __init__(self) -> None:
            self.session_id: UUID = resolved_session_id

        async def map_slot(self, descriptor: "RingSlotDescriptor") -> bytes:
            return b""

        async def release_slot(self, descriptor: "RingSlotDescriptor") -> None:
            pass

    resolved_session_id: UUID = (
        UUID(str(session_id)) if session_id is not None else _uuid4()
    )

    # --- Voice session -------------------------------------------------
    session = VoiceSession(resolved_session_id)

    # --- UDS server ----------------------------------------------------
    server = VoiceUnixServer(
        runtime_dir=runtime_dir,
        socket_path=socket_path,
        session_id=resolved_session_id,
        capability=capability,
        on_message=None,          # pipeline wires its own transcript handler
        on_turn_discarded=on_turn_discarded,
    )

    # --- Pipecat ingress processors ------------------------------------
    frame_adapter = PipecatFrameAdapter()
    ring_reader = _NullRingSlotReader()
    ingress = VoiceIngressProcessors(
        session=session,
        ring_reader=ring_reader,
        frame_adapter=frame_adapter,
    )

    # --- Pipeline -------------------------------------------------------
    pipeline = VoiceSessionPipeline(
        session=session,
        ingress=ingress,
        sinks=VoicePipelineSinks(),
    )

    # Wire the UDS server's transcript messages into the pipeline so that
    # final Transcript_Events from Swift route to Pipecat immediately.
    async def _on_transcript_message(validated: "object") -> None:
        try:
            await pipeline.ingest_transcript_message(validated)
        except Exception:
            logger.debug("replacement_session: transcript ingress error", exc_info=True)

    server._on_message = _on_transcript_message  # type: ignore[assignment]

    # Start the server first so the socket is ready before the pipeline.
    await server.start()
    try:
        await pipeline.start()
    except Exception:
        await server.stop()
        raise

    return ReplacementSessionHandle(server=server, session=session, pipeline=pipeline)


__all__ = [
    "AUTH_PREFIX",
    "DEFAULT_RECONNECT_DELAYS",
    "ReconnectBackoff",
    "RUNTIME_DIRECTORY_MODE",
    "SOCKET_MODE",
    "ReplacementSessionHandle",
    "VoiceUnixServer",
    "VoiceUnixServerError",
    "create_owner_only_runtime_directory",
    "default_voice_runtime_directory",
    "peer_uid_from_socket",
    "remove_validated_stale_socket",
    "start_replacement_session",
]
