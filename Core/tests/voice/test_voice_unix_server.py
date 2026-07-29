"""Temporary same-UID integration coverage for the dedicated voice UDS server.

Validates: Requirements 3.4, 3.5, 3.6, 3.8
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import stat
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from core.ipc.voice_protocol import PCM_CHUNK, PeerSecurityError
from core.ipc.voice_unix_server import (
    AUTH_PREFIX,
    DEFAULT_RECONNECT_DELAYS,
    ReconnectBackoff,
    VoiceUnixServer,
)


@pytest.fixture
def short_runtime_dir():
    """Use a real short directory because macOS limits UDS pathname length."""
    path = Path(tempfile.mkdtemp(prefix="haki-voice-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _transcript(*, session_id: str, turn_id: str, event_seq: int, is_final: bool) -> dict[str, object]:
    return {
        "version": 1,
        "type": "TRANSCRIPT_EVENT",
        "event_id": str(uuid4()),
        "session_id": session_id,
        "turn_id": turn_id,
        "event_seq": event_seq,
        "text": "Kal meeting" if not is_final else "Kal meeting reschedule kar do",
        "is_final": is_final,
        "language": "hinglish",
        "capture_started_monotonic_ns": 1,
        "capture_ended_monotonic_ns": 2,
    }


async def _authenticated_client(server: VoiceUnixServer, capability: str | None = None):
    reader, writer = await asyncio.open_unix_connection(server.socket_path)
    writer.write(AUTH_PREFIX + (capability or server.capability).encode("ascii") + b"\n")
    await writer.drain()
    return reader, writer


async def _send_json_and_read_ack(writer, reader, message: dict[str, object]) -> dict[str, object]:
    writer.write(json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n")
    await writer.drain()
    return json.loads((await asyncio.wait_for(reader.readline(), timeout=1)).decode("utf-8"))


async def _close(writer) -> None:
    writer.close()
    try:
        await writer.wait_closed()
    except ConnectionError:
        pass


async def _wait_for(predicate) -> None:
    for _ in range(50):
        if predicate():
            return
        await asyncio.sleep(0.01)
    assert predicate()


@pytest.mark.asyncio
async def test_same_uid_server_creates_owner_only_paths_and_acks_text_transcript(short_runtime_dir: Path) -> None:
    """The dedicated socket is 0600 under a 0700 directory and ACKs only text events."""
    received: list[dict[str, object]] = []
    server = VoiceUnixServer(
        runtime_dir=short_runtime_dir / "voice-runtime",
        on_message=lambda message: received.append(message.as_dict()),
    )
    await server.start()
    try:
        assert stat.S_IMODE(os.lstat(server.runtime_dir).st_mode) == 0o700
        assert stat.S_IMODE(os.lstat(server.socket_path).st_mode) == 0o600

        reader, writer = await _authenticated_client(server)
        turn_id = str(uuid4())
        event = _transcript(
            session_id=server.session_id,
            turn_id=turn_id,
            event_seq=0,
            is_final=False,
        )
        ack = await _send_json_and_read_ack(writer, reader, event)
        assert ack == {
            "version": 1,
            "type": "EVENT_ACK",
            "event_id": event["event_id"],
            "status": "accepted",
        }
        assert received == [event]
        await _close(writer)
    finally:
        await server.stop()
    assert not server.socket_path.exists()


@pytest.mark.asyncio
async def test_rejects_wrong_peer_uid_and_wrong_session_capability(short_runtime_dir: Path) -> None:
    """Both OS credential and inherited capability gates close unauthenticated clients."""
    uid_rejecting_server = VoiceUnixServer(
        socket_path=short_runtime_dir / "wrong-uid.sock",
        peer_uid_provider=lambda _socket: os.getuid() + 1,
    )
    await uid_rejecting_server.start()
    try:
        reader, writer = await _authenticated_client(uid_rejecting_server)
        assert await asyncio.wait_for(reader.readline(), timeout=1) == b""
        await _close(writer)
    finally:
        await uid_rejecting_server.stop()

    server = VoiceUnixServer(socket_path=short_runtime_dir / "wrong-capability.sock")
    await server.start()
    try:
        reader, writer = await _authenticated_client(server, "0" * 32)
        assert await asyncio.wait_for(reader.readline(), timeout=1) == b""
        await _close(writer)

        good_reader, good_writer = await _authenticated_client(server)
        event = _transcript(
            session_id=server.session_id,
            turn_id=str(uuid4()),
            event_seq=0,
            is_final=True,
        )
        assert (await _send_json_and_read_ack(good_writer, good_reader, event))["status"] == "accepted"
        await _close(good_writer)
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_refuses_symlink_and_non_socket_stale_paths_then_reclaims_valid_stale_socket(short_runtime_dir: Path) -> None:
    """Lifecycle cleanup never follows a link or deletes an arbitrary stale file."""
    target = short_runtime_dir / "target"
    target.write_text("must survive", encoding="utf-8")
    symlink_path = short_runtime_dir / "voice.sock"
    symlink_path.symlink_to(target)
    symlink_server = VoiceUnixServer(socket_path=symlink_path)
    with pytest.raises(PeerSecurityError, match="socket_path_symlink"):
        await symlink_server.start()
    assert target.read_text(encoding="utf-8") == "must survive"
    symlink_path.unlink()

    stale_file = short_runtime_dir / "voice.sock"
    stale_file.write_text("not a socket", encoding="utf-8")
    file_server = VoiceUnixServer(socket_path=stale_file)
    with pytest.raises(PeerSecurityError, match="socket_path_not_unix_socket"):
        await file_server.start()
    stale_file.unlink()

    stale_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale_listener.bind(os.fspath(stale_file))
    stale_listener.close()
    os.chmod(stale_file, 0o600)
    stale_server = VoiceUnixServer(socket_path=stale_file)
    await stale_server.start()
    try:
        assert stat.S_ISSOCK(os.lstat(stale_file).st_mode)
    finally:
        await stale_server.stop()


def test_reconnect_backoff_is_bounded_and_resets_after_a_connection() -> None:
    """Reconnect scheduling stops after the prescribed 100/250/500/1000 ms attempts."""
    backoff = ReconnectBackoff()
    assert tuple(backoff.next_delay() for _ in DEFAULT_RECONNECT_DELAYS) == DEFAULT_RECONNECT_DELAYS
    assert backoff.next_delay() is None
    assert backoff.exhausted
    backoff.reset()
    assert not backoff.exhausted
    assert backoff.attempts == 0
    assert backoff.next_delay() == 0.1


@pytest.mark.asyncio
async def test_disconnect_before_final_discards_turn_and_late_final_is_acked_discarded(short_runtime_dir: Path) -> None:
    """A reconnect cannot resurrect the partial turn that was abandoned on disconnect."""
    accepted: list[dict[str, object]] = []
    discarded: list[tuple[str, str]] = []
    server = VoiceUnixServer(
        socket_path=short_runtime_dir / "terminal.sock",
        on_message=lambda message: accepted.append(message.as_dict()),
        on_turn_discarded=lambda turn_id, reason: discarded.append((turn_id, reason)),
    )
    await server.start()
    try:
        turn_id = str(uuid4())
        reader, writer = await _authenticated_client(server)
        partial = _transcript(
            session_id=server.session_id,
            turn_id=turn_id,
            event_seq=0,
            is_final=False,
        )
        assert (await _send_json_and_read_ack(writer, reader, partial))["status"] == "accepted"
        await _close(writer)
        await _wait_for(lambda: discarded == [(turn_id, "disconnect_before_final")])

        reconnect_reader, reconnect_writer = await _authenticated_client(server)
        late_final = _transcript(
            session_id=server.session_id,
            turn_id=turn_id,
            event_seq=1,
            is_final=True,
        )
        ack = await _send_json_and_read_ack(reconnect_writer, reconnect_reader, late_final)
        assert ack["status"] == "discarded"
        assert ack["reason"] == "turn_discarded"
        assert accepted == [partial]
        await _close(reconnect_writer)
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_rejects_client_originated_pcm_even_when_metadata_is_valid(short_runtime_dir: Path) -> None:
    """Only Python can emit PCM_CHUNK; client microphone PCM has no UDS route."""
    server = VoiceUnixServer(socket_path=short_runtime_dir / "no-input-pcm.sock")
    await server.start()
    try:
        reader, writer = await _authenticated_client(server)
        metadata = {
            "version": 1,
            "type": PCM_CHUNK,
            "session_id": server.session_id,
            "turn_id": str(uuid4()),
            "sentence_id": str(uuid4()),
            "sequence": 0,
            "sample_rate_hz": 16_000,
            "channels": 1,
            "format": "s16le",
            "byte_length": 2,
        }
        writer.write(json.dumps(metadata).encode("utf-8") + b"\n")
        await writer.drain()
        assert await asyncio.wait_for(reader.readline(), timeout=1) == b""
        await _close(writer)
    finally:
        await server.stop()
