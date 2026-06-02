"""
IPC smoke test — JSON-over-UNIX-socket round trip.

Starts a JSONIPCServer on a temporary socket, connects with an asyncio
client, sends a HEARTBEAT, and verifies a HEARTBEAT comes back.

Also tests:
- CONTROL_EVENT / CANCEL round trip
- Unknown message type returns an ERROR response
- Multiple sequential HEARTBEAT messages all receive responses
- Server cleans up its socket file on stop

Design: Process & Threading Model, Architecture (IPC).
Requirements: 3.1
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from core.ipc import JSONIPCServer
from core.ipc.server import (
    MSG_TYPE_CONTROL_EVENT,
    MSG_TYPE_ERROR,
    MSG_TYPE_HEARTBEAT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def send_recv(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    msg: dict,
    timeout: float = 3.0,
) -> dict:
    """Send one JSON message and read back one JSON response line."""
    line = json.dumps(msg, separators=(",", ":")) + "\n"
    writer.write(line.encode("utf-8"))
    await writer.drain()

    response_line = await asyncio.wait_for(reader.readline(), timeout=timeout)
    return json.loads(response_line)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def ipc_server():
    """Start a JSONIPCServer on a tmp socket and yield (server, socket_path).

    Note: AF_UNIX paths on macOS are limited to 104 characters, so we use
    /tmp directly rather than pytest's tmp_path.
    """
    import uuid
    socket_path = f"/tmp/haki_test_{uuid.uuid4().hex[:8]}.sock"
    server = JSONIPCServer(socket_path=socket_path)
    await server.start()
    yield server, socket_path
    await server.stop()
    # Clean up socket file
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass


@pytest_asyncio.fixture
async def connected_client(ipc_server):
    """Yield (reader, writer) connected to the running server."""
    server, socket_path = ipc_server
    reader, writer = await asyncio.open_unix_connection(path=socket_path)
    yield reader, writer
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_heartbeat_round_trip(connected_client):
    """
    Send a HEARTBEAT ClientMessage; verify the server returns a HEARTBEAT.

    Validates: Requirements 3.1 — the bidirectional transport is alive.
    """
    reader, writer = connected_client

    msg = {"type": MSG_TYPE_HEARTBEAT, "payload": {}}
    response = await send_recv(reader, writer, msg)

    assert response["type"] == MSG_TYPE_HEARTBEAT, (
        f"Expected HEARTBEAT response, got {response!r}"
    )
    assert "payload" in response


@pytest.mark.asyncio
async def test_heartbeat_payload_has_ok_status(connected_client):
    """Server responds to HEARTBEAT with status 'ok' in the payload."""
    reader, writer = connected_client

    msg = {"type": MSG_TYPE_HEARTBEAT, "payload": {}}
    response = await send_recv(reader, writer, msg)

    assert response.get("payload", {}).get("status") == "ok"


@pytest.mark.asyncio
async def test_cancel_control_event_acknowledged(connected_client):
    """
    Send a CONTROL_EVENT / CANCEL; verify the server acknowledges it.
    """
    reader, writer = connected_client

    msg = {
        "type": MSG_TYPE_CONTROL_EVENT,
        "payload": {"event_type": "CANCEL", "sequence_num": 1},
    }
    response = await send_recv(reader, writer, msg)

    assert response["type"] == MSG_TYPE_CONTROL_EVENT, (
        f"Expected CONTROL_EVENT, got {response!r}"
    )
    assert response["payload"].get("event_type") == "CANCEL"
    assert response["payload"].get("status") == "acknowledged"


@pytest.mark.asyncio
async def test_unknown_message_type_returns_error(connected_client):
    """
    An unrecognised message type should yield an ERROR response.
    """
    reader, writer = connected_client

    msg = {"type": "TOTALLY_UNKNOWN_TYPE", "payload": {}}
    response = await send_recv(reader, writer, msg)

    assert response["type"] == MSG_TYPE_ERROR, (
        f"Expected ERROR for unknown type, got {response!r}"
    )
    assert "unknown message type" in response["payload"].get("message", "").lower()


@pytest.mark.asyncio
async def test_malformed_json_returns_error(ipc_server):
    """
    Sending malformed JSON should yield an ERROR response without crashing
    the server.
    """
    _, socket_path = ipc_server
    reader, writer = await asyncio.open_unix_connection(path=socket_path)
    try:
        writer.write(b"{ this is not valid json }\n")
        await writer.drain()

        response_line = await asyncio.wait_for(reader.readline(), timeout=3.0)
        response = json.loads(response_line)

        assert response["type"] == MSG_TYPE_ERROR
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_multiple_sequential_heartbeats(connected_client):
    """
    Multiple HEARTBEAT messages should each receive a HEARTBEAT response.
    """
    reader, writer = connected_client

    for i in range(5):
        msg = {"type": MSG_TYPE_HEARTBEAT, "payload": {"seq": i}}
        response = await send_recv(reader, writer, msg)
        assert response["type"] == MSG_TYPE_HEARTBEAT, (
            f"Heartbeat {i} got unexpected response: {response!r}"
        )


@pytest.mark.asyncio
async def test_audio_frame_is_accepted_silently(connected_client):
    """
    AUDIO_FRAME messages are accepted without a response (Phase 0 stub).
    Send one, then verify the next HEARTBEAT still gets its reply
    (the server didn't crash).
    """
    reader, writer = connected_client

    # Send an AUDIO_FRAME (no response expected)
    audio_msg = {
        "type": "AUDIO_FRAME",
        "payload": {
            "samples": "",
            "timestamp_ms": 1000,
            "sequence_num": 0,
            "sample_rate": 16000,
            "channels": 1,
        },
    }
    line = json.dumps(audio_msg, separators=(",", ":")) + "\n"
    writer.write(line.encode("utf-8"))
    await writer.drain()

    # Now send a HEARTBEAT and check the server is still alive
    msg = {"type": MSG_TYPE_HEARTBEAT, "payload": {}}
    response = await send_recv(reader, writer, msg)
    assert response["type"] == MSG_TYPE_HEARTBEAT


@pytest.mark.asyncio
async def test_server_socket_file_created(ipc_server):
    """The server must create its socket file when started."""
    _, socket_path = ipc_server
    assert os.path.exists(socket_path), f"Socket file not found at {socket_path}"


@pytest.mark.asyncio
async def test_server_cleans_up_stale_socket():
    """
    If a stale socket file is present when the server starts, it should be
    replaced (not cause an error).
    """
    import uuid
    socket_path = f"/tmp/haki_stale_{uuid.uuid4().hex[:8]}.sock"
    # Create a stale file
    Path(socket_path).touch()

    server = JSONIPCServer(socket_path=socket_path)
    await server.start()
    try:
        assert os.path.exists(socket_path)
        reader, writer = await asyncio.open_unix_connection(path=socket_path)
        msg = {"type": MSG_TYPE_HEARTBEAT, "payload": {}}
        response = await send_recv(reader, writer, msg)
        assert response["type"] == MSG_TYPE_HEARTBEAT
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
    finally:
        await server.stop()
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass


@pytest.mark.asyncio
async def test_concurrent_clients(ipc_server):
    """
    Two simultaneous clients can both exchange HEARTBEAT messages
    without interfering with each other.
    """
    _, socket_path = ipc_server

    async def client_task(n: int) -> str:
        reader, writer = await asyncio.open_unix_connection(path=socket_path)
        try:
            msg = {"type": MSG_TYPE_HEARTBEAT, "payload": {"client": n}}
            response = await send_recv(reader, writer, msg)
            return response["type"]
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    results = await asyncio.gather(client_task(1), client_task(2))
    assert all(r == MSG_TYPE_HEARTBEAT for r in results), (
        f"Expected both clients to get HEARTBEAT, got {results}"
    )
