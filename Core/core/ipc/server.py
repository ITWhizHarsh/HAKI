"""
IPC server — gRPC / UNIX domain socket entry point.

Listens on a UNIX domain socket scoped to the app (never a network port,
Req 20.4 / Security Considerations).  Exposes a bidirectional streaming
gRPC service (HAKICore.StreamTurn) to the Swift shell.

The .proto definition lives at:  proto/haki_ipc.proto
Generated stubs live at:         core/ipc/proto/

Also provides JSONIPCServer: a simpler JSON-over-UNIX-socket transport
using asyncio.start_unix_server for Phase 0 integration without grpc-swift.

Design: Process & Threading Model, Architecture (IPC).
Requirements: 3.1 — streaming transport for first-audio ≤ 300 ms.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator

import grpc
import grpc.aio

from core.ipc.proto import (
    ClientMessage,
    ServerMessage,
    PartialTranscript,
    LLMToken,
    TTSAudioChunk,
    ControlEvent,
    HAKICoreServicer,
    add_HAKICoreServicer_to_server,
)

logger = logging.getLogger(__name__)

# Default socket path — scoped to the app's container so it is never
# reachable off-device.
DEFAULT_SOCKET_PATH: str = str(Path.home() / ".haki" / "core.sock")


# ---------------------------------------------------------------------------
# Servicer implementation
# ---------------------------------------------------------------------------


class HAKICoreServicerImpl(HAKICoreServicer):
    """
    Pass-through servicer that implements the HAKICore gRPC interface.

    In Phase 0 this class is a minimal stub that accepts the stream and
    immediately sends a single HEARTBEAT control event to confirm the
    transport is alive.  Full pipeline wiring happens in Task 1.4 once
    both sides of the socket are connected.

    The structure matches the design's bidirectional streaming contract:
    - Inbound:  ClientMessage (audio_frame | partial_transcript |
                               turn_request | control_event)
    - Outbound: ServerMessage (partial_transcript | llm_token |
                               tts_audio_chunk | control_event | error)
    """

    def __init__(self, orchestrator: Any | None = None) -> None:
        self._orchestrator = orchestrator

    async def StreamTurn(
        self,
        request_iterator: AsyncIterator[ClientMessage],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[ServerMessage]:
        """
        Main voice/command pipeline — bidirectional streaming RPC.

        Stub behaviour (Phase 0):
          1. Acknowledge the stream with a HEARTBEAT control event.
          2. Drain inbound messages (log them at DEBUG level).
          3. When the client closes the upload side, finish gracefully.

        Full implementation is wired in Task 1.4.
        """
        logger.debug("StreamTurn: stream opened")

        # Send an immediate HEARTBEAT so the Swift client knows the server
        # is alive and the transport is healthy.
        heartbeat = ServerMessage(
            control_event=ControlEvent(
                event_type=ControlEvent.HEARTBEAT,
                sequence_num=0,
            )
        )
        yield heartbeat

        # Drain inbound messages until the client closes the stream.
        async for msg in request_iterator:
            kind = msg.WhichOneof("payload")
            logger.debug("StreamTurn: received client message kind=%s", kind)
            # TODO (Task 1.4): route to Orchestrator based on message kind:
            #   audio_frame        → Voice_Engine (VAD / STT)
            #   partial_transcript → Voice_Engine (finalize)
            #   turn_request       → Orchestrator turn loop
            #   control_event      → handle CANCEL / BARGE_IN

        logger.debug("StreamTurn: stream closed by client")


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


class IPCServer:
    """
    gRPC server bound to a UNIX domain socket.

    Phase 0 stub:  builds the server with the HAKICoreServicerImpl and the
    generated service descriptor but does not yet wire the Orchestrator.

    Full implementation (Task 1.4) adds:
    - Real Orchestrator reference passed into the servicer
    - Child-process health reporting and clean shutdown handshake
    - Reconnect / back-off logic
    """

    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET_PATH,
        orchestrator: Any | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._orchestrator = orchestrator
        self._server: grpc.aio.Server | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the gRPC server on the configured UNIX socket."""
        # Ensure parent directory exists
        Path(self._socket_path).parent.mkdir(parents=True, exist_ok=True)
        # Remove stale socket file if present
        try:
            os.unlink(self._socket_path)
        except FileNotFoundError:
            pass

        servicer = HAKICoreServicerImpl(orchestrator=self._orchestrator)

        self._server = grpc.aio.server()
        add_HAKICoreServicer_to_server(servicer, self._server)

        # Bind to the UNIX domain socket.  The "unix:" prefix is required by
        # gRPC's address resolver; it is never reachable off-device (Req 20.4).
        listen_addr = f"unix:{self._socket_path}"
        self._server.add_insecure_port(listen_addr)

        await self._server.start()
        logger.info("IPCServer listening on %s", listen_addr)

    async def stop(self, grace: float = 5.0) -> None:
        """Gracefully stop the server within *grace* seconds."""
        if self._server is not None:
            await self._server.stop(grace)
            self._server = None
            logger.info("IPCServer stopped")

    async def serve_forever(self) -> None:
        """Start the server and block until it terminates."""
        await self.start()
        if self._server is not None:
            await self._server.wait_for_termination()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "IPCServer":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()


# ---------------------------------------------------------------------------
# JSON-over-UNIX-socket server (simpler Phase 0 transport)
# ---------------------------------------------------------------------------

# Message type constants shared between client and server
MSG_TYPE_HEARTBEAT = "HEARTBEAT"
MSG_TYPE_AUDIO_FRAME = "AUDIO_FRAME"
MSG_TYPE_PARTIAL_TRANSCRIPT = "PARTIAL_TRANSCRIPT"
MSG_TYPE_TURN_REQUEST = "TURN_REQUEST"
MSG_TYPE_CONTROL_EVENT = "CONTROL_EVENT"
MSG_TYPE_LLM_TOKEN = "LLM_TOKEN"
MSG_TYPE_TTS_AUDIO_CHUNK = "TTS_AUDIO_CHUNK"
MSG_TYPE_ERROR = "ERROR"


class JSONIPCServer:
    """
    Simple JSON-over-UNIX-socket IPC server for Phase 0.

    Reads newline-delimited JSON ClientMessage dicts from each connected
    client and writes newline-delimited JSON ServerMessage dicts back.

    Message format:
      ClientMessage: {"type": "...", "payload": {...}}
      ServerMessage: {"type": "...", "payload": {...}}

    Supported client message types:
      HEARTBEAT          → responds with HEARTBEAT
      AUDIO_FRAME        → logged at DEBUG (future: pipe to Voice_Engine)
      PARTIAL_TRANSCRIPT → logged at DEBUG
      TURN_REQUEST       → logged at DEBUG (future: route to Orchestrator)
      CONTROL_EVENT      → handled: CANCEL echoed; others logged

    Design: Architecture, Security Considerations (local IPC only).
    Requirements: 3.1
    """

    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET_PATH,
        orchestrator: Any | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._orchestrator = orchestrator
        self._server: asyncio.AbstractServer | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the JSON IPC server on the configured UNIX socket."""
        Path(self._socket_path).parent.mkdir(parents=True, exist_ok=True)
        # Remove stale socket file if present
        try:
            os.unlink(self._socket_path)
        except FileNotFoundError:
            pass

        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=self._socket_path,
        )
        logger.info("JSONIPCServer listening on unix:%s", self._socket_path)

    async def stop(self, grace: float = 5.0) -> None:
        """Gracefully stop the server."""
        if self._server is not None:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=grace)
            except asyncio.TimeoutError:
                logger.warning("JSONIPCServer did not close cleanly within %ss", grace)
            self._server = None
            logger.info("JSONIPCServer stopped")

    async def serve_forever(self) -> None:
        """Start the server and block until it is stopped."""
        await self.start()
        if self._server is not None:
            async with self._server:
                await self._server.serve_forever()

    # ------------------------------------------------------------------
    # Connection handler
    # ------------------------------------------------------------------

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername") or "unix"
        logger.debug("JSONIPCServer: client connected from %s", peer)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break  # client closed connection
                line = line.rstrip(b"\n")
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("JSONIPCServer: malformed JSON from client: %s", exc)
                    await self._write_message(
                        writer,
                        {"type": MSG_TYPE_ERROR, "payload": {"message": "malformed JSON"}},
                    )
                    continue

                await self._dispatch(msg, writer)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            logger.debug("JSONIPCServer: client disconnected")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.debug("JSONIPCServer: connection closed")

    async def _dispatch(
        self,
        msg: dict[str, Any],
        writer: asyncio.StreamWriter,
    ) -> None:
        """Route an incoming client message and write the appropriate response."""
        msg_type = msg.get("type", "")
        payload = msg.get("payload", {})

        if msg_type == MSG_TYPE_HEARTBEAT:
            # Echo a HEARTBEAT so the Swift client knows the server is alive
            await self._write_message(
                writer,
                {"type": MSG_TYPE_HEARTBEAT, "payload": {"status": "ok"}},
            )
        elif msg_type == MSG_TYPE_AUDIO_FRAME:
            logger.debug("JSONIPCServer: received AUDIO_FRAME seq=%s", payload.get("sequence_num"))
            # TODO (Task 1.4+): pipe to Voice_Engine
        elif msg_type == MSG_TYPE_PARTIAL_TRANSCRIPT:
            logger.debug(
                "JSONIPCServer: received PARTIAL_TRANSCRIPT text=%r is_final=%s",
                payload.get("text"),
                payload.get("is_final"),
            )
            # TODO (Task 1.4+): forward to STT pipeline
        elif msg_type == MSG_TYPE_TURN_REQUEST:
            logger.debug(
                "JSONIPCServer: received TURN_REQUEST turn_id=%s",
                payload.get("turn_id"),
            )
            # TODO (Task 1.4+): route to Orchestrator turn loop
        elif msg_type == MSG_TYPE_CONTROL_EVENT:
            event_type = payload.get("event_type", "")
            logger.debug("JSONIPCServer: received CONTROL_EVENT type=%s", event_type)
            if event_type == "CANCEL":
                # Acknowledge the cancel
                await self._write_message(
                    writer,
                    {
                        "type": MSG_TYPE_CONTROL_EVENT,
                        "payload": {"event_type": "CANCEL", "status": "acknowledged"},
                    },
                )
            # BARGE_IN / END_OF_SPEECH forwarded to pipeline in later tasks
        else:
            logger.warning("JSONIPCServer: unknown message type %r", msg_type)
            await self._write_message(
                writer,
                {
                    "type": MSG_TYPE_ERROR,
                    "payload": {"message": f"unknown message type: {msg_type!r}"},
                },
            )

    @staticmethod
    async def _write_message(
        writer: asyncio.StreamWriter,
        msg: dict[str, Any],
    ) -> None:
        """Serialise *msg* as a single JSON line and flush."""
        line = json.dumps(msg, separators=(",", ":")) + "\n"
        writer.write(line.encode("utf-8"))
        await writer.drain()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "JSONIPCServer":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()
