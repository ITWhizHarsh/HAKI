"""
IPC server — gRPC / UNIX domain socket entry point.

Listens on a UNIX domain socket scoped to the app (never a network port,
Req 20.4 / Security Considerations).  Exposes a bidirectional streaming
gRPC service (HAKICore.StreamTurn) to the Swift shell.

The .proto definition lives at:  proto/haki_ipc.proto
Generated stubs live at:         core/ipc/proto/

Also provides JSONIPCServer: a simpler JSON-over-UNIX-socket transport
using asyncio.start_unix_server for Phase 0 integration without grpc-swift.

Live voice is handled exclusively by VoiceUnixServer + VoiceSessionPipeline
on a separate session-scoped socket (see voice_unix_server.py and
haki_core_service.py).  AUDIO_FRAME, legacy STT/TTS traffic, Edge TTS,
afplay, and say subprocess paths have been removed from this server.

Design: Process & Threading Model, Architecture (IPC).
Requirements: 3.1 — streaming transport for first-audio ≤ 300 ms.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
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
        scheduler: Any | None = None,
        task_tracker: Any | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._orchestrator = orchestrator
        self._scheduler = scheduler
        self._task_tracker = task_tracker
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
# Phase 5–6 integration message types (Task 36.1)
MSG_TYPE_IMAGE_RESPONSE = "IMAGE_RESPONSE"
MSG_TYPE_PROPOSAL = "PROPOSAL"
MSG_TYPE_REMINDER = "REMINDER"
MSG_TYPE_AUTOMATION_PROGRESS = "AUTOMATION_PROGRESS"
MSG_TYPE_TASK_ADDED = "TASK_ADDED"
# GUI Agent event message type (Req 7.1)
MSG_TYPE_AGENT_EVENT = "AGENT_EVENT"


class AgentEventType:
    """String constants for AGENT_EVENT sub-types (Req 7.2).

    All seven sub-types are defined here so every component that emits or
    validates AGENT_EVENT messages imports from a single location.
    """

    AGENT_START             = "agent_start"
    AGENT_STEP              = "agent_step"
    AGENT_DONE              = "agent_done"
    AGENT_ERROR             = "agent_error"
    AGENT_MAX_STEPS_REACHED = "agent_max_steps_reached"
    AGENT_HITL_PAUSE        = "agent_hitl_pause"
    AGENT_HITL_RESUME       = "agent_hitl_resume"

    _VALID: frozenset[str] = frozenset({
        AGENT_START,
        AGENT_STEP,
        AGENT_DONE,
        AGENT_ERROR,
        AGENT_MAX_STEPS_REACHED,
        AGENT_HITL_PAUSE,
        AGENT_HITL_RESUME,
    })

    @classmethod
    def is_valid(cls, event_type: str) -> bool:
        """Return True iff *event_type* is one of the seven defined constants."""
        return event_type in cls._VALID

# ---------------------------------------------------------------------------
# Voice path note
# ---------------------------------------------------------------------------
# Live voice is handled exclusively by VoiceUnixServer / VoiceSessionPipeline
# on a dedicated session-scoped UNIX socket (see core/ipc/voice_unix_server.py
# and haki_core_service.py).  The non-voice JSON socket handled here never
# carries raw microphone audio, legacy STT/TTS traffic, or voice playback
# subprocesses.  AUDIO_FRAME and legacy voice control messages arriving on
# this socket are dropped as unsupported.


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
      TURN_REQUEST       → routed to Orchestrator with Phase 5–6 extras
      CONTROL_EVENT      → handled: CANCEL echoed; others logged
      PROPOSAL_ACTION    → user confirmed/rejected/edited a calendar proposal
      TASK_COMPLETE      → user marked a task complete

    Phase 5–6 server-sent message types (Task 36.1):
      IMAGE_RESPONSE       — image generated by Image_Studio
      PROPOSAL             — calendar event proposal for user confirmation
      REMINDER             — task reminder surfaced in-app
      AUTOMATION_PROGRESS  — step-by-step automation progress
      TASK_ADDED           — new task created by the Task_Tracker

    Design: Architecture, Security Considerations (local IPC only).
    Requirements: 3.1, 6.1, 11.1, 12.6, 13.1, 17.5
    """

    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET_PATH,
        orchestrator: Any | None = None,
        scheduler: Any | None = None,
        task_tracker: Any | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._orchestrator = orchestrator
        self._scheduler = scheduler
        self._task_tracker = task_tracker
        self._server: asyncio.AbstractServer | None = None
        # Track all connected stream writers so AGENT_EVENTs can be fanned out
        # to every connected client (Req 7.3).
        self._connected_writers: set[asyncio.StreamWriter] = set()

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

        # Register writer for AGENT_EVENT broadcast (Req 7.3)
        self._connected_writers.add(writer)

        # Current active turn task (for barge-in cancellation)
        active_turn_task: asyncio.Task | None = None

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

                msg_type = msg.get("type", "")
                payload = msg.get("payload", {})

                # ── AUDIO_FRAME: live voice is handled by VoiceUnixServer ─
                if msg_type == MSG_TYPE_AUDIO_FRAME:
                    # Raw audio on this socket is unsupported; live voice uses
                    # the dedicated VoiceUnixServer session socket.
                    logger.debug(
                        "JSONIPCServer: received AUDIO_FRAME on non-voice socket — dropped"
                    )
                    continue

                # ── PARTIAL_TRANSCRIPT on this socket is legacy-only ──────
                if msg_type == MSG_TYPE_PARTIAL_TRANSCRIPT:
                    logger.debug(
                        "JSONIPCServer: received PARTIAL_TRANSCRIPT on non-voice socket — dropped"
                    )
                    continue

                # ── Control events ────────────────────────────────────────
                if msg_type == MSG_TYPE_CONTROL_EVENT:
                    event_type = payload.get("event_type", "")
                    logger.debug("JSONIPCServer: received CONTROL_EVENT type=%s", event_type)

                    if event_type == "END_OF_SPEECH":
                        # END_OF_SPEECH on the non-voice socket is legacy;
                        # live voice turns are driven by VoiceSessionPipeline VAD.
                        logger.debug(
                            "JSONIPCServer: END_OF_SPEECH on non-voice socket — dropped"
                        )
                        continue

                    elif event_type in ("CANCEL", "BARGE_IN"):
                        if self._orchestrator is not None:
                            self._orchestrator.cancel()
                        if active_turn_task and not active_turn_task.done():
                            active_turn_task.cancel()
                        await self._write_message(writer, {
                            "type": MSG_TYPE_CONTROL_EVENT,
                            "payload": {"event_type": event_type, "status": "acknowledged"},
                        })
                        continue

                    # All other control events fall through to _dispatch
                    await self._dispatch(msg, writer)
                    continue

                await self._dispatch(msg, writer)

        except (asyncio.IncompleteReadError, ConnectionResetError):
            logger.debug("JSONIPCServer: client disconnected")
        finally:
            # Deregister writer from broadcast set (Req 7.3)
            self._connected_writers.discard(writer)
            if active_turn_task and not active_turn_task.done():
                active_turn_task.cancel()
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.debug("JSONIPCServer: connection closed")

    async def broadcast_agent_event(self, event_type: str, payload: dict) -> None:
        """Broadcast an AGENT_EVENT to all connected IPC clients (Req 7.3, 7.4).

        Validates *event_type* against AgentEventType._VALID before sending.
        Unknown types are logged and dropped; dead connections are silently
        discarded from the writers set.
        """
        if not AgentEventType.is_valid(event_type):
            logger.warning(
                "JSONIPCServer: unknown AGENT_EVENT type %r — dropped", event_type
            )
            return
        msg = {
            "type": MSG_TYPE_AGENT_EVENT,
            "payload": {"event_type": event_type, **payload},
        }
        dead: list[asyncio.StreamWriter] = []
        for writer in list(self._connected_writers):
            try:
                await self._write_message(writer, msg)
            except Exception:
                logger.debug(
                    "JSONIPCServer: failed to write AGENT_EVENT to client — removing"
                )
                dead.append(writer)
        for w in dead:
            self._connected_writers.discard(w)

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
            # Raw audio on this socket is unsupported; live voice uses the
            # dedicated VoiceUnixServer session socket.
            logger.debug("JSONIPCServer: received AUDIO_FRAME on non-voice socket — dropped")
        elif msg_type == MSG_TYPE_PARTIAL_TRANSCRIPT:
            # Legacy partial transcript on this socket is no longer routed.
            logger.debug(
                "JSONIPCServer: received PARTIAL_TRANSCRIPT on non-voice socket — dropped"
            )
        elif msg_type == MSG_TYPE_TURN_REQUEST:
            turn_id = payload.get("turn_id", "")
            transcript = payload.get("transcript", "")
            audio_features = payload.get("audio_features", {})
            logger.debug(
                "JSONIPCServer: received TURN_REQUEST turn_id=%s transcript=%r",
                turn_id,
                transcript[:40] if transcript else "",
            )
            if self._orchestrator is not None and transcript:
                # Build an ipc_writer callable bound to this connection's writer
                # so capability handlers can push non-text events (proposals,
                # reminders, automation progress, images) to the Swift UI.
                async def _ipc_writer(msg_dict: dict) -> None:
                    await self._write_message(writer, msg_dict)

                # Assemble Phase 5–6 extras for the capability handlers.
                turn_extras: dict[str, Any] = {
                    "ipc_writer": _ipc_writer,
                }
                if self._scheduler is not None:
                    turn_extras["scheduler"] = self._scheduler
                if self._task_tracker is not None:
                    turn_extras["task_tracker"] = self._task_tracker

                # Schedule the turn as a cancellable asyncio Task so that a
                # concurrent CANCEL/BARGE_IN control event can abort it.
                async def _run_and_stream() -> None:
                    try:
                        async for token in self._orchestrator.stream_turn(
                            transcript, audio_features, extras=turn_extras
                        ):
                            await self._write_message(
                                writer,
                                {
                                    "type": MSG_TYPE_LLM_TOKEN,
                                    "payload": {
                                        "turn_id": turn_id,
                                        "token": token,
                                        "is_final": False,
                                    },
                                },
                            )
                        # Signal turn completion
                        await self._write_message(
                            writer,
                            {
                                "type": MSG_TYPE_CONTROL_EVENT,
                                "payload": {
                                    "event_type": "TURN_COMPLETE",
                                    "turn_id": turn_id,
                                },
                            },
                        )
                    except asyncio.CancelledError:
                        logger.debug("JSONIPCServer: turn %s cancelled", turn_id)
                        await self._write_message(
                            writer,
                            {
                                "type": MSG_TYPE_CONTROL_EVENT,
                                "payload": {
                                    "event_type": "TURN_CANCELLED",
                                    "turn_id": turn_id,
                                },
                            },
                        )
                    except Exception as exc:
                        logger.exception("JSONIPCServer: turn %s error: %r", turn_id, exc)
                        await self._write_message(
                            writer,
                            {
                                "type": MSG_TYPE_ERROR,
                                "payload": {
                                    "turn_id": turn_id,
                                    "message": str(exc),
                                },
                            },
                        )

                task = asyncio.ensure_future(_run_and_stream())
                self._orchestrator.set_current_task(task)
            else:
                # No orchestrator or empty transcript — echo back an error.
                await self._write_message(
                    writer,
                    {
                        "type": MSG_TYPE_ERROR,
                        "payload": {
                            "turn_id": turn_id,
                            "message": "No orchestrator configured or empty transcript.",
                        },
                    },
                )
        elif msg_type == "PROPOSAL_ACTION":
            # User confirmed/rejected/edited a calendar proposal from the Swift UI.
            proposal_id = payload.get("proposal_id", "")
            action = payload.get("action", "")  # "confirm" | "reject" | "edit"
            logger.debug(
                "JSONIPCServer: received PROPOSAL_ACTION proposal_id=%s action=%s",
                proposal_id, action,
            )
            # The Scheduler stores proposals by id; wire confirm/reject when
            # a full CalendarProposal round-trip is implemented.
            await self._write_message(
                writer,
                {
                    "type": MSG_TYPE_CONTROL_EVENT,
                    "payload": {
                        "event_type": "PROPOSAL_ACTION_ACKNOWLEDGED",
                        "proposal_id": proposal_id,
                        "action": action,
                    },
                },
            )
        elif msg_type == "TASK_COMPLETE":
            # User marked a task complete from the Swift UI (Req 13.4).
            task_id = payload.get("task_id", "")
            logger.debug("JSONIPCServer: received TASK_COMPLETE task_id=%s", task_id)
            if self._task_tracker is not None and task_id:
                self._task_tracker.mark_complete(task_id)
                await self._write_message(
                    writer,
                    {
                        "type": MSG_TYPE_CONTROL_EVENT,
                        "payload": {"event_type": "TASK_COMPLETE_ACKNOWLEDGED", "task_id": task_id},
                    },
                )
        elif msg_type == MSG_TYPE_CONTROL_EVENT:
            event_type = payload.get("event_type", "")
            logger.debug("JSONIPCServer: received CONTROL_EVENT type=%s", event_type)
            if event_type in ("CANCEL", "BARGE_IN"):
                # Cancel the active turn (barge-in / explicit cancel).
                if self._orchestrator is not None:
                    self._orchestrator.cancel()
                await self._write_message(
                    writer,
                    {
                        "type": MSG_TYPE_CONTROL_EVENT,
                        "payload": {"event_type": event_type, "status": "acknowledged"},
                    },
                )
            # END_OF_SPEECH on this socket is legacy voice; it is not routed.
        elif msg_type == MSG_TYPE_AGENT_EVENT:
            # Route inbound AGENT_EVENT (from internal components) to all clients
            # (Req 7.3, 7.4, 7.5).
            event_type = payload.get("event_type", "")
            if not AgentEventType.is_valid(event_type):
                logger.warning(
                    "JSONIPCServer: unknown AGENT_EVENT sub-type %r — dropped", event_type
                )
                return
            await self.broadcast_agent_event(
                event_type,
                {k: v for k, v in payload.items() if k != "event_type"},
            )
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
