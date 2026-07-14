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

        # Per-connection audio frame buffer: accumulate between speech segments
        audio_buffer: list[bytes] = []
        audio_sample_rate: int = 16_000
        # Current active turn task (for barge-in cancellation)
        active_turn_task: asyncio.Task | None = None
        # Barge-in / overlap control:
        #   speaking_state["proc"]   — the live `afplay` subprocess, so it can be
        #                              killed instantly on barge-in / cancel.
        #   speaking_state["active"] — True from the moment a turn is spawned
        #                              until it fully finishes, so a second
        #                              END_OF_SPEECH cannot start a turn that
        #                              would play OVER the first (the root cause
        #                              of the "two voices at once" overlap).
        speaking_state: dict[str, Any] = {"proc": None, "active": False}

        def _kill_playback() -> None:
            proc = speaking_state.get("proc")
            if proc is not None and proc.returncode is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
            speaking_state["proc"] = None

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

                # ── Audio frame accumulation ────────────────────────────
                if msg_type == MSG_TYPE_AUDIO_FRAME:
                    raw = payload.get("samples_b64") or payload.get("samples")
                    if raw:
                        import base64
                        if isinstance(raw, str):
                            audio_buffer.append(base64.b64decode(raw))
                        elif isinstance(raw, list):
                            import struct
                            audio_buffer.append(struct.pack(f"<{len(raw)}h", *raw))
                    audio_sample_rate = payload.get("sample_rate", 16_000)
                    continue

                # ── End of speech: run STT then LLM ────────────────────
                if msg_type == MSG_TYPE_CONTROL_EVENT:
                    event_type = payload.get("event_type", "")
                    logger.debug("JSONIPCServer: received CONTROL_EVENT type=%s", event_type)

                    if event_type == "END_OF_SPEECH":
                        frames_bytes = b"".join(audio_buffer)
                        audio_buffer.clear()

                        if not frames_bytes:
                            logger.debug("JSONIPCServer: END_OF_SPEECH with no audio — skipping")
                            continue

                        # Overlap guard: if a turn is already running or HAKI is
                        # still speaking, do NOT start a competing turn — that is
                        # exactly what produced two voices playing at once. A real
                        # interruption arrives as a CANCEL / BARGE_IN event, which
                        # is handled separately below.
                        if speaking_state["active"]:
                            logger.info(
                                "JSONIPCServer: a turn is already active — ignoring overlapping END_OF_SPEECH"
                            )
                            continue

                        # Run STT + LLM + TTS in a background task so we don't block the read loop
                        async def _stt_then_llm_then_tts(
                            audio: bytes,
                            sample_rate: int,
                            w: asyncio.StreamWriter,
                        ) -> None:
                            # ── STT ────────────────────────────────────
                            transcript = ""
                            try:
                                if self._orchestrator and hasattr(self._orchestrator, "_stt_engine"):
                                    result = await self._orchestrator._stt_engine.transcribe(
                                        audio, sample_rate
                                    )
                                    transcript = result.text.strip()
                                else:
                                    logger.warning("JSONIPCServer: no STT engine available")
                            except Exception as exc:
                                logger.exception("JSONIPCServer: STT error: %s", exc)

                            logger.info("JSONIPCServer: transcript=%r", transcript)

                            # Send partial transcript result back to Swift STTService
                            await self._write_message(w, {
                                "type": MSG_TYPE_PARTIAL_TRANSCRIPT,
                                "payload": {
                                    "text": transcript,
                                    "is_final": True,
                                    "confidence": 1.0,
                                },
                            })

                            if not transcript:
                                return

                            # ── LLM turn ───────────────────────────────
                            if self._orchestrator is None:
                                return

                            turn_id = f"turn_{id(audio)}"
                            async def _ipc_writer(msg_dict: dict) -> None:
                                await self._write_message(w, msg_dict)

                            turn_extras: dict = {"ipc_writer": _ipc_writer}
                            if self._scheduler is not None:
                                turn_extras["scheduler"] = self._scheduler
                            if self._task_tracker is not None:
                                turn_extras["task_tracker"] = self._task_tracker

                            try:
                                # Collect LLM response text
                                response_text = []
                                async for token in self._orchestrator.stream_turn(
                                    transcript, {}, extras=turn_extras
                                ):
                                    response_text.append(token)
                                    # Send token for display purposes (optional)
                                    await self._write_message(w, {
                                        "type": MSG_TYPE_LLM_TOKEN,
                                        "payload": {
                                            "turn_id": turn_id,
                                            "text": token,
                                            "is_last": False,
                                        },
                                    })

                                # ── TTS: speak directly via Microsoft Edge-TTS + afplay ──
                                # The Core runs locally on the user's Mac, so we
                                # speak out loud directly. We synthesise speech with
                                # edge-tts (hi-IN-MadhurNeural) into a temp .mp3 file
                                # then play it with `afplay -v` at a boosted volume
                                # so HAKI is clearly audible even while the mic's
                                # voice-processing unit ducks other audio.
                                full_text = " ".join(response_text)
                                logger.info("JSONIPCServer: speaking response: %r", full_text[:100])
                                # afplay -v is a volume multiplier; the old 4.0
                                # was so loud it defeated the mic's echo
                                # cancellation and made HAKI hear itself. Default
                                # to a saner 2.0, overridable via HAKI_TTS_VOLUME.
                                _tts_vol = os.environ.get("HAKI_TTS_VOLUME", "2.0")
                                # Tell the Swift shell we are speaking so it can
                                # arm barge-in detection and stop treating HAKI's
                                # own voice as a new user turn.
                                await self._write_message(w, {
                                    "type": MSG_TYPE_CONTROL_EVENT,
                                    "payload": {"event_type": "SPEAKING_STARTED", "turn_id": turn_id},
                                })
                                try:
                                    import tempfile as _tf
                                    import edge_tts
                                    from core.model_provider.tts_engine import (
                                        get_edge_voice_settings,
                                    )

                                    _voice, _rate, _pitch = get_edge_voice_settings()
                                    _mp3 = _tf.NamedTemporaryFile(suffix=".mp3", delete=False).name
                                    communicate = edge_tts.Communicate(
                                        text=full_text,
                                        voice=_voice,
                                        rate=_rate,
                                        pitch=_pitch,
                                    )
                                    await communicate.save(_mp3)
                                    # afplay -v is a volume multiplier; >1.0 amplifies
                                    # to overcome voice-processing ducking. afplay
                                    # plays mp3 natively.
                                    play_proc = await asyncio.create_subprocess_exec(
                                        "afplay", "-v", _tts_vol, _mp3,
                                        stdout=asyncio.subprocess.DEVNULL,
                                        stderr=asyncio.subprocess.DEVNULL,
                                    )
                                    speaking_state["proc"] = play_proc
                                    try:
                                        await play_proc.wait()
                                    except asyncio.CancelledError:
                                        # Barge-in / cancel: stop the audio NOW.
                                        if play_proc.returncode is None:
                                            try:
                                                play_proc.terminate()
                                            except ProcessLookupError:
                                                pass
                                        raise
                                    finally:
                                        speaking_state["proc"] = None
                                    try:
                                        os.unlink(_mp3)
                                    except OSError:
                                        pass
                                    logger.info("JSONIPCServer: finished speaking")
                                except Exception as tts_exc:
                                    logger.exception("JSONIPCServer: edge-tts TTS failed: %s — falling back to `say`", tts_exc)
                                    # Fall back to macOS `say` + afplay so audio
                                    # never fully breaks if edge-tts is unavailable.
                                    try:
                                        import tempfile as _tf
                                        _aiff = _tf.NamedTemporaryFile(suffix=".aiff", delete=False).name
                                        say_proc = await asyncio.create_subprocess_exec(
                                            "say", "-r", "188", "-o", _aiff, full_text,
                                            stdout=asyncio.subprocess.DEVNULL,
                                            stderr=asyncio.subprocess.DEVNULL,
                                        )
                                        await say_proc.wait()
                                        play_proc = await asyncio.create_subprocess_exec(
                                            "afplay", "-v", _tts_vol, _aiff,
                                            stdout=asyncio.subprocess.DEVNULL,
                                            stderr=asyncio.subprocess.DEVNULL,
                                        )
                                        speaking_state["proc"] = play_proc
                                        try:
                                            await play_proc.wait()
                                        except asyncio.CancelledError:
                                            if play_proc.returncode is None:
                                                try:
                                                    play_proc.terminate()
                                                except ProcessLookupError:
                                                    pass
                                            raise
                                        finally:
                                            speaking_state["proc"] = None
                                        try:
                                            os.unlink(_aiff)
                                        except OSError:
                                            pass
                                        logger.info("JSONIPCServer: finished speaking (say fallback)")
                                    except Exception as say_exc:
                                        logger.exception("JSONIPCServer: `say` fallback TTS failed: %s", say_exc)

                                # Tell the Swift shell we have stopped speaking.
                                await self._write_message(w, {
                                    "type": MSG_TYPE_CONTROL_EVENT,
                                    "payload": {"event_type": "SPEAKING_STOPPED", "turn_id": turn_id},
                                })

                                await self._write_message(w, {
                                    "type": MSG_TYPE_CONTROL_EVENT,
                                    "payload": {
                                        "event_type": "TURN_COMPLETE",
                                        "turn_id": turn_id,
                                    },
                                })
                            except asyncio.CancelledError:
                                logger.debug("JSONIPCServer: turn cancelled")
                            except Exception as exc:
                                logger.exception("JSONIPCServer: LLM error: %s", exc)
                                await self._write_message(w, {
                                    "type": MSG_TYPE_ERROR,
                                    "payload": {"message": str(exc)},
                                })

                        speaking_state["active"] = True
                        active_turn_task = asyncio.ensure_future(
                            _stt_then_llm_then_tts(frames_bytes, audio_sample_rate, writer)
                        )
                        # Clear the active flag the moment the turn finishes
                        # (success, error, or cancellation) so the next utterance
                        # can be processed.
                        active_turn_task.add_done_callback(
                            lambda _t: speaking_state.__setitem__("active", False)
                        )
                        if self._orchestrator is not None:
                            self._orchestrator.set_current_task(active_turn_task)
                        continue

                    elif event_type in ("CANCEL", "BARGE_IN"):
                        if self._orchestrator is not None:
                            self._orchestrator.cancel()
                        # Kill any audio that is currently playing so HAKI goes
                        # quiet immediately and listens (true barge-in).
                        _kill_playback()
                        if active_turn_task and not active_turn_task.done():
                            active_turn_task.cancel()
                        speaking_state["active"] = False
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
            _kill_playback()
            if active_turn_task and not active_turn_task.done():
                active_turn_task.cancel()
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
            # END_OF_SPEECH forwarded to pipeline in later tasks
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
