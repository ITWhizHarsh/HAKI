#!/usr/bin/env python3
"""
haki_core_service.py — HAKI Core entry-point script.

Usage
-----
    python haki_core_service.py --socket <path> [--transport grpc|json]

The Swift CoreProcessManager spawns this script as a child process, passing
the UNIX domain socket path via --socket.  Output is written to stderr so the
parent process can capture it via the Process pipe.

Lifecycle
---------
1. Parse CLI arguments (--socket, --transport).
2. Instantiate the chosen transport server (JSONIPCServer by default).
3. Register SIGTERM / SIGINT handlers for graceful shutdown.
4. Start the server and block in ``serve_forever()`` until a signal arrives.
5. On signal: call ``server.stop(grace=5.0)`` then exit cleanly.

Design: Architecture, Security Considerations (local IPC only).
Requirements: 3.1
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

# override=True makes Core/.env the SINGLE source of truth for keys, even when
# a stale value (e.g. an old AIza... Gemini key) is already exported in the
# environment the Core process inherits from the launcher. Without this, a
# stale exported key silently wins over the value in Core/.env.
load_dotenv(override=True)

import argparse
import asyncio
import logging
import os
import signal
import sys

# ---------------------------------------------------------------------------
# Logging — to stderr so the Swift shell can pipe it
# ---------------------------------------------------------------------------
# Default to INFO for a clean, readable log (what the user said, what HAKI
# heard, and the key API operations). Set HAKI_DEBUG=1 for full verbose output.
_log_level = logging.DEBUG if os.environ.get("HAKI_DEBUG") == "1" else logging.INFO
logging.basicConfig(
    stream=sys.stderr,
    level=_log_level,
    format="[HAKI Core %(levelname)s] %(asctime)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("haki_core_service")

# Silence noisy third-party libraries unless HAKI_DEBUG is set. These flood the
# terminal with per-request HTTP/websocket frame logging that hides the lines
# the user actually cares about.
if os.environ.get("HAKI_DEBUG") != "1":
    for _noisy in (
        "httpx", "httpcore", "httpcore.connection", "httpcore.http11",
        "groq", "groq._base_client", "openai", "websockets", "websockets.client",
        "urllib3", "urllib3.connectionpool", "chromadb", "chromadb.config",
        "deepgram", "sentence_transformers", "asyncio", "filelock",
        "huggingface_hub", "transformers",
    ):
        logging.getLogger(_noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(name: str, default: int, min_val: int = 1, max_val: int = 1440) -> int:
    """Read an integer from the environment with range validation.

    Returns *default* when the variable is unset, non-integer, or out of the
    [min_val, max_val] range.  Logs an error for invalid/out-of-range values so
    operators know their override was ignored.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (ValueError, TypeError):
        logger.error(
            "Environment variable %s=%r is not a valid integer — using default %d",
            name, raw, default,
        )
        return default
    if value < min_val or value > max_val:
        logger.error(
            "Environment variable %s=%d is out of range [%d, %d] — using default %d",
            name, value, min_val, max_val, default,
        )
        return default
    return value


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="haki_core_service",
        description="HAKI Core — local orchestration service",
    )
    parser.add_argument(
        "--socket",
        required=True,
        metavar="PATH",
        help="UNIX domain socket path the IPC server will listen on",
    )
    parser.add_argument(
        "--transport",
        choices=["grpc", "json"],
        default="json",
        help=(
            "IPC transport to use.  "
            "'json' (default) uses a simple JSON-over-UNIX-socket transport. "
            "'grpc' uses the full gRPC transport (requires grpc-swift on the client)."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Async main
# ---------------------------------------------------------------------------

async def _run(socket_path: str, transport: str) -> None:
    """Start the IPC server and block until a stop signal is received."""

    import os
    from pathlib import Path
    from core.memory import MemoryBrain, HAKIBrain
    from core.orchestrator import Orchestrator
    from core.scheduler import Scheduler, create_sqlite_task_tracker
    from core.model_provider import (
        LLMRouter, LLMRouterConfig,
        EmbeddingsEngine,
    )
    from core.voice.cloud_gate import CloudEscalationGate
    from core.ipc.voice_unix_server import start_replacement_session

    # ----------------------------------------------------------------
    # Read env vars — see Misc/api_keys_setup.md for the full list
    # ----------------------------------------------------------------
    groq_key      = os.environ.get("HAKI_GROQ_API_KEY")
    cerebras_key  = os.environ.get("HAKI_CEREBRAS_API_KEY")
    gemini_key    = os.environ.get("HAKI_GEMINI_API_KEY")

    if not any([groq_key, cerebras_key, gemini_key]):
        logger.warning(
            "No cloud LLM keys found. Set HAKI_GROQ_API_KEY / HAKI_CEREBRAS_API_KEY "
            "/ HAKI_GEMINI_API_KEY. Falling back to local MLX models only."
        )

    # ----------------------------------------------------------------
    # LLM Router  Groq → Cerebras → Gemini → local MLX  (non-voice only)
    # Live voice uses VoiceLLMRouter / VoiceLocalMLXService, not this router.
    # ----------------------------------------------------------------
    llm_router = LLMRouter(
        config=LLMRouterConfig(
            groq_api_key=groq_key,
            cerebras_api_key=cerebras_key,
            gemini_api_key=gemini_key,
        )
    )
    logger.info(
        "LLMRouter ready (groq=%s cerebras=%s gemini=%s)",
        bool(groq_key), bool(cerebras_key), bool(gemini_key),
    )

    # ----------------------------------------------------------------
    # CloudEscalationGate — initialised with Gemini disabled for every
    # new voice session (Req 8.1).  The gate is passed into the voice
    # session pipeline; no voice session starts with Gemini enabled.
    # ----------------------------------------------------------------
    cloud_escalation_gate = CloudEscalationGate()
    logger.info("CloudEscalationGate ready (Gemini disabled by default for every session)")

    # ----------------------------------------------------------------
    # Embeddings  Granite ModernBERT 32k → Gemini Embed API fallback
    # ----------------------------------------------------------------
    chroma_dir = Path.home() / ".haki" / "chroma_db"
    embeddings_engine = EmbeddingsEngine(
        gemini_api_key=gemini_key,
        chroma_persist_dir=chroma_dir,
    )
    logger.info("EmbeddingsEngine ready (chroma=%s)", chroma_dir)

    # ----------------------------------------------------------------
    # Initialise Memory_Brain (existing vault-based note store)
    # ----------------------------------------------------------------
    vault_path = Path.home() / ".haki" / "vault"
    memory_brain = MemoryBrain(
        vault_path=vault_path,
        embeddings_provider=embeddings_engine,
    )
    memory_brain.init()
    logger.info("MemoryBrain initialised (vault=%s)", vault_path)

    # ----------------------------------------------------------------
    # Initialise HAKI Brain (LLM Wiki — 3-folder Obsidian pipeline)
    # ----------------------------------------------------------------
    obsidian_root = Path(
        os.environ.get("HAKI_OBSIDIAN_VAULT",
                       str(Path.home() / "Obsidian" / "HAKI_Brain"))
    )
    haki_brain = HAKIBrain(
        obsidian_vault_path=obsidian_root,
        llm_router=llm_router,
        embeddings_engine=embeddings_engine,
        auto_watch_interval=30.0,
    )
    haki_brain.init()

    # --- PipelineScheduler replaces the old start_watching() approach ---
    from core.memory.pipeline_scheduler import PipelineScheduler

    raw_interval = _safe_int("HAKI_PIPELINE_RAW_INTERVAL_MINUTES", default=30, min_val=1, max_val=1440)
    conv_time = os.environ.get("HAKI_PIPELINE_CONV_RUN_TIME", "02:00").strip()

    pipeline_scheduler = PipelineScheduler(
        haki_brain=haki_brain,
        raw_interval_minutes=raw_interval,
        conv_run_time=conv_time,
    )

    if haki_brain._vault_valid:
        pipeline_scheduler.start()
        logger.info("PipelineScheduler started (raw every %d min, conv at %s)", raw_interval, conv_time)
    else:
        logger.warning("PipelineScheduler NOT started — vault path validation failed")

    logger.info("HAKIBrain initialised (vault=%s)", obsidian_root)

    # Ingest any files already sitting in raw/ from before startup
    if haki_brain._vault_valid:
        pending = await haki_brain.ingest_pending()
        if pending:
            logger.info("HAKIBrain: ingested %d pending file(s) at startup", len(pending))

    # ----------------------------------------------------------------
    # Initialise Orchestrator (non-voice turns only)
    # ----------------------------------------------------------------
    orchestrator = Orchestrator(
        memory_brain=memory_brain,
        llm_router=llm_router,
        haki_brain=haki_brain,
    )
    logger.info("Orchestrator created (non-voice, stt_engine/tts_engine removed)")

    # Seed cross-session memory from the Obsidian conversation logs
    try:
        loaded = orchestrator.load_persistent_history()
        if loaded:
            logger.info("Orchestrator: restored %d message(s) of prior conversation", loaded)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Orchestrator: failed to restore conversation history: %r", exc)

    # ----------------------------------------------------------------
    # Initialise Scheduler and TaskTracker (Phase 5 subsystems)
    # ----------------------------------------------------------------
    scheduler = Scheduler()
    task_tracker = create_sqlite_task_tracker(
        db_path=Path.home() / ".haki" / "tasks.db",
    )
    logger.info("Scheduler and TaskTracker initialised")

    if transport == "grpc":
        from core.ipc import IPCServer
        server: IPCServer | object = IPCServer(
            socket_path=socket_path,
            orchestrator=orchestrator,
            scheduler=scheduler,
            task_tracker=task_tracker,
        )
        logger.info("Starting gRPC IPC server on unix:%s", socket_path)
    else:
        from core.ipc import JSONIPCServer
        server = JSONIPCServer(
            socket_path=socket_path,
            orchestrator=orchestrator,
            scheduler=scheduler,
            task_tracker=task_tracker,
        )
        logger.info("Starting JSON IPC server on unix:%s", socket_path)

    # Expose the JSONIPCServer instance for GUI agent wiring (Req 5.1, 7.3, 8.1).
    # Only JSONIPCServer supports broadcast_agent_event; gRPC server is left unwired.
    _json_ipc_server = server if transport == "json" else None

    # Install signal handlers that request graceful shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_stop(sig_name: str) -> None:
        logger.info("Received %s — requesting graceful shutdown", sig_name)
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _request_stop, sig.name)

    # Start the non-voice IPC server
    await server.start()  # type: ignore[attr-defined]
    logger.info("HAKI Core started (transport=%s, socket=%s)", transport, socket_path)

    # ----------------------------------------------------------------
    # Start the replacement voice session (production route).
    # VoiceUnixServer + VoiceSessionPipeline is the live voice path.
    # CloudEscalationGate initialises Gemini disabled for every session
    # (Req 8.1).  Failures are reported directly without legacy fallback.
    # ----------------------------------------------------------------
    voice_handle = None
    try:
        # Monkeypatch: start_replacement_session normally checks the dev gate;
        # for the production cutover we call the components directly without
        # the gate env-var requirement.
        from core.voice.session import VoiceSession
        from core.voice.pipeline import (
            VoiceSessionPipeline,
            VoiceIngressProcessors,
            PipecatFrameAdapter,
            VoicePipelineSinks,
        )
        from core.voice.asr_bridge import AuthenticatedRingSlotReader, RingSlotDescriptor
        from core.ipc.voice_unix_server import (
            VoiceUnixServer,
            ReplacementSessionHandle,
        )
        from uuid import uuid4 as _uuid4

        _session_id = _uuid4()

        class _NullRingSlotReader:
            def __init__(self) -> None:
                self.session_id = _session_id

            async def map_slot(self, descriptor: "RingSlotDescriptor") -> bytes:
                return b""

            async def release_slot(self, descriptor: "RingSlotDescriptor") -> None:
                pass

        _session = VoiceSession(_session_id)
        # Register the session with Gemini disabled (Req 8.1)
        cloud_escalation_gate.register_session(_session_id)

        _server_voice = VoiceUnixServer(
            session_id=_session_id,
            on_message=None,
            on_turn_discarded=None,
        )
        _frame_adapter = PipecatFrameAdapter()
        _ring_reader = _NullRingSlotReader()
        _ingress = VoiceIngressProcessors(
            session=_session,
            ring_reader=_ring_reader,
            frame_adapter=_frame_adapter,
        )
        _pipeline = VoiceSessionPipeline(
            session=_session,
            ingress=_ingress,
            sinks=VoicePipelineSinks(),
        )

        async def _on_transcript_message(validated: object) -> None:
            try:
                await _pipeline.ingest_transcript_message(validated)
            except Exception:
                logger.debug("voice_session: transcript ingress error", exc_info=True)

        _server_voice._on_message = _on_transcript_message  # type: ignore[assignment]

        # Wire the JSONIPCServer into the VoiceToolAdapter so gui_agent.spawn
        # calls can broadcast AGENT_EVENTs and launch SidecarAgentLoop threads
        # (Req 5.1, 7.3, 8.1, 13.1).
        if _json_ipc_server is not None:
            try:
                from core.voice.tools import VoiceToolAdapter  # noqa: PLC0415
                _tool_adapter = VoiceToolAdapter(ipc_server=_json_ipc_server)
                _pipeline._tool_adapter = _tool_adapter
                _pipeline.wire_ipc_server(_json_ipc_server)
                logger.info("haki_core_service: wired JSONIPCServer into VoiceToolAdapter")
            except Exception as exc:
                logger.debug("haki_core_service: could not wire ipc_server into tool adapter: %s", exc)

        await _server_voice.start()
        try:
            await _pipeline.start()
        except Exception as exc:
            await _server_voice.stop()
            logger.error(
                "HAKI voice session pipeline failed to start: %s — voice unavailable",
                exc,
            )
        else:
            voice_handle = ReplacementSessionHandle(
                server=_server_voice,
                session=_session,
                pipeline=_pipeline,
            )
            logger.info(
                "VoiceUnixServer started (session=%s socket=%s)",
                _session_id,
                _server_voice.socket_path,
            )
    except Exception as exc:
        logger.error(
            "HAKI voice session could not start: %s — voice unavailable (no legacy fallback)",
            exc,
        )

    # Check locally provisioned replacement-voice prerequisites after the IPC
    # server is listening.
    from core.voice.resources import run_startup_voice_health_check

    voice_health_task = asyncio.create_task(
        run_startup_voice_health_check(logger=logger),
        name="haki-local-voice-availability",
    )

    # Pre-warm the embedding model in the background
    async def _prewarm_embeddings() -> None:
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="haki_embed_prewarm") as ex:
                await loop.run_in_executor(ex, embeddings_engine._embed_fn._local_embed, ["HAKI prewarm"])
            embeddings_engine._embed_fn._model_ready = True
            logger.info("EmbeddingsEngine: model pre-warm complete")
        except Exception as exc:
            logger.warning("EmbeddingsEngine: pre-warm failed: %s", exc)

    asyncio.ensure_future(_prewarm_embeddings())

    # Block until a stop signal arrives
    await stop_event.wait()

    # Graceful shutdown
    logger.info("Stopping HAKI Core…")
    if not voice_health_task.done():
        voice_health_task.cancel()
        try:
            await voice_health_task
        except asyncio.CancelledError:
            pass
    if voice_handle is not None:
        try:
            await voice_handle.shutdown()
            logger.info("VoiceSessionPipeline stopped cleanly")
        except Exception as exc:
            logger.warning("Voice session shutdown raised: %s", exc)
        if hasattr(voice_handle, 'session'):
            try:
                cloud_escalation_gate.end_session(voice_handle.session.session_id)
            except Exception:
                pass
    pipeline_scheduler.stop()
    await server.stop(grace=5.0)  # type: ignore[attr-defined]
    logger.info("HAKI Core stopped cleanly")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        asyncio.run(_run(socket_path=args.socket, transport=args.transport))
    except KeyboardInterrupt:
        # asyncio.run() may re-raise KeyboardInterrupt on SIGINT; handle cleanly
        logger.info("HAKI Core interrupted — exiting")
        sys.exit(0)


if __name__ == "__main__":
    main()
