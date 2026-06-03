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

import argparse
import asyncio
import logging
import signal
import sys

# ---------------------------------------------------------------------------
# Logging — to stderr so the Swift shell can pipe it
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stderr,
    level=logging.DEBUG,
    format="[HAKI Core %(levelname)s] %(asctime)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("haki_core_service")


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

    # ----------------------------------------------------------------
    # Initialise Memory_Brain and Orchestrator (Task 14.3)
    # ----------------------------------------------------------------
    from pathlib import Path
    from core.memory import MemoryBrain
    from core.orchestrator import Orchestrator

    vault_path = Path.home() / ".haki" / "vault"
    # No embeddings_provider: retrieve() returns [] until an API key is
    # configured — identical behaviour to the stub but uses the real class.
    memory_brain = MemoryBrain(vault_path=vault_path)
    memory_brain.init()  # Req 7.4: ensure vault exists at startup
    logger.info("MemoryBrain initialised (vault=%s)", vault_path)

    orchestrator = Orchestrator(memory_brain=memory_brain)
    logger.info("Orchestrator created with real MemoryBrain")

    if transport == "grpc":
        from core.ipc import IPCServer
        server: IPCServer | object = IPCServer(
            socket_path=socket_path,
            orchestrator=orchestrator,
        )
        logger.info("Starting gRPC IPC server on unix:%s", socket_path)
    else:
        from core.ipc import JSONIPCServer
        server = JSONIPCServer(
            socket_path=socket_path,
            orchestrator=orchestrator,
        )
        logger.info("Starting JSON IPC server on unix:%s", socket_path)

    # Install signal handlers that request graceful shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_stop(sig_name: str) -> None:
        logger.info("Received %s — requesting graceful shutdown", sig_name)
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _request_stop, sig.name)

    # Start the server
    await server.start()  # type: ignore[attr-defined]
    logger.info("HAKI Core started (transport=%s, socket=%s)", transport, socket_path)

    # Block until a stop signal arrives
    await stop_event.wait()

    # Graceful shutdown
    logger.info("Stopping HAKI Core…")
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
