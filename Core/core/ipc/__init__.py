"""
IPC sub-package.

Provides the gRPC / JSON-RPC server stub and UNIX domain socket server
that the Swift shell connects to for bidirectional streaming of audio
frames, partial/final transcripts, LLM tokens, TTS audio chunks, and
control/cancel events.

Design reference: Process & Threading Model, Architecture.
Requirements: 3.1, 1.4.
"""

from .server import IPCServer, JSONIPCServer

__all__ = ["IPCServer", "JSONIPCServer"]
