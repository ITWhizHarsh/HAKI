"""
core.ipc.proto — Generated gRPC / protobuf stubs for the HAKI IPC contract.

Generated from: proto/haki_ipc.proto
Generator:     grpc_tools.protoc  (grpcio-tools)
Regenerate:    See proto/README.md for the exact commands.

Re-exports the most frequently used classes so callers can do:
    from core.ipc.proto import (
        ClientMessage, ServerMessage,
        AudioFrame, PartialTranscript, AudioFeatures,
        LLMToken, TTSAudioChunk, ControlEvent,
        TurnRequest, TurnResponse,
        HAKICoreServicer, add_HAKICoreServicer_to_server,
    )
"""

# Fix the relative import emitted by protoc so it works as a package.
# The generated haki_ipc_pb2_grpc.py uses `import haki_ipc_pb2` (bare),
# which fails when the module lives inside a package.  We patch the
# sys.modules alias before importing the grpc stub so the bare name resolves.
import sys
from . import haki_ipc_pb2  # noqa: E402

# Register the bare name alias that the generated grpc stub expects
sys.modules.setdefault("haki_ipc_pb2", haki_ipc_pb2)

from . import haki_ipc_pb2_grpc  # noqa: E402

# ---------------------------------------------------------------------------
# Message types (from haki_ipc_pb2)
# ---------------------------------------------------------------------------
AudioFrame        = haki_ipc_pb2.AudioFrame
PartialTranscript = haki_ipc_pb2.PartialTranscript
AudioFeatures     = haki_ipc_pb2.AudioFeatures
LLMToken          = haki_ipc_pb2.LLMToken
TTSAudioChunk     = haki_ipc_pb2.TTSAudioChunk
ControlEvent      = haki_ipc_pb2.ControlEvent
TurnRequest       = haki_ipc_pb2.TurnRequest
TurnResponse      = haki_ipc_pb2.TurnResponse
ClientMessage     = haki_ipc_pb2.ClientMessage
ServerMessage     = haki_ipc_pb2.ServerMessage

# ---------------------------------------------------------------------------
# Service stubs (from haki_ipc_pb2_grpc)
# ---------------------------------------------------------------------------
HAKICoreStub                 = haki_ipc_pb2_grpc.HAKICoreStub
HAKICoreServicer             = haki_ipc_pb2_grpc.HAKICoreServicer
add_HAKICoreServicer_to_server = haki_ipc_pb2_grpc.add_HAKICoreServicer_to_server

__all__ = [
    # messages
    "AudioFrame",
    "PartialTranscript",
    "AudioFeatures",
    "LLMToken",
    "TTSAudioChunk",
    "ControlEvent",
    "TurnRequest",
    "TurnResponse",
    "ClientMessage",
    "ServerMessage",
    # service
    "HAKICoreStub",
    "HAKICoreServicer",
    "add_HAKICoreServicer_to_server",
]
