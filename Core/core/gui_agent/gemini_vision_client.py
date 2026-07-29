"""
gemini_vision_client — Data models, error types, and GeminiVisionClient.

This module defines the core data structures used by the Gemini-Sidecar GUI
agent to represent bounding boxes, pixel coordinates, agent actions, and
retriable API errors. It also implements GeminiVisionClient which connects to
the Swift sidecar socket and queries Gemini 2.5 Flash for next-action decisions.

Requirements: 3.2, 3.3, 3.4, 3.6, 10.5
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
from dataclasses import dataclass

__all__ = [
    "BoundingBox",
    "NativePixelBox",
    "GeminiAction",
    "GeminiAPIError",
    "GeminiVisionClient",
]


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Normalized Gemini bounding box in 0–1000 integer space.

    Coordinates are returned by the Gemini 2.5 Flash API in the format
    [ymin, xmin, ymax, xmax] where each value is an integer in [0, 1000].
    """

    ymin: int  # 0–1000
    xmin: int  # 0–1000
    ymax: int  # 0–1000
    xmax: int  # 0–1000


@dataclass(frozen=True, slots=True)
class NativePixelBox:
    """Pixel coordinates in NativeResolution (2560×1600) space.

    Derived from a BoundingBox by multiplying each component by
    NativeResolution / 1000 (see GeminiVisionClient.bbox_to_native).
    """

    ymin: int
    xmin: int
    ymax: int
    xmax: int


@dataclass(frozen=True, slots=True)
class GeminiAction:
    """A single next-action decision returned by the Gemini 2.5 Flash model.

    Attributes:
        action_type: The type of action to perform, e.g. "click", "type",
                     "scroll", or "done".
        bbox:        Target area in NativePixelBox (pixel) coordinates.
        text:        Populated for "type" actions; the text to type.
        summary:     Populated for "done" actions; a brief completion summary.
    """

    action_type: str
    bbox: NativePixelBox
    text: str | None = None
    summary: str | None = None


class GeminiAPIError(Exception):
    """Retriable error raised when the Gemini API returns a non-200 status or
    a network timeout occurs.

    Attributes:
        status_code: The HTTP status code returned by the API, or ``None`` for
                     network-level errors (e.g. timeouts).
        retriable:   Always ``True``; signals to SidecarAgentLoop that the
                     current cognitive step may be retried.

    Requirements: 3.4
    """

    def __init__(self, status_code: int | None, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retriable = True


class GeminiVisionClient:
    """Client that fetches screen frames from the Swift sidecar and queries
    Gemini 2.5 Flash for next-action decisions.

    The Gemini API key is read exclusively from the ``HAKI_GEMINI_API_KEY``
    environment variable at instantiation time — it is never accepted as a
    function argument (Requirements 3.2, 10.5).

    Class-level constants
    ---------------------
    NATIVE_WIDTH : int
        Physical pixel width of the target display (2560).
    NATIVE_HEIGHT : int
        Physical pixel height of the target display (1600).
    SOCKET_PATH : str
        UNIX domain socket path for the Swift sidecar.
    GEMINI_MODEL : str
        Gemini model identifier used for all API calls.
    TIMEOUT_SECONDS : int
        Per-request timeout for the Gemini API (15 seconds).
    """

    NATIVE_WIDTH: int = 2560
    NATIVE_HEIGHT: int = 1600
    SOCKET_PATH: str = os.path.expanduser("~/.haki/sidecar_frames.sock")
    GEMINI_MODEL: str = "gemini-2.5-flash"
    TIMEOUT_SECONDS: int = 15

    def __init__(self) -> None:
        """Initialise the client.

        Reads ``HAKI_GEMINI_API_KEY`` from the process environment, configures
        the ``google-generativeai`` SDK, and stores the GenerativeModel
        instance.  Raises ``KeyError`` if the environment variable is absent.

        Requirements: 3.2, 10.5
        """
        # API key is read exclusively from environment — never passed as argument
        api_key = os.environ["HAKI_GEMINI_API_KEY"]
        import google.generativeai as genai  # noqa: PLC0415
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(self.GEMINI_MODEL)
        self._display_scale: float | None = None

    async def perform_handshake(self) -> float:
        """Connect to the sidecar UNIX socket and complete the initial handshake.

        The Swift sidecar sends a newline-delimited JSON message immediately
        after a client connects:

            {"display_scale": 2.0, "width": 2560, "height": 1600}\\n

        This method reads that message, stores the ``display_scale`` value on
        the instance for later use, and returns it to the caller.

        Returns:
            The ``display_scale`` float received from the sidecar (e.g. ``2.0``
            on an M2 MacBook Air Retina display).

        Requirements: 2.7, 3.1
        """
        reader, _writer = await asyncio.open_unix_connection(self.SOCKET_PATH)
        try:
            line = await reader.readline()
            handshake = json.loads(line.decode())
            self._display_scale = float(handshake["display_scale"])
            return self._display_scale
        finally:
            _writer.close()
            try:
                await _writer.wait_closed()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass

    async def request_frame(self) -> bytes:
        """Request the latest JPEG frame from the sidecar over the UNIX socket.

        Protocol:
        1. Open a new connection to ``SOCKET_PATH``.
        2. Read and discard the handshake line (required by the sidecar protocol
           before the connection is ready to accept ``REQUEST_FRAME``).
        3. Send ``REQUEST_FRAME\\n``.
        4. Read the 4-byte big-endian unsigned-int length prefix.
        5. Read exactly that many bytes (the JPEG payload) and return them.

        Returns:
            Raw JPEG bytes for the most recently captured frame.

        Requirements: 2.3, 3.1
        """
        reader, writer = await asyncio.open_unix_connection(self.SOCKET_PATH)
        try:
            # Consume the mandatory handshake line before sending commands.
            await reader.readline()

            # Request a frame.
            writer.write(b"REQUEST_FRAME\n")
            await writer.drain()

            # Read the 4-byte big-endian uint32 length prefix.
            length_bytes = await reader.readexactly(4)
            (frame_length,) = struct.unpack(">I", length_bytes)

            # Read the JPEG payload.
            jpeg_bytes = await reader.readexactly(frame_length)
            return jpeg_bytes
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass

    def bbox_to_native(self, bbox: BoundingBox) -> NativePixelBox:
        """Convert a 0–1000 normalised bounding box to 2560×1600 pixel coordinates.

        Each coordinate component is scaled by the corresponding native
        dimension divided by 1000, then truncated to an integer:

            native_coord = int(normalised_coord * dimension / 1000)

        Args:
            bbox: A ``BoundingBox`` with components in the range [0, 1000].

        Returns:
            A ``NativePixelBox`` with pixel coordinates in the 2560×1600 space.

        Requirements: 3.6
        """
        return NativePixelBox(
            ymin=int(bbox.ymin * self.NATIVE_HEIGHT / 1000),
            xmin=int(bbox.xmin * self.NATIVE_WIDTH / 1000),
            ymax=int(bbox.ymax * self.NATIVE_HEIGHT / 1000),
            xmax=int(bbox.xmax * self.NATIVE_WIDTH / 1000),
        )

    async def _call_gemini(
        self, frame_bytes: bytes, task_description: str
    ) -> GeminiAction:
        """Submit a JPEG frame and task description to Gemini and return the action.

        Builds a structured prompt, sends the frame bytes alongside it to the
        ``google-generativeai`` SDK, parses the JSON response, and converts the
        normalised bounding box to native pixel coordinates.

        Args:
            frame_bytes:      Raw JPEG bytes for the current screen frame.
            task_description: Natural-language description of the task to
                              complete.

        Returns:
            A ``GeminiAction`` with native-pixel ``bbox`` coordinates.

        Raises:
            GeminiAPIError: On non-200 HTTP status, network errors, or
                            unparseable responses.

        Requirements: 3.2, 3.3, 3.4
        """
        import google.generativeai as genai  # noqa: PLC0415
        import google.api_core.exceptions as gapi_exceptions  # noqa: PLC0415

        prompt = (
            "You are a GUI automation agent. Given this screenshot, determine "
            "the next action to complete the task.\n\n"
            f"Task: {task_description}\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            "{{\n"
            '  "action_type": "click" | "type" | "scroll" | "done",\n'
            '  "bbox": [ymin, xmin, ymax, xmax],  // integers 0-1000\n'
            '  "text": "text to type",  // only for "type" actions\n'
            '  "summary": "brief summary"  // only for "done" actions\n'
            "}}"
        )

        image_part = {"mime_type": "image/jpeg", "data": frame_bytes}

        try:
            response = await self._model.generate_content_async([image_part, prompt])
        except gapi_exceptions.GoogleAPICallError as exc:
            # Extract HTTP status code when available; fall back to None.
            status_code: int | None = None
            if hasattr(exc, "code") and exc.code is not None:  # type: ignore[union-attr]
                try:
                    status_code = int(exc.code)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    pass
            raise GeminiAPIError(status_code, "Gemini API error") from exc
        except Exception as exc:
            raise GeminiAPIError(None, "Gemini API error") from exc

        # Extract text and strip optional markdown code fences.
        raw_text: str = response.text.strip()
        if raw_text.startswith("```"):
            # Remove opening fence (```json or ```)
            raw_text = raw_text.split("\n", 1)[-1]
            # Remove closing fence
            if raw_text.endswith("```"):
                raw_text = raw_text[: raw_text.rfind("```")]

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise GeminiAPIError(None, "Gemini API error") from exc

        try:
            ymin, xmin, ymax, xmax = data["bbox"]
            bbox = BoundingBox(ymin=int(ymin), xmin=int(xmin), ymax=int(ymax), xmax=int(xmax))
            native_box = self.bbox_to_native(bbox)
            return GeminiAction(
                action_type=data["action_type"],
                bbox=native_box,
                text=data.get("text"),
                summary=data.get("summary"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GeminiAPIError(None, "Gemini API error") from exc

    async def query_gemini(self, task_description: str) -> GeminiAction:
        """Fetch the current screen frame and query Gemini for the next action.

        Fetches a JPEG frame from the sidecar, submits it to Gemini via
        ``_call_gemini``, and returns the parsed action.  The frame bytes are
        explicitly deleted in a ``finally`` block regardless of success or
        failure to comply with the in-memory lifecycle requirement.

        Args:
            task_description: Natural-language description of the task to
                              complete.

        Returns:
            A ``GeminiAction`` describing the next step.

        Raises:
            GeminiAPIError: On timeout (``status_code=None``) or any API /
                            network error surfaced by ``_call_gemini``.

        Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 10.1
        """
        frame_bytes = await self.request_frame()
        try:
            return await asyncio.wait_for(
                self._call_gemini(frame_bytes, task_description),
                timeout=self.TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise GeminiAPIError(None, "Gemini API timeout") from exc
        except GeminiAPIError:
            raise
        finally:
            del frame_bytes  # Req 10.1 — explicit in-scope deletion
