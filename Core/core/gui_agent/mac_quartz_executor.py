"""
MacQuartzExecutor — HID event dispatch via pyobjc-framework-Quartz.

Dispatches physical mouse and keyboard events using CGEvent APIs with
runtime display-scale correction for accurate Retina coordinate mapping.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["MacQuartzExecutor", "ExecutorUnavailableError"]

if TYPE_CHECKING:
    from .gemini_vision_client import NativePixelBox


class ExecutorUnavailableError(RuntimeError):
    """Raised at instantiation when pyobjc-framework-Quartz is not importable."""


class MacQuartzExecutor:
    """
    Dispatches physical HID events via pyobjc-framework-Quartz.

    Converts NativeCoordinates (raw Retina pixels) to LogicalCoordinates
    (macOS points) at runtime using the live display scale factor queried
    from Quartz, then posts CGEvents to the HID event tap.

    Raises:
        ExecutorUnavailableError: At instantiation if pyobjc-framework-Quartz
            is not installed in the current environment.
    """

    def __init__(self) -> None:
        try:
            import Quartz  # noqa: F401
            self._Quartz = Quartz
        except ImportError as exc:
            raise ExecutorUnavailableError(
                "pyobjc-framework-Quartz is not installed. "
                "Install with: pip install pyobjc-framework-Quartz"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def click(self, native_box: "NativePixelBox") -> None:
        """
        Click the center of *native_box*.

        Computes the center of the bounding box in native pixel space,
        queries the runtime display scale, converts to logical coordinates,
        then dispatches a left-mouse-down / left-mouse-up pair.

        Args:
            native_box: Bounding box in native (Retina) pixel coordinates.
        """
        # Compute center in native (Retina) pixel space
        cx_native = (native_box.xmin + native_box.xmax) / 2
        cy_native = (native_box.ymin + native_box.ymax) / 2

        # Query display scale at runtime (Req 4.2, 4.3)
        scale = self._get_display_scale()

        # Convert to logical (points) coordinate space
        lx, ly = self._to_logical(cx_native, cy_native, scale)

        # Dispatch the HID mouse events
        self._post_click(lx, ly)

    def type_text(self, text: str) -> None:
        """
        Type *text* by dispatching a CGEventCreateKeyboardEvent per character.

        For each character a key-down and key-up event are created, the Unicode
        string is set on each event via ``CGEventKeyboardSetUnicodeString``, and
        both events are posted to ``kCGHIDEventTap`` (Req 4.5, 4.7).

        Args:
            text: The string to type into the currently focused UI element.
        """
        Q = self._Quartz

        for char in text:
            # virtual key code 0 — actual key routing is driven by the Unicode
            # string we set on the event rather than the keycode
            key_down = Q.CGEventCreateKeyboardEvent(None, 0, True)
            Q.CGEventKeyboardSetUnicodeString(key_down, len(char), char)
            Q.CGEventPost(Q.kCGHIDEventTap, key_down)

            key_up = Q.CGEventCreateKeyboardEvent(None, 0, False)
            Q.CGEventKeyboardSetUnicodeString(key_up, len(char), char)
            Q.CGEventPost(Q.kCGHIDEventTap, key_up)

    def scroll(
        self,
        native_box: "NativePixelBox",
        direction: str,
        amount: int,
    ) -> None:
        """
        Scroll within the area described by *native_box*.

        Moves the cursor to the logical center of *native_box* first, then
        posts a ``CGEventCreateScrollWheelEvent`` with the appropriate axis
        delta (Req 4.7).

        Args:
            native_box: Target area in native pixel coordinates.
            direction: One of ``"up"``, ``"down"``, ``"left"``, ``"right"``.
            amount: Number of scroll line units to apply.
        """
        Q = self._Quartz

        # Compute logical center of the target area
        cx_native = (native_box.xmin + native_box.xmax) / 2
        cy_native = (native_box.ymin + native_box.ymax) / 2
        scale = self._get_display_scale()
        lx, ly = self._to_logical(cx_native, cy_native, scale)

        # Move mouse to the scroll target before scrolling so the OS delivers
        # the wheel event to the correct window
        point = Q.CGPointMake(lx, ly)
        move_event = Q.CGEventCreateMouseEvent(
            None,
            Q.kCGEventMouseMoved,
            point,
            Q.kCGMouseButtonLeft,
        )
        Q.CGEventPost(Q.kCGHIDEventTap, move_event)

        # Map direction to (scroll_y, scroll_x) deltas.
        # kCGScrollEventUnitLine / axis1 = vertical, axis2 = horizontal.
        # Positive axis1 scrolls up; positive axis2 scrolls left.
        direction_map = {
            "up":    ( amount, 0),
            "down":  (-amount, 0),
            "left":  (0,  amount),
            "right": (0, -amount),
        }
        scroll_y, scroll_x = direction_map.get(direction, (0, 0))

        scroll_event = Q.CGEventCreateScrollWheelEvent(
            None,
            Q.kCGScrollEventUnitLine,
            2,          # number of scroll axes
            scroll_y,   # axis 1 — vertical
            scroll_x,   # axis 2 — horizontal
        )
        Q.CGEventPost(Q.kCGHIDEventTap, scroll_event)

    def dispatch(self, action: "GeminiAction") -> None:
        """
        Route a ``GeminiAction`` to the appropriate low-level HID method.

        Dispatches based on ``action.action_type``:
        - ``"click"``  → ``click(action.bbox)``
        - ``"type"``   → ``type_text(action.text or "")``
        - ``"scroll"`` → ``scroll(action.bbox, direction="down", amount=3)``
        - ``"done"``   → no physical action (loop will terminate after this)
        - anything else → logs a warning and skips

        Args:
            action: The structured action returned by GeminiVisionClient.

        Requirements: 4.1, 4.4, 4.5
        """
        import logging as _logging
        _log = _logging.getLogger(__name__)

        if action.action_type == "click":
            self.click(action.bbox)
        elif action.action_type == "type":
            self.type_text(action.text or "")
        elif action.action_type == "scroll":
            self.scroll(action.bbox, direction="down", amount=3)
        elif action.action_type == "done":
            pass  # no physical action needed; loop handles termination
        else:
            _log.warning(
                "MacQuartzExecutor.dispatch: unknown action_type %r — skipping dispatch",
                action.action_type,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _post_click(self, lx: float, ly: float) -> None:
        """
        Post a kCGEventLeftMouseDown then kCGEventLeftMouseUp at *(lx, ly)*.

        Both events are posted to kCGHIDEventTap (Req 4.1, 4.4, 4.7).

        Args:
            lx: Logical (points) x coordinate.
            ly: Logical (points) y coordinate.
        """
        Q = self._Quartz
        point = Q.CGPointMake(lx, ly)

        # Create and post mouse-down event
        down = Q.CGEventCreateMouseEvent(
            None,
            Q.kCGEventLeftMouseDown,
            point,
            Q.kCGMouseButtonLeft,
        )
        Q.CGEventPost(Q.kCGHIDEventTap, down)

        # Create and post mouse-up event immediately after
        up = Q.CGEventCreateMouseEvent(
            None,
            Q.kCGEventLeftMouseUp,
            point,
            Q.kCGMouseButtonLeft,
        )
        Q.CGEventPost(Q.kCGHIDEventTap, up)

    def _get_display_scale(self) -> float:
        """
        Query the runtime display scale factor for the main display.

        Uses ``CGMainDisplayID``, ``CGDisplayScreenSize``, and
        ``CGDisplayBounds`` to derive the scale.  Falls back to ``2.0``
        if the physical width cannot be determined.

        Returns:
            The display scale factor (typically ``2.0`` on Retina displays).
        """
        Q = self._Quartz
        display_id = Q.CGMainDisplayID()
        phys = Q.CGDisplayScreenSize(display_id)   # CGSize in mm
        pix = Q.CGDisplayBounds(display_id)        # CGRect in logical points
        if phys.width == 0:
            return 2.0
        native_pixel_width = float(Q.CGDisplayPixelsWide(display_id))
        logical_point_width = float(pix.size.width)
        if logical_point_width == 0:
            return 2.0
        return native_pixel_width / logical_point_width

    @staticmethod
    def _to_logical(
        cx_native: float,
        cy_native: float,
        display_scale: float,
    ) -> tuple[float, float]:
        """
        Convert native pixel coordinates to logical (points) coordinates.

        Formula: ``logical = native / display_scale``

        Args:
            cx_native: X coordinate in native pixels.
            cy_native: Y coordinate in native pixels.
            display_scale: Runtime display scale factor.

        Returns:
            ``(lx, ly)`` in logical point space.
        """
        return (cx_native / display_scale, cy_native / display_scale)
