"""
CDPWebActuator — Chrome DevTools Protocol actuator for Arc/Chromium.

Drives Arc (the Default_Browser) via CDP over a loopback remote-debugging
connection to open tabs, navigate, click elements, and fill form fields.

Security: only connects to 127.0.0.1 — never a remote host.
If the site is unreachable, raises WebsiteUnreachableError so the
ExecutionEngine stops dependents and informs the user (Req 21.13).

Design: Mac_Controller > Arc/web > CDP (Chrome DevTools Protocol 21.5, 21.6).
Requirements: 21.5, 21.6, 21.12, 21.13.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .errors import ElementNotFoundError, WebsiteUnreachableError

logger = logging.getLogger(__name__)

# ── Conditional import of aiohttp (graceful if not installed) ──────────────
try:
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _AIOHTTP_AVAILABLE = False
    aiohttp = None  # type: ignore[assignment]

# Loopback-only remote debugging endpoint (security: never allow remote hosts)
_CDP_HOST = "127.0.0.1"
_CDP_PORT = 9222
_CDP_BASE_URL = f"http://{_CDP_HOST}:{_CDP_PORT}"


# ---------------------------------------------------------------------------
# ActuatorResult
# ---------------------------------------------------------------------------


@dataclass
class ActuatorResult:
    """
    Result of a CDP actuator operation.

    Attributes
    ----------
    success:
        ``True`` when the operation completed without error.
    error_type:
        A machine-readable tag when ``success`` is ``False``.
        One of: ``"unreachable"``, ``"element_not_found"``,
        ``"permission_denied"``, ``"failed"``, or ``None``.
    detail:
        Optional human-readable detail (URL, selector, or error message).
    """

    success: bool
    error_type: str | None = None
    detail: str | None = None

    # ── Convenience factory methods ──────────────────────────────────────

    @classmethod
    def ok(cls, detail: str | None = None) -> "ActuatorResult":
        """Return a successful result."""
        return cls(success=True, detail=detail)

    @classmethod
    def unreachable(cls, url: str) -> "ActuatorResult":
        """Return an unreachable-site result (Req 21.13)."""
        return cls(success=False, error_type="unreachable", detail=url)

    @classmethod
    def element_not_found(cls, selector: str) -> "ActuatorResult":
        """Return an element-not-found result (Req 21.12)."""
        return cls(
            success=False, error_type="element_not_found", detail=selector
        )

    @classmethod
    def permission_denied(cls, detail: str | None = None) -> "ActuatorResult":
        """Return a permission-denied result."""
        return cls(
            success=False, error_type="permission_denied", detail=detail
        )

    @classmethod
    def failed(cls, detail: str | None = None) -> "ActuatorResult":
        """Return a generic failure result."""
        return cls(success=False, error_type="failed", detail=detail)


# ---------------------------------------------------------------------------
# CDPWebActuator
# ---------------------------------------------------------------------------


class CDPWebActuator:
    """
    Arc/Chromium CDP actuator.

    Drives Arc (or any Chromium-based browser exposing a remote-debugging
    port) over CDP to open tabs, navigate, click elements, and fill form
    fields on behalf of the Mac_Controller.

    Security constraint: the remote-debugging endpoint MUST be on the
    loopback address (127.0.0.1).  Attempts to connect to any other host
    are refused.

    Parameters
    ----------
    host:
        The loopback host.  Default: ``127.0.0.1``.  Any non-loopback
        value raises :class:`ValueError`.
    port:
        The Chrome DevTools Protocol port.  Default: 9222 (Arc's default).

    Design: Mac_Controller > Arc/web > Chrome DevTools Protocol.
    Requirements: 21.5 (open tabs), 21.6 (navigate / click / fill),
                  21.12 (element not found), 21.13 (site unreachable).
    """

    # ---- security: allow only loopback hosts --------------------------------
    _ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

    def __init__(
        self,
        host: str = _CDP_HOST,
        port: int = _CDP_PORT,
    ) -> None:
        if host not in self._ALLOWED_HOSTS:
            raise ValueError(
                f"CDPWebActuator only connects to loopback addresses "
                f"({self._ALLOWED_HOSTS}), not '{host}'."
            )
        self._host = host
        self._port = port
        self._base_url = f"http://{host}:{port}"

    # ------------------------------------------------------------------
    # Public API (Req 21.5, 21.6)
    # ------------------------------------------------------------------

    async def open_tabs(self, urls: list[str]) -> ActuatorResult:
        """
        Open each URL as a new tab in Arc.  (Req 21.5)

        Uses ``/json/new?<url>`` endpoint per the DevTools protocol.
        If the browser is not reachable, raises
        :class:`~core.execution.errors.WebsiteUnreachableError` so the
        ExecutionEngine stops dependent steps.

        Parameters
        ----------
        urls:
            List of URLs to open; each becomes a separate new tab.

        Returns
        -------
        ActuatorResult
            ``ok()`` when all tabs were opened successfully; ``unreachable``
            when the browser cannot be contacted; ``failed`` on any other
            error.
        """
        if not urls:
            return ActuatorResult.ok("No URLs provided.")
        if not _AIOHTTP_AVAILABLE:
            return ActuatorResult.failed("aiohttp is not installed.")

        try:
            async with aiohttp.ClientSession() as session:
                opened: list[str] = []
                for url in urls:
                    tab_url = f"{self._base_url}/json/new?{url}"
                    async with session.get(tab_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status != 200:
                            logger.warning(
                                "CDPWebActuator.open_tabs: /json/new returned %d for %s",
                                resp.status,
                                url,
                            )
                            # Non-200 but browser is up — mark individual failure.
                            return ActuatorResult.failed(
                                f"Browser returned HTTP {resp.status} for {url}"
                            )
                        opened.append(url)
                        logger.debug("CDPWebActuator.open_tabs: opened %s", url)
                return ActuatorResult.ok(f"Opened {len(opened)} tab(s).")
        except (aiohttp.ClientConnectionError, aiohttp.ServerConnectionError) as exc:
            # Browser not running or port not open — surface as unreachable.
            raise WebsiteUnreachableError(str(urls)) from exc
        except Exception as exc:
            logger.error("CDPWebActuator.open_tabs: unexpected error: %s", exc)
            return ActuatorResult.failed(str(exc))

    async def navigate(
        self,
        url: str,
        tab_id: str | None = None,
    ) -> ActuatorResult:
        """
        Navigate the active (or specified) tab to *url*.

        Uses the ``Page.navigate`` CDP command.  If the target URL is
        unreachable (network error, DNS failure, etc.), raises
        :class:`~core.execution.errors.WebsiteUnreachableError` so the
        ExecutionEngine propagates the failure to dependent steps (Req 21.13).

        Parameters
        ----------
        url:
            The destination URL.
        tab_id:
            Optional tab ``webSocketDebuggerUrl`` fragment or target ID.
            When ``None``, the first available non-extension tab is used.

        Returns
        -------
        ActuatorResult
            ``ok()`` on success; ``unreachable`` when the browser or the
            target URL cannot be reached; ``failed`` otherwise.
        """
        if not _AIOHTTP_AVAILABLE:
            return ActuatorResult.failed("aiohttp is not installed.")

        try:
            ws_url = await self._resolve_tab_ws_url(tab_id)
        except WebsiteUnreachableError:
            raise
        except Exception as exc:
            return ActuatorResult.failed(f"Could not resolve tab: {exc}")

        try:
            result = await self._send_command(
                ws_url, "Page.navigate", {"url": url}
            )
        except WebsiteUnreachableError:
            raise
        except Exception as exc:
            return ActuatorResult.failed(str(exc))

        # CDP returns errorText on navigation failure (e.g. DNS error).
        error_text = result.get("errorText") if isinstance(result, dict) else None
        if error_text:
            raise WebsiteUnreachableError(url)

        logger.debug("CDPWebActuator.navigate: navigated to %s", url)
        return ActuatorResult.ok(url)

    async def click_element(
        self,
        selector: str,
        tab_id: str | None = None,
    ) -> ActuatorResult:
        """
        Find an element by CSS *selector* in the given tab and click it.

        Uses ``Runtime.evaluate`` to locate the element bounding rect
        and ``Input.dispatchMouseEvent`` to synthesize the click.
        If no element matches *selector*, raises
        :class:`~core.execution.errors.ElementNotFoundError` (Req 21.12).

        Parameters
        ----------
        selector:
            A CSS selector string (e.g. ``"button#submit"``).
        tab_id:
            Optional tab identifier.  When ``None``, uses the first
            available non-extension tab.

        Returns
        -------
        ActuatorResult
            ``ok()`` on success; ``element_not_found`` when the selector
            matches nothing; ``failed`` on other errors.
        """
        if not _AIOHTTP_AVAILABLE:
            return ActuatorResult.failed("aiohttp is not installed.")

        try:
            ws_url = await self._resolve_tab_ws_url(tab_id)
        except WebsiteUnreachableError:
            raise
        except Exception as exc:
            return ActuatorResult.failed(f"Could not resolve tab: {exc}")

        # 1. Locate element and get bounding rect.
        js = (
            f"(function() {{"
            f"  var el = document.querySelector({json.dumps(selector)});"
            f"  if (!el) return null;"
            f"  var r = el.getBoundingClientRect();"
            f"  return {{x: r.left + r.width/2, y: r.top + r.height/2}};"
            f"}})()"
        )
        try:
            eval_result = await self._send_command(
                ws_url, "Runtime.evaluate", {"expression": js, "returnByValue": True}
            )
        except WebsiteUnreachableError:
            raise
        except Exception as exc:
            return ActuatorResult.failed(str(exc))

        coords = (
            eval_result.get("result", {}).get("value")
            if isinstance(eval_result, dict)
            else None
        )
        if not coords:
            raise ElementNotFoundError(selector)

        x, y = float(coords["x"]), float(coords["y"])

        # 2. Dispatch mousePressed + mouseReleased to synthesize click.
        try:
            for event_type in ("mousePressed", "mouseReleased"):
                await self._send_command(
                    ws_url,
                    "Input.dispatchMouseEvent",
                    {
                        "type": event_type,
                        "x": x,
                        "y": y,
                        "button": "left",
                        "clickCount": 1,
                    },
                )
        except WebsiteUnreachableError:
            raise
        except Exception as exc:
            return ActuatorResult.failed(str(exc))

        logger.debug("CDPWebActuator.click_element: clicked '%s' at (%s, %s)", selector, x, y)
        return ActuatorResult.ok(f"Clicked '{selector}'")

    async def fill_field(
        self,
        selector: str,
        value: str,
        tab_id: str | None = None,
    ) -> ActuatorResult:
        """
        Focus a form field identified by CSS *selector* and type *value*.

        Focuses the element via ``Runtime.evaluate``, then uses
        ``Input.insertText`` to insert the text (CDP recommended pattern).
        If the element is not found, raises
        :class:`~core.execution.errors.ElementNotFoundError` (Req 21.12).

        Parameters
        ----------
        selector:
            A CSS selector identifying the input/textarea element.
        value:
            The text to insert into the field.
        tab_id:
            Optional tab identifier.

        Returns
        -------
        ActuatorResult
            ``ok()`` on success; ``element_not_found`` when the selector
            matches nothing; ``failed`` on other errors.
        """
        if not _AIOHTTP_AVAILABLE:
            return ActuatorResult.failed("aiohttp is not installed.")

        try:
            ws_url = await self._resolve_tab_ws_url(tab_id)
        except WebsiteUnreachableError:
            raise
        except Exception as exc:
            return ActuatorResult.failed(f"Could not resolve tab: {exc}")

        # 1. Focus the element (and confirm it exists).
        js_focus = (
            f"(function() {{"
            f"  var el = document.querySelector({json.dumps(selector)});"
            f"  if (!el) return false;"
            f"  el.focus();"
            f"  el.value = '';"
            f"  return true;"
            f"}})()"
        )
        try:
            focus_result = await self._send_command(
                ws_url, "Runtime.evaluate", {"expression": js_focus, "returnByValue": True}
            )
        except WebsiteUnreachableError:
            raise
        except Exception as exc:
            return ActuatorResult.failed(str(exc))

        focused = (
            focus_result.get("result", {}).get("value")
            if isinstance(focus_result, dict)
            else None
        )
        if not focused:
            raise ElementNotFoundError(selector)

        # 2. Insert text via Input.insertText.
        try:
            await self._send_command(
                ws_url, "Input.insertText", {"text": value}
            )
        except WebsiteUnreachableError:
            raise
        except Exception as exc:
            return ActuatorResult.failed(str(exc))

        logger.debug(
            "CDPWebActuator.fill_field: filled '%s' with %d chars", selector, len(value)
        )
        return ActuatorResult.ok(f"Filled '{selector}'")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _list_targets(self) -> list[dict]:
        """
        Fetch the list of open tabs/targets from CDP /json/list.

        Raises :class:`~core.execution.errors.WebsiteUnreachableError`
        when the browser is not reachable (Req 21.13).
        """
        if not _AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is not installed.")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._base_url}/json/list",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        raise WebsiteUnreachableError(self._base_url)
                    return await resp.json(content_type=None)
        except (aiohttp.ClientConnectionError, aiohttp.ServerConnectionError) as exc:
            raise WebsiteUnreachableError(self._base_url) from exc

    async def _resolve_tab_ws_url(self, tab_id: str | None) -> str:
        """
        Return the WebSocket debugger URL for the active (or specified) tab.

        Parameters
        ----------
        tab_id:
            An explicit tab ID (``target.id``).  When ``None``, the first
            page-type target that is not a Chrome extension is selected.

        Raises
        ------
        WebsiteUnreachableError
            When the browser is not running.
        RuntimeError
            When no matching tab can be found.
        """
        targets = await self._list_targets()

        if tab_id is not None:
            # Find the specified target by id or webSocketDebuggerUrl.
            for t in targets:
                if t.get("id") == tab_id or tab_id in t.get("webSocketDebuggerUrl", ""):
                    ws_url = t.get("webSocketDebuggerUrl")
                    if ws_url:
                        return ws_url
            raise RuntimeError(f"Tab with id '{tab_id}' not found.")

        # Default: pick the first page-type non-extension target.
        for t in targets:
            if t.get("type") == "page" and not t.get("url", "").startswith("chrome-extension://"):
                ws_url = t.get("webSocketDebuggerUrl")
                if ws_url:
                    return ws_url

        raise RuntimeError("No navigable page tab found in the browser.")

    async def _send_command(
        self,
        ws_url: str,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Send a single CDP command over a new WebSocket connection and
        return the result payload.

        Opens a fresh WebSocket connection per call to avoid managing
        persistent connections. This is simple and sufficient for the
        actuator's low-frequency command pattern.

        Parameters
        ----------
        ws_url:
            The ``webSocketDebuggerUrl`` for the target tab.
        method:
            CDP method name (e.g. ``"Page.navigate"``).
        params:
            Optional parameters dict.

        Returns
        -------
        dict
            The ``result`` field of the CDP response, or an empty dict.

        Raises
        ------
        WebsiteUnreachableError
            When the WebSocket connection fails (browser closed, tab
            crashed, etc.).
        RuntimeError
            On protocol-level errors returned by CDP.
        """
        if not _AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is not installed.")

        payload = json.dumps(
            {"id": 1, "method": method, "params": params or {}}
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    ws_url,
                    timeout=aiohttp.ClientWSTimeout(ws_receive=10),
                ) as ws:
                    await ws.send_str(payload)
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if data.get("id") == 1:
                                if "error" in data:
                                    raise RuntimeError(
                                        f"CDP error: {data['error'].get('message', 'unknown')}"
                                    )
                                return data.get("result") or {}
                        elif msg.type in (
                            aiohttp.WSMsgType.ERROR,
                            aiohttp.WSMsgType.CLOSED,
                        ):
                            break
        except (aiohttp.ClientConnectionError, aiohttp.ServerConnectionError) as exc:
            raise WebsiteUnreachableError(ws_url) from exc

        return {}
