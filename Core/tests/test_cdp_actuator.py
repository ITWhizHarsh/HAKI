"""
Unit tests for CDPWebActuator.

Tests the Arc/Chromium CDP actuator (task 24.2):
  - ActuatorResult dataclass and factory methods
  - CDPWebActuator security: only loopback addresses are accepted
  - open_tabs, navigate, click_element, fill_field behaviour
    with mocked aiohttp / network layer

Requirements covered: 21.5, 21.6, 21.12, 21.13
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from core.execution.cdp_actuator import (
    ActuatorResult,
    CDPWebActuator,
)
from core.execution.errors import ElementNotFoundError, WebsiteUnreachableError

# ---------------------------------------------------------------------------
# aiohttp availability guard
# ---------------------------------------------------------------------------

try:
    import aiohttp as _aiohttp_module
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False
    _aiohttp_module = None  # type: ignore[assignment]

requires_aiohttp = pytest.mark.skipif(
    not _HAS_AIOHTTP,
    reason="aiohttp is not installed — skipping network-layer tests",
)


# ---------------------------------------------------------------------------
# Helpers to mock aiohttp
# ---------------------------------------------------------------------------


def _make_json_response(data: Any, status: int = 200):
    """Build a mock aiohttp response that returns JSON."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=data)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _make_ws_session(messages: list[dict]):
    """
    Build a mock aiohttp ClientSession whose ws_connect returns an async
    iterator of text WebSocket messages.

    Each dict in *messages* is JSON-serialised and wrapped in a fake msg.
    """

    # Determine the TEXT type value at call time.
    if _HAS_AIOHTTP:
        _text_type = _aiohttp_module.WSMsgType.TEXT
    else:
        _text_type = 1  # arbitrary fallback (tests are skipped when no aiohttp)

    serialised = [json.dumps(m) for m in messages]

    class _FakeMsg:
        def __init__(self, data: str):
            self.data = data
            self.type = _text_type

    class _FakeWS:
        def __init__(self):
            self._iter = iter([_FakeMsg(s) for s in serialised])

        async def send_str(self, _s: str) -> None:
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration:
                raise StopAsyncIteration

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

    class _FakeSession:
        def ws_connect(self, *_a, **_kw):
            return _FakeWS()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

    class _FakeSessionCls:
        def __call__(self, *_a, **_kw):
            return _FakeSession()

        def __enter__(self):
            return _FakeSession()

        def __exit__(self, *_):
            pass

    return _FakeSessionCls()


# ---------------------------------------------------------------------------
# ActuatorResult tests
# ---------------------------------------------------------------------------


class TestActuatorResult:
    """Unit tests for ActuatorResult dataclass and factory methods."""

    def test_ok_success_true(self):
        r = ActuatorResult.ok()
        assert r.success is True
        assert r.error_type is None

    def test_ok_with_detail(self):
        r = ActuatorResult.ok("opened 3 tabs")
        assert r.success is True
        assert r.detail == "opened 3 tabs"

    def test_unreachable_success_false(self):
        r = ActuatorResult.unreachable("https://example.com")
        assert r.success is False
        assert r.error_type == "unreachable"
        assert r.detail == "https://example.com"

    def test_element_not_found_success_false(self):
        r = ActuatorResult.element_not_found("#submit-button")
        assert r.success is False
        assert r.error_type == "element_not_found"
        assert r.detail == "#submit-button"

    def test_permission_denied(self):
        r = ActuatorResult.permission_denied("Screen Recording")
        assert r.success is False
        assert r.error_type == "permission_denied"

    def test_failed(self):
        r = ActuatorResult.failed("timeout")
        assert r.success is False
        assert r.error_type == "failed"
        assert r.detail == "timeout"

    def test_direct_construction(self):
        r = ActuatorResult(success=True)
        assert r.error_type is None
        assert r.detail is None


# ---------------------------------------------------------------------------
# CDPWebActuator security / construction tests (no network — always run)
# ---------------------------------------------------------------------------


class TestCDPWebActuatorSecurity:
    """Security: only loopback addresses are accepted (Req 21.13)."""

    def test_default_host_is_loopback(self):
        actuator = CDPWebActuator()
        assert actuator._host == "127.0.0.1"

    def test_localhost_accepted(self):
        actuator = CDPWebActuator(host="localhost")
        assert actuator._host == "localhost"

    def test_ipv6_loopback_accepted(self):
        actuator = CDPWebActuator(host="::1")
        assert actuator._host == "::1"

    def test_remote_host_raises_value_error(self):
        with pytest.raises(ValueError, match="loopback"):
            CDPWebActuator(host="192.168.1.100")

    def test_arbitrary_domain_raises_value_error(self):
        with pytest.raises(ValueError, match="loopback"):
            CDPWebActuator(host="example.com")

    def test_custom_port_stored(self):
        actuator = CDPWebActuator(port=9999)
        assert actuator._port == 9999


# ---------------------------------------------------------------------------
# open_tabs tests (Req 21.5)
# ---------------------------------------------------------------------------


class TestOpenTabs:
    """Tests for CDPWebActuator.open_tabs (Req 21.5)."""

    @pytest.mark.asyncio
    async def test_empty_list_returns_ok(self):
        actuator = CDPWebActuator()
        result = await actuator.open_tabs([])
        assert result.success is True

    @pytest.mark.asyncio
    async def test_no_aiohttp_returns_failed(self):
        """When aiohttp is not installed, return failed result."""
        actuator = CDPWebActuator()
        with patch("core.execution.cdp_actuator._AIOHTTP_AVAILABLE", False):
            result = await actuator.open_tabs(["https://example.com"])
        assert result.success is False
        assert result.error_type == "failed"

    @pytest.mark.asyncio
    @requires_aiohttp
    async def test_successful_tab_open(self):
        """Successful /json/new?<url> returns ok result."""
        actuator = CDPWebActuator()
        resp = _make_json_response({"id": "tab1", "url": "https://github.com"}, status=200)

        with patch("core.execution.cdp_actuator._AIOHTTP_AVAILABLE", True):
            with patch.object(_aiohttp_module, "ClientSession") as mock_session_cls:
                mock_session = AsyncMock()
                mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                mock_session.get.return_value = resp

                result = await actuator.open_tabs(["https://github.com"])

        assert result.success is True

    @pytest.mark.asyncio
    @requires_aiohttp
    async def test_connection_error_raises_website_unreachable(self):
        """Connection refused → WebsiteUnreachableError (Req 21.13)."""
        actuator = CDPWebActuator()

        with patch("core.execution.cdp_actuator._AIOHTTP_AVAILABLE", True):
            with patch.object(_aiohttp_module, "ClientSession") as mock_session_cls:
                mock_session = AsyncMock()
                mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                mock_session.get.side_effect = _aiohttp_module.ClientConnectionError("refused")

                with pytest.raises(WebsiteUnreachableError):
                    await actuator.open_tabs(["https://example.com"])

    @pytest.mark.asyncio
    @requires_aiohttp
    async def test_non_200_status_returns_failed(self):
        """Non-200 HTTP response from /json/new → failed result."""
        actuator = CDPWebActuator()
        resp = _make_json_response({}, status=500)

        with patch("core.execution.cdp_actuator._AIOHTTP_AVAILABLE", True):
            with patch.object(_aiohttp_module, "ClientSession") as mock_session_cls:
                mock_session = AsyncMock()
                mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)
                mock_session.get.return_value = resp

                result = await actuator.open_tabs(["https://example.com"])

        assert result.success is False
        assert result.error_type == "failed"


# ---------------------------------------------------------------------------
# navigate tests (Req 21.6, 21.13)
# ---------------------------------------------------------------------------


class TestNavigate:
    """Tests for CDPWebActuator.navigate."""

    @pytest.mark.asyncio
    async def test_no_aiohttp_returns_failed(self):
        actuator = CDPWebActuator()
        with patch("core.execution.cdp_actuator._AIOHTTP_AVAILABLE", False):
            result = await actuator.navigate("https://example.com")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_browser_unreachable_during_navigate(self):
        """When browser not running, _list_targets raises WebsiteUnreachableError."""
        actuator = CDPWebActuator()
        with patch("core.execution.cdp_actuator._AIOHTTP_AVAILABLE", True):
            with patch.object(
                actuator, "_list_targets",
                AsyncMock(side_effect=WebsiteUnreachableError("http://127.0.0.1:9222"))
            ):
                with pytest.raises(WebsiteUnreachableError):
                    await actuator.navigate("https://example.com")

    @pytest.mark.asyncio
    async def test_cdp_error_text_raises_unreachable(self):
        """When CDP returns errorText (DNS failure), raise WebsiteUnreachableError."""
        actuator = CDPWebActuator()
        targets = [{"type": "page", "url": "about:blank", "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/1"}]
        cdp_response = {"id": 1, "result": {"errorText": "net::ERR_NAME_NOT_RESOLVED"}}

        with patch("core.execution.cdp_actuator._AIOHTTP_AVAILABLE", True):
            with patch.object(actuator, "_list_targets", AsyncMock(return_value=targets)):
                with patch.object(
                    actuator, "_send_command",
                    AsyncMock(return_value={"errorText": "net::ERR_NAME_NOT_RESOLVED"})
                ):
                    with pytest.raises(WebsiteUnreachableError):
                        await actuator.navigate("https://nonexistent.invalid")

    @pytest.mark.asyncio
    async def test_successful_navigation(self):
        """Page.navigate returns a valid frameId → ok result."""
        actuator = CDPWebActuator()
        targets = [{"type": "page", "url": "about:blank", "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/1"}]

        with patch("core.execution.cdp_actuator._AIOHTTP_AVAILABLE", True):
            with patch.object(actuator, "_list_targets", AsyncMock(return_value=targets)):
                with patch.object(
                    actuator, "_send_command",
                    AsyncMock(return_value={"frameId": "frame1"})
                ):
                    result = await actuator.navigate("https://github.com")

        assert result.success is True


# ---------------------------------------------------------------------------
# click_element tests (Req 21.6, 21.12)
# ---------------------------------------------------------------------------


class TestClickElement:
    """Tests for CDPWebActuator.click_element."""

    @pytest.mark.asyncio
    async def test_no_aiohttp_returns_failed(self):
        actuator = CDPWebActuator()
        with patch("core.execution.cdp_actuator._AIOHTTP_AVAILABLE", False):
            result = await actuator.click_element("#btn")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_element_not_found_raises_error(self):
        """When JS returns null (element absent), raise ElementNotFoundError."""
        actuator = CDPWebActuator()
        targets = [{"type": "page", "url": "about:blank", "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/1"}]

        with patch("core.execution.cdp_actuator._AIOHTTP_AVAILABLE", True):
            with patch.object(actuator, "_list_targets", AsyncMock(return_value=targets)):
                with patch.object(
                    actuator, "_send_command",
                    AsyncMock(return_value={"result": {"value": None}})
                ):
                    with pytest.raises(ElementNotFoundError):
                        await actuator.click_element("#nonexistent")

    @pytest.mark.asyncio
    async def test_successful_click(self):
        """When element found, synthesize clicks and return ok."""
        actuator = CDPWebActuator()
        targets = [{"type": "page", "url": "about:blank", "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/1"}]

        call_count = 0

        async def mock_send_command(ws_url, method, params=None):
            nonlocal call_count
            call_count += 1
            if method == "Runtime.evaluate":
                return {"result": {"value": {"x": 100.0, "y": 200.0}}}
            return {}

        with patch("core.execution.cdp_actuator._AIOHTTP_AVAILABLE", True):
            with patch.object(actuator, "_list_targets", AsyncMock(return_value=targets)):
                with patch.object(actuator, "_send_command", side_effect=mock_send_command):
                    result = await actuator.click_element("button#submit")

        assert result.success is True
        # 3 calls: Runtime.evaluate + mousePressed + mouseReleased
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_browser_unreachable_during_click(self):
        """WebsiteUnreachableError propagates from _list_targets."""
        actuator = CDPWebActuator()
        with patch("core.execution.cdp_actuator._AIOHTTP_AVAILABLE", True):
            with patch.object(
                actuator, "_list_targets",
                AsyncMock(side_effect=WebsiteUnreachableError("http://127.0.0.1:9222"))
            ):
                with pytest.raises(WebsiteUnreachableError):
                    await actuator.click_element("#btn")


# ---------------------------------------------------------------------------
# fill_field tests (Req 21.6, 21.12)
# ---------------------------------------------------------------------------


class TestFillField:
    """Tests for CDPWebActuator.fill_field."""

    @pytest.mark.asyncio
    async def test_no_aiohttp_returns_failed(self):
        actuator = CDPWebActuator()
        with patch("core.execution.cdp_actuator._AIOHTTP_AVAILABLE", False):
            result = await actuator.fill_field("#inp", "hello")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_element_not_found_raises_error(self):
        """When focus JS returns False (element absent), raise ElementNotFoundError."""
        actuator = CDPWebActuator()
        targets = [{"type": "page", "url": "about:blank", "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/1"}]

        with patch("core.execution.cdp_actuator._AIOHTTP_AVAILABLE", True):
            with patch.object(actuator, "_list_targets", AsyncMock(return_value=targets)):
                with patch.object(
                    actuator, "_send_command",
                    AsyncMock(return_value={"result": {"value": False}})
                ):
                    with pytest.raises(ElementNotFoundError):
                        await actuator.fill_field("#missing-input", "test")

    @pytest.mark.asyncio
    async def test_successful_fill(self):
        """When element focused successfully, insert text and return ok."""
        actuator = CDPWebActuator()
        targets = [{"type": "page", "url": "about:blank", "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/1"}]

        call_results = [
            {"result": {"value": True}},  # Runtime.evaluate (focus) → success
            {},                            # Input.insertText → ok
        ]
        call_iter = iter(call_results)

        async def mock_send(ws_url, method, params=None):
            return next(call_iter)

        with patch("core.execution.cdp_actuator._AIOHTTP_AVAILABLE", True):
            with patch.object(actuator, "_list_targets", AsyncMock(return_value=targets)):
                with patch.object(actuator, "_send_command", side_effect=mock_send):
                    result = await actuator.fill_field("input[name='q']", "HAKI test")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_browser_unreachable_during_fill(self):
        """WebsiteUnreachableError propagates from _list_targets."""
        actuator = CDPWebActuator()
        with patch("core.execution.cdp_actuator._AIOHTTP_AVAILABLE", True):
            with patch.object(
                actuator, "_list_targets",
                AsyncMock(side_effect=WebsiteUnreachableError("http://127.0.0.1:9222"))
            ):
                with pytest.raises(WebsiteUnreachableError):
                    await actuator.fill_field("#q", "test")


# ---------------------------------------------------------------------------
# _resolve_tab_ws_url tests
# ---------------------------------------------------------------------------


class TestResolveTabWsUrl:
    """Tests for tab resolution helpers."""

    @pytest.mark.asyncio
    async def test_picks_first_page_tab(self):
        actuator = CDPWebActuator()
        targets = [
            {
                "type": "page",
                "url": "https://github.com",
                "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/1",
            },
        ]
        with patch.object(actuator, "_list_targets", AsyncMock(return_value=targets)):
            ws_url = await actuator._resolve_tab_ws_url(None)
        assert "page/1" in ws_url

    @pytest.mark.asyncio
    async def test_skips_extension_tabs(self):
        actuator = CDPWebActuator()
        targets = [
            {
                "type": "page",
                "url": "chrome-extension://abc/bg.html",
                "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/ext",
            },
            {
                "type": "page",
                "url": "https://github.com",
                "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/real",
            },
        ]
        with patch.object(actuator, "_list_targets", AsyncMock(return_value=targets)):
            ws_url = await actuator._resolve_tab_ws_url(None)
        assert "page/real" in ws_url

    @pytest.mark.asyncio
    async def test_explicit_tab_id_resolved(self):
        actuator = CDPWebActuator()
        targets = [
            {
                "type": "page",
                "id": "tab42",
                "url": "https://example.com",
                "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/42",
            },
        ]
        with patch.object(actuator, "_list_targets", AsyncMock(return_value=targets)):
            ws_url = await actuator._resolve_tab_ws_url("tab42")
        assert "page/42" in ws_url

    @pytest.mark.asyncio
    async def test_no_page_tabs_raises(self):
        actuator = CDPWebActuator()
        with patch.object(actuator, "_list_targets", AsyncMock(return_value=[])):
            with pytest.raises(RuntimeError, match="No navigable"):
                await actuator._resolve_tab_ws_url(None)
