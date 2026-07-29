"""
MacController — native macOS automation and screen analysis.

This module gives HAKI the ability to actually control the host Mac by:

  * launching applications (``open -a "<app>"``),
  * capturing the full screen to a PNG (``screencapture -x``),
  * analyzing what is on screen using Google Gemini's vision model, and
  * running arbitrary AppleScript (``osascript -e``) for keystrokes /
    UI automation.

All blocking subprocess / network work is dispatched off the event loop
(via :func:`asyncio.create_subprocess_exec` or :func:`asyncio.to_thread`)
so the async orchestrator is never stalled.

================================================================
macOS PRIVACY PERMISSIONS  (REQUIRED — read this)
================================================================
These capabilities are powerful local operations and depend on macOS
privacy permissions being granted to whichever process hosts HAKI
(the HAKI app bundle, or the terminal / IDE you launched it from):

  * **Screen Recording** — required for ``screencapture`` to capture any
    real on-screen content. Without it macOS returns a blank/black or
    desktop-only image. Grant in:
        System Settings → Privacy & Security → Screen Recording
  * **Accessibility / Automation** — required for controlling other apps
    and sending keystrokes via AppleScript / ``osascript``. Grant in:
        System Settings → Privacy & Security → Accessibility   (and)
        System Settings → Privacy & Security → Automation

When a call fails because of a missing permission, the methods below
return a clear, human-readable message instructing the user to grant
HAKI (or the launching terminal app) Screen Recording and Accessibility
permissions in System Settings → Privacy & Security.

SECURITY NOTE: screen captures can contain sensitive on-screen data, and
``analyze_screen`` uploads the captured PNG to the Google Gemini API for
analysis. Treat both the local control surface and the outbound image
transfer as sensitive.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile

logger = logging.getLogger(__name__)


# Kept for the rare places where a permission really is the likely blocker
# (screen capture). Spoken replies stay short — we do NOT tack this onto every
# failure anymore (the user found the long reminder annoying).
_PERMISSION_HINT = "Screen Recording / Accessibility permission check kar lena."


class MacController:
    """Native macOS automation surface used by the mac_command handler."""

    # ------------------------------------------------------------------
    # App launching
    # ------------------------------------------------------------------

    async def open_app(self, app_name: str) -> str:
        """
        Launch a macOS application via ``open -a "<app_name>"``.

        Fuzzy handling: if the given name fails (e.g. ``"whatsapp"``), the
        name is title-cased (``"Whatsapp"`` → also tries ``"WhatsApp"``-style
        capitalization) and retried.

        Returns a short, spoken-friendly success or failure message.
        """
        app_name = (app_name or "").strip()
        if not app_name:
            return "Mujhe app ka naam nahi mila. Kaunsa app kholu?"

        # Build an ordered list of candidate names to try (dedup, keep order).
        candidates: list[str] = []
        for cand in (app_name, app_name.title(), app_name.upper(), app_name.capitalize()):
            if cand and cand not in candidates:
                candidates.append(cand)

        for cand in candidates:
            rc, _out, _err = await self._run_process("open", "-a", cand)
            if rc == 0:
                return f"Done — {cand} khol diya."

        return f"{app_name} open nahi kar paya."

    # ------------------------------------------------------------------
    # Screen capture
    # ------------------------------------------------------------------

    async def capture_screen(self, path: str | None = None) -> str:
        """
        Capture the full screen to a PNG using ``screencapture -x <path>``.

        The ``-x`` flag silences the camera/shutter sound. When ``path`` is
        None a temp file is used. Returns the captured file path.

        Raises :class:`RuntimeError` on failure so callers (e.g.
        :meth:`analyze_screen`) can surface a permission-aware message.
        """
        if path is None:
            fd, path = tempfile.mkstemp(prefix="haki_screen_", suffix=".png")
            os.close(fd)

        rc, _out, err = await self._run_process("screencapture", "-x", path)
        if rc != 0:
            raise RuntimeError(
                f"screencapture failed (rc={rc}): {err.strip()}. {_PERMISSION_HINT}"
            )
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise RuntimeError(
                "screencapture produced no image — this usually means Screen "
                f"Recording permission is missing. {_PERMISSION_HINT}"
            )
        return path

    # ------------------------------------------------------------------
    # Screen analysis (Gemini vision)
    # ------------------------------------------------------------------

    async def analyze_screen(
        self, question: str, gemini_api_key: str | None
    ) -> str:
        """
        Capture the screen and ask Google Gemini's vision model about it.

        Sends the captured PNG plus ``question`` to ``gemini-2.5-flash``.
        The blocking Gemini call runs in a thread executor. Returns the
        analysis as plain text. If no API key is supplied, returns a clear
        message that screen analysis needs the Gemini API key.
        """
        if not gemini_api_key:
            return (
                "Screen analyze karne ke liye mujhe Gemini API key chahiye. "
                "Please set HAKI_GEMINI_API_KEY."
            )

        # 1) Capture the screen (permission-aware errors surface here).
        try:
            png_path = await self.capture_screen()
        except Exception as exc:  # noqa: BLE001
            logger.warning("analyze_screen: capture failed: %r", exc)
            return f"Screen capture nahi ho payi: {exc}"

        # 2) Run the blocking Gemini vision call off the event loop.
        try:
            text = await asyncio.to_thread(
                self._gemini_analyze_sync, png_path, question, gemini_api_key
            )
            return text
        except Exception as exc:  # noqa: BLE001
            logger.warning("analyze_screen: Gemini call failed: %r", exc)
            return f"Screen analyze karte waqt error aaya: {exc}"
        finally:
            # Best-effort cleanup of the temp capture.
            try:
                os.remove(png_path)
            except OSError:
                pass

    @staticmethod
    def _gemini_analyze_sync(
        png_path: str, question: str, gemini_api_key: str
    ) -> str:
        """
        Synchronous Gemini vision call. Runs inside a thread executor.

        Uses the google-genai SDK (required for the new AQ.* authentication
        keys). The PNG is sent inline as raw bytes via ``types.Part.from_bytes``.
        """
        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415

        client = genai.Client(api_key=gemini_api_key)

        with open(png_path, "rb") as fh:
            image_bytes = fh.read()
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

        prompt = question or "What is currently shown on the screen?"
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image_part],
        )

        text = getattr(response, "text", None)
        if not text:
            # Some responses surface text via candidates/parts only.
            try:
                parts = response.candidates[0].content.parts  # type: ignore[attr-defined]
                text = "".join(getattr(p, "text", "") for p in parts)
            except Exception:  # noqa: BLE001
                text = ""
        return (text or "").strip() or "Mujhe screen par kuch clear nahi dikha."

    # ------------------------------------------------------------------
    # AppleScript
    # ------------------------------------------------------------------

    async def run_applescript(self, script: str) -> str:
        """
        Run an arbitrary AppleScript via ``osascript -e <script>``.

        Returns stdout on success, or a permission-aware error message that
        includes stderr on failure. Enables keystrokes / UI automation.
        """
        if not script or not script.strip():
            return "Koi AppleScript nahi mila chalane ke liye."

        rc, out, err = await self._run_process("osascript", "-e", script)
        if rc == 0:
            return out.strip() or "AppleScript chal gaya."
        return (
            f"AppleScript fail hua (rc={rc}): {err.strip() or out.strip()}. "
            + _PERMISSION_HINT
        )

    # ------------------------------------------------------------------
    # Volume control
    # ------------------------------------------------------------------

    async def set_volume(self, level: int) -> str:
        """Set the system output volume to ``level`` (0–100)."""
        level = max(0, min(100, int(level)))
        rc, _out, err = await self._run_process(
            "osascript", "-e", f"set volume output volume {level}"
        )
        if rc == 0:
            return f"Volume {level}% kar diya."
        return "Volume set nahi kar paya."

    async def adjust_volume(self, direction: str, step: int = 15) -> str:
        """Raise or lower the system volume by ``step`` percent.

        ``direction`` is ``"up"`` / ``"down"``.
        """
        # Read current volume, then clamp the adjusted value.
        rc, out, _err = await self._run_process(
            "osascript", "-e", "output volume of (get volume settings)"
        )
        try:
            current = int(out.strip())
        except (ValueError, AttributeError):
            current = 50
        new_level = current + step if direction == "up" else current - step
        return await self.set_volume(new_level)

    async def set_mute(self, muted: bool) -> str:
        """Mute or unmute the system output."""
        flag = "true" if muted else "false"
        rc, _out, err = await self._run_process(
            "osascript", "-e", f"set volume output muted {flag}"
        )
        if rc == 0:
            return "Mute kar diya." if muted else "Unmute kar diya."
        return "Mute toggle nahi ho paya."

    # ------------------------------------------------------------------
    # Brightness control
    # ------------------------------------------------------------------

    async def adjust_brightness(self, direction: str, steps: int = 4) -> str:
        """Raise or lower the built-in display brightness.

        macOS has no public CLI to set an absolute brightness level without a
        third-party tool, but the brightness media keys are exposed as System
        Events key codes:  144 = brightness up, 145 = brightness down.  We send
        the key ``steps`` times so the change is noticeable (each press is a
        small increment, same as tapping the keyboard key).

        ``direction`` is ``"up"`` / ``"down"``.  Needs Accessibility permission.
        """
        steps = max(1, min(16, int(steps)))
        key_code = 144 if direction == "up" else 145
        # One osascript call sends all key presses to minimise process spawns.
        repeats = "\n".join(
            f'  key code {key_code}' for _ in range(steps)
        )
        script = f'tell application "System Events"\n{repeats}\nend tell'
        rc, _out, _err = await self._run_process("osascript", "-e", script)
        if rc == 0:
            return (
                "Brightness badha di." if direction == "up" else "Brightness kam kar di."
            )
        return "Brightness change nahi kar paya."

    async def set_brightness_extreme(self, maximum: bool) -> str:
        """Push brightness to (near) full or minimum by sending many key presses."""
        return await self.adjust_brightness("up" if maximum else "down", steps=16)

    # ------------------------------------------------------------------
    # Media playback (Music / Spotify)
    # ------------------------------------------------------------------

    async def media_control(self, command: str) -> str:
        """Control the active media player (Apple Music or Spotify).

        ``command`` is one of: ``play``, ``pause``, ``playpause``, ``next``,
        ``previous``.  We target Spotify if it is running, else Apple Music,
        using each app's AppleScript dictionary.  No third-party tools needed.
        """
        verb_map = {
            "play": "play",
            "pause": "pause",
            "playpause": "playpause",
            "toggle": "playpause",
            "next": "next track",
            "previous": "previous track",
            "prev": "previous track",
        }
        verb = verb_map.get(command, "playpause")

        # Prefer whichever player is already running; default to Music.
        script = (
            'tell application "System Events"\n'
            '  set spotifyRunning to (exists (processes whose name is "Spotify"))\n'
            '  set musicRunning to (exists (processes whose name is "Music"))\n'
            "end tell\n"
            "if spotifyRunning then\n"
            f'  tell application "Spotify" to {verb}\n'
            "else\n"
            f'  tell application "Music" to {verb}\n'
            "end if"
        )
        rc, _out, _err = await self._run_process("osascript", "-e", script)
        if rc == 0:
            spoken = {
                "play": "Gaana chala diya.",
                "pause": "Gaana pause kar diya.",
                "playpause": "Playback toggle kar diya.",
                "toggle": "Playback toggle kar diya.",
                "next": "Agla gaana laga diya.",
                "previous": "Pichla gaana laga diya.",
                "prev": "Pichla gaana laga diya.",
            }
            return spoken.get(command, "Ho gaya.")
        return "Music control nahi kar paya."

    # ------------------------------------------------------------------
    # App lifecycle (quit / close)
    # ------------------------------------------------------------------

    async def close_app(self, app_name: str) -> str:
        """Quit a running macOS application via AppleScript ``quit``."""
        app_name = (app_name or "").strip()
        if not app_name:
            return "Kaunsa app band karu? Naam batao."

        # Try the given name and common capitalisations.
        candidates: list[str] = []
        for cand in (app_name, app_name.title(), app_name.upper(), app_name.capitalize()):
            if cand and cand not in candidates:
                candidates.append(cand)

        for cand in candidates:
            rc, _out, _err = await self._run_process(
                "osascript", "-e", f'quit app "{cand}"'
            )
            if rc == 0:
                return f"Done — {cand} band kar diya."

        return f"{app_name} band nahi kar paya."

    # ------------------------------------------------------------------
    # Keyboard / typing
    # ------------------------------------------------------------------

    async def type_text(self, text: str, *, press_return: bool = False) -> str:
        """Type ``text`` into the frontmost app via System Events keystrokes.

        When ``press_return`` is True, a Return key press is sent after the
        text (used to send a chat message once it has been typed).
        """
        text = text or ""
        if not text.strip():
            return "Kya type karu? Text to mila hi nahi."

        # Escape backslashes and double quotes for the AppleScript string.
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        script = f'tell application "System Events" to keystroke "{escaped}"'
        if press_return:
            script += '\ntell application "System Events" to key code 36'  # 36 = Return

        rc, _out, err = await self._run_process("osascript", "-e", script)
        if rc == 0:
            return "Type kar diya." if not press_return else "Type karke bhej diya."
        return "Type nahi kar paya."

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    async def send_imessage(self, recipient: str, message: str) -> str:
        """Send an iMessage/SMS to ``recipient`` (name or number) via Messages.

        Uses the Messages AppleScript dictionary, which works for contacts that
        Messages can already resolve (buddies on the iMessage service).
        """
        recipient = (recipient or "").strip()
        message = (message or "").strip()
        if not recipient or not message:
            return "Kisko aur kya message bhejna hai, dono batao."

        r = recipient.replace('"', '\\"')
        m = message.replace("\\", "\\\\").replace('"', '\\"')
        script = (
            'tell application "Messages"\n'
            '  set targetService to 1st service whose service type = iMessage\n'
            f'  set targetBuddy to buddy "{r}" of targetService\n'
            f'  send "{m}" to targetBuddy\n'
            "end tell"
        )
        rc, _out, err = await self._run_process("osascript", "-e", script)
        if rc == 0:
            return f"{recipient} ko message bhej diya."
        return f"{recipient} ko message nahi bhej paya."

    async def send_whatsapp(self, recipient: str, message: str) -> str:
        """Best-effort WhatsApp send.

        WhatsApp has no AppleScript dictionary, so we open the chat via the
        ``whatsapp://`` URL scheme (search), then type the message and press
        Return through System Events.  This depends on Accessibility
        permission and WhatsApp Desktop being installed and logged in.
        """
        recipient = (recipient or "").strip()
        message = (message or "").strip()
        if not recipient or not message:
            return "WhatsApp pe kisko aur kya bhejna hai, dono batao."

        # Open WhatsApp and bring it to the front.
        await self._run_process("open", "-a", "WhatsApp")
        await asyncio.sleep(1.5)

        # Open the in-app search, type the contact, open the top chat.
        search_open = (
            'tell application "System Events" to keystroke "f" using command down'
        )
        await self._run_process("osascript", "-e", search_open)
        await asyncio.sleep(0.6)
        await self.type_text(recipient)
        await asyncio.sleep(1.0)
        # Return selects the first matching chat.
        await self._run_process(
            "osascript", "-e", 'tell application "System Events" to key code 36'
        )
        await asyncio.sleep(0.8)
        # Type the message into the chat box and send it.
        result = await self.type_text(message, press_return=True)
        if "nahi" in result.lower():
            return "WhatsApp pe message type nahi kar paya."
        return f"WhatsApp pe {recipient} ko message bhej diya."

    # ------------------------------------------------------------------
    # Calling
    # ------------------------------------------------------------------

    async def make_call(self, contact: str, *, facetime_audio: bool = True) -> str:
        """Place a call to ``contact`` (a name or a phone number).

        If a name is given, the phone number is resolved from the Contacts app
        via AppleScript, then dialled through FaceTime using the ``tel:`` /
        ``facetime-audio:`` URL scheme.
        """
        contact = (contact or "").strip()
        if not contact:
            return "Kisko call karu? Naam ya number batao."

        number = contact
        # If it's not already a phone number, resolve via Contacts.
        if not re.fullmatch(r"[+\d\s\-()]+", contact):
            resolved = await self._lookup_contact_number(contact)
            if not resolved:
                return (
                    f"'{contact}' ka number Contacts mein nahi mila. "
                    "Contact save hai? Ya number bol dein."
                )
            number = resolved

        clean_number = re.sub(r"[\s\-()]", "", number)
        scheme = "facetime-audio" if facetime_audio else "tel"
        rc, _out, err = await self._run_process("open", f"{scheme}://{clean_number}")
        if rc == 0:
            return f"{contact} ko call laga raha hoon ({clean_number})."
        return f"{contact} ko call nahi laga paya."

    async def _lookup_contact_number(self, name: str) -> str | None:
        """Return the first phone number for ``name`` from the Contacts app."""
        n = name.replace('"', '\\"')
        script = (
            'tell application "Contacts"\n'
            f'  set matches to (every person whose name contains "{n}")\n'
            "  if matches is {} then return \"\"\n"
            "  set thePerson to item 1 of matches\n"
            "  if (count of phones of thePerson) is 0 then return \"\"\n"
            "  return value of phone 1 of thePerson\n"
            "end tell"
        )
        rc, out, _err = await self._run_process("osascript", "-e", script)
        if rc == 0 and out.strip():
            return out.strip()
        return None

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    async def _run_process(
        program: str, *args: str, timeout: float = 20.0
    ) -> tuple[int, str, str]:
        """
        Run ``program`` with ``args`` and return ``(returncode, stdout, stderr)``.

        Uses :func:`asyncio.create_subprocess_exec` so the event loop is not
        blocked. On a hard launch failure (e.g. command not found) returns a
        non-zero code with the exception text in stderr.

        A ``timeout`` (seconds) guards against a subprocess that never returns
        — e.g. an ``osascript`` keystroke call that blocks on a macOS
        permission prompt or an unresponsive target app. On timeout the
        process is killed and a non-zero code is returned, so the turn can
        finish cleanly instead of hanging the whole assistant (which would
        otherwise leave the IPC turn marked "active" and stop HAKI from
        listening until a restart).
        """
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                program,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            return (
                proc.returncode if proc.returncode is not None else -1,
                stdout_b.decode("utf-8", errors="replace"),
                stderr_b.decode("utf-8", errors="replace"),
            )
        except asyncio.TimeoutError:
            # Kill the stuck process so it cannot linger.
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await proc.wait()
                except Exception:  # noqa: BLE001
                    pass
            return (
                -1,
                "",
                f"timed out after {timeout:.0f}s: {program} "
                f"{' '.join(args)} (possible permission prompt or hung app)",
            )
        except FileNotFoundError as exc:
            return (-1, "", f"command not found: {program} ({exc})")
        except Exception as exc:  # noqa: BLE001
            return (-1, "", f"{type(exc).__name__}: {exc}")
