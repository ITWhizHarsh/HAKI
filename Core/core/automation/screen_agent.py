"""
ScreenAgent — see the screen, understand it, and act.

This is the core of HAKI's agentic automation: the LLM is given a
screenshot + accessibility tree snapshot, reasons about what needs to
happen next, and dispatches real macOS actions (AppleScript, keyboard
input, accessibility clicks). The loop continues — take screenshot,
decide, act — until the goal is achieved or the agent gives up.

Architecture:
    ┌─────────────┐
    │  User goal  │  "call mummy", "send papa a message", "type hello in Discord"
    └──────┬──────┘
           │
    ┌──────▼──────────┐
    │  ScreenAgent    │  one outer agentic loop
    │  .run(goal)     │
    └──────┬──────────┘
           │  for up to MAX_STEPS
    ┌──────▼──────────────────────────────┐
    │  Step 1: capture_screen()           │  PNG → base64
    │  Step 2: ax_snapshot()              │  macOS Accessibility API text dump
    │  Step 3: llm_decide(goal, screen,   │  Gemini vision → JSON action
    │           ax, history)             │
    │  Step 4: execute_action(action)     │  AppleScript / keyboard / AX
    │  Step 5: speak_step(description)    │  TTS feedback
    └─────────────────────────────────────┘
           │  action.type == "done" or "error"
    ┌──────▼──────────┐
    │  done / report  │
    └─────────────────┘

Actions the LLM can emit (JSON):
    {"type": "open_app",   "app": "WhatsApp"}
    {"type": "click",      "description": "Send button", "x": 123, "y": 456}
    {"type": "type_text",  "text": "Hello bhai", "send": false}
    {"type": "keystroke",  "key": "return"}
    {"type": "applescript","script": "tell application ..."}
    {"type": "wait",       "seconds": 1.5}
    {"type": "done",       "message": "Message sent."}
    {"type": "error",      "message": "Cannot find the element."}

The LLM NEVER receives raw base64 image bytes via token count — it uses
Gemini's multimodal API (vision) to see the screen as an image.

Privacy: screenshots are sent to Gemini only for the duration of the
agentic turn and are deleted immediately afterwards.  The user can set
HAKI_AGENT_NO_CLOUD=1 to disable cloud vision (agent will use AX-only
heuristics instead).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Maximum agentic loop iterations before giving up.
MAX_STEPS = 12

# Seconds to wait after each action before the next screenshot.
ACTION_SETTLE_SECS = 0.8

# System prompt given to the LLM at every step of the agentic loop.
_AGENT_SYSTEM_PROMPT = """\
You are HAKI, an AI agent controlling a macOS computer on behalf of the user.
You see screenshots of the current screen and an Accessibility tree snapshot.
Your job is to figure out what action to take NEXT to achieve the user's goal.

## RULES
1. Reply with ONLY valid JSON — no prose, no markdown fences.
2. Each reply is ONE action only (the next single step).
3. Prefer Accessibility (AX) actions over vision clicks when the tree shows the element.
4. Always check the screenshot to confirm the current state before acting.
5. If the goal is already achieved, reply {"type": "done", "message": "<brief confirmation>"}.
6. If you are stuck after 3 tries on the same goal, reply {"type": "error", "message": "<why stuck>"}.
7. Keep "message" fields in Roman-script Hinglish (the user speaks Hinglish).

## ACTION SCHEMA
Each action is a JSON object with a "type" field:

open_app:      {"type": "open_app",   "app": "AppName"}
click:         {"type": "click",      "description": "what to click", "x": <int>, "y": <int>}
               -- x/y are REQUIRED pixel coordinates from the screenshot
ax_click:      {"type": "ax_click",   "ax_id": "<AX element id from tree>", "description": "what"}
type_text:     {"type": "type_text",  "text": "...", "send": <true|false>}
               -- send:true presses Return after typing (to send messages)
keystroke:     {"type": "keystroke",  "key": "<return|escape|tab|cmd+a|cmd+v|...>"}
applescript:   {"type": "applescript","script": "<one-line AppleScript>"}
scroll:        {"type": "scroll",     "direction": "up|down", "amount": <int 1-10>}
wait:          {"type": "wait",       "seconds": <float>}
done:          {"type": "done",       "message": "<confirmation in Hinglish>"}
error:         {"type": "error",      "message": "<why stuck, in Hinglish>"}

## EXAMPLE FLOW (goal: "discord mein papa ko hello bhejo")
Step 1 — screen shows macOS desktop
  → {"type": "open_app", "app": "Discord"}
Step 2 — Discord opens, DM list visible
  → {"type": "applescript", "script": "tell application \\"System Events\\" to keystroke \\"k\\" using command down"}
Step 3 — Quick switcher open
  → {"type": "type_text", "text": "Papa", "send": false}
Step 4 — Papa's chat shows in list
  → {"type": "keystroke", "key": "return"}
Step 5 — Papa's chat is open, message box is focused
  → {"type": "type_text", "text": "Hello bhai!", "send": true}
Step 6 — message sent
  → {"type": "done", "message": "Papa ko Discord pe hello bhej diya."}
"""

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    """One iteration of the agentic loop."""
    step_num: int
    screenshot_path: str | None = None
    ax_snapshot: str = ""
    action: dict = field(default_factory=dict)
    action_result: str = ""
    error: str | None = None


@dataclass
class AgentResult:
    """Final result returned to the caller."""
    success: bool
    message: str
    steps: list[AgentStep] = field(default_factory=list)
    goal: str = ""


# ---------------------------------------------------------------------------
# ScreenAgent
# ---------------------------------------------------------------------------


class ScreenAgent:
    """
    Agentic screen-control loop for HAKI.

    Parameters
    ----------
    llm_router:
        HAKI's LLMRouter (used for the text-only planning tiers).
    gemini_api_key:
        Gemini API key for multimodal vision analysis.  When None, the
        agent falls back to AX-only reasoning (text-based, no vision).
    mac_controller:
        A MacController instance for low-level macOS actions.
    max_steps:
        Maximum agentic loop iterations before giving up.
    ipc_writer:
        Optional async callable for streaming progress IPC messages to
        the Swift UI (AGENT_STEP events).
    """

    def __init__(
        self,
        llm_router: Any | None = None,
        gemini_api_key: str | None = None,
        mac_controller: Any | None = None,
        max_steps: int = MAX_STEPS,
        ipc_writer: Any | None = None,
    ) -> None:
        from core.automation.mac_controller import MacController  # noqa: PLC0415

        self._llm = llm_router
        self._gemini_key = gemini_api_key
        self._mac = mac_controller or MacController()
        self._max_steps = max_steps
        self._ipc_writer = ipc_writer

    # ------------------------------------------------------------------
    # Public: run an agentic goal
    # ------------------------------------------------------------------

    async def run(self, goal: str) -> AgentResult:
        """
        Execute an agentic loop to accomplish *goal*.

        Takes a screenshot + AX snapshot at each step, asks Gemini/LLM
        what to do next, executes the action, and loops until the goal
        is achieved or MAX_STEPS is exhausted.

        Returns an :class:`AgentResult` with the final status.
        """
        logger.info("[ScreenAgent] Starting agentic run: %r", goal)
        steps: list[AgentStep] = []
        history: list[dict] = []  # action history for the LLM

        await self._send_ipc("agent_start", {"goal": goal, "max_steps": self._max_steps})

        for step_num in range(1, self._max_steps + 1):
            step = AgentStep(step_num=step_num)
            steps.append(step)

            # 1. Capture screenshot.
            try:
                png_path = await self._mac.capture_screen()
                step.screenshot_path = png_path
            except Exception as exc:
                step.error = f"Screenshot fail: {exc}"
                await self._send_ipc("agent_step", {
                    "step": step_num, "status": "error", "message": step.error
                })
                return AgentResult(
                    success=False,
                    message=f"Screen capture nahi ho payi: {exc}",
                    steps=steps,
                    goal=goal,
                )

            # 2. Read Accessibility tree (best-effort — many apps don't expose it).
            try:
                ax_text = await self._read_ax_tree()
                step.ax_snapshot = ax_text[:3000]  # truncate for LLM context
            except Exception as exc:
                logger.debug("[ScreenAgent] AX read failed (ok): %s", exc)
                step.ax_snapshot = "(AX tree unavailable)"

            # 3. Ask the LLM what to do next.
            try:
                action = await self._decide(goal, png_path, step.ax_snapshot, history)
                step.action = action
            except Exception as exc:
                step.error = f"LLM decide error: {exc}"
                logger.warning("[ScreenAgent] decide failed: %s", exc)
                # Try a graceful error response.
                action = {"type": "error", "message": f"Soch nahi paya: {exc}"}
                step.action = action

            # Clean up screenshot immediately after LLM has processed it.
            if png_path and step_num > 0:
                try:
                    os.unlink(png_path)
                except OSError:
                    pass
                step.screenshot_path = None

            await self._send_ipc("agent_step", {
                "step": step_num,
                "action_type": action.get("type"),
                "action": action,
                "status": "executing",
            })

            action_type = action.get("type", "")

            # 4. Terminal states.
            if action_type == "done":
                msg = action.get("message", "Kaam ho gaya.")
                logger.info("[ScreenAgent] Goal achieved at step %d: %s", step_num, msg)
                await self._send_ipc("agent_done", {"message": msg, "steps": step_num})
                return AgentResult(success=True, message=msg, steps=steps, goal=goal)

            if action_type == "error":
                msg = action.get("message", "Kuch error aa gaya.")
                logger.warning("[ScreenAgent] Agent error at step %d: %s", step_num, msg)
                await self._send_ipc("agent_done", {"message": msg, "steps": step_num, "error": True})
                return AgentResult(success=False, message=msg, steps=steps, goal=goal)

            # 5. Execute the action.
            try:
                result_str = await self._execute_action(action)
                step.action_result = result_str
                logger.info("[ScreenAgent] Step %d (%s): %s", step_num, action_type, result_str[:80])
            except Exception as exc:
                step.error = str(exc)
                result_str = f"Action failed: {exc}"
                logger.warning("[ScreenAgent] Step %d action failed: %s", step_num, exc)

            # Add this step to the history so the LLM has context.
            history.append({
                "step": step_num,
                "action": action,
                "result": result_str,
            })

            # 6. Wait for UI to settle before next screenshot.
            wait_secs = float(action.get("seconds", ACTION_SETTLE_SECS)) if action_type == "wait" else ACTION_SETTLE_SECS
            if action_type == "open_app":
                wait_secs = 2.0  # apps take longer to open
            await asyncio.sleep(wait_secs)

        # Exhausted max steps.
        msg = f"Max iterations ({self._max_steps}) pe bhi kaam poora nahi hua. Manually check karein."
        await self._send_ipc("agent_done", {"message": msg, "steps": self._max_steps, "error": True})
        return AgentResult(success=False, message=msg, steps=steps, goal=goal)

    # ------------------------------------------------------------------
    # Vision + reasoning: decide the next action
    # ------------------------------------------------------------------

    async def _decide(
        self,
        goal: str,
        screenshot_path: str,
        ax_text: str,
        history: list[dict],
    ) -> dict:
        """
        Call Gemini (multimodal) or the text LLM to decide the next action.

        Returns a parsed action dict.
        """
        # Build the text context block.
        history_text = ""
        if history:
            lines = []
            for h in history[-5:]:  # last 5 steps in context
                lines.append(
                    f"Step {h['step']}: {json.dumps(h['action'])} → {h['result'][:120]}"
                )
            history_text = "Recent actions:\n" + "\n".join(lines)

        ax_block = f"\nAccessibility tree snapshot:\n{ax_text}" if ax_text and ax_text != "(AX tree unavailable)" else ""

        user_text = (
            f"Goal: {goal}\n"
            f"{history_text}"
            f"{ax_block}\n\n"
            "What is the SINGLE next action to take? Reply with only JSON."
        )

        # Prefer Gemini vision when available (it can actually see the screen).
        if self._gemini_key and screenshot_path and os.path.exists(screenshot_path):
            try:
                raw_json = await asyncio.to_thread(
                    self._gemini_vision_decide,
                    screenshot_path,
                    user_text,
                    self._gemini_key,
                )
                return self._parse_action_json(raw_json)
            except Exception as exc:
                logger.warning("[ScreenAgent] Gemini vision decide failed: %s — falling back to text", exc)

        # Fallback: text-only LLM (less capable but works offline).
        if self._llm is not None:
            raw_json = await self._llm.chat(
                user_text,
                system_prompt=_AGENT_SYSTEM_PROMPT,
            )
            return self._parse_action_json(raw_json)

        raise RuntimeError("No LLM available for agentic reasoning.")

    @staticmethod
    def _gemini_vision_decide(
        screenshot_path: str,
        user_text: str,
        api_key: str,
    ) -> str:
        """
        Synchronous Gemini multimodal call. Runs in a thread executor.

        Sends the PNG screenshot + text prompt to gemini-2.5-flash and
        returns the raw JSON string that the model produces.
        """
        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415

        client = genai.Client(api_key=api_key)

        # Send the screenshot inline as raw PNG bytes.
        with open(screenshot_path, "rb") as fh:
            image_bytes = fh.read()
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[user_text, image_part],
            config=types.GenerateContentConfig(
                system_instruction=_AGENT_SYSTEM_PROMPT,
            ),
        )

        text = getattr(response, "text", None)
        if not text:
            try:
                parts = response.candidates[0].content.parts  # type: ignore[attr-defined]
                text = "".join(getattr(p, "text", "") for p in parts)
            except Exception:
                text = ""
        return (text or "").strip()

    @staticmethod
    def _parse_action_json(raw: str) -> dict:
        """Extract and parse the first JSON object from the LLM response."""
        raw = (raw or "").strip()

        # Strip markdown code fences if any.
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)

        # Find the first {...} block.
        start = raw.find("{")
        if start == -1:
            logger.warning("[ScreenAgent] No JSON in LLM response: %r", raw[:200])
            return {"type": "error", "message": "LLM ne JSON nahi diya."}

        depth = 0
        end = start
        for i, ch in enumerate(raw[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        try:
            action = json.loads(raw[start: end + 1])
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("[ScreenAgent] JSON parse failed: %s — raw: %r", exc, raw[:300])
            return {"type": "error", "message": f"JSON parse error: {exc}"}

        if not isinstance(action, dict) or "type" not in action:
            return {"type": "error", "message": "Invalid action JSON from LLM."}

        return action

    # ------------------------------------------------------------------
    # Accessibility tree snapshot
    # ------------------------------------------------------------------

    async def _read_ax_tree(self) -> str:
        """
        Read a concise text dump of the frontmost window's Accessibility tree.

        Uses ``osascript`` with System Events to enumerate the focused
        window's AX elements.  Returns a formatted text block.

        Falls back to an empty string if Accessibility permission is missing
        or the app does not support AX.
        """
        # AppleScript: get all UI element descriptions from the front window.
        script = r"""
tell application "System Events"
    set frontApp to name of first application process whose frontmost is true
    set frontPID to unix id of first application process whose frontmost is true
    try
        set frontWin to (first window of application process frontApp)
        set elemList to {}
        set allElems to entire contents of frontWin
        repeat with elem in allElems
            try
                set desc to (description of elem) & " [" & (role of elem) & "]"
                if value of elem is not missing value then
                    set desc to desc & " val=" & (value of elem as text)
                end if
                set end of elemList to desc
            end try
        end repeat
        return (elemList as text)
    on error
        return "(no AX data)"
    end try
end tell
"""
        rc, out, _err = await self._mac._run_process("osascript", "-e", script)
        if rc == 0 and out.strip() and out.strip() != "(no AX data)":
            return out.strip()
        return "(AX tree unavailable)"

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    async def _execute_action(self, action: dict) -> str:
        """
        Execute one action dict and return a short result string.

        Each action type dispatches to the appropriate macOS primitive.
        """
        action_type = action.get("type", "")

        if action_type == "open_app":
            return await self._mac.open_app(action.get("app", ""))

        if action_type == "click":
            # Pixel-coordinate click via CGEvent / AppleScript.
            x = int(action.get("x", 0))
            y = int(action.get("y", 0))
            desc = action.get("description", "element")
            result = await self._ax_click_coords(x, y)
            return result or f"Clicked {desc} at ({x}, {y})"

        if action_type == "ax_click":
            ax_id = action.get("ax_id", "")
            desc = action.get("description", ax_id)
            return await self._ax_click_by_id(ax_id, desc)

        if action_type == "type_text":
            text = action.get("text", "")
            send = bool(action.get("send", False))
            return await self._mac.type_text(text, press_return=send)

        if action_type == "keystroke":
            key = action.get("key", "")
            return await self._send_keystroke(key)

        if action_type == "applescript":
            script = action.get("script", "")
            return await self._mac.run_applescript(script)

        if action_type == "scroll":
            direction = action.get("direction", "down")
            amount = int(action.get("amount", 3))
            return await self._scroll(direction, amount)

        if action_type == "wait":
            secs = float(action.get("seconds", 1.0))
            await asyncio.sleep(secs)
            return f"{secs}s wait complete."

        return f"Unknown action type: {action_type!r}"

    # ------------------------------------------------------------------
    # Low-level input helpers
    # ------------------------------------------------------------------

    async def _ax_click_coords(self, x: int, y: int) -> str:
        """Click at pixel coordinates using a CGEvent AppleScript."""
        script = (
            f'tell application "System Events" to '
            f'click at {{x:{x}, y:{y}}}'
        )
        rc, out, err = await self._mac._run_process("osascript", "-e", script)
        if rc == 0:
            return f"Clicked at ({x}, {y})."
        # Fallback: use cliclick if available (more reliable for pixel clicks).
        rc2, _out2, _err2 = await self._mac._run_process("cliclick", f"c:{x},{y}")
        if rc2 == 0:
            return f"cliclick: clicked at ({x}, {y})."
        return f"Click at ({x},{y}) failed: {err.strip()}"

    async def _ax_click_by_id(self, ax_id: str, description: str) -> str:
        """
        Click an AX element identified by its role+description in the tree.

        Tries to locate the element via System Events AX and click it
        using 'click' action on the AXElement.
        """
        # Build an AppleScript that finds and clicks by description/role hint.
        desc_escaped = description.replace('"', '\\"')
        script = (
            'tell application "System Events"\n'
            '  set frontApp to name of first application process whose frontmost is true\n'
            f'  set matchElem to first UI element of (first window of '
            f'application process frontApp) whose description contains "{desc_escaped}"\n'
            '  click matchElem\n'
            'end tell'
        )
        rc, _out, err = await self._mac._run_process("osascript", "-e", script)
        if rc == 0:
            return f"AX clicked: {description}"
        # Couldn't click by description — log and fail gracefully.
        return f"AX click '{description}' failed: {err.strip()}"

    async def _send_keystroke(self, key: str) -> str:
        """
        Send a keystroke (e.g. "return", "escape", "cmd+a") via System Events.

        Supports modifier combos: "cmd+shift+k", "ctrl+c", etc.
        """
        key = (key or "").strip().lower()
        if not key:
            return "No key specified."

        # Key code mapping for special keys.
        _KEY_CODES: dict[str, int] = {
            "return": 36, "enter": 36,
            "escape": 53, "esc": 53,
            "tab": 48,
            "delete": 51, "backspace": 51,
            "space": 49,
            "up": 126, "down": 125, "left": 123, "right": 124,
            "f1": 122, "f2": 120, "f3": 99, "f4": 118,
        }

        # Parse modifier+key combos.
        parts = key.split("+")
        base_key = parts[-1].strip()
        modifiers = [m.strip() for m in parts[:-1]] if len(parts) > 1 else []

        mod_map = {
            "cmd": "command down", "command": "command down",
            "ctrl": "control down", "control": "control down",
            "shift": "shift down",
            "alt": "option down", "opt": "option down", "option": "option down",
        }

        using_clause = ""
        if modifiers:
            mod_list = [mod_map.get(m, f"{m} down") for m in modifiers]
            using_clause = " using {" + ", ".join(mod_list) + "}"

        if base_key in _KEY_CODES:
            # Use key code for special keys.
            code = _KEY_CODES[base_key]
            script = f'tell application "System Events" to key code {code}{using_clause}'
        else:
            # Use keystroke for regular characters.
            esc = base_key.replace('"', '\\"')
            script = f'tell application "System Events" to keystroke "{esc}"{using_clause}'

        rc, _out, err = await self._mac._run_process("osascript", "-e", script)
        if rc == 0:
            return f"Keystroke sent: {key}"
        return f"Keystroke {key!r} failed: {err.strip()}"

    async def _scroll(self, direction: str, amount: int) -> str:
        """Scroll the frontmost window via System Events."""
        direction = (direction or "down").lower()
        amount = max(1, min(10, amount))
        # Positive delta = up, negative = down in AppleScript scroll.
        delta = amount if direction == "up" else -amount
        script = (
            'tell application "System Events"\n'
            f'  scroll (first window of (first application process whose frontmost is true)) '
            f'by {delta}\n'
            'end tell'
        )
        rc, _out, err = await self._mac._run_process("osascript", "-e", script)
        if rc == 0:
            return f"Scrolled {direction} by {amount}."
        # Fallback: send scroll keystroke.
        key = "up" if direction == "up" else "down"
        for _ in range(amount):
            await self._send_keystroke(key)
        return f"Scrolled {direction} x{amount} via keys."

    # ------------------------------------------------------------------
    # IPC progress events
    # ------------------------------------------------------------------

    async def _send_ipc(self, event_type: str, payload: dict) -> None:
        """Send an AGENT_* IPC event to the Swift UI (best-effort)."""
        if self._ipc_writer is None:
            return
        try:
            await self._ipc_writer({
                "type": "AGENT_EVENT",
                "payload": {"event_type": event_type, **payload},
            })
        except Exception as exc:
            logger.debug("[ScreenAgent] IPC send failed: %s", exc)
