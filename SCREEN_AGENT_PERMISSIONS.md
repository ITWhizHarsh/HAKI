# HAKI Screen Agent — macOS Permissions Required

The agentic screen control (ScreenAgent) needs three macOS permissions
granted to whichever process runs HAKI (the app bundle, or Terminal/iTerm
if you're running via `start_haki.sh`).

## 1. Screen Recording (required for every agentic turn)

> System Settings → Privacy & Security → Screen Recording

Enable for: **Terminal** (or your IDE / HAKI app bundle).

Without this `screencapture` returns a blank image and HAKI cannot see
anything on screen.

## 2. Accessibility (required for clicking, typing, AX tree reading)

> System Settings → Privacy & Security → Accessibility

Enable for: **Terminal** (or your IDE / HAKI app bundle).

Without this `osascript` System Events commands (click, keystroke,
scroll, AX tree enumeration) silently fail.

## 3. Automation (required for controlling specific apps via AppleScript)

> System Settings → Privacy & Security → Automation

Enable the apps you want HAKI to control, e.g.:
- Terminal → control **System Events** ✓
- Terminal → control **Messages** ✓
- Terminal → control **Contacts** ✓
- Terminal → control **Calendar** ✓

## Quick test after granting permissions

```bash
# Should print something other than blank/error:
screencapture -x /tmp/test_cap.png && ls -lh /tmp/test_cap.png

# Should print all UI element names of the frontmost window:
osascript -e 'tell application "System Events"
  set frontApp to name of first application process whose frontmost is true
  return (name of every UI element of first window of application process frontApp)
end tell'
```

## Notes

- Gemini API key (`HAKI_GEMINI_API_KEY`) must be set for vision analysis.
  Without it HAKI falls back to text-only AX reasoning (still works but
  less accurate for visually complex UIs).

- Set `HAKI_AGENT_NO_CLOUD=1` to disable screenshot uploads to Gemini
  entirely (AX-only mode — works for standard apps like Discord/WhatsApp
  but may struggle on custom/games UI).

- The agentic loop runs up to 12 steps before giving up. Complex goals
  may need more — adjust `MAX_STEPS` in `core/automation/screen_agent.py`.
