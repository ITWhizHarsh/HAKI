#!/usr/bin/env bash
# HAKI Unified Startup Script
# Usage: ./start_haki.sh

set -e  # Exit on first error

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🧠 HAKI — Personal AI Assistant Startup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# ────────────────────────────────────────────────────────────────────
# 1. Load environment variables
# ────────────────────────────────────────────────────────────────────
echo "1. Loading API keys…"
# Single source of truth for all keys: Core/.env  (loaded by the Core too).
if [ -f "Core/.env" ]; then
    set -a                # export every variable defined while sourcing
    source Core/.env
    set +a
    echo "   ✓ Loaded keys from Core/.env"
else
    echo "⚠️  Core/.env not found — no API keys loaded"
    echo "   Cloud LLM/STT/TTS will fail without keys. See Core/.env.example."
fi
echo

# ────────────────────────────────────────────────────────────────────
# 1b. Resolve the Core Python interpreter (single source of truth)
#     The Swift app's CoreProcessManager also uses Core/.venv, so we
#     must use the SAME interpreter here or dependencies won't match.
# ────────────────────────────────────────────────────────────────────
VENV_PY="$PROJECT_ROOT/Core/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "   📦 Core/.venv not found — creating it…"
    python3 -m venv "$PROJECT_ROOT/Core/.venv"
    "$VENV_PY" -m pip install --upgrade pip >/dev/null
fi
echo

# ────────────────────────────────────────────────────────────────────
# 2. Create Obsidian vault structure
# ────────────────────────────────────────────────────────────────────
echo "2. Setting up Obsidian vault…"
if [ -z "$HAKI_OBSIDIAN_VAULT" ]; then
    HAKI_OBSIDIAN_VAULT="$HOME/Obsidian/HAKI_Brain"
    export HAKI_OBSIDIAN_VAULT
fi
mkdir -p "$HAKI_OBSIDIAN_VAULT"/{raw,processed,wiki}
echo "   ✓ Vault: $HAKI_OBSIDIAN_VAULT/{raw,processed,wiki}"
echo

# ────────────────────────────────────────────────────────────────────
# 3. Check system dependencies
# ────────────────────────────────────────────────────────────────────
echo "3. Checking system dependencies…"
if ! command -v espeak-ng &> /dev/null; then
    echo "   ⚠️  espeak-ng not found (required for Kokoro TTS phonemizer)"
    echo "   Install: brew install espeak-ng"
    echo "   Continuing anyway — TTS may fall back to Cartesia cloud…"
else
    echo "   ✓ espeak-ng found"
fi
echo

# ────────────────────────────────────────────────────────────────────
# 4. Install Python dependencies if needed
# ────────────────────────────────────────────────────────────────────
echo "4. Checking Python dependencies…"
# Probe a few representative packages across tiers (IPC + local STT). If any
# are missing we (re)install the full requirements file. Pip is idempotent, so
# this is a no-op when everything is already present.
if ! "$VENV_PY" -c "import grpc, mlx_whisper" 2>/dev/null; then
    echo "   📦 Installing Core/requirements.txt (this may take a few minutes)…"
    "$VENV_PY" -m pip install -r Core/requirements.txt
    echo "   ✓ Dependencies installed"
else
    echo "   ✓ Dependencies already installed"
fi
echo

# ────────────────────────────────────────────────────────────────────
# 5. Build Swift app and assemble .app bundle
# ────────────────────────────────────────────────────────────────────
echo "5. Building Swift app…"
cd HAKI

if [ ! -f ".build/release/HAKI" ]; then
    echo "   🔨 Running: swift build --configuration release"
    swift build --configuration release
    echo "   ✓ Swift build complete"
else
    echo "   ✓ Swift already built"
    echo "   (To rebuild: rm HAKI/.build/release/HAKI then re-run)"
fi

# Reassemble the bundle so plist/binary changes are picked up
APP_BUNDLE=".build/arm64-apple-macosx/release/HAKI.app"
echo "   📦 Assembling ${APP_BUNDLE}…"
rm -rf "${APP_BUNDLE}"
mkdir -p "${APP_BUNDLE}/Contents/MacOS"
mkdir -p "${APP_BUNDLE}/Contents/Resources"
cp ".build/release/HAKI"                             "${APP_BUNDLE}/Contents/MacOS/HAKI"
cp "Sources/HAKI/Resources/Info.plist"               "${APP_BUNDLE}/Contents/Info.plist"
cp "Sources/HAKI/Resources/HAKI.entitlements"        "${APP_BUNDLE}/Contents/Resources/HAKI.entitlements"

echo "   🔏 Ad-hoc codesigning…"
codesign --force --deep --sign - \
    --entitlements "Sources/HAKI/Resources/HAKI.entitlements" \
    "${APP_BUNDLE}" 2>&1 | sed 's/^/   /'

# Install to /Applications so macOS LaunchServices uses a stable,
# trusted path for TCC grants (microphone, etc.)
echo "   📲 Installing to /Applications/HAKI.app…"
rm -rf /Applications/HAKI.app
cp -R "${APP_BUNDLE}" /Applications/HAKI.app
echo "   ✓ Installed at /Applications/HAKI.app"

cd ..
echo

# ────────────────────────────────────────────────────────────────────
# 6. Export API keys into the per-user launchd environment so that
#    apps launched via LaunchServices (open command) inherit them.
#    This is necessary because `open` does not inherit shell env vars.
# ────────────────────────────────────────────────────────────────────
echo "6. Exporting environment to launchd…"
_export_to_launchd() {
    local key="$1"
    local val="${!key}"
    if [ -n "$val" ]; then
        launchctl setenv "$key" "$val"
        echo "   ✓ $key"
    fi
}
# Export the HAKI_-prefixed keys the Core actually reads (see haki_core_service.py)
_export_to_launchd HAKI_GROQ_API_KEY
_export_to_launchd HAKI_CEREBRAS_API_KEY
_export_to_launchd HAKI_GEMINI_API_KEY
_export_to_launchd HAKI_DEEPGRAM_API_KEY
_export_to_launchd HAKI_CARTESIA_API_KEY
_export_to_launchd HAKI_OBSIDIAN_VAULT
echo

# ────────────────────────────────────────────────────────────────────
# 7. Start Python Core service in background
# ────────────────────────────────────────────────────────────────────
echo "7. Starting Python Core service…"
SOCKET_PATH="$HOME/Library/Application Support/HAKI/haki_core.sock"
mkdir -p "$HOME/Library/Application Support/HAKI"

# Kill any stale socket from a previous run
rm -f "$SOCKET_PATH"

cd Core
"$VENV_PY" -m haki_core_service --socket "$SOCKET_PATH" --transport json &
CORE_PID=$!
echo "   ✓ Core service started (PID: $CORE_PID)"
cd ..
echo

# ────────────────────────────────────────────────────────────────────
# 8. Wait for IPC socket to be ready
# ────────────────────────────────────────────────────────────────────
echo "8. Waiting for Core server to be fully initialized…"
TIMEOUT=15
for i in $(seq 1 $TIMEOUT); do
    if [ -S "$SOCKET_PATH" ]; then
        sleep 1  # let server fully bind
        echo "   ✓ Core service is ready at: $SOCKET_PATH"
        break
    fi
    if [ $i -eq $TIMEOUT ]; then
        echo "   ⚠️  Socket not found after ${TIMEOUT}s — Core may have failed."
        echo "   Check logs: tail -f Core/logs/haki_core.log"
        kill $CORE_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done
echo

# ────────────────────────────────────────────────────────────────────
# 9. Launch HAKI.app via LaunchServices (open command)
#
#    IMPORTANT: We MUST use `open` here, not `exec` or a direct path.
#    Running the binary directly from a shell means macOS attributes
#    TCC permissions (microphone, etc.) to the terminal/parent process,
#    not to HAKI. `open` goes through LaunchServices which correctly
#    registers the app under its bundle ID (com.haki.app) so that:
#      - The microphone prompt says "HAKI" not "Terminal"
#      - AVAudioEngine can start capture without hardwareUnavailable
# ────────────────────────────────────────────────────────────────────
APP_BUNDLE_ABS="/Applications/HAKI.app"

echo "9. Launching HAKI.app via LaunchServices…"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  🧠 HAKI is starting up…"
echo "  🎤 Grant microphone permission when prompted (should say 'HAKI')"
echo "  🔇 To stop: press Ctrl+C here (shuts down Core too)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Cleanup: kill Core and HAKI when this script exits (Ctrl+C or normal exit)
trap "echo; echo '🛑 Shutting down…'; pkill -f 'HAKI.app/Contents/MacOS/HAKI' 2>/dev/null; kill $CORE_PID 2>/dev/null; exit 0" EXIT INT TERM

# Kill any previously running HAKI instance
pkill -f "HAKI.app/Contents/MacOS/HAKI" 2>/dev/null || true
sleep 0.5

# Launch via open so LaunchServices registers it correctly for TCC
# This ensures the microphone prompt and menu bar dot say "HAKI", not Kiro/Terminal
open "$APP_BUNDLE_ABS"
echo "   ✓ HAKI.app launched via LaunchServices"
echo "   (Check /Applications/HAKI.app for the running process)"
echo "   Press Ctrl+C here to shut down the Core."

# Keep the script running until Ctrl+C
wait $CORE_PID
