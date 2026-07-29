import Foundation

/// Internal development replacement gate.
///
/// Mirrors the Python ``core.voice.dev_gate`` module so both the Swift and
/// Python sides of the voice pipeline read from the same environment variable.
///
/// Gate rules (Design §11 step 3, Requirements 1.5–1.6):
/// - When ``isEnabled`` is ``true``: only the new local path
///   (``VoiceSessionCompositionRoot``) is activated.  No legacy STT/TTS
///   engine, ``afplay``, ``say``, Deepgram, or Groq voice route is used.
/// - When ``isEnabled`` is ``false``: the existing non-voice IPC handlers
///   and app routing are left completely untouched.
/// - Non-voice IPC handlers are always preserved regardless of gate state.
///
/// This file contains no archive imports and no references to legacy voice
/// components.
public enum VoiceDevGate {
    /// ``true`` only when ``HAKI_VOICE_DEV_REPLACEMENT=1`` is present in the
    /// process environment.  Evaluated once at app startup.
    public static let isEnabled: Bool = {
        ProcessInfo.processInfo.environment["HAKI_VOICE_DEV_REPLACEMENT"] == "1"
    }()
}
