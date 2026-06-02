// HAKI — Heuristic Augmented Knowledge Interface
// main.swift — App entry point
//
// This is the Swift shell ("Body") of the two-process HAKI architecture.
// The Body owns: TCC permissions, ScreenCaptureKit, Accessibility, Vision OCR,
// EventKit, AppleScript/Apple Events bridge, global hotkey/Wake_Invocation,
// audio I/O (AVAudioEngine), the menu-bar UI, notifications, and the secure
// on-device store.
//
// The "Mind" (HAKI Core, a local Python service) is spawned as a child process
// and communicates over a UNIX domain socket via gRPC/JSON-RPC streaming.
//
// Requirements: 20.1 (macOS native shell)

import AppKit

// Create and start the application on the main thread.
// NSApplicationMain is called from here because we use a SwiftPM executable target
// (which does not support @NSApplicationMain attribute in a library target).
autoreleasepool {
    let app = NSApplication.shared
    let delegate = AppDelegate()
    app.delegate = delegate
    app.run()
}
