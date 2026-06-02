// CoreProcessManager.swift
// HAKI — Swift / SwiftUI Shell
//
// Manages the lifecycle of the HAKI Core (Python) child process.
//
// The Core is bundled at:
//   HAKI.app/Contents/Resources/haki_core/haki_core_service
//
// It is launched as a child of this process, inheriting the sandbox, and
// communicates over a UNIX domain socket at the path returned by `socketPath`.
// If the Core exits unexpectedly it is restarted with exponential back-off
// (max 5 attempts before giving up and notifying the user).

import Foundation

/// Manages spawn, health-check, and teardown of the Python HAKI Core process.
final class CoreProcessManager {

    // MARK: - Constants

    /// Path to the bundled Core executable, relative to the app bundle's Resources directory.
    private static let coreBundleRelativePath = "haki_core/haki_core_service"

    /// Maximum number of automatic restart attempts before giving up.
    private static let maxRestartAttempts = 5

    // MARK: - State

    private var process: Process?
    private var restartAttempts = 0

    // MARK: - Callbacks

    /// Called on the main queue once the Core socket is ready and the IPC
    /// client should connect.  Set this before calling `start()`.
    var onCoreReady: (() -> Void)?

    // MARK: - Public interface

    /// Spawn the Core process.
    func start() {
        guard let executableURL = resolveExecutableURL() else {
            // Core not yet bundled (development mode). No-op.
            print("[CoreProcessManager] Core executable not found — running shell-only (development mode).")
            return
        }

        launchProcess(at: executableURL)
    }

    /// Terminate the Core process gracefully.
    func stop() {
        process?.terminate()
        process = nil
    }

    // MARK: - IPC socket path

    /// UNIX domain socket path scoped to this app's container.
    var socketPath: URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? URL(fileURLWithPath: NSTemporaryDirectory())

        return appSupport
            .appendingPathComponent("HAKI", isDirectory: true)
            .appendingPathComponent("haki_core.sock")
    }

    // MARK: - Private helpers

    private func resolveExecutableURL() -> URL? {
        guard
            let resourcesURL = Bundle.main.resourceURL
        else { return nil }

        let execURL = resourcesURL
            .appendingPathComponent(Self.coreBundleRelativePath)

        return FileManager.default.fileExists(atPath: execURL.path) ? execURL : nil
    }

    private func launchProcess(at url: URL) {
        let proc = Process()
        proc.executableURL = url
        proc.arguments = ["--socket", socketPath.path]
        proc.environment = ProcessInfo.processInfo.environment

        proc.terminationHandler = { [weak self] terminatedProcess in
            self?.handleTermination(of: terminatedProcess)
        }

        do {
            try proc.run()
            process = proc
            restartAttempts = 0
            print("[CoreProcessManager] Core process started (PID \(proc.processIdentifier)).")
            // Poll for the socket to appear, then notify the IPC client
            waitForSocketReady(path: socketPath, timeout: 5.0) { [weak self] in
                self?.onCoreReady?()
            }
        } catch {
            print("[CoreProcessManager] Failed to start Core process: \(error).")
        }
    }

    /// Poll for the UNIX socket file to appear, up to *timeout* seconds.
    /// Calls *completion* on the main queue when ready, or logs a warning on timeout.
    ///
    /// - Parameters:
    ///   - path: The socket URL to poll.
    ///   - timeout: Maximum wait in seconds (default 5).
    ///   - completion: Called when the socket file is detected.
    private func waitForSocketReady(path: URL, timeout: Double, completion: @escaping () -> Void) {
        let deadline = Date(timeIntervalSinceNow: timeout)
        let pollInterval = 0.1  // 100 ms

        func poll() {
            if FileManager.default.fileExists(atPath: path.path) {
                print("[CoreProcessManager] Core socket ready at \(path.path).")
                DispatchQueue.main.async { completion() }
                return
            }
            if Date() > deadline {
                print("[CoreProcessManager] Timed out waiting for Core socket at \(path.path).")
                return
            }
            DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + pollInterval) {
                poll()
            }
        }

        DispatchQueue.global(qos: .utility).async { poll() }
    }

    private func handleTermination(of proc: Process) {
        print("[CoreProcessManager] Core process exited with status \(proc.terminationStatus).")

        guard proc.terminationStatus != 0 else {
            // Clean exit — app is quitting.
            return
        }

        restartAttempts += 1
        guard restartAttempts <= Self.maxRestartAttempts else {
            print("[CoreProcessManager] Core process failed \(Self.maxRestartAttempts) times. Giving up.")
            // TODO: surface a user notification via NSUserNotificationCenter
            return
        }

        let delay = pow(2.0, Double(restartAttempts - 1))
        print("[CoreProcessManager] Restarting Core in \(delay)s (attempt \(restartAttempts)/\(Self.maxRestartAttempts))…")
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self, let url = self.resolveExecutableURL() else { return }
            self.launchProcess(at: url)
        }
    }
}
