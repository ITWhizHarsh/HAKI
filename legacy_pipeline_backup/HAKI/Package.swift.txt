// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "HAKI",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "HAKI",
            targets: ["HAKI"]
        )
    ],
    dependencies: [
        // SwiftCheck — property-based testing for Swift
        .package(
            url: "https://github.com/typelift/SwiftCheck.git",
            from: "0.12.0"
        ),
        // SQLite.swift — structured local database
        .package(
            url: "https://github.com/stephencelis/SQLite.swift.git",
            from: "0.15.0"
        )
    ],
    targets: [
        // MARK: - Main app target
        .executableTarget(
            name: "HAKI",
            dependencies: ["HAKIIPC", "HAKIUI", "HAKIAudio", "HAKIPermissions"],
            path: "Sources/HAKI",
            swiftSettings: [
                .define("DEBUG", .when(configuration: .debug))
            ]
        ),

        // MARK: - Library targets (one per subsystem)
        .target(
            name: "HAKIAudio",
            dependencies: ["HAKIIPC"],
            path: "Sources/Subsystems/Audio"
        ),
        .target(
            name: "HAKICapture",
            dependencies: ["HAKIAudio", "HAKIPermissions"],
            path: "Sources/Subsystems/Capture"
        ),
        .target(
            name: "HAKIOSActions",
            dependencies: ["HAKIPermissions"],
            path: "Sources/Subsystems/OSActions"
        ),
        .target(
            name: "HAKIPermissions",
            dependencies: [],
            path: "Sources/Subsystems/Permissions"
        ),
        .target(
            name: "HAKIIPC",
            dependencies: [],
            path: "Sources/Subsystems/IPC"
        ),
        .target(
            name: "HAKIUI",
            dependencies: ["HAKIIPC"],
            path: "Sources/Subsystems/UI"
        ),
        .target(
            name: "HAKIStore",
            dependencies: [.product(name: "SQLite", package: "sqlite.swift")],
            path: "Sources/Subsystems/Store"
        ),
        .target(
            name: "HAKITextInput",
            dependencies: [],
            path: "Sources/Subsystems/TextInput"
        ),
        .target(
            name: "HAKIControl",
            dependencies: ["HAKIOSActions", "HAKIPermissions"],
            path: "Sources/Subsystems/Control"
        ),
        .target(
            name: "HAKIScheduler",
            dependencies: ["HAKIOSActions"],
            path: "Sources/Subsystems/Scheduler"
        ),

        // MARK: - Test targets
        .testTarget(
            name: "HAKITests",
            dependencies: [
                "HAKI",
                "HAKIAudio",
                "HAKICapture",
                "HAKIOSActions",
                "HAKIPermissions",
                "HAKIIPC",
                "HAKIUI",
                "HAKIStore",
                "HAKITextInput",
                "HAKIControl",
                "HAKIScheduler"
            ],
            path: "Tests/HAKITests"
        ),
        .testTarget(
            name: "HAKIPropertyTests",
            dependencies: [
                "HAKI",
                "HAKIAudio",
                "HAKICapture",
                "HAKIOSActions",
                "HAKIPermissions",
                "HAKIIPC",
                "HAKIUI",
                "HAKIStore",
                "HAKITextInput",
                "HAKIControl",
                "HAKIScheduler",
                .product(name: "SwiftCheck", package: "SwiftCheck")
            ],
            path: "Tests/HAKIPropertyTests"
        )
    ]
)
