// swift-tools-version:6.0
import PackageDescription

// fluoddity-metal — a Metal-native fork of the Fluoddity flow-field engine.
// Builds with Command Line Tools only (no full Xcode): runtime-compiled shaders,
// SwiftPM executable. See README.md for the project intent.
let package = Package(
    name: "fluoddity-metal",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "fluoddity-metal",
            path: "Sources/fluoddity-metal",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation"),
            ]
        )
    ]
)
