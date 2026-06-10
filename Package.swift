// swift-tools-version:6.0
import PackageDescription

// plankton — a Metal-native fork of the Fluoddity flow-field engine.
// Builds with Command Line Tools only (no full Xcode): runtime-compiled shaders,
// SwiftPM executable. See README.md for the project intent.
let package = Package(
    name: "plankton",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "plankton",
            path: "Sources/plankton",
            swiftSettings: [
                // Swift 5 language mode for now — the app is main-thread-only
                // (AppKit + Metal); adopt Swift 6 strict concurrency once the
                // architecture settles.
                .swiftLanguageMode(.v5)
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("AppKit"),
                .linkedFramework("QuartzCore"),
                .linkedFramework("Foundation"),
                .linkedFramework("AVFoundation"),   // mp4 recording (AVAssetWriter)
                .linkedFramework("CoreMedia"),       // CMTime
                .linkedFramework("CoreVideo"),       // CVPixelBuffer
                .linkedFramework("ImageIO"),         // animated-GIF export
            ]
        )
    ]
)
