import AppKit

// ───────────────────────────────────────────────────────────────────────────
// fluoddity-metal — entry point.
//
//   swift run fluoddity-metal             → opens the Metal window (the engine)
//   swift run fluoddity-metal --simtest   → headless: run the compute pipeline,
//                                           verify the field (no window)
//   swift run fluoddity-metal --smoke     → headless atomic_float toolchain test
//
// The windowed path needs a GUI session. The --simtest / --smoke paths are
// headless and are what a remote agent / CI runs to verify the engine.
// ───────────────────────────────────────────────────────────────────────────

if CommandLine.arguments.contains("--smoke") {
    runSmoke()
    exit(0)
}
if CommandLine.arguments.contains("--simtest") {
    runSimTest()
    exit(0)
}
if CommandLine.arguments.contains("--3dtest") {
    runSim3DTest()
    exit(0)
}
if CommandLine.arguments.contains("--spectest") {
    runSpecTest()
    exit(0)
}
if CommandLine.arguments.contains("--sweep") {
    runSweep()
    exit(0)
}
if CommandLine.arguments.contains("--map") {
    runMap()
    exit(0)
}

let app = NSApplication.shared
// `--3d` launches the 3D engine; default is the 2D app. Both delegates conform
// to NSApplicationDelegate; the top-level `let` keeps the chosen one retained.
let appDelegate: NSApplicationDelegate =
    CommandLine.arguments.contains("--3d") ? App3D() : AppDelegate()
app.delegate = appDelegate
app.setActivationPolicy(.regular)
app.activate(ignoringOtherApps: true)
app.run()
