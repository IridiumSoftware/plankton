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

let app = NSApplication.shared
let appDelegate = AppDelegate()        // top-level `let` → retained for program lifetime
app.delegate = appDelegate
app.setActivationPolicy(.regular)
app.activate(ignoringOtherApps: true)
app.run()
