import AppKit

// ───────────────────────────────────────────────────────────────────────────
// fluoddity-metal — entry point.
//
//   swift run fluoddity-metal           → opens the Metal window (the engine)
//   swift run fluoddity-metal --smoke   → headless compute/atomic_float smoke
//
// The windowed path needs a GUI session. The --smoke path is headless and is
// what a remote agent / CI runs to verify the toolchain end-to-end.
// ───────────────────────────────────────────────────────────────────────────

if CommandLine.arguments.contains("--smoke") {
    runSmoke()
    exit(0)
}

let app = NSApplication.shared
let appDelegate = AppDelegate()        // top-level `let` → retained for program lifetime
app.delegate = appDelegate
app.setActivationPolicy(.regular)
app.activate(ignoringOtherApps: true)
app.run()
