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
if CommandLine.arguments.contains("--capturetest") {
    runCaptureTest()
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
if CommandLine.arguments.contains("--map3") {
    runMap3()
    exit(0)
}
if CommandLine.arguments.contains("--sdscan") {
    runSdScan()
    exit(0)
}
if CommandLine.arguments.contains("--bistab") {
    runBistab()
    exit(0)
}
if CommandLine.arguments.contains("--vortprobe") {
    runVortProbe()
    exit(0)
}
if CommandLine.arguments.contains("--3dspec") {
    run3dSpec()
    exit(0)
}

let app = NSApplication.shared
// `--3d` launches the 3D engine; default is the 2D app. Both delegates conform
// to NSApplicationDelegate; the top-level `let` keeps the chosen one retained.
let appDelegate: NSApplicationDelegate =
    CommandLine.arguments.contains("--3d") ? App3D() : AppDelegate()
app.delegate = appDelegate
app.setActivationPolicy(.regular)

// Minimal menu bar — a SwiftPM AppKit app gets none by default, which means
// no ⌘Q / ⌘H / ⌘M. One app menu with the standard items fixes that.
let mainMenu = NSMenu()
let appMenuItem = NSMenuItem()
mainMenu.addItem(appMenuItem)
let appMenu = NSMenu()
appMenu.addItem(withTitle: "About fluoddity-metal",
                action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)),
                keyEquivalent: "")
appMenu.addItem(.separator())
appMenu.addItem(withTitle: "Hide fluoddity-metal",
                action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
appMenu.addItem(.separator())
appMenu.addItem(withTitle: "Quit fluoddity-metal",
                action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
appMenuItem.submenu = appMenu
let windowMenuItem = NSMenuItem()
mainMenu.addItem(windowMenuItem)
let windowMenu = NSMenu(title: "Window")
windowMenu.addItem(withTitle: "Minimize",
                   action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
windowMenuItem.submenu = windowMenu
app.mainMenu = mainMenu
app.windowsMenu = windowMenu

app.activate(ignoringOtherApps: true)
app.run()
