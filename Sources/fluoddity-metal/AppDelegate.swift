import AppKit
import MetalKit

// Owns the window, the Metal view, the on-screen slider panel, and the
// (secondary) keyboard tuning. Instantiated at top level in main.swift and
// held for the program's lifetime.
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow!
    private var mtkView: EngineView!
    private var renderer: Renderer!
    private var controls: ControlsPanel!
    private let params = Params()
    private let mouse = MouseInput()
    private var tuning: Tuning!
    private var loadIndex = -1

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device available.")
        }

        let frame = NSRect(x: 0, y: 0, width: 1024, height: 1024)
        let container = NSView(frame: frame)
        container.autoresizesSubviews = true

        mtkView = EngineView(frame: frame, device: device)
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.preferredFramesPerSecond = 60
        mtkView.autoresizingMask = [.width, .height]

        tuning = Tuning(params: params)
        mtkView.onKey = { [weak self] key in
            guard let self else { return }
            if key == "r" { self.renderer?.reroll() } else { self.tuning.handleKey(key) }
        }
        mtkView.mouseInput = mouse

        renderer = Renderer(device: device, pixelFormat: mtkView.colorPixelFormat,
                            params: params, mouse: mouse)
        mtkView.delegate = renderer
        container.addSubview(mtkView)

        // slider panel, pinned top-left
        controls = ControlsPanel(params: params, knobs: engineKnobs)
        controls.setFrameOrigin(NSPoint(x: 12, y: frame.height - controls.frame.height - 12))
        controls.autoresizingMask = [.maxXMargin, .minYMargin]
        container.addSubview(controls)

        controls.onReset  = { [weak self] in self?.renderer?.reset() }
        controls.onReroll = { [weak self] in self?.renderer?.reroll() }
        controls.onSave   = { [weak self] in self?.savePreset() }
        controls.onLoad   = { [weak self] in self?.loadNextPreset() }

        window = NSWindow(contentRect: frame,
                          styleMask: [.titled, .closable, .resizable, .miniaturizable],
                          backing: .buffered,
                          defer: false)
        window.title = "fluoddity-metal"
        window.contentView = container
        window.center()
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(mtkView)   // keyboard tuning (secondary)
    }

    // ── presets ─────────────────────────────────────────────────────────
    private func savePreset() {
        let dict = Dictionary(uniqueKeysWithValues:
            engineKnobs.map { ($0.name, params[keyPath: $0.kp]) })
        let data = PresetData(params: dict, rule: renderer.ruleSnapshot())
        if let url = Presets.save(data) { print("saved \(url.lastPathComponent)") }
    }

    private func loadNextPreset() {
        let files = Presets.list()
        guard !files.isEmpty else { print("no presets saved yet"); return }
        loadIndex = (loadIndex + 1) % files.count
        guard let p = Presets.load(files[loadIndex]) else { return }
        for k in engineKnobs { if let v = p.params[k.name] { params[keyPath: k.kp] = v } }
        renderer.loadRule(p.rule)
        controls.refresh()
        print("loaded \(files[loadIndex].lastPathComponent)")
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
