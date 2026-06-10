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
    private var hud: NSTextField!
    private var plot: PlotView!
    private var spectrum: SpectrumView!
    private var help: HelpView!
    private var captureLabel: NSTextField!
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
        mtkView.framebufferOnly = false        // lets the recorder blit the drawable
        mtkView.autoresizingMask = [.width, .height]

        tuning = Tuning(params: params)
        mtkView.onKey = { [weak self] key in
            guard let self else { return }
            switch key {
            case "r": self.renderer?.reroll()
            case "c": self.renderer?.captureCreature()                       // capture creature (full state)
            case "x": self.renderer?.restoreCreature(); self.controls?.refresh()  // restore (cycles captures)
            case "j": self.renderer?.recordPathToggle()                      // record path (toggle)
            case "k": self.renderer?.replayLastPath(); self.controls?.refresh()   // replay latest path
            case "v": self.renderer?.toggleVideo(size: self.mtkView.drawableSize)  // record mp4 (toggle)
            case "g": self.renderer?.toggleGIF(size: self.mtkView.drawableSize)    // record gif (toggle)
            default: self.tuning.handleKey(key)
            }
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
        controls.onToggleDiag = { [weak self] on in
            guard let self else { return }
            self.params.diagnosticsOn = on
            self.hud.isHidden = !on
            self.plot.isHidden = !on
            self.spectrum.isHidden = !on
        }
        controls.onResetAvg = { [weak self] in self?.renderer?.resetSpectrumAvg() }

        // research HUD (top-right): live E / Z / |ω|max / div readout
        hud = NSTextField(frame: NSRect(x: frame.width - 392, y: frame.height - 34, width: 380, height: 22))
        hud.isEditable = false
        hud.isBordered = false
        hud.drawsBackground = true
        hud.backgroundColor = NSColor(white: 0, alpha: 0.45)
        hud.textColor = NSColor(calibratedRed: 0.6, green: 0.9, blue: 1.0, alpha: 1)
        hud.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        hud.alignment = .right
        hud.autoresizingMask = [.minXMargin, .minYMargin]
        container.addSubview(hud)

        // E/Z time-series plot (below the HUD)
        plot = PlotView(frame: NSRect(x: frame.width - 312, y: frame.height - 132, width: 300, height: 88))
        plot.autoresizingMask = [.minXMargin, .minYMargin]
        container.addSubview(plot)

        // energy spectrum E(k) — log-log with a -5/3 reference (below the E/Z plot)
        spectrum = SpectrumView(frame: NSRect(x: frame.width - 312, y: frame.height - 252, width: 300, height: 108))
        spectrum.autoresizingMask = [.minXMargin, .minYMargin]
        container.addSubview(spectrum)
        renderer.onSpectrum = { [weak self] ek, frames in self?.spectrum.update(ek, frames) }

        renderer.onDiagnostics = { [weak self] d in
            self?.hud.stringValue = String(format: "E %.3f    Z %.3f    |\u{03C9}|max %.2f    div %.4f",
                                           Double(d.e), Double(d.z), Double(d.maxW), Double(d.div))
            self?.plot.push(d.e, d.z)
        }

        // keyboard-help button (bottom-left; expands upward)
        help = HelpView(text: """
        drag         stir fluid + inject dye
        right-click  adopt + mutate cohort (breed)
        r            re-roll brains
        c / x        capture creature / restore (cycle captures)
        j / k        record path (toggle) / replay last path
        v / g        record mp4 / gif clip (toggle) → captures/video
        [ ] / - =    keyboard tuning (sliders are primary)
        space        print all params
        """)
        help.setFrameOrigin(NSPoint(x: 12, y: 12))
        help.autoresizingMask = [.maxXMargin, .maxYMargin]
        container.addSubview(help)

        // capture-zoo status (bottom-right): which creature slot is loaded/saved
        captureLabel = NSTextField(frame: NSRect(x: frame.width - 412, y: 12, width: 400, height: 22))
        captureLabel.isEditable = false
        captureLabel.isBordered = false
        captureLabel.drawsBackground = true
        captureLabel.backgroundColor = NSColor(white: 0, alpha: 0.55)
        captureLabel.textColor = NSColor(calibratedRed: 1.0, green: 0.82, blue: 0.4, alpha: 1)  // amber (≠ cyan HUD)
        captureLabel.font = .monospacedSystemFont(ofSize: 11, weight: .medium)
        captureLabel.alignment = .right
        captureLabel.stringValue = "capture: press c to save · x to load"
        captureLabel.autoresizingMask = [.minXMargin, .maxYMargin]
        container.addSubview(captureLabel)
        renderer.onCaptureStatus = { [weak self] s in self?.captureLabel.stringValue = s }

        window = NSWindow(contentRect: frame,
                          styleMask: [.titled, .closable, .resizable, .miniaturizable],
                          backing: .buffered,
                          defer: false)
        window.title = "plankton"
        window.contentView = container
        window.center()
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(mtkView)   // keyboard tuning (secondary)

        // Smoke hook for the live record path (no keystrokes needed): with
        // PLANKTON_AUTOREC set, record ~2.4 s of mp4 then quit. Lets the GUI
        // blit→encode path be verified non-interactively. Harmless when unset.
        if let mode = ProcessInfo.processInfo.environment["PLANKTON_AUTOREC"] {
            let gif = (mode == "gif")
            let go: () -> Void = { if gif { self.renderer?.toggleGIF(size: self.mtkView.drawableSize) }
                                   else { self.renderer?.toggleVideo(size: self.mtkView.drawableSize) } }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.6, execute: go)
            DispatchQueue.main.asyncAfter(deadline: .now() + 3.0, execute: go)
            DispatchQueue.main.asyncAfter(deadline: .now() + 4.5) { NSApp.terminate(nil) }
        }
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
