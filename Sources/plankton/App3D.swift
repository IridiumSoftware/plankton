import AppKit
import MetalKit

// 3D app delegate (launched with `--3d`): an orbital volumetric view over the
// agent-driven 3D fluid. Same layout as the 2D app: a fixed-width control
// sidebar on the left (scrollable knob panel + description footer — controls
// never overlap the rendering) and the canvas to its right.
final class App3D: NSObject, NSApplicationDelegate {
    private var window: NSWindow!
    private var view: View3D!
    private var renderer: Renderer3D!
    private var panel: KnobPanel!
    private var sidebar: Sidebar!
    private var help: HelpView!
    private var captureLabel: NSTextField!

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device available.")
        }
        let canvas: CGFloat = 1024, sideW = Sidebar.width
        let frame = NSRect(x: 0, y: 0, width: sideW + canvas, height: canvas)
        let container = NSView(frame: frame)
        container.autoresizesSubviews = true

        view = View3D(frame: NSRect(x: sideW, y: 0, width: canvas, height: canvas), device: device)
        view.colorPixelFormat = .bgra8Unorm
        view.preferredFramesPerSecond = 60
        view.framebufferOnly = false        // lets the recorder blit the drawable
        view.autoresizingMask = [.width, .height]
        renderer = Renderer3D(device: device, pixelFormat: view.colorPixelFormat)
        view.delegate = renderer
        view.camera = renderer.camera
        view.onReroll = { [weak self] in self?.renderer.reroll() }
        view.onDensity = { [weak self] f in self?.renderer.adjustDensity(f); self?.panel.refresh() }
        container.addSubview(view)

        // control sidebar (left), knobs bound to the sim + renderer.
        let sim = renderer.sim
        let r = renderer!
        let knobs: [UIKnob] = [
            // ARRAY ORDER = capture/path serialization order: append-only.
            // `group` is display-only bucketing (the panel regroups visually).
            UIKnob(name: "swim",         info: "Agent self-propulsion speed.",
                   get: { sim.swim },         set: { sim.swim = $0 },         lo: 0,     hi: 0.30,  group: "Agents"),
            UIKnob(name: "sensorDist",   info: "How far ahead the sensors reach — sets the eddy scale agents react to.",
                   get: { sim.sensorDist },   set: { sim.sensorDist = $0 },   lo: 0.001, hi: 0.08,  group: "Agents"),
            UIKnob(name: "sensorAngle",  info: "Angular spread between the paired sensors (radians).",
                   get: { sim.sensorAngle },  set: { sim.sensorAngle = $0 },  lo: 0,     hi: 1.5,   group: "Agents"),
            UIKnob(name: "turn",         info: "Steering gain — how sharply the brain can turn the agent per step.",
                   get: { sim.turn },         set: { sim.turn = $0 },         lo: 0,     hi: 0.30,  group: "Agents"),
            UIKnob(name: "axialForce",   info: "Thrust modulation along the heading (the brain's speed-up / slow-down).",
                   get: { sim.axialForce },   set: { sim.axialForce = $0 },   lo: 0,     hi: 0.10,  group: "Agents"),
            UIKnob(name: "senseScale",   info: "Gain on the sensed flow before it enters the brain (higher = more reactive).",
                   get: { sim.senseScale },   set: { sim.senseScale = $0 },   lo: 0,     hi: 10,    group: "Agents"),
            UIKnob(name: "planeSamples", info: "Tangent planes the 3D brain samples per step (more = smoother steering, slower).",
                   get: { sim.planeSamples }, set: { sim.planeSamples = $0 }, lo: 1,     hi: 6,     group: "Agents"),
            UIKnob(name: "fluidPull",    info: "How strongly the fluid carries agents along (their advection).",
                   get: { sim.fluidPull },    set: { sim.fluidPull = $0 },    lo: 0,     hi: 6,     group: "Fluid"),
            UIKnob(name: "velDamp",      info: "Residual large-scale drag per step (1 = none) — the big-eddy energy sink.",
                   get: { sim.velDamp },      set: { sim.velDamp = $0 },      lo: 0.80,  hi: 0.999, group: "Fluid"),
            UIKnob(name: "viscosity",    info: "Real ν∇² viscosity — dissipates small scales selectively (the physical damping).",
                   get: { sim.viscosity },    set: { sim.viscosity = $0 },    lo: 0,     hi: 0.16,  group: "Fluid"),
            UIKnob(name: "dipoleLen",    info: "Spacing of each agent's +f/−f force pair, in cells (net-zero swimmer forcing).",
                   get: { sim.dipoleLen },    set: { sim.dipoleLen = $0 },    lo: 0,     hi: 6,     group: "Fluid"),
            UIKnob(name: "cohesion",     info: "Chemotaxis toward higher dye = toward other agents. The creature-maker — crank it to grow volumetric creatures.",
                   get: { sim.cohesion },     set: { sim.cohesion = $0 },     lo: 0,     hi: 0.60,  group: "Agents"),
            UIKnob(name: "forceGain",    info: "How hard each agent pushes on the fluid.",
                   get: { sim.forceGain },    set: { sim.forceGain = $0 },    lo: 0,     hi: 3,     group: "Fluid"),
            UIKnob(name: "dyeDecay",     info: "Dye persistence per step (closer to 1 = structures linger longer).",
                   get: { sim.dyeDecay },     set: { sim.dyeDecay = $0 },     lo: 0.90,  hi: 0.999, group: "Dye"),
            UIKnob(name: "dyeAmount",    info: "Dye each agent deposits per step (the chemotaxis signal strength).",
                   get: { sim.dyeAmount },    set: { sim.dyeAmount = $0 },    lo: 0,     hi: 4,     group: "Dye"),
            UIKnob(name: "densityScale", info: "Volume exposure: dye density → opacity ([ and ] nudge it too).",
                   get: { r.densityScale },   set: { r.densityScale = $0 },   lo: 0.001, hi: 0.10,  group: "Display"),
            UIKnob(name: "colorMode",    info: "Color the volume by dye, flow direction, flow speed — or show the fluid-only vortex tubes.",
                   get: { r.colorMode },      set: { r.colorMode = $0 },      lo: 0,     hi: 3,     group: "Display",
                   options: ["dye density", "flow direction", "flow speed", "fluid only: vorticity"]),
            UIKnob(name: "simSpeed",     info: "Sim steps per rendered frame: 0 pauses, <1 slow motion, >1 fast-forward.",
                   get: { sim.simSpeed },     set: { sim.simSpeed = $0 },     lo: 0,     hi: 4,     group: "Time"),
            UIKnob(name: "sharpness",    info: "Transfer gamma — higher crisps structure edges by cutting the faint halo.",
                   get: { r.sharpness },      set: { r.sharpness = $0 },      lo: 0.5,   hi: 4,     group: "Display"),
            UIKnob(name: "mutationStrength", info: "How strongly right-click breeding (and ecology reseeding) mutates brains.",
                   get: { sim.mutationStrength }, set: { sim.mutationStrength = $0 }, lo: 0, hi: 1.5, group: "Agents"),
            UIKnob(name: "pointAlpha",   info: "Cohort-coloured agent dots over the volume — raise to see species / the ecology mix.",
                   get: { r.pointAlpha },     set: { r.pointAlpha = $0 },     lo: 0,     hi: 0.6,   group: "Display"),
            UIKnob(name: "vortScale",    info: "Opacity gain for the fluid-only vorticity view.",
                   get: { r.vortScale },      set: { r.vortScale = $0 },      lo: 1,     hi: 200,   group: "Display"),
            UIKnob(name: "jacobiIters",  info: "Pressure-solve iterations: projection quality vs speed — the main perf dial at 192³.",
                   get: { Float(sim.jacobiIters) }, set: { sim.jacobiIters = max(1, Int($0.rounded())) },
                   lo: 5, hi: 40, step: 1, group: "Fluid"),
        ]
        panel = KnobPanel(
            knobs: knobs,
            buttons: [("Brain", "Re-roll all 8 cohort brains (same as the r key)", { [weak self] in self?.renderer.reroll() })]
        )
        sidebar = Sidebar(panel: panel, height: frame.height)
        sidebar.autoresizingMask = [.height]
        container.addSubview(sidebar)

        // capture: bind the param read/write to the 3D knobs, route the capture keys
        renderer.paramRead = { knobs.map { $0.get() } }
        renderer.paramWrite = { i, v in if i < knobs.count { knobs[i].set(v) } }
        renderer.onReplayTick = { [weak self] in self?.panel.refresh() }   // path replay → keep sliders live
        view.onBreed = { [weak self] uv in self?.renderer.breed(at: uv) }
        view.onKey = { [weak self] key in
            guard let self else { return }
            switch key {
            case "c": self.renderer.captureCreature()
            case "x": self.renderer.restoreCreature(); self.panel.refresh()
            case "j": self.renderer.recordPathToggle()
            case "k": self.renderer.replayLastPath(); self.panel.refresh()
            case "v": self.renderer.toggleVideo(size: self.view.drawableSize)
            case "g": self.renderer.toggleGIF(size: self.view.drawableSize)
            case "e": self.renderer.cycleEcology(); self.panel.refresh()    // ecology mode: off→rps→coex→dom
            default: break
            }
        }

        // keyboard-help button (bottom-left of the canvas; expands upward)
        help = HelpView(text: """
        drag         orbit camera
        scroll       zoom
        right-click  adopt + mutate cohort (breed)
        r            re-roll brains (8 cohorts)
        [  /  ]      dim / brighten volume
        c / x        capture creature / restore (cycle)
        j / k        record path (toggle) / replay path
        v / g        record mp4 / gif clip (toggle) → captures/video
        e            ecology mode: off → rps → coexistence → dominance
        """)
        help.setFrameOrigin(NSPoint(x: sideW + 12, y: 12))
        help.autoresizingMask = [.maxXMargin, .maxYMargin]
        container.addSubview(help)

        // capture-zoo status (bottom-right): which 3D creature slot is loaded/saved
        captureLabel = NSTextField(frame: NSRect(x: frame.width - 412, y: 12, width: 400, height: 22))
        captureLabel.isEditable = false
        captureLabel.isBordered = false
        captureLabel.drawsBackground = true
        captureLabel.backgroundColor = NSColor(white: 0, alpha: 0.55)
        captureLabel.textColor = NSColor(calibratedRed: 1.0, green: 0.82, blue: 0.4, alpha: 1)
        captureLabel.font = .monospacedSystemFont(ofSize: 11, weight: .medium)
        captureLabel.alignment = .right
        captureLabel.stringValue = "capture: c save · x load · j/k path"
        captureLabel.autoresizingMask = [.minXMargin, .maxYMargin]
        container.addSubview(captureLabel)
        renderer.onCaptureStatus = { [weak self] s in self?.captureLabel.stringValue = s }

        window = NSWindow(contentRect: frame,
                          styleMask: [.titled, .closable, .resizable, .miniaturizable],
                          backing: .buffered, defer: false)
        window.title = "plankton — 3D"
        window.contentView = container
        window.minSize = NSSize(width: sideW + 560, height: 640)
        window.center()
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(view)
        print("3D: drag orbit · scroll zoom · r re-roll · [ ] density")

        // Smoke hook for the live 3D record path (no keystrokes needed): with
        // PLANKTON_AUTOREC set, record ~2.4 s of mp4 then quit. Harmless when unset.
        if let mode = ProcessInfo.processInfo.environment["PLANKTON_AUTOREC"] {
            let gif = (mode == "gif")
            let go: () -> Void = { if gif { self.renderer.toggleGIF(size: self.view.drawableSize) }
                                   else { self.renderer.toggleVideo(size: self.view.drawableSize) } }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.6, execute: go)
            DispatchQueue.main.asyncAfter(deadline: .now() + 3.0, execute: go)
            DispatchQueue.main.asyncAfter(deadline: .now() + 4.5) { NSApp.terminate(nil) }
        }
        // Smoke hook for the live ecology render path (cohort-tinted points + reallocation).
        if ProcessInfo.processInfo.environment["PLANKTON_AUTOECO"] != nil {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) { self.renderer.cycleEcology() }   // → rps
            DispatchQueue.main.asyncAfter(deadline: .now() + 3.5) { NSApp.terminate(nil) }
        }
        // Perf hook: with PLANKTON_AUTOPERF set, the renderer prints the average
        // GPU ms/frame (sim + render) every 240 frames; quit after two reports.
        if ProcessInfo.processInfo.environment["PLANKTON_AUTOPERF"] != nil {
            DispatchQueue.main.asyncAfter(deadline: .now() + 10.0) { NSApp.terminate(nil) }
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
