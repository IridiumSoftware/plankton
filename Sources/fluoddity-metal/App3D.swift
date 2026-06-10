import AppKit
import MetalKit

// 3D app delegate (launched with `--3d`): an orbital volumetric view over the
// agent-driven 3D fluid, with a slider panel + a keyboard-help button.
final class App3D: NSObject, NSApplicationDelegate {
    private var window: NSWindow!
    private var view: View3D!
    private var renderer: Renderer3D!
    private var panel: Panel3D!
    private var help: HelpView!
    private var captureLabel: NSTextField!

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device available.")
        }
        let frame = NSRect(x: 0, y: 0, width: 1024, height: 1024)
        let container = NSView(frame: frame)
        container.autoresizesSubviews = true

        view = View3D(frame: frame, device: device)
        view.colorPixelFormat = .bgra8Unorm
        view.preferredFramesPerSecond = 60
        view.autoresizingMask = [.width, .height]
        renderer = Renderer3D(device: device, pixelFormat: view.colorPixelFormat)
        view.delegate = renderer
        view.camera = renderer.camera
        view.onReroll = { [weak self] in self?.renderer.reroll() }
        view.onDensity = { [weak self] f in self?.renderer.adjustDensity(f) }
        container.addSubview(view)

        // slider panel (top-left), knobs bound to the sim + renderer
        let sim = renderer.sim
        let r = renderer!
        let knobs: [Knob3D] = [
            // ARRAY ORDER = capture/path serialization order: append-only.
            // `group` is display-only bucketing (Panel3D regroups visually).
            Knob3D(name: "swim",         get: { sim.swim },         set: { sim.swim = $0 },         lo: 0,     hi: 0.30,  group: "Agents"),
            Knob3D(name: "sensorDist",   get: { sim.sensorDist },   set: { sim.sensorDist = $0 },   lo: 0.001, hi: 0.08,  group: "Agents"),
            Knob3D(name: "sensorAngle",  get: { sim.sensorAngle },  set: { sim.sensorAngle = $0 },  lo: 0,     hi: 1.5,   group: "Agents"),
            Knob3D(name: "turn",         get: { sim.turn },         set: { sim.turn = $0 },         lo: 0,     hi: 0.30,  group: "Agents"),
            Knob3D(name: "axialForce",   get: { sim.axialForce },   set: { sim.axialForce = $0 },   lo: 0,     hi: 0.10,  group: "Agents"),
            Knob3D(name: "senseScale",   get: { sim.senseScale },   set: { sim.senseScale = $0 },   lo: 0,     hi: 10,    group: "Agents"),
            Knob3D(name: "planeSamples", get: { sim.planeSamples }, set: { sim.planeSamples = $0 }, lo: 1,     hi: 6,     group: "Agents"),
            Knob3D(name: "fluidPull",    get: { sim.fluidPull },    set: { sim.fluidPull = $0 },    lo: 0,     hi: 6,     group: "Fluid"),
            Knob3D(name: "velDamp",      get: { sim.velDamp },      set: { sim.velDamp = $0 },      lo: 0.80,  hi: 0.999, group: "Fluid"),
            Knob3D(name: "viscosity",    get: { sim.viscosity },    set: { sim.viscosity = $0 },    lo: 0,     hi: 0.16,  group: "Fluid"),
            Knob3D(name: "dipoleLen",    get: { sim.dipoleLen },    set: { sim.dipoleLen = $0 },    lo: 0,     hi: 6,     group: "Fluid"),
            Knob3D(name: "cohesion",     get: { sim.cohesion },     set: { sim.cohesion = $0 },     lo: 0,     hi: 0.60,  group: "Agents"),
            Knob3D(name: "forceGain",    get: { sim.forceGain },    set: { sim.forceGain = $0 },    lo: 0,     hi: 3,     group: "Fluid"),
            Knob3D(name: "dyeDecay",     get: { sim.dyeDecay },     set: { sim.dyeDecay = $0 },     lo: 0.90,  hi: 0.999, group: "Dye"),
            Knob3D(name: "dyeAmount",    get: { sim.dyeAmount },    set: { sim.dyeAmount = $0 },    lo: 0,     hi: 4,     group: "Dye"),
            Knob3D(name: "densityScale", get: { r.densityScale },   set: { r.densityScale = $0 },   lo: 0.001, hi: 0.10,  group: "Display"),
            Knob3D(name: "colorMode",    get: { r.colorMode },      set: { r.colorMode = $0 },      lo: 0,     hi: 3,     group: "Display",
                   options: ["dye density", "flow direction", "flow speed", "fluid only: vorticity"]),
            Knob3D(name: "simSpeed",     get: { sim.simSpeed },     set: { sim.simSpeed = $0 },     lo: 0,     hi: 4,     group: "Time"),
            Knob3D(name: "sharpness",    get: { r.sharpness },      set: { r.sharpness = $0 },      lo: 0.5,   hi: 4,     group: "Display"),
            Knob3D(name: "mutationStrength", get: { sim.mutationStrength }, set: { sim.mutationStrength = $0 }, lo: 0, hi: 1.5, group: "Agents"),
            Knob3D(name: "pointAlpha",   get: { r.pointAlpha },     set: { r.pointAlpha = $0 },     lo: 0,     hi: 0.6,   group: "Display"),
            Knob3D(name: "vortScale",    get: { r.vortScale },      set: { r.vortScale = $0 },      lo: 1,     hi: 200,   group: "Display"),
        ]
        panel = Panel3D(knobs: knobs)
        panel.onReroll = { [weak self] in self?.renderer.reroll() }
        panel.setFrameOrigin(NSPoint(x: 12, y: frame.height - panel.frame.height - 12))
        panel.autoresizingMask = [.maxXMargin, .minYMargin]
        container.addSubview(panel)

        // capture: bind the param read/write to the 3D knobs, route the capture keys
        renderer.paramRead = { knobs.map { $0.get() } }
        renderer.paramWrite = { i, v in if i < knobs.count { knobs[i].set(v) } }
        view.onBreed = { [weak self] uv in self?.renderer.breed(at: uv) }
        view.onKey = { [weak self] key in
            guard let self else { return }
            switch key {
            case "c": self.renderer.captureCreature()
            case "x": self.renderer.restoreCreature(); self.panel.refresh()
            case "j": self.renderer.recordPathToggle()
            case "k": self.renderer.replayLastPath(); self.panel.refresh()
            default: break
            }
        }

        // keyboard-help button (bottom-left; expands upward)
        help = HelpView(text: """
        drag         orbit camera
        scroll       zoom
        right-click  adopt + mutate cohort (breed)
        r            re-roll brains (8 cohorts)
        [  /  ]      dim / brighten volume
        c / x        capture creature / restore (cycle)
        j / k        record path (toggle) / replay path
        """)
        help.setFrameOrigin(NSPoint(x: 12, y: 12))
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
        window.title = "fluoddity-metal — 3D"
        window.contentView = container
        window.center()
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(view)
        print("3D: drag orbit · scroll zoom · r re-roll · [ ] density")
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
