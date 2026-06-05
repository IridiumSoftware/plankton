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
            Knob3D(name: "swim",         get: { sim.swim },         set: { sim.swim = $0 },         lo: 0,     hi: 0.30),
            Knob3D(name: "sensorDist",   get: { sim.sensorDist },   set: { sim.sensorDist = $0 },   lo: 0.001, hi: 0.08),
            Knob3D(name: "sensorAngle",  get: { sim.sensorAngle },  set: { sim.sensorAngle = $0 },  lo: 0,     hi: 1.5),
            Knob3D(name: "turn",         get: { sim.turn },         set: { sim.turn = $0 },         lo: 0,     hi: 0.30),
            Knob3D(name: "axialForce",   get: { sim.axialForce },   set: { sim.axialForce = $0 },   lo: 0,     hi: 0.10),
            Knob3D(name: "senseScale",   get: { sim.senseScale },   set: { sim.senseScale = $0 },   lo: 0,     hi: 10),
            Knob3D(name: "planeSamples", get: { sim.planeSamples }, set: { sim.planeSamples = $0 }, lo: 1,     hi: 6),
            Knob3D(name: "fluidPull",    get: { sim.fluidPull },    set: { sim.fluidPull = $0 },    lo: 0,     hi: 6),
            Knob3D(name: "velDamp",      get: { sim.velDamp },      set: { sim.velDamp = $0 },      lo: 0.80,  hi: 0.999),
            Knob3D(name: "viscosity",    get: { sim.viscosity },    set: { sim.viscosity = $0 },    lo: 0,     hi: 0.16),
            Knob3D(name: "dipoleLen",    get: { sim.dipoleLen },    set: { sim.dipoleLen = $0 },    lo: 0,     hi: 6),
            Knob3D(name: "cohesion",     get: { sim.cohesion },     set: { sim.cohesion = $0 },     lo: 0,     hi: 0.60),
            Knob3D(name: "forceGain",    get: { sim.forceGain },    set: { sim.forceGain = $0 },    lo: 0,     hi: 3),
            Knob3D(name: "dyeDecay",     get: { sim.dyeDecay },     set: { sim.dyeDecay = $0 },     lo: 0.90,  hi: 0.999),
            Knob3D(name: "dyeAmount",    get: { sim.dyeAmount },    set: { sim.dyeAmount = $0 },    lo: 0,     hi: 4),
            Knob3D(name: "densityScale", get: { r.densityScale },   set: { r.densityScale = $0 },   lo: 0.001, hi: 0.10),
            Knob3D(name: "colorMode",    get: { r.colorMode },      set: { r.colorMode = $0 },      lo: 0,     hi: 2),
        ]
        panel = Panel3D(knobs: knobs)
        panel.onReroll = { [weak self] in self?.renderer.reroll() }
        panel.setFrameOrigin(NSPoint(x: 12, y: frame.height - panel.frame.height - 12))
        panel.autoresizingMask = [.maxXMargin, .minYMargin]
        container.addSubview(panel)

        // keyboard-help button (bottom-left; expands upward)
        help = HelpView(text: """
        drag      orbit camera
        scroll    zoom
        r         re-roll brain
        [  /  ]   dim / brighten volume
        """)
        help.setFrameOrigin(NSPoint(x: 12, y: 12))
        help.autoresizingMask = [.maxXMargin, .maxYMargin]
        container.addSubview(help)

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
