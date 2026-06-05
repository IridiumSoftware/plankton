import MetalKit

// MTKView for the 3D viewport: mouse-drag orbits the camera, scroll zooms.
final class View3D: MTKView {
    var camera: Camera3D?
    var onReroll: (() -> Void)?
    var onDensity: ((Float) -> Void)?
    var onKey: ((String) -> Void)?          // capture keys (c/x/j/k) → App3D

    override var acceptsFirstResponder: Bool { true }

    override func keyDown(with e: NSEvent) {
        switch e.charactersIgnoringModifiers {
        case "r": onReroll?()
        case "[": onDensity?(0.83)   // dimmer volume
        case "]": onDensity?(1.2)    // denser volume
        default: if let s = e.charactersIgnoringModifiers { onKey?(s) }
        }
    }

    override func mouseDragged(with e: NSEvent) {
        guard let c = camera else { return }
        c.azimuth -= Float(e.deltaX) * 0.01
        c.elevation = max(-1.5, min(1.5, c.elevation + Float(e.deltaY) * 0.01))
    }

    override func scrollWheel(with e: NSEvent) {
        guard let c = camera else { return }
        c.distance = max(0.5, min(6.0, c.distance * (1.0 - Float(e.deltaY) * 0.02)))
    }
}
