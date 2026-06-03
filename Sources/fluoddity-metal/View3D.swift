import MetalKit

// MTKView for the 3D viewport: mouse-drag orbits the camera, scroll zooms.
final class View3D: MTKView {
    var camera: Camera3D?

    override var acceptsFirstResponder: Bool { true }

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
