import MetalKit
import Foundation

// v0 render loop. Clears the view to a slow, breathing dark gradient so the
// live (display-linked) loop is visibly running on screen. This proves the
// second risky unknown — a windowed Metal loop from SwiftPM, no Xcode.
//
// Next: a particle buffer + a compute pass (sense → move → atomic-splat into a
// flow-field texture), then the field update (decay/diffuse → advect → project).
final class Renderer: NSObject, MTKViewDelegate {
    private let queue: MTLCommandQueue
    private var frame: Int = 0

    init(device: MTLDevice) {
        guard let q = device.makeCommandQueue() else {
            fatalError("could not create command queue")
        }
        self.queue = q
        super.init()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let rpd = view.currentRenderPassDescriptor,
              let cmd = queue.makeCommandBuffer() else { return }

        // Breathing dark background — a calm hue drift, so the loop is visibly alive.
        let t = Float(frame) * 0.012
        let r = (0.5 + 0.5 * sin(t))         * 0.18
        let g = (0.5 + 0.5 * sin(t + 2.094)) * 0.20   // +120°
        let b = (0.5 + 0.5 * sin(t + 4.188)) * 0.32   // +240°
        rpd.colorAttachments[0].clearColor =
            MTLClearColor(red: Double(r), green: Double(g), blue: Double(b), alpha: 1.0)
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].storeAction = .store

        guard let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }
        // v0: clear only — nothing drawn yet. Particles next.
        enc.endEncoding()
        cmd.present(drawable)
        cmd.commit()
        frame += 1
    }
}
