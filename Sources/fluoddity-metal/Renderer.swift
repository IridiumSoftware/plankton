import MetalKit
import simd
import Foundation

// Drives one frame: advance the simulation (advect → splat → project → advect
// dye → sense/move), then draw the dye fullscreen and the agents as additive
// points on top. Compute + render encoders share one command buffer; Metal
// serializes them, so the fields are updated before they're drawn.
final class Renderer: NSObject, MTKViewDelegate {
    private let queue: MTLCommandQueue
    private let sim: Simulation
    private let dyePipe: MTLRenderPipelineState
    private let pointsPipe: MTLRenderPipelineState

    init(device: MTLDevice, pixelFormat: MTLPixelFormat) {
        guard let q = device.makeCommandQueue() else {
            fatalError("could not create command queue")
        }
        self.queue = q

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: Shaders.source, options: nil)
        } catch {
            fatalError("MSL compile failed: \(error)")
        }
        self.sim = Simulation(device: device, library: library)

        // dye fullscreen pipeline (opaque)
        let ddesc = MTLRenderPipelineDescriptor()
        ddesc.vertexFunction = library.makeFunction(name: "fs_vertex")
        ddesc.fragmentFunction = library.makeFunction(name: "fs_fragment")
        ddesc.colorAttachments[0].pixelFormat = pixelFormat
        self.dyePipe = try! device.makeRenderPipelineState(descriptor: ddesc)

        // agent-points pipeline (additive blend)
        let pdesc = MTLRenderPipelineDescriptor()
        pdesc.vertexFunction = library.makeFunction(name: "point_vertex")
        pdesc.fragmentFunction = library.makeFunction(name: "point_fragment")
        let att = pdesc.colorAttachments[0]!
        att.pixelFormat = pixelFormat
        att.isBlendingEnabled = true
        att.rgbBlendOperation = .add
        att.alphaBlendOperation = .add
        att.sourceRGBBlendFactor = .one
        att.sourceAlphaBlendFactor = .one
        att.destinationRGBBlendFactor = .one
        att.destinationAlphaBlendFactor = .one
        self.pointsPipe = try! device.makeRenderPipelineState(descriptor: pdesc)

        super.init()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let rpd = view.currentRenderPassDescriptor,
              let cmd = queue.makeCommandBuffer() else { return }

        // advance the simulation
        sim.encode(into: cmd)

        // draw: dye field, then agent points on top
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0.01, green: 0.01, blue: 0.03, alpha: 1)
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }

        enc.setRenderPipelineState(dyePipe)
        enc.setFragmentBuffer(sim.dye, offset: 0, index: 0)
        var dim = sim.dim
        enc.setFragmentBytes(&dim, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 1)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

        enc.setRenderPipelineState(pointsPipe)
        enc.setVertexBuffer(sim.particleBuffer, offset: 0, index: 0)
        enc.drawPrimitives(type: .point, vertexStart: 0, vertexCount: sim.particleCount)

        enc.endEncoding()
        cmd.present(drawable)
        cmd.commit()
    }
}
