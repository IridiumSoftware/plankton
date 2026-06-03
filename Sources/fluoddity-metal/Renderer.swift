import MetalKit
import simd
import Foundation

// Drives one frame: advance the simulation, then draw the dye fullscreen and
// the agents as additive points on top. Render-side tunables (toneK,
// pointAlpha) come from the shared Params each frame.
final class Renderer: NSObject, MTKViewDelegate {
    private let queue: MTLCommandQueue
    private let sim: Simulation
    private let params: Params
    private let dyePipe: MTLRenderPipelineState
    private let pointsPipe: MTLRenderPipelineState

    init(device: MTLDevice, pixelFormat: MTLPixelFormat, params: Params, mouse: MouseInput) {
        guard let q = device.makeCommandQueue() else {
            fatalError("could not create command queue")
        }
        self.queue = q
        self.params = params

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: Shaders.source, options: nil)
        } catch {
            fatalError("MSL compile failed: \(error)")
        }
        self.sim = Simulation(device: device, library: library, params: params, mouse: mouse)

        let ddesc = MTLRenderPipelineDescriptor()
        ddesc.vertexFunction = library.makeFunction(name: "fs_vertex")
        ddesc.fragmentFunction = library.makeFunction(name: "fs_fragment")
        ddesc.colorAttachments[0].pixelFormat = pixelFormat
        self.dyePipe = try! device.makeRenderPipelineState(descriptor: ddesc)

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

    func reroll() { sim.rerollRule() }
    func reset() { sim.reset() }
    func ruleSnapshot() -> [Float] { sim.ruleSnapshot() }
    func loadRule(_ floats: [Float]) { sim.loadRule(floats) }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let rpd = view.currentRenderPassDescriptor,
              let cmd = queue.makeCommandBuffer() else { return }

        sim.encode(into: cmd)

        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0.01, green: 0.01, blue: 0.03, alpha: 1)
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }

        var dim = sim.dim
        var toneK = params.toneK
        var satGain = params.satGain
        var bloomStrength = params.bloomStrength
        var palette = params.palette
        enc.setRenderPipelineState(dyePipe)
        enc.setFragmentBuffer(sim.dye, offset: 0, index: 0)
        enc.setFragmentBytes(&dim, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 1)
        enc.setFragmentBytes(&toneK, length: 4, index: 2)
        enc.setFragmentBuffer(sim.vel, offset: 0, index: 3)
        enc.setFragmentBytes(&satGain, length: 4, index: 4)
        enc.setFragmentBuffer(sim.dyeBlur, offset: 0, index: 5)
        enc.setFragmentBytes(&bloomStrength, length: 4, index: 6)
        enc.setFragmentBytes(&palette, length: 4, index: 7)
        var viewMode = params.viewMode
        var vortScale = params.vortScale
        enc.setFragmentBuffer(sim.vort, offset: 0, index: 8)
        enc.setFragmentBytes(&viewMode, length: 4, index: 9)
        enc.setFragmentBytes(&vortScale, length: 4, index: 10)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

        // agent points — only over the dye art, not the diagnostic field views
        if params.viewMode < 0.5 {
            var pointAlpha = params.pointAlpha
            var pointSize = params.pointSize
            var count = UInt32(sim.particleCount)
            enc.setRenderPipelineState(pointsPipe)
            enc.setVertexBuffer(sim.particleBuffer, offset: 0, index: 0)
            enc.setVertexBytes(&pointSize, length: 4, index: 1)
            enc.setVertexBytes(&count, length: 4, index: 2)
            enc.setFragmentBytes(&pointAlpha, length: 4, index: 0)
            enc.drawPrimitives(type: .point, vertexStart: 0, vertexCount: sim.particleCount)
        }

        enc.endEncoding()
        cmd.present(drawable)
        cmd.commit()
    }
}
