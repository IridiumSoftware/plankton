import MetalKit
import simd

// Stage-1 3D renderer: advance the particle flow, then draw it as an additive
// point cloud through the orbital camera.
final class Renderer3D: NSObject, MTKViewDelegate {
    private let queue: MTLCommandQueue
    private let sim: Sim3D
    private let pointsPipe: MTLRenderPipelineState
    let camera = Camera3D()

    init(device: MTLDevice, pixelFormat: MTLPixelFormat) {
        queue = device.makeCommandQueue()!
        let lib: MTLLibrary
        do { lib = try device.makeLibrary(source: Shaders3D.source, options: nil) }
        catch { fatalError("MSL3D compile failed: \(error)") }
        sim = Sim3D(device: device, library: lib)

        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = lib.makeFunction(name: "point3d_vertex")
        d.fragmentFunction = lib.makeFunction(name: "point3d_fragment")
        let a = d.colorAttachments[0]!
        a.pixelFormat = pixelFormat
        a.isBlendingEnabled = true
        a.rgbBlendOperation = .add
        a.alphaBlendOperation = .add
        a.sourceRGBBlendFactor = .one
        a.sourceAlphaBlendFactor = .one
        a.destinationRGBBlendFactor = .one
        a.destinationAlphaBlendFactor = .one
        pointsPipe = try! device.makeRenderPipelineState(descriptor: d)
        super.init()
    }

    func reroll() { sim.rerollRule() }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        camera.aspect = Float(size.width / max(size.height, 1))
    }

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let rpd = view.currentRenderPassDescriptor,
              let cmd = queue.makeCommandBuffer() else { return }

        sim.encode(into: cmd)

        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0.01, green: 0.01, blue: 0.03, alpha: 1)
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }
        enc.setRenderPipelineState(pointsPipe)
        enc.setVertexBuffer(sim.particleBuffer, offset: 0, index: 0)
        var vp = camera.viewProj()
        enc.setVertexBytes(&vp, length: MemoryLayout<float4x4>.stride, index: 1)
        enc.drawPrimitives(type: .point, vertexStart: 0, vertexCount: sim.count)
        enc.endEncoding()

        cmd.present(drawable)
        cmd.commit()
    }
}
