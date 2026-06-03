import MetalKit
import simd

// Stage-3 3D renderer: advance the agent-driven fluid, then volumetric ray-march
// the dye density into a glowing volume through the orbital camera.
final class Renderer3D: NSObject, MTKViewDelegate {
    private let queue: MTLCommandQueue
    private let sim: Sim3D
    private let volumePipe: MTLRenderPipelineState
    let camera = Camera3D()
    var densityScale: Float = 0.05

    init(device: MTLDevice, pixelFormat: MTLPixelFormat) {
        queue = device.makeCommandQueue()!
        let lib: MTLLibrary
        do { lib = try device.makeLibrary(source: Shaders3D.source, options: nil) }
        catch { fatalError("MSL3D compile failed: \(error)") }
        sim = Sim3D(device: device, library: lib)

        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = lib.makeFunction(name: "raymarch_vertex")
        d.fragmentFunction = lib.makeFunction(name: "raymarch_fragment")
        d.colorAttachments[0].pixelFormat = pixelFormat
        volumePipe = try! device.makeRenderPipelineState(descriptor: d)
        super.init()
    }

    func reroll() { sim.rerollRule() }
    func adjustDensity(_ f: Float) {
        densityScale = max(0.005, min(2.0, densityScale * f))
        print(String(format: "densityScale = %.3f", densityScale))
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        camera.aspect = Float(size.width / max(size.height, 1))
    }

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let rpd = view.currentRenderPassDescriptor,
              let cmd = queue.makeCommandBuffer() else { return }

        sim.encode(into: cmd)

        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0.02, green: 0.02, blue: 0.05, alpha: 1)
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }
        enc.setRenderPipelineState(volumePipe)
        enc.setFragmentBuffer(sim.dye, offset: 0, index: 0)
        var dim = sim.dim
        enc.setFragmentBytes(&dim, length: 16, index: 1)
        var invVP = camera.invViewProj()
        enc.setFragmentBytes(&invVP, length: MemoryLayout<float4x4>.stride, index: 2)
        var ds = densityScale
        enc.setFragmentBytes(&ds, length: 4, index: 3)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()

        cmd.present(drawable)
        cmd.commit()
    }
}
