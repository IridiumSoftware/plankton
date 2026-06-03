import Metal
import simd

// Mirrors `struct Particle3D` in Shaders3D.source (two float3 = 32 bytes).
struct Particle3D {
    var pos: SIMD3<Float>
    var vel: SIMD3<Float>
}

// Stage-1 3D simulation: a buffer of particles advected by the ABC flow each frame.
final class Sim3D {
    let count: Int
    let particleBuffer: MTLBuffer
    private let movePipe: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary, count: Int = 1 << 20) {
        self.count = count
        var rng = SystemRandomNumberGenerator()
        var parts = [Particle3D](); parts.reserveCapacity(count)
        for _ in 0..<count {
            let p = SIMD3(Float.random(in: 0..<1, using: &rng),
                          Float.random(in: 0..<1, using: &rng),
                          Float.random(in: 0..<1, using: &rng))
            parts.append(Particle3D(pos: p, vel: .zero))
        }
        particleBuffer = device.makeBuffer(bytes: &parts,
                                           length: MemoryLayout<Particle3D>.stride * count,
                                           options: .storageModeShared)!
        guard let fn = library.makeFunction(name: "move3d") else { fatalError("move3d not found") }
        movePipe = try! device.makeComputePipelineState(function: fn)
    }

    func encode(into cmd: MTLCommandBuffer) {
        guard let e = cmd.makeComputeCommandEncoder() else { return }
        e.setComputePipelineState(movePipe)
        e.setBuffer(particleBuffer, offset: 0, index: 0)
        let w = movePipe.maxTotalThreadsPerThreadgroup
        e.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        e.endEncoding()
    }
}
