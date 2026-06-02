import Metal
import simd
import Foundation

// Mirrors `struct Particle` in Shaders.source (16 bytes: pos.xy, vel.xy).
struct Particle {
    var pos: SIMD2<Float>
    var vel: SIMD2<Float>
}

// Owns the particle + field buffers and the three compute pipelines, and
// encodes one simulation step (decay → splat → sense/move) into a command
// buffer. Each pass is its own encoder so Metal serializes them (the field
// must be decayed before splat, splatted before sensing reads it).
final class Simulation {
    let particleCount: Int
    let dim: SIMD2<UInt32>            // field resolution
    let particleBuffer: MTLBuffer
    let fieldBuffer: MTLBuffer

    private let decayPipe: MTLComputePipelineState
    private let splatPipe: MTLComputePipelineState
    private let movePipe:  MTLComputePipelineState

    // v0 tunables (CPU side)
    private var decay: Float = 0.96
    private var splatAmount: Float = 1.0
    private var dt: Float = 1.0 / 60.0

    init(device: MTLDevice, library: MTLLibrary,
         particleCount: Int = 1 << 20, fieldDim: Int = 1024) {
        self.particleCount = particleCount
        self.dim = SIMD2<UInt32>(UInt32(fieldDim), UInt32(fieldDim))

        // Particles: random position, random heading, modest speed.
        var particles = [Particle]()
        particles.reserveCapacity(particleCount)
        var rng = SystemRandomNumberGenerator()
        for _ in 0..<particleCount {
            let px = Float.random(in: 0..<1, using: &rng)
            let py = Float.random(in: 0..<1, using: &rng)
            let ang = Float.random(in: 0..<(2 * .pi), using: &rng)
            let spd = Float.random(in: 0.05..<0.15, using: &rng)
            particles.append(Particle(pos: SIMD2(px, py),
                                      vel: SIMD2(cos(ang) * spd, sin(ang) * spd)))
        }
        particleBuffer = device.makeBuffer(
            bytes: &particles,
            length: MemoryLayout<Particle>.stride * particleCount,
            options: .storageModeShared)!

        let fieldCount = fieldDim * fieldDim
        fieldBuffer = device.makeBuffer(
            length: MemoryLayout<Float>.stride * fieldCount,
            options: .storageModeShared)!
        memset(fieldBuffer.contents(), 0, fieldBuffer.length)

        func pipe(_ name: String) -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                fatalError("kernel `\(name)` not found in library")
            }
            return try! device.makeComputePipelineState(function: fn)
        }
        decayPipe = pipe("decay_field")
        splatPipe = pipe("splat_particles")
        movePipe  = pipe("move_particles")
    }

    func encode(into cmd: MTLCommandBuffer) {
        let fieldCount = Int(dim.x) * Int(dim.y)
        var dimv = dim
        var decayv = decay
        var amountv = splatAmount
        var dtv = dt

        // 1. decay the field in place
        if let e = cmd.makeComputeCommandEncoder() {
            e.setComputePipelineState(decayPipe)
            e.setBuffer(fieldBuffer, offset: 0, index: 0)
            e.setBytes(&decayv, length: MemoryLayout<Float>.stride, index: 1)
            dispatch1D(e, decayPipe, fieldCount)
            e.endEncoding()
        }
        // 2. splat particle trails (atomic)
        if let e = cmd.makeComputeCommandEncoder() {
            e.setComputePipelineState(splatPipe)
            e.setBuffer(fieldBuffer, offset: 0, index: 0)
            e.setBuffer(particleBuffer, offset: 0, index: 1)
            e.setBytes(&dimv, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
            e.setBytes(&amountv, length: MemoryLayout<Float>.stride, index: 3)
            dispatch1D(e, splatPipe, particleCount)
            e.endEncoding()
        }
        // 3. sense + move
        if let e = cmd.makeComputeCommandEncoder() {
            e.setComputePipelineState(movePipe)
            e.setBuffer(particleBuffer, offset: 0, index: 0)
            e.setBuffer(fieldBuffer, offset: 0, index: 1)
            e.setBytes(&dimv, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
            e.setBytes(&dtv, length: MemoryLayout<Float>.stride, index: 3)
            dispatch1D(e, movePipe, particleCount)
            e.endEncoding()
        }
    }

    private func dispatch1D(_ e: MTLComputeCommandEncoder,
                            _ pipe: MTLComputePipelineState, _ count: Int) {
        let w = pipe.maxTotalThreadsPerThreadgroup
        e.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    }
}
