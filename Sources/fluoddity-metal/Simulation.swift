import Metal
import simd
import Foundation

// Mirrors `struct Particle` in Shaders.source (16 bytes: pos.xy, vel.xy).
struct Particle {
    var pos: SIMD2<Float>
    var vel: SIMD2<Float>
}

// Owns the grid fields + particle buffer and encodes one Stable-Fluids step:
//   advect velocity → damp → splat (force + dye) → project (divergence,
//   Jacobi pressure, subtract gradient) → advect dye → particles ride the flow.
// Each pass is its own encoder; Metal's hazard tracking serializes them.
final class Simulation {
    let particleCount: Int
    let dim: SIMD2<UInt32>

    // ping-ponged buffers are `var` so we can swap references
    private(set) var vel: MTLBuffer            // interleaved x,y  (2N)
    private var velTmp: MTLBuffer              // 2N
    private var pressure: MTLBuffer            // N
    private var pressureTmp: MTLBuffer         // N
    private let divg: MTLBuffer                // N
    private(set) var dye: MTLBuffer            // N
    private var dyeTmp: MTLBuffer              // N
    let particleBuffer: MTLBuffer

    private let scalePipe, advectVelPipe, splatPipe, divPipe, jacobiPipe,
                subGradPipe, advectDyePipe, movePipe: MTLComputePipelineState

    // tunables
    var velDamp: Float = 0.95
    var forceGain: Float = 0.5
    var dyeAmount: Float = 1.0
    var jacobiIters: Int = 30

    init(device: MTLDevice, library: MTLLibrary,
         particleCount: Int = 1 << 20, fieldDim: Int = 1024) {
        self.particleCount = particleCount
        self.dim = SIMD2<UInt32>(UInt32(fieldDim), UInt32(fieldDim))
        let n = fieldDim * fieldDim

        func zeroed(_ floats: Int) -> MTLBuffer {
            let b = device.makeBuffer(length: MemoryLayout<Float>.stride * floats,
                                      options: .storageModeShared)!
            memset(b.contents(), 0, b.length)
            return b
        }
        vel = zeroed(2 * n); velTmp = zeroed(2 * n)
        pressure = zeroed(n); pressureTmp = zeroed(n)
        divg = zeroed(n)
        dye = zeroed(n); dyeTmp = zeroed(n)

        // Particles: random position; velocity = a few overlaid Gaussian
        // vortices so the opening flow is visibly swirling, not noise.
        struct Vortex { var c: SIMD2<Float>; var s: Float; var r: Float }
        var rng = SystemRandomNumberGenerator()
        var vortices = [Vortex]()
        for _ in 0..<6 {
            vortices.append(Vortex(
                c: SIMD2(Float.random(in: 0..<1, using: &rng), Float.random(in: 0..<1, using: &rng)),
                s: Float.random(in: -1..<1, using: &rng),
                r: Float.random(in: 0.12..<0.30, using: &rng)))
        }
        var particles = [Particle](); particles.reserveCapacity(particleCount)
        for _ in 0..<particleCount {
            let pos = SIMD2(Float.random(in: 0..<1, using: &rng),
                            Float.random(in: 0..<1, using: &rng))
            var v = SIMD2<Float>(0, 0)
            for vt in vortices {
                let dvec = pos - vt.c
                let fall = exp(-simd_length_squared(dvec) / (vt.r * vt.r))
                v += vt.s * SIMD2(-dvec.y, dvec.x) * fall
            }
            v *= 0.12
            particles.append(Particle(pos: pos, vel: v))
        }
        particleBuffer = device.makeBuffer(
            bytes: &particles,
            length: MemoryLayout<Particle>.stride * particleCount,
            options: .storageModeShared)!

        func pipe(_ name: String) -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                fatalError("kernel `\(name)` not found")
            }
            return try! device.makeComputePipelineState(function: fn)
        }
        scalePipe     = pipe("scale_buffer")
        advectVelPipe = pipe("advect_velocity")
        splatPipe     = pipe("splat")
        divPipe       = pipe("divergence")
        jacobiPipe    = pipe("jacobi")
        subGradPipe   = pipe("subtract_gradient")
        advectDyePipe = pipe("advect_dye")
        movePipe      = pipe("move_particles")
    }

    func encode(into cmd: MTLCommandBuffer) {
        let n = Int(dim.x) * Int(dim.y)
        var dimv = dim, velDampv = velDamp, forceGainv = forceGain, dyeAmtv = dyeAmount

        // 1. advect velocity (vel → velTmp)
        field(cmd, advectVelPipe) { e in
            e.setBuffer(self.vel, offset: 0, index: 0)
            e.setBuffer(self.velTmp, offset: 0, index: 1)
            e.setBytes(&dimv, length: 8, index: 2)
        }
        // 2. damp velTmp (over 2N scalars)
        elementwise(cmd, scalePipe, count: 2 * n) { e in
            e.setBuffer(self.velTmp, offset: 0, index: 0)
            e.setBytes(&velDampv, length: 4, index: 1)
        }
        // 3. splat particle force + dye into velTmp / dye
        particles(cmd, splatPipe) { e in
            e.setBuffer(self.velTmp, offset: 0, index: 0)
            e.setBuffer(self.dye, offset: 0, index: 1)
            e.setBuffer(self.particleBuffer, offset: 0, index: 2)
            e.setBytes(&dimv, length: 8, index: 3)
            e.setBytes(&forceGainv, length: 4, index: 4)
            e.setBytes(&dyeAmtv, length: 4, index: 5)
        }
        // 4. divergence(velTmp → divg)
        field(cmd, divPipe) { e in
            e.setBuffer(self.velTmp, offset: 0, index: 0)
            e.setBuffer(self.divg, offset: 0, index: 1)
            e.setBytes(&dimv, length: 8, index: 2)
        }
        // 5. Jacobi pressure solve (warm-started; ping-pong)
        for _ in 0..<jacobiIters {
            field(cmd, jacobiPipe) { e in
                e.setBuffer(self.pressure, offset: 0, index: 0)
                e.setBuffer(self.pressureTmp, offset: 0, index: 1)
                e.setBuffer(self.divg, offset: 0, index: 2)
                e.setBytes(&dimv, length: 8, index: 3)
            }
            swap(&pressure, &pressureTmp)   // `pressure` now holds the latest
        }
        // 6. subtract gradient → velTmp divergence-free
        field(cmd, subGradPipe) { e in
            e.setBuffer(self.velTmp, offset: 0, index: 0)
            e.setBuffer(self.pressure, offset: 0, index: 1)
            e.setBytes(&dimv, length: 8, index: 2)
        }
        swap(&vel, &velTmp)                 // vel = new divergence-free field
        // 7. advect dye by the projected field (+ decay)
        field(cmd, advectDyePipe) { e in
            e.setBuffer(self.dye, offset: 0, index: 0)
            e.setBuffer(self.dyeTmp, offset: 0, index: 1)
            e.setBuffer(self.vel, offset: 0, index: 2)
            e.setBytes(&dimv, length: 8, index: 3)
        }
        swap(&dye, &dyeTmp)
        // 8. agents sense the dye, steer, swim + get carried by the flow
        particles(cmd, movePipe) { e in
            e.setBuffer(self.particleBuffer, offset: 0, index: 0)
            e.setBuffer(self.vel, offset: 0, index: 1)
            e.setBuffer(self.dye, offset: 0, index: 2)
            e.setBytes(&dimv, length: 8, index: 3)
        }
    }

    // ── encoder helpers ─────────────────────────────────────────────────
    private func field(_ cmd: MTLCommandBuffer, _ pipe: MTLComputePipelineState,
                       _ setup: (MTLComputeCommandEncoder) -> Void) {
        guard let e = cmd.makeComputeCommandEncoder() else { return }
        e.setComputePipelineState(pipe)
        setup(e)
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        e.dispatchThreads(MTLSize(width: Int(dim.x), height: Int(dim.y), depth: 1),
                          threadsPerThreadgroup: tg)
        e.endEncoding()
    }
    private func elementwise(_ cmd: MTLCommandBuffer, _ pipe: MTLComputePipelineState,
                             count: Int, _ setup: (MTLComputeCommandEncoder) -> Void) {
        guard let e = cmd.makeComputeCommandEncoder() else { return }
        e.setComputePipelineState(pipe)
        setup(e)
        let w = pipe.maxTotalThreadsPerThreadgroup
        e.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        e.endEncoding()
    }
    private func particles(_ cmd: MTLCommandBuffer, _ pipe: MTLComputePipelineState,
                           _ setup: (MTLComputeCommandEncoder) -> Void) {
        elementwise(cmd, pipe, count: particleCount, setup)
    }
}
