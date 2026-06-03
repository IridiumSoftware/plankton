import Metal
import simd
import Foundation

// Mirrors `struct Particle` in Shaders.source (16 bytes: pos.xy, vel.xy).
struct Particle {
    var pos: SIMD2<Float>
    var vel: SIMD2<Float>
}

// Owns the grid fields + particle buffer and encodes one Stable-Fluids step:
//   advect velocity → damp → splat (force + dye) → project (divergence, Jacobi
//   pressure, subtract gradient) → advect dye → agents sense/steer/ride.
// Live tunables are read from `params` each frame. Each pass is its own
// encoder; Metal's hazard tracking serializes them.
final class Simulation {
    let particleCount: Int
    let dim: SIMD2<UInt32>
    private let params: Params
    private let mouse: MouseInput

    private(set) var vel: MTLBuffer
    private var velTmp: MTLBuffer
    private var pressure: MTLBuffer
    private var pressureTmp: MTLBuffer
    private let divg: MTLBuffer
    private(set) var dye: MTLBuffer
    private var dyeTmp: MTLBuffer
    let particleBuffer: MTLBuffer
    let ruleBuffer: MTLBuffer          // 20 float4: [0..9] freqs, [10..19] amps

    private let scalePipe, advectVelPipe, splatPipe, mouseStirPipe, divPipe,
                jacobiPipe, subGradPipe, advectDyePipe, movePipe: MTLComputePipelineState

    private let dyeAmount: Float = 1.0     // not live-tuned
    var jacobiIters: Int = 30              // not live-tuned (changes encoder count)

    init(device: MTLDevice, library: MTLLibrary, params: Params, mouse: MouseInput,
         particleCount: Int = 1 << 20, fieldDim: Int = 1024) {
        self.params = params
        self.mouse = mouse
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

        var particles = Simulation.seedParticles(count: particleCount)
        particleBuffer = device.makeBuffer(
            bytes: &particles,
            length: MemoryLayout<Particle>.stride * particleCount,
            options: .storageModeShared)!

        var rule = Simulation.randomRule()
        ruleBuffer = device.makeBuffer(
            bytes: &rule,
            length: 20 * MemoryLayout<SIMD4<Float>>.stride,
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
        mouseStirPipe = pipe("mouse_stir")
        divPipe       = pipe("divergence")
        jacobiPipe    = pipe("jacobi")
        subGradPipe   = pipe("subtract_gradient")
        advectDyePipe = pipe("advect_dye")
        movePipe      = pipe("move_particles")
    }

    func encode(into cmd: MTLCommandBuffer) {
        let n = Int(dim.x) * Int(dim.y)
        var dimv = dim
        var velDampv = params.velDamp
        var forceGainv = params.forceGain
        var dyeAmtv = dyeAmount
        var dyeDecayv = params.dyeDecay
        var mp = MoveParamsGPU(swim: params.swim, sensorDist: params.sensorDist,
                               sensorAngle: params.sensorAngle, turn: params.turn,
                               fluidPull: params.fluidPull, senseScale: params.senseScale,
                               speedGain: params.speedGain)

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
        // 3. splat force + dye
        particles(cmd, splatPipe) { e in
            e.setBuffer(self.velTmp, offset: 0, index: 0)
            e.setBuffer(self.dye, offset: 0, index: 1)
            e.setBuffer(self.particleBuffer, offset: 0, index: 2)
            e.setBytes(&dimv, length: 8, index: 3)
            e.setBytes(&forceGainv, length: 4, index: 4)
            e.setBytes(&dyeAmtv, length: 4, index: 5)
        }
        // 3b. mouse stir (force + dye), only while dragging
        if mouse.active {
            var mu = MouseUniformGPU(pos: mouse.posN, vel: mouse.velN,
                                     radius: params.mouseRadius, forceGain: params.mouseForce,
                                     dyeGain: params.mouseDye, active: 1)
            field(cmd, mouseStirPipe) { e in
                e.setBuffer(self.velTmp, offset: 0, index: 0)
                e.setBuffer(self.dye, offset: 0, index: 1)
                e.setBytes(&dimv, length: 8, index: 2)
                e.setBytes(&mu, length: MemoryLayout<MouseUniformGPU>.stride, index: 3)
            }
            mouse.velN = .zero
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
            swap(&pressure, &pressureTmp)
        }
        // 6. subtract gradient → velTmp divergence-free
        field(cmd, subGradPipe) { e in
            e.setBuffer(self.velTmp, offset: 0, index: 0)
            e.setBuffer(self.pressure, offset: 0, index: 1)
            e.setBytes(&dimv, length: 8, index: 2)
        }
        swap(&vel, &velTmp)
        // 7. advect dye by the projected field (+ decay)
        field(cmd, advectDyePipe) { e in
            e.setBuffer(self.dye, offset: 0, index: 0)
            e.setBuffer(self.dyeTmp, offset: 0, index: 1)
            e.setBuffer(self.vel, offset: 0, index: 2)
            e.setBytes(&dimv, length: 8, index: 3)
            e.setBytes(&dyeDecayv, length: 4, index: 4)
        }
        swap(&dye, &dyeTmp)
        // 8. agents: sense the flow → Fourier brain → steer + ride the fluid
        particles(cmd, movePipe) { e in
            e.setBuffer(self.particleBuffer, offset: 0, index: 0)
            e.setBuffer(self.vel, offset: 0, index: 1)
            e.setBytes(&dimv, length: 8, index: 2)
            e.setBytes(&mp, length: MemoryLayout<MoveParamsGPU>.stride, index: 3)
            e.setBuffer(self.ruleBuffer, offset: 0, index: 4)
        }
    }

    // Particles at random positions with velocities from a few overlaid Gaussian
    // vortices, so the opening / post-reset flow is visibly swirling.
    static func seedParticles(count: Int) -> [Particle] {
        struct Vortex { var c: SIMD2<Float>; var s: Float; var r: Float }
        var rng = SystemRandomNumberGenerator()
        var vortices = [Vortex]()
        for _ in 0..<6 {
            vortices.append(Vortex(
                c: SIMD2(Float.random(in: 0..<1, using: &rng), Float.random(in: 0..<1, using: &rng)),
                s: Float.random(in: -1..<1, using: &rng),
                r: Float.random(in: 0.12..<0.30, using: &rng)))
        }
        var particles = [Particle](); particles.reserveCapacity(count)
        for _ in 0..<count {
            let pos = SIMD2(Float.random(in: 0..<1, using: &rng),
                            Float.random(in: 0..<1, using: &rng))
            var v = SIMD2<Float>(0, 0)
            for vt in vortices {
                let dvec = pos - vt.c
                let fall = exp(-simd_length_squared(dvec) / (vt.r * vt.r))
                v += vt.s * SIMD2(-dvec.y, dvec.x) * fall
            }
            particles.append(Particle(pos: pos, vel: v * 0.12))
        }
        return particles
    }

    // Clear all fields and re-seed the particles (fresh canvas, same params/brain).
    func reset() {
        for b in [vel, velTmp, pressure, pressureTmp, divg, dye, dyeTmp] {
            memset(b.contents(), 0, b.length)
        }
        var particles = Simulation.seedParticles(count: particleCount)
        memcpy(particleBuffer.contents(), &particles,
               MemoryLayout<Particle>.stride * particleCount)
        print("reset")
    }

    // Read / write the 80-float brain (for presets).
    func ruleSnapshot() -> [Float] {
        let p = ruleBuffer.contents().bindMemory(to: Float.self, capacity: 80)
        return Array(UnsafeBufferPointer(start: p, count: 80))
    }
    func loadRule(_ floats: [Float]) {
        guard floats.count == 80 else { return }
        var f = floats
        memcpy(ruleBuffer.contents(), &f, 80 * MemoryLayout<Float>.stride)
    }

    // Generate a fresh random brain (10 freq float4 + 10 amp float4).
    static func randomRule() -> [SIMD4<Float>] {
        var rng = SystemRandomNumberGenerator()
        var r = [SIMD4<Float>]()
        for _ in 0..<10 {
            r.append(SIMD4(Float.random(in: -2...2, using: &rng), Float.random(in: -2...2, using: &rng),
                           Float.random(in: -2...2, using: &rng), Float.random(in: -2...2, using: &rng)))
        }
        for _ in 0..<10 {
            r.append(SIMD4(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                           Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng)))
        }
        return r
    }

    // Re-roll the global brain in place (bound to the `r` key).
    func rerollRule() {
        var r = Simulation.randomRule()
        memcpy(ruleBuffer.contents(), &r, 20 * MemoryLayout<SIMD4<Float>>.stride)
        print("brain re-rolled")
    }

    // ── encoder helpers ─────────────────────────────────────────────────
    private func field(_ cmd: MTLCommandBuffer, _ pipe: MTLComputePipelineState,
                       _ setup: (MTLComputeCommandEncoder) -> Void) {
        guard let e = cmd.makeComputeCommandEncoder() else { return }
        e.setComputePipelineState(pipe)
        setup(e)
        e.dispatchThreads(MTLSize(width: Int(dim.x), height: Int(dim.y), depth: 1),
                          threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
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
