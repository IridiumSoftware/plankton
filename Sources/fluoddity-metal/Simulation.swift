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
    static let nCohorts = 8        // sub-populations; matches N_COHORTS in Shaders

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
    private(set) var dyeBlur: MTLBuffer        // blurred dye for the bloom glow
    private var blurTmp: MTLBuffer             // separable-blur scratch
    private(set) var vort: MTLBuffer           // vorticity ω of the projected field
    let particleBuffer: MTLBuffer
    let ruleBuffer: MTLBuffer          // 20 float4: [0..9] freqs, [10..19] amps

    private let scalePipe, advectVelPipe, splatPipe, mouseStirPipe, divPipe,
                jacobiPipe, subGradPipe, advectDyePipe, movePipe, blurPipe,
                vortPipe: MTLComputePipelineState

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
        dyeBlur = zeroed(n); blurTmp = zeroed(n)
        vort = zeroed(n)

        var particles = Simulation.seedParticles(count: particleCount)
        particleBuffer = device.makeBuffer(
            bytes: &particles,
            length: MemoryLayout<Particle>.stride * particleCount,
            options: .storageModeShared)!

        var rule = Simulation.randomRule()
        ruleBuffer = device.makeBuffer(
            bytes: &rule,
            length: rule.count * MemoryLayout<SIMD4<Float>>.stride,
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
        blurPipe      = pipe("box_blur")
        vortPipe      = pipe("vorticity")
    }

    func encode(into cmd: MTLCommandBuffer) {
        if mouse.breedRequested {
            breedAt(mouse.breedPos)
            mouse.breedRequested = false
        }
        let n = Int(dim.x) * Int(dim.y)
        var dimv = dim
        var countv = UInt32(particleCount)
        var velDampv = params.velDamp
        var forceGainv = params.forceGain
        var dyeAmtv = dyeAmount
        var dyeDecayv = params.dyeDecay
        var mp = MoveParamsGPU(swim: params.swim, sensorDist: params.sensorDist,
                               sensorAngle: params.sensorAngle, turn: params.turn,
                               fluidPull: params.fluidPull, senseScale: params.senseScale,
                               speedGain: params.speedGain, cohesion: params.cohesion)

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
        // 7a. vorticity of the projected field (research-viz overlay)
        field(cmd, vortPipe) { e in
            e.setBuffer(self.vel, offset: 0, index: 0)
            e.setBuffer(self.vort, offset: 0, index: 1)
            e.setBytes(&dimv, length: 8, index: 2)
        }
        // 7. advect dye by the projected field (+ decay)
        field(cmd, advectDyePipe) { e in
            e.setBuffer(self.dye, offset: 0, index: 0)
            e.setBuffer(self.dyeTmp, offset: 0, index: 1)
            e.setBuffer(self.vel, offset: 0, index: 2)
            e.setBytes(&dimv, length: 8, index: 3)
            e.setBytes(&dyeDecayv, length: 4, index: 4)
        }
        swap(&dye, &dyeTmp)
        // 7b. blur dye → dyeBlur (separable box blur) for the bloom glow
        var stepH = SIMD2<Int32>(1, 0), stepV = SIMD2<Int32>(0, 1)
        var radius: Int32 = 10
        field(cmd, blurPipe) { e in
            e.setBuffer(self.dye, offset: 0, index: 0)
            e.setBuffer(self.blurTmp, offset: 0, index: 1)
            e.setBytes(&dimv, length: 8, index: 2)
            e.setBytes(&stepH, length: 8, index: 3)
            e.setBytes(&radius, length: 4, index: 4)
        }
        field(cmd, blurPipe) { e in
            e.setBuffer(self.blurTmp, offset: 0, index: 0)
            e.setBuffer(self.dyeBlur, offset: 0, index: 1)
            e.setBytes(&dimv, length: 8, index: 2)
            e.setBytes(&stepV, length: 8, index: 3)
            e.setBytes(&radius, length: 4, index: 4)
        }
        // 8. agents: sense the flow → Fourier brain → steer + ride the fluid
        particles(cmd, movePipe) { e in
            e.setBuffer(self.particleBuffer, offset: 0, index: 0)
            e.setBuffer(self.vel, offset: 0, index: 1)
            e.setBytes(&dimv, length: 8, index: 2)
            e.setBytes(&mp, length: MemoryLayout<MoveParamsGPU>.stride, index: 3)
            e.setBuffer(self.ruleBuffer, offset: 0, index: 4)
            e.setBytes(&countv, length: 4, index: 5)
            e.setBuffer(self.dye, offset: 0, index: 6)
        }
    }

    // Particles seeded into per-cohort tiles (so the sub-populations start
    // spatially separated for selection), with small random headings.
    static func seedParticles(count: Int) -> [Particle] {
        var rng = SystemRandomNumberGenerator()
        let cols = 4, rows = max(1, (nCohorts + cols - 1) / cols)   // 8 → 4×2
        var particles = [Particle](); particles.reserveCapacity(count)
        for i in 0..<count {
            let cohort = (i * nCohorts) / count
            let tx = cohort % cols, ty = cohort / cols
            let px = (Float(tx) + Float.random(in: 0.05..<0.95, using: &rng)) / Float(cols)
            let py = (Float(ty) + Float.random(in: 0.05..<0.95, using: &rng)) / Float(rows)
            let ang = Float.random(in: 0..<(2 * .pi), using: &rng)
            particles.append(Particle(pos: SIMD2(px, py),
                                      vel: SIMD2(cos(ang), sin(ang)) * 0.08))
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

    // Read / write all cohort brains (for presets).
    func ruleSnapshot() -> [Float] {
        let n = Simulation.nCohorts * 80
        let p = ruleBuffer.contents().bindMemory(to: Float.self, capacity: n)
        return Array(UnsafeBufferPointer(start: p, count: n))
    }
    func loadRule(_ floats: [Float]) {
        let n = Simulation.nCohorts * 80
        guard floats.count == n else { return }
        var f = floats
        memcpy(ruleBuffer.contents(), &f, n * MemoryLayout<Float>.stride)
    }

    // Generate nCohorts fresh random brains (each: 10 freq float4 + 10 amp float4).
    static func randomRule() -> [SIMD4<Float>] {
        var rng = SystemRandomNumberGenerator()
        func rand4(_ lo: Float, _ hi: Float) -> SIMD4<Float> {
            SIMD4(Float.random(in: lo...hi, using: &rng), Float.random(in: lo...hi, using: &rng),
                  Float.random(in: lo...hi, using: &rng), Float.random(in: lo...hi, using: &rng))
        }
        var r = [SIMD4<Float>]()
        for _ in 0..<nCohorts {
            for _ in 0..<10 { r.append(rand4(-2, 2)) }   // frequencies
            for _ in 0..<10 { r.append(rand4(-1, 1)) }   // amplitudes
        }
        return r
    }

    // Re-roll all cohort brains in place (bound to the `r` key / Brain button).
    func rerollRule() {
        var r = Simulation.randomRule()
        memcpy(ruleBuffer.contents(), &r, r.count * MemoryLayout<SIMD4<Float>>.stride)
        print("brains re-rolled")
    }

    // Right-click breed: vote the dominant cohort near `clickPos`, then set every
    // cohort to a mutation of it (cohort 0 = exact parent). All CPU (shared bufs).
    func breedAt(_ clickPos: SIMD2<Float>) {
        let nc = Simulation.nCohorts
        let p = particleBuffer.contents().bindMemory(to: Particle.self, capacity: particleCount)
        var votes = [Int](repeating: 0, count: nc)
        let radius2: Float = 0.10 * 0.10
        let stepN = max(1, particleCount / 200_000)   // subsample for speed
        var i = 0
        while i < particleCount {
            if simd_length_squared(p[i].pos - clickPos) < radius2 {
                votes[(i * nc) / particleCount] += 1
            }
            i += stepN
        }
        guard let parent = votes.indices.max(by: { votes[$0] < votes[$1] }),
              votes[parent] > 0 else { print("breed: no agents near click"); return }

        let rf = ruleBuffer.contents().bindMemory(to: Float.self, capacity: nc * 80)
        let parentRule = Array(UnsafeBufferPointer(start: rf + parent * 80, count: 80))
        var rng = SystemRandomNumberGenerator()
        let m = params.mutationStrength
        for c in 0..<nc {
            for k in 0..<80 {
                rf[c * 80 + k] = (c == 0) ? parentRule[k]
                    : parentRule[k] + Float.random(in: -m...m, using: &rng)
            }
        }
        print("bred from cohort \(parent) (\(votes[parent]) votes)")
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
