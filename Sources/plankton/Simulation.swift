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
    private(set) var divDisp: MTLBuffer         // divergence of the projected field (display)
    let particleBuffer: MTLBuffer
    let cohortBuffer: MTLBuffer        // uint per agent: which cohort/strategy it runs
    let ruleBuffer: MTLBuffer          // 20 float4: [0..9] freqs, [10..19] amps

    private let scalePipe, advectVelPipe, diffusePipe, splatPipe, mouseStirPipe, divPipe,
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
        vort = zeroed(n); divDisp = zeroed(n)

        var particles = Simulation.seedParticles(count: particleCount)
        particleBuffer = device.makeBuffer(
            bytes: &particles,
            length: MemoryLayout<Particle>.stride * particleCount,
            options: .storageModeShared)!

        // per-agent cohort index (was a fixed index partition; now explicit so
        // ecology mode can reassign membership). Seeded to the same partition.
        var coh = Simulation.partitionCohorts(count: particleCount)
        cohortBuffer = device.makeBuffer(
            bytes: &coh, length: MemoryLayout<UInt32>.stride * particleCount,
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
        diffusePipe   = pipe("diffuse_velocity")
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

    var gpuSync: (() -> Void)?   // set by the renderer: wait for the in-flight frame

    func encode(into cmd: MTLCommandBuffer) {
        if mouse.breedRequested {
            gpuSync?()                 // breedAt reads particle positions on the CPU — don't race the kernels
            breedAt(mouse.breedPos)
            mouse.breedRequested = false
        }
        let n = Int(dim.x) * Int(dim.y)
        var dimv = dim
        var countv = UInt32(particleCount)
        var velDampv = params.velDamp
        var viscv = min(params.viscosity, 0.24)   // clamp below the explicit-diffusion stability limit
        var forceGainv = params.forceGain
        var dipoleLenv = params.dipoleLen
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
        // 1b. viscous diffusion ν∇² (the FAITHFUL fix vs uniform drag): diffuse
        //     velTmp → vel (scratch), then swap so velTmp holds the diffused field.
        //     `vel` is free scratch here — advect already consumed it; the final
        //     swap (after projection) restores the projected field into `vel`.
        field(cmd, diffusePipe) { e in
            e.setBuffer(self.velTmp, offset: 0, index: 0)
            e.setBuffer(self.vel, offset: 0, index: 1)
            e.setBytes(&dimv, length: 8, index: 2)
            e.setBytes(&viscv, length: 4, index: 3)
        }
        swap(&vel, &velTmp)
        // 2. weak large-scale drag on velTmp (Rayleigh; viscosity does the cascade)
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
            e.setBytes(&dipoleLenv, length: 4, index: 6)
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
        // (vorticity/divergence viz passes live in encodeViz(), called by the
        // renderer EVERY frame — not here — so the research views stay current
        // even when the sim is paused (simSpeed 0) or a capture was restored.)
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
            e.setBuffer(self.cohortBuffer, offset: 0, index: 7)
        }
    }

    // the fixed index-partition cohort assignment (default / pre-ecology layout)
    static func partitionCohorts(count: Int) -> [UInt32] {
        (0..<count).map { UInt32(($0 * nCohorts) / count) }
    }

    // ── ecology mode (replicator-mutator over the 8 cohort strategies) ──────
    // STAGE 1 (global): the cohort frequencies evolve by the replicator equation
    // under a well-mixed payoff π = A·p; agents are relabelled each step so the
    // on-screen colour proportions track p. Extinct strategies are reseeded as a
    // mutation of the leader (the mutator). Stage 2 will make the payoff spatial.
    var ecology = Ecology()
    var ecologyOn = false
    private var ecoFrame = 0
    private let ecoEvery = 8           // step the replicator every N rendered frames
    private let ecoDt = 0.05
    // tracked cohort counts so reallocation never re-scans all N to recount; a
    // bounded scan per step keeps the main thread cheap (the recount was the
    // O(N)-every-step cost that made ecology mode janky live).
    private var cohortCounts = [Int](repeating: 0, count: Simulation.nCohorts)
    private var reallocCursor = 0
    private var ecoSteps = 0

    // full recount of cohortCounts from the buffer (one O(N) pass — used on enable
    // and periodically to correct any drift from external edits).
    private func recountCohorts() {
        let coh = cohortBuffer.contents().bindMemory(to: UInt32.self, capacity: particleCount)
        var cur = [Int](repeating: 0, count: Simulation.nCohorts)
        for i in 0..<particleCount { cur[Int(coh[i])] += 1 }
        cohortCounts = cur
    }

    // sync the ecology's frequencies from the current agent counts (call on enable)
    func syncEcologyFromAgents() {
        recountCohorts()
        let s = Double(particleCount)
        ecology.p = s > 0 ? cohortCounts.map { Double($0) / s } : ecology.p
        ecoFrame = 0; ecoSteps = 0; reallocCursor = 0
    }

    func setEcologyPreset(_ preset: Ecology.Preset) { ecology.A = Ecology.matrix(preset) }

    // advance the ecology one tick (throttled); call once per rendered frame.
    func stepEcology() {
        guard ecologyOn else { return }
        ecoFrame += 1
        guard ecoFrame % ecoEvery == 0 else { return }
        ecology.step(dt: ecoDt)
        ecology.reseedExtinct { dead, leader in self.mutateRuleBlock(dead, from: leader) }
        ecoSteps += 1
        if ecoSteps % 256 == 0 { recountCohorts() }      // rare drift correction (counts are exact during normal play)
        reallocateCohorts(to: ecology.p)
    }

    // mutate cohort `dead`'s 80-param brain from `leader` (the mutator's innovation)
    private func mutateRuleBlock(_ dead: Int, from leader: Int) {
        let rf = ruleBuffer.contents().bindMemory(to: Float.self, capacity: Simulation.nCohorts * 80)
        var rng = SystemRandomNumberGenerator()
        let m = max(0.05, params.mutationStrength)
        for k in 0..<80 { rf[dead * 80 + k] = rf[leader * 80 + k] + Float.random(in: -m...m, using: &rng) }
    }

    // Relabel agents toward round(p_i · N) using TRACKED counts and a BOUNDED scan
    // (a persistent cursor sweeps the buffer across steps). Per step touches at most
    // `scanBudget` agents, so the cost is O(scanBudget), not O(N); membership lags p
    // by a few steps, which is invisible since p moves slowly.
    private func reallocateCohorts(to p: [Double]) {
        let n = Simulation.nCohorts, N = particleCount
        var tgt = p.map { Int(($0 * Double(N)).rounded()) }
        var drift = N - tgt.reduce(0, +)                 // fix rounding so Σtgt == N
        var k = 0
        while drift != 0 { let s = drift > 0 ? 1 : -1; if tgt[k % n] + s >= 0 { tgt[k % n] += s; drift -= s }; k += 1 }
        var need = (0..<n).map { tgt[$0] - cohortCounts[$0] }     // >0 wants more, <0 surplus
        var recvNeed = need.reduce(0) { $0 + max(0, $1) }
        if recvNeed == 0 { return }

        let c = cohortBuffer.contents().bindMemory(to: UInt32.self, capacity: N)
        let scanBudget = 60_000                          // ~1/17 of N → full sweep over ~17 steps
        var i = reallocCursor, scanned = 0, recv = 0
        while scanned < scanBudget && recvNeed > 0 {
            let old = Int(c[i])
            if need[old] < 0 {                            // this agent's cohort has a surplus → move it
                while recv < n && need[recv] <= 0 { recv += 1 }
                if recv >= n { break }
                c[i] = UInt32(recv)
                cohortCounts[old] -= 1; cohortCounts[recv] += 1
                need[old] += 1; need[recv] -= 1; recvNeed -= 1
            }
            i += 1; if i >= N { i = 0 }
            scanned += 1
        }
        reallocCursor = i
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
    // dyeBlur/vort/divDisp are display-derived but must clear too: paused at
    // simSpeed 0 nothing recomputes them, so a stale bloom/ω ghost would persist.
    func reset() {
        for b in [vel, velTmp, pressure, pressureTmp, divg, dye, dyeTmp,
                  dyeBlur, blurTmp, vort, divDisp] {
            memset(b.contents(), 0, b.length)
        }
        var particles = Simulation.seedParticles(count: particleCount)
        memcpy(particleBuffer.contents(), &particles,
               MemoryLayout<Particle>.stride * particleCount)
        // restore the even 8-cohort partition (ecology may have scrambled membership)
        var coh = Simulation.partitionCohorts(count: particleCount)
        memcpy(cohortBuffer.contents(), &coh, particleCount * MemoryLayout<UInt32>.stride)
        recountCohorts()
        ecology.p = [Double](repeating: 1.0 / Double(Simulation.nCohorts), count: Simulation.nCohorts)
        ecoFrame = 0; ecoSteps = 0; reallocCursor = 0
        print("reset")
    }

    // Read / write all cohort brains (for presets).
    // Research-viz fields (ω, ∇·u) of the CURRENT velocity — called by the
    // renderer every frame, independent of the sim stepping, so the vorticity/
    // enstrophy/divergence views can never go stale or blank (e.g. switching
    // viewMode while paused at simSpeed 0, or right after a capture restore).
    func encodeViz(into cmd: MTLCommandBuffer) {
        var dimv = dim
        let vm = Int(params.viewMode + 0.5)
        if params.diagnosticsOn || vm == 1 || vm == 2 {
            field(cmd, vortPipe) { e in
                e.setBuffer(self.vel, offset: 0, index: 0)
                e.setBuffer(self.vort, offset: 0, index: 1)
                e.setBytes(&dimv, length: 8, index: 2)
            }
        }
        if params.diagnosticsOn || vm == 3 {
            field(cmd, divPipe) { e in
                e.setBuffer(self.vel, offset: 0, index: 0)
                e.setBuffer(self.divDisp, offset: 0, index: 1)
                e.setBytes(&dimv, length: 8, index: 2)
            }
        }
    }

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

    // ── full-state capture: vel + dye + particles + brain + params → one blob ──
    // Captures the EXACT grown configuration (every agent's pos+heading, the velocity
    // and dye fields, the brains, and all params). The reliable way to "save a creature":
    // because the system is hysteretic (AT-7), a creature can't be reproduced from
    // params alone — only its full state replays it.
    func serializeState(_ params: Params) -> Data {
        let n = Int(dim.x) * Int(dim.y)
        var data = Data()
        func putI(_ v: Int32) { var x = v; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        func putBuf(_ b: MTLBuffer, _ floats: Int) { data.append(Data(bytes: b.contents(), count: floats * 4)) }
        putI(0x464C554F); putI(2)                                  // magic "FLUO", version (2 = + per-agent cohorts)
        putI(Int32(dim.x)); putI(Int32(particleCount)); putI(Int32(Simulation.nCohorts))
        putBuf(vel, 2 * n); putBuf(dye, n)
        putBuf(particleBuffer, particleCount * 4); putBuf(ruleBuffer, Simulation.nCohorts * 80)
        putBuf(cohortBuffer, particleCount)                        // v2: per-agent cohort index
        putI(Int32(engineKnobs.count))
        for k in engineKnobs { var v = params[keyPath: k.kp]; withUnsafeBytes(of: &v) { data.append(contentsOf: $0) } }
        return data
    }

    @discardableResult
    func applyState(_ data: Data, _ params: Params) -> Bool {
        let n = Int(dim.x) * Int(dim.y); var ok = false
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard raw.count >= 20 else { return }
            var off = 0
            func i32() -> Int32 { let v = raw.loadUnaligned(fromByteOffset: off, as: Int32.self); off += 4; return v }
            func getBuf(_ b: MTLBuffer, _ floats: Int) {
                memcpy(b.contents(), raw.baseAddress!.advanced(by: off), floats * 4); off += floats * 4
            }
            guard i32() == 0x464C554F else { print("capture: bad magic"); return }
            let version = Int(i32())
            let fd = Int(i32()), pc = Int(i32()), ncoh = Int(i32())
            guard fd == Int(dim.x), pc == particleCount, ncoh == Simulation.nCohorts else {
                print("capture: dims mismatch (\(fd)/\(pc)/\(ncoh) vs \(Int(dim.x))/\(particleCount)/\(Simulation.nCohorts))"); return
            }
            getBuf(vel, 2 * n); getBuf(dye, n)
            getBuf(particleBuffer, particleCount * 4); getBuf(ruleBuffer, Simulation.nCohorts * 80)
            if version >= 2 {
                getBuf(cohortBuffer, particleCount)                // v2: per-agent cohort index
            } else {
                var coh = Simulation.partitionCohorts(count: particleCount)   // v1: rebuild the partition
                memcpy(cohortBuffer.contents(), &coh, particleCount * 4)
            }
            recountCohorts()   // keep ecology's tracked counts in sync with the restored membership
            let np = Int(i32())
            for (idx, k) in engineKnobs.enumerated() where idx < np {
                params[keyPath: k.kp] = raw.loadUnaligned(fromByteOffset: off, as: Float.self); off += 4
            }
            ok = true
        }
        return ok
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
        let coh = cohortBuffer.contents().bindMemory(to: UInt32.self, capacity: particleCount)
        var votes = [Int](repeating: 0, count: nc)
        let radius2: Float = 0.10 * 0.10
        let stepN = max(1, particleCount / 200_000)   // subsample for speed
        var i = 0
        while i < particleCount {
            if simd_length_squared(p[i].pos - clickPos) < radius2 {
                votes[Int(coh[i])] += 1          // vote by the agent's actual cohort
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
