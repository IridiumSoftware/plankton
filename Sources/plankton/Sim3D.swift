import Metal
import simd
import Foundation

// Mirrors `struct Particle3D` in Shaders3D.source (two float3 = 32 bytes).
struct Particle3D {
    var pos: SIMD3<Float>
    var vel: SIMD3<Float>
}

// Mirrors `struct MoveParams3` in Shaders3D.source (9 floats, 36 bytes).
struct MoveParams3GPU {
    var swim, sensorDist, sensorAngle, turn, axialForce, fluidPull, senseScale, planeSamples, cohesion: Float
}

// Stage-2a 3D simulation: agents force an N³ incompressible fluid (default 128,
// the app passes 160) and ride it. Per frame: advect velocity → ν∇² diffuse →
// weak drag → net-zero dipole splat → project (divergence, 6-neighbour Jacobi,
// subtract gradient) → MacCormack dye advect → agents move (brain + chemotaxis).
final class Sim3D {
    let count: Int
    let dim: SIMD3<UInt32>
    let particleBuffer: MTLBuffer
    let cohortBuffer: MTLBuffer        // uint per agent: which cohort/strategy it runs
    let ruleBuffer: MTLBuffer          // nCohorts × 20 float4 (one 80-param brain per cohort)
    private(set) var vel: MTLBuffer
    private var velTmp: MTLBuffer
    private var pressure: MTLBuffer
    private var pressureTmp: MTLBuffer
    private let divg: MTLBuffer
    private(set) var dye: MTLBuffer            // 3D density volume (rendered)
    private var dyeTmp: MTLBuffer
    private var dyeTmp2: MTLBuffer   // MacCormack backward-advect scratch

    private let scalePipe, advectPipe, diffusePipe, splatPipe, divPipe, jacobiPipe,
                subPipe, movePipe, advectDyePipe, advectDyeBackPipe,
                maccormackPipe: MTLComputePipelineState

    var velDamp: Float = 0.97        // weak large-scale drag; viscosity does the cascade
    var viscosity: Float = 0.10      // ν∇² scale-selective viscosity — the FAITHFUL fix vs uniform drag
    var dipoleLen: Float = 2.5       // net-zero force-dipole separation (cells)
    var forceGain: Float = 0.5
    var fluidPull: Float = 2.0
    var jacobiIters: Int = 25
    // brain (Stage 2b)
    var swim: Float = 0.10
    var sensorDist: Float = 0.015
    var sensorAngle: Float = 0.5
    var turn: Float = 0.05
    var axialForce: Float = 0.02
    var senseScale: Float = 3.0
    var planeSamples: Float = 2
    var cohesion: Float = 0.15       // chemotaxis up the dye gradient → aggregation (the creature-maker)
    var dyeDecay: Float = 0.985
    var dyeAmount: Float = 1.0
    var simSpeed: Float = 1.0        // sim steps per rendered frame (<1 = slow-mo via frame skip, 0 = pause)
    var mutationStrength: Float = 0.3 // right-click breed: mutation amount
    private var frame: UInt32 = 0

    init(device: MTLDevice, library: MTLLibrary, count: Int = 1 << 20, fieldDim: Int = 128) {
        self.count = count
        dim = SIMD3<UInt32>(UInt32(fieldDim), UInt32(fieldDim), UInt32(fieldDim))
        let n3 = fieldDim * fieldDim * fieldDim

        func zeroed(_ floats: Int) -> MTLBuffer {
            let b = device.makeBuffer(length: MemoryLayout<Float>.stride * floats,
                                      options: .storageModeShared)!
            memset(b.contents(), 0, b.length)
            return b
        }
        vel = zeroed(3 * n3); velTmp = zeroed(3 * n3)
        pressure = zeroed(n3); pressureTmp = zeroed(n3); divg = zeroed(n3)
        dye = zeroed(n3); dyeTmp = zeroed(n3); dyeTmp2 = zeroed(n3)

        // particles: random position; velocity = ABC flow (persistent forcing)
        func abc(_ p: SIMD3<Float>) -> SIMD3<Float> {
            let x = p.x * 2 * .pi, y = p.y * 2 * .pi, z = p.z * 2 * .pi
            let A: Float = 1.0, B: Float = 0.7, C: Float = 0.43
            return SIMD3(A * sin(z) + C * cos(y), B * sin(x) + A * cos(z), C * sin(y) + B * cos(x))
        }
        var rng = SystemRandomNumberGenerator()
        var parts = [Particle3D](); parts.reserveCapacity(count)
        for _ in 0..<count {
            let p = SIMD3(Float.random(in: 0..<1, using: &rng),
                          Float.random(in: 0..<1, using: &rng),
                          Float.random(in: 0..<1, using: &rng))
            parts.append(Particle3D(pos: p, vel: abc(p) * 0.06))
        }
        particleBuffer = device.makeBuffer(bytes: &parts,
                                           length: MemoryLayout<Particle3D>.stride * count,
                                           options: .storageModeShared)!

        // per-agent cohort index (was a fixed index partition; now explicit so
        // ecology mode can reassign membership). Seeded to the same partition.
        var coh = Sim3D.partitionCohorts(count: count)
        cohortBuffer = device.makeBuffer(bytes: &coh, length: MemoryLayout<UInt32>.stride * count,
                                         options: .storageModeShared)!

        var rule = Sim3D.randomRule()
        ruleBuffer = device.makeBuffer(bytes: &rule,
                                       length: rule.count * MemoryLayout<SIMD4<Float>>.stride,
                                       options: .storageModeShared)!

        func pipe(_ name: String) -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else { fatalError("\(name) not found") }
            return try! device.makeComputePipelineState(function: fn)
        }
        scalePipe = pipe("scale3d"); advectPipe = pipe("advect3d"); splatPipe = pipe("splat3d")
        diffusePipe = pipe("diffuse3d")
        divPipe = pipe("divergence3d"); jacobiPipe = pipe("jacobi3d")
        subPipe = pipe("subgrad3d"); movePipe = pipe("move3d")
        advectDyePipe = pipe("advectDye3d")
        advectDyeBackPipe = pipe("advectDyeBack3d")
        maccormackPipe = pipe("maccormackDye3d")
    }

    func encode(into cmd: MTLCommandBuffer) {
        var dimv = dim, velDampv = velDamp, forceGainv = forceGain
        var viscv = min(viscosity, 0.16)   // clamp below the 3D explicit-diffusion stability limit (1/6)

        field(cmd, advectPipe) { e in
            e.setBuffer(self.vel, offset: 0, index: 0)
            e.setBuffer(self.velTmp, offset: 0, index: 1)
            e.setBytes(&dimv, length: 16, index: 2)
        }
        // viscous diffusion ν∇² FUSED with the large-scale drag (velDamp): velTmp →
        // vel, then swap so velTmp holds the diffused+damped field. The old separate
        // scale3d damping pass is gone — one fewer full velocity read+write per step.
        field(cmd, diffusePipe) { e in
            e.setBuffer(self.velTmp, offset: 0, index: 0)
            e.setBuffer(self.vel, offset: 0, index: 1)
            e.setBytes(&dimv, length: 16, index: 2)
            e.setBytes(&viscv, length: 4, index: 3)
            e.setBytes(&velDampv, length: 4, index: 4)
        }
        swap(&vel, &velTmp)
        var dyeAmtv = dyeAmount
        var dipoleLenv = dipoleLen
        particles(cmd, splatPipe) { e in
            e.setBuffer(self.velTmp, offset: 0, index: 0)
            e.setBuffer(self.dye, offset: 0, index: 1)
            e.setBuffer(self.particleBuffer, offset: 0, index: 2)
            e.setBytes(&dimv, length: 16, index: 3)
            e.setBytes(&forceGainv, length: 4, index: 4)
            e.setBytes(&dyeAmtv, length: 4, index: 5)
            e.setBytes(&dipoleLenv, length: 4, index: 6)
        }
        field(cmd, divPipe) { e in
            e.setBuffer(self.velTmp, offset: 0, index: 0)
            e.setBuffer(self.divg, offset: 0, index: 1)
            e.setBytes(&dimv, length: 16, index: 2)
        }
        for _ in 0..<jacobiIters {
            field(cmd, jacobiPipe) { e in
                e.setBuffer(self.pressure, offset: 0, index: 0)
                e.setBuffer(self.pressureTmp, offset: 0, index: 1)
                e.setBuffer(self.divg, offset: 0, index: 2)
                e.setBytes(&dimv, length: 16, index: 3)
            }
            swap(&pressure, &pressureTmp)
        }
        field(cmd, subPipe) { e in
            e.setBuffer(self.velTmp, offset: 0, index: 0)
            e.setBuffer(self.pressure, offset: 0, index: 1)
            e.setBytes(&dimv, length: 16, index: 2)
        }
        swap(&vel, &velTmp)
        // MacCormack dye advection (3 passes — see Shaders3D): forward into dyeTmp,
        // backward into dyeTmp2, then correct+clamp+decay in place in dyeTmp2.
        var dyeDecayv = dyeDecay
        var noDecay: Float = 1.0
        field(cmd, advectDyePipe) { e in
            e.setBuffer(self.dye, offset: 0, index: 0)
            e.setBuffer(self.dyeTmp, offset: 0, index: 1)
            e.setBuffer(self.vel, offset: 0, index: 2)
            e.setBytes(&dimv, length: 16, index: 3)
            e.setBytes(&noDecay, length: 4, index: 4)
        }
        field(cmd, advectDyeBackPipe) { e in
            e.setBuffer(self.dyeTmp, offset: 0, index: 0)
            e.setBuffer(self.dyeTmp2, offset: 0, index: 1)
            e.setBuffer(self.vel, offset: 0, index: 2)
            e.setBytes(&dimv, length: 16, index: 3)
        }
        field(cmd, maccormackPipe) { e in
            e.setBuffer(self.dye, offset: 0, index: 0)
            e.setBuffer(self.dyeTmp, offset: 0, index: 1)
            e.setBuffer(self.dyeTmp2, offset: 0, index: 2)
            e.setBuffer(self.vel, offset: 0, index: 3)
            e.setBytes(&dimv, length: 16, index: 4)
            e.setBytes(&dyeDecayv, length: 4, index: 5)
        }
        swap(&dye, &dyeTmp2)
        frame &+= 1
        var framev = frame
        var mp = MoveParams3GPU(swim: swim, sensorDist: sensorDist, sensorAngle: sensorAngle,
                                turn: turn, axialForce: axialForce, fluidPull: fluidPull,
                                senseScale: senseScale, planeSamples: planeSamples, cohesion: cohesion)
        var countv = UInt32(count)
        particles(cmd, movePipe) { e in
            e.setBuffer(self.particleBuffer, offset: 0, index: 0)
            e.setBuffer(self.vel, offset: 0, index: 1)
            e.setBytes(&dimv, length: 16, index: 2)
            e.setBytes(&mp, length: MemoryLayout<MoveParams3GPU>.stride, index: 3)
            e.setBuffer(self.ruleBuffer, offset: 0, index: 4)
            e.setBytes(&framev, length: 4, index: 5)
            e.setBuffer(self.dye, offset: 0, index: 6)
            e.setBytes(&countv, length: 4, index: 7)
            e.setBuffer(self.cohortBuffer, offset: 0, index: 8)
        }
    }

    // the fixed index-partition cohort assignment (default / pre-ecology layout)
    static func partitionCohorts(count: Int) -> [UInt32] {
        (0..<count).map { UInt32(($0 * nCohorts) / count) }
    }

    // 8 sub-populations, each with its own 80-param brain (mirrors the 2D engine)
    static let nCohorts = 8

    static func randomRule() -> [SIMD4<Float>] {
        var rng = SystemRandomNumberGenerator()
        func r4(_ lo: Float, _ hi: Float) -> SIMD4<Float> {
            SIMD4(Float.random(in: lo...hi, using: &rng), Float.random(in: lo...hi, using: &rng),
                  Float.random(in: lo...hi, using: &rng), Float.random(in: lo...hi, using: &rng))
        }
        var r = [SIMD4<Float>]()
        for _ in 0..<nCohorts {
            for _ in 0..<10 { r.append(r4(-2, 2)) }   // frequencies
            for _ in 0..<10 { r.append(r4(-1, 1)) }   // amplitudes
        }
        return r
    }

    func rerollRule() {
        var r = Sim3D.randomRule()
        memcpy(ruleBuffer.contents(), &r, r.count * MemoryLayout<SIMD4<Float>>.stride)
        print("3D brains re-rolled (\(Sim3D.nCohorts) cohorts)")
    }

    // Right-click breed (mirrors the 2D breedAt): vote the dominant cohort among
    // agents near the click RAY (a 3D click is a ray, not a point), then set every
    // cohort to a mutation of the winner (cohort 0 = exact parent). All CPU.
    func breedAt(rayOrigin ro: SIMD3<Float>, rayDir rd: SIMD3<Float>) {
        let nc = Sim3D.nCohorts
        let p = particleBuffer.contents().bindMemory(to: Particle3D.self, capacity: count)
        let coh = cohortBuffer.contents().bindMemory(to: UInt32.self, capacity: count)
        var votes = [Int](repeating: 0, count: nc)
        let radius2: Float = 0.08 * 0.08
        let stepN = max(1, count / 200_000)   // subsample for speed
        var i = 0
        while i < count {
            // perpendicular distance from agent to the ray (forward half only)
            let w = p[i].pos - ro
            let t = simd_dot(w, rd)
            if t > 0 {
                let perp = w - rd * t
                if simd_length_squared(perp) < radius2 { votes[Int(coh[i])] += 1 }   // vote by actual cohort
            }
            i += stepN
        }
        guard let parent = votes.indices.max(by: { votes[$0] < votes[$1] }),
              votes[parent] > 0 else { print("3D breed: no agents near click ray"); return }

        let rf = ruleBuffer.contents().bindMemory(to: Float.self, capacity: nc * 80)
        let parentRule = Array(UnsafeBufferPointer(start: rf + parent * 80, count: 80))
        var rng = SystemRandomNumberGenerator()
        let m = mutationStrength
        for c in 0..<nc {
            for k in 0..<80 {
                rf[c * 80 + k] = (c == 0) ? parentRule[k]
                    : parentRule[k] + Float.random(in: -m...m, using: &rng)
            }
        }
        print("3D bred from cohort \(parent) (\(votes[parent]) votes)")
    }

    // ── ecology mode (replicator-mutator over the 8 cohort strategies) ──────
    // Mirrors the 2D Simulation: cohort frequencies evolve by the replicator
    // equation under a well-mixed payoff π = A·p; agents are relabelled each step
    // so the cohort colours track p; extinct strategies reseed from the leader.
    var ecology = Ecology()
    var ecologyOn = false
    private var ecoFrame = 0
    private let ecoEvery = 8
    private let ecoDt = 0.05
    private var cohortCounts = [Int](repeating: 0, count: Sim3D.nCohorts)
    private var reallocCursor = 0
    private var ecoSteps = 0

    private func recountCohorts() {
        let c = cohortBuffer.contents().bindMemory(to: UInt32.self, capacity: count)
        var cur = [Int](repeating: 0, count: Sim3D.nCohorts)
        for i in 0..<count { cur[Int(c[i])] += 1 }
        cohortCounts = cur
    }

    func syncEcologyFromAgents() {
        recountCohorts()
        let s = Double(count)
        ecology.p = s > 0 ? cohortCounts.map { Double($0) / s } : ecology.p
        ecoFrame = 0; ecoSteps = 0; reallocCursor = 0
    }

    func setEcologyPreset(_ preset: Ecology.Preset) { ecology.A = Ecology.matrix(preset) }

    func stepEcology() {
        guard ecologyOn else { return }
        ecoFrame += 1
        guard ecoFrame % ecoEvery == 0 else { return }
        ecology.step(dt: ecoDt)
        ecology.reseedExtinct { dead, leader in self.mutateRuleBlock(dead, from: leader) }
        ecoSteps += 1
        if ecoSteps % 256 == 0 { recountCohorts() }
        reallocateCohorts(to: ecology.p)
    }

    private func mutateRuleBlock(_ dead: Int, from leader: Int) {
        let rf = ruleBuffer.contents().bindMemory(to: Float.self, capacity: Sim3D.nCohorts * 80)
        var rng = SystemRandomNumberGenerator()
        let m = max(0.05, mutationStrength)
        for k in 0..<80 { rf[dead * 80 + k] = rf[leader * 80 + k] + Float.random(in: -m...m, using: &rng) }
    }

    // bounded incremental relabel toward round(p_i·N) — same cheap scan as 2D
    private func reallocateCohorts(to p: [Double]) {
        let n = Sim3D.nCohorts, N = count
        var tgt = p.map { Int(($0 * Double(N)).rounded()) }
        var drift = N - tgt.reduce(0, +)
        var k = 0
        while drift != 0 { let s = drift > 0 ? 1 : -1; if tgt[k % n] + s >= 0 { tgt[k % n] += s; drift -= s }; k += 1 }
        var need = (0..<n).map { tgt[$0] - cohortCounts[$0] }
        var recvNeed = need.reduce(0) { $0 + max(0, $1) }
        if recvNeed == 0 { return }
        let c = cohortBuffer.contents().bindMemory(to: UInt32.self, capacity: N)
        let scanBudget = 60_000
        var i = reallocCursor, scanned = 0, recv = 0
        while scanned < scanBudget && recvNeed > 0 {
            let old = Int(c[i])
            if need[old] < 0 {
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

    // ── full-state capture: vel + dye + particles + brain + the param values (extra) ──
    func serializeState(_ extra: [Float]) -> Data {
        var data = Data()
        func putI(_ v: Int32) { var x = v; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        putI(0x464C5533); putI(3)                                  // magic "FLU3", version (3 = + per-agent cohorts)
        putI(Int32(dim.x)); putI(Int32(count))
        for b in [vel, dye, particleBuffer, ruleBuffer, cohortBuffer] { data.append(Data(bytes: b.contents(), count: b.length)) }
        putI(Int32(extra.count))
        for v in extra { var vv = v; withUnsafeBytes(of: &vv) { data.append(contentsOf: $0) } }
        return data
    }

    @discardableResult
    func applyState(_ data: Data) -> [Float]? {
        var out: [Float]? = nil
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard raw.count >= 16 else { return }
            var off = 0
            func i32() -> Int32 { let v = raw.loadUnaligned(fromByteOffset: off, as: Int32.self); off += 4; return v }
            guard i32() == 0x464C5533 else { print("3D capture: bad magic"); return }
            let version = Int(i32())
            let fd = Int(i32()), c = Int(i32())
            guard fd == Int(dim.x), c == count else { print("3D capture: dims mismatch (\(fd)/\(c) vs \(Int(dim.x))/\(count))"); return }
            for b in [vel, dye, particleBuffer] {
                memcpy(b.contents(), raw.baseAddress!.advanced(by: off), b.length); off += b.length
            }
            if version >= 2 {
                memcpy(ruleBuffer.contents(), raw.baseAddress!.advanced(by: off), ruleBuffer.length)
                off += ruleBuffer.length
            } else {
                // v1 stored a single 80-float brain — replicate it into all cohorts
                let single = 80 * 4
                let rp = ruleBuffer.contents()
                for cix in 0..<Sim3D.nCohorts {
                    memcpy(rp.advanced(by: cix * single), raw.baseAddress!.advanced(by: off), single)
                }
                off += single
            }
            if version >= 3 {
                memcpy(cohortBuffer.contents(), raw.baseAddress!.advanced(by: off), cohortBuffer.length)
                off += cohortBuffer.length
            } else {
                var coh = Sim3D.partitionCohorts(count: count)   // pre-v3: rebuild the partition
                memcpy(cohortBuffer.contents(), &coh, count * 4)
            }
            recountCohorts()
            let np = Int(i32()); var p = [Float]()
            for _ in 0..<np { p.append(raw.loadUnaligned(fromByteOffset: off, as: Float.self)); off += 4 }
            out = p
        }
        return out
    }

    private func field(_ cmd: MTLCommandBuffer, _ p: MTLComputePipelineState,
                       _ setup: (MTLComputeCommandEncoder) -> Void) {
        guard let e = cmd.makeComputeCommandEncoder() else { return }
        e.setComputePipelineState(p); setup(e)
        e.dispatchThreads(MTLSize(width: Int(dim.x), height: Int(dim.y), depth: Int(dim.z)),
                          threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 4))
        e.endEncoding()
    }
    private func elementwise(_ cmd: MTLCommandBuffer, _ p: MTLComputePipelineState,
                             _ count: Int, _ setup: (MTLComputeCommandEncoder) -> Void) {
        guard let e = cmd.makeComputeCommandEncoder() else { return }
        e.setComputePipelineState(p); setup(e)
        let w = p.maxTotalThreadsPerThreadgroup
        e.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        e.endEncoding()
    }
    private func particles(_ cmd: MTLCommandBuffer, _ p: MTLComputePipelineState,
                           _ setup: (MTLComputeCommandEncoder) -> Void) {
        elementwise(cmd, p, count, setup)
    }
}
