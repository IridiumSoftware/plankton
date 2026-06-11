import Foundation
import Metal

// Headless verification of the replicator-mutator core (run via `--ecologytest`).
// Proves the dynamics are correct before any sim coupling:
//   • simplex invariance — Σṗ ≈ 0 every step, Σp stays 1
//   • RPS payoff      → neutral CYCLES (frequencies oscillate, nobody wins/dies)
//   • coexistence     → interior fixed point (all strategies survive, ṗ → 0)
//   • dominance       → a vertex (one strategy takes over)
//   • mutator         → an extinct strategy is reseeded from the leader
func runEcologyTest() {
    let n = Ecology.n
    var allPass = true
    func check(_ name: String, _ ok: Bool, _ detail: String) {
        let pad = name.padding(toLength: 14, withPad: " ", startingAt: 0)
        print("\(pad) \(ok ? "PASS" : "FAIL")  \(detail)")
        if !ok { allPass = false }
    }

    // ── simplex invariance on a random-ish matrix (deterministic, no RNG) ──
    var A0 = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
    for i in 0..<n { for j in 0..<n { A0[i][j] = Double((i * 7 + j * 13) % 11) - 5.0 } }
    var p0 = [Double](repeating: 0, count: n)
    for i in 0..<n { p0[i] = Double((i * i + 3) % 7 + 1) }     // arbitrary start
    let s0 = p0.reduce(0, +); for i in 0..<n { p0[i] /= s0 }   // onto simplex
    var inv = Ecology(p: p0, A: A0)
    var maxSumDp = 0.0, maxSumErr = 0.0
    for _ in 0..<2000 {
        let dp = inv.step(dt: 0.01)
        maxSumDp = max(maxSumDp, abs(dp.reduce(0,+)))
        maxSumErr = max(maxSumErr, abs(inv.p.reduce(0,+) - 1.0))
    }
    check("invariance", maxSumDp < 1e-9 && maxSumErr < 1e-9,
          String(format: "max|Σṗ|=%.2e  max|Σp−1|=%.2e", maxSumDp, maxSumErr))

    // ── RPS → cycling: count crossings of the centre by strategy 0 ──
    var rps = Ecology(A: Ecology.rps())
    rps.p[0] += 0.06; rps.p[4] -= 0.06            // perturb off-centre
    let centre = 1.0 / Double(n)
    var crossings = 0, minP = 1.0, maxP = 0.0
    var prevSign = (rps.p[0] - centre) >= 0
    for _ in 0..<6000 {
        rps.step(dt: 0.01)
        let sign = (rps.p[0] - centre) >= 0
        if sign != prevSign { crossings += 1; prevSign = sign }
        minP = min(minP, rps.p.min()!); maxP = max(maxP, rps.p.max()!)
    }
    check("rps cycles", crossings >= 4 && minP > 1e-3 && maxP < 0.9,
          String(format: "crossings=%d  min_p=%.3f  max_p=%.3f", crossings, minP, maxP))

    // ── coexistence → interior fixed point (uniform), ṗ → 0 ──
    var co = Ecology(A: Ecology.coexistence())
    co.p[2] += 0.05; co.p[5] -= 0.05              // perturb (stays interior: all p > 0)
    var lastDp = 1.0
    for _ in 0..<5000 { lastDp = co.step(dt: 0.05).map(abs).max()! }
    let dev = co.p.map { abs($0 - centre) }.max()!
    check("coexistence", dev < 1e-3 && lastDp < 1e-5,
          String(format: "max|p−1/n|=%.2e  |ṗ|=%.2e", dev, lastDp))

    // ── dominance → vertex (strategy 0 → 1) ──
    var dom = Ecology(A: Ecology.dominance())
    for _ in 0..<5000 { dom.step(dt: 0.05) }
    check("dominance", dom.p[0] > 0.99, String(format: "p_0=%.4f", dom.p[0]))

    // ── mutator → reseed extinct strategies from the leader ──
    var mut = Ecology(A: Ecology.coexistence())
    mut.p = [Double](repeating: 0, count: n); mut.p[0] = 1.0      // only strategy 0 alive
    var reseededFrom: [Int: Int] = [:]
    let reseeded = mut.reseedExtinct { dead, leader in reseededFrom[dead] = leader }
    let allFromLeader = reseededFrom.values.allSatisfy { $0 == 0 }
    check("mutator", reseeded.count == n - 1 && allFromLeader && abs(mut.p.reduce(0,+) - 1) < 1e-9,
          "reseeded \(reseeded.count)/\(n - 1) strategies from leader 0")

    print(allPass ? "RESULT       : PASS — replicator-mutator core verified."
                  : "RESULT       : CHECK — a dynamics check failed.")
    if !allPass { exit(1) }
}

// In-engine check (run via `--ecologysim`): drive a real Simulation through ecology
// mode and confirm the per-agent cohort buffer (cohortBuffer) actually tracks the
// replicator frequencies — the glue between the ODE and the GPU population. No
// window needed (the reallocation + mutator are CPU; the GPU sim isn't required).
func runEcologySimTest() {
    guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device.") }
    let library = try! device.makeLibrary(source: Shaders.source, options: nil)
    let n = Simulation.nCohorts, N = 1 << 18
    let sim = Simulation(device: device, library: library, params: Params(),
                         mouse: MouseInput(), particleCount: N, fieldDim: 256)

    func counts() -> [Int] {
        let c = sim.cohortBuffer.contents().bindMemory(to: UInt32.self, capacity: N)
        var k = [Int](repeating: 0, count: n); for i in 0..<N { k[Int(c[i])] += 1 }; return k
    }
    let start = counts()
    print("start counts : \(start)  (Σ=\(start.reduce(0,+)))")

    // dominance → strategy 0 should take over the agent population
    sim.syncEcologyFromAgents(); sim.setEcologyPreset(.dominance); sim.ecologyOn = true
    for _ in 0..<3000 { sim.stepEcology() }
    let dom = counts(); let domSum = dom.reduce(0, +)
    print("dominance    : \(dom)  (Σ=\(domSum))")
    let domOK = domSum == N && dom[0] > Int(0.95 * Double(N))

    // rps → all cohorts stay populated (cycling, nobody eliminated)
    sim.ecologyOn = false
    let fresh = Simulation(device: device, library: library, params: Params(),
                           mouse: MouseInput(), particleCount: N, fieldDim: 256)
    fresh.syncEcologyFromAgents(); fresh.setEcologyPreset(.rps)
    fresh.ecology.p[0] += 0.06; fresh.ecology.p[4] -= 0.06   // perturb off the uniform rest point
    fresh.ecologyOn = true
    var minAcross = N, maxAcross = 0
    for _ in 0..<3000 {
        fresh.stepEcology()
        let c = fresh.cohortBuffer.contents().bindMemory(to: UInt32.self, capacity: N)
        var k = [Int](repeating: 0, count: n); for i in 0..<N { k[Int(c[i])] += 1 }
        minAcross = min(minAcross, k.min()!); maxAcross = max(maxAcross, k.max()!)
    }
    let rpsCounts = (0..<n).map { j -> Int in
        let c = fresh.cohortBuffer.contents().bindMemory(to: UInt32.self, capacity: N)
        var t = 0; for i in 0..<N where Int(c[i]) == j { t += 1 }; return t
    }
    print("rps end      : \(rpsCounts)  (Σ=\(rpsCounts.reduce(0,+)))  min-ever=\(minAcross)  max-ever=\(maxAcross)")
    // cycles in-engine: nobody eliminated (min>0) AND counts actually moved (max grew past uniform)
    let rpsOK = rpsCounts.reduce(0, +) == N && minAcross > 0 && maxAcross > N / n + 1000

    // ── 3D engine: same reallocation glue on Sim3D ──
    let lib3 = try! device.makeLibrary(source: Shaders3D.source, options: nil)
    let N3 = 1 << 18
    let sim3 = Sim3D(device: device, library: lib3, count: N3, fieldDim: 64)
    func counts3() -> [Int] {
        let c = sim3.cohortBuffer.contents().bindMemory(to: UInt32.self, capacity: N3)
        var k = [Int](repeating: 0, count: n); for i in 0..<N3 { k[Int(c[i])] += 1 }; return k
    }
    sim3.syncEcologyFromAgents(); sim3.setEcologyPreset(.dominance); sim3.ecologyOn = true
    for _ in 0..<3000 { sim3.stepEcology() }
    let dom3 = counts3()
    print("3D dominance : \(dom3)  (Σ=\(dom3.reduce(0,+)))")
    let dom3OK = dom3.reduce(0, +) == N3 && dom3[0] > Int(0.95 * Double(N3))

    let ok = domOK && rpsOK && dom3OK
    print(ok ? "RESULT       : PASS — cohort buffer tracks the replicator in-engine (2D + 3D)."
             : "RESULT       : CHECK — in-engine reallocation failed.")
    if !ok { exit(1) }
}
