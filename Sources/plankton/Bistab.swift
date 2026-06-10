import Metal
import Foundation

// Headless multistability probe (run via `--bistab`). The sensorDist sweep
// (--sdscan §2.6) found a noisy transition zone (0.004–0.012) where peakK
// scattered across adjacent settings. Two explanations: (a) genuine
// multistability — at a FIXED sensorDist the flow has several attractors and the
// initial condition selects which one; or (b) just a steep/sensitive single-valued
// dependence sampled once each. This distinguishes them: at each sensorDist run
// many INDEPENDENT replicates (each a fresh random particle seeding via reset(),
// the SAME baseline brain), and look at the spread of steady-state peakK. A
// multimodal spread at fixed sensorDist = multistability; a tight cluster = noise.
//
// Key guard: a GENEROUS warmup, so each run reaches its attractor — under-
// convergence would manufacture spurious "multistability". Writes bistab_results.csv
// incrementally (crash-safe).
func runBistab() {
    guard let device = MTLCreateSystemDefaultDevice() else { print("bistab: no Metal device"); exit(1) }
    let library: MTLLibrary
    do { library = try device.makeLibrary(source: Shaders.source, options: nil) }
    catch { print("bistab: MSL compile failed: \(error)"); exit(1) }

    let params = Params()
    params.diagnosticsOn = true
    let mouse = MouseInput()
    let sim = Simulation(device: device, library: library, params: params, mouse: mouse)
    guard let queue = device.makeCommandQueue() else { print("bistab: no queue"); exit(1) }
    let dim = Int(sim.dim.x)
    let n = dim * dim
    let spectrum = Spectrum(n: dim)
    let warmup = 240, accum = 55     // generous warmup: reach the attractor, don't fake bistability

    func step() {
        guard let cmd = queue.makeCommandBuffer() else { return }
        sim.encode(into: cmd); cmd.commit(); cmd.waitUntilCompleted()
    }
    func energyEnstrophy() -> (e: Float, z: Float) {
        let v = sim.vel.contents().bindMemory(to: Float.self, capacity: 2 * n)
        let w = sim.vort.contents().bindMemory(to: Float.self, capacity: n)
        var e: Float = 0, z: Float = 0
        for i in 0..<n { e += v[2*i]*v[2*i] + v[2*i+1]*v[2*i+1]; z += w[i]*w[i] }
        return (e / Float(n), z / Float(n))
    }
    func say(_ s: String) { print(s); fflush(stdout) }

    var baseParams: [String: Float] = [:]
    var baseRule: [Float]
    if let p = Presets.load(URL(fileURLWithPath: "presets/preset_003.json")) {
        baseParams = p.params; baseRule = p.rule; say("baseline: preset_003")
    } else {
        for k in engineKnobs { baseParams[k.name] = params[keyPath: k.kp] }
        sim.rerollRule(); baseRule = sim.ruleSnapshot()
        say("baseline: defaults + one rolled brain (preset_003 not found)")
    }
    func applyBaseline() {
        for k in engineKnobs { if let v = baseParams[k.name] { params[keyPath: k.kp] = v } }
        params.diagnosticsOn = true
        sim.loadRule(baseRule)              // SAME forcing structure every replicate
    }

    let sensorDists: [Float] = [0.003, 0.006, 0.009, 0.012]   // 0.003 = sub-threshold control
    let reps = 12

    say("\nbistab: \(sensorDists.count) sensorDist × \(reps) replicates (fresh IC each), "
      + "\(warmup)+\(accum) frames/run\n")
    var rows: [String] = ["sensorDist,rep,peakK,inSlope,inR2,E,Z"]
    let url = Study.url("bistab_results.csv")
    var done = 0
    let total = sensorDists.count * reps

    for sd in sensorDists {
        for rep in 0..<reps {
            applyBaseline(); params.sensorDist = sd
            sim.reset()                      // fresh random particle seeding = independent IC
            spectrum.resetAverage()
            for _ in 0..<warmup { step() }
            for _ in 0..<accum {
                step()
                let v = sim.vel.contents().bindMemory(to: Float.self, capacity: 2 * n)
                spectrum.compute(v, dim: dim)
            }
            let f = SpectrumFit.compute(spectrum.ek)
            let (e, z) = energyEnstrophy()
            done += 1
            rows.append(String(format: "%g,%d,%d,%.3f,%.3f,%.5f,%.5f",
                               sd, rep, f.peakK, f.inSlope, f.inR2, e, z))
            try? rows.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
            say(String(format: "  [%2d/%2d] sensorDist=%6g rep=%2d  ->  peakK=%3d  inSlope=%+.2f (R\u{00B2}%.2f)",
                       done, total, sd, rep, f.peakK, f.inSlope, f.inR2))
        }
    }
    say("\nwrote \(url.path)  (\(rows.count - 1) runs)")
}
