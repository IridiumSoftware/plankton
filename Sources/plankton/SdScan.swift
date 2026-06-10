import Metal
import Foundation

// Headless characterization of the injection-scale axis (run via `--sdscan`).
// A fine log-spaced 1D sweep over `sensorDist` (everything else at the preset_003
// baseline), resolving the non-monotonic peak-k "hump" the 3-axis scan exposed
// (--map3 §2.5) and testing whether it is a resonance between the agents' sensing
// distance and a fluid length scale. At each point it records summary stats —
// peakK, slopes, E, Z, and the Taylor microscale λ=√(E/Z) — AND dumps the full
// time-averaged spectrum E(k), so the spectral *shape* (not just the peak) can be
// followed. Writes sdscan_summary.csv (incrementally, crash-safe) + sdscan_spectra.csv.
func runSdScan() {
    guard let device = MTLCreateSystemDefaultDevice() else { print("sdscan: no Metal device"); exit(1) }
    let library: MTLLibrary
    do { library = try device.makeLibrary(source: Shaders.source, options: nil) }
    catch { print("sdscan: MSL compile failed: \(error)"); exit(1) }

    let params = Params()
    params.diagnosticsOn = true
    let mouse = MouseInput()
    let sim = Simulation(device: device, library: library, params: params, mouse: mouse)
    guard let queue = device.makeCommandQueue() else { print("sdscan: no queue"); exit(1) }
    let dim = Int(sim.dim.x)
    let n = dim * dim
    let spectrum = Spectrum(n: dim)
    let warmup = 220, accum = 80

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
        sim.loadRule(baseRule)
    }

    // log-spaced sensorDist across (most of) its knob range [0.001, 0.10]
    let nSD = 20
    let sdMin = 0.001, sdMax = 0.10
    let sensorDists = (0..<nSD).map { Float(sdMin * pow(sdMax / sdMin, Double($0) / Double(nSD - 1))) }

    say("\nsdscan: \(nSD) log-spaced sensorDist in [\(sdMin), \(sdMax)], "
      + "\(warmup)+\(accum) frames each (others = baseline)\n")
    var summary: [String] = ["sensorDist,E,Z,lambda,krms,peakK,invPeakK,inSlope,inR2,highSlope,highR2"]
    var spectra: [String] = ["sensorDist,k,Ek"]
    let sumURL = URL(fileURLWithPath: "sdscan_summary.csv")

    for (i, sd) in sensorDists.enumerated() {
        applyBaseline(); params.sensorDist = sd
        sim.reset(); spectrum.resetAverage()
        for _ in 0..<warmup { step() }
        for _ in 0..<accum {
            step()
            let v = sim.vel.contents().bindMemory(to: Float.self, capacity: 2 * n)
            spectrum.compute(v, dim: dim)
        }
        let f = SpectrumFit.compute(spectrum.ek)
        let (e, z) = energyEnstrophy()
        let lambda = z > 0 ? (e / z).squareRoot() : 0     // Taylor microscale √(E/Z), in grid cells
        let krms = e > 0 ? (z / e).squareRoot() : 0
        let invPk = f.peakK > 0 ? 1.0 / Float(f.peakK) : 0
        summary.append(String(format: "%g,%.5f,%.5f,%.4f,%.4f,%d,%.4f,%.3f,%.3f,%.3f,%.3f",
                              sd, e, z, lambda, krms, f.peakK, invPk, f.inSlope, f.inR2,
                              f.highSlope, f.highR2))
        let ek = spectrum.ek
        for k in 1..<ek.count where ek[k] > 0 { spectra.append(String(format: "%g,%d,%.6g", sd, k, ek[k])) }
        // crash-safe: rewrite the (short) summary after every point
        try? summary.joined(separator: "\n").write(to: sumURL, atomically: true, encoding: .utf8)
        say(String(format: "  [%2d/%2d] sensorDist=%7g  peakK=%3d  inSlope=%+.2f (R\u{00B2}%.2f)  lambda=%6.2f  E=%.4f",
                   i + 1, nSD, sd, f.peakK, f.inSlope, f.inR2, lambda, e))
    }
    try? spectra.joined(separator: "\n").write(to: URL(fileURLWithPath: "sdscan_spectra.csv"),
                                               atomically: true, encoding: .utf8)
    say("\nwrote sdscan_summary.csv (\(summary.count - 1) rows) + sdscan_spectra.csv")
}
