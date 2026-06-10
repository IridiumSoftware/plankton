import Metal
import Foundation

// Headless 2D exponent map (run via `--map`): the trimmed inertial slope over a
// forceGain × velDamp grid — the two strongest INDEPENDENT levers from the OAT
// survey (one drive, one dissipation). Everything else is held at the preset_003
// baseline + its brain. Writes map_results.csv and prints an ASCII heatmap of the
// inertial slope, plus a re-run of the collapse stress-test on this denser,
// jointly-sampled grid (does slope = f(drive, drag) hold here?).
func runMap() {
    guard let device = MTLCreateSystemDefaultDevice() else { print("map: no Metal device"); exit(1) }
    let library: MTLLibrary
    do { library = try device.makeLibrary(source: Shaders.source, options: nil) }
    catch { print("map: MSL compile failed: \(error)"); exit(1) }

    let params = Params()
    params.diagnosticsOn = true
    let mouse = MouseInput()
    let sim = Simulation(device: device, library: library, params: params, mouse: mouse)
    guard let queue = device.makeCommandQueue() else { print("map: no queue"); exit(1) }
    let dim = Int(sim.dim.x)
    let n = dim * dim
    let spectrum = Spectrum(n: dim)
    let warmup = 220, accum = 70

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
    func measure() -> (f: SpectrumFit, e: Float, z: Float) {
        sim.reset(); spectrum.resetAverage()
        for _ in 0..<warmup { step() }
        for _ in 0..<accum {
            step()
            let v = sim.vel.contents().bindMemory(to: Float.self, capacity: 2 * n)
            spectrum.compute(v, dim: dim)
        }
        let f = SpectrumFit.compute(spectrum.ek)
        let (e, z) = energyEnstrophy()
        return (f, e, z)
    }

    // ── grid: forceGain (cols) × velDamp (rows) ─────────────────────────────
    let forceGains: [Float] = [0.25, 0.5, 1.0, 2.0, 3.5, 5.0]
    let velDamps:   [Float] = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]

    say("\nmap: forceGain × velDamp, \(forceGains.count)×\(velDamps.count) cells, "
      + "\(warmup)+\(accum) frames/cell\n")
    var rows: [String] = ["forceGain,velDamp,E,Z,peakK,highSlope,highR2,inSlope,inR2"]
    var slope = [[Float]](repeating: [Float](repeating: 0, count: forceGains.count),
                          count: velDamps.count)
    for (vi, vd) in velDamps.enumerated() {
        for (fi, fg) in forceGains.enumerated() {
            applyBaseline(); params.velDamp = vd; params.forceGain = fg
            let m = measure()
            slope[vi][fi] = m.f.inSlope
            rows.append(String(format: "%g,%g,%.5f,%.4f,%d,%.3f,%.3f,%.3f,%.3f",
                               fg, vd, m.e, m.z, m.f.peakK, m.f.highSlope, m.f.highR2,
                               m.f.inSlope, m.f.inR2))
            say(String(format: "  forceGain=%4g velDamp=%5g  ->  inSlope=%+.2f (R\u{00B2}%.2f) peakK=%d",
                       fg, vd, m.f.inSlope, m.f.inR2, m.f.peakK))
        }
    }

    // ── ASCII heatmap of the inertial slope ─────────────────────────────────
    say("\nINERTIAL SLOPE  (rows velDamp ↓, cols forceGain →)")
    var head = "velDamp\\fG"
    for fg in forceGains { head += String(format: "  %5g", fg) }
    say(head)
    for (vi, vd) in velDamps.enumerated().reversed() {
        var line = String(format: "%8g  ", vd)
        for fi in 0..<forceGains.count { line += String(format: "  %+5.2f", slope[vi][fi]) }
        say(line)
    }
    say("\n(-5/3 = -1.67 contour runs diagonally if drive & dissipation trade off)")

    let url = Study.url("map_results.csv")
    try? rows.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
    say("\nwrote \(url.path)  (\(rows.count - 1) cells)")
}
