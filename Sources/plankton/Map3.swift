import Metal
import Foundation

// Headless 3-axis exponent scan (run via `--map3`): the trimmed inertial slope
// over a forceGain × velDamp × sensorDist grid. Extends the 2D `--map` (drive ×
// dissipation) with the injection scale to test whether the in-plane two-group
// law (slope ≈ a·log fG + b·log γ) is a clean THREE-group extension, or whether
// the injection scale breaks the low-dimensional structure. Everything else is
// held at the preset_003 baseline + its brain. Writes map3_results.csv and prints
// the inSlope surface as one forceGain×velDamp table per sensorDist slice.
func runMap3() {
    guard let device = MTLCreateSystemDefaultDevice() else { print("map3: no Metal device"); exit(1) }
    let library: MTLLibrary
    do { library = try device.makeLibrary(source: Shaders.source, options: nil) }
    catch { print("map3: MSL compile failed: \(error)"); exit(1) }

    let params = Params()
    params.diagnosticsOn = true
    let mouse = MouseInput()
    let sim = Simulation(device: device, library: library, params: params, mouse: mouse)
    guard let queue = device.makeCommandQueue() else { print("map3: no queue"); exit(1) }
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
        let (e, z) = energyEnstrophy()
        return (SpectrumFit.compute(spectrum.ek), e, z)
    }

    // ── grid: forceGain × velDamp × sensorDist ──────────────────────────────
    let forceGains:  [Float] = [0.5, 1.0, 2.0, 3.5]
    let velDamps:    [Float] = [0.95, 0.98, 0.99, 0.999]
    let sensorDists: [Float] = [0.004, 0.012, 0.03, 0.06]
    let total = forceGains.count * velDamps.count * sensorDists.count

    say("\nmap3: forceGain × velDamp × sensorDist, "
      + "\(forceGains.count)×\(velDamps.count)×\(sensorDists.count) = \(total) cells, "
      + "\(warmup)+\(accum) frames/cell\n")
    var rows: [String] = ["forceGain,velDamp,sensorDist,E,Z,peakK,highSlope,highR2,inSlope,inR2"]
    var done = 0
    for sd in sensorDists {
        var slice = [[Float]](repeating: [Float](repeating: 0, count: forceGains.count),
                              count: velDamps.count)
        for (vi, vd) in velDamps.enumerated() {
            for (fi, fg) in forceGains.enumerated() {
                applyBaseline(); params.forceGain = fg; params.velDamp = vd; params.sensorDist = sd
                let m = measure()
                slice[vi][fi] = m.f.inSlope
                done += 1
                rows.append(String(format: "%g,%g,%g,%.5f,%.4f,%d,%.3f,%.3f,%.3f,%.3f",
                                   fg, vd, sd, m.e, m.z, m.f.peakK, m.f.highSlope, m.f.highR2,
                                   m.f.inSlope, m.f.inR2))
                say(String(format: "  [%2d/%2d] sensorDist=%5g forceGain=%4g velDamp=%5g  ->  inSlope=%+.2f (R\u{00B2}%.2f) peakK=%d",
                           done, total, sd, fg, vd, m.f.inSlope, m.f.inR2, m.f.peakK))
            }
        }
        // per-sensorDist inSlope table (forceGain → cols, velDamp ↓ rows)
        say(String(format: "\n  -- sensorDist = %g --  inertial slope (velDamp ↓, forceGain →)", sd))
        var head = "  velDamp\\fG"
        for fg in forceGains { head += String(format: "  %5g", fg) }
        say(head)
        for (vi, vd) in velDamps.enumerated().reversed() {
            var line = String(format: "  %8g  ", vd)
            for fi in 0..<forceGains.count { line += String(format: "  %+5.2f", slice[vi][fi]) }
            say(line)
        }
        say("")
    }

    let url = Study.url("map3_results.csv")
    try? rows.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
    say("wrote \(url.path)  (\(rows.count - 1) cells)")
}
