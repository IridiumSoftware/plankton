import Metal
import Foundation

// Headless parameter survey (run via `--sweep`). One-axis-at-a-time around a
// baseline: hold every parameter at the baseline (preset_003 if present, else
// the Params defaults + a rolled brain), then walk ONE axis across its range and
// log how the spectrum responds — peak k (injection scale), the full + trimmed
// inertial slopes, and R². Slopes use SpectrumFit — the SAME code as the live
// plot. No window, no convergence-by-eyeball, no hand-transcription.
//
// Reads per setting: E (energy), Z (enstrophy), peakK, highSlope/highR2 (full
// right limb), inLo/inHi/inSlope/inR2 (trimmed inertial range). Writes
// sweep_results.csv and prints a per-axis impact summary. Grid + frame budgets
// are the constants below — ask to widen/refine.
func runSweep() {
    guard let device = MTLCreateSystemDefaultDevice() else { print("sweep: no Metal device"); exit(1) }
    let library: MTLLibrary
    do { library = try device.makeLibrary(source: Shaders.source, options: nil) }
    catch { print("sweep: MSL compile failed: \(error)"); exit(1) }

    let params = Params()
    params.diagnosticsOn = true            // so vorticity (→ enstrophy Z) is computed
    let mouse = MouseInput()               // defaults inactive: no spurious stir/breed
    let sim = Simulation(device: device, library: library, params: params, mouse: mouse)
    guard let queue = device.makeCommandQueue() else { print("sweep: no queue"); exit(1) }
    let dim = Int(sim.dim.x)
    let n = dim * dim
    let spectrum = Spectrum(n: dim)         // full-res, anti-aliased

    let warmup = 220, accum = 70

    func step() {
        guard let cmd = queue.makeCommandBuffer() else { return }
        sim.encode(into: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    func energyEnstrophy() -> (e: Float, z: Float) {
        let v = sim.vel.contents().bindMemory(to: Float.self, capacity: 2 * n)
        let w = sim.vort.contents().bindMemory(to: Float.self, capacity: n)
        var e: Float = 0, z: Float = 0
        for i in 0..<n { e += v[2*i]*v[2*i] + v[2*i+1]*v[2*i+1]; z += w[i]*w[i] }
        return (e / Float(n), z / Float(n))
    }
    func say(_ s: String) { print(s); fflush(stdout) }

    // ── baseline: preset_003 (params + brain) if present, else defaults + a roll ─
    var baseParams: [String: Float] = [:]
    var baseRule: [Float]
    if let p = Presets.load(URL(fileURLWithPath: "presets/preset_003.json")) {
        baseParams = p.params; baseRule = p.rule
        say("baseline: preset_003")
    } else {
        for k in engineKnobs { baseParams[k.name] = params[keyPath: k.kp] }
        sim.rerollRule(); baseRule = sim.ruleSnapshot()
        say("baseline: Params defaults + one rolled brain (preset_003 not found)")
    }
    func applyBaseline() {
        for k in engineKnobs { if let v = baseParams[k.name] { params[keyPath: k.kp] = v } }
        params.diagnosticsOn = true
        sim.loadRule(baseRule)
    }
    // extra brains for the brain axis (index 0 = baseline brain)
    var brains = [baseRule]
    for _ in 0..<2 { sim.rerollRule(); brains.append(sim.ruleSnapshot()) }

    // measure one setting (baseline + the single axis value already applied)
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

    // ── axes: param keypath (nil = brain) + values to walk ──────────────────
    let axes: [(name: String, kp: ReferenceWritableKeyPath<Params, Float>?, vals: [Float])] = [
        ("velDamp",    \.velDamp,    [0.90, 0.95, 0.98, 0.99, 0.999]),
        ("forceGain",  \.forceGain,  [0.1, 0.5, 1.0, 2.5, 5.0]),
        ("sensorDist", \.sensorDist, [0.002, 0.006, 0.012, 0.03, 0.08]),
        ("swim",       \.swim,       [0.0, 0.1, 0.2, 0.35, 0.5]),
        ("cohesion",   \.cohesion,   [0.0, 0.05, 0.15, 0.3, 0.5]),
        ("fluidPull",  \.fluidPull,  [0.0, 1.0, 2.0, 5.0, 10.0]),
        ("senseScale", \.senseScale, [1.0, 5.0, 10.0, 15.0, 20.0]),
        ("speedGain",  \.speedGain,  [0.0, 0.5, 1.0, 1.5, 2.0]),
        ("brain",      nil,          [0, 1, 2]),
    ]

    let header = "axis,value,E,Z,peakK,highSlope,highR2,inLo,inHi,inSlope,inR2"
    var rows: [String] = [header]
    say("\nsweep: one-axis-at-a-time, \(warmup) warmup + \(accum) accum frames/setting\n")

    // baseline reference row
    applyBaseline()
    let b = measure()
    let baseRow = String(format: "baseline,,%.5f,%.4f,%d,%.3f,%.3f,%d,%d,%.3f,%.3f",
                         b.e, b.z, b.f.peakK, b.f.highSlope, b.f.highR2,
                         b.f.inLo, b.f.inHi, b.f.inSlope, b.f.inR2)
    rows.append(baseRow)
    say(String(format: "BASELINE  peakK=%d  high=%.2f (R\u{00B2}%.2f)  inertial=%.2f (R\u{00B2}%.2f)  E=%.4f",
               b.f.peakK, b.f.highSlope, b.f.highR2, b.f.inSlope, b.f.inR2, b.e))

    for axis in axes {
        say("\n== \(axis.name) (others = baseline) ==")
        say("value     E         Z        peakK  high   hiR2   inLo inHi inSlope inR2")
        var pk = [Int](), ins = [Float](), hr = [Float]()
        for v in axis.vals {
            applyBaseline()
            var label: String
            if let kp = axis.kp { params[keyPath: kp] = v; label = String(format: "%-8g", v) }
            else { sim.loadRule(brains[Int(v)]); label = "brain\(Int(v)) " }
            let m = measure()
            pk.append(m.f.peakK); ins.append(m.f.inSlope); hr.append(m.f.highR2)
            say(String(format: "%@  %.5f  %.4f  %4d   %+.2f  %.2f   %4d %4d  %+.2f  %.2f",
                       label, m.e, m.z, m.f.peakK, m.f.highSlope, m.f.highR2,
                       m.f.inLo, m.f.inHi, m.f.inSlope, m.f.inR2))
            rows.append(String(format: "%@,%g,%.5f,%.4f,%d,%.3f,%.3f,%d,%d,%.3f,%.3f",
                               axis.name, v, m.e, m.z, m.f.peakK, m.f.highSlope, m.f.highR2,
                               m.f.inLo, m.f.inHi, m.f.inSlope, m.f.inR2))
        }
        say(String(format: "  impact: peakK %d\u{2192}%d   inSlope %+.2f\u{2192}%+.2f   highR\u{00B2} %.2f\u{2192}%.2f",
                   pk.min() ?? 0, pk.max() ?? 0, ins.min() ?? 0, ins.max() ?? 0,
                   hr.min() ?? 0, hr.max() ?? 0))
    }

    let url = URL(fileURLWithPath: "sweep_results.csv")
    try? rows.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
    say("\nwrote \(url.path)  (\(rows.count - 1) data rows)")
}
