import Metal
import Foundation

// Morphology atlas (run via `--morphology`): sweep cohesion × dyeDecay and classify
// the resulting dye structure (foam / network / spots / dispersed) via Euler χ +
// characteristic + contrast, mapping the engine's attractor landscape. cohesion is
// the aggregation strength (the foam-maker); dyeDecay is how long walls persist.
// The brain is held fixed, so this is the chemotaxis-driven morphology plane —
// the low-cohesion corner is where the brain re-asserts and structure depends on
// the genome (a caveat, not captured by one fixed brain). Writes data/morphology.csv
// + an ASCII class map + per-cell dye thumbnails (data/morphology_thumbs.bin).
func runMorphology() {
    guard let device = MTLCreateSystemDefaultDevice() else { print("morph: no Metal device"); exit(1) }
    let library: MTLLibrary
    do { library = try device.makeLibrary(source: Shaders.source, options: nil) }
    catch { print("morph: MSL compile failed: \(error)"); exit(1) }

    // NS_DIM / NS_PRESET / NS_FRAMES tunable; default to the FOAM baseline (preset_005,
    // cohesion-cranked) at 1 agent/cell — preset_003 is the spectrum-study config and
    // produces NO structure (washed-out dye), so it's the wrong neighbourhood to map.
    func envI(_ k: String, _ d: Int) -> Int { (ProcessInfo.processInfo.environment[k]).flatMap { Int($0) } ?? d }
    let dim = envI("NS_DIM", 384)
    let presetName = ProcessInfo.processInfo.environment["NS_PRESET"] ?? "005"
    let warmup = envI("NS_FRAMES", 600)
    let params = Params()
    let mouse = MouseInput()
    let sim = Simulation(device: device, library: library, params: params, mouse: mouse,
                         particleCount: dim * dim, fieldDim: dim)   // 1 agent/cell, like the live app
    guard let queue = device.makeCommandQueue() else { print("morph: no queue"); exit(1) }
    let n = dim * dim
    func say(_ s: String) { print(s); fflush(stdout) }
    func step() { if let cmd = queue.makeCommandBuffer() { sim.encode(into: cmd); cmd.commit(); cmd.waitUntilCompleted() } }

    // baseline params + brain from the foam preset
    var baseParams: [String: Float] = [:]
    var baseRule: [Float]
    if let p = Presets.load(URL(fileURLWithPath: "presets/preset_\(presetName).json")) {
        baseParams = p.params; baseRule = p.rule; say("baseline: preset_\(presetName) (foam config)")
    } else {
        for k in engineKnobs { baseParams[k.name] = params[keyPath: k.kp] }
        sim.rerollRule(); baseRule = sim.ruleSnapshot()
        say("baseline: defaults + one rolled brain (preset_\(presetName) not found)")
    }
    func applyBaseline() {
        for k in engineKnobs { if let v = baseParams[k.name] { params[keyPath: k.kp] = v } }
        sim.loadRule(baseRule)
    }
    func dyeField() -> [Float] {
        let p = sim.dye.contents().bindMemory(to: Float.self, capacity: n)
        return Array(UnsafeBufferPointer(start: p, count: n))
    }

    // sweep the two morphology-determining knobs across the regimes:
    // cohesion (aggregation strength) × dyeDecay (wall persistence).
    let cohesions: [Float] = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    let dyeDecays: [Float] = [0.50, 0.70, 0.85, 0.95, 0.99]
    say("\nmorphology: cohesion × dyeDecay, \(cohesions.count)×\(dyeDecays.count) cells, "
      + "\(warmup) frames/cell at \(dim)²\n")

    var rows = ["cohesion,dyeDecay,class,contrast,coverage,euler,eulerDensity,lcc,lengthScale,mean"]
    let letter: [String: String] = ["foam": "F", "network": "N", "spots": "s", "dispersed": "·"]
    var grid = [[String]](repeating: [String](repeating: " ", count: cohesions.count), count: dyeDecays.count)
    // per-cell dye thumbnails (downsampled + per-cell normalised) → a visual montage
    let thumbDim = 96
    var thumbs = [UInt8]()   // row-major over the sweep grid; thumbDim² bytes per cell

    for (ci, coh) in cohesions.enumerated() {
        for (di, dd) in dyeDecays.enumerated() {
            applyBaseline(); params.cohesion = coh; params.dyeDecay = dd
            sim.reset()
            for _ in 0..<warmup { step() }
            let field = dyeField()
            let m = Morphology.analyze(field, dim: dim)
            thumbs.append(contentsOf: Morphology.thumbnail(field, dim: dim, out: thumbDim))
            grid[di][ci] = letter[m.klass] ?? "?"
            rows.append(String(format: "%g,%g,%@,%.3f,%.3f,%d,%.2f,%.3f,%.4f,%.4f",
                               coh, dd, m.klass, m.contrast, m.coverage, m.euler,
                               m.eulerDensity, m.lcc, m.lengthScale, m.mean))
            say(String(format: "  cohesion=%4g dyeDecay=%5g  ->  %-9@  (LCC=%.2f  χ/10⁴=%+6.1f  CV=%.2f  L=%.3f)",
                       coh, dd, m.klass, m.lcc, m.eulerDensity, m.contrast, m.lengthScale))
        }
    }

    // ── ASCII class map (F foam · N network · s spots · · dispersed) ──
    say("\nMORPHOLOGY  (rows dyeDecay ↑, cols cohesion →)   F=foam(closed cells) N=network(open) s=spots(discrete) ·=dispersed")
    var head = "dyeD\\coh"
    for coh in cohesions { head += String(format: "  %4g", coh) }
    say(head)
    for (di, dd) in dyeDecays.enumerated().reversed() {
        var line = String(format: "%7g ", dd)
        for ci in 0..<cohesions.count { line += "     " + grid[di][ci] }
        say(line)
    }

    let url = Study.url("morphology.csv")
    try? rows.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
    say("\nwrote \(url.path)  (\(rows.count - 1) cells)")

    // thumbnails: header [cols, rows, thumbDim] (Int32) then the uint8 thumbs,
    // ordered cohesion-major (matches the CSV row order). Python tiles them.
    var tdata = Data()
    for v in [Int32(cohesions.count), Int32(dyeDecays.count), Int32(thumbDim)] {
        var x = v; withUnsafeBytes(of: &x) { tdata.append(contentsOf: $0) }
    }
    tdata.append(contentsOf: thumbs)
    let turl = Study.url("morphology_thumbs.bin")
    try? tdata.write(to: turl)
    say("wrote \(turl.path)  (\(cohesions.count)×\(dyeDecays.count) thumbnails, \(thumbDim)²)")
}

// Diagnostic (run via `--morphdump`): run ONE config (env NS_COH / NS_DD / NS_DIM /
// NS_FRAMES) and dump the raw dye field to /tmp/dye_dump.f32 so we can
// actually SEE what the sweep is measuring instead of guessing.
func runMorphDump() {
    guard let device = MTLCreateSystemDefaultDevice() else { fatalError("no Metal") }
    let library = try! device.makeLibrary(source: Shaders.source, options: nil)
    func env(_ k: String, _ d: Float) -> Float { (ProcessInfo.processInfo.environment[k]).flatMap { Float($0) } ?? d }
    let dim = Int(env("NS_DIM", 256))
    let frames = Int(env("NS_FRAMES", 500))
    let agents = Int(env("NS_AGENTS", Float(dim * dim)))   // default 1 agent/cell (the live-app density)
    let params = Params()
    let sim = Simulation(device: device, library: library, params: params, mouse: MouseInput(),
                         particleCount: agents, fieldDim: dim)
    let queue = device.makeCommandQueue()!
    let presetName = ProcessInfo.processInfo.environment["NS_PRESET"] ?? "003"
    if let p = Presets.load(URL(fileURLWithPath: "presets/preset_\(presetName).json")) {
        for k in engineKnobs { if let v = p.params[k.name] { params[keyPath: k.kp] = v } }
        sim.loadRule(p.rule)
        print("loaded preset_\(presetName)")
    }
    let ev = ProcessInfo.processInfo.environment
    if ev["NS_COH"] != nil { params.cohesion = env("NS_COH", params.cohesion) }   // else keep preset's
    if ev["NS_DD"] != nil { params.dyeDecay = env("NS_DD", params.dyeDecay) }
    sim.reset()
    for _ in 0..<frames { if let cmd = queue.makeCommandBuffer() { sim.encode(into: cmd); cmd.commit(); cmd.waitUntilCompleted() } }
    let n = dim * dim
    let p = sim.dye.contents().bindMemory(to: Float.self, capacity: n)
    let arr = Array(UnsafeBufferPointer(start: p, count: n))
    let m = Morphology.analyze(arr, dim: dim)
    var lo = Float.greatestFiniteMagnitude, hi = -Float.greatestFiniteMagnitude
    for v in arr { lo = min(lo, v); hi = max(hi, v) }
    print(String(format: "coh=%g dyeDecay=%g dim=%d agents=%d (%.2f/cell) frames=%d", params.cohesion, params.dyeDecay, dim, agents, Double(agents)/Double(n), frames))
    print(String(format: "dye range [%.3f, %.3f]  mean=%.3f  CV=%.3f  class=%@  χ=%d  L=%.3f", lo, hi, m.mean, m.contrast, m.klass, m.euler, m.lengthScale))
    var data = Data(); var d32 = Int32(dim); withUnsafeBytes(of: &d32) { data.append(contentsOf: $0) }
    arr.withUnsafeBytes { data.append(contentsOf: $0) }
    try? data.write(to: URL(fileURLWithPath: "/tmp/dye_dump.f32"))
    print("wrote /tmp/dye_dump.f32 (\(dim)² float32, 4-byte dim header)")
}
