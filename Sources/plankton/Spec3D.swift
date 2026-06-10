import Metal
import Foundation

// Headless 3D energy-spectrum probe (run via `--3dspec`). The scientific point:
// in 3D, turbulence has a forward ENERGY cascade with the Kolmogorov -5/3 spectrum
// (unlike 2D, where -5/3 turned out to be a dial position). So the 3D engine MIGHT
// show a genuine cascade. This sweeps the agent forcing amplitude over a FIXED
// (snapshotted) brain, runs the 128^3 incompressible 3D sim to steady state at
// each, and dumps the raw velocity field (3 floats/cell, interleaved, cell index
// (z*N+y)*N+x) for numpy 3D-FFT analysis (analyze_3dspec.py). If the spectral
// slope is ~ -5/3 AND invariant to forcing, that's a real 3D cascade; if it dials
// like 2D, the agent forcing dominates over the fluid dynamics in 3D too.
func run3dSpec() {
    guard let device = MTLCreateSystemDefaultDevice() else { print("3dspec: no Metal device"); exit(1) }
    let library: MTLLibrary
    do { library = try device.makeLibrary(source: Shaders3D.source, options: nil) }
    catch { print("3dspec: MSL3D compile failed: \(error)"); exit(1) }
    guard let queue = device.makeCommandQueue() else { print("3dspec: no queue"); exit(1) }
    func say(_ s: String) { print(s); fflush(stdout) }

    let N = 128, n3 = N * N * N
    let forceGains: [Float] = [0.25, 0.5, 1.0, 2.0, 4.0]
    let warmup = 250

    // snapshot a shared brain so forceGain is the only thing varying across runs
    let sim0 = Sim3D(device: device, library: library, fieldDim: N)
    let rcount = Sim3D.nCohorts * 80             // all cohort brains (kept identical across runs)
    let rule = Array(UnsafeBufferPointer(
        start: sim0.ruleBuffer.contents().bindMemory(to: Float.self, capacity: rcount), count: rcount))

    say("3dspec: \(forceGains.count) forceGain × \(warmup) warmup steps at \(N)^3 (shared brain)\n")
    var manifest = ["i,forceGain,file"]
    for (i, fg) in forceGains.enumerated() {
        let sim = Sim3D(device: device, library: library, fieldDim: N)
        var r = rule
        memcpy(sim.ruleBuffer.contents(), &r, rcount * MemoryLayout<Float>.stride)
        sim.forceGain = fg
        for _ in 0..<warmup {
            guard let cmd = queue.makeCommandBuffer() else { break }
            sim.encode(into: cmd); cmd.commit(); cmd.waitUntilCompleted()
        }
        let fn = "vel3d_\(i).bin"
        let data = Data(bytes: sim.vel.contents(), count: 3 * n3 * MemoryLayout<Float>.stride)
        try? data.write(to: URL(fileURLWithPath: fn))
        manifest.append("\(i),\(fg),\(fn)")
        // quick mean|vel| sanity (subsampled)
        let v = sim.vel.contents().bindMemory(to: Float.self, capacity: 3 * n3)
        var s: Float = 0; let stp = max(1, n3 / 50000); var c = 0, j = 0
        while j < n3 {
            let a = v[3*j], b = v[3*j+1], cc = v[3*j+2]
            s += (a*a + b*b + cc*cc).squareRoot(); c += 1; j += stp
        }
        say(String(format: "  [%d/%d] forceGain=%4g  mean|vel|=%.4f  wrote %@",
                   i + 1, forceGains.count, fg, s / Float(max(1, c)), fn))
    }
    try? manifest.joined(separator: "\n").write(to: URL(fileURLWithPath: "3dspec_manifest.csv"),
                                                atomically: true, encoding: .utf8)
    say("\nwrote 3dspec_manifest.csv + vel3d_*.bin (\(forceGains.count) snapshots, ~25 MB each)")
}
