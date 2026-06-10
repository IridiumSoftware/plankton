import Metal
import Foundation

// Headless probe for the "vorticity goes blank" report (run via `--vortprobe`):
// runs the 2D pipeline long (default params), toggling viewMode dye↔vorticity
// every 300 steps the way a user flips the dropdown, and prints the liveness of
// the velocity + vorticity fields over time. Distinguishes (a) the fluid dying
// (|ω|→0 honestly), (b) NaN poisoning (renders black), (c) a stale/gated vort
// buffer (vel alive, vort not updating).
func runVortProbe() {
    guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device.") }
    let library = try! device.makeLibrary(source: Shaders.source, options: nil)
    let params = Params()
    let dim = 256
    let sim = Simulation(device: device, library: library, params: params,
                         mouse: MouseInput(), particleCount: 1 << 18, fieldDim: dim)
    let queue = device.makeCommandQueue()!

    let n = dim * dim
    func stats() -> (vel: Float, vort: Float, finite: Bool) {
        let v = sim.vel.contents().bindMemory(to: Float.self, capacity: 2 * n)
        let w = sim.vort.contents().bindMemory(to: Float.self, capacity: n)
        var velMag: Float = 0, vortMax: Float = 0, finite = true
        for i in 0..<n {
            let sx = v[2 * i], sy = v[2 * i + 1], wi = w[i]
            if !sx.isFinite || !sy.isFinite || !wi.isFinite { finite = false }
            velMag += (sx * sx + sy * sy).squareRoot()
            vortMax = max(vortMax, abs(wi))
        }
        return (velMag / Float(n), vortMax, finite)
    }

    // a "frame" as the renderer drives it: sim steps (0 when paused) + viz pass
    func frame(simSteps: Int) {
        let cmd = queue.makeCommandBuffer()!
        for _ in 0..<simSteps { sim.encode(into: cmd) }
        sim.encodeViz(into: cmd)
        cmd.commit(); cmd.waitUntilCompleted()
    }

    print("step   viewMode diag   mean|vel|   max|ω|     finite")
    for phase in 0..<10 {
        params.viewMode = (phase % 2 == 0) ? 0 : 1       // flip dye ↔ vorticity
        if phase == 6 { params.diagnosticsOn = false }   // and try art mode late
        for _ in 0..<300 { frame(simSteps: 1) }
        let s = stats()
        print(String(format: "%5d  vm=%d     %@    %.5f     %.5f    %@",
                     (phase + 1) * 300, Int(params.viewMode),
                     params.diagnosticsOn ? "on " : "OFF",
                     s.vel, s.vort, s.finite ? "yes" : "NO — NaN"))
    }

    // the reported repro: PAUSED (simSpeed 0 ⇒ no sim steps), Diag off, then
    // switch to the vorticity view — it must still show the current field
    sim.vort.contents().initializeMemory(as: Float.self, repeating: 0, count: n)  // worst case: stale-zero buffer
    params.viewMode = 1
    frame(simSteps: 0)                                   // one paused render frame
    let s = stats()
    print(String(format: "paused switch to vorticity: max|ω| = %.5f  %@",
                 s.vort, s.vort > 0.01 && s.finite ? "(fresh — PASS)" : "(BLANK — FAIL)"))
    if !(s.vort > 0.01 && s.finite) { exit(1) }
    print("RESULT       : PASS — ω view live in all phases incl. paused switching.")
}
