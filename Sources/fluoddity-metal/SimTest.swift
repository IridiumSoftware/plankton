import Metal
import Foundation

// Headless verification of the compute substrate (run via `--simtest`).
//
// This is how a headless agent verifies the engine without a window: it
// compiles the MSL (catching shader errors that `swift build` cannot, since
// shaders compile at runtime), runs the full decay→splat→sense/move pipeline
// for a number of steps, then reads the field back and sanity-checks it
// (non-empty, finite). The *look* still needs a human at the window; this
// proves the pipeline executes correctly.
func runSimTest() {
    guard let device = MTLCreateSystemDefaultDevice() else {
        fatalError("No Metal device available.")
    }
    print("Metal device : \(device.name)")

    let library: MTLLibrary
    do {
        library = try device.makeLibrary(source: Shaders.source, options: nil)
    } catch {
        fatalError("MSL compile FAILED: \(error)")
    }
    print("MSL          : compiled OK")

    // Smaller than the window run, for a fast headless check.
    let sim = Simulation(device: device, library: library,
                         particleCount: 1 << 18, fieldDim: 256)
    guard let queue = device.makeCommandQueue() else { fatalError("no command queue") }

    let steps = 60
    let t0 = Date()
    for _ in 0..<steps {
        guard let cmd = queue.makeCommandBuffer() else { fatalError("no command buffer") }
        sim.encode(into: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        if let err = cmd.error { fatalError("command buffer error: \(err)") }
    }
    let ms = Date().timeIntervalSince(t0) * 1000

    let count = Int(sim.dim.x) * Int(sim.dim.y)
    let ptr = sim.fieldBuffer.contents().bindMemory(to: Float.self, capacity: count)
    var sum: Float = 0, mx: Float = 0
    var finite = true
    for i in 0..<count {
        let v = ptr[i]
        if !v.isFinite { finite = false }
        sum += v
        mx = max(mx, v)
    }
    let mean = sum / Float(count)
    print(String(format: "%d steps in %.0f ms  (%.2f ms/step)", steps, ms, ms / Double(steps)))
    print(String(format: "field        : sum=%.0f  max=%.2f  mean=%.3f  finite=%@",
                 sum, mx, mean, finite ? "yes" : "NO"))
    if sum > 0 && finite {
        print("RESULT       : PASS — compute substrate (decay+splat+sense/move) is live.")
    } else {
        print("RESULT       : FAIL — field empty or non-finite.")
        exit(1)
    }
}
