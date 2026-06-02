import Metal
import Foundation

// Headless verification of the fluid step (run via `--simtest`).
//
// Compiles the MSL (catching shader errors `swift build` can't, since shaders
// compile at runtime), runs the full Stable-Fluids pipeline for some steps,
// then checks the projected velocity field on the CPU. The headline claim —
// the field is divergence-free (incompressible) after projection — is verified
// by measuring mean|div| against mean|vel|. The *look* still needs the window.
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

    let dim = 256
    let sim = Simulation(device: device, library: library,
                         particleCount: 1 << 18, fieldDim: dim)
    guard let queue = device.makeCommandQueue() else { fatalError("no command queue") }

    let steps = 120
    let t0 = Date()
    for _ in 0..<steps {
        guard let cmd = queue.makeCommandBuffer() else { fatalError("no command buffer") }
        sim.encode(into: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
        if let err = cmd.error { fatalError("command buffer error: \(err)") }
    }
    let ms = Date().timeIntervalSince(t0) * 1000

    // ── CPU diagnostics on the final (projected) velocity field ──
    let n = dim * dim
    let v = sim.vel.contents().bindMemory(to: Float.self, capacity: 2 * n)
    func vx(_ x: Int, _ y: Int) -> Float {
        let xx = ((x % dim) + dim) % dim, yy = ((y % dim) + dim) % dim
        return v[2 * (yy * dim + xx)]
    }
    func vy(_ x: Int, _ y: Int) -> Float {
        let xx = ((x % dim) + dim) % dim, yy = ((y % dim) + dim) % dim
        return v[2 * (yy * dim + xx) + 1]
    }
    var velMag: Float = 0, divAbs: Float = 0
    var finite = true
    for y in 0..<dim {
        for x in 0..<dim {
            let sx = v[2 * (y * dim + x)], sy = v[2 * (y * dim + x) + 1]
            if !sx.isFinite || !sy.isFinite { finite = false }
            velMag += (sx * sx + sy * sy).squareRoot()
            let d = 0.5 * ((vx(x + 1, y) - vx(x - 1, y)) + (vy(x, y + 1) - vy(x, y - 1)))
            divAbs += abs(d)
        }
    }
    velMag /= Float(n); divAbs /= Float(n)

    let dye = sim.dye.contents().bindMemory(to: Float.self, capacity: n)
    var dyeSum: Float = 0
    for i in 0..<n { let dv = dye[i]; if !dv.isFinite { finite = false }; dyeSum += dv }

    let ratio = velMag > 0 ? 100 * divAbs / velMag : 0
    print(String(format: "%d steps in %.0f ms  (%.2f ms/step)", steps, ms, ms / Double(steps)))
    print(String(format: "mean |vel|   : %.4f cells/frame", velMag))
    print(String(format: "mean |div|   : %.5f   (div/vel = %.1f%%, lower = more incompressible)",
                 divAbs, ratio))
    print(String(format: "dye sum      : %.0f   finite=%@", dyeSum, finite ? "yes" : "NO"))

    let ok = finite && velMag > 0 && dyeSum > 0 && ratio < 10.0
    if ok {
        print("RESULT       : PASS — projected field is near-divergence-free (incompressible).")
    } else {
        print("RESULT       : CHECK — review the numbers above (finite/bounded/divergence).")
        exit(1)
    }
}
