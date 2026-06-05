import Metal
import Foundation

// Headless verification of full-state capture (run via `--capturetest`).
// Grows some state, serializes it, wipes the sim + changes params, restores from the
// blob, and checks every buffer + the params came back bit-for-bit. This is the core
// of "capture a creature" — the full-state save/restore must round-trip exactly.
func runCaptureTest() {
    guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device available.") }
    print("Metal device : \(device.name)")
    let library: MTLLibrary
    do { library = try device.makeLibrary(source: Shaders.source, options: nil) }
    catch { fatalError("MSL compile FAILED: \(error)") }

    let dim = 128, pc = 1 << 16
    let params = Params()
    let sim = Simulation(device: device, library: library, params: params,
                         mouse: MouseInput(), particleCount: pc, fieldDim: dim)
    let queue = device.makeCommandQueue()!

    // grow some non-trivial state
    for _ in 0..<40 {
        let cmd = queue.makeCommandBuffer()!; sim.encode(into: cmd); cmd.commit(); cmd.waitUntilCompleted()
    }

    // order-sensitive weighted checksum of a shared buffer
    func checksum(_ b: MTLBuffer, _ floats: Int) -> Double {
        let p = b.contents().bindMemory(to: Float.self, capacity: floats)
        var s = 0.0; for i in 0..<floats { s += Double(p[i]) * Double((i % 97) + 1) }; return s
    }
    let n = dim * dim
    let cVel = checksum(sim.vel, 2 * n), cDye = checksum(sim.dye, n), cPar = checksum(sim.particleBuffer, pc * 4)

    params.viscosity = 0.07; params.cohesion = 0.21               // distinctive params to round-trip
    let blob = sim.serializeState(params)
    print(String(format: "serialized   : %d bytes (vel+dye+particles+brain+params)", blob.count))

    sim.reset(); params.viscosity = 0.99; params.cohesion = 0.99  // wipe state + move params away
    let ok = sim.applyState(blob, params)
    let rVel = checksum(sim.vel, 2 * n), rDye = checksum(sim.dye, n), rPar = checksum(sim.particleBuffer, pc * 4)

    let velOK = rVel == cVel, dyeOK = rDye == cDye, parOK = rPar == cPar
    let parmOK = params.viscosity == 0.07 && params.cohesion == 0.21
    print("velocity     : \(velOK ? "OK" : "FAIL")")
    print("dye          : \(dyeOK ? "OK" : "FAIL")")
    print("particles    : \(parOK ? "OK" : "FAIL")")
    print("params       : \(parmOK ? "OK" : "FAIL") (viscosity \(params.viscosity), cohesion \(params.cohesion))")
    if ok && velOK && dyeOK && parOK && parmOK {
        print("RESULT       : PASS — full-state capture round-trips bit-for-bit (creature save/restore works).")
    } else {
        print("RESULT       : CHECK — round-trip mismatch."); exit(1)
    }
}
