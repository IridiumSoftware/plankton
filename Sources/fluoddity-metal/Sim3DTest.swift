import Metal
import Foundation

// Headless verification of the 3D fluid (run via `--3dtest`): compiles the 3D
// MSL, runs the agent-forced fluid for some steps at 64³, then CPU-checks the
// projected velocity field — mean|vel| (bounded), mean|div| (≈0, incompressible),
// finiteness — plus particles staying inside the cube. The look needs the window.
func runSim3DTest() {
    guard let device = MTLCreateSystemDefaultDevice() else {
        fatalError("No Metal device available.")
    }
    print("Metal device : \(device.name)")
    let lib: MTLLibrary
    do { lib = try device.makeLibrary(source: Shaders3D.source, options: nil) }
    catch { fatalError("MSL3D compile FAILED: \(error)") }
    print("MSL3D        : compiled OK")

    let dim = 64
    let sim = Sim3D(device: device, library: lib, count: 1 << 16, fieldDim: dim)
    guard let q = device.makeCommandQueue() else { fatalError("no queue") }
    let t0 = Date()
    for _ in 0..<60 {
        guard let cmd = q.makeCommandBuffer() else { fatalError("no cmd") }
        sim.encode(into: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
    }
    let ms = Date().timeIntervalSince(t0) * 1000

    let n3 = dim * dim * dim
    let v = sim.vel.contents().bindMemory(to: Float.self, capacity: 3 * n3)
    func comp(_ x: Int, _ y: Int, _ z: Int, _ c: Int) -> Float {
        let xx = ((x % dim) + dim) % dim, yy = ((y % dim) + dim) % dim, zz = ((z % dim) + dim) % dim
        return v[3 * ((zz * dim + yy) * dim + xx) + c]
    }
    var velMag: Float = 0, divAbs: Float = 0
    var finite = true, samples = 0
    let step = max(1, n3 / 65536)
    var i = 0
    while i < n3 {
        let z = i / (dim * dim), y = (i / dim) % dim, x = i % dim
        let vx = v[3 * i], vy = v[3 * i + 1], vz = v[3 * i + 2]
        if !vx.isFinite || !vy.isFinite || !vz.isFinite { finite = false }
        velMag += (vx * vx + vy * vy + vz * vz).squareRoot()
        let dvg = 0.5 * ((comp(x + 1, y, z, 0) - comp(x - 1, y, z, 0))
                       + (comp(x, y + 1, z, 1) - comp(x, y - 1, z, 1))
                       + (comp(x, y, z + 1, 2) - comp(x, y, z - 1, 2)))
        divAbs += abs(dvg)
        samples += 1
        i += step
    }
    velMag /= Float(samples); divAbs /= Float(samples)

    let p = sim.particleBuffer.contents().bindMemory(to: Particle3D.self, capacity: sim.count)
    var inCube = true
    for j in 0..<sim.count {
        let q = p[j].pos
        if q.x < 0 || q.x >= 1 || q.y < 0 || q.y >= 1 || q.z < 0 || q.z >= 1 { inCube = false }
    }

    print(String(format: "60 steps in %.0f ms  (%.2f ms/step, %d³)", ms, ms / 60, dim))
    print(String(format: "mean |vel|   : %.4f cells/frame", velMag))
    print(String(format: "mean |div|   : %.5f", divAbs))
    print("particles    : in-cube=\(inCube)  finite=\(finite)")
    if finite && inCube && velMag < 50 && divAbs < 0.2 {
        print("RESULT       : PASS — 3D agent-forced fluid is incompressible + bounded.")
    } else {
        print("RESULT       : CHECK — review numbers above.")
        exit(1)
    }
}
