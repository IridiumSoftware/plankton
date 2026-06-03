import Metal
import Foundation

// Headless verification of the 3D Stage-1 sim (run via `--3dtest`): compiles the
// 3D MSL, advects the particles for some steps, and checks they stay finite and
// inside the unit cube. The look (the orbiting cloud) needs the window.
func runSim3DTest() {
    guard let device = MTLCreateSystemDefaultDevice() else {
        fatalError("No Metal device available.")
    }
    print("Metal device : \(device.name)")
    let lib: MTLLibrary
    do { lib = try device.makeLibrary(source: Shaders3D.source, options: nil) }
    catch { fatalError("MSL3D compile FAILED: \(error)") }
    print("MSL3D        : compiled OK")

    let sim = Sim3D(device: device, library: lib, count: 1 << 16)
    guard let q = device.makeCommandQueue() else { fatalError("no queue") }
    for _ in 0..<60 {
        guard let cmd = q.makeCommandBuffer() else { fatalError("no cmd") }
        sim.encode(into: cmd)
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    let p = sim.particleBuffer.contents().bindMemory(to: Particle3D.self, capacity: sim.count)
    var finite = true, inCube = true
    for i in 0..<sim.count {
        let pos = p[i].pos
        if !pos.x.isFinite || !pos.y.isFinite || !pos.z.isFinite { finite = false }
        if pos.x < 0 || pos.x >= 1 || pos.y < 0 || pos.y >= 1 || pos.z < 0 || pos.z >= 1 { inCube = false }
    }
    print("particles    : finite=\(finite)  in-cube=\(inCube)")
    if finite && inCube {
        print("RESULT       : PASS — 3D ABC-flow advection live + bounded.")
    } else {
        print("RESULT       : CHECK — particles non-finite or escaped the cube.")
        exit(1)
    }
}
