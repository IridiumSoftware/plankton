import Metal
import simd
import Foundation

// Mirrors `struct Particle3D` in Shaders3D.source (two float3 = 32 bytes).
struct Particle3D {
    var pos: SIMD3<Float>
    var vel: SIMD3<Float>
}

// Mirrors `struct MoveParams3` in Shaders3D.source (9 floats, 36 bytes).
struct MoveParams3GPU {
    var swim, sensorDist, sensorAngle, turn, axialForce, fluidPull, senseScale, planeSamples, cohesion: Float
}

// Stage-2a 3D simulation: agents force a 128³ incompressible fluid and ride it.
// Per frame: advect velocity → damp → splat agent velocities → project
// (divergence, 6-neighbour Jacobi, subtract gradient) → agents ride the field.
final class Sim3D {
    let count: Int
    let dim: SIMD3<UInt32>
    let particleBuffer: MTLBuffer
    let ruleBuffer: MTLBuffer          // 20 float4: single global brain
    private(set) var vel: MTLBuffer
    private var velTmp: MTLBuffer
    private var pressure: MTLBuffer
    private var pressureTmp: MTLBuffer
    private let divg: MTLBuffer
    private(set) var dye: MTLBuffer            // 3D density volume (rendered)
    private var dyeTmp: MTLBuffer

    private let scalePipe, advectPipe, diffusePipe, splatPipe, divPipe, jacobiPipe,
                subPipe, movePipe, advectDyePipe: MTLComputePipelineState

    var velDamp: Float = 0.97        // weak large-scale drag; viscosity does the cascade
    var viscosity: Float = 0.10      // ν∇² scale-selective viscosity — the FAITHFUL fix vs uniform drag
    var dipoleLen: Float = 2.5       // net-zero force-dipole separation (cells)
    var forceGain: Float = 0.5
    var fluidPull: Float = 2.0
    var jacobiIters: Int = 25
    // brain (Stage 2b)
    var swim: Float = 0.10
    var sensorDist: Float = 0.015
    var sensorAngle: Float = 0.5
    var turn: Float = 0.05
    var axialForce: Float = 0.02
    var senseScale: Float = 3.0
    var planeSamples: Float = 2
    var cohesion: Float = 0.15       // chemotaxis up the dye gradient → aggregation (the creature-maker)
    var dyeDecay: Float = 0.985
    var dyeAmount: Float = 1.0
    private var frame: UInt32 = 0

    init(device: MTLDevice, library: MTLLibrary, count: Int = 1 << 20, fieldDim: Int = 128) {
        self.count = count
        dim = SIMD3<UInt32>(UInt32(fieldDim), UInt32(fieldDim), UInt32(fieldDim))
        let n3 = fieldDim * fieldDim * fieldDim

        func zeroed(_ floats: Int) -> MTLBuffer {
            let b = device.makeBuffer(length: MemoryLayout<Float>.stride * floats,
                                      options: .storageModeShared)!
            memset(b.contents(), 0, b.length)
            return b
        }
        vel = zeroed(3 * n3); velTmp = zeroed(3 * n3)
        pressure = zeroed(n3); pressureTmp = zeroed(n3); divg = zeroed(n3)
        dye = zeroed(n3); dyeTmp = zeroed(n3)

        // particles: random position; velocity = ABC flow (persistent forcing)
        func abc(_ p: SIMD3<Float>) -> SIMD3<Float> {
            let x = p.x * 2 * .pi, y = p.y * 2 * .pi, z = p.z * 2 * .pi
            let A: Float = 1.0, B: Float = 0.7, C: Float = 0.43
            return SIMD3(A * sin(z) + C * cos(y), B * sin(x) + A * cos(z), C * sin(y) + B * cos(x))
        }
        var rng = SystemRandomNumberGenerator()
        var parts = [Particle3D](); parts.reserveCapacity(count)
        for _ in 0..<count {
            let p = SIMD3(Float.random(in: 0..<1, using: &rng),
                          Float.random(in: 0..<1, using: &rng),
                          Float.random(in: 0..<1, using: &rng))
            parts.append(Particle3D(pos: p, vel: abc(p) * 0.06))
        }
        particleBuffer = device.makeBuffer(bytes: &parts,
                                           length: MemoryLayout<Particle3D>.stride * count,
                                           options: .storageModeShared)!

        var rule = Sim3D.randomRule()
        ruleBuffer = device.makeBuffer(bytes: &rule,
                                       length: rule.count * MemoryLayout<SIMD4<Float>>.stride,
                                       options: .storageModeShared)!

        func pipe(_ name: String) -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else { fatalError("\(name) not found") }
            return try! device.makeComputePipelineState(function: fn)
        }
        scalePipe = pipe("scale3d"); advectPipe = pipe("advect3d"); splatPipe = pipe("splat3d")
        diffusePipe = pipe("diffuse3d")
        divPipe = pipe("divergence3d"); jacobiPipe = pipe("jacobi3d")
        subPipe = pipe("subgrad3d"); movePipe = pipe("move3d")
        advectDyePipe = pipe("advectDye3d")
    }

    func encode(into cmd: MTLCommandBuffer) {
        let fieldDim = Int(dim.x)
        let n3 = fieldDim * fieldDim * fieldDim
        var dimv = dim, velDampv = velDamp, forceGainv = forceGain
        var viscv = min(viscosity, 0.16)   // clamp below the 3D explicit-diffusion stability limit (1/6)

        field(cmd, advectPipe) { e in
            e.setBuffer(self.vel, offset: 0, index: 0)
            e.setBuffer(self.velTmp, offset: 0, index: 1)
            e.setBytes(&dimv, length: 16, index: 2)
        }
        // viscous diffusion ν∇² (the FAITHFUL fix vs uniform drag): velTmp → vel
        // scratch, then swap so velTmp holds the diffused field (vel is restored by
        // the post-projection swap below).
        field(cmd, diffusePipe) { e in
            e.setBuffer(self.velTmp, offset: 0, index: 0)
            e.setBuffer(self.vel, offset: 0, index: 1)
            e.setBytes(&dimv, length: 16, index: 2)
            e.setBytes(&viscv, length: 4, index: 3)
        }
        swap(&vel, &velTmp)
        elementwise(cmd, scalePipe, 3 * n3) { e in
            e.setBuffer(self.velTmp, offset: 0, index: 0)
            e.setBytes(&velDampv, length: 4, index: 1)
        }
        var dyeAmtv = dyeAmount
        var dipoleLenv = dipoleLen
        particles(cmd, splatPipe) { e in
            e.setBuffer(self.velTmp, offset: 0, index: 0)
            e.setBuffer(self.dye, offset: 0, index: 1)
            e.setBuffer(self.particleBuffer, offset: 0, index: 2)
            e.setBytes(&dimv, length: 16, index: 3)
            e.setBytes(&forceGainv, length: 4, index: 4)
            e.setBytes(&dyeAmtv, length: 4, index: 5)
            e.setBytes(&dipoleLenv, length: 4, index: 6)
        }
        field(cmd, divPipe) { e in
            e.setBuffer(self.velTmp, offset: 0, index: 0)
            e.setBuffer(self.divg, offset: 0, index: 1)
            e.setBytes(&dimv, length: 16, index: 2)
        }
        for _ in 0..<jacobiIters {
            field(cmd, jacobiPipe) { e in
                e.setBuffer(self.pressure, offset: 0, index: 0)
                e.setBuffer(self.pressureTmp, offset: 0, index: 1)
                e.setBuffer(self.divg, offset: 0, index: 2)
                e.setBytes(&dimv, length: 16, index: 3)
            }
            swap(&pressure, &pressureTmp)
        }
        field(cmd, subPipe) { e in
            e.setBuffer(self.velTmp, offset: 0, index: 0)
            e.setBuffer(self.pressure, offset: 0, index: 1)
            e.setBytes(&dimv, length: 16, index: 2)
        }
        swap(&vel, &velTmp)
        var dyeDecayv = dyeDecay
        field(cmd, advectDyePipe) { e in
            e.setBuffer(self.dye, offset: 0, index: 0)
            e.setBuffer(self.dyeTmp, offset: 0, index: 1)
            e.setBuffer(self.vel, offset: 0, index: 2)
            e.setBytes(&dimv, length: 16, index: 3)
            e.setBytes(&dyeDecayv, length: 4, index: 4)
        }
        swap(&dye, &dyeTmp)
        frame &+= 1
        var framev = frame
        var mp = MoveParams3GPU(swim: swim, sensorDist: sensorDist, sensorAngle: sensorAngle,
                                turn: turn, axialForce: axialForce, fluidPull: fluidPull,
                                senseScale: senseScale, planeSamples: planeSamples, cohesion: cohesion)
        particles(cmd, movePipe) { e in
            e.setBuffer(self.particleBuffer, offset: 0, index: 0)
            e.setBuffer(self.vel, offset: 0, index: 1)
            e.setBytes(&dimv, length: 16, index: 2)
            e.setBytes(&mp, length: MemoryLayout<MoveParams3GPU>.stride, index: 3)
            e.setBuffer(self.ruleBuffer, offset: 0, index: 4)
            e.setBytes(&framev, length: 4, index: 5)
            e.setBuffer(self.dye, offset: 0, index: 6)
        }
    }

    static func randomRule() -> [SIMD4<Float>] {
        var rng = SystemRandomNumberGenerator()
        func r4(_ lo: Float, _ hi: Float) -> SIMD4<Float> {
            SIMD4(Float.random(in: lo...hi, using: &rng), Float.random(in: lo...hi, using: &rng),
                  Float.random(in: lo...hi, using: &rng), Float.random(in: lo...hi, using: &rng))
        }
        var r = [SIMD4<Float>]()
        for _ in 0..<10 { r.append(r4(-2, 2)) }   // frequencies
        for _ in 0..<10 { r.append(r4(-1, 1)) }   // amplitudes
        return r
    }

    func rerollRule() {
        var r = Sim3D.randomRule()
        memcpy(ruleBuffer.contents(), &r, r.count * MemoryLayout<SIMD4<Float>>.stride)
        print("3D brain re-rolled")
    }

    // ── full-state capture: vel + dye + particles + brain + the param values (extra) ──
    func serializeState(_ extra: [Float]) -> Data {
        var data = Data()
        func putI(_ v: Int32) { var x = v; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        putI(0x464C5533); putI(1)                                  // magic "FLU3", version
        putI(Int32(dim.x)); putI(Int32(count))
        for b in [vel, dye, particleBuffer, ruleBuffer] { data.append(Data(bytes: b.contents(), count: b.length)) }
        putI(Int32(extra.count))
        for v in extra { var vv = v; withUnsafeBytes(of: &vv) { data.append(contentsOf: $0) } }
        return data
    }

    @discardableResult
    func applyState(_ data: Data) -> [Float]? {
        var out: [Float]? = nil
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            guard raw.count >= 16 else { return }
            var off = 0
            func i32() -> Int32 { let v = raw.loadUnaligned(fromByteOffset: off, as: Int32.self); off += 4; return v }
            guard i32() == 0x464C5533 else { print("3D capture: bad magic"); return }
            _ = i32()
            let fd = Int(i32()), c = Int(i32())
            guard fd == Int(dim.x), c == count else { print("3D capture: dims mismatch (\(fd)/\(c) vs \(Int(dim.x))/\(count))"); return }
            for b in [vel, dye, particleBuffer, ruleBuffer] {
                memcpy(b.contents(), raw.baseAddress!.advanced(by: off), b.length); off += b.length
            }
            let np = Int(i32()); var p = [Float]()
            for _ in 0..<np { p.append(raw.loadUnaligned(fromByteOffset: off, as: Float.self)); off += 4 }
            out = p
        }
        return out
    }

    private func field(_ cmd: MTLCommandBuffer, _ p: MTLComputePipelineState,
                       _ setup: (MTLComputeCommandEncoder) -> Void) {
        guard let e = cmd.makeComputeCommandEncoder() else { return }
        e.setComputePipelineState(p); setup(e)
        e.dispatchThreads(MTLSize(width: Int(dim.x), height: Int(dim.y), depth: Int(dim.z)),
                          threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 4))
        e.endEncoding()
    }
    private func elementwise(_ cmd: MTLCommandBuffer, _ p: MTLComputePipelineState,
                             _ count: Int, _ setup: (MTLComputeCommandEncoder) -> Void) {
        guard let e = cmd.makeComputeCommandEncoder() else { return }
        e.setComputePipelineState(p); setup(e)
        let w = p.maxTotalThreadsPerThreadgroup
        e.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        e.endEncoding()
    }
    private func particles(_ cmd: MTLCommandBuffer, _ p: MTLComputePipelineState,
                           _ setup: (MTLComputeCommandEncoder) -> Void) {
        elementwise(cmd, p, count, setup)
    }
}
