import Metal
import simd
import Foundation

// Mirrors `struct Particle3D` in Shaders3D.source (two float3 = 32 bytes).
struct Particle3D {
    var pos: SIMD3<Float>
    var vel: SIMD3<Float>
}

// Stage-2a 3D simulation: agents force a 128³ incompressible fluid and ride it.
// Per frame: advect velocity → damp → splat agent velocities → project
// (divergence, 6-neighbour Jacobi, subtract gradient) → agents ride the field.
final class Sim3D {
    let count: Int
    let dim: SIMD3<UInt32>
    let particleBuffer: MTLBuffer
    private(set) var vel: MTLBuffer
    private var velTmp: MTLBuffer
    private var pressure: MTLBuffer
    private var pressureTmp: MTLBuffer
    private let divg: MTLBuffer

    private let scalePipe, advectPipe, splatPipe, divPipe, jacobiPipe,
                subPipe, movePipe: MTLComputePipelineState

    var velDamp: Float = 0.96
    var forceGain: Float = 0.5
    var fluidPull: Float = 4.0
    var jacobiIters: Int = 25

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

        func pipe(_ name: String) -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else { fatalError("\(name) not found") }
            return try! device.makeComputePipelineState(function: fn)
        }
        scalePipe = pipe("scale3d"); advectPipe = pipe("advect3d"); splatPipe = pipe("splat3d")
        divPipe = pipe("divergence3d"); jacobiPipe = pipe("jacobi3d")
        subPipe = pipe("subgrad3d"); movePipe = pipe("move3d")
    }

    func encode(into cmd: MTLCommandBuffer) {
        let fieldDim = Int(dim.x)
        let n3 = fieldDim * fieldDim * fieldDim
        var dimv = dim, velDampv = velDamp, forceGainv = forceGain, fluidPullv = fluidPull

        field(cmd, advectPipe) { e in
            e.setBuffer(self.vel, offset: 0, index: 0)
            e.setBuffer(self.velTmp, offset: 0, index: 1)
            e.setBytes(&dimv, length: 16, index: 2)
        }
        elementwise(cmd, scalePipe, 3 * n3) { e in
            e.setBuffer(self.velTmp, offset: 0, index: 0)
            e.setBytes(&velDampv, length: 4, index: 1)
        }
        particles(cmd, splatPipe) { e in
            e.setBuffer(self.velTmp, offset: 0, index: 0)
            e.setBuffer(self.particleBuffer, offset: 0, index: 1)
            e.setBytes(&dimv, length: 16, index: 2)
            e.setBytes(&forceGainv, length: 4, index: 3)
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
        particles(cmd, movePipe) { e in
            e.setBuffer(self.particleBuffer, offset: 0, index: 0)
            e.setBuffer(self.vel, offset: 0, index: 1)
            e.setBytes(&dimv, length: 16, index: 2)
            e.setBytes(&fluidPullv, length: 4, index: 3)
        }
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
