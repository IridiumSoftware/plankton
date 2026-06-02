import Metal
import Foundation

// ───────────────────────────────────────────────────────────────────────────
// fluoddity-metal — toolchain smoke test (foundation N=1)
//
// Proves, headless and WITHOUT full Xcode, the four things the build rests on:
//   1. SwiftPM builds a Metal program with Command Line Tools only
//   2. runtime shader compilation  (device.makeLibrary(source:))
//   3. compute dispatch
//   4. atomic_float  — the trail-splat primitive that, on the OpenGL engine,
//      required the NVIDIA-only GL_NV_shader_atomic_float extension. On Apple
//      Silicon it's native Metal. This is THE risky primitive; prove it first.
//
// Each thread atomically adds 1.0 into a single float. N is exact in float32
// (N < 2^24), so a correct run yields sum == N deterministically.
// ───────────────────────────────────────────────────────────────────────────

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("No Metal device available — cannot continue.")
}
print("Metal device : \(device.name)")

let shaderSource = """
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

kernel void atomic_smoke(device atomic_float *sink [[buffer(0)]],
                         uint                 gid  [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(sink, 1.0f, memory_order_relaxed);
}
"""

do {
    let library = try device.makeLibrary(source: shaderSource, options: nil)
    guard let fn = library.makeFunction(name: "atomic_smoke") else {
        fatalError("kernel `atomic_smoke` not found in compiled library")
    }
    let pipeline = try device.makeComputePipelineState(function: fn)
    guard let queue = device.makeCommandQueue() else { fatalError("no command queue") }

    // Single float accumulator, zero-initialized, CPU-visible (UMA shared).
    var seed: Float = 0
    guard let sink = device.makeBuffer(bytes: &seed,
                                       length: MemoryLayout<Float>.stride,
                                       options: .storageModeShared) else {
        fatalError("buffer allocation failed")
    }

    let n = 1_000_000
    guard let cmd = queue.makeCommandBuffer(),
          let enc = cmd.makeComputeCommandEncoder() else {
        fatalError("command encoder allocation failed")
    }
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(sink, offset: 0, index: 0)

    let tgWidth = pipeline.maxTotalThreadsPerThreadgroup
    enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
    enc.endEncoding()

    let t0 = Date()
    cmd.commit()
    cmd.waitUntilCompleted()
    let ms = Date().timeIntervalSince(t0) * 1000

    if let err = cmd.error {
        fatalError("command buffer error: \(err)")
    }

    let sum = sink.contents().load(as: Float.self)
    print(String(format: "atomic_float : %.0f / %d expected   (%.2f ms, %d threads)",
                 sum, n, ms, tgWidth))
    if sum == Float(n) {
        print("RESULT       : PASS — Metal compute + atomic_float are green on this Mac.")
    } else {
        print("RESULT       : FAIL — atomic sum mismatch (got \(sum)).")
        exit(1)
    }
} catch {
    fatalError("Metal smoke test failed: \(error)")
}
