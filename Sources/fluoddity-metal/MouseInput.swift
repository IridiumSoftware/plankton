import Foundation
import simd

// Shared mouse-stir state: written by EngineView (mouse events), read + consumed
// by Simulation each frame. posN/velN are in normalized [0,1] sim coordinates
// (origin bottom-left, matching the render + particle mapping).
final class MouseInput {
    var posN = SIMD2<Float>(0.5, 0.5)
    var velN = SIMD2<Float>(0, 0)   // accumulated drag delta this frame
    var active = false
    // right-click → breed: select the cohort near breedPos and mutate from it
    var breedRequested = false
    var breedPos = SIMD2<Float>(0.5, 0.5)
}

// Mirrors `struct MouseUniform` in Shaders.source (8 floats, 32 bytes).
struct MouseUniformGPU {
    var pos: SIMD2<Float>
    var vel: SIMD2<Float>
    var radius: Float
    var forceGain: Float
    var dyeGain: Float
    var active: Float
}
