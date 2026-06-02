import Foundation

// Live-tunable engine parameters: mutated by keyboard (see Tuning), read each
// frame by Simulation + Renderer and pushed to the shaders as uniforms.
// Defaults are the values dialed in so far.
final class Params {
    var toneK: Float       = 0.020   // dye → brightness
    var swim: Float        = 0.10    // agent self-propulsion
    var sensorDist: Float  = 0.012   // how far ahead agents sense
    var sensorAngle: Float = 0.40    // left/right sensor offset (rad)
    var turn: Float        = 0.35    // steer per step (rad)
    var fluidPull: Float   = 2.0     // how strongly the fluid carries agents
    var velDamp: Float     = 0.95    // fluid energy bleed / frame
    var dyeDecay: Float    = 0.985   // dye fade / frame
    var forceGain: Float   = 0.5     // agent velocity → fluid forcing
    var pointAlpha: Float  = 0.10    // agent dot brightness
}

// Mirrors `struct MoveParams` in Shaders.source (5 floats, 20 bytes).
struct MoveParamsGPU {
    var swim: Float
    var sensorDist: Float
    var sensorAngle: Float
    var turn: Float
    var fluidPull: Float
}
