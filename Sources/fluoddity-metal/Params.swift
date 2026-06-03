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
    var satGain: Float     = 0.7     // flow speed → color saturation
}

// Mirrors `struct MoveParams` in Shaders.source (5 floats, 20 bytes).
struct MoveParamsGPU {
    var swim: Float
    var sensorDist: Float
    var sensorAngle: Float
    var turn: Float
    var fluidPull: Float
}

// One tunable knob: name, where it lives in Params, range, keyboard step.
// Shared source of truth for both the slider panel and keyboard tuning.
struct Knob {
    let name: String
    let kp: ReferenceWritableKeyPath<Params, Float>
    let lo: Float
    let hi: Float
    let step: Float
}

let engineKnobs: [Knob] = [
    Knob(name: "toneK",       kp: \.toneK,       lo: 0.001, hi: 0.20,  step: 0.002),
    Knob(name: "swim",        kp: \.swim,        lo: 0.0,   hi: 0.50,  step: 0.01),
    Knob(name: "sensorDist",  kp: \.sensorDist,  lo: 0.001, hi: 0.10,  step: 0.002),
    Knob(name: "sensorAngle", kp: \.sensorAngle, lo: 0.0,   hi: 1.50,  step: 0.05),
    Knob(name: "turn",        kp: \.turn,        lo: 0.0,   hi: 1.50,  step: 0.05),
    Knob(name: "fluidPull",   kp: \.fluidPull,   lo: 0.0,   hi: 10.0,  step: 0.25),
    Knob(name: "velDamp",     kp: \.velDamp,     lo: 0.50,  hi: 0.999, step: 0.005),
    Knob(name: "dyeDecay",    kp: \.dyeDecay,    lo: 0.50,  hi: 0.999, step: 0.005),
    Knob(name: "forceGain",   kp: \.forceGain,   lo: 0.0,   hi: 5.0,   step: 0.05),
    Knob(name: "pointAlpha",  kp: \.pointAlpha,  lo: 0.0,   hi: 1.0,   step: 0.02),
    Knob(name: "satGain",     kp: \.satGain,     lo: 0.0,   hi: 3.0,   step: 0.05),
]
