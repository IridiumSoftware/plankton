import Foundation

// Live-tunable engine parameters: mutated by keyboard (see Tuning), read each
// frame by Simulation + Renderer and pushed to the shaders as uniforms.
// Defaults are the values dialed in so far.
final class Params {
    var toneK: Float       = 0.020   // dye → brightness
    var swim: Float        = 0.10    // agent self-propulsion
    var sensorDist: Float  = 0.012   // how far ahead agents sense
    var sensorAngle: Float = 0.40    // left/right sensor offset (rad)
    var turn: Float        = 0.35    // brain turn gain (rad)
    var fluidPull: Float   = 2.0     // how strongly the fluid carries agents
    var senseScale: Float  = 5.0     // scales sensed flow into the brain input
    var speedGain: Float   = 0.5     // brain axial output → speed modulation
    var cohesion: Float    = 0.05    // chemotaxis up the dye gradient → aggregation
    var velDamp: Float     = 0.95    // fluid energy bleed / frame
    var dyeDecay: Float    = 0.985   // dye fade / frame
    var forceGain: Float   = 0.5     // agent velocity → fluid forcing
    var pointAlpha: Float  = 0.10    // agent dot brightness
    var satGain: Float     = 0.7     // flow speed → color saturation
    var mouseForce: Float    = 0.15  // mouse-drag stir strength
    var mouseDye: Float      = 6.0   // mouse dye injection
    var mouseRadius: Float   = 0.045 // mouse brush radius
    var bloomStrength: Float = 0.6   // bloom glow intensity
    var palette: Float       = 0.0   // 0 dir-hue, 1 thermal, 2 teal
    var pointSize: Float     = 1.5   // agent dot base size
    var mutationStrength: Float = 0.3 // right-click breed: mutation amount
    var viewMode: Float    = 0.0     // 0 dye art, 1 vorticity, 2 enstrophy, 3 divergence
    var vortScale: Float   = 3.0     // vorticity/enstrophy display scale
    var diagnosticsOn = true         // research HUD + plot + field calcs (toggle off for perf)
}

// Mirrors `struct MoveParams` in Shaders.source (5 floats, 20 bytes).
struct MoveParamsGPU {
    var swim: Float
    var sensorDist: Float
    var sensorAngle: Float
    var turn: Float
    var fluidPull: Float
    var senseScale: Float
    var speedGain: Float
    var cohesion: Float
}

// One tunable knob: name, where it lives in Params, range, keyboard step, group.
// Shared source of truth for both the slider panel and keyboard tuning.
struct Knob {
    let name: String
    let kp: ReferenceWritableKeyPath<Params, Float>
    let lo: Float
    let hi: Float
    let step: Float
    let group: String
}

// Ordered by group; the panel draws a header whenever the group changes.
let engineKnobs: [Knob] = [
    // ── Particles (simulation behavior) ──
    Knob(name: "swim",        kp: \.swim,        lo: 0.0,   hi: 0.50,  step: 0.01,  group: "Particles"),
    Knob(name: "sensorDist",  kp: \.sensorDist,  lo: 0.001, hi: 0.10,  step: 0.002, group: "Particles"),
    Knob(name: "sensorAngle", kp: \.sensorAngle, lo: 0.0,   hi: 1.50,  step: 0.05,  group: "Particles"),
    Knob(name: "turn",        kp: \.turn,        lo: 0.0,   hi: 1.50,  step: 0.05,  group: "Particles"),
    Knob(name: "senseScale",  kp: \.senseScale,  lo: 0.0,   hi: 20.0,  step: 0.5,   group: "Particles"),
    Knob(name: "speedGain",   kp: \.speedGain,   lo: 0.0,   hi: 2.0,   step: 0.05,  group: "Particles"),
    Knob(name: "cohesion",    kp: \.cohesion,    lo: 0.0,   hi: 0.50,  step: 0.01,  group: "Particles"),
    Knob(name: "forceGain",   kp: \.forceGain,   lo: 0.0,   hi: 5.0,   step: 0.05,  group: "Particles"),
    Knob(name: "fluidPull",   kp: \.fluidPull,   lo: 0.0,   hi: 10.0,  step: 0.25,  group: "Particles"),
    Knob(name: "velDamp",     kp: \.velDamp,     lo: 0.50,  hi: 0.999, step: 0.005, group: "Particles"),
    Knob(name: "dyeDecay",    kp: \.dyeDecay,    lo: 0.50,  hi: 0.999, step: 0.005, group: "Particles"),
    Knob(name: "mutationStrength", kp: \.mutationStrength, lo: 0.0, hi: 1.5, step: 0.05, group: "Particles"),
    // ── VFX (rendering) ──
    Knob(name: "toneK",       kp: \.toneK,       lo: 0.001, hi: 0.20,  step: 0.002, group: "VFX"),
    Knob(name: "satGain",     kp: \.satGain,     lo: 0.0,   hi: 3.0,   step: 0.05,  group: "VFX"),
    Knob(name: "palette",     kp: \.palette,     lo: 0.0,   hi: 2.0,   step: 1.0,   group: "VFX"),
    Knob(name: "bloomStrength", kp: \.bloomStrength, lo: 0.0, hi: 3.0, step: 0.05,  group: "VFX"),
    Knob(name: "pointAlpha",  kp: \.pointAlpha,  lo: 0.0,   hi: 1.0,   step: 0.02,  group: "VFX"),
    Knob(name: "pointSize",   kp: \.pointSize,   lo: 0.0,   hi: 6.0,   step: 0.5,   group: "VFX"),
    // ── Mouse (interaction) ──
    Knob(name: "mouseForce",  kp: \.mouseForce,  lo: 0.0,   hi: 1.0,   step: 0.02,  group: "Mouse"),
    Knob(name: "mouseDye",    kp: \.mouseDye,    lo: 0.0,   hi: 20.0,  step: 0.5,   group: "Mouse"),
    Knob(name: "mouseRadius", kp: \.mouseRadius, lo: 0.01,  hi: 0.20,  step: 0.005, group: "Mouse"),
    // ── Research (diagnostics) ──
    Knob(name: "viewMode",    kp: \.viewMode,    lo: 0.0,   hi: 3.0,   step: 1.0,   group: "Research"),
    Knob(name: "vortScale",   kp: \.vortScale,   lo: 0.5,   hi: 20.0,  step: 0.5,   group: "Research"),
]
