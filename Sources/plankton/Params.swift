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
    var viscosity: Float   = 0.12    // ν∇² scale-selective viscosity — the FAITHFUL fix vs uniform drag
    var dipoleLen: Float   = 2.5     // net-zero force-dipole separation (cells) — faithful swimmer forcing
    var velDamp: Float     = 0.95    // weak large-scale (Rayleigh) drag / frame; raise → 1 to let ν∇² dominate
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
    var simSpeed: Float    = 1.0     // sim steps per rendered frame (<1 = slow-mo via frame skip, 0 = pause)
    var diagnosticsOn = true         // research HUD + plot + field calcs (toggle off for perf)
}

// Mirrors `struct MoveParams` in Shaders.source (8 floats, 32 bytes).
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

// One tunable knob: name, where it lives in Params, range, keyboard step, group,
// and a plain-language `info` line (tooltip + sidebar footer). Shared source of
// truth for the slider panel, keyboard tuning, AND the capture/path serialization
// (which writes params by ARRAY index — so the array is append-only; `group` is
// display-only bucketing). `options` non-nil ⇒ a dropdown (value = option index).
struct Knob {
    let name: String
    let kp: ReferenceWritableKeyPath<Params, Float>
    let lo: Float
    let hi: Float
    let step: Float
    let group: String
    var info: String = ""
    var options: [String]? = nil
}

// ARRAY ORDER = serialization order (append-only). Groups mirror the 3D panel:
// Agents / Fluid / Dye / Display / Mouse / Research / Time.
let engineKnobs: [Knob] = [
    Knob(name: "swim",        kp: \.swim,        lo: 0.0,   hi: 0.50,  step: 0.01,  group: "Agents",
         info: "Agent self-propulsion speed."),
    Knob(name: "sensorDist",  kp: \.sensorDist,  lo: 0.001, hi: 0.10,  step: 0.002, group: "Agents",
         info: "How far ahead the two sensors reach — sets the eddy scale agents react to."),
    Knob(name: "sensorAngle", kp: \.sensorAngle, lo: 0.0,   hi: 1.50,  step: 0.05,  group: "Agents",
         info: "Angular spread between the left/right sensors (radians)."),
    Knob(name: "turn",        kp: \.turn,        lo: 0.0,   hi: 1.50,  step: 0.05,  group: "Agents",
         info: "Steering gain — how sharply the brain can turn the agent per step."),
    Knob(name: "senseScale",  kp: \.senseScale,  lo: 0.0,   hi: 20.0,  step: 0.5,   group: "Agents",
         info: "Gain on the sensed flow before it enters the brain (higher = more reactive)."),
    Knob(name: "speedGain",   kp: \.speedGain,   lo: 0.0,   hi: 2.0,   step: 0.05,  group: "Agents",
         info: "How much the brain can speed up / slow down the swim speed."),
    Knob(name: "cohesion",    kp: \.cohesion,    lo: 0.0,   hi: 0.50,  step: 0.01,  group: "Agents",
         info: "Chemotaxis toward higher dye = toward other agents. The creature-maker — crank it to grow membranes and cells."),
    Knob(name: "viscosity",   kp: \.viscosity,   lo: 0.0,   hi: 0.25,  step: 0.01,  group: "Fluid",
         info: "Real ν∇² viscosity — dissipates small scales selectively (the physical damping)."),
    Knob(name: "dipoleLen",   kp: \.dipoleLen,   lo: 0.0,   hi: 6.0,   step: 0.5,   group: "Fluid",
         info: "Spacing of each agent's +f/−f force pair, in grid cells (net-zero swimmer forcing)."),
    Knob(name: "forceGain",   kp: \.forceGain,   lo: 0.0,   hi: 5.0,   step: 0.05,  group: "Fluid",
         info: "How hard each agent pushes on the fluid."),
    Knob(name: "fluidPull",   kp: \.fluidPull,   lo: 0.0,   hi: 10.0,  step: 0.25,  group: "Fluid",
         info: "How strongly the fluid carries agents along (their advection)."),
    Knob(name: "velDamp",     kp: \.velDamp,     lo: 0.50,  hi: 0.999, step: 0.005, group: "Fluid",
         info: "Residual large-scale drag per step (1 = none) — the big-eddy energy sink."),
    Knob(name: "dyeDecay",    kp: \.dyeDecay,    lo: 0.50,  hi: 0.999, step: 0.005, group: "Dye",
         info: "Dye persistence per step (closer to 1 = trails linger longer)."),
    Knob(name: "mutationStrength", kp: \.mutationStrength, lo: 0.0, hi: 1.5, step: 0.05, group: "Agents",
         info: "How strongly right-click breeding (and ecology reseeding) mutates brains."),
    Knob(name: "toneK",       kp: \.toneK,       lo: 0.001, hi: 0.20,  step: 0.002, group: "Display",
         info: "Exposure: dye density → brightness."),
    Knob(name: "satGain",     kp: \.satGain,     lo: 0.0,   hi: 3.0,   step: 0.05,  group: "Display",
         info: "Flow speed → color saturation."),
    Knob(name: "palette",     kp: \.palette,     lo: 0.0,   hi: 2.0,   step: 1.0,   group: "Display",
         info: "Color scheme for the dye art (0 direction-hue, 1 thermal, 2 teal)."),
    Knob(name: "bloomStrength", kp: \.bloomStrength, lo: 0.0, hi: 3.0, step: 0.05,  group: "Display",
         info: "Soft glow bled off bright dye."),
    Knob(name: "pointAlpha",  kp: \.pointAlpha,  lo: 0.0,   hi: 1.0,   step: 0.02,  group: "Display",
         info: "Agent dot brightness, tinted by cohort (0 hides the agents)."),
    Knob(name: "pointSize",   kp: \.pointSize,   lo: 0.0,   hi: 6.0,   step: 0.5,   group: "Display",
         info: "Agent dot size."),
    Knob(name: "mouseForce",  kp: \.mouseForce,  lo: 0.0,   hi: 1.0,   step: 0.02,  group: "Mouse",
         info: "Stir strength of a mouse drag."),
    Knob(name: "mouseDye",    kp: \.mouseDye,    lo: 0.0,   hi: 20.0,  step: 0.5,   group: "Mouse",
         info: "Dye injected while dragging."),
    Knob(name: "mouseRadius", kp: \.mouseRadius, lo: 0.01,  hi: 0.20,  step: 0.005, group: "Mouse",
         info: "Brush radius of the mouse stir."),
    Knob(name: "viewMode",    kp: \.viewMode,    lo: 0.0,   hi: 3.0,   step: 1.0,   group: "Research",
         info: "Dye art, or the fluid's vorticity / enstrophy / divergence fields.",
         options: ["dye art", "vorticity ω", "enstrophy ω²", "divergence"]),
    Knob(name: "vortScale",   kp: \.vortScale,   lo: 0.5,   hi: 20.0,  step: 0.5,   group: "Research",
         info: "Display gain for the vorticity / enstrophy / divergence views."),
    // (MUST stay last: captures + path journals serialize params by knob index,
    // so new knobs are append-only to keep old .fluo files valid)
    Knob(name: "simSpeed",    kp: \.simSpeed,    lo: 0.0,   hi: 4.0,   step: 0.05,  group: "Time",
         info: "Sim steps per rendered frame: 0 pauses, <1 slow motion, >1 fast-forward."),
]
