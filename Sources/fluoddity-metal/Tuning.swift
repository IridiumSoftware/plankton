import Foundation

// Keyboard live-tuning. [ ] select a knob, - = adjust it, space prints all.
// Prints to the terminal the app was launched from.
final class Tuning {
    private struct Knob {
        let name: String
        let kp: ReferenceWritableKeyPath<Params, Float>
        let step: Float
        let lo: Float
        let hi: Float
    }

    private let params: Params
    private var sel = 0
    private let knobs: [Knob]

    init(params: Params) {
        self.params = params
        knobs = [
            Knob(name: "toneK",       kp: \.toneK,       step: 0.002, lo: 0.001, hi: 0.20),
            Knob(name: "swim",        kp: \.swim,        step: 0.01,  lo: 0.0,   hi: 0.50),
            Knob(name: "sensorDist",  kp: \.sensorDist,  step: 0.002, lo: 0.001, hi: 0.10),
            Knob(name: "sensorAngle", kp: \.sensorAngle, step: 0.05,  lo: 0.0,   hi: 1.50),
            Knob(name: "turn",        kp: \.turn,        step: 0.05,  lo: 0.0,   hi: 1.50),
            Knob(name: "fluidPull",   kp: \.fluidPull,   step: 0.25,  lo: 0.0,   hi: 10.0),
            Knob(name: "velDamp",     kp: \.velDamp,     step: 0.005, lo: 0.50,  hi: 0.999),
            Knob(name: "dyeDecay",    kp: \.dyeDecay,    step: 0.005, lo: 0.50,  hi: 0.999),
            Knob(name: "forceGain",   kp: \.forceGain,   step: 0.05,  lo: 0.0,   hi: 5.0),
            Knob(name: "pointAlpha",  kp: \.pointAlpha,  step: 0.02,  lo: 0.0,   hi: 1.0),
        ]
        printKeymap()
        printSelected()
    }

    func handleKey(_ s: String) {
        switch s {
        case "[": sel = (sel - 1 + knobs.count) % knobs.count; printSelected()
        case "]": sel = (sel + 1) % knobs.count; printSelected()
        case "-", "_": adjust(-1)
        case "=", "+": adjust(+1)
        case " ": printAll()
        default: break
        }
    }

    private func adjust(_ dir: Float) {
        let k = knobs[sel]
        let v = min(k.hi, max(k.lo, params[keyPath: k.kp] + dir * k.step))
        params[keyPath: k.kp] = v
        print(String(format: "  %@ = %.3f", k.name, v))
    }
    private func printSelected() {
        let k = knobs[sel]
        print(String(format: "▶ %@ = %.3f   ([ ] select · - = adjust · space = all)",
                     k.name, params[keyPath: k.kp]))
    }
    private func printAll() {
        print("── params ──")
        for k in knobs { print(String(format: "  %@ = %.3f", k.name, params[keyPath: k.kp])) }
    }
    private func printKeymap() {
        print("""

        ── fluoddity-metal live tuning ──
          [ / ]   select parameter
          - / =   decrease / increase
          space   print all values

        """)
    }
}
