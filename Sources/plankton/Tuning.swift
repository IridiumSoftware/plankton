import Foundation

// Keyboard live-tuning (secondary to the on-screen sliders). Reads the shared
// engineKnobs list. [ ] select, - = adjust, space prints all. Prints to the
// launching terminal.
final class Tuning {
    private let params: Params
    private var sel = 0

    init(params: Params) {
        self.params = params
        print("keyboard: [ ] select · - = adjust · space = all   (or use the on-screen sliders)")
    }

    func handleKey(_ s: String) {
        switch s {
        case "[": sel = (sel - 1 + engineKnobs.count) % engineKnobs.count; printSelected()
        case "]": sel = (sel + 1) % engineKnobs.count; printSelected()
        case "-", "_": adjust(-1)
        case "=", "+": adjust(+1)
        case " ": printAll()
        default: break
        }
    }

    private func adjust(_ dir: Float) {
        let k = engineKnobs[sel]
        let v = min(k.hi, max(k.lo, params[keyPath: k.kp] + dir * k.step))
        params[keyPath: k.kp] = v
        print(String(format: "  %@ = %.3f", k.name, v))
    }
    private func printSelected() {
        let k = engineKnobs[sel]
        print(String(format: "▶ %@ = %.3f", k.name, params[keyPath: k.kp]))
    }
    private func printAll() {
        print("── params ──")
        for k in engineKnobs { print(String(format: "  %@ = %.3f", k.name, params[keyPath: k.kp])) }
    }
}
