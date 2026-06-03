import AppKit

// A live-tuning knob backed by get/set closures (so it can drive Sim3D's vars
// and the renderer directly, without a shared params class).
struct Knob3D {
    let name: String
    let get: () -> Float
    let set: (Float) -> Void
    let lo: Float
    let hi: Float
}

// On-screen slider panel for the 3D engine: a Brain (re-roll) button + a labeled
// slider with live value per knob. Mirrors the 2D ControlsPanel layout.
final class Panel3D: NSView {
    var onReroll: (() -> Void)?

    private let knobs: [Knob3D]
    private var valueLabels: [NSTextField] = []

    init(knobs: [Knob3D]) {
        self.knobs = knobs
        let rowH: CGFloat = 26, btnH: CGFloat = 30, pad: CGFloat = 8, width: CGFloat = 300
        let height = pad * 2 + btnH + CGFloat(knobs.count) * rowH
        super.init(frame: NSRect(x: 0, y: 0, width: width, height: height))

        wantsLayer = true
        layer?.backgroundColor = NSColor(white: 0, alpha: 0.55).cgColor
        layer?.cornerRadius = 8

        let b = NSButton(title: "Brain", target: self, action: #selector(brain))
        b.bezelStyle = .rounded
        b.font = .systemFont(ofSize: 11)
        b.frame = NSRect(x: 8, y: height - pad - btnH + 2, width: 80, height: 24)
        addSubview(b)

        for (i, k) in knobs.enumerated() {
            let y = height - pad - btnH - CGFloat(i + 1) * rowH + 3
            addSubview(label(k.name, NSRect(x: 8, y: y, width: 96, height: 18), .left))

            let s = NSSlider(frame: NSRect(x: 108, y: y, width: 140, height: 20))
            s.minValue = Double(k.lo)
            s.maxValue = Double(k.hi)
            s.doubleValue = Double(k.get())
            s.isContinuous = true
            s.target = self
            s.action = #selector(sliderChanged(_:))
            s.tag = i
            addSubview(s)

            let v = label(fmt(k.get()), NSRect(x: 250, y: y, width: 42, height: 18), .right)
            valueLabels.append(v)
            addSubview(v)
        }
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    @objc private func brain() { onReroll?() }

    @objc private func sliderChanged(_ s: NSSlider) {
        let k = knobs[s.tag]
        let v = Float(s.doubleValue)
        k.set(v)
        valueLabels[s.tag].stringValue = fmt(v)
    }

    private func fmt(_ v: Float) -> String { String(format: "%.3f", v) }

    private func label(_ s: String, _ frame: NSRect, _ align: NSTextAlignment) -> NSTextField {
        let t = NSTextField(frame: frame)
        t.stringValue = s
        t.isEditable = false; t.isBordered = false; t.drawsBackground = false; t.isSelectable = false
        t.textColor = .white
        t.font = .systemFont(ofSize: 11)
        t.alignment = align
        return t
    }
}
