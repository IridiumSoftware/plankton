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
    private var sliders: [NSSlider] = []

    init(knobs: [Knob3D]) {
        self.knobs = knobs
        let rowH: CGFloat = 26, btnH: CGFloat = 30, pad: CGFloat = 8, width: CGFloat = 300
        let height = pad * 2 + btnH + CGFloat(knobs.count) * rowH
        super.init(frame: NSRect(x: 0, y: 0, width: width, height: height))

        wantsLayer = true
        layer?.backgroundColor = NSColor(white: 0, alpha: 0.55).cgColor
        layer?.cornerRadius = 8

        let b = NSButton(title: "Brain", target: self, action: #selector(brain))
        b.isBordered = false            // bordered NSButtons vanish on the dark panel
        b.wantsLayer = true
        b.layer?.cornerRadius = 5
        b.layer?.borderWidth = 1.2
        b.layer?.backgroundColor = NSColor(calibratedRed: 0.10, green: 0.27, blue: 0.40, alpha: 0.98).cgColor
        b.layer?.borderColor = NSColor(calibratedRed: 0.45, green: 0.82, blue: 1.0, alpha: 0.95).cgColor
        b.attributedTitle = NSAttributedString(string: "Brain", attributes: [
            .foregroundColor: NSColor(calibratedRed: 0.85, green: 0.96, blue: 1.0, alpha: 1.0),
            .font: NSFont.boldSystemFont(ofSize: 11)])
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
            sliders.append(s)
            addSubview(s)

            let v = label(fmt(k.get()), NSRect(x: 250, y: y, width: 42, height: 18), .right)
            valueLabels.append(v)
            addSubview(v)
        }
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    // Re-sync every slider + value label from the knobs (after a creature restore / path replay).
    func refresh() {
        for (i, k) in knobs.enumerated() {
            sliders[i].doubleValue = Double(k.get())
            valueLabels[i].stringValue = fmt(k.get())
        }
    }

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
