import AppKit

// On-screen tuning panel: a labeled NSSlider per knob with a live value readout,
// overlaid (translucent) on the Metal view. Sliders write straight into the
// shared Params, so dragging one updates the running sim immediately.
final class ControlsPanel: NSView {
    private let params: Params
    private let knobs: [Knob]
    private var valueLabels: [NSTextField] = []

    init(params: Params, knobs: [Knob]) {
        self.params = params
        self.knobs = knobs

        let rowH: CGFloat = 26, pad: CGFloat = 8, width: CGFloat = 324
        let height = pad * 2 + CGFloat(knobs.count) * rowH
        super.init(frame: NSRect(x: 0, y: 0, width: width, height: height))

        wantsLayer = true
        layer?.backgroundColor = NSColor(white: 0, alpha: 0.55).cgColor
        layer?.cornerRadius = 8

        for (i, k) in knobs.enumerated() {
            // rows laid out top-to-bottom (AppKit origin is bottom-left)
            let y = height - pad - CGFloat(i + 1) * rowH + 3

            addSubview(label(k.name, NSRect(x: 8, y: y, width: 100, height: 18), .left))

            let slider = NSSlider(frame: NSRect(x: 112, y: y, width: 156, height: 20))
            slider.minValue = Double(k.lo)
            slider.maxValue = Double(k.hi)
            slider.doubleValue = Double(params[keyPath: k.kp])
            slider.isContinuous = true
            slider.target = self
            slider.action = #selector(sliderChanged(_:))
            slider.tag = i
            addSubview(slider)

            let v = label(fmt(params[keyPath: k.kp]), NSRect(x: 272, y: y, width: 46, height: 18), .right)
            valueLabels.append(v)
            addSubview(v)
        }
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    @objc private func sliderChanged(_ s: NSSlider) {
        let k = knobs[s.tag]
        let v = Float(s.doubleValue)
        params[keyPath: k.kp] = v
        valueLabels[s.tag].stringValue = fmt(v)
    }

    private func fmt(_ v: Float) -> String { String(format: "%.3f", v) }

    private func label(_ s: String, _ frame: NSRect, _ align: NSTextAlignment) -> NSTextField {
        let t = NSTextField(frame: frame)
        t.stringValue = s
        t.isEditable = false
        t.isBordered = false
        t.drawsBackground = false
        t.isSelectable = false
        t.textColor = .white
        t.font = .systemFont(ofSize: 11)
        t.alignment = align
        return t
    }
}
