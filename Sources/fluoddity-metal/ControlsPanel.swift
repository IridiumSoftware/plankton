import AppKit

// On-screen tuning panel: a button row (Save / Load / Reset / Brain) plus a
// labeled NSSlider + live value per knob, overlaid (translucent) on the Metal
// view. Sliders write straight into the shared Params; buttons fire closures
// the app wires up. refresh() re-syncs the sliders after a preset load/reset.
final class ControlsPanel: NSView {
    var onSave: (() -> Void)?
    var onLoad: (() -> Void)?
    var onReset: (() -> Void)?
    var onReroll: (() -> Void)?

    private let params: Params
    private let knobs: [Knob]
    private var sliders: [NSSlider] = []
    private var valueLabels: [NSTextField] = []

    init(params: Params, knobs: [Knob]) {
        self.params = params
        self.knobs = knobs

        let rowH: CGFloat = 26, pad: CGFloat = 8, btnH: CGFloat = 30, width: CGFloat = 324
        let height = pad * 2 + btnH + CGFloat(knobs.count) * rowH
        super.init(frame: NSRect(x: 0, y: 0, width: width, height: height))

        wantsLayer = true
        layer?.backgroundColor = NSColor(white: 0, alpha: 0.55).cgColor
        layer?.cornerRadius = 8

        // button row (top)
        let titles = ["Save", "Load", "Reset", "Brain"]
        for (i, title) in titles.enumerated() {
            let b = NSButton(title: title, target: self, action: #selector(buttonClicked(_:)))
            b.bezelStyle = .rounded
            b.font = .systemFont(ofSize: 11)
            b.tag = i
            b.frame = NSRect(x: 8 + CGFloat(i) * 79, y: height - pad - btnH + 2, width: 74, height: 24)
            addSubview(b)
        }

        // slider rows (below the buttons)
        for (i, k) in knobs.enumerated() {
            let y = height - pad - btnH - CGFloat(i + 1) * rowH + 3

            addSubview(label(k.name, NSRect(x: 8, y: y, width: 100, height: 18), .left))

            let slider = NSSlider(frame: NSRect(x: 112, y: y, width: 156, height: 20))
            slider.minValue = Double(k.lo)
            slider.maxValue = Double(k.hi)
            slider.doubleValue = Double(params[keyPath: k.kp])
            slider.isContinuous = true
            slider.target = self
            slider.action = #selector(sliderChanged(_:))
            slider.tag = i
            sliders.append(slider)
            addSubview(slider)

            let v = label(fmt(params[keyPath: k.kp]), NSRect(x: 272, y: y, width: 46, height: 18), .right)
            valueLabels.append(v)
            addSubview(v)
        }
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    // Re-read params into every slider + value label (after load / reset).
    func refresh() {
        for (i, k) in knobs.enumerated() {
            let val = params[keyPath: k.kp]
            sliders[i].doubleValue = Double(val)
            valueLabels[i].stringValue = fmt(val)
        }
    }

    @objc private func sliderChanged(_ s: NSSlider) {
        let k = knobs[s.tag]
        let v = Float(s.doubleValue)
        params[keyPath: k.kp] = v
        valueLabels[s.tag].stringValue = fmt(v)
    }

    @objc private func buttonClicked(_ b: NSButton) {
        switch b.tag {
        case 0: onSave?()
        case 1: onLoad?()
        case 2: onReset?()
        case 3: onReroll?()
        default: break
        }
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
