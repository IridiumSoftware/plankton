import AppKit

// On-screen tuning panel: a button row (Save / Load / Reset / Brain), then the
// knobs grouped under headers (Particles / VFX / Mouse), each a labeled NSSlider
// with a live value. Sliders write straight into the shared Params; buttons fire
// closures the app wires up. refresh() re-syncs sliders after a preset load/reset.
final class ControlsPanel: NSView {
    var onSave: (() -> Void)?
    var onLoad: (() -> Void)?
    var onReset: (() -> Void)?
    var onReroll: (() -> Void)?
    var onToggleDiag: ((Bool) -> Void)?

    private let params: Params
    private let knobs: [Knob]
    private var sliders: [NSSlider] = []
    private var valueLabels: [NSTextField] = []

    init(params: Params, knobs: [Knob]) {
        self.params = params
        self.knobs = knobs

        let rowH: CGFloat = 26, headerH: CGFloat = 24, btnH: CGFloat = 30
        let pad: CGFloat = 8, width: CGFloat = 324
        let nGroups = CGFloat(Set(knobs.map { $0.group }).count)
        let headersH: CGFloat = nGroups * headerH
        let knobsH: CGFloat = CGFloat(knobs.count) * rowH
        let height: CGFloat = pad * 2 + btnH + headersH + knobsH
        super.init(frame: NSRect(x: 0, y: 0, width: width, height: height))

        wantsLayer = true
        layer?.backgroundColor = NSColor(white: 0, alpha: 0.55).cgColor
        layer?.cornerRadius = 8

        // button row (top): 4 actions + a Diag on/off toggle
        let titles = ["Save", "Load", "Reset", "Brain", "Diag"]
        for (i, title) in titles.enumerated() {
            let b = NSButton(title: title, target: self, action: #selector(buttonClicked(_:)))
            b.bezelStyle = .rounded
            b.font = .systemFont(ofSize: 11)
            b.tag = i
            if i == 4 { b.setButtonType(.pushOnPushOff); b.state = .on }   // Diag toggle (on)
            b.frame = NSRect(x: 8 + CGFloat(i) * 62, y: height - pad - btnH + 2, width: 58, height: 24)
            addSubview(b)
        }

        // grouped knob rows (a header is drawn whenever the group changes)
        var cursor = height - pad - btnH
        var lastGroup = ""
        for (i, k) in knobs.enumerated() {
            if k.group != lastGroup {
                lastGroup = k.group
                addSubview(header(k.group, NSRect(x: 8, y: cursor - headerH + 5, width: width - 16, height: 16)))
                cursor -= headerH
            }
            let y = cursor - rowH + 3
            addSubview(label(k.name, NSRect(x: 10, y: y, width: 98, height: 18), .left))

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

            cursor -= rowH
        }
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    // Re-read params into every slider + value label (after load / reset).
    func refresh() {
        for (i, k) in knobs.enumerated() {
            sliders[i].doubleValue = Double(params[keyPath: k.kp])
            valueLabels[i].stringValue = fmt(params[keyPath: k.kp])
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
        case 4: onToggleDiag?(b.state == .on)
        default: break
        }
    }

    private func fmt(_ v: Float) -> String { String(format: "%.3f", v) }

    private func header(_ s: String, _ frame: NSRect) -> NSTextField {
        let t = label(s.uppercased(), frame, .left)
        t.textColor = NSColor(calibratedRed: 0.5, green: 0.85, blue: 1.0, alpha: 1.0)
        t.font = .boldSystemFont(ofSize: 10)
        return t
    }

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
