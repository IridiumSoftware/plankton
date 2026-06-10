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
    var onResetAvg: (() -> Void)?

    private let params: Params
    private let knobs: [Knob]
    private var sliders: [NSSlider?] = []        // nil where the knob is a dropdown
    private var popups: [NSPopUpButton?] = []    // nil where the knob is a slider
    private var valueLabels: [NSTextField?] = [] // nil for dropdowns (title shows the value)
    private var diagButton: NSButton?

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

        // button row (top): 5 actions + a Diag on/off toggle. Custom-styled (bordered
        // NSButtons render with no contrast on the dark translucent panel) — a solid
        // dark-teal fill + a bright border + bold bright text so they POP on the dark UI.
        let titles = ["Save", "Load", "Reset", "Brain", "Avg", "Diag"]
        for (i, title) in titles.enumerated() {
            let b = NSButton(title: title, target: self, action: #selector(buttonClicked(_:)))
            b.tag = i
            b.isBordered = false
            b.wantsLayer = true
            b.layer?.cornerRadius = 5
            b.layer?.borderWidth = 1.2
            b.frame = NSRect(x: 8 + CGFloat(i) * 52, y: height - pad - btnH + 2, width: 50, height: 24)
            if i == 5 { b.setButtonType(.pushOnPushOff); b.state = .on; diagButton = b; styleToggle(b, on: true) }
            else { styleAction(b, title) }
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

            if let opts = k.options {
                let pop = NSPopUpButton(frame: NSRect(x: 112, y: y - 2, width: 206, height: 24), pullsDown: false)
                pop.addItems(withTitles: opts)
                pop.selectItem(at: min(max(Int(params[keyPath: k.kp] + 0.5), 0), opts.count - 1))
                pop.target = self
                pop.action = #selector(popupChanged(_:))
                pop.tag = i
                popups.append(pop); sliders.append(nil); valueLabels.append(nil)
                addSubview(pop)
            } else {
                let slider = NSSlider(frame: NSRect(x: 112, y: y, width: 156, height: 20))
                slider.minValue = Double(k.lo)
                slider.maxValue = Double(k.hi)
                slider.doubleValue = Double(params[keyPath: k.kp])
                slider.isContinuous = true
                slider.target = self
                slider.action = #selector(sliderChanged(_:))
                slider.tag = i
                sliders.append(slider); popups.append(nil)
                addSubview(slider)

                let v = label(fmt(params[keyPath: k.kp]), NSRect(x: 272, y: y, width: 46, height: 18), .right)
                valueLabels.append(v)
                addSubview(v)
            }

            cursor -= rowH
        }
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    // Re-read params into every slider/dropdown + value label (after load / reset).
    func refresh() {
        for (i, k) in knobs.enumerated() {
            let v = params[keyPath: k.kp]
            sliders[i]?.doubleValue = Double(v)
            valueLabels[i]?.stringValue = fmt(v)
            if let pop = popups[i] { pop.selectItem(at: min(max(Int(v + 0.5), 0), pop.numberOfItems - 1)) }
        }
    }

    @objc private func sliderChanged(_ s: NSSlider) {
        let k = knobs[s.tag]
        let v = Float(s.doubleValue)
        params[keyPath: k.kp] = v
        valueLabels[s.tag]?.stringValue = fmt(v)
        onResetAvg?()   // a param change alters the flow → restart the spectrum average
    }

    @objc private func popupChanged(_ p: NSPopUpButton) {
        params[keyPath: knobs[p.tag].kp] = Float(p.indexOfSelectedItem)
        onResetAvg?()
    }

    @objc private func buttonClicked(_ b: NSButton) {
        switch b.tag {
        case 0: onSave?()
        case 1: onLoad?()
        case 2: onReset?()
        case 3: onReroll?()
        case 4: onResetAvg?()
        case 5: let on = (b.state == .on); styleToggle(b, on: on); onToggleDiag?(on)
        default: break
        }
    }

    private func fmt(_ v: Float) -> String { String(format: "%.3f", v) }

    // ── high-contrast button styling (so the controls pop on the dark panel) ──
    private func styleAction(_ b: NSButton, _ title: String) {
        b.layer?.backgroundColor = NSColor(calibratedRed: 0.10, green: 0.27, blue: 0.40, alpha: 0.98).cgColor
        b.layer?.borderColor = NSColor(calibratedRed: 0.45, green: 0.82, blue: 1.0, alpha: 0.95).cgColor
        b.attributedTitle = NSAttributedString(string: title, attributes: [
            .foregroundColor: NSColor(calibratedRed: 0.85, green: 0.96, blue: 1.0, alpha: 1.0),
            .font: NSFont.boldSystemFont(ofSize: 11)])
    }
    private func styleToggle(_ b: NSButton, on: Bool) {
        let bg  = on ? NSColor(calibratedRed: 0.13, green: 0.52, blue: 0.42, alpha: 0.98)
                     : NSColor(calibratedRed: 0.20, green: 0.20, blue: 0.23, alpha: 0.95)
        let brd = on ? NSColor(calibratedRed: 0.40, green: 1.00, blue: 0.70, alpha: 0.95)
                     : NSColor(calibratedRed: 0.50, green: 0.50, blue: 0.55, alpha: 0.90)
        let txt = on ? NSColor(calibratedRed: 0.88, green: 1.00, blue: 0.92, alpha: 1.0)
                     : NSColor(calibratedRed: 0.62, green: 0.62, blue: 0.68, alpha: 1.0)
        b.layer?.backgroundColor = bg.cgColor
        b.layer?.borderColor = brd.cgColor
        b.attributedTitle = NSAttributedString(string: on ? "Diag●" : "Diag○",
            attributes: [.foregroundColor: txt, .font: NSFont.boldSystemFont(ofSize: 11)])
    }

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
