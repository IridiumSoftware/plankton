import AppKit

// A live-tuning knob backed by get/set closures (so it can drive Sim3D's vars
// and the renderer directly, without a shared params class).
// `group` buckets the knob under a panel header (display only — the knobs ARRAY
// order is the capture/path serialization order and must stay append-only).
// `options` non-nil ⇒ rendered as a dropdown (value = option index).
struct Knob3D {
    let name: String
    let get: () -> Float
    let set: (Float) -> Void
    let lo: Float
    let hi: Float
    var group: String = "Agents"
    var options: [String]? = nil
}

// On-screen panel for the 3D engine: a Brain (re-roll) button + the knobs
// bucketed under group headers (mirrors the 2D ControlsPanel). Display order is
// group-by-group; control tags stay the knob's ARRAY index, so captures/paths
// (which serialize by array index) are unaffected by the visual grouping.
final class Panel3D: NSView {
    var onReroll: (() -> Void)?

    private let knobs: [Knob3D]
    private var valueLabels: [NSTextField?] = []
    private var sliders: [NSSlider?] = []
    private var popups: [NSPopUpButton?] = []

    init(knobs: [Knob3D]) {
        self.knobs = knobs
        // groups in order of first appearance in the array
        var groupOrder: [String] = []
        for k in knobs where !groupOrder.contains(k.group) { groupOrder.append(k.group) }

        let rowH: CGFloat = 26, headerH: CGFloat = 24, btnH: CGFloat = 30
        let pad: CGFloat = 8, width: CGFloat = 300
        let height = pad * 2 + btnH + CGFloat(groupOrder.count) * headerH + CGFloat(knobs.count) * rowH
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

        sliders = Array(repeating: nil, count: knobs.count)
        popups = Array(repeating: nil, count: knobs.count)
        valueLabels = Array(repeating: nil, count: knobs.count)

        var cursor = height - pad - btnH
        for g in groupOrder {
            addSubview(header(g, NSRect(x: 8, y: cursor - headerH + 5, width: width - 16, height: 16)))
            cursor -= headerH
            for (i, k) in knobs.enumerated() where k.group == g {
                let y = cursor - rowH + 3
                addSubview(label(k.name, NSRect(x: 8, y: y, width: 96, height: 18), .left))

                if let opts = k.options {
                    let pop = NSPopUpButton(frame: NSRect(x: 108, y: y - 2, width: 184, height: 24), pullsDown: false)
                    pop.addItems(withTitles: opts)
                    pop.selectItem(at: min(max(Int(k.get() + 0.5), 0), opts.count - 1))
                    pop.target = self
                    pop.action = #selector(popupChanged(_:))
                    pop.tag = i
                    popups[i] = pop
                    addSubview(pop)
                } else {
                    let s = NSSlider(frame: NSRect(x: 108, y: y, width: 140, height: 20))
                    s.minValue = Double(k.lo)
                    s.maxValue = Double(k.hi)
                    s.doubleValue = Double(k.get())
                    s.isContinuous = true
                    s.target = self
                    s.action = #selector(sliderChanged(_:))
                    s.tag = i
                    sliders[i] = s
                    addSubview(s)

                    let v = label(fmt(k.get()), NSRect(x: 250, y: y, width: 42, height: 18), .right)
                    valueLabels[i] = v
                    addSubview(v)
                }
                cursor -= rowH
            }
        }
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    // Re-sync every control from the knobs (after a creature restore / path replay).
    func refresh() {
        for (i, k) in knobs.enumerated() {
            let v = k.get()
            sliders[i]?.doubleValue = Double(v)
            valueLabels[i]?.stringValue = fmt(v)
            if let pop = popups[i] { pop.selectItem(at: min(max(Int(v + 0.5), 0), pop.numberOfItems - 1)) }
        }
    }

    @objc private func brain() { onReroll?() }

    @objc private func sliderChanged(_ s: NSSlider) {
        let k = knobs[s.tag]
        let v = Float(s.doubleValue)
        k.set(v)
        valueLabels[s.tag]?.stringValue = fmt(v)
    }

    @objc private func popupChanged(_ p: NSPopUpButton) {
        knobs[p.tag].set(Float(p.indexOfSelectedItem))
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
        t.isEditable = false; t.isBordered = false; t.drawsBackground = false; t.isSelectable = false
        t.textColor = .white
        t.font = .systemFont(ofSize: 11)
        t.alignment = align
        return t
    }
}
