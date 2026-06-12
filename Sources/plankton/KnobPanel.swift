import AppKit

// One live-tunable knob for the shared control panel: closure-backed so it can
// bind the 2D Params keypaths and the 3D Sim3D/renderer vars alike.
// `group` buckets it under a header — DISPLAY ONLY; the knobs ARRAY order is the
// capture/path serialization order in both engines and must stay append-only.
// `info` is the plain-language one-liner shown in the tooltip + footer strip.
// `options` non-nil ⇒ rendered as a dropdown (value = option index).
struct UIKnob {
    let name: String
    let info: String
    let get: () -> Float
    let set: (Float) -> Void
    let lo: Float
    let hi: Float
    var step: Float = 0                 // 0 ⇒ (hi−lo)/100 (−/+ stepper increment)
    var group: String = "Agents"
    var options: [String]? = nil
    var stepValue: Float { step > 0 ? step : (hi - lo) / 100 }
}

// The control panel shared by the 2D and 3D apps: an action-button row, then the
// knobs bucketed under group headers (each row: name · − · slider · + · value, or
// name · dropdown), all tagged by ARRAY index so captures/paths are unaffected by
// the visual grouping. Hovering a row (or adjusting it) reports the knob's
// description through `onInfo` — the Sidebar pins that into an always-visible
// footer. Flipped (y down) so layout is top-anchored and scrolls naturally.
final class KnobPanel: NSView {
    static let panelWidth: CGFloat = 340

    var onAnyChange: (() -> Void)?       // fired on every knob change (2D restarts the spectrum average)
    var onInfo: ((String) -> Void)?      // hovered/adjusted knob description → Sidebar footer

    private let knobs: [UIKnob]
    private var sliders: [NSSlider?]
    private var popups: [NSPopUpButton?]
    private var valueLabels: [NSTextField?]
    private var buttonActions: [() -> Void] = []
    private var toggleAction: ((Bool) -> Void)?
    private var toggleTitle = ""
    private var rowHitRects: [(NSRect, Int)] = []   // row rect → knob array index (hover → footer)
    private var lastHover = -2                       // dedup hover callbacks (-2 none, -1 background)

    override var isFlipped: Bool { true }

    init(knobs: [UIKnob],
         buttons: [(title: String, tip: String, action: () -> Void)] = [],
         toggle: (title: String, tip: String, initial: Bool, onChange: (Bool) -> Void)? = nil) {
        self.knobs = knobs
        sliders = Array(repeating: nil, count: knobs.count)
        popups = Array(repeating: nil, count: knobs.count)
        valueLabels = Array(repeating: nil, count: knobs.count)
        super.init(frame: NSRect(x: 0, y: 0, width: Self.panelWidth, height: 100))

        let W = Self.panelWidth
        let rowH: CGFloat = 24, headerH: CGFloat = 22, pad: CGFloat = 10
        var y = pad

        // ── action buttons (one row) ──
        if !buttons.isEmpty || toggle != nil {
            let slotW: CGFloat = 52
            var x: CGFloat = 8
            for (i, b) in buttons.enumerated() {
                let btn = actionButton(b.title, tip: b.tip)
                btn.tag = i
                btn.target = self
                btn.action = #selector(actionClicked(_:))
                btn.frame = NSRect(x: x, y: y, width: slotW - 4, height: 22)
                buttonActions.append(b.action)
                addSubview(btn)
                x += slotW
            }
            if let t = toggle {
                toggleTitle = t.title
                toggleAction = t.onChange
                let btn = actionButton(t.title, tip: t.tip)
                btn.setButtonType(.pushOnPushOff)
                btn.state = t.initial ? .on : .off
                btn.target = self
                btn.action = #selector(toggleClicked(_:))
                btn.frame = NSRect(x: x, y: y, width: slotW - 4, height: 22)
                styleToggle(btn, on: t.initial)
                addSubview(btn)
            }
            y += 22 + 8
        }

        // ── knob rows, bucketed by group (first-appearance order) ──
        var groupOrder: [String] = []
        for k in knobs where !groupOrder.contains(k.group) { groupOrder.append(k.group) }
        for g in groupOrder {
            addSubview(headerLabel(g, NSRect(x: 10, y: y + 5, width: W - 20, height: 14)))
            y += headerH
            for (i, k) in knobs.enumerated() where k.group == g {
                rowHitRects.append((NSRect(x: 0, y: y, width: W, height: rowH), i))
                let name = label(k.name, NSRect(x: 10, y: y + 3, width: 104, height: 16), .left)
                name.toolTip = k.info
                addSubview(name)

                if let opts = k.options {
                    let pop = NSPopUpButton(frame: NSRect(x: 116, y: y, width: 214, height: 22), pullsDown: false)
                    pop.addItems(withTitles: opts)
                    pop.selectItem(at: min(max(Int(k.get() + 0.5), 0), opts.count - 1))
                    pop.font = .systemFont(ofSize: 11)
                    pop.target = self
                    pop.action = #selector(popupChanged(_:))
                    pop.tag = i
                    pop.toolTip = k.info
                    popups[i] = pop
                    addSubview(pop)
                } else {
                    // −/+ steppers flank the slider: one fine step per click
                    addSubview(stepButton("−", NSRect(x: 116, y: y + 2, width: 18, height: 18), i, #selector(stepDown(_:)), k.info))
                    let s = NSSlider(frame: NSRect(x: 138, y: y + 1, width: 116, height: 20))
                    s.minValue = Double(k.lo)
                    s.maxValue = Double(k.hi)
                    s.doubleValue = Double(k.get())
                    s.isContinuous = true
                    s.controlSize = .small
                    s.target = self
                    s.action = #selector(sliderChanged(_:))
                    s.tag = i
                    s.toolTip = k.info
                    sliders[i] = s
                    addSubview(s)
                    addSubview(stepButton("+", NSRect(x: 258, y: y + 2, width: 18, height: 18), i, #selector(stepUp(_:)), k.info))
                    let v = label(fmt(k.get()), NSRect(x: 280, y: y + 3, width: W - 280 - 10, height: 16), .right)
                    v.toolTip = k.info
                    valueLabels[i] = v
                    addSubview(v)
                }
                y += rowH
            }
            y += 4
        }
        setFrameSize(NSSize(width: W, height: y + pad))
    }

    required init?(coder: NSCoder) { fatalError("not used") }

    // Re-sync every control from the knobs (after preset load / capture restore / replay).
    func refresh() {
        for (i, k) in knobs.enumerated() {
            let v = k.get()
            sliders[i]?.doubleValue = Double(v)
            valueLabels[i]?.stringValue = fmt(v)
            if let pop = popups[i] { pop.selectItem(at: min(max(Int(v + 0.5), 0), pop.numberOfItems - 1)) }
        }
    }

    // ── hover → description footer ──
    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        trackingAreas.forEach(removeTrackingArea)
        addTrackingArea(NSTrackingArea(rect: bounds,
                                       options: [.mouseMoved, .mouseEnteredAndExited, .activeInKeyWindow, .inVisibleRect],
                                       owner: self, userInfo: nil))
    }
    override func mouseMoved(with e: NSEvent) {
        let p = convert(e.locationInWindow, from: nil)
        let hit = rowHitRects.first(where: { $0.0.contains(p) })?.1 ?? -1
        guard hit != lastHover else { return }
        lastHover = hit
        onInfo?(hit >= 0 ? describe(hit) : "")
    }
    override func mouseExited(with e: NSEvent) {
        lastHover = -2
        onInfo?("")
    }
    private func describe(_ i: Int) -> String { "\(knobs[i].name) — \(knobs[i].info)" }

    // ── actions ──
    @objc private func actionClicked(_ b: NSButton) { buttonActions[b.tag]() }
    @objc private func toggleClicked(_ b: NSButton) {
        let on = (b.state == .on)
        styleToggle(b, on: on)
        toggleAction?(on)
    }
    @objc private func sliderChanged(_ s: NSSlider) {
        knobs[s.tag].set(Float(s.doubleValue))
        valueLabels[s.tag]?.stringValue = fmt(Float(s.doubleValue))
        onInfo?(describe(s.tag))
        onAnyChange?()
    }
    @objc private func popupChanged(_ p: NSPopUpButton) {
        knobs[p.tag].set(Float(p.indexOfSelectedItem))
        onInfo?(describe(p.tag))
        onAnyChange?()
    }
    @objc private func stepDown(_ b: NSButton) { step(b.tag, -1) }
    @objc private func stepUp(_ b: NSButton) { step(b.tag, +1) }
    private func step(_ i: Int, _ dir: Float) {
        let k = knobs[i]
        let v = min(max(k.get() + dir * k.stepValue, k.lo), k.hi)
        k.set(v)
        sliders[i]?.doubleValue = Double(v)
        valueLabels[i]?.stringValue = fmt(v)
        onInfo?(describe(i))
        onAnyChange?()
    }

    private func fmt(_ v: Float) -> String { String(format: "%.3f", v) }

    // ── widget factories (high-contrast styling for the dark sidebar) ──
    private func actionButton(_ title: String, tip: String) -> NSButton {
        let b = NSButton(title: title, target: nil, action: nil)
        b.isBordered = false               // bordered NSButtons vanish on the dark panel
        b.wantsLayer = true
        b.layer?.cornerRadius = 5
        b.layer?.borderWidth = 1.2
        b.layer?.backgroundColor = NSColor(calibratedRed: 0.10, green: 0.27, blue: 0.40, alpha: 0.98).cgColor
        b.layer?.borderColor = NSColor(calibratedRed: 0.45, green: 0.82, blue: 1.0, alpha: 0.95).cgColor
        b.attributedTitle = NSAttributedString(string: title, attributes: [
            .foregroundColor: NSColor(calibratedRed: 0.85, green: 0.96, blue: 1.0, alpha: 1.0),
            .font: NSFont.boldSystemFont(ofSize: 11)])
        b.toolTip = tip
        return b
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
        b.attributedTitle = NSAttributedString(string: on ? "\(toggleTitle)●" : "\(toggleTitle)○",
            attributes: [.foregroundColor: txt, .font: NSFont.boldSystemFont(ofSize: 11)])
    }
    private func stepButton(_ title: String, _ frame: NSRect, _ tag: Int, _ action: Selector, _ tip: String) -> NSButton {
        let b = NSButton(title: title, target: self, action: action)
        b.tag = tag
        b.isBordered = false
        b.wantsLayer = true
        b.layer?.cornerRadius = 4
        b.layer?.borderWidth = 1
        b.layer?.backgroundColor = NSColor(calibratedRed: 0.10, green: 0.27, blue: 0.40, alpha: 0.98).cgColor
        b.layer?.borderColor = NSColor(calibratedRed: 0.45, green: 0.82, blue: 1.0, alpha: 0.7).cgColor
        b.attributedTitle = NSAttributedString(string: title, attributes: [
            .foregroundColor: NSColor(calibratedRed: 0.85, green: 0.96, blue: 1.0, alpha: 1.0),
            .font: NSFont.boldSystemFont(ofSize: 11)])
        b.toolTip = tip
        b.frame = frame
        return b
    }
    private func headerLabel(_ s: String, _ frame: NSRect) -> NSTextField {
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

// Fixed-width control sidebar: the scrollable KnobPanel on top + an always-visible
// description strip pinned at the bottom (fed by the panel's onInfo). The app
// delegates place it at x = 0 and put the Metal view to its right, so the knobs
// can never overlap the rendering.
final class Sidebar: NSView {
    static let width: CGFloat = KnobPanel.panelWidth
    private let footer: NSTextField
    private let hint = "hover a knob to see what it does · drag sliders, or click − / + for one fine step"

    init(panel: KnobPanel, height: CGFloat) {
        footer = NSTextField(wrappingLabelWithString: "")
        super.init(frame: NSRect(x: 0, y: 0, width: Self.width, height: height))
        wantsLayer = true
        layer?.backgroundColor = NSColor(calibratedRed: 0.075, green: 0.085, blue: 0.11, alpha: 1.0).cgColor

        let footerH: CGFloat = 76
        footer.frame = NSRect(x: 10, y: 8, width: Self.width - 20, height: footerH - 16)
        footer.isEditable = false; footer.isSelectable = false
        footer.textColor = NSColor(calibratedRed: 0.75, green: 0.82, blue: 0.88, alpha: 1.0)
        footer.font = .systemFont(ofSize: 10.5)
        footer.stringValue = hint
        footer.autoresizingMask = [.maxYMargin]
        addSubview(footer)

        let sep = NSView(frame: NSRect(x: 0, y: footerH, width: Self.width, height: 1))
        sep.wantsLayer = true
        sep.layer?.backgroundColor = NSColor(white: 1.0, alpha: 0.12).cgColor
        sep.autoresizingMask = [.maxYMargin, .width]
        addSubview(sep)

        let sv = NSScrollView(frame: NSRect(x: 0, y: footerH + 1, width: Self.width, height: height - footerH - 1))
        sv.autoresizingMask = [.height]
        sv.hasVerticalScroller = true
        sv.drawsBackground = false
        sv.verticalScrollElasticity = .allowed
        sv.documentView = panel
        addSubview(sv)

        panel.onInfo = { [weak self] s in
            guard let self else { return }
            self.footer.stringValue = s.isEmpty ? self.hint : s
        }
    }

    required init?(coder: NSCoder) { fatalError("not used") }
}
