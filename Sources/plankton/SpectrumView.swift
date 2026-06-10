import AppKit

// Log-log plot of the energy spectrum E(k) with a dashed −5/3 reference line.
// Where the measured curve runs parallel to the dashed line is an inertial
// (Kolmogorov-like) cascade range.
final class SpectrumView: NSView {
    private var ek: [Float] = []
    private var frames = 0

    func update(_ e: [Float], _ n: Int) { ek = e; frames = n; needsDisplay = true }

    override func draw(_ dirtyRect: NSRect) {
        NSColor(white: 0, alpha: 0.45).setFill()
        NSBezierPath(roundedRect: bounds, xRadius: 6, yRadius: 6).fill()
        guard ek.count > 3 else { return }

        let pad: CGFloat = 8
        let W = bounds.width - pad * 2, H = bounds.height - pad * 2 - 6
        let kmax = ek.count - 1

        var emax: Float = 0
        for k in 1..<ek.count { emax = max(emax, ek[k]) }
        guard emax > 0 else { return }
        let lemax = log10(emax), lemin = lemax - 6        // show the top 6 decades
        let lkmax = log10(Float(kmax))
        func px(_ k: Float) -> CGFloat { pad + W * CGFloat(log10(k) / lkmax) }
        func py(_ e: Float) -> CGFloat {
            let le = max(log10(max(e, 1e-30)), lemin)
            return pad + H * CGFloat((le - lemin) / 6)
        }

        // −5/3 reference (anchored at k=2, E=emax)
        let ref = NSBezierPath()
        ref.move(to: NSPoint(x: px(2), y: py(emax)))
        ref.line(to: NSPoint(x: px(Float(kmax)), y: py(emax * pow(Float(kmax) / 2, -5.0 / 3.0))))
        NSColor(calibratedRed: 1.0, green: 0.6, blue: 0.3, alpha: 0.7).setStroke()
        ref.lineWidth = 1.0
        ref.setLineDash([4, 3], count: 2, phase: 0)
        ref.stroke()

        // measured E(k)
        let path = NSBezierPath()
        var started = false
        for k in 1...kmax where ek[k] > 0 {
            let p = NSPoint(x: px(Float(k)), y: py(ek[k]))
            if started { path.line(to: p) } else { path.move(to: p); started = true }
        }
        NSColor(calibratedRed: 0.4, green: 0.85, blue: 1.0, alpha: 1).setStroke()
        path.lineWidth = 1.5
        path.stroke()

        // Slope fits via the SHARED SpectrumFit (identical to the --sweep harness).
        // The faint dashed verticals mark the trimmed inertial window [inLo, inHi].
        let f = SpectrumFit.compute(ek)
        NSColor(white: 1, alpha: 0.22).setStroke()
        for kk in [f.inLo, f.inHi] where kk >= 1 {
            let g = NSBezierPath()
            g.move(to: NSPoint(x: px(Float(kk)), y: pad))
            g.line(to: NSPoint(x: px(Float(kk)), y: pad + H))
            g.lineWidth = 0.5
            g.setLineDash([2, 2], count: 2, phase: 0)
            g.stroke()
        }
        let attrs: [NSAttributedString.Key: Any] = [
            .foregroundColor: NSColor.white,
            .font: NSFont.monospacedSystemFont(ofSize: 9, weight: .regular)]
        let l1 = String(format: "peak k=%d   high %.2f (R\u{00B2}%.2f)   avg N=%d",
                        f.peakK, f.highSlope, f.highR2, frames)
        let l2 = String(format: "inertial k%d\u{2013}%d: %.2f (R\u{00B2}%.2f)   \u{2212}5/3=\u{2212}1.67",
                        f.inLo, f.inHi, f.inSlope, f.inR2)
        (l1 as NSString).draw(at: NSPoint(x: 8, y: bounds.height - 15), withAttributes: attrs)
        (l2 as NSString).draw(at: NSPoint(x: 8, y: bounds.height - 28), withAttributes: attrs)
    }
}
