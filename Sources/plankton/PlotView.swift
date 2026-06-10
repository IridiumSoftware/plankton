import AppKit

// Scrolling time-series of E (energy, cyan) and Z (enstrophy, orange). Each
// series is auto-scaled to its own recent max, so both show their dynamics
// regardless of absolute magnitude. Fed by the diagnostics callback.
final class PlotView: NSView {
    private var eHist: [CGFloat] = []
    private var zHist: [CGFloat] = []
    private let cap = 240

    private let eColor = NSColor(calibratedRed: 0.4, green: 0.85, blue: 1.0, alpha: 1)
    private let zColor = NSColor(calibratedRed: 1.0, green: 0.6, blue: 0.3, alpha: 1)

    func push(_ e: Float, _ z: Float) {
        eHist.append(CGFloat(e)); if eHist.count > cap { eHist.removeFirst() }
        zHist.append(CGFloat(z)); if zHist.count > cap { zHist.removeFirst() }
        needsDisplay = true
    }

    override func draw(_ dirtyRect: NSRect) {
        NSColor(white: 0, alpha: 0.45).setFill()
        NSBezierPath(roundedRect: bounds, xRadius: 6, yRadius: 6).fill()

        drawSeries(eHist, color: eColor)
        drawSeries(zHist, color: zColor)

        label("E", NSPoint(x: 6, y: bounds.height - 15), eColor)
        label("Z", NSPoint(x: 22, y: bounds.height - 15), zColor)
    }

    private func drawSeries(_ hist: [CGFloat], color: NSColor) {
        guard hist.count > 1 else { return }
        let mx = max(hist.max() ?? 1, 1e-6)
        let pad: CGFloat = 6
        let w = bounds.width - pad * 2, h = bounds.height - pad * 2
        let path = NSBezierPath()
        for (i, v) in hist.enumerated() {
            let x = pad + w * CGFloat(i) / CGFloat(cap - 1)
            let y = pad + h * (v / mx)
            if i == 0 { path.move(to: NSPoint(x: x, y: y)) } else { path.line(to: NSPoint(x: x, y: y)) }
        }
        color.setStroke()
        path.lineWidth = 1.5
        path.stroke()
    }

    private func label(_ s: String, _ p: NSPoint, _ color: NSColor) {
        (s as NSString).draw(at: p, withAttributes: [
            .foregroundColor: color,
            .font: NSFont.monospacedSystemFont(ofSize: 10, weight: .bold)])
    }
}
