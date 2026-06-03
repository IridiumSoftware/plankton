import MetalKit
import simd

// MTKView that forwards key presses (live tuning) and mouse drags (fluid stir).
// keyDown is consumed so unhandled keys don't beep. Mouse position is converted
// to normalized [0,1] sim coords; drag deltas accumulate into MouseInput.velN.
// (Property is `mouseInput`, not `mouse`, to avoid NSView.mouse(_:in:).)
final class EngineView: MTKView {
    var onKey: ((String) -> Void)?
    var mouseInput: MouseInput?
    private var lastPosN = SIMD2<Float>(0, 0)

    override var acceptsFirstResponder: Bool { true }

    override func keyDown(with event: NSEvent) {
        onKey?(event.charactersIgnoringModifiers ?? "")
    }

    private func normPos(_ event: NSEvent) -> SIMD2<Float> {
        let p = convert(event.locationInWindow, from: nil)
        let w = max(bounds.width, 1), h = max(bounds.height, 1)
        return SIMD2(Float(p.x / w), Float(p.y / h))
    }

    override func mouseDown(with event: NSEvent) {
        let n = normPos(event); lastPosN = n
        mouseInput?.posN = n; mouseInput?.velN = .zero; mouseInput?.active = true
    }
    override func mouseDragged(with event: NSEvent) {
        let n = normPos(event)
        mouseInput?.velN += (n - lastPosN)
        mouseInput?.posN = n; mouseInput?.active = true
        lastPosN = n
    }
    override func mouseUp(with event: NSEvent) {
        mouseInput?.active = false
    }

    // right-click selects the cohort under the cursor and breeds from it
    override func rightMouseDown(with event: NSEvent) {
        mouseInput?.breedPos = normPos(event)
        mouseInput?.breedRequested = true
    }
}
