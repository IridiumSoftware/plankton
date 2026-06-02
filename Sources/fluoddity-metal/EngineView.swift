import MetalKit

// MTKView that becomes first responder and forwards key presses for live
// tuning. keyDown is consumed (no super call) so unhandled keys don't beep.
final class EngineView: MTKView {
    var onKey: ((String) -> Void)?

    override var acceptsFirstResponder: Bool { true }

    override func keyDown(with event: NSEvent) {
        onKey?(event.charactersIgnoringModifiers ?? "")
    }
}
