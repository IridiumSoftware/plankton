import AppKit
import MetalKit

// Owns the window, the Metal view, and the live-tuning wiring. Instantiated at
// top level in main.swift and held for the program's lifetime.
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow!
    private var mtkView: EngineView!
    private var renderer: Renderer!
    private let params = Params()
    private var tuning: Tuning!

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device available.")
        }

        let frame = NSRect(x: 0, y: 0, width: 1024, height: 1024)

        mtkView = EngineView(frame: frame, device: device)
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.preferredFramesPerSecond = 60

        tuning = Tuning(params: params)
        mtkView.onKey = { [weak self] key in self?.tuning.handleKey(key) }

        renderer = Renderer(device: device, pixelFormat: mtkView.colorPixelFormat, params: params)
        mtkView.delegate = renderer

        window = NSWindow(contentRect: frame,
                          styleMask: [.titled, .closable, .resizable, .miniaturizable],
                          backing: .buffered,
                          defer: false)
        window.title = "fluoddity-metal"
        window.contentView = mtkView
        window.center()
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(mtkView)   // route key events for live tuning
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
