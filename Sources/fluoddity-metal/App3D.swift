import AppKit
import MetalKit

// 3D app delegate (launched with `--3d`). Stage 1: a window + orbital 3D view
// over the ABC-flow point cloud. Controls/diagnostics come in later stages.
final class App3D: NSObject, NSApplicationDelegate {
    private var window: NSWindow!
    private var view: View3D!
    private var renderer: Renderer3D!

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device available.")
        }
        let frame = NSRect(x: 0, y: 0, width: 1024, height: 1024)

        view = View3D(frame: frame, device: device)
        view.colorPixelFormat = .bgra8Unorm
        view.preferredFramesPerSecond = 60

        renderer = Renderer3D(device: device, pixelFormat: view.colorPixelFormat)
        view.delegate = renderer
        view.camera = renderer.camera
        view.onReroll = { [weak self] in self?.renderer.reroll() }
        view.onDensity = { [weak self] f in self?.renderer.adjustDensity(f) }

        window = NSWindow(contentRect: frame,
                          styleMask: [.titled, .closable, .resizable, .miniaturizable],
                          backing: .buffered,
                          defer: false)
        window.title = "fluoddity-metal — 3D"
        window.contentView = view
        window.center()
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(view)
        print("3D: drag to orbit · scroll to zoom")
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
