import AppKit
import MetalKit

// Owns the window and the MTKView. Instantiated at top level in main.swift and
// held for the program's lifetime (NSApplication.delegate is a weak reference).
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow!
    private var mtkView: MTKView!
    private var renderer: Renderer!

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device available.")
        }

        let frame = NSRect(x: 0, y: 0, width: 1024, height: 1024)

        mtkView = MTKView(frame: frame, device: device)
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.preferredFramesPerSecond = 60

        renderer = Renderer(device: device)
        mtkView.delegate = renderer

        window = NSWindow(contentRect: frame,
                          styleMask: [.titled, .closable, .resizable, .miniaturizable],
                          backing: .buffered,
                          defer: false)
        window.title = "fluoddity-metal"
        window.contentView = mtkView
        window.center()
        window.makeKeyAndOrderFront(nil)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
