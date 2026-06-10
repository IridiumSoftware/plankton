import MetalKit
import simd

// Stage-3 3D renderer: advance the agent-driven fluid, then volumetric ray-march
// the dye density into a glowing volume through the orbital camera.
final class Renderer3D: NSObject, MTKViewDelegate {
    private let queue: MTLCommandQueue
    let sim: Sim3D                              // internal: the 3D panel tunes it
    private let volumePipe: MTLRenderPipelineState
    let camera = Camera3D()
    var densityScale: Float = 0.006
    var colorMode: Float = 0                    // 0 density, 1 flow direction, 2 speed
    // capture (set by App3D, which owns the knob list)
    var onCaptureStatus: ((String) -> Void)?
    var paramRead: () -> [Float] = { [] }
    var paramWrite: (Int, Float) -> Void = { _, _ in }
    private let journal = PathJournal()
    private var creatureLoadIdx = -1
    private var simAcc: Float = 0   // fractional sim-step accumulator (simSpeed)

    init(device: MTLDevice, pixelFormat: MTLPixelFormat) {
        queue = device.makeCommandQueue()!
        let lib: MTLLibrary
        do { lib = try device.makeLibrary(source: Shaders3D.source, options: nil) }
        catch { fatalError("MSL3D compile failed: \(error)") }
        sim = Sim3D(device: device, library: lib, fieldDim: 160)   // 160³ (was 128³) — finer dye structures

        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = lib.makeFunction(name: "raymarch_vertex")
        d.fragmentFunction = lib.makeFunction(name: "raymarch_fragment")
        d.colorAttachments[0].pixelFormat = pixelFormat
        volumePipe = try! device.makeRenderPipelineState(descriptor: d)
        super.init()
    }

    func reroll() { sim.rerollRule() }
    func adjustDensity(_ f: Float) {
        densityScale = max(0.001, min(2.0, densityScale * f))
        print(String(format: "densityScale = %.3f", densityScale))
    }

    // ── capture creatures (full state) + paths (parameter trajectory), 3D ──
    func captureCreature() {                                   // c
        let url = Captures.nextURL("creatures3d", "creature3d", "fluo3")
        do {
            try sim.serializeState(paramRead()).write(to: url)
            let n = Captures.list("creatures3d", "fluo3").count
            onCaptureStatus?("⦿ \(url.deletingPathExtension().lastPathComponent)   (\(n)/\(n) — captured)")
        } catch { onCaptureStatus?("capture failed") }
    }
    func restoreCreature() {                                   // x — cycles through captures
        let files = Captures.list("creatures3d", "fluo3")
        guard !files.isEmpty else { onCaptureStatus?("no creatures yet — press c"); return }
        creatureLoadIdx = (creatureLoadIdx + 1) % files.count
        guard let data = try? Data(contentsOf: files[creatureLoadIdx]), let p = sim.applyState(data) else { return }
        for (i, v) in p.enumerated() { paramWrite(i, v) }
        onCaptureStatus?("◆ \(files[creatureLoadIdx].deletingPathExtension().lastPathComponent)   (\(creatureLoadIdx + 1)/\(files.count) — loaded)")
    }
    func recordPathToggle() {                                  // j
        if journal.recording {
            let url = journal.stopRecordingAndSave("paths3d", "path3d", "fluo3path")
            onCaptureStatus?("■ saved \(url?.deletingPathExtension().lastPathComponent ?? "path")")
        } else {
            journal.startRecording(startState: sim.serializeState(paramRead()), read: paramRead)
            onCaptureStatus?("● REC path — tune, then press j to save")
        }
    }
    func replayLastPath() {                                    // k
        guard let url = Captures.list("paths3d", "fluo3path").last else { onCaptureStatus?("no paths yet — press j"); return }
        if let start = journal.beginReplay(url), let p = sim.applyState(start) {
            for (i, v) in p.enumerated() { paramWrite(i, v) }
            onCaptureStatus?("▶ replaying \(url.deletingPathExtension().lastPathComponent)")
        }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        camera.aspect = Float(size.width / max(size.height, 1))
    }

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let rpd = view.currentRenderPassDescriptor,
              let cmd = queue.makeCommandBuffer() else { return }

        journal.tickReplay(paramWrite)   // applies scheduled param changes (no-op unless replaying)
        journal.tickRecord(paramRead)    // records param deltas (no-op unless recording)

        // simSpeed: accumulate fractional steps — ≥1 runs that many sim steps per
        // frame (capped), <1 steps only on some frames (slow-mo), 0 pauses.
        simAcc += sim.simSpeed
        var steps = 0
        while simAcc >= 1, steps < 8 { sim.encode(into: cmd); simAcc -= 1; steps += 1 }
        if steps == 8 { simAcc = 0 }   // don't bank a backlog the GPU can't pay down

        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0.02, green: 0.02, blue: 0.05, alpha: 1)
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }
        enc.setRenderPipelineState(volumePipe)
        enc.setFragmentBuffer(sim.dye, offset: 0, index: 0)
        var dim = sim.dim
        enc.setFragmentBytes(&dim, length: 16, index: 1)
        var invVP = camera.invViewProj()
        enc.setFragmentBytes(&invVP, length: MemoryLayout<float4x4>.stride, index: 2)
        var ds = densityScale
        enc.setFragmentBytes(&ds, length: 4, index: 3)
        enc.setFragmentBuffer(sim.vel, offset: 0, index: 4)
        var cm = colorMode
        enc.setFragmentBytes(&cm, length: 4, index: 5)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()

        cmd.present(drawable)
        cmd.commit()
    }
}
