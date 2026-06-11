import MetalKit
import simd

// Stage-3 3D renderer: advance the agent-driven fluid, then volumetric ray-march
// the dye density into a glowing volume through the orbital camera.
final class Renderer3D: NSObject, MTKViewDelegate {
    private let queue: MTLCommandQueue
    private let recorder: Recorder             // mp4 / gif clip recording (v / g)
    let sim: Sim3D                              // internal: the 3D panel tunes it
    private let volumePipe: MTLRenderPipelineState
    private let pointPipe: MTLRenderPipelineState
    let camera = Camera3D()
    var densityScale: Float = 0.006
    var sharpness: Float = 1.5      // ray-march transfer gamma — >1 cuts faint halo, crisper edges
    var colorMode: Float = 0                    // 0 density, 1 flow dir, 2 speed, 3 vorticity (fluid-only)
    var pointAlpha: Float = 0.0     // cohort-tinted agent overlay (0 = off; raise to see species when breeding)
    var vortScale: Float = 30       // |ω| → opacity scale for the fluid-only vorticity view
    // capture (set by App3D, which owns the knob list)
    var onCaptureStatus: ((String) -> Void)?
    var paramRead: () -> [Float] = { [] }
    var paramWrite: (Int, Float) -> Void = { _, _ in }
    private let journal = PathJournal()
    private var creatureLoadIdx = -1
    private var simAcc: Float = 0   // fractional sim-step accumulator (simSpeed)
    private var lastCommand: MTLCommandBuffer?   // for GPU sync before CPU reads/writes

    // Block until the in-flight frame's GPU work is done, so capture/breed/restore
    // read/write the shared buffers consistently (they're triggered off the render
    // loop and would otherwise race the move/splat kernels → torn snapshots). All
    // on the main thread + a rare user action, so a one-frame stall is fine.
    private func syncGPU() { lastCommand?.waitUntilCompleted() }

    init(device: MTLDevice, pixelFormat: MTLPixelFormat) {
        queue = device.makeCommandQueue()!
        recorder = Recorder(device: device)
        let lib: MTLLibrary
        do { lib = try device.makeLibrary(source: Shaders3D.source, options: nil) }
        catch { fatalError("MSL3D compile failed: \(error)") }
        sim = Sim3D(device: device, library: lib, fieldDim: 160)   // 160³ (was 128³) — finer dye structures

        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = lib.makeFunction(name: "raymarch_vertex")
        d.fragmentFunction = lib.makeFunction(name: "raymarch_fragment")
        d.colorAttachments[0].pixelFormat = pixelFormat
        volumePipe = try! device.makeRenderPipelineState(descriptor: d)

        // cohort-tinted agent points, additive over the volume
        let pd = MTLRenderPipelineDescriptor()
        pd.vertexFunction = lib.makeFunction(name: "point3d_vertex")
        pd.fragmentFunction = lib.makeFunction(name: "point3d_fragment")
        pd.colorAttachments[0].pixelFormat = pixelFormat
        pd.colorAttachments[0].isBlendingEnabled = true
        pd.colorAttachments[0].rgbBlendOperation = .add
        pd.colorAttachments[0].alphaBlendOperation = .add
        pd.colorAttachments[0].sourceRGBBlendFactor = .one
        pd.colorAttachments[0].destinationRGBBlendFactor = .one
        pd.colorAttachments[0].sourceAlphaBlendFactor = .one
        pd.colorAttachments[0].destinationAlphaBlendFactor = .one
        pointPipe = try! device.makeRenderPipelineState(descriptor: pd)
        super.init()
        recorder.onStatus = { [weak self] s in self?.onCaptureStatus?(s) }
    }

    func toggleVideo(size: CGSize) { recorder.toggleVideo(size: size) }
    func toggleGIF(size: CGSize) { recorder.toggleGIF(size: size) }

    func reroll() { sim.rerollRule() }

    // right-click breed: unproject the click into a world ray (same math as the
    // ray-march fragment), vote + mutate in Sim3D
    func breed(at uv: SIMD2<Float>) {
        syncGPU()
        let invVP = camera.invViewProj()
        let ndc = uv * 2 - SIMD2<Float>(1, 1)
        let nh = invVP * SIMD4<Float>(ndc.x, ndc.y, 0, 1)
        let fh = invVP * SIMD4<Float>(ndc.x, ndc.y, 1, 1)
        let ro = SIMD3(nh.x, nh.y, nh.z) / nh.w
        let rd = simd_normalize(SIMD3(fh.x, fh.y, fh.z) / fh.w - ro)
        sim.breedAt(rayOrigin: ro, rayDir: rd)
        onCaptureStatus?("bred — cohort 0 keeps the parent, 1–7 mutate")
    }
    func adjustDensity(_ f: Float) {
        densityScale = max(0.001, min(2.0, densityScale * f))
        print(String(format: "densityScale = %.3f", densityScale))
    }

    // ── capture creatures (full state) + paths (parameter trajectory), 3D ──
    func captureCreature() {                                   // c
        syncGPU()
        let url = Captures.nextURL("creatures3d", "creature3d", "fluo3")
        do {
            try sim.serializeState(paramRead()).write(to: url)
            let n = Captures.list("creatures3d", "fluo3").count
            onCaptureStatus?("⦿ \(url.deletingPathExtension().lastPathComponent)   (\(n)/\(n) — captured)")
        } catch { onCaptureStatus?("capture failed") }
    }
    func restoreCreature() {                                   // x — cycles through captures
        syncGPU()
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
            syncGPU()
            journal.startRecording(startState: sim.serializeState(paramRead()), read: paramRead)
            onCaptureStatus?("● REC path — tune, then press j to save")
        }
    }
    func replayLastPath() {                                    // k
        guard let url = Captures.list("paths3d", "fluo3path").last else { onCaptureStatus?("no paths yet — press j"); return }
        syncGPU()
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
        var sh = sharpness
        enc.setFragmentBytes(&sh, length: 4, index: 6)
        var vs = vortScale
        enc.setFragmentBytes(&vs, length: 4, index: 7)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

        // cohort-tinted agent overlay (species identity for breeding)
        if pointAlpha > 0.001 {
            enc.setRenderPipelineState(pointPipe)
            enc.setVertexBuffer(sim.particleBuffer, offset: 0, index: 0)
            var vp = camera.viewProj()
            enc.setVertexBytes(&vp, length: MemoryLayout<float4x4>.stride, index: 1)
            var cnt = UInt32(sim.count)
            enc.setVertexBytes(&cnt, length: 4, index: 2)
            var pa = pointAlpha
            enc.setFragmentBytes(&pa, length: 4, index: 0)
            enc.drawPrimitives(type: .point, vertexStart: 0, vertexCount: sim.count)
        }
        enc.endEncoding()

        recorder.grab(drawable: drawable, commandBuffer: cmd)   // no-op unless recording
        cmd.present(drawable)
        cmd.commit()
        lastCommand = cmd   // capture/breed wait on this before touching shared buffers
    }
}
