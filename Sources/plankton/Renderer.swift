import MetalKit
import simd
import Foundation

// One sample of the research diagnostics (fed to the HUD + the time plot).
struct Diag { var e: Float; var z: Float; var maxW: Float; var div: Float }

// Drives one frame: advance the simulation, then draw the dye fullscreen and
// the agents as additive points on top. Render-side tunables (toneK,
// pointAlpha) come from the shared Params each frame.
final class Renderer: NSObject, MTKViewDelegate {
    private let queue: MTLCommandQueue
    private let sim: Simulation
    private let params: Params
    private let dyePipe: MTLRenderPipelineState
    private let pointsPipe: MTLRenderPipelineState
    var onDiagnostics: ((Diag) -> Void)?       // diagnostics sink (set by AppDelegate)
    var onSpectrum: (([Float], Int) -> Void)?  // energy spectrum E(k) + avg frame count
    var onCaptureStatus: ((String) -> Void)?   // capture/restore/path status → on-screen slot label
    private let spectrum = Spectrum(n: 1024)   // == fieldDim: full-res, no aliasing
    private var hudFrame = 0
    private var simAcc: Float = 0   // fractional sim-step accumulator (simSpeed)
    private let journal = PathJournal()        // param-trajectory record/replay (capture paths)
    private var creatureLoadIdx = -1           // cycles through captured creatures on restore
    private let recorder: Recorder             // mp4 / gif clip recording (v / g)

    init(device: MTLDevice, pixelFormat: MTLPixelFormat, params: Params, mouse: MouseInput) {
        guard let q = device.makeCommandQueue() else {
            fatalError("could not create command queue")
        }
        self.queue = q
        self.params = params

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: Shaders.source, options: nil)
        } catch {
            fatalError("MSL compile failed: \(error)")
        }
        self.sim = Simulation(device: device, library: library, params: params, mouse: mouse)
        self.recorder = Recorder(device: device)

        let ddesc = MTLRenderPipelineDescriptor()
        ddesc.vertexFunction = library.makeFunction(name: "fs_vertex")
        ddesc.fragmentFunction = library.makeFunction(name: "fs_fragment")
        ddesc.colorAttachments[0].pixelFormat = pixelFormat
        self.dyePipe = try! device.makeRenderPipelineState(descriptor: ddesc)

        let pdesc = MTLRenderPipelineDescriptor()
        pdesc.vertexFunction = library.makeFunction(name: "point_vertex")
        pdesc.fragmentFunction = library.makeFunction(name: "point_fragment")
        let att = pdesc.colorAttachments[0]!
        att.pixelFormat = pixelFormat
        att.isBlendingEnabled = true
        att.rgbBlendOperation = .add
        att.alphaBlendOperation = .add
        att.sourceRGBBlendFactor = .one
        att.sourceAlphaBlendFactor = .one
        att.destinationRGBBlendFactor = .one
        att.destinationAlphaBlendFactor = .one
        self.pointsPipe = try! device.makeRenderPipelineState(descriptor: pdesc)

        super.init()
        recorder.onStatus = { [weak self] s in self?.onCaptureStatus?(s) }
    }

    // clip recording (v = mp4, g = gif); size is the view's drawable size
    func toggleVideo(size: CGSize) { recorder.toggleVideo(size: size) }
    func toggleGIF(size: CGSize) { recorder.toggleGIF(size: size) }

    // ecology mode: cycle off → rps → coexistence → dominance → off (the `e` key).
    // The 8 cohorts become game-theory strategies whose frequencies evolve by the
    // replicator equation; cohort colours (raise pointAlpha) show the live mix.
    private var ecoState = 0
    private let ecoPresets: [Ecology.Preset] = [.rps, .coexistence, .dominance]
    func cycleEcology() {
        ecoState = (ecoState + 1) % 4
        if ecoState == 0 {
            sim.ecologyOn = false
            onCaptureStatus?("ecology: off")
        } else {
            if !sim.ecologyOn {
                sim.syncEcologyFromAgents()
                if params.pointAlpha < 0.25 { params.pointAlpha = 0.35 }   // make the cohort colours visible
            }
            let preset = ecoPresets[ecoState - 1]
            sim.setEcologyPreset(preset)
            sim.ecologyOn = true
            onCaptureStatus?("ecology: \(preset.rawValue) — cohort colours = strategy mix")
        }
    }

    func reroll() { sim.rerollRule(); spectrum.resetAverage() }
    func reset() { sim.reset(); spectrum.resetAverage() }
    func ruleSnapshot() -> [Float] { sim.ruleSnapshot() }
    func loadRule(_ floats: [Float]) { sim.loadRule(floats); spectrum.resetAverage() }
    func resetSpectrumAvg() { spectrum.resetAverage() }

    // ── capture creatures (full state) + paths (parameter trajectory) ─────
    var isReplaying: Bool { journal.replaying }
    func captureCreature() {                                   // c
        let url = Captures.nextURL("creatures", "creature", "fluo")
        do {
            try sim.serializeState(params).write(to: url)
            let name = url.deletingPathExtension().lastPathComponent
            creatureLoadIdx = Captures.list("creatures", "fluo").firstIndex(of: url) ?? creatureLoadIdx
            let n = Captures.list("creatures", "fluo").count
            print("📸 captured \(url.lastPathComponent)")
            onCaptureStatus?("⦿ \(name)   (\(n)/\(n) — captured)")
        } catch { print("capture failed: \(error)"); onCaptureStatus?("capture failed") }
    }
    func restoreCreature() {                                   // x — cycles through captures
        let files = Captures.list("creatures", "fluo")
        guard !files.isEmpty else { print("no creatures captured yet (press c)"); onCaptureStatus?("no creatures yet — press c"); return }
        creatureLoadIdx = (creatureLoadIdx + 1) % files.count
        guard let data = try? Data(contentsOf: files[creatureLoadIdx]) else { return }
        if sim.applyState(data, params) {
            spectrum.resetAverage()
            let name = files[creatureLoadIdx].deletingPathExtension().lastPathComponent
            print("↩︎ restored \(files[creatureLoadIdx].lastPathComponent)")
            onCaptureStatus?("◆ \(name)   (\(creatureLoadIdx + 1)/\(files.count) — loaded)")
        }
    }
    func recordPathToggle() {                                  // j — start/stop recording
        if journal.recording {
            let url = journal.stopRecordingAndSave()
            onCaptureStatus?("■ saved \(url?.deletingPathExtension().lastPathComponent ?? "path")")
        } else {
            journal.startRecording(startState: sim.serializeState(params),
                                   read: { engineKnobs.map { self.params[keyPath: $0.kp] } })
            onCaptureStatus?("● REC path — tune, then press j to save")
        }
    }
    func replayLastPath() {                                    // k — replay the latest path
        guard let url = Captures.list("paths", "fluopath").last else { print("no paths recorded yet (press j)"); onCaptureStatus?("no paths yet — press j"); return }
        if let start = journal.beginReplay(url) {
            sim.applyState(start, params); spectrum.resetAverage()
            onCaptureStatus?("▶ replaying \(url.deletingPathExtension().lastPathComponent)")
        }
    }

    // Full-field CPU reductions for the research HUD (caller throttles the rate):
    // E = mean|u|², Z = mean ω² (enstrophy), |ω|max (intermittency), mean|div|
    // (should sit near 0 — the incompressibility holding).
    private func diagnostics() -> Diag {
        let dim = Int(sim.dim.x)
        let n = dim * dim
        let v = sim.vel.contents().bindMemory(to: Float.self, capacity: 2 * n)
        let w = sim.vort.contents().bindMemory(to: Float.self, capacity: n)
        var energy: Float = 0, enstrophy: Float = 0, maxW: Float = 0, divAbs: Float = 0
        let step = max(1, n / 65536)   // subsample ~64k cells to keep it light
        var count = 0, i = 0
        while i < n {
            let x = i % dim, y = i / dim
            let xR = (x + 1) % dim, xL = (x + dim - 1) % dim
            let yU = (y + 1) % dim, yD = (y + dim - 1) % dim
            let ux = v[2 * i], uy = v[2 * i + 1]
            energy += ux * ux + uy * uy
            let wi = w[i]
            enstrophy += wi * wi
            if abs(wi) > maxW { maxW = abs(wi) }
            let dvg = 0.5 * ((v[2 * (y * dim + xR)] - v[2 * (y * dim + xL)])
                           + (v[2 * (yU * dim + x) + 1] - v[2 * (yD * dim + x) + 1]))
            divAbs += abs(dvg)
            count += 1
            i += step
        }
        let nn = Float(count)
        return Diag(e: energy / nn, z: enstrophy / nn, maxW: maxW, div: divAbs / nn)
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let rpd = view.currentRenderPassDescriptor,
              let cmd = queue.makeCommandBuffer() else { return }

        // path capture/replay: apply this frame's scheduled param changes (replay) and
        // record any manual changes (record). No-ops unless one mode is active.
        journal.tickReplay { i, v in self.params[keyPath: engineKnobs[i].kp] = v }
        journal.tickRecord { engineKnobs.map { self.params[keyPath: $0.kp] } }

        sim.stepEcology()   // replicator-mutator reallocation (no-op unless ecology mode is on)

        // research-viz HUD: compute diagnostics a few times a second from the
        // last completed frame's fields (before this frame's encode touches them)
        hudFrame += 1
        if params.diagnosticsOn && hudFrame % 20 == 0 { onDiagnostics?(diagnostics()) }
        if params.diagnosticsOn && hudFrame % 30 == 0 {
            let n = Int(sim.dim.x)
            let v = sim.vel.contents().bindMemory(to: Float.self, capacity: 2 * n * n)
            spectrum.compute(v, dim: n)
            onSpectrum?(spectrum.ek, spectrum.frames)
        }

        // simSpeed: accumulate fractional steps — ≥1 runs that many sim steps per
        // frame (capped), <1 steps only on some frames (slow-mo), 0 pauses.
        simAcc += params.simSpeed
        var steps = 0
        while simAcc >= 1, steps < 8 { sim.encode(into: cmd); simAcc -= 1; steps += 1 }
        if steps == 8 { simAcc = 0 }   // don't bank a backlog the GPU can't pay down
        sim.encodeViz(into: cmd)       // ω/div views refresh every frame, even paused

        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0.01, green: 0.01, blue: 0.03, alpha: 1)
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }

        var dim = sim.dim
        var toneK = params.toneK
        var satGain = params.satGain
        var bloomStrength = params.bloomStrength
        var palette = params.palette
        enc.setRenderPipelineState(dyePipe)
        enc.setFragmentBuffer(sim.dye, offset: 0, index: 0)
        enc.setFragmentBytes(&dim, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 1)
        enc.setFragmentBytes(&toneK, length: 4, index: 2)
        enc.setFragmentBuffer(sim.vel, offset: 0, index: 3)
        enc.setFragmentBytes(&satGain, length: 4, index: 4)
        enc.setFragmentBuffer(sim.dyeBlur, offset: 0, index: 5)
        enc.setFragmentBytes(&bloomStrength, length: 4, index: 6)
        enc.setFragmentBytes(&palette, length: 4, index: 7)
        var viewMode = params.viewMode
        var vortScale = params.vortScale
        enc.setFragmentBuffer(sim.vort, offset: 0, index: 8)
        enc.setFragmentBytes(&viewMode, length: 4, index: 9)
        enc.setFragmentBytes(&vortScale, length: 4, index: 10)
        enc.setFragmentBuffer(sim.divDisp, offset: 0, index: 11)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

        // agent points — only over the dye art, not the diagnostic field views
        if params.viewMode < 0.5 {
            var pointAlpha = params.pointAlpha
            var pointSize = params.pointSize
            var count = UInt32(sim.particleCount)
            enc.setRenderPipelineState(pointsPipe)
            enc.setVertexBuffer(sim.particleBuffer, offset: 0, index: 0)
            enc.setVertexBytes(&pointSize, length: 4, index: 1)
            enc.setVertexBytes(&count, length: 4, index: 2)
            enc.setVertexBuffer(sim.cohortBuffer, offset: 0, index: 3)
            enc.setFragmentBytes(&pointAlpha, length: 4, index: 0)
            enc.drawPrimitives(type: .point, vertexStart: 0, vertexCount: sim.particleCount)
        }

        enc.endEncoding()
        recorder.grab(drawable: drawable, commandBuffer: cmd)   // no-op unless recording
        cmd.present(drawable)
        cmd.commit()
    }
}
