import AVFoundation
import CoreVideo
import ImageIO
import Metal
import QuartzCore

// Records the live engine to a shareable clip. Two outputs, one frame-grab path:
//   • mp4  (key `v`) — AVAssetWriter H.264, streamed: efficient, full quality.
//   • gif  (key `g`) — ImageIO animated GIF, downscaled + frame-capped so memory
//                      stays bounded; for quick loops to drop in chat/issues.
//
// Each rendered frame, grab() blits the drawable into a pooled BGRA pixel buffer
// on the SAME command buffer, then appends it once the GPU finishes (completion
// handler) — so recording never stalls the render loop. Files land in
// captures/video/. Verify the encoder headlessly with `--rectest`.
final class Recorder {
    enum Mode { case mp4, gif }

    private let device: MTLDevice
    private var cache: CVMetalTextureCache?

    // mp4 writer
    private var writer: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    // gif accumulation
    private var gifFrames: [CGImage] = []
    private let gifMaxFrames = 240        // ~8 s at 30fps cap
    private let gifEvery = 2              // sample every Nth frame → ~30fps from 60
    private let gifMaxDim = 512           // longest side, downscaled
    private var gifFrameCounter = 0

    private var mode: Mode = .mp4
    private var outURL: URL?
    private var startTime: CFTimeInterval = 0
    private(set) var isRecording = false
    private let q = DispatchQueue(label: "plankton.recorder")
    private var pending = 0
    private var stopping = false
    var onStatus: ((String) -> Void)?

    init(device: MTLDevice) {
        self.device = device
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache)
    }

    // ── public toggles (wired to the v / g keys) ───────────────────────────
    func toggleVideo(size: CGSize) { isRecording ? stop() : startVideo(size: size) }
    func toggleGIF(size: CGSize)   { isRecording ? stop() : startGIF() }

    private func startVideo(size: CGSize) {
        let url = Captures.nextURL("video", "clip", "mp4")
        if beginVideo(url: url, width: Int(size.width), height: Int(size.height)) {
            mode = .mp4; isRecording = true; startTime = 0
            onStatus?("● REC video → \(url.lastPathComponent)")
        } else {
            onStatus?("video: failed to start")
        }
    }

    private func startGIF() {
        mode = .gif; gifFrames.removeAll(); gifFrameCounter = 0
        outURL = Captures.nextURL("video", "clip", "gif")
        isRecording = true; startTime = 0
        onStatus?("● REC gif (downscaled) — press g to save")
    }

    // ── mp4 encoder (also driven directly by --rectest) ────────────────────
    @discardableResult
    func beginVideo(url: URL, width: Int, height: Int) -> Bool {
        let w = max(2, width & ~1), h = max(2, height & ~1)   // h264 wants even dims
        try? FileManager.default.removeItem(at: url)
        guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mp4) else { return false }
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: w, AVVideoHeightKey: h,
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = true
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferMetalCompatibilityKey as String: true,
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: attrs)
        guard writer.canAdd(input) else { return false }
        writer.add(input)
        guard writer.startWriting() else { return false }
        writer.startSession(atSourceTime: .zero)
        self.writer = writer; self.input = input; self.adaptor = adaptor
        self.outURL = url; self.pending = 0; self.stopping = false
        return true
    }

    func pooledBuffer() -> CVPixelBuffer? {
        guard let pool = adaptor?.pixelBufferPool else { return nil }
        var pb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pb)
        return pb
    }

    private func texture(for pb: CVPixelBuffer) -> MTLTexture? {
        guard let cache = cache else { return nil }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        var cvtex: CVMetalTexture?
        let r = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache, pb, nil,
                                                          .bgra8Unorm, w, h, 0, &cvtex)
        guard r == kCVReturnSuccess, let cvtex = cvtex else { return nil }
        return CVMetalTextureGetTexture(cvtex)
    }

    // append a finished mp4 frame at wall-clock `seconds`; serialized on q.
    func append(_ pb: CVPixelBuffer, at seconds: CFTimeInterval) {
        q.async { [weak self] in
            guard let self, let input = self.input, let adaptor = self.adaptor,
                  let writer = self.writer, writer.status == .writing,
                  input.isReadyForMoreMediaData else { return }
            adaptor.append(pb, withPresentationTime: CMTime(seconds: max(0, seconds), preferredTimescale: 600))
        }
    }

    // ── the live per-frame grab (called from each renderer's draw) ─────────
    func grab(drawable: CAMetalDrawable, commandBuffer cmd: MTLCommandBuffer) {
        guard isRecording, !stopping else { return }
        if startTime == 0 { startTime = CACurrentMediaTime() }
        switch mode {
        case .mp4:
            guard let input = input, input.isReadyForMoreMediaData,
                  let pb = pooledBuffer(), let dst = texture(for: pb) else { return }
            blit(drawable.texture, to: dst, on: cmd)
            let t = CACurrentMediaTime() - startTime
            q.sync { pending += 1 }
            cmd.addCompletedHandler { [weak self] _ in
                guard let self else { return }
                self.append(pb, at: t)
                self.q.async {
                    self.pending -= 1
                    if self.stopping && self.pending == 0 { self.finalize() }
                }
            }
        case .gif:
            gifFrameCounter += 1
            guard gifFrameCounter % gifEvery == 0, gifFrames.count < gifMaxFrames else { return }
            // gif path reads back on the CPU: copy the drawable into a managed
            // texture, then build a downscaled CGImage in the completion handler.
            guard let readable = managedCopy(of: drawable.texture, on: cmd) else { return }
            q.sync { pending += 1 }
            cmd.addCompletedHandler { [weak self] _ in
                guard let self else { return }
                if let img = self.cgImage(from: readable, maxDim: self.gifMaxDim) {
                    self.q.async { if self.gifFrames.count < self.gifMaxFrames { self.gifFrames.append(img) } }
                }
                self.q.async {
                    self.pending -= 1
                    if self.stopping && self.pending == 0 { self.finalize() }
                }
            }
        }
    }

    func stop() {
        guard isRecording else { return }
        isRecording = false
        q.async { [weak self] in
            guard let self else { return }
            self.stopping = true
            if self.pending == 0 { self.finalize() }
        }
    }

    // runs on q once all in-flight frames have been consumed
    private func finalize() {
        switch mode {
        case .mp4:
            guard let writer = writer, let input = input else { return }
            input.markAsFinished()
            let url = outURL
            writer.finishWriting { [weak self] in
                DispatchQueue.main.async { self?.onStatus?("■ saved \(url?.lastPathComponent ?? "clip.mp4")") }
            }
            self.writer = nil; self.input = nil; self.adaptor = nil
        case .gif:
            let url = outURL; let frames = gifFrames
            let ok = url != nil && Recorder.writeGIF(frames, to: url!, fps: 30)
            gifFrames.removeAll()
            DispatchQueue.main.async {
                self.onStatus?(ok ? "■ saved \(url?.lastPathComponent ?? "clip.gif") (\(frames.count) frames)"
                                  : "gif: write failed")
            }
        }
        stopping = false
    }

    // ── Metal helpers ──────────────────────────────────────────────────────
    private func blit(_ src: MTLTexture, to dst: MTLTexture, on cmd: MTLCommandBuffer) {
        let w = min(src.width, dst.width), h = min(src.height, dst.height)
        guard let b = cmd.makeBlitCommandEncoder() else { return }
        b.copy(from: src, sourceSlice: 0, sourceLevel: 0,
               sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
               sourceSize: MTLSize(width: w, height: h, depth: 1),
               to: dst, destinationSlice: 0, destinationLevel: 0,
               destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        b.endEncoding()
    }

    private func managedCopy(of src: MTLTexture, on cmd: MTLCommandBuffer) -> MTLTexture? {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm,
                                                         width: src.width, height: src.height, mipmapped: false)
        d.usage = [.shaderRead]; d.storageMode = .managed
        guard let dst = device.makeTexture(descriptor: d) else { return nil }
        blit(src, to: dst, on: cmd)
        if let b = cmd.makeBlitCommandEncoder() { b.synchronize(resource: dst); b.endEncoding() }
        return dst
    }

    private func cgImage(from tex: MTLTexture, maxDim: Int) -> CGImage? {
        let w = tex.width, h = tex.height
        var bgra = [UInt8](repeating: 0, count: w * h * 4)
        tex.getBytes(&bgra, bytesPerRow: w * 4,
                     from: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                                     size: MTLSize(width: w, height: h, depth: 1)), mipmapLevel: 0)
        let cs = CGColorSpaceCreateDeviceRGB()
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue
                                | CGBitmapInfo.byteOrder32Little.rawValue)   // BGRA
        guard let ctx = CGContext(data: &bgra, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: cs, bitmapInfo: info.rawValue),
              let full = ctx.makeImage() else { return nil }
        let scale = min(1.0, Double(maxDim) / Double(max(w, h)))
        if scale >= 1.0 { return full }
        let sw = Int(Double(w) * scale), sh = Int(Double(h) * scale)
        guard let sctx = CGContext(data: nil, width: sw, height: sh, bitsPerComponent: 8,
                                   bytesPerRow: 0, space: cs, bitmapInfo: info.rawValue) else { return full }
        sctx.interpolationQuality = .high
        sctx.draw(full, in: CGRect(x: 0, y: 0, width: sw, height: sh))
        return sctx.makeImage() ?? full
    }

    // ── GIF writer (ImageIO) ───────────────────────────────────────────────
    static func writeGIF(_ frames: [CGImage], to url: URL, fps: Int) -> Bool {
        guard !frames.isEmpty else { return false }
        try? FileManager.default.removeItem(at: url)
        let gifType = "com.compuserve.gif" as CFString
        guard let dst = CGImageDestinationCreateWithURL(url as CFURL, gifType, frames.count, nil) else { return false }
        let fileProps = [kCGImagePropertyGIFDictionary as String: [kCGImagePropertyGIFLoopCount as String: 0]]
        CGImageDestinationSetProperties(dst, fileProps as CFDictionary)
        let delay = 1.0 / Double(max(1, fps))
        let frameProps = [kCGImagePropertyGIFDictionary as String:
                          [kCGImagePropertyGIFUnclampedDelayTime as String: delay]]
        for img in frames { CGImageDestinationAddImage(dst, img, frameProps as CFDictionary) }
        return CGImageDestinationFinalize(dst)
    }

    // ── headless test entry (used by appendForTest in RecTest) ─────────────
    func appendForTest(_ pb: CVPixelBuffer, at s: CFTimeInterval) {
        q.sync {
            guard let input = input, let adaptor = adaptor else { return }
            var spins = 0
            while !input.isReadyForMoreMediaData && spins < 1000 { usleep(500); spins += 1 }
            adaptor.append(pb, withPresentationTime: CMTime(seconds: s, preferredTimescale: 600))
        }
    }
    func finishForTest(_ done: @escaping (Bool) -> Void) {
        q.async { [weak self] in
            guard let self, let writer = self.writer, let input = self.input else { done(false); return }
            input.markAsFinished()
            writer.finishWriting {
                done(writer.status == .completed)
                self.writer = nil; self.input = nil; self.adaptor = nil
            }
        }
    }
}
