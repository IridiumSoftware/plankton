import AVFoundation
import CoreVideo
import CoreGraphics
import ImageIO
import Metal
import Foundation

// Headless check of the mp4 encoder path (run via `--rectest`). The live blit
// from a drawable needs a window, but the risky part — AVAssetWriter producing a
// valid, playable mp4 — is testable here: feed synthetic gradient frames through
// the exact Recorder encoder, then reopen the file and assert it has a video
// track of the right size and a non-zero duration.
func runRecTest() {
    guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device.") }
    print("Metal device : \(device.name)")
    let rec = Recorder(device: device)

    let W = 256, H = 192, N = 45, fps = 30.0
    let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("plankton_rectest.mp4")
    guard rec.beginVideo(url: url, width: W, height: H) else { print("RESULT       : CHECK — beginVideo failed"); exit(1) }

    // synthetic frames: a diagonal gradient that shifts with the frame index
    for i in 0..<N {
        guard let pb = rec.pooledBuffer() else { print("RESULT       : CHECK — no pooled buffer"); exit(1) }
        CVPixelBufferLockBaseAddress(pb, [])
        if let base = CVPixelBufferGetBaseAddress(pb) {
            let bpr = CVPixelBufferGetBytesPerRow(pb)
            let px = base.assumingMemoryBound(to: UInt8.self)
            for y in 0..<H {
                for x in 0..<W {
                    let o = y * bpr + x * 4
                    px[o + 0] = UInt8((x + i * 4) & 0xFF)         // B
                    px[o + 1] = UInt8((y + i * 2) & 0xFF)         // G
                    px[o + 2] = UInt8((x + y + i * 6) & 0xFF)     // R
                    px[o + 3] = 255                               // A
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(pb, [])
        rec.appendForTest(pb, at: Double(i) / fps)
    }

    let sem = DispatchSemaphore(value: 0)
    var ok = false
    rec.finishForTest { done in ok = done; sem.signal() }
    _ = sem.wait(timeout: .now() + 30)

    guard ok else { print("RESULT       : CHECK — writer did not complete"); exit(1) }

    // reopen and validate
    let asset = AVURLAsset(url: url)
    let dur = CMTimeGetSeconds(asset.duration)
    let tracks = asset.tracks(withMediaType: .video)
    let size = tracks.first?.naturalSize ?? .zero
    let bytes = ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int) ?? 0
    print(String(format: "wrote        : %@ (%d bytes)", url.lastPathComponent, bytes))
    print(String(format: "duration     : %.2f s   tracks: %d   size: %.0f×%.0f",
                 dur, tracks.count, size.width, size.height))

    let good = tracks.count == 1 && dur > 0.5 && Int(size.width) == W && Int(size.height) == H && bytes > 1000
    if !good { print("RESULT       : CHECK — output mp4 failed validation."); exit(1) }
    print("mp4          : PASS — valid \(W)×\(H) clip (\(N) frames).")

    // ── gif encoder (ImageIO) — synthetic CGImages → animated GIF → reopen ──
    let gframes = (0..<12).compactMap { syntheticCGImage(64, 64, $0) }
    let gurl = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("plankton_rectest.gif")
    let gok = Recorder.writeGIF(gframes, to: gurl, fps: 12)
    let gsrc = CGImageSourceCreateWithURL(gurl as CFURL, nil)
    let gcount = gsrc.map { CGImageSourceGetCount($0) } ?? 0
    let gtype = gsrc.flatMap { CGImageSourceGetType($0) as String? } ?? ""
    print("gif          : \(gurl.lastPathComponent)  frames=\(gcount)  type=\(gtype)")
    let ggood = gok && gcount == gframes.count && gtype.contains("gif")
    if !ggood { print("RESULT       : CHECK — gif encoder failed validation."); exit(1) }

    print("RESULT       : PASS — mp4 + gif encoders both write valid clips.")
}

private func syntheticCGImage(_ w: Int, _ h: Int, _ i: Int) -> CGImage? {
    var bgra = [UInt8](repeating: 0, count: w * h * 4)
    for y in 0..<h { for x in 0..<w {
        let o = (y * w + x) * 4
        bgra[o + 0] = UInt8((x + i * 8) & 0xFF)
        bgra[o + 1] = UInt8((y + i * 4) & 0xFF)
        bgra[o + 2] = UInt8((x + y + i * 12) & 0xFF)
        bgra[o + 3] = 255
    }}
    let cs = CGColorSpaceCreateDeviceRGB()
    let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
    return bgra.withUnsafeMutableBytes { raw in
        CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8,
                  bytesPerRow: w * 4, space: cs, bitmapInfo: info.rawValue)?.makeImage()
    }
}
