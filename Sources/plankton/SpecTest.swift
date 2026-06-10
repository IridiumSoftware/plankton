import Foundation

// Headless verification of the FFT energy spectrum (run via `--spectest`).
// Feeds a synthetic field vx = cos(2π·k0·x/n) (a pure wave at wavenumber k0)
// and checks the radial E(k) peaks at bin k0 — i.e. the FFT + radial binning
// are correct, before the live plot is built on top.
func runSpecTest() {
    let n = 64, k0 = 8
    let spec = Spectrum(n: n)
    var vel = [Float](repeating: 0, count: n * n * 2)
    for j in 0..<n {
        for i in 0..<n {
            let idx = j * n + i
            vel[2 * idx] = cos(2 * .pi * Float(k0) * Float(i) / Float(n))   // vx wave along x
            vel[2 * idx + 1] = 0
        }
    }
    vel.withUnsafeBufferPointer { p in spec.compute(p.baseAddress!, dim: n) }

    var peak = 0
    var peakV: Float = 0
    for k in 1..<spec.ek.count where spec.ek[k] > peakV { peakV = spec.ek[k]; peak = k }

    print("Spectrum     : peak at k=\(peak)  (expected \(k0))")
    if abs(peak - k0) <= 1 {
        print("RESULT       : PASS — FFT spectrum peaks at the right wavenumber.")
    } else {
        print("RESULT       : CHECK — peak at \(peak), expected \(k0).")
        exit(1)
    }
}
