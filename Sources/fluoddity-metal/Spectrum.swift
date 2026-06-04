import Accelerate

// Energy spectrum E(k) of the 2D velocity field via a complex 2D FFT (vDSP),
// radially binned by wavenumber. On log-log axes a Kolmogorov inertial range is
// a straight line: slope -5/3 for a forward *energy* cascade, -3 for the 2D
// *enstrophy* cascade.
//
// ESTIMATOR NOTE — load-bearing for slope honesty. The field is reduced to the
// n×n FFT grid by *block-averaging* each stride×stride block (a box low-pass),
// NOT by point-decimation. Naked decimation (taking every stride-th cell, the
// previous behaviour) aliases all field energy above the n-Nyquist back into the
// resolved band, where it piles onto the high-k tail and artificially FLATTENS
// the measured slope — it reads shallower (less negative) than the true cascade,
// and does so across almost any parameter setting, which is the tell-tale of an
// estimator artifact rather than physics. The Renderer runs at n == fieldDim
// (stride 1: no reduction, no aliasing) — the trustworthy instrument. The
// block-average path only engages if a caller sets n < fieldDim.
final class Spectrum {
    let n: Int
    private let log2n: vDSP_Length
    private let setup: FFTSetup
    private var vxr: [Float], vxi: [Float], vyr: [Float], vyi: [Float], powr: [Float]
    private var acc: [Float]             // running sum of E(k) for time-averaging
    private(set) var frames = 0          // snapshots accumulated since last reset
    private(set) var ek: [Float]   // time-averaged E(k) for k = 0 ..< n/2

    init(n: Int = 1024) {
        self.n = n
        self.log2n = vDSP_Length(log2(Double(n)).rounded())
        self.setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))!
        self.vxr = [Float](repeating: 0, count: n * n)
        self.vxi = [Float](repeating: 0, count: n * n)
        self.vyr = [Float](repeating: 0, count: n * n)
        self.vyi = [Float](repeating: 0, count: n * n)
        self.powr = [Float](repeating: 0, count: n * n)
        self.acc = [Float](repeating: 0, count: n / 2)
        self.ek = [Float](repeating: 0, count: n / 2)
    }
    deinit { vDSP_destroy_fftsetup(setup) }

    // vel: interleaved (vx, vy) of a dim×dim field. Block-averages stride×stride
    // blocks down to n×n (stride 1 = full resolution, no reduction, no aliasing).
    func compute(_ vel: UnsafePointer<Float>, dim: Int) {
        let stride = max(1, dim / n)
        let inv = 1.0 / Float(stride * stride)
        for j in 0..<n {
            for i in 0..<n {
                var sx: Float = 0, sy: Float = 0
                for bj in 0..<stride {
                    for bi in 0..<stride {
                        let src = (j * stride + bj) * dim + (i * stride + bi)
                        sx += vel[2 * src]
                        sy += vel[2 * src + 1]
                    }
                }
                let o = j * n + i
                vxr[o] = sx * inv
                vyr[o] = sy * inv
            }
        }
        for k in 0..<(n * n) { vxi[k] = 0; vyi[k] = 0; powr[k] = 0 }
        fftPower(&vxr, &vxi, &powr)
        fftPower(&vyr, &vyi, &powr)

        var bins = [Float](repeating: 0, count: n / 2)
        for j in 0..<n {
            let ky = j < n / 2 ? j : j - n
            for i in 0..<n {
                let kx = i < n / 2 ? i : i - n
                let k = Int((Float(kx * kx + ky * ky)).squareRoot().rounded())
                if k >= 1 && k < n / 2 { bins[k] += powr[j * n + i] }
            }
        }
        // time-average: accumulate this snapshot, publish the running mean. A
        // single-frame spectrum fluctuates frame-to-frame, so its slope/R² bounce;
        // the averaged curve is what gives a trustworthy fit. resetAverage()
        // restarts it after any flow change (param edit, reset, reroll, load).
        for k in 0..<(n / 2) { acc[k] += bins[k] }
        frames += 1
        let invF = 1.0 / Float(frames)
        for k in 0..<(n / 2) { bins[k] = acc[k] * invF }
        ek = bins
    }

    func resetAverage() {
        for k in 0..<acc.count { acc[k] = 0 }
        frames = 0
    }

    private func fftPower(_ real: inout [Float], _ imag: inout [Float], _ powr: inout [Float]) {
        let count = n * n
        real.withUnsafeMutableBufferPointer { rp in
            imag.withUnsafeMutableBufferPointer { ip in
                var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                vDSP_fft2d_zip(setup, &split, 1, 0, log2n, log2n,
                               FFTDirection(kFFTDirection_Forward))
                for k in 0..<count { powr[k] += rp[k] * rp[k] + ip[k] * ip[k] }
            }
        }
    }
}
