import Accelerate

// Energy spectrum E(k) of the 2D velocity field via a complex 2D FFT (vDSP),
// radially binned by wavenumber. The cascade diagnostic: on log-log axes a
// Kolmogorov inertial range is a straight line of slope -5/3.
//
// The field is strided-downsampled to n×n (n a power of two) before the FFT.
final class Spectrum {
    let n: Int
    private let log2n: vDSP_Length
    private let setup: FFTSetup
    private(set) var ek: [Float]   // E(k) for k = 0 ..< n/2

    init(n: Int = 256) {
        self.n = n
        self.log2n = vDSP_Length(log2(Double(n)).rounded())
        self.setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))!
        self.ek = [Float](repeating: 0, count: n / 2)
    }
    deinit { vDSP_destroy_fftsetup(setup) }

    // vel: interleaved (vx, vy) of a dim×dim field. Downsamples to n×n.
    func compute(_ vel: UnsafePointer<Float>, dim: Int) {
        let stride = max(1, dim / n)
        var vxr = [Float](repeating: 0, count: n * n), vxi = [Float](repeating: 0, count: n * n)
        var vyr = [Float](repeating: 0, count: n * n), vyi = [Float](repeating: 0, count: n * n)
        for j in 0..<n {
            for i in 0..<n {
                let src = (j * stride) * dim + (i * stride)
                vxr[j * n + i] = vel[2 * src]
                vyr[j * n + i] = vel[2 * src + 1]
            }
        }
        var powr = [Float](repeating: 0, count: n * n)
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
        ek = bins
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
