import Foundation

// Shared log-log least-squares slope fits over the radial energy spectrum E(k).
// Used by BOTH the live SpectrumView readout and the headless --sweep harness so
// the reported numbers are guaranteed identical (single source of truth).
//
//   peakK   : the spectrum's peak bin — the forcing / energy-containing scale.
//   high*   : fit over the full right limb [peakK, kmax]. Includes the
//             energy-containing shoulder and the dissipation tail, so it's a
//             blended slope, not a pure inertial exponent.
//   in*     : trimmed inertial-range fit over [inLo, inHi] = [3·peak, kmax/2],
//             dropping the shoulder and the dissipation tail. This is the slope
//             to watch under a velDamp sweep: if it's damping-INSENSITIVE it's a
//             genuine inertial range; if it tracks velDamp it's forced-dissipative.
//   R²      : straightness of the fit (a real power law sits at ~0.97+).
struct SpectrumFit {
    let peakK: Int
    let highSlope: Float, highR2: Float, highBins: Int
    let inLo: Int, inHi: Int, inSlope: Float, inR2: Float

    static func compute(_ ek: [Float]) -> SpectrumFit {
        let kmax = ek.count - 1
        guard kmax >= 4 else {
            return SpectrumFit(peakK: 0, highSlope: 0, highR2: 0, highBins: 0,
                               inLo: 0, inHi: 0, inSlope: 0, inR2: 0)
        }
        var peak = 1
        var pv: Float = 0
        for k in 1...kmax where ek[k] > pv { pv = ek[k]; peak = k }

        func fit(_ a: Int, _ b: Int) -> (m: Float, r2: Float, n: Int) {
            guard b > a + 1 else { return (0, 0, 0) }
            var sx: Float = 0, sy: Float = 0, sxx: Float = 0, syy: Float = 0, sxy: Float = 0, n: Float = 0
            for k in a...b where ek[k] > 0 {
                let x = log10(Float(k)), y = log10(ek[k])
                sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y; n += 1
            }
            let den = n * sxx - sx * sx
            guard n >= 2, abs(den) > 1e-9 else { return (0, 0, Int(n)) }
            let cov = n * sxy - sx * sy
            let vy = n * syy - sy * sy
            let r2 = vy > 1e-12 ? (cov * cov) / (den * vy) : 0
            return (cov / den, r2, Int(n))
        }

        let hi = fit(peak, kmax)
        let kLo = min(kmax - 2, max(3, peak * 3))
        let kHi = max(kLo + 2, kmax / 2)
        let inr = fit(kLo, kHi)
        return SpectrumFit(peakK: peak,
                           highSlope: hi.m, highR2: hi.r2, highBins: hi.n,
                           inLo: kLo, inHi: kHi, inSlope: inr.m, inR2: inr.r2)
    }
}
