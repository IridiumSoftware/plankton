import Foundation

// Morphology descriptors of a 2D scalar field (the dye), for mapping the engine's
// structure regimes. The key discriminator is the Euler characteristic χ of the
// bright (above-threshold) set, computed by 2×2 bit-quad counting (a Minkowski
// functional): isolated bright islands give χ > 0 (spots), a connected wall
// network enclosing dark cells gives χ < 0 (foam), and a balanced labyrinth gives
// χ ≈ 0. Coefficient-of-variation contrast separates "structured" from "dispersed";
// a radial-autocorrelation first-zero gives the characteristic length scale.
struct Morphology {
    let mean: Double
    let contrast: Double        // std/mean (coefficient of variation) — low ⇒ dispersed
    let coverage: Double        // fraction of cells above threshold (bright area)
    let euler: Int              // Euler characteristic of the bright set
    let eulerDensity: Double    // χ per 10⁴ cells (size-independent sign/magnitude)
    let lcc: Double             // fraction of the bright set in its largest connected component
    let lengthScale: Double     // characteristic spacing, in domain units (0..1)
    let klass: String           // dispersed | spots | network | foam

    // thresholds (documented, tunable): bright = dye > mean + K·std
    static let brightK = 0.5
    static let dispersedContrast = 0.15   // below this CV ⇒ no real structure
    static let connectedLCC = 0.35        // ≥ this fraction in one component ⇒ a connected network

    static func analyze(_ dye: [Float], dim: Int) -> Morphology {
        let n = dim * dim
        // mean / std
        var s = 0.0, s2 = 0.0
        for i in 0..<n { let v = Double(dye[i]); s += v; s2 += v * v }
        let mean = s / Double(n)
        let varr = max(0, s2 / Double(n) - mean * mean)
        let std = varr.squareRoot()
        let contrast = mean > 1e-12 ? std / mean : 0

        // binary bright set
        let thr = mean + brightK * std
        var b = [UInt8](repeating: 0, count: n)
        var bright = 0
        for i in 0..<n where Double(dye[i]) > thr { b[i] = 1; bright += 1 }
        let coverage = Double(bright) / Double(n)

        let euler = Morphology.eulerChar(b, dim: dim)
        let eulerDensity = Double(euler) / Double(n) * 1.0e4
        let lcc = Morphology.largestComponentFraction(b, dim: dim, brightCount: bright)
        let lengthScale = Morphology.lengthScale(dye, dim: dim, mean: mean, varr: varr)

        // classify: low contrast ⇒ dispersed; else connectivity (largest component
        // fraction) separates a connected network from discrete islands, and the
        // Euler sign tells closed-cell foam (χ<0) from an open network/labyrinth.
        let klass: String
        if contrast < dispersedContrast {
            klass = "dispersed"
        } else if lcc >= connectedLCC {
            klass = euler < 0 ? "foam" : "network"     // connected; closed cells vs open
        } else {
            klass = "spots"                            // discrete aggregates
        }
        return Morphology(mean: mean, contrast: contrast, coverage: coverage, euler: euler,
                          eulerDensity: eulerDensity, lcc: lcc, lengthScale: lengthScale, klass: klass)
    }

    // Euler number of a binary image (8-connected foreground), by bit-quad counting:
    // χ = (n₁ − n₃ + 2·n_D)/4, where over every 2×2 window n₁ = #windows with one set
    // pixel, n₃ = three set, n_D = two set on a diagonal. Windows run over a 1-pixel
    // background border so components touching the edge are counted.
    static func eulerChar(_ b: [UInt8], dim: Int) -> Int {
        func at(_ x: Int, _ y: Int) -> Int {
            (x < 0 || y < 0 || x >= dim || y >= dim) ? 0 : Int(b[y * dim + x])
        }
        var n1 = 0, n3 = 0, nD = 0
        for y in -1..<dim {
            for x in -1..<dim {
                let tl = at(x, y), tr = at(x + 1, y), bl = at(x, y + 1), br = at(x + 1, y + 1)
                let sum = tl + tr + bl + br
                if sum == 1 { n1 += 1 }
                else if sum == 3 { n3 += 1 }
                else if sum == 2 && ((tl == 1 && br == 1) || (tr == 1 && bl == 1)) { nD += 1 }
            }
        }
        return (n1 - n3 + 2 * nD) / 4
    }

    // Downsample a dye field to out×out by block-averaging, then normalise per-cell
    // to 0..255 (so each thumbnail shows its own structure regardless of brightness).
    static func thumbnail(_ dye: [Float], dim: Int, out: Int) -> [UInt8] {
        var small = [Float](repeating: 0, count: out * out)
        let bs = max(1, dim / out)
        for oy in 0..<out {
            for ox in 0..<out {
                var acc = 0.0; var cnt = 0
                for yy in 0..<bs { for xx in 0..<bs {
                    let x = ox * bs + xx, y = oy * bs + yy
                    if x < dim && y < dim { acc += Double(dye[y * dim + x]); cnt += 1 }
                }}
                small[oy * out + ox] = cnt > 0 ? Float(acc / Double(cnt)) : 0
            }
        }
        let lo = small.min() ?? 0, hi = small.max() ?? 1
        let span = max(1e-9, hi - lo)
        return small.map { UInt8(max(0, min(255, ($0 - lo) / span * 255))) }
    }

    // Fraction of the bright set contained in its single largest 8-connected
    // component (union-find). Near 1 ⇒ one connected network (foam / labyrinth);
    // small ⇒ many discrete islands (spots). Distinguishes connectivity, which the
    // Euler sign alone can't (open foam and discrete blobs are both χ > 0).
    static func largestComponentFraction(_ b: [UInt8], dim: Int, brightCount: Int) -> Double {
        guard brightCount > 0 else { return 0 }
        let n = dim * dim
        var parent = [Int](0..<n)
        func find(_ x: Int) -> Int { var r = x; while parent[r] != r { parent[r] = parent[parent[r]]; r = parent[r] }; return r }
        func union(_ a: Int, _ c: Int) { let ra = find(a), rc = find(c); if ra != rc { parent[ra] = rc } }
        for y in 0..<dim {
            for x in 0..<dim where b[y * dim + x] == 1 {
                let i = y * dim + x
                if x + 1 < dim && b[i + 1] == 1 { union(i, i + 1) }
                if y + 1 < dim && b[i + dim] == 1 { union(i, i + dim) }
                if x + 1 < dim && y + 1 < dim && b[i + dim + 1] == 1 { union(i, i + dim + 1) }
                if x > 0 && y + 1 < dim && b[i + dim - 1] == 1 { union(i, i + dim - 1) }
            }
        }
        var sizes = [Int: Int](); var maxS = 0
        for i in 0..<n where b[i] == 1 { let r = find(i); let s = (sizes[r] ?? 0) + 1; sizes[r] = s; if s > maxS { maxS = s } }
        return Double(maxS) / Double(brightCount)
    }

    // Characteristic spacing from the radial autocorrelation g(r): the first zero
    // crossing r₀ marks half a wavelength, so length ≈ 2·r₀. Computed along x and y
    // (subsampled) and averaged; returned in domain units (length/dim).
    static func lengthScale(_ dye: [Float], dim: Int, mean: Double, varr: Double) -> Double {
        guard varr > 1e-12 else { return 0 }
        let L = max(4, dim / 4)
        let stepRow = max(1, dim / 128)        // subsample rows/cols for speed
        func g(_ lag: Int) -> Double {
            var acc = 0.0, cnt = 0
            var y = 0
            while y < dim {                     // horizontal pairs
                let row = y * dim
                for x in 0..<(dim - lag) {
                    acc += (Double(dye[row + x]) - mean) * (Double(dye[row + x + lag]) - mean); cnt += 1
                }
                y += stepRow
            }
            var x = 0
            while x < dim {                     // vertical pairs
                for yy in 0..<(dim - lag) {
                    acc += (Double(dye[yy * dim + x]) - mean) * (Double(dye[(yy + lag) * dim + x]) - mean); cnt += 1
                }
                x += stepRow
            }
            return cnt > 0 ? acc / Double(cnt) / varr : 0
        }
        var r0 = 0.0
        var prev = 1.0
        for lag in 1...L {
            let cur = g(lag)
            if cur <= 0 { r0 = Double(lag - 1) + prev / (prev - cur); break }   // linear-interp the crossing
            prev = cur
        }
        return r0 > 0 ? 2.0 * r0 / Double(dim) : 0
    }
}
