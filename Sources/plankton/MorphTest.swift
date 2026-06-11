import Foundation

// Verify the morphology descriptors on synthetic patterns (run via `--morphtest`)
// before trusting them on the sim: a lattice of dots → spots (χ > 0), the inverse
// (a wall network around dark cells) → foam (χ < 0), vertical bands → connected
// (not foam), and a flat field → dispersed (no contrast). The Euler sign is the
// load-bearing claim, so these pin it down.
func runMorphTest() {
    let dim = 256
    var pass = true
    func check(_ name: String, _ ok: Bool, _ m: Morphology) {
        let p = name.padding(toLength: 12, withPad: " ", startingAt: 0)
        print("\(p) \(ok ? "PASS" : "FAIL")  class=\(m.klass)  χ=\(m.euler)  LCC=\(String(format: "%.2f", m.lcc))  CV=\(String(format: "%.2f", m.contrast))")
        if !ok { pass = false }
    }

    // lattice of 3×3 bright dots on a dark field → isolated islands → spots
    var dots = [Float](repeating: 0, count: dim * dim)
    for y in 0..<dim { for x in 0..<dim where (x % 16) < 3 && (y % 16) < 3 { dots[y * dim + x] = 1 } }
    let mDots = Morphology.analyze(dots, dim: dim)
    check("dots", mDots.klass == "spots", mDots)

    // thin bright walls (minority) around large dark cells → a connected network
    // with many holes → foam (matches real dye: filaments enclose voids)
    var foam = [Float](repeating: 0, count: dim * dim)
    for y in 0..<dim { for x in 0..<dim where (x % 16) < 2 || (y % 16) < 2 { foam[y * dim + x] = 1 } }
    let mFoam = Morphology.analyze(foam, dim: dim)
    check("foam", mFoam.klass == "foam", mFoam)

    // a thick square frame → ONE connected component enclosing ONE hole (χ=0,
    // LCC=1): connected but not a closed-cell foam → network
    var frame = [Float](repeating: 0, count: dim * dim)
    for y in 0..<dim { for x in 0..<dim where x < 12 || x >= dim - 12 || y < 12 || y >= dim - 12 { frame[y * dim + x] = 1 } }
    let mFrame = Morphology.analyze(frame, dim: dim)
    check("frame", mFrame.klass == "network", mFrame)

    // flat field (+ negligible noise) → no structure → dispersed
    var flat = [Float](repeating: 1.0, count: dim * dim)
    for i in 0..<(dim * dim) { flat[i] += Float((i % 7)) * 1.0e-4 }
    let mFlat = Morphology.analyze(flat, dim: dim)
    check("flat", mFlat.klass == "dispersed", mFlat)

    // length scale of the dot lattice should be ~ the 16-px period → ~0.0625 domain
    let lenOK = mDots.lengthScale > 0.02 && mDots.lengthScale < 0.18
    check("lengthscale", lenOK, mDots)

    print(pass ? "RESULT       : PASS — morphology descriptors classify the synthetic patterns."
               : "RESULT       : CHECK — a descriptor misclassified a known pattern.")
    if !pass { exit(1) }
}
