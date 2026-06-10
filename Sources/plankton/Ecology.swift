import Foundation

// Replicator-mutator dynamics over the 8 cohort "strategies" (evolutionary game
// theory). p_i is the fraction of the population running cohort brain i; it
// evolves by the replicator equation ṗ_i = p_i (π_i − π̄), where π_i is strategy
// i's payoff against the current mix and π̄ = p·π is the mean. Higher-than-average
// payoff strategies grow; lower ones shrink. Σṗ_i = 0, so p stays on the simplex.
//
// STAGE 1 (here): GLOBAL / well-mixed payoff π = A·p from a payoff matrix A —
// pure dynamics, verified by --ecologytest (no sim coupling). STAGE 2 will make
// the payoff SPATIAL (agents play their neighbours) for pattern formation
// (RPS spiral waves, coexistence polymorphism).
//
// "Mutator": replicator alone can't innovate (no new strategies), so when a
// strategy goes extinct it's reseeded as a mutation of the current leader and
// re-injected at a small frequency — that keeps the ecology alive and exploring.
struct Ecology {
    static let n = 8

    var p: [Double]                 // strategy frequencies, on the simplex (Σ = 1)
    var A: [[Double]]               // n×n payoff matrix (only relative values matter)
    var extinctFloor = 1.0e-3       // below this a strategy is "extinct" → reseed
    var reinjectEps = 1.0e-2        // frequency a reseeded strategy comes back at

    init(p: [Double]? = nil, A: [[Double]]? = nil) {
        self.p = p ?? [Double](repeating: 1.0 / Double(Ecology.n), count: Ecology.n)
        self.A = A ?? Ecology.rps()
    }

    // well-mixed payoff of each strategy against the current mix: π = A·p
    func payoffs() -> [Double] {
        (0..<Ecology.n).map { i in
            (0..<Ecology.n).reduce(0.0) { $0 + A[i][$1] * p[$1] }
        }
    }

    // one replicator step (explicit Euler), returns ṗ (pre-renormalisation) so
    // callers/tests can check simplex invariance Σṗ ≈ 0.
    @discardableResult
    mutating func step(dt: Double) -> [Double] {
        let pi = payoffs()
        let mean = zip(p, pi).reduce(0.0) { $0 + $1.0 * $1.1 }     // π̄ = p·π
        let dp = (0..<Ecology.n).map { p[$0] * (pi[$0] - mean) }
        for i in 0..<Ecology.n { p[i] = max(0, p[i] + dt * dp[i]) }
        normalize()
        return dp
    }

    // replicator-MUTATOR: reseed any extinct strategy as a mutation of the leader.
    // `onReseed(dead, leader)` lets the engine mutate genome `dead` from `leader`;
    // in the pure core it's just bookkeeping. Returns the reseeded indices.
    @discardableResult
    mutating func reseedExtinct(_ onReseed: (Int, Int) -> Void = { _, _ in }) -> [Int] {
        guard let leader = p.indices.max(by: { p[$0] < p[$1] }) else { return [] }
        var reseeded: [Int] = []
        for i in 0..<Ecology.n where i != leader && p[i] < extinctFloor {
            onReseed(i, leader)
            p[i] = reinjectEps
            reseeded.append(i)
        }
        if !reseeded.isEmpty { normalize() }
        return reseeded
    }

    private mutating func normalize() {
        let s = p.reduce(0.0, +)
        if s > 0 { for i in 0..<Ecology.n { p[i] /= s } }
    }

    // ── payoff-matrix presets ──────────────────────────────────────────────
    // Generalised Rock-Paper-Scissors: strategy i beats the next n/2−1 strategies
    // (cyclically) and loses to the previous n/2−1; the antipode and self are ties.
    // The uniform mix p=1/n is a rest point; off-centre starts cycle (neutrally).
    static func rps() -> [[Double]] {
        let n = Ecology.n
        var A = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        for i in 0..<n { for j in 0..<n {
            let d = (j - i + n) % n
            if d == 0 || d == n / 2 { A[i][j] = 0 }
            else if d < n / 2 { A[i][j] = 1 }       // i beats j
            else { A[i][j] = -1 }                    // i loses to j
        }}
        return A
    }

    // Anti-coordination: you score for being DIFFERENT from the rest (rare → high
    // payoff). Drives every start to the interior, all strategies coexisting.
    static func coexistence() -> [[Double]] {
        let n = Ecology.n
        return (0..<n).map { i in (0..<n).map { j in i == j ? 0.0 : 1.0 } }
    }

    // Strict dominance: strategy 0 always pays best → takes over to a vertex.
    static func dominance() -> [[Double]] {
        let n = Ecology.n
        return (0..<n).map { i in (0..<n).map { _ in i == 0 ? 1.0 : 0.0 } }
    }

    enum Preset: String, CaseIterable { case rps, coexistence, dominance }
    static func matrix(_ preset: Preset) -> [[Double]] {
        switch preset {
        case .rps: return rps()
        case .coexistence: return coexistence()
        case .dominance: return dominance()
        }
    }
}
