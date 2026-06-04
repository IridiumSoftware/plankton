#!/usr/bin/env python3
"""
Collapse test for the fluoddity-metal spectrum survey (--sweep output).

Question: the OAT survey showed the inertial slope moves with nearly every knob.
Is there a single control variable onto which ALL 44 slopes collapse — i.e.
slope = f(one number)? If yes, that's a law (and reframes the messy survey into
one curve). If no, the exponent is genuinely multi-parameter.

We reconstruct each row's full parameter context (preset_003 baseline with the
swept axis overridden), build a basket of candidate controls — both intrinsic
(from the emergent state E, Z, peakK) and input-based (drag/drive) — and rank
them by |Pearson r| against inSlope (testing linear-x and log10-x).

stdlib only (no numpy). Writes collapse_table.csv for plotting.
"""
import csv, json, math

BASE = json.load(open("presets/preset_003.json"))["params"]

rows = []
with open("sweep_results.csv") as f:
    for r in csv.DictReader(f):
        E = float(r["E"]); Z = float(r["Z"]); pk = int(r["peakK"])
        inSlope = float(r["inSlope"]); inR2 = float(r["inR2"])
        # full param context: baseline, with the swept axis overridden
        p = dict(BASE)
        axis, val = r["axis"], r["value"]
        if axis in p and val != "":
            p[axis] = float(val)
        rows.append(dict(axis=axis, value=val, E=E, Z=Z, pk=pk,
                         inSlope=inSlope, inR2=inR2, p=p))

# drop degenerate rows (dead flow / non-power-law): keep clean fits with energy
clean = [r for r in rows if r["inR2"] >= 0.88 and r["E"] > 1e-3]
dropped = [r for r in rows if r not in clean]
print(f"rows: {len(rows)} total, {len(clean)} kept, "
      f"{len(dropped)} dropped (inR2<0.88 or E~0): "
      f"{[ (d['axis'],d['value']) for d in dropped ]}\n")

def gamma(r):  # continuous per-frame drag rate
    return -math.log(max(r["p"]["velDamp"], 1e-6))

# candidate control variables: name -> function(row) -> value
candidates = {
    "E (energy)":               lambda r: r["E"],
    "Z (enstrophy)":            lambda r: r["Z"],
    "peakK":                    lambda r: r["pk"],
    "krms = sqrt(Z/E)":         lambda r: math.sqrt(r["Z"]/r["E"]),
    "sep = krms/peakK":         lambda r: math.sqrt(r["Z"]/r["E"])/r["pk"],
    "ReE = E/sqrt(Z)":          lambda r: r["E"]/math.sqrt(r["Z"]),
    "gamma = -ln(velDamp)":     lambda r: gamma(r),
    "Re_drag = sqrt(E)*pk/g":   lambda r: math.sqrt(r["E"])*r["pk"]/gamma(r),
    "Re_drag2 = sqrt(E)/g":     lambda r: math.sqrt(r["E"])/gamma(r),
    "drive = forceGain*swim":   lambda r: r["p"]["forceGain"]*r["p"]["swim"],
    "drive/gamma":              lambda r: r["p"]["forceGain"]*r["p"]["swim"]/gamma(r),
    "E/gamma":                  lambda r: r["E"]/gamma(r),
}

def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs)/n; my = sum(ys)/n
    sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    sxx = sum((x-mx)**2 for x in xs); syy = sum((y-my)**2 for y in ys)
    return sxy/math.sqrt(sxx*syy) if sxx > 0 and syy > 0 else 0.0

ys = [r["inSlope"] for r in clean]
results = []
for name, fn in candidates.items():
    xs = [fn(r) for r in clean]
    r_lin = pearson(xs, ys)
    r_log = pearson([math.log10(x) for x in xs], ys) if all(x > 0 for x in xs) else float("nan")
    best = max([abs(r_lin), abs(r_log) if r_log == r_log else 0.0])
    results.append((name, r_lin, r_log, best))

results.sort(key=lambda t: t[3], reverse=True)
print(f"{'control variable':28s}  r(linear)  r(log10)   |r|best   R^2best")
print("-"*72)
for name, rl, rlog, best in results:
    rlog_s = f"{rlog:+.3f}" if rlog == rlog else "   -  "
    print(f"{name:28s}   {rl:+.3f}    {rlog_s}    {best:.3f}    {best**2:.3f}")

win = results[0]
print(f"\nBEST COLLAPSE: {win[0]}   |r|={win[3]:.3f}  (R^2={win[3]**2:.3f})")
print("interpretation:",
      "strong single-variable collapse — candidate law" if win[3] >= 0.95 else
      "moderate — partial collapse, likely still multi-parameter" if win[3] >= 0.85 else
      "weak — the exponent does NOT collapse onto one control variable")

# ── multivariate input-law test: does slope collapse onto a few INPUT groups? ──
def solve(A, b):                       # Gaussian elimination, small systems
    n = len(b); M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c and M[c][c] != 0:
                f = M[r][c]/M[c][c]
                M[r] = [M[r][j] - f*M[c][j] for j in range(n+1)]
    return [M[i][n]/M[i][i] if M[i][i] != 0 else 0.0 for i in range(n)]

def ols(feats):                        # feats: list of predictor-lists (no intercept)
    X = [[1.0] + f for f in feats]
    k = len(X[0]); n = len(X)
    XtX = [[sum(X[i][a]*X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    Xty = [sum(X[i][a]*ys[i] for i in range(n)) for a in range(k)]
    beta = solve(XtX, Xty)
    yhat = [sum(beta[a]*X[i][a] for a in range(k)) for i in range(n)]
    ybar = sum(ys)/n
    ss_res = sum((ys[i]-yhat[i])**2 for i in range(n))
    ss_tot = sum((y-ybar)**2 for y in ys)
    return beta, (1 - ss_res/ss_tot if ss_tot > 0 else 0.0)

def L(r, key):  return math.log10(r["p"][key])
lg = lambda r: gamma(r)
models = {
    "slope ~ log(drive)":                      lambda r: [math.log10(r["p"]["forceGain"]*r["p"]["swim"])],
    "slope ~ log(drive) + log(gamma)":         lambda r: [math.log10(r["p"]["forceGain"]*r["p"]["swim"]), math.log10(lg(r))],
    "slope ~ log(fG) + log(swim) + log(gamma)":lambda r: [L(r,"forceGain"), L(r,"swim"), math.log10(lg(r))],
    "slope ~ +log(sensorDist) [4 groups]":     lambda r: [L(r,"forceGain"), L(r,"swim"), math.log10(lg(r)), L(r,"sensorDist")],
}
print("\nMULTIVARIATE INPUT-LAW TEST (pure forcing/dissipation inputs):")
print(f"{'model':45s}  R^2")
print("-"*56)
for name, fn in models.items():
    _, r2 = ols([fn(r) for r in clean])
    print(f"{name:45s}  {r2:.3f}")

# dump a plot-ready table with all controls
with open("collapse_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    cols = list(candidates.keys())
    w.writerow(["axis", "value", "E", "Z", "peakK", "inSlope", "inR2"] + cols)
    for r in clean:
        w.writerow([r["axis"], r["value"], r["E"], r["Z"], r["pk"],
                    r["inSlope"], r["inR2"]] + [f"{candidates[c](r):.5g}" for c in cols])
print("\nwrote collapse_table.csv")
