#!/usr/bin/env python3
"""
Collapse test for the plankton spectrum study.

  python3 analyze_collapse.py            # OAT survey   (sweep_results.csv, 44 sparse pts)
  python3 analyze_collapse.py map        # 2D map       (map_results.csv, 36 joint cells)

Question: is there a single control variable onto which the inertial slope
collapses — slope = f(one number)? We reconstruct each row's full parameter
context, build a basket of candidate controls (intrinsic state E/Z/peakK and
input-based drive/drag), and rank by |Pearson r| against inSlope. The MAP is the
proper substrate — its forceGain×velDamp grid samples the joint space, unlike the
OAT excursions which confound. stdlib only. Writes collapse_table_<src>.csv.
"""
import csv, json, math, sys

src = sys.argv[1] if len(sys.argv) > 1 else "sweep"
BASE = json.load(open("presets/preset_003.json"))["params"]

rows = []
if src in ("map", "map3"):
    path = f"data/{src}_results.csv"
    for r in csv.DictReader(open(path)):
        p = dict(BASE)
        p["forceGain"] = float(r["forceGain"]); p["velDamp"] = float(r["velDamp"])
        if "sensorDist" in r:
            p["sensorDist"] = float(r["sensorDist"])
        lbl = f"fG={r['forceGain']},vD={r['velDamp']}" + (f",sD={r['sensorDist']}" if "sensorDist" in r else "")
        rows.append(dict(label=lbl, E=float(r["E"]), Z=float(r["Z"]), pk=int(r["peakK"]),
                         inSlope=float(r["inSlope"]), inR2=float(r["inR2"]), p=p))
else:
    path = "data/sweep_results.csv"
    for r in csv.DictReader(open(path)):
        p = dict(BASE)
        if r["axis"] in p and r["value"] != "":
            p[r["axis"]] = float(r["value"])
        rows.append(dict(label=f"{r['axis']}={r['value']}",
                         E=float(r["E"]), Z=float(r["Z"]), pk=int(r["peakK"]),
                         inSlope=float(r["inSlope"]), inR2=float(r["inR2"]), p=p))

clean = [r for r in rows if r["inR2"] >= 0.88 and r["E"] > 1e-3]
dropped = [r for r in rows if r not in clean]
print(f"=== COLLAPSE STRESS-TEST on {src.upper()} ({path}) ===")
print(f"rows: {len(rows)} total, {len(clean)} kept, {len(dropped)} dropped "
      f"{[d['label'] for d in dropped]}\n")

def gamma(r): return -math.log(max(r["p"]["velDamp"], 1e-6))

candidates = {
    "E (energy)":              lambda r: r["E"],
    "Z (enstrophy)":           lambda r: r["Z"],
    "peakK":                   lambda r: r["pk"],
    "krms = sqrt(Z/E)":        lambda r: math.sqrt(r["Z"]/r["E"]),
    "sep = krms/peakK":        lambda r: math.sqrt(r["Z"]/r["E"])/r["pk"],
    "ReE = E/sqrt(Z)":         lambda r: r["E"]/math.sqrt(r["Z"]),
    "gamma = -ln(velDamp)":    lambda r: gamma(r),
    "forceGain":               lambda r: r["p"]["forceGain"],
    "Re_drag = sqrt(E)*pk/g":  lambda r: math.sqrt(r["E"])*r["pk"]/gamma(r),
    "Re_drag2 = sqrt(E)/g":    lambda r: math.sqrt(r["E"])/gamma(r),
    "E/gamma":                 lambda r: r["E"]/gamma(r),
    "fG * gamma^0.38 (2grp)":  lambda r: r["p"]["forceGain"]*gamma(r)**0.379,
}

def pearson(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    sxx = sum((x-mx)**2 for x in xs); syy = sum((y-my)**2 for y in ys)
    return sxy/math.sqrt(sxx*syy) if sxx > 0 and syy > 0 else 0.0

ys = [r["inSlope"] for r in clean]
results = []
for name, fn in candidates.items():
    xs = [fn(r) for r in clean]
    r_lin = pearson(xs, ys)
    r_log = pearson([math.log10(x) for x in xs], ys) if all(x > 0 for x in xs) else float("nan")
    best = max(abs(r_lin), abs(r_log) if r_log == r_log else 0.0)
    results.append((name, r_lin, r_log, best))
results.sort(key=lambda t: t[3], reverse=True)

print(f"{'control variable':26s}  r(lin)   r(log)   |r|best  R^2best")
print("-"*64)
for name, rl, rlog, best in results:
    rlog_s = f"{rlog:+.3f}" if rlog == rlog else "  -   "
    print(f"{name:26s}  {rl:+.3f}  {rlog_s}   {best:.3f}   {best**2:.3f}")

# multivariate input-law fit (drop constant predictors → robust on the map)
def solve(A, b):
    n = len(b); M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c and M[c][c] != 0:
                f = M[r][c]/M[c][c]; M[r] = [M[r][j]-f*M[c][j] for j in range(n+1)]
    return [M[i][n]/M[i][i] if M[i][i] != 0 else 0.0 for i in range(n)]

def ols(feats):
    keep = [j for j in range(len(feats[0])) if len(set(round(f[j], 9) for f in feats)) > 1]
    X = [[1.0] + [f[j] for j in keep] for f in feats]
    k = len(X[0]); n = len(X)
    XtX = [[sum(X[i][a]*X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    Xty = [sum(X[i][a]*ys[i] for i in range(n)) for a in range(k)]
    beta = solve(XtX, Xty)
    yhat = [sum(beta[a]*X[i][a] for a in range(k)) for i in range(n)]
    ybar = sum(ys)/n
    ss = sum((ys[i]-yhat[i])**2 for i in range(n)); tot = sum((y-ybar)**2 for y in ys)
    return (1 - ss/tot) if tot > 0 else 0.0, len(keep)

models = {
    "slope ~ log(forceGain)":               lambda r: [math.log10(r["p"]["forceGain"])],
    "slope ~ log(forceGain)+log(gamma)":    lambda r: [math.log10(r["p"]["forceGain"]), math.log10(gamma(r))],
    "slope ~ log(fG)+log(gamma)+log(sensorDist)": lambda r: [math.log10(r["p"]["forceGain"]), math.log10(gamma(r)), math.log10(r["p"]["sensorDist"])],
    "slope ~ log(fG)+log(gamma)+log(senseScale)": lambda r: [math.log10(r["p"]["forceGain"]), math.log10(gamma(r)), math.log10(r["p"]["senseScale"])],
}
print("\nMULTIVARIATE LAW:")
for name, fn in models.items():
    r2, kk = ols([fn(r) for r in clean])
    print(f"  {name:46s}  R^2={r2:.3f}  ({kk} live groups)")

with open(f"data/collapse_table_{src}.csv", "w", newline="") as f:
    w = csv.writer(f); cols = list(candidates.keys())
    w.writerow(["label", "E", "Z", "peakK", "inSlope", "inR2"] + cols)
    for r in clean:
        w.writerow([r["label"], r["E"], r["Z"], r["pk"], r["inSlope"], r["inR2"]]
                   + [f"{candidates[c](r):.5g}" for c in cols])
print(f"\nwrote data/collapse_table_{src}.csv")
