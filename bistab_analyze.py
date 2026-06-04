#!/usr/bin/env python3
"""
Analyze the --bistab multistability probe. For each sensorDist, look at the
spread of steady-state peakK across independent replicates (fresh IC each, same
brain). Multimodal (>=2 populated, well-separated clusters) = multistability;
a tight single cluster = noise / single-valued dependence.
Writes figures/fig7_bistability.png. Run: .venv/bin/python bistab_analyze.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, csv
from collections import defaultdict

PK = defaultdict(list); SL = defaultdict(list)
for r in csv.DictReader(open("bistab_results.csv")):
    PK[float(r["sensorDist"])].append(int(r["peakK"]))
    SL[float(r["sensorDist"])].append(float(r["inSlope"]))
sds = sorted(PK.keys())

def clusters(vals, gap=3):
    """Group sorted values; a jump > `gap` between consecutive values starts a
    new cluster. peakK are integers, so gap=3 separates distinct modes."""
    v = sorted(vals)
    groups = [[v[0]]]
    for x in v[1:]:
        if x - groups[-1][-1] <= gap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups

print(f"{'sensorDist':11s} {'n':>2s}  {'peakK (sorted)':32s} {'mean':>5s} {'std':>5s}  clusters")
print("-"*86)
verdict = []
for sd in sds:
    pk = PK[sd]; cl = clusters(pk)
    populated = [g for g in cl if len(g) >= 2]          # ignore lone outliers
    multi = len(populated) >= 2
    verdict.append((sd, multi))
    desc = ", ".join(f"k~{np.mean(g):.0f}×{len(g)}" for g in cl)
    print(f"{sd:<11g} {len(pk):>2d}  {str(sorted(pk)):32s} {np.mean(pk):>5.1f} {np.std(pk):>5.1f}  "
          f"{len(cl)} ({desc})  {'<-- MULTISTABLE' if multi else ''}")

ms = [f"{sd:g}" for sd, m in verdict if m]
print(f"\nVERDICT: multistable sensorDist (>=2 populated peakK clusters): "
      f"{', '.join(ms) if ms else 'NONE -- spread is unimodal (noise / single-valued)'}")

# ── Fig 7: per-replicate strip plots ────────────────────────────────────────
rng = np.random.default_rng(0)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for sd in sds:
    j = np.exp(rng.uniform(-0.05, 0.05, len(PK[sd])))
    ax[0].scatter(np.full(len(PK[sd]), sd) * j, PK[sd], s=34, alpha=0.7,
                  edgecolor="k", linewidth=0.3, color="teal")
ax[0].set_xscale("log"); ax[0].set_xlabel("sensorDist"); ax[0].set_ylabel("steady-state peakK")
ax[0].set_title("peakK across independent replicates (fresh IC)")
for sd in sds:
    j = np.exp(rng.uniform(-0.05, 0.05, len(SL[sd])))
    ax[1].scatter(np.full(len(SL[sd]), sd) * j, SL[sd], s=34, alpha=0.7,
                  edgecolor="k", linewidth=0.3, color="darkgreen")
ax[1].axhline(-5/3, color="red", ls=":", lw=1, label="-5/3")
ax[1].set_xscale("log"); ax[1].set_xlabel("sensorDist"); ax[1].set_ylabel("inertial slope")
ax[1].set_title("inertial slope across replicates"); ax[1].legend()
plt.tight_layout(); plt.savefig("figures/fig7_bistability.png", dpi=140); plt.close()
print("wrote figures/fig7_bistability.png")
