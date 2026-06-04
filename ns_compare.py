#!/usr/bin/env python3
"""
Compare the real 2D NS DNS (ns2d_dns.py) to the fluoddity-metal engine.
Two contrasts (fig8):
  (L) spectral shape: DNS E(k) vs a representative fluoddity E(k), with -5/3 / -3 refs.
  (R) UNIVERSALITY: inertial slope vs forcing amplitude. Real 2D NS = flat
      (amplitude-independent); fluoddity = sloped (a dial). This is the calibration.
Run: .venv/bin/python ns_compare.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, csv, os

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))

# ── (L) spectral-shape overlay ──────────────────────────────────────────────
if os.path.exists("ns2d_spectrum.csv"):
    D = [(int(r["k"]), float(r["Ek"])) for r in csv.DictReader(open("ns2d_spectrum.csv"))]
    dk = np.array([k for k, _ in D]); de = np.array([e for _, e in D])
    de = de / de.max()
    ax[0].loglog(dk, de, color="crimson", lw=1.6, label="real 2D NS DNS")
# representative fluoddity spectrum (sdscan point nearest the preset_003 baseline sensorDist≈0.06)
if os.path.exists("sdscan_spectra.csv"):
    SP = {}
    for r in csv.DictReader(open("sdscan_spectra.csv")):
        SP.setdefault(float(r["sensorDist"]), ([], []))
        SP[float(r["sensorDist"])][0].append(int(r["k"])); SP[float(r["sensorDist"])][1].append(float(r["Ek"]))
    sd0 = min(SP, key=lambda s: abs(s - 0.06))
    fk = np.array(SP[sd0][0]); fe = np.array(SP[sd0][1]); fe = fe / fe.max()
    ax[0].loglog(fk, fe, color="teal", lw=1.6, label=f"fluoddity (sd={sd0:.3g})")
for sl, lbl, x0 in [(-5/3, "-5/3", 8), (-3, "-3", 8)]:
    xs = np.array([x0, 80])
    ax[0].loglog(xs, 0.5 * (xs / x0)**sl, "k--", lw=0.8, alpha=0.6)
    ax[0].text(82, 0.5 * (80 / x0)**sl, lbl, fontsize=8, va="center")
ax[0].set_xlabel("k"); ax[0].set_ylabel("E(k) (normalized)")
ax[0].set_title("Spectral shape: real 2D NS vs the toy"); ax[0].legend(fontsize=9)

# ── (R) universality: slope vs forcing amplitude ────────────────────────────
def norm(x): x = np.asarray(x, float); return x / np.median(x)
if os.path.exists("ns2d_sweep.csv"):
    rows = list(csv.DictReader(open("ns2d_sweep.csv")))
    amp = np.array([float(r["amp"]) for r in rows]); ens = np.array([float(r["enstSlope"]) for r in rows])
    ax[1].plot(norm(amp), ens, "o-", color="crimson", lw=1.8,
               label=f"real 2D NS (enstrophy range)  Δ={ens.max()-ens.min():.2f}")
# fluoddity: forceGain axis from the OAT survey
if os.path.exists("sweep_results.csv"):
    fg = [(float(r["value"]), float(r["inSlope"])) for r in csv.DictReader(open("sweep_results.csv"))
          if r["axis"] == "forceGain"]
    fg.sort()
    fa = np.array([a for a, _ in fg]); fs = np.array([s for _, s in fg])
    ax[1].plot(norm(fa), fs, "s-", color="teal", lw=1.8,
               label=f"fluoddity (dial)  Δ={fs.max()-fs.min():.2f}")
ax[1].axhline(-3, color="purple", ls="--", lw=1, alpha=0.7, label="-3 (2D enstrophy, universal)")
ax[1].axhline(-5/3, color="gray", ls=":", lw=1, label="-5/3")
ax[1].set_xscale("log")
ax[1].set_xlabel("forcing amplitude (normalized to each model's median)")
ax[1].set_ylabel("inertial slope")
ax[1].set_title("Real 2D NS stays steep (→ -3 as Re grows); the toy dials shallow")
ax[1].legend(fontsize=8)

plt.tight_layout(); plt.savefig("figures/fig8_dns_compare.png", dpi=140); plt.close()
print("wrote figures/fig8_dns_compare.png")
