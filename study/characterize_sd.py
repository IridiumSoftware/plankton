#!/usr/bin/env python3
"""
Characterize the sensorDist (injection-scale) axis from the --sdscan output.
Reads sdscan_summary.csv + sdscan_spectra.csv; writes figures/fig5_sensordist.png
(4-panel) and figures/fig6_sd_spectra.png (spectral-shape overlay), and prints the
resonance test. Run with: .venv/bin/python study/characterize_sd.py
"""
import os as _os; _os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # run from anywhere: paths resolve against the repo root
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, csv

DIM = 1024  # field grid; sensorDist is normalized [0,1], lambda is in grid cells

S = [ {k: float(v) for k, v in r.items()} for r in csv.DictReader(open("data/sdscan_summary.csv")) ]
sd  = np.array([r["sensorDist"] for r in S])
pk  = np.array([r["peakK"] for r in S])
sl  = np.array([r["inSlope"] for r in S])
E   = np.array([r["E"] for r in S]); Z = np.array([r["Z"] for r in S])
lam = np.array([r["lambda"] for r in S])      # Taylor microscale, in grid cells
lam_n = lam / DIM                              # in domain units (to compare with sensorDist)
inv_pk = np.where(pk > 0, 1.0 / pk, np.nan)    # injection wavelength, domain units

imax = int(np.argmax(pk))
sd_opt = sd[imax]
print(f"peak-k MAXIMUM: peakK = {pk[imax]:.0f}  at sensorDist = {sd_opt:.4g}")
print(f"  there: lambda = {lam[imax]:.1f} cells (= {lam_n[imax]:.4g} domain), 1/peakK = {inv_pk[imax]:.4g} domain")
print(f"  sensorDist / lambda_norm = {sd_opt/lam_n[imax]:.2f}")
print(f"  sensorDist * peakK       = {sd_opt*pk[imax]:.2f}   (sensorDist in units of injection wavelength)")

# where does sensorDist cross the Taylor microscale (self-consistent resonance)?
diff = sd - lam_n
cross = None
for i in range(len(sd)-1):
    if diff[i] == 0 or diff[i]*diff[i+1] < 0:
        t = diff[i]/(diff[i]-diff[i+1])
        cross = sd[i]*(sd[i+1]/sd[i])**t
        break
print(f"sensorDist == lambda_norm crossing: sd ~ {cross}"
      + (f"  (vs hump at {sd_opt:.4g})" if cross else ""))

# ── Fig 5: 4-panel characterization ─────────────────────────────────────────
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax[0,0].plot(sd, pk, "o-", color="teal")
ax[0,0].axvline(sd_opt, color="red", ls=":", label=f"peakK max @ {sd_opt:.3g}")
ax[0,0].set_xscale("log"); ax[0,0].set_xlabel("sensorDist"); ax[0,0].set_ylabel("peakK")
ax[0,0].set_title("Injection wavenumber vs sensing distance"); ax[0,0].legend()

ax[0,1].plot(sd, sd, "k--", lw=0.8, label="sensorDist (identity)")
ax[0,1].plot(sd, lam_n, "o-", color="purple", label="λ/dim (Taylor microscale)")
ax[0,1].plot(sd, inv_pk, "s-", color="orange", label="1/peakK (injection λ)")
ax[0,1].axvline(sd_opt, color="red", ls=":")
ax[0,1].set_xscale("log"); ax[0,1].set_yscale("log")
ax[0,1].set_xlabel("sensorDist"); ax[0,1].set_ylabel("length (domain units)")
ax[0,1].set_title("Resonance test: sensing distance vs fluid scales"); ax[0,1].legend(fontsize=8)

ax[1,0].plot(sd, sl, "o-", color="darkgreen")
ax[1,0].axhline(-5/3, color="red", ls=":", label="-5/3")
ax[1,0].axvline(sd_opt, color="red", ls=":")
ax[1,0].set_xscale("log"); ax[1,0].set_xlabel("sensorDist"); ax[1,0].set_ylabel("inertial slope")
ax[1,0].set_title("Slope vs sensing distance"); ax[1,0].legend()

a2 = ax[1,1]; a2.plot(sd, E, "o-", color="blue", label="E")
a2.set_xscale("log"); a2.set_xlabel("sensorDist"); a2.set_ylabel("E (energy)", color="blue")
ab = a2.twinx(); ab.plot(sd, Z, "s-", color="crimson", label="Z"); ab.set_ylabel("Z (enstrophy)", color="crimson")
a2.axvline(sd_opt, color="red", ls=":"); a2.set_title("Energy / enstrophy vs sensing distance")
plt.tight_layout(); plt.savefig("figures/fig5_sensordist.png", dpi=140); plt.close()

# ── Fig 6: spectral-shape overlay ───────────────────────────────────────────
SP = {}
for r in csv.DictReader(open("data/sdscan_spectra.csv")):
    s = float(r["sensorDist"]); SP.setdefault(s, ([], []))
    SP[s][0].append(int(r["k"])); SP[s][1].append(float(r["Ek"]))
sds = sorted(SP.keys())
idx = np.unique(np.linspace(0, len(sds)-1, 7).astype(int))
fig, ax = plt.subplots(figsize=(7, 5))
cmap = plt.cm.viridis
for j, ii in enumerate(idx):
    s = sds[ii]; ks, eks = SP[s]
    ax.loglog(ks, eks, color=cmap(j/max(1, len(idx)-1)), lw=1.2, label=f"sd={s:.3g}")
ax.set_xlabel("k"); ax.set_ylabel("E(k)")
ax.set_title("Spectral shape across the sensing-distance sweep")
ax.legend(fontsize=8, title="sensorDist")
plt.tight_layout(); plt.savefig("figures/fig6_sd_spectra.png", dpi=140); plt.close()
print("wrote figures/fig5_sensordist.png, fig6_sd_spectra.png")
