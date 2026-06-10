#!/usr/bin/env python3
"""
3D energy-spectrum analysis from the --3dspec dumps. Reads 3dspec_manifest.csv +
vel3d_*.bin (raw float32, 3 floats/cell interleaved, 128^3), computes the
spherically-binned 3D energy spectrum E(k) for each forcing value, fits the
inertial slope, and compares to the Kolmogorov -5/3 EXPECTED in 3D. The question:
is the 3D engine's slope ~ -5/3 and forcing-invariant (a real 3D cascade), or does
it dial like the 2D engine? Writes figures/fig9_3d_spectrum.png.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, csv

N = 128
k1 = np.fft.fftfreq(N, d=1.0 / N)
KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
kflat = np.round(np.sqrt(KX**2 + KY**2 + KZ**2)).astype(int).ravel()

def spectrum3d(fn):
    v = np.fromfile(fn, dtype=np.float32).reshape(N, N, N, 3)
    Etot = np.zeros((N, N, N))
    for c in range(3):
        Etot += 0.5 * np.abs(np.fft.fftn(v[..., c]))**2 / N**6
    return np.bincount(kflat, weights=Etot.ravel(), minlength=N // 2)[:N // 2]

def fit_slope(E, klo, khi):
    ks = np.arange(klo, khi + 1); ek = E[klo:khi + 1]; m = ek > 0
    if m.sum() < 3: return 0.0, 0.0
    x = np.log10(ks[m]); y = np.log10(ek[m])
    A = np.vstack([x, np.ones_like(x)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c[0], 1 - np.sum((y - A @ c)**2) / np.sum((y - y.mean())**2)

rows = list(csv.DictReader(open("data/3dspec_manifest.csv")))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
cmap = plt.cm.viridis
print("forceGain  peakK  fitwindow  forward-slope  R2")
fgs, slopes = [], []
for j, r in enumerate(rows):
    fg = float(r["forceGain"]); E = spectrum3d(r["file"])
    peak = int(np.argmax(E[1:]) + 1)
    klo = min(peak + 3, N // 2 - 8); khi = min(55, N // 2 - 1)   # forward cascade: ABOVE the peak
    sl, r2 = fit_slope(E, klo, khi)
    fgs.append(fg); slopes.append(sl)
    print(f"{fg:<9g}  {peak:<5d}  [{klo},{khi}]    {sl:+.2f}          {r2:.2f}")
    ax[0].loglog(np.arange(1, N // 2), E[1:], color=cmap(j / max(1, len(rows) - 1)),
                 lw=1.4, label=f"fG={fg:g} (k>{peak}: {sl:+.2f})")
xs = np.array([18, 55])
ax[0].loglog(xs, 5e-4 * (xs / 18)**(-5 / 3), "k--", lw=1.2, label="-5/3")
ax[0].set_xlabel("k"); ax[0].set_ylabel("E(k)")
ax[0].set_title("3D energy spectrum vs forcing"); ax[0].legend(fontsize=8)

ax[1].plot(fgs, slopes, "o-", color="crimson", lw=1.8)
ax[1].axhline(-5 / 3, color="purple", ls="--", label="-5/3 (3D Kolmogorov)")
ax[1].set_xscale("log"); ax[1].set_xlabel("forceGain"); ax[1].set_ylabel("inertial slope")
ax[1].set_title(f"3D slope vs forcing  (Δ={max(slopes) - min(slopes):.2f})"); ax[1].legend()
plt.tight_layout(); plt.savefig("figures/fig9_3d_spectrum.png", dpi=140); plt.close()
print("wrote figures/fig9_3d_spectrum.png")
