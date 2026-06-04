#!/usr/bin/env python3
"""
Figures + collapse fit for the fluoddity-metal spectrum study.
Reads map_results.csv (2D forceGain×velDamp grid) and sweep_results.csv (OAT).
Writes figures/*.png. Run with the project venv: .venv/bin/python make_figures.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, csv, os

os.makedirs("figures", exist_ok=True)

# ── load the 2D map ──────────────────────────────────────────────────────────
M = [ {k: float(v) for k, v in r.items()} for r in csv.DictReader(open("map_results.csv")) ]
fgs = sorted(set(r["forceGain"] for r in M))
vds = sorted(set(r["velDamp"] for r in M))
S = np.full((len(vds), len(fgs)), np.nan)
for r in M:
    S[vds.index(r["velDamp"]), fgs.index(r["forceGain"])] = r["inSlope"]

# ── Fig 1: heatmap of the inertial slope over drive × dissipation ────────────
fig, ax = plt.subplots(figsize=(6.2, 4.6))
im = ax.imshow(S, origin="lower", aspect="auto", cmap="viridis")
cs = ax.contour(np.arange(len(fgs)), np.arange(len(vds)), S,
                levels=[-3, -5/3, -1], colors=["white", "red", "white"],
                linewidths=[1, 2.2, 1])
ax.clabel(cs, fmt={-3: "-3", -5/3: "-5/3", -1: "-1"}, fontsize=8)
ax.set_xticks(range(len(fgs))); ax.set_xticklabels(fgs)
ax.set_yticks(range(len(vds))); ax.set_yticklabels(vds)
ax.set_xlabel("forceGain  (drive →)"); ax.set_ylabel("velDamp  (retention / less dissipation →)")
ax.set_title("Inertial spectral slope: drive × dissipation")
for i in range(len(vds)):
    for j in range(len(fgs)):
        ax.text(j, i, f"{S[i,j]:.2f}", ha="center", va="center", color="0.85", fontsize=7)
plt.colorbar(im, label="inertial slope")
plt.tight_layout(); plt.savefig("figures/fig1_slope_map.png", dpi=140); plt.close()

# ── collapse fit on the map: slope vs log10(drive / drag-rate) ───────────────
fg = np.array([r["forceGain"] for r in M])
vd = np.array([r["velDamp"] for r in M])
sl = np.array([r["inSlope"] for r in M])
gam = -np.log(vd)                          # continuous drag rate
ratio = np.log10(fg / gam)                 # log(drive / dissipation)

def fit(X, y):
    X = np.column_stack(X + [np.ones_like(y)])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ c
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return c, r2

_,  R2_ratio = fit([ratio], sl)                      # 1-var: the naive ratio (fails)
c2, R2_2 = fit([np.log10(fg), np.log10(gam)], sl)    # 2-var: drive + dissipation, separate
a, b, c0 = c2
u = a * np.log10(fg) + b * np.log10(gam)             # the 2-group linear predictor
print(f"MAP ratio collapse  slope~log10(drive/drag):       R^2 = {R2_ratio:.3f}  (FAILS)")
print(f"MAP 2-group fit  slope = {a:+.2f} log(fG) {b:+.2f} log(gamma) {c0:+.2f}:  R^2 = {R2_2:.3f}")
print(f"   both coefficients {'same sign (both shallow the slope)' if a*b > 0 else 'opposite sign'}")

# ── Fig 2: the collapse onto the 2-group predictor ──────────────────────────
fig, ax = plt.subplots(figsize=(6.2, 4.6))
sc = ax.scatter(u, sl, c=vd, cmap="plasma", s=45, edgecolor="k", linewidth=0.3)
xs = np.linspace(u.min(), u.max(), 50)
ax.plot(xs, xs + c0, "k--", lw=1.2, label=f"slope = predictor + {c0:.2f}   R²={R2_2:.2f}")
ax.axhline(-5/3, color="red", lw=1, ls=":", label="-5/3")
ax.set_xlabel(f"2-group predictor:  {a:.2f}·log₁₀(forceGain) {b:+.2f}·log₁₀(drag-rate)")
ax.set_ylabel("inertial slope")
ax.set_title("Exponent collapses onto a 2-group drive+dissipation law")
plt.colorbar(sc, label="velDamp"); ax.legend()
plt.tight_layout(); plt.savefig("figures/fig2_collapse.png", dpi=140); plt.close()

# ── Fig 3: sensorDist sets the injection scale (peak k) ──────────────────────
sd, pk = [], []
for r in csv.DictReader(open("sweep_results.csv")):
    if r["axis"] == "sensorDist":
        sd.append(float(r["value"])); pk.append(int(r["peakK"]))
fig, ax = plt.subplots(figsize=(6.2, 4.6))
ax.plot(sd, pk, "o-", color="teal", lw=1.5)
ax.set_xscale("log")
ax.set_xlabel("sensorDist (agent sensing distance)")
ax.set_ylabel("peak wavenumber k  (injection scale)")
ax.set_title("sensorDist sets the injection scale")
plt.tight_layout(); plt.savefig("figures/fig3_injection_scale.png", dpi=140); plt.close()

print("wrote figures/fig1_slope_map.png, fig2_collapse.png, fig3_injection_scale.png")

# ── Fig 4 (optional): 3-axis scan — does sensorDist extend the law? ──────────
import os
if os.path.exists("map3_results.csv"):
    M3 = [ {k: float(v) for k, v in r.items()} for r in csv.DictReader(open("map3_results.csv")) ]
    fg3 = np.array([r["forceGain"] for r in M3]); vd3 = np.array([r["velDamp"] for r in M3])
    sd3 = np.array([r["sensorDist"] for r in M3]); sl3 = np.array([r["inSlope"] for r in M3])
    gam3 = -np.log(vd3)
    c2, R2_2g = fit([np.log10(fg3), np.log10(gam3)], sl3)              # drive + dissipation
    c3, R2_3g = fit([np.log10(fg3), np.log10(gam3), np.log10(sd3)], sl3)  # + injection scale
    a, b, d, c0 = c3
    print(f"\nMAP3 ({len(M3)} cells):")
    print(f"  2-group  slope ~ log(fG)+log(gamma):           R^2 = {R2_2g:.3f}")
    print(f"  3-group  + log(sensorDist)  [coef={d:+.2f}]:     R^2 = {R2_3g:.3f}")
    u3 = a*np.log10(fg3) + b*np.log10(gam3) + d*np.log10(sd3)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    sc = ax.scatter(u3, sl3, c=np.log10(sd3), cmap="cividis", s=42, edgecolor="k", linewidth=0.3)
    xs = np.linspace(u3.min(), u3.max(), 50)
    ax.plot(xs, xs + c0, "k--", lw=1.2, label=f"3-group fit  R²={R2_3g:.2f}")
    ax.axhline(-5/3, color="red", lw=1, ls=":", label="-5/3")
    ax.set_xlabel(f"3-group predictor:  {a:.2f}·logfG {b:+.2f}·logγ {d:+.2f}·log(sensorDist)")
    ax.set_ylabel("inertial slope")
    ax.set_title("3-axis collapse: drive + dissipation + injection scale")
    plt.colorbar(sc, label="log₁₀(sensorDist)"); ax.legend()
    plt.tight_layout(); plt.savefig("figures/fig4_collapse3.png", dpi=140); plt.close()
    print("wrote figures/fig4_collapse3.png")
