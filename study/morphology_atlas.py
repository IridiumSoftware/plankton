#!/usr/bin/env python3
import os as _os; _os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # run from anywhere: paths resolve against the repo root
"""
Render the morphology atlas from data/morphology.csv + data/morphology_thumbs.bin
(written by `swift run plankton --morphology`): a cohesion × dyeDecay montage of
the ACTUAL dye thumbnails, each bordered by its classified regime (foam / network
/ spots / dispersed).  Run with the project venv:
.venv/bin/python study/morphology_atlas.py  →  figures/fig_morphology.png
"""
import csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("data/morphology.csv")))
cohs = sorted(set(float(r["cohesion"]) for r in rows))
dds = sorted(set(float(r["dyeDecay"]) for r in rows))
BORDER = {"dispersed": (0.3, 0.3, 0.4), "spots": (0.23, 0.63, 1.0),
          "network": (1.0, 0.82, 0.23), "foam": (1.0, 0.35, 0.54)}

cls = {(float(r["cohesion"]), float(r["dyeDecay"])): r["class"] for r in rows}
length = {(float(r["cohesion"]), float(r["dyeDecay"])): float(r["lengthScale"]) for r in rows}

# thumbnails: header [cols, rows, td] int32, then cols*rows td² uint8, cohesion-major
raw = open("data/morphology_thumbs.bin", "rb").read()
cols, nrows, td = np.frombuffer(raw[:12], dtype=np.int32)
thumbs = np.frombuffer(raw[12:12 + cols * nrows * td * td], dtype=np.uint8).reshape(cols, nrows, td, td)

fig, axes = plt.subplots(len(dds), len(cohs), figsize=(1.5 * len(cohs), 1.5 * len(dds)))
for di, dd in enumerate(dds):          # top row = highest dyeDecay
    for ci, coh in enumerate(cohs):
        ax = axes[len(dds) - 1 - di][ci]
        ax.imshow(thumbs[ci][di], cmap="magma", vmin=0, vmax=255)
        k = cls.get((coh, dd), "dispersed")
        for s in ax.spines.values():
            s.set_color(BORDER[k]); s.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])
        if di == 0:
            ax.set_xlabel(f"{coh:g}", fontsize=9)
        if ci == 0:
            ax.set_ylabel(f"{dd:g}", fontsize=9, rotation=0, ha="right", va="center")

fig.supxlabel("cohesion (chemotaxis strength) →", fontsize=11)
fig.supylabel("dyeDecay (wall persistence) →", fontsize=11)
fig.suptitle("plankton morphology atlas — dye structure vs (cohesion, dyeDecay)\n"
             "border: foam(pink) · network(gold) · spots(blue) · dispersed(grey)", fontsize=11)
fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])
_os.makedirs("figures", exist_ok=True)
out = "figures/fig_morphology.png"
fig.savefig(out, dpi=130)
print("wrote", out)
