"""Render the US-DP paper figure from recorded simulator frames.

Run from the workspace root with ``us_dp/.venv/bin/python
us_dp/docs/figures/render_framework.py``. Ultrasound PNGs are exact frames extracted from acq_c/demo_154; this script
changes their display size only. The Panda/ABDFAN inset is an illustration.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.image import imread
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "framework-v2"
NAVY = "#163454"
BLUE = "#2b6da8"
TEAL = "#078887"
ORANGE = "#dc7429"
GRAY = "#5a6c7d"
PALE = "#f4f8fc"

fig, ax = plt.subplots(figsize=(20, 10), dpi=180)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
ax.axis("off")


def box(x, y, w, h, edge="#c5d5e3", fill="white", radius=0.14, lw=1.2):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={radius}",
                           edgecolor=edge, facecolor=fill, linewidth=lw)
    ax.add_patch(patch)
    return patch


def label(x, y, value, size=12, color=NAVY, weight="normal", ha="center", va="center"):
    ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, family="DejaVu Sans", linespacing=1.25)


def arrow(a, b, color=BLUE, lw=1.8, dashed=False, curve=0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15,
                                linewidth=lw, color=color,
                                linestyle="--" if dashed else "-",
                                connectionstyle=f"arc3,rad={curve}"))


def photo(path, extent):
    ax.imshow(imread(ROOT / "references" / path), extent=extent, aspect="auto",
              interpolation="nearest", zorder=3)


label(10, 9.72, "Ultrasound-conditioned diffusion spline policy", 23, NAVY, "bold")
label(10, 9.25, "Measured imaging  →  local spline targets  →  conditional DDPM  →  receding-horizon control",
      12, GRAY)

panels = [(0.3, 4.65, "1   ACQUISITION", NAVY),
          (5.05, 4.65, "2   PREPARATION", TEAL),
          (9.8, 4.75, "3   TRAINING", BLUE),
          (14.65, 5.05, "4   DEPLOYMENT", ORANGE)]
for x, w, title, accent in panels:
    box(x, 0.85, w, 8.1, edge="#d4dfea", fill="white", radius=0.18)
    box(x, 8.25, w, 0.7, edge=accent, fill=accent, radius=0.14)
    label(x + w / 2, 8.59, title, 15, "white", "bold")

# 1. Recorded simulator camera and true recorded ultrasound frames.
box(0.57, 4.6, 4.1, 3.25, fill=PALE)
label(2.62, 7.58, "Franka Panda + Kyoto Kagaku ABDFAN", 11.5, NAVY, "bold")
photo("panda_abdfan_illustration.png", (0.73, 4.50, 5.43, 7.38))
for i, x in zip((116, 117, 118), (1.38, 2.22, 3.06)):
    photo(f"acq_c_demo_154_frame_{i:03d}.png", (x, x + 0.73, 4.67, 5.39))
label(0.95, 5.0, "US →", 10, GRAY)
box(0.6, 2.95, 4.05, 1.02, edge="#b9d6d5", fill="#eff9f8")
label(2.62, 3.61, "Synchronized measurements", 13, NAVY, "bold")
label(2.62, 3.22, "ultrasound + TCP pose at 50 Hz", 11, TEAL)
label(2.62, 2.1, "Expert motion generates measured\nTCP trajectories; no anatomy at inference", 10.5, GRAY)

# 2. Target construction; deliberately no group-split box.
box(5.33, 6.63, 4.06, 1.03, edge="#b9d6d5", fill="#eff9f8")
label(7.36, 7.24, "Observation history", 13, NAVY, "bold")
label(7.36, 6.91, "3 US frames  +  3 TCP poses (9D)", 11, TEAL)
arrow((7.36, 6.6), (7.36, 6.17), TEAL)
box(5.33, 5.1, 4.06, 0.97, fill=PALE)
label(7.36, 5.74, "Current TCP frame", 12.5, NAVY, "bold")
label(7.36, 5.39, "xᵢ = Rₜᵀ (pₜ₊ᵢ − pₜ)", 13, BLUE)
arrow((7.36, 5.08), (7.36, 4.67), TEAL)
box(5.33, 3.56, 4.06, 1.0, fill=PALE)
label(7.36, 4.18, "Measured future: 1.0 s", 12.5, NAVY, "bold")
label(7.36, 3.83, "50 intervals + starting pose = 51 points", 10.5, GRAY)
arrow((7.36, 3.52), (7.36, 3.1), TEAL)
box(5.33, 1.48, 4.06, 1.53, edge="#b9d6d5", fill="#eff9f8")
label(7.36, 2.68, "Anchored C¹ quadratic spline fit", 12.5, NAVY, "bold")
label(7.36, 2.23, "4 segments  •  5 free XYZ vectors", 11.2, TEAL)
label(7.36, 1.84, "15 learned scalars per target", 10.5, GRAY)

# 3. Image and state pathways converge into the denoiser.
box(10.1, 5.59, 2.02, 1.93, edge="#b4cde8", fill="#edf5fc")
photo("acq_c_demo_154_frame_118.png", (10.64, 11.59, 6.44, 7.27))
label(11.11, 6.17, "Frozen USFM\nViT-B/16", 11.5, BLUE, "bold")
box(12.32, 5.59, 2.02, 1.93, edge="#b9d6d5", fill="#eff9f8")
label(13.33, 6.83, "TCP pose", 11, GRAY)
label(13.33, 6.17, "State MLP", 12, TEAL, "bold")
arrow((11.11, 5.56), (11.87, 5.07), BLUE)
arrow((13.33, 5.56), (12.54, 5.07), TEAL)
box(10.72, 4.27, 3.0, 0.7, edge="#b9cbe1", fill=PALE)
label(12.22, 4.62, "Concatenate features", 11.8, NAVY, "bold")
arrow((12.22, 4.23), (12.22, 3.8), BLUE)
box(10.08, 2.42, 4.3, 1.25, edge="#a7b8eb", fill="#f2f1ff")
label(12.23, 3.27, "Conditional 1D U-Net + DDPM", 12.4, NAVY, "bold")
label(12.23, 2.82, "Denoise spline coefficients", 11, BLUE)
arrow((12.23, 2.39), (12.23, 2.0), BLUE)
box(10.72, 1.25, 3.0, 0.66, edge="#edb6b3", fill="#fff3f1")
label(12.22, 1.58, "Noise prediction MSE", 11.8, "#b34543", "bold")

# 4. Full trajectory and executed prefix.
box(14.96, 6.77, 4.42, 0.82, fill=PALE)
label(17.17, 7.32, "Sample 15 spline scalars", 12.5, NAVY, "bold")
label(17.17, 6.99, "full prediction: 1.0 s / 50 steps", 10.8, GRAY)
arrow((17.17, 6.72), (17.17, 6.3), ORANGE)
box(14.96, 5.42, 4.42, 0.78, edge="#eed3bd", fill="#fff6ed")
label(17.17, 5.82, "C¹ spline decoder → world TCP", 12.5, NAVY, "bold")
arrow((17.17, 5.38), (17.17, 4.99), ORANGE)
box(14.97, 2.75, 4.4, 2.12, fill=PALE)
xx = np.linspace(15.35, 19.0, 51)
yy = 3.25 + 0.76 * np.sin((xx - 15.35) / 3.65 * np.pi * 0.8) + 0.13 * (xx - 15.35)
ax.plot(xx, yy, color="#a0aebc", lw=3.2, zorder=3)
ax.plot(xx[:41], yy[:41], color=ORANGE, lw=4.5, zorder=4)
ax.scatter([xx[0], xx[40], xx[-1]], [yy[0], yy[40], yy[-1]],
           c=[NAVY, ORANGE, GRAY], s=[55, 55, 55], zorder=5)
label(17.16, 2.98, "Execute first 0.8 s / 40 steps", 11.4, ORANGE, "bold")
box(14.96, 1.28, 4.42, 0.9, edge="#eed3bd", fill="#fff6ed")
label(17.17, 1.83, "Observe measured TCP → replan", 11.4, NAVY, "bold")
label(17.17, 1.48, "Position predicted; orientation fixed", 10.2, GRAY)
arrow((19.42, 1.68), (19.43, 6.87), ORANGE, 1.6, True, 0.45)

# Stage-to-stage flow and source credit.
for start, end in [((4.67, 4.88), (5.02, 4.88)),
                   ((9.42, 4.88), (9.77, 4.88)),
                   ((14.4, 4.88), (14.74, 4.88))]:
    arrow(start, end, NAVY, 2.2)
label(10, 0.44,
      "US frames: recorded simulator data (acq_c/demo_154, 116–118). Panda/ABDFAN: schematic illustration. ABDFAN model: Kyoto Kagaku US-1B.",
      9.3, GRAY)

fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(OUT.with_suffix(".png"), dpi=180, facecolor="white")
fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
print(OUT.with_suffix(".png"))
print(OUT.with_suffix(".svg"))
