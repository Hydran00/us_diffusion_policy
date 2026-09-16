"""Render the original four-column US-DP paper layout with updated assets.

The ultrasound panels embed unchanged frames 116–118 of acq_c/demo_154.
The Panda/ABDFAN scene is a generated schematic illustration.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.image import imread
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "framework-v3"
NAVY = "#19365b"
BLUE = "#2566a7"
TEAL = "#11898f"
ORANGE = "#e16b27"
GRAY = "#546273"
SOFT = "#f2f7fc"
fig, ax = plt.subplots(figsize=(16.72, 9.41), dpi=200)
ax.set_xlim(0, 16.72)
ax.set_ylim(0, 9.41)
ax.axis("off")
fig.patch.set_facecolor("white")


def card(x, y, w, h, edge="#b9d0e7", face="white", lw=1.0, dashed=False, radius=0.12):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.025,rounding_size={radius}",
                       facecolor=face, edgecolor=edge, linewidth=lw,
                       linestyle=(0, (4, 3)) if dashed else "-")
    ax.add_patch(p)
    return p


def txt(x, y, s, size=10, color=NAVY, bold=False, ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
            fontweight="bold" if bold else "normal", family="DejaVu Sans",
            linespacing=1.2, zorder=6)


def arrow(x1, y1, x2, y2, color=NAVY, lw=1.3, dashed=False, curve=0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                mutation_scale=13, color=color, linewidth=lw,
                                linestyle="--" if dashed else "-",
                                connectionstyle=f"arc3,rad={curve}"))


def img(name, x1, x2, y1, y2):
    ax.imshow(imread(ROOT / "references" / name), extent=(x1, x2, y1, y2),
              interpolation="nearest", aspect="auto", zorder=3)


def pose(x, y, scale=0.28):
    ax.plot([x, x + scale], [y, y], color="#df3d32", lw=1.6)
    ax.plot([x, x + scale * .12], [y, y + scale], color="#2b9a5b", lw=1.6)
    ax.plot([x, x - scale * .20], [y, y + scale * .67], color="#2676d1", lw=1.6)
    ax.scatter([x], [y], s=11, c=NAVY, zorder=5)


# Four original-style numbered banners and thin stage separators.
headers = [(0.08, 3.94, "DATA ACQUISITION", NAVY),
           (4.20, 3.94, "PREPARATION", TEAL),
           (8.32, 3.94, "TRAINING", BLUE),
           (12.44, 4.20, "DEPLOYMENT", ORANGE)]
for index, (x, w, title, color) in enumerate(headers, 1):
    card(x, 8.86, w, .46, edge=color, face=color, lw=0, radius=.1)
    ax.add_patch(Circle((x + .25, 9.09), .27, facecolor=color, edgecolor="white", lw=1.4, zorder=5))
    txt(x + .25, 9.09, str(index), 16, "white", True)
    txt(x + w / 2 + .16, 9.09, title, 14, "white", True)
for x in (4.11, 8.23, 12.35):
    ax.plot([x, x], [.83, 8.78], color="#92b9d6", lw=.75, linestyle=(0, (4, 4)))

# Acquisition: keep the original privileged expert box, but use no fake US.
card(.12, 6.22, 3.82, 2.37, edge=ORANGE, face="white", lw=1.1, dashed=True)
txt(.34, 8.29, "Privileged anatomy (expert only)", 11.7, "#9e4b17", True, ha="left")
# stylized skin surface and trajectory reference
xs = np.linspace(.50, 3.52, 40)
ys = 6.70 + .16 * np.sin((xs - .50) * 2.6)
ax.fill_between(xs, ys - .29, ys, color="#d59b69", alpha=.35)
ax.plot(xs, ys, color="#ae6d3f", lw=2.5)
ax.plot(xs, ys + .38, color=ORANGE, lw=1.4, linestyle=(0, (4, 3)))
ax.scatter(xs[::7], (ys + .38)[::7], color=ORANGE, s=17)
txt(2.02, 6.42, "ABDFAN surface → expert reference", 9.7, GRAY)
arrow(2.03, 6.17, 2.03, 5.99, ORANGE, dashed=True)
card(.12, 1.16, 3.82, 4.70, edge="#bed4e7", face="#f8fbff")
txt(2.03, 5.57, "Measured ultrasound + TCP pose", 12.0, NAVY, True)
img("panda_abdfan_illustration.png", .30, 3.75, 3.20, 5.25)
txt(2.03, 3.03, "Franka Panda scanning Kyoto Kagaku ABDFAN", 9.3, GRAY)
for i, x in zip((116, 117, 118), (.55, 1.47, 2.39)):
    img(f"acq_c_demo_154_frame_{i:03d}.png", x, x + .82, 1.55, 2.78)
txt(2.03, 1.35, "Recorded liver ultrasound frames 116–118", 9.2, GRAY)

# Preparation: replace the original phantom split box with temporal windows.
card(4.30, 6.22, 3.82, 2.37, edge="#bad5e4", face="#f7fbfe")
txt(6.21, 8.30, "Sliding windows from one episode", 11.8, NAVY, True)
for i in range(20):
    x = 4.51 + i * .17
    ax.add_patch(Rectangle((x, 7.49), .11, .42,
                           facecolor=TEAL if 1 <= i <= 3 else ("#efad77" if 3 < i <= 15 else "#cbd8e3"),
                           edgecolor="white", lw=.45))
txt(6.21, 7.18, "120 frames → 68 overlapping training samples", 9.7, GRAY)
txt(6.21, 6.74, "3-frame history  +  1.0 s future", 10.3, TEAL, True)
arrow(6.21, 6.18, 6.21, 5.98, TEAL)
card(4.30, 3.28, 3.82, 2.57, edge="#bed4e7", face="#f8fbff")
txt(6.21, 5.58, "Construct training sample", 12.0, NAVY, True)
txt(5.21, 5.24, "3 US frames", 9.6, NAVY, True)
txt(7.18, 5.24, "3 TCP poses (9D)", 9.6, NAVY, True)
for i, x in zip((116, 117, 118), (4.52, 4.99, 5.46)):
    img(f"acq_c_demo_154_frame_{i:03d}.png", x, x + .45, 4.20, 5.02)
for x in (6.62, 7.11, 7.60):
    pose(x, 4.51, .29)
ax.plot(np.linspace(4.66, 7.75, 45),
        3.70 + .25 * np.sin(np.linspace(0, np.pi * 1.15, 45)),
        color=BLUE, lw=2.1)
txt(6.21, 3.47, "Measured future in current TCP frame (51 poses)", 9.3, GRAY)
arrow(6.21, 3.24, 6.21, 3.02, TEAL)
card(4.30, 1.16, 3.82, 1.73, edge="#b9dce0", face="#ecf8f9")
txt(6.21, 2.62, "C¹ quadratic spline fit", 11.7, NAVY, True)
xx = np.linspace(4.66, 7.72, 60)
yy = 1.77 + .40 * np.sin((xx - 4.66) / 3.06 * np.pi * .85)
ax.plot(xx, yy, color=BLUE, lw=2.2)
ax.scatter(xx[::12], yy[::12], color=BLUE, s=14)
txt(6.21, 1.38, "4 segments • 5 free XYZ vectors (15 scalars)", 9.7, NAVY, True)

# Training: the same two-branch model layout as the original.
card(8.42, 4.88, 1.82, 3.28, edge="#a9cbea", face="#f5faff")
card(10.37, 4.88, 1.82, 3.28, edge="#abd6d5", face="#f1faf9")
txt(9.33, 7.91, "Ultrasound history", 10.2, NAVY, True)
txt(11.28, 7.91, "TCP pose history", 10.2, NAVY, True)
for i, x in zip((116, 117, 118), (8.52, 8.88, 9.24)):
    img(f"acq_c_demo_154_frame_{i:03d}.png", x, x + .47, 6.55, 7.56)
for x in (10.58, 11.05, 11.52):
    pose(x, 6.97, .28)
arrow(9.33, 6.48, 9.33, 6.11, BLUE)
arrow(11.28, 6.48, 11.28, 6.11, TEAL)
card(8.60, 5.37, 1.46, .63, edge="#9fc7e8", face="#e9f4ff")
card(10.55, 5.37, 1.46, .63, edge="#a8d6d3", face="#e9f8f5")
txt(9.33, 5.68, "Frozen USFM\nViT-B/16", 9.7, BLUE, True)
txt(11.28, 5.68, "State MLP", 9.7, TEAL, True)
arrow(9.33, 4.83, 10.18, 4.55, BLUE)
arrow(11.28, 4.83, 10.43, 4.55, TEAL)
card(8.95, 4.07, 2.73, .43, edge="#bdd0e4", face="#f6f9fd")
txt(10.31, 4.28, "Concatenate features", 9.5, NAVY, True)
arrow(10.31, 4.02, 10.31, 3.78, BLUE)
card(8.42, 2.18, 3.77, 1.47, edge="#aaaee7", face="#f2f1ff")
txt(10.31, 3.39, "Conditional 1D U-Net + DDPM", 11.1, NAVY, True)
for i, h in enumerate((.27, .42, .65, .42, .27)):
    ax.add_patch(Rectangle((9.24 + i * .42, 2.53), .25, h,
                           facecolor="#a9bce9", edgecolor=BLUE, lw=.8))
txt(10.31, 2.34, "Denoise spline coefficients", 9.2, BLUE)
arrow(10.31, 2.13, 10.31, 1.92, BLUE)
card(9.04, 1.18, 2.54, .59, edge="#eeaab0", face="#fff0f2")
txt(10.31, 1.47, "Noise prediction MSE", 10.3, "#a63342", True)

# Deployment: original stacked sampler/decoder/trajectory architecture.
card(12.54, 7.21, 4.02, 1.19, edge="#b8cce1", face="#f7fbff")
txt(14.55, 8.12, "Sample spline coefficients", 12.0, NAVY, True)
for i in range(6):
    ax.add_patch(Rectangle((13.33 + i * .36, 7.46), .27, .38,
                           facecolor="#e1b596", edgecolor="white", lw=.5))
txt(14.55, 7.29, "15 free scalars", 8.8, GRAY)
arrow(14.55, 7.15, 14.55, 6.92, NAVY)
card(12.54, 6.08, 4.02, .72, edge=ORANGE, face="#fff4e9")
txt(14.55, 6.43, "C¹ spline decoder → world TCP", 11.4, NAVY, True)
arrow(14.55, 6.03, 14.55, 5.78, NAVY)
card(12.54, 1.83, 4.02, 3.78, edge="#bfd3e8", face="#f8fbff")
txt(14.55, 5.33, "World-frame Cartesian references", 11.5, NAVY, True)
# Full 1 s trajectory, orange first 0.8 s.
xx = np.linspace(12.91, 16.20, 51)
yy = 3.54 + .54 * np.sin(np.linspace(0, np.pi * 1.00, 51)) + .18 * (xx - 12.91)
ax.plot(xx, yy, color="#93a6b6", lw=2.5)
ax.plot(xx[:41], yy[:41], color=ORANGE, lw=3.2)
ax.scatter([xx[0], xx[40], xx[-1]], [yy[0], yy[40], yy[-1]],
           s=[30, 30, 30], c=[NAVY, ORANGE, GRAY], zorder=5)
txt(14.55, 4.66, "Full plan: 1.0 s / 50 steps", 10.0, GRAY)
txt(14.55, 2.91, "Execute first 0.8 s / 40 steps", 10.2, ORANGE, True)
txt(14.55, 2.43, "Position predicted; orientation fixed", 9.7, GRAY)
arrow(16.33, 1.98, 16.36, 3.25, ORANGE, dashed=True, curve=-.45)
card(12.54, 1.16, 4.02, .47, edge="#edd2bc", face="#fff8f2")
txt(14.55, 1.39, "Replan from newly measured TCP", 10.2, ORANGE, True)

# Bold transitions between columns and unobtrusive footer legend.
for x in (4.01, 8.13, 12.25):
    arrow(x, 4.42, x + .20, 4.42, NAVY, 2.2)
ax.plot([.12, 16.55], [.71, .71], color="#8aa9c4", lw=.7)
arrow(3.55, .36, 4.27, .36, NAVY, 1.8)
txt(6.38, .36, "Training and deployment data flow", 9.3, GRAY)
arrow(9.80, .36, 10.52, .36, ORANGE, 1.8, dashed=True)
txt(13.15, .36, "Expert reference / replanning feedback", 9.3, GRAY)
fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(OUT.with_suffix(".png"), dpi=200, facecolor="white")
fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
print(OUT.with_suffix(".png"))
print(OUT.with_suffix(".svg"))
