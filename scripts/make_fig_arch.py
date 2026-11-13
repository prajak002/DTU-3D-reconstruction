"""Architecture and training diagram (fig_arch), with real photographs and the head's real output.

Layout: (a) training framework: N unposed photos -> frozen backbone (per-image encoder, cross-view attention) ->
cross-view tokens -> support head -> training objective; a label-free teacher (reprojection of the backbone's own
depth and cameras) supplies the labels; a per-image control taps the encoder. (b) what the head outputs: the
photographs of one unseen DTU scan beside the head's logged score in its three fixed bands (observed > 0.7,
abstain, imagined < 0.3). The scan is the one of Figure 1 (largest single-view share in the five-view dump); the
views shown are the two with the largest share of reference depth. Nothing in the figure is drawn by hand except
boxes, arrows and the two icons.

  python scripts/make_fig_arch.py [--out paper/figures/fig_arch]"""
import argparse
import glob
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK, MUTED = "#1D2B36", "#5C6B77"
BLUE_FC, BLUE_EC = "#E7EEF6", "#7C9CBF"
GREEN_FC, GREEN_EC = "#E3F3F1", "#2A9D8F"
YEL_FC, YEL_EC = "#FBFAF7", "#B9B3A6"
GRAY_FC, GRAY_EC = "#EEF0F2", "#7B8794"
PURPLE = "#5B8DB8"
OBS, ABST, IMAG = "#2A9D8F", "#E9C46A", "#D1495B"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
                     "pdf.fonttype": 42})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(ROOT, "results", "P20-dump", "dtu"))
    ap.add_argument("--hd", default=os.path.join(ROOT, "results", "P20-dump-hd", "dtu", "hd"))
    ap.add_argument("--base", default="vggt")
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "figures", "fig_arch_training"))
    a = ap.parse_args()
    key = max(glob.glob(os.path.join(a.dump, "labels", "*__s0__v5.npz")), key=lambda p: float(np.load(p)["U"].mean()))
    key = os.path.basename(key)[:-4]
    L = np.load(os.path.join(a.dump, "labels", key + ".npz"))
    rgb = np.load(os.path.join(a.hd, "rgb", key + ".npz"))["rgb"]
    sc = np.load(os.path.join(a.hd, a.base + "_oiavisself", key + ".npz"))["conf"].astype(np.float32)
    order = np.argsort(-(L["gt"] > 0).reshape(len(rgb), -1).mean(1))
    h, w = rgb.shape[1:3]

    def bands(v):
        s = cv2.resize(sc[v], (w, h), interpolation=cv2.INTER_LINEAR)
        out = np.zeros((h, w, 3), np.float32)
        for m, col in ((s > 0.7, OBS), ((s <= 0.7) & (s >= 0.3), ABST), (s < 0.3, IMAG)):
            out[m] = [int(col[i:i + 2], 16) for i in (1, 3, 5)]
        return (0.72 * out + 0.28 * rgb[v]).astype(np.uint8)

    W, H = 6.875, 2.45
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(0, H)

    def box(x, y, bw, bh, fc, ec, lw=1.0, r=0.07, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))

    def txt(x, y, t, fs=6.6, **kw):
        kw.setdefault("ha", "center"); kw.setdefault("va", "center"); kw.setdefault("color", INK)
        ax.text(x, y, t, fontsize=fs, zorder=5, linespacing=1.15, **kw)

    def arrow(x0, y0, x1, y1, ls="-", rad=0.0, color=INK):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=1.0, ls=ls, color=color, zorder=4,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))

    def img(im, x, y, iw, ih, ec="#C9D1D9"):
        a_ = fig.add_axes([x / W, y / H, iw / W, ih / H]); a_.imshow(im, interpolation="lanczos", aspect="auto")
        a_.set_xticks([]); a_.set_yticks([])
        for sp in a_.spines.values():
            sp.set_color(ec); sp.set_linewidth(0.7)

    def snowflake(x, y, r=0.075, color=BLUE_EC):
        for k in range(3):
            t = np.pi * k / 3
            ax.plot([x - r * np.cos(t), x + r * np.cos(t)], [y - r * np.sin(t), y + r * np.sin(t)], color=color, lw=1.3, zorder=6,
                    solid_capstyle="round")
            for sgn in (-1, 1):
                bx, by = x + sgn * 0.62 * r * np.cos(t), y + sgn * 0.62 * r * np.sin(t)
                for d in (-0.6, 0.6):
                    ax.plot([bx, bx + sgn * 0.3 * r * np.cos(t + d)], [by, by + sgn * 0.3 * r * np.sin(t + d)], color=color, lw=0.9, zorder=6)

    def flame(x, y, s=0.085):
        outer = np.array([[0, -1], [0.62, -0.55], [0.7, 0.05], [0.38, 0.5], [0.3, 0.15], [0.05, 1.0], [-0.35, 0.35], [-0.5, 0.55],
                          [-0.72, 0.0], [-0.6, -0.6]]) * s + [x, y]
        inner = np.array([[0, -0.9], [0.32, -0.55], [0.3, -0.1], [0.05, 0.3], [-0.2, -0.05], [-0.34, -0.5]]) * s + [x, y]
        ax.add_patch(Polygon(outer, closed=True, fc="#E8541E", ec="none", zorder=6))
        ax.add_patch(Polygon(inner, closed=True, fc="#FFC83D", ec="none", zorder=7))

    # ---------------- (a) training framework
    yc = 1.47
    pw = 0.80; ph = pw * h / w                                   # two whole photographs, aligned, one above the other
    for j, v in enumerate(order[:2]):
        img(rgb[v], 0.07, 1.50 - j * (ph + 0.05), pw, ph)
    txt(0.07 + pw / 2, 2.22, "N unposed photos", weight="bold", fs=6.2)
    arrow(0.07 + pw + 0.03, yc, 1.10, yc)
    box(1.10, 0.80, 1.02, 1.50, BLUE_FC, BLUE_EC, lw=1.3)
    txt(1.61, 2.21, "frozen 3D backbone", weight="bold", fs=6.2)
    txt(1.61, 2.035, "VGGT-1B, Pi3, Pi3X,\nDepth Anything 3", fs=4.4, color=MUTED)
    box(1.19, 1.47, 0.84, 0.44, "#F3F8FE", BLUE_EC, lw=0.8); txt(1.61, 1.69, "per-image\nencoder", fs=6.2)
    box(1.19, 0.90, 0.84, 0.44, "#F3F8FE", BLUE_EC, lw=0.8); txt(1.61, 1.12, "cross-view\nattention", fs=6.2)
    arrow(2.12, 1.12, 2.36, 1.12)
    for k, bh in enumerate((0.14, 0.24, 0.34, 0.44)):
        ax.add_patch(Rectangle((2.42 + 0.105 * k, 0.94), 0.075, bh, fc=PURPLE, ec="none", zorder=3))
    txt(2.60, 0.78, "cross-view tokens\n(4 tapped layers)", fs=5.2, weight="bold", color=PURPLE)
    arrow(2.86, 1.12, 3.05, 1.12)
    box(3.05, 0.72, 0.80, 0.80, GREEN_FC, GREEN_EC, lw=1.3); txt(3.45, 1.40, "support head", weight="bold", fs=6.4)
    txt(3.45, 1.17, "LayerNorm, 1\u00d71 proj.\ntwo 3\u00d73 conv\n1 logit per patch", fs=4.7, color=INK)
    txt(3.45, 0.83, "trained, 1.7\u20133.3 M param.", fs=4.5, color=GREEN_EC, weight="bold")
    arrow(3.85, 1.12, 4.02, 1.12)
    box(4.02, 0.72, 0.98, 0.80, "white", GREEN_EC, lw=1.0)
    txt(4.51, 1.39, "Training objective", weight="bold", fs=6.2)
    txt(4.51, 1.14, "class-balanced\nbinary cross-entropy", fs=5.3)
    txt(4.51, 0.86, "1500 steps, 79 DTU scans", fs=4.6, color=MUTED)
    # per-image control
    box(2.42, 1.80, 1.36, 0.44, GRAY_FC, GRAY_EC, lw=0.9, ls=(0, (3, 2)))
    txt(3.10, 2.02, "per-image control\n(same head, no cross-view tokens)", fs=5.5)
    arrow(2.03, 1.75, 2.42, 1.96, ls=(0, (2, 2)), color=GRAY_EC)
    # label-free teacher
    ax.plot([1.61, 1.61, 2.02], [0.80, 0.33, 0.33], color=INK, lw=1.0, zorder=4); arrow(2.0, 0.33, 2.20, 0.33)
    txt(1.52, 0.52, "predicted depth\n+ cameras", fs=5.5, ha="right", color=MUTED)
    box(2.20, 0.08, 1.92, 0.50, YEL_FC, YEL_EC, lw=1.0)
    txt(3.16, 0.45, "Label-free teacher", weight="bold", fs=6.4)
    txt(3.16, 0.23, "reproject each point into the other views,\ncount the views that agree", fs=5.5)
    ax.plot([4.12, 4.51], [0.33, 0.33], color=INK, lw=1.0, zorder=4); arrow(4.51, 0.33, 4.51, 0.72)
    txt(4.57, 0.52, "agreement labels\n(no ground truth)", fs=5.2, ha="left", color=MUTED)
    txt(2.55, 2.38, "(a) Training framework", weight="bold", fs=6.8)
    # ---------------- (b) what the head outputs
    x0 = 5.14
    box(x0, 0.08, W - x0 - 0.04, 2.24, "white", "#7B8794", lw=1.0, r=0.09)
    txt(x0 + (W - x0 - 0.04) / 2, 2.38, "(b) What the head outputs", weight="bold", fs=6.8)
    iw = 0.715; ih = iw * h / w
    for r_, v in enumerate(order[:2]):
        y = 1.66 - r_ * (ih + 0.07)
        img(rgb[v], x0 + 0.07, y, iw, ih); arrow(x0 + 0.07 + iw + 0.02, y + ih / 2, x0 + 0.07 + iw + 0.14, y + ih / 2)
        img(bands(v), x0 + 0.07 + iw + 0.16, y, iw, ih)
    yl = 0.345
    for k, (col, name, xo) in enumerate(((OBS, "observed > 0.7", 0.07), (ABST, "abstain", 0.74), (IMAG, "imagined < 0.3", 1.10))):
        ax.add_patch(Rectangle((x0 + xo, yl - 0.04), 0.08, 0.08, fc=col, ec="none", zorder=5))
        txt(x0 + xo + 0.10, yl, name, fs=4.6, ha="left")
    txt(x0 + (W - x0 - 0.04) / 2, 0.18, "one extra forward, under 1% of the backbone cost", fs=4.8, color=MUTED)
    # legend
    box(0.08, 0.29, 0.13, 0.09, BLUE_FC, BLUE_EC, lw=0.8, r=0.02); txt(0.26, 0.335, "frozen", fs=5.4, ha="left")
    box(0.08, 0.12, 0.13, 0.09, GREEN_FC, GREEN_EC, lw=0.8, r=0.02); txt(0.26, 0.165, "trained", fs=5.4, ha="left")
    box(0.72, 0.29, 0.13, 0.09, GRAY_FC, GRAY_EC, lw=0.8, r=0.02, ls=(0, (2, 1.5))); txt(0.90, 0.335, "control", fs=5.4, ha="left")
    fig.savefig(a.out + ".pdf", dpi=500); fig.savefig(a.out + ".png", dpi=500)
    print("->", a.out, key, "views", [int(v) for v in order[:2]],
          "bands (observed/abstain/imagined):", [round(float(x), 3) for x in ((sc > 0.7).mean(), ((sc <= 0.7) & (sc >= 0.3)).mean(), (sc < 0.3).mean())])


if __name__ == "__main__":
    main()
