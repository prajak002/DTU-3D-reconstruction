"""Architecture figure in the training-diagram layout, with the content and the photographs of the original fig_arch
(scripts/make_fig_design.py): (a) training framework: photographs, frozen backbone with its four tapped cross-view
layers, the support head's chain, the per-image control, the training objective and the label-free teacher;
(b) what the head outputs: a photograph beside the head's bands, both cropped from the logged qualitative figure.

  python scripts/make_fig_arch.py [--out paper/figures/fig_arch]"""
import argparse
import os
import sys

import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_fig_design as D  # noqa: E402

INK, MUTED, OBS, ABST, IMAG, CTRL = D.INK, D.MUTED, D.OBS, D.ABST, D.IMAG, D.CTRL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(D.ROOT, "paper", "figures", "fig_arch"))
    a = ap.parse_args()
    c = D.Canvas(6.875, 2.50)
    c.text(2.70, 2.42, "(a) Training framework", ha="center", va="center", fontsize=6.8, weight="bold")
    c.text(6.005, 2.42, "(b) What the head outputs", ha="center", va="center", fontsize=6.8, weight="bold")
    # photographs: three whole tiles, aligned
    pw, ph = 0.60, 0.60 * 0.76
    for k, im in enumerate([D.tpanel(0), D.qpanel(0, 0), D.tpanel(3)]):
        c.image(im, 0.07, 1.72 - k * (ph + 0.035), pw, ph, lw=0.8)
    c.text(0.37, 2.26, "N unposed photos", ha="center", va="center", fontsize=5.6, weight="bold")
    c.text(0.37, 0.64, "3 to 20 views\nno cameras", ha="center", va="center", fontsize=5.0, color=MUTED, linespacing=1.1)
    c.arrow(0.70, 1.46, 0.84, 1.46)
    # frozen backbone
    c.box(0.84, 0.74, 1.46, 1.56, fc=D.FROZEN_FC, ec=D.FROZEN_EC, lw=1.1, r=0.08)
    c.text(1.57, 2.19, "frozen 3D backbone", ha="center", va="center", fontsize=6.8, weight="bold")
    c.text(1.57, 2.07, "VGGT-1B, Pi3, Pi3X, Depth Anything 3", ha="center", va="center", fontsize=5.2, color=MUTED)
    c.box(0.91, 1.00, 0.40, 0.96, "per-image\nencoder", ec=D.FROZEN_EC, fs=5.8)
    c.box(1.39, 1.00, 0.84, 0.96, ec=D.FROZEN_EC)
    c.text(1.81, 1.88, "cross-view attention", ha="center", va="center", fontsize=5.6)
    taps = []
    for k, yy in enumerate(np.linspace(1.05, 1.71, 8)):
        tapped = k in (1, 3, 5, 7)
        c.ax.add_patch(Rectangle((1.45, yy), 0.72, 0.06, fc=OBS if tapped else "#C8D6E6", ec="none", zorder=3))
        if tapped:
            taps.append(yy + 0.03)
            c.text(1.81, yy + 0.03, f"layer {(4, 11, 17, 23)[len(taps) - 1]}", ha="center", va="center", fontsize=4.8, color="white",
                   weight="bold", zorder=4)
    c.arrow(1.31, 1.48, 1.39, 1.48, ms=5)
    c.box(0.91, 0.79, 1.32, 0.16, "depth, cameras, native confidence", ec=D.FROZEN_EC, fs=5.2)
    # support head
    hx, hy, hw, hh = 2.50, 1.18, 2.05, 1.10
    c.box(hx, hy, hw, hh, fc=D.HEAD_FC, ec=D.HEAD_EC, lw=1.2, r=0.08)
    c.text(hx + hw / 2, hy + hh - 0.11, "support head (trained)", ha="center", va="center", fontsize=6.8, weight="bold", color=OBS)
    c.text(hx + hw / 2, hy + hh - 0.23, "1.7 to 3.3 M parameters, under 1% of the backbone's FLOPs", ha="center", va="center", fontsize=5.0,
           color=MUTED)
    chips = [("LayerNorm\n+ 1×1 proj\nper layer", 0.48), ("Σ", 0.17), ("3×3 conv\n× 2", 0.34), ("logit\nper patch", 0.34),
             ("bilinear\nupsample", 0.36)]
    x, cy, ch = hx + 0.06, hy + 0.21, 0.50
    starts = []
    for k, (t, w) in enumerate(chips):
        c.box(x, cy, w, ch, t, ec=D.HEAD_EC, fs=5.2)
        starts.append(x)
        if k < len(chips) - 1:
            c.arrow(x + w, cy + ch / 2, x + w + 0.07, cy + ch / 2, color=OBS, ms=4.5)
        x += w + 0.07
    for k, yy in enumerate(taps):
        c.arrow(2.17, yy, starts[0], cy + 0.08 + k * 0.11, color=OBS, lw=0.7, ms=4)
    c.text(hx + hw / 2, hy + 0.09, "score = 1 − P(imagined) per pixel, one forward", ha="center", va="center", fontsize=5.2, color=MUTED)
    c.arrow(hx + hw, cy + ch / 2, 5.16, cy + ch / 2, color=OBS)
    # per-image control and training objective
    c.box(2.50, 0.76, 1.62, 0.30, "per-image control: the same head on the encoder\ntokens, before any cross-view attention", ec=CTRL,
          ls="--", fs=5.0)
    c.arrow(1.11, 1.00, 2.50, 0.88, color=CTRL, lw=0.8, ls="--", ms=4.5, rad=0.2)
    c.box(4.22, 0.76, 0.86, 0.30, "training objective\nclass-balanced BCE", fc="white", ec=OBS, fs=5.0)
    c.arrow(4.40, hy, 4.40, 1.06, color=OBS, ms=4.5)
    # label-free teacher
    c.box(0.84, 0.05, 4.24, 0.60, fc=D.LANE_FC, ec=D.LANE_EC, ls="--", lw=0.8, r=0.06)
    c.text(0.92, 0.555, "Label-free teacher (no ground truth)", fontsize=6.2, weight="bold", va="center")
    c.text(4.68, 0.555, "1500 steps, 79 DTU scans; matched split adds observed patches as negatives", fontsize=4.3, color=MUTED,
           va="center", ha="right")
    for t, x0, w in (("run the backbone\nonce on clean views", 0.90, 0.80),
                     ("project each predicted point into the\nother views, predicted depth and cameras", 1.80, 1.36),
                     ("count the views whose\npredicted surface it hits", 3.26, 0.84)):
        c.box(x0, 0.11, w, 0.30, t, ec=D.LANE_EC, fs=4.9)
    for x0, x1 in ((1.70, 1.80), (3.16, 3.26), (4.10, 4.20)):
        c.arrow(x0, 0.26, x1, 0.26, color=MUTED, ms=4.5)
    c.box(4.20, 0.275, 0.82, 0.135, "0 other views: imagined", fc="#FBE8EA", ec=IMAG, fs=4.7, r=0.03)
    c.box(4.20, 0.11, 0.82, 0.135, "2 or more: observed", fc="#E3F3F1", ec=OBS, fs=4.7, r=0.03)
    c.arrow(4.75, 0.41, 4.75, 0.76, color=OBS, ms=4.5)
    c.text(4.79, 0.70, "labels", fontsize=4.6, color=OBS, va="center", ha="left")
    # (b) what the head outputs
    bx, bw = 5.18, 1.65
    c.box(bx, 0.05, bw, 2.25, fc="white", ec="#7B8794", lw=1.0, r=0.08)
    iw = 0.72; ih = iw * 219 / 273
    c.image(D.qpanel(0, 0), bx + 0.07, 1.56, iw, ih, "photograph", tfs=5.6)
    c.image(D.qpanel(0, 5), bx + 0.07 + iw + 0.07, 1.56, iw, ih, "head output", tfs=5.6, tcolor=OBS)
    for k, (lab, col, rg) in enumerate((("observed", OBS, "score > 0.7"), ("abstain", ABST, "0.3 to 0.7"), ("imagined", IMAG, "score < 0.3"))):
        yy = 1.22 - k * 0.27
        c.box(bx + 0.30, yy, 0.13, 0.13, fc=col, ec="none", r=0.02)
        c.text(bx + 0.50, yy + 0.065, lab, fontsize=6.0, va="center", weight="bold")
        c.text(bx + 1.02, yy + 0.065, rg, fontsize=5.0, va="center", color=MUTED)
    c.text(bx + bw / 2, 0.42, "bands fixed in advance", fontsize=5.0, color=MUTED, ha="center", va="center", style="italic")
    c.text(bx + bw / 2, 0.20, "one extra forward pass,\nbackbone unchanged", fontsize=4.9, color=MUTED, ha="center", va="center", linespacing=1.15)
    c.fig.savefig(a.out + ".pdf", dpi=500); c.fig.savefig(a.out + ".png", dpi=500)
    print("->", a.out)


if __name__ == "__main__":
    main()
