"""Radar chart of the true-visibility result (supplement): AUROC of native confidence, reprojection and the label-free
support head on every (dataset, backbone) pair, from docs/assets/numbers.json. One spoke per pair, grouped by dataset.
The radial axis starts at chance (0.5).

  python scripts/make_fig_radar.py [--out paper/figures/fig_radar]"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS, BASE, REPRO, INK, MUTED = "#2A9D8F", "#8D99AE", "#5B8DB8", "#1D2B36", "#5C6B77"
DS = [("dtu", "DTU"), ("co3d", "CO3Dv2"), ("eth3d", "ETH3D"), ("imc", "Phototourism")]
BASES = [("vggt", "VGGT"), ("pi3x", "Pi3X"), ("pi3", "Pi3"), ("da3", "DA3")]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 7, "pdf.fonttype": 42})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "figures", "fig_radar"))
    a = ap.parse_args()
    V = json.load(open(os.path.join(ROOT, "docs", "assets", "numbers.json"), encoding="utf-8"))["vis"]
    spokes = [(d, b) for d, _ in DS for b, _ in BASES]
    ang = np.linspace(0, 2 * np.pi, len(spokes), endpoint=False) + np.pi / len(spokes)
    fig = plt.figure(figsize=(3.4, 3.25))
    ax = fig.add_axes([0.11, 0.13, 0.78, 0.74], projection="polar")
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    for suf, name, col, lw, fill in (("", "native confidence", BASE, 1.1, 0.10), ("_reproj", "reprojection", REPRO, 1.1, 0.0),
                                     ("_oiavisself", "support head (ours)", OBS, 1.6, 0.16)):
        v = np.array([V[d][b + suf]["auroc"][0] for d, b in spokes])
        ax.plot(np.r_[ang, ang[0]], np.r_[v, v[0]], color=col, lw=lw, label=name, marker="o", ms=2.2, zorder=3)
        if fill:
            ax.fill(np.r_[ang, ang[0]], np.r_[v, v[0]], color=col, alpha=fill, zorder=2)
    ax.set_ylim(0.5, 1.0); ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0]); ax.set_yticklabels(["0.6", "", "0.8", "", "1.0"], fontsize=5.4, color=MUTED)
    ax.set_rlabel_position(0)
    ax.set_xticks(ang); ax.set_xticklabels([n for _ in DS for _, n in BASES], fontsize=5.8, color=INK)
    ax.tick_params(axis="x", pad=-1)
    ax.grid(color="#C9D1D9", lw=0.5); ax.spines["polar"].set_color("#C9D1D9")
    for k, (_, name) in enumerate(DS):          # dataset sectors
        a0 = 2 * np.pi * k / len(DS)
        ax.plot([a0, a0], [0.5, 1.0], color=MUTED, lw=0.7, zorder=1)
        mid = a0 + np.pi / len(DS)
        ax.text(mid, 1.2, name, ha="center", va="center", fontsize=6.8, weight="bold", color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=5.8, frameon=False, columnspacing=1.0, handlelength=1.4)
    fig.savefig(a.out + ".pdf"); fig.savefig(a.out + ".png", dpi=450)
    print("->", a.out)


if __name__ == "__main__":
    main()
