"""Qualitative figure of the sixteenth pass (moving scenes, full video): frames of one TUM RGB-D sequence with a
walking person, the sensor-derived labels, native confidence and the support head, from the per-pixel dump of
scripts/dynamic_eval.py --dump-scenes.

Rows are chosen by a fixed rule, not by eye: the --rows windows of the sequence with the largest share of moving
pixels (from the result table), middle frame of each window. Scores are shown between their 2nd and 98th percentile
within the window.

  python scripts/make_fig_dynamic.py --dump results/P23-dynamic/dump --base vggt [--out paper/figures/fig_dynamic]"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAG, OBS, MOV, INK, MUTED = "#D1495B", "#2A9D8F", "#8E44AD", "#1D2B36", "#5C6B77"

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 7, "pdf.fonttype": 42})


def column_figure(a, nat, head, w):
    L = np.load(os.path.join(a.dump, "labels", f"{a.scene}__w{w:03d}.npz"))
    f = L["rgb"].shape[0] // 2
    fig, ax = plt.subplots(1, 3, figsize=(3.3, 0.98))
    fig.subplots_adjust(left=0.004, right=0.996, top=0.87, bottom=0.01, wspace=0.03)
    ax[0].imshow(L["rgb"][f]); ax[0].set_title("video frame", fontsize=6.6, weight="bold", pad=2, color=INK)
    for j, m, t, col, name in ((1, a.base, nat, IMAG, "native confidence"), (2, a.base + "_oiavisself", head, OBS, "support head (ours)")):
        c = np.load(os.path.join(a.dump, m, f"{a.scene}__w{w:03d}.npz"))["conf"].astype(np.float32)
        lo, hi = np.quantile(c, 0.02), np.quantile(c, 0.98)
        ax[j].imshow(np.clip((c[f] - lo) / (hi - lo + 1e-9), 0, 1), cmap="viridis", vmin=0, vmax=1, interpolation="bilinear")
        ax[j].set_title(name, fontsize=6.6, weight="bold", pad=2, color=col)
        ax[j].text(0.03, 0.04, f"static {t.auroc_SU[w]:.2f}  moving {t.auroc_SM[w]:.2f}", transform=ax[j].transAxes, fontsize=5.4,
                   color="white", weight="bold", va="bottom", bbox=dict(fc="black", ec="none", alpha=0.55, pad=1.2))
    for x in ax:
        x.contour(L["M"][f].astype(float), levels=[0.5], colors="white", linewidths=1.3)
        x.contour(L["M"][f].astype(float), levels=[0.5], colors=MOV, linewidths=0.7)
        x.set_xticks([]); x.set_yticks([])
        for sp in x.spines.values():
            sp.set_color("#C9D1D9"); sp.set_linewidth(0.6)
    fig.savefig(a.out + ".pdf", dpi=500); fig.savefig(a.out + ".png", dpi=500)
    print("->", a.out, "window", w, "frame", f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(ROOT, "results", "P23-dynamic", "dump"))
    ap.add_argument("--base", default="vggt")
    ap.add_argument("--scene", default="freiburg3_walking_xyz")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--column", action="store_true", help="one-column figure: frame, native confidence, head")
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "figures", "fig_dynamic"))
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(ROOT, "results", "P23-dynamic", a.base, "P23-dynamic.csv"))
    x = df[(df.scan == a.scene)]
    nat, head = x[x.model == a.base].set_index("window"), x[x.model == a.base + "_oiavisself"].set_index("window")
    wins = sorted(nat.frac_M.sort_values(ascending=False).index[:a.rows])
    if a.column:          # main-paper version, one column wide: the window with the largest moving share only
        return column_figure(a, nat, head, int(nat.frac_M.idxmax()))
    fig, ax = plt.subplots(len(wins), 4, figsize=(6.875, 1.36 * len(wins) + 0.2))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.01, wspace=0.03, hspace=0.16)
    titles = ["frame of the video", "labels from sensor depth and poses", "native confidence", "support head (ours)"]
    for r, w in enumerate(wins):
        L = np.load(os.path.join(a.dump, "labels", f"{a.scene}__w{w:03d}.npz"))
        f = L["rgb"].shape[0] // 2
        rgb = L["rgb"][f]
        lab = (0.45 * rgb + 0.55 * 255).astype(np.float32)
        for k, col in (("S", OBS), ("U", IMAG), ("M", MOV)):
            c = np.array([int(col[i:i + 2], 16) for i in (1, 3, 5)], np.float32)
            lab[L[k][f]] = 0.25 * rgb[L[k][f]] + 0.75 * c
        ax[r, 0].imshow(rgb); ax[r, 1].imshow(lab.astype(np.uint8))
        for j, m in ((2, a.base), (3, a.base + "_oiavisself")):
            c = np.load(os.path.join(a.dump, m, f"{a.scene}__w{w:03d}.npz"))["conf"].astype(np.float32)
            lo, hi = np.quantile(c, 0.02), np.quantile(c, 0.98)
            ax[r, j].imshow(np.clip((c[f] - lo) / (hi - lo + 1e-9), 0, 1), cmap="viridis", vmin=0, vmax=1, interpolation="bilinear")
            ax[r, j].contour(L["M"][f].astype(float), levels=[0.5], colors="white", linewidths=1.5)
            ax[r, j].contour(L["M"][f].astype(float), levels=[0.5], colors=MOV, linewidths=0.8)
            t = nat if j == 2 else head
            ax[r, j].text(0.02, 0.03, f"static {t.auroc_SU[w]:.2f}   moving {t.auroc_SM[w]:.2f}", transform=ax[r, j].transAxes,
                          fontsize=6.2, color="white", weight="bold", va="bottom",
                          bbox=dict(fc="black", ec="none", alpha=0.55, pad=1.5))
        for j in range(4):
            ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
            for sp in ax[r, j].spines.values():
                sp.set_color("#C9D1D9"); sp.set_linewidth(0.6)
            if r == 0:
                ax[r, j].set_title(titles[j], fontsize=7.2, weight="bold", pad=3,
                                   color=IMAG if j == 2 else OBS if j == 3 else INK)
    fig.savefig(a.out + ".pdf", dpi=400); fig.savefig(a.out + ".png", dpi=400)
    print("->", a.out, "windows", wins)


if __name__ == "__main__":
    main()
