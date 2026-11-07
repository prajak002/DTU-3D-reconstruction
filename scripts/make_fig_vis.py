"""Qualitative figure of the true-visibility protocol from the per-pixel dumps (stage `dump`, results/P6-dump/<dataset>):
no model is run. One row per selected view: the photograph, the ground-truth support (single-view pixels in red,
supported in green, unlabelled dark), native confidence, reprojection consistency, the label-free head's score and
its observed / abstain / imagined bands (0.7 / 0.3). Pick a cell by scene, seed and view count.

Usage: python scripts/make_fig_vis.py results/P6-dump/dtu --scene 3 --views 5 --seed 0 --base vggt --rows 0,2
       [--detectors vggt,vggt_reproj,vggt_oiavisself] [--out paper/figures/fig_vis_dtu.pdf]"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--views", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default="vggt")
    ap.add_argument("--detectors", default=None, help="comma list; default <base>, <base>_reproj, <base>_oiavisself")
    ap.add_argument("--rows", default="0,1", help="which of the input views to show")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    key = f"{a.scene}__s{a.seed}__v{a.views}"
    L = np.load(os.path.join(a.dump_dir, "labels", key + ".npz"))
    dets = a.detectors.split(",") if a.detectors else [a.base, f"{a.base}_reproj", f"{a.base}_oiavisself"]
    maps = {}
    for d in dets:
        p = os.path.join(a.dump_dir, d, key + ".npz")
        if os.path.exists(p):
            maps[d] = np.load(p)["conf"].astype(np.float32)
        else:
            print("missing", p)
    rows = [int(r) for r in a.rows.split(",")]
    U, S = L["U"], L["S"]
    rgb = L["rgb"] if "rgb" in L.files else None
    head = [d for d in maps if "oia" in d]
    cols = ["photograph", "ground-truth support"] + [d for d in maps] + ([f"{head[-1]} bands"] if head else [])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h, w = U.shape[1:]
    fig, ax = plt.subplots(len(rows), len(cols), figsize=(len(cols) * 2.6, len(rows) * 2.6 * h / w + 0.5), squeeze=False)
    for i, r in enumerate(rows):
        j = 0
        if rgb is not None:
            ax[i, j].imshow(rgb[r])
        else:
            ax[i, j].imshow(np.zeros((h, w)), cmap="gray")
        j += 1
        lab = np.zeros((h, w, 3), np.float32) + 0.15
        lab[S[r]] = (0.2, 0.65, 0.3)
        lab[U[r]] = (0.85, 0.2, 0.2)
        ax[i, j].imshow(lab); j += 1
        for d, m in maps.items():
            mm = m[r]
            if "oia" in d:
                ax[i, j].imshow(mm, cmap="viridis", vmin=0, vmax=1)
            else:
                lo, hi = np.percentile(mm, 2), np.percentile(mm, 98)
                ax[i, j].imshow(np.clip((mm - lo) / (hi - lo + 1e-9), 0, 1), cmap="viridis", vmin=0, vmax=1)
            j += 1
        if head:
            s = maps[head[-1]][r]
            band = np.zeros((h, w, 3), np.float32)
            band[s > 0.7] = (0.2, 0.65, 0.3)
            band[(s <= 0.7) & (s >= 0.3)] = (0.95, 0.75, 0.15)
            band[s < 0.3] = (0.85, 0.2, 0.2)
            base = rgb[r].astype(np.float32) / 255 if rgb is not None else np.zeros((h, w, 3), np.float32)
            ax[i, j].imshow(0.35 * base + 0.65 * band)
        for jj in range(len(cols)):
            ax[i, jj].set_xticks([]); ax[i, jj].set_yticks([])
            if i == 0:
                ax[i, jj].set_title(cols[jj].replace(f"{a.base}_", f"{a.base} "), fontsize=8)
        ax[i, 0].set_ylabel(f"view {r}", fontsize=8)
    fig.tight_layout()
    out = a.out or os.path.join("paper", "figures", f"fig_vis_{os.path.basename(os.path.normpath(a.dump_dir))}_{a.scene}.pdf")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.rsplit(".", 1)[0] + ".png", dpi=120, bbox_inches="tight")
    print("->", out, "single-view fraction per shown view:", [float(U[r].sum() / max((U[r] | S[r]).sum(), 1)) for r in rows])


if __name__ == "__main__":
    main()
