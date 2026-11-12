"""Detector comparison figure from the per-pixel dumps of scripts/dump_compare.py, no model run.

One row per (scan, view). Columns: the photograph, the ground-truth support (single-view red, supported green),
then one panel per detector. Every detector panel is drawn by the same rule, the one the paper's CUR uses: the
detector's threshold is set on that capture so that 90% of the supported pixels pass, and the panel shows what
passes. Green: a supported pixel that passes. Red: a single-view pixel that passes (the leak). Dark: rejected or
unlabelled. The number in the corner is the share of single-view pixels that pass (lower is better).

  python scripts/make_fig_compare.py --dump results/P20-dump/dtu --base vggt \
      --rows "28__s0__v5:3" "56__s0__v5:1" --out paper/figures/fig_compare.pdf

  python scripts/make_fig_compare.py --dump results/P20-dump/dtu --base vggt --rank
      prints, for every dumped capture, each detector's leak, so that rows can be chosen by a stated rule."""
import argparse
import glob
import os

import numpy as np

DETS = [("{b}", "native confidence"), ("{b}_pvf8", "perturbation variance"), ("{b}_reproj", "reprojection"),
        ("{b}_had", "HAD scorer (GT cameras)"), ("{b}_oiavisself", "support head (ours)")]
GREEN, RED, DARK = np.array([0.16, 0.62, 0.56]), np.array([0.82, 0.29, 0.36]), np.array([0.11, 0.17, 0.21])


def load_cell(dump, base, key):
    L = np.load(os.path.join(dump, "labels", key + ".npz"))
    out = {"U": L["U"].astype(bool), "S": L["S"].astype(bool),
           "rgb": L["rgb"] if "rgb" in L.files else None, "conf": {}}
    for pat, _ in DETS:
        p = os.path.join(dump, pat.format(b=base), key + ".npz")
        out["conf"][pat] = np.load(p)["conf"].astype(np.float64) if os.path.exists(p) else None
    return out


def threshold(conf, S, keep=0.9):
    """Score above which `keep` of the supported pixels of the capture lie (all its views together)."""
    s = conf[S]
    s = s[np.isfinite(s)]
    return float(np.quantile(s, 1.0 - keep)) if len(s) else float("nan")


def leak(conf, U, S, keep=0.9):
    tau = threshold(conf, S, keep)
    u = conf[U]
    return float((u >= tau).mean()) if len(u) else float("nan"), tau


def rank(a):
    rows = []
    for lp in sorted(glob.glob(os.path.join(a.dump, "labels", "*.npz"))):
        key = os.path.splitext(os.path.basename(lp))[0]
        c = load_cell(a.dump, a.base, key)
        if any(v is None for v in c["conf"].values()):
            continue
        leaks = [leak(c["conf"][pat], c["U"], c["S"], a.keep)[0] for pat, _ in DETS]
        rows.append((key, float(c["U"].sum() / max((c["U"] | c["S"]).sum(), 1)), leaks))
    print(f"{'capture':28s} single-view  " + "  ".join(f"{n[:14]:>14s}" for _, n in DETS))
    for key, fr, leaks in rows:
        print(f"{key:28s} {fr:10.3f}  " + "  ".join(f"{x:14.3f}" for x in leaks))
    if rows:
        m = np.nanmean(np.array([r[2] for r in rows]), 0)
        print(f"{'mean':28s} {'':10s}  " + "  ".join(f"{x:14.3f}" for x in m))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join("results", "P20-dump", "dtu"))
    ap.add_argument("--base", default="vggt")
    ap.add_argument("--rows", nargs="*", default=[], help="cell_key:view_index, e.g. 28__s0__v5:3")
    ap.add_argument("--keep", type=float, default=0.9)
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--out", default=os.path.join("paper", "figures", "fig_compare.pdf"))
    a = ap.parse_args()
    # a detector that was not dumped (HAD needs its own checkout and checkpoint) gets no column
    DETS[:] = [d for d in DETS if os.path.isdir(os.path.join(a.dump, d[0].format(b=a.base)))]
    if a.rank or not a.rows:
        rank(a)
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    specs = [r.rsplit(":", 1) for r in a.rows]
    cols = ["photograph", "ground truth"] + [n for _, n in DETS]
    h0, w0 = load_cell(a.dump, a.base, specs[0][0])["U"].shape[1:]
    pw = 1.75
    fig, ax = plt.subplots(len(specs), len(cols), figsize=(len(cols) * pw, len(specs) * pw * h0 / w0 + 0.3), squeeze=False)
    for i, (key, row) in enumerate(specs):
        r = int(row)
        c = load_cell(a.dump, a.base, key)
        U, S = c["U"], c["S"]
        rgb = (c["rgb"][r].astype(np.float32) / 255) if c["rgb"] is not None else np.zeros((*U.shape[1:], 3), np.float32)
        gray = rgb.mean(-1, keepdims=True)
        gt = np.repeat(0.25 * gray, 3, -1) + 0.75 * DARK
        gt[S[r]] = 0.3 * gray[S[r]] + 0.7 * GREEN
        gt[U[r]] = 0.3 * gray[U[r]] + 0.7 * RED
        panels, notes = [rgb, gt], ["", f"{100 * U[r].sum() / max((U[r] | S[r]).sum(), 1):.0f}% single-view"]
        for pat, _ in DETS:
            conf = c["conf"][pat]
            if conf is None:
                panels.append(np.zeros_like(rgb) + DARK); notes.append("not run"); continue
            lk, tau = leak(conf, U, S, a.keep)
            keep = conf[r] >= tau
            P = np.repeat(0.25 * gray, 3, -1) + 0.75 * DARK
            m = keep & S[r]; P[m] = 0.3 * gray[m] + 0.7 * GREEN
            m = keep & U[r]; P[m] = 0.15 * gray[m] + 0.85 * RED
            panels.append(P)
            lv = float((keep & U[r]).sum() / max(U[r].sum(), 1))
            notes.append(f"leak {100 * lv:.0f}%")
        for j, P in enumerate(panels):
            ax[i, j].imshow(np.clip(P, 0, 1)); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            for sp in ax[i, j].spines.values():
                sp.set_linewidth(0.4)
            if i == 0:
                ax[i, j].set_title(cols[j], fontsize=7.5, pad=3,
                                   fontweight="bold" if j == len(cols) - 1 else "normal")
            if notes[j]:
                ax[i, j].text(0.03, 0.04, notes[j], transform=ax[i, j].transAxes, fontsize=6.5, color="white",
                              bbox=dict(facecolor="black", alpha=0.6, pad=1.2, edgecolor="none"))
    fig.subplots_adjust(left=0.004, right=0.996, top=0.94, bottom=0.004, wspace=0.03, hspace=0.03)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=300)
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=200)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
