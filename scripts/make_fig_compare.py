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
GREEN, RED, DARK = np.array([0.10, 0.62, 0.52]), np.array([0.88, 0.18, 0.26]), np.array([0.11, 0.17, 0.21])
OURS, INK, GOOD, AMBER, BAD = "#0f8a73", "#2b303b", "#12866f", "#c9820a", "#cf2e40"


def load_cell(dump, base, key):
    L = np.load(os.path.join(dump, "labels", key + ".npz"))
    out = {"U": L["U"].astype(bool), "S": L["S"].astype(bool),
           "rgb": L["rgb"] if "rgb" in L.files else None, "conf": {}}
    for pat, _ in DETS:
        p = os.path.join(dump, pat.format(b=base), key + ".npz")
        out["conf"][pat] = np.load(p)["conf"].astype(np.float64) if os.path.exists(p) else None
    return out


def _up(a, hw, thr=None):
    """Display only: bilinear resize to hw; a mask is resized as a float map and cut at 0.5 (smooth outline)."""
    import cv2
    x = cv2.resize(a.astype(np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return x > thr if thr is not None else x


def load_hd(hd, base, key, r, c):
    """Full-size photograph and each detector's score at the model's resolution (dump_compare.py --hd), resized to
    the photograph. Labels and thresholds stay the low-resolution ones every number is computed from; the labels
    are upsampled for display. A detector without a full-resolution dump is upsampled from its low-resolution map."""
    rgb = np.load(os.path.join(hd, "hd", "rgb", key + ".npz"))["rgb"][r].astype(np.float32) / 255
    hw = rgb.shape[:2]
    conf = {}
    for pat, _ in DETS:
        q = os.path.join(hd, "hd", pat.format(b=base), key + ".npz")
        lo = c["conf"][pat]
        src = np.load(q)["conf"][r].astype(np.float64) if os.path.exists(q) else (lo[r] if lo is not None else None)
        conf[pat] = None if src is None else _up(np.nan_to_num(src, nan=-1e30, posinf=1e30, neginf=-1e30), hw)
    return rgb, _up(c["U"][r], hw, 0.5), _up(c["S"][r], hw, 0.5), conf


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
    ap.add_argument("--hd", default=None, help="dump written with dump_compare.py --hd: draw at full resolution")
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
    pw = 2.3 if a.hd else 1.75
    fig, ax = plt.subplots(len(specs), len(cols), figsize=(len(cols) * pw, len(specs) * pw * h0 / w0 + 0.3), squeeze=False)
    for i, (key, row) in enumerate(specs):
        r = int(row)
        c = load_cell(a.dump, a.base, key)
        U, S = c["U"], c["S"]
        rgb = (c["rgb"][r].astype(np.float32) / 255) if c["rgb"] is not None else np.zeros((*U.shape[1:], 3), np.float32)
        Ur, Sr, chd = U[r], S[r], None
        if a.hd:
            c_lo = load_cell(a.dump, a.base, key)      # thresholds and leak from the dump the numbers come from
            rgb, Ur, Sr, chd = load_hd(a.hd, a.base, key, r, c_lo)
        gray = rgb.mean(-1, keepdims=True)
        pale = np.repeat(0.35 * gray + 0.62, 3, -1)            # rejected or unlabelled: washed-out photograph
        gt = pale.copy()
        gt[Sr] = 0.45 * rgb[Sr] + 0.55 * GREEN
        gt[Ur] = 0.30 * rgb[Ur] + 0.70 * RED
        panels, notes = [rgb, gt], ["", f"{100 * U[r].sum() / max((U[r] | S[r]).sum(), 1):.0f}% single-view"]
        leaks = [None, None]
        for pat, _ in DETS:
            conf = c["conf"][pat]
            if conf is None:
                panels.append(pale.copy()); notes.append("not run"); leaks.append(None); continue
            lk, tau = leak(conf, U, S, a.keep)
            keep = (chd[pat] >= tau) if chd is not None else (conf[r] >= tau)
            P = pale.copy()
            m = keep & Sr; P[m] = 0.45 * rgb[m] + 0.55 * GREEN
            m = keep & Ur; P[m] = 0.22 * rgb[m] + 0.78 * RED
            panels.append(P)
            lv = float(((conf[r] >= tau) & U[r]).sum() / max(U[r].sum(), 1))   # always from the measured maps
            notes.append(f"leak {100 * lv:.0f}%"); leaks.append(lv)
        for j, P in enumerate(panels):
            ax[i, j].imshow(np.clip(P, 0, 1), interpolation="lanczos" if a.hd else None); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            ours = j == len(cols) - 1
            for sp in ax[i, j].spines.values():
                sp.set_linewidth(1.4 if ours else 0.5)
                sp.set_edgecolor(OURS if ours else "#c9ced6")
            if i == 0:
                ax[i, j].set_title(cols[j], fontsize=8, pad=5, color=OURS if ours else INK,
                                   fontweight="bold" if ours else "medium")
            if notes[j]:
                lv = leaks[j]                       # badge colour by the measured leak: low green, middle amber, high red
                face = "#3b4252" if lv is None else (GOOD if lv < 0.05 else AMBER if lv < 0.20 else BAD)
                ax[i, j].text(0.035, 0.05, notes[j], transform=ax[i, j].transAxes, fontsize=7, color="white",
                              fontweight="bold", bbox=dict(boxstyle="round,pad=0.32", facecolor=face, alpha=0.95, edgecolor="none"))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.006, right=0.994, top=0.93, bottom=0.008, wspace=0.035, hspace=0.045)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=300)
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=300 if a.hd else 200)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
