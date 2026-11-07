"""Qualitative provenance figure (run P10-prov-dump): for one scene, which input views the head says corroborate a
pixel, against the ground truth.

  python scripts/make_fig_prov.py [--dump results/P10-prov-dump] [--out paper/figures/fig_prov] [--model vggt_oiaprovself]

Layout, one row per dataset found under the dump folder: the reference view with three query pixels (a supported
pixel seen by two or more other views, a pixel seen by exactly one, a single-view pixel); the ground-truth number of
other views that see each pixel and the head's predicted number (sum of its per-view probabilities); then, for each
query, one bar per other input view with the head's probability, the bar drawn green where that view does see the
point and red where it does not (hatched where the ground truth is undecided). Rendered from the released per-view
dumps; nothing is typed by hand."""
import argparse
import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = {"dtu": "DTU", "co3d": "CO3Dv2", "eth3d": "ETH3D", "imc": "Phototourism"}


def pick_queries(seen, unknown, U, S, valid, rng):
    """Three (y, x) pixels of view 0: seen by >= 2 views (in S), by exactly 1, by none (in U)."""
    cnt = seen[0].sum(0)
    und = unknown[0].any(0)
    cands = [(S[0] & (cnt >= 2)), (valid[0] & (cnt == 1) & ~und), (U[0])]
    out = []
    for c in cands:
        ys, xs = np.nonzero(c)
        if len(ys) == 0:
            out.append(None); continue
        k = rng.integers(len(ys))
        out.append((int(ys[k]), int(xs[k])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(ROOT, "results", "P10-prov-dump"))
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "figures", "fig_prov"))
    ap.add_argument("--model", default="vggt_oiaprovself")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for ds in ("dtu", "co3d", "eth3d", "imc"):
        files = sorted(glob.glob(os.path.join(a.dump, ds, a.model, "*.npz")))
        if files:
            rows.append((ds, files[0]))
    if not rows:
        print("no provenance dumps under", a.dump); return
    rng = np.random.default_rng(a.seed)
    fig = plt.figure(figsize=(16, 3.6 * len(rows)))
    gs = fig.add_gridspec(len(rows), 6, width_ratios=[1.6, 1.3, 1.3, 1, 1, 1], wspace=0.08, hspace=0.35)
    marks = ["o", "s", "^"]
    titles = ["seen by two or more views", "seen by one view", "seen by no other view"]
    for r, (ds, f) in enumerate(rows):
        z = np.load(f)
        P, seen, unknown, rgb, gt, U, S = z["P"].astype(np.float32), z["seen"], z["unknown"], z["rgb"], z["gt"], z["U"], z["S"]
        N = P.shape[0]
        valid = gt > 0
        qs = pick_queries(seen, unknown, U, S, valid, rng)
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(rgb[0]); ax.set_xticks([]); ax.set_yticks([])
        for q, m, c in zip(qs, marks, ("#1E8A7A", "#B86E00", "#C8102E")):
            if q is not None:
                ax.plot(q[1], q[0], m, ms=11, mfc="none", mec=c, mew=2.2)
        ax.set_title(f"{NAMES[ds]}: reference view, {N} input views", fontsize=10, loc="left")
        cnt_gt = seen[0].sum(0).astype(float); cnt_gt[~valid[0]] = np.nan
        cnt_pr = P[0].sum(0); cnt_pr[~valid[0]] = np.nan
        vmax = max(1, N - 1)
        for k, (img, t) in enumerate(((cnt_gt, "views that see the pixel (ground truth)"), (cnt_pr, "views the head attributes (sum of probabilities)"))):
            ax = fig.add_subplot(gs[r, 1 + k])
            im = ax.imshow(img, cmap="viridis", vmin=0, vmax=vmax); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(t, fontsize=9, loc="left")
            if k == 1:
                plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        for k, (q, t) in enumerate(zip(qs, titles)):
            ax = fig.add_subplot(gs[r, 3 + k])
            if q is None:
                ax.axis("off"); ax.set_title(t + ": none in this view set", fontsize=9); continue
            y, x = q
            others = [j for j in range(N) if j != 0]
            p = [P[0, j, y, x] for j in others]
            sees = [bool(seen[0, j, y, x]) for j in others]
            und = [bool(unknown[0, j, y, x]) for j in others]
            cols = ["#1E8A7A" if s_ else "#C8102E" for s_ in sees]
            bars = ax.bar(range(len(others)), p, color=cols, edgecolor="#17201F", linewidth=0.6)
            for b, u in zip(bars, und):
                if u:
                    b.set_hatch("//"); b.set_alpha(0.5)
            ax.set_ylim(0, 1); ax.set_xticks(range(len(others))); ax.set_xticklabels([f"v{j}" for j in others], fontsize=7)
            ax.axhline(0.5, color="#4C5A58", lw=0.6, ls=":")
            ax.set_title(f"{marks[k]}  {t}", fontsize=9, loc="left")
            ax.set_ylabel("P(view corroborates)" if k == 0 else "", fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
    fig.text(0.5, 0.005, "bar colour: ground truth, green = that view does see the point, red = it does not (hatched: undecided); the head never saw these scenes",
             ha="center", fontsize=9, color="#4C5A58")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=170, bbox_inches="tight")
    print("->", a.out, [d for d, _ in rows])


if __name__ == "__main__":
    main()
