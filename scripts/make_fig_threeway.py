"""Qualitative figure: the support head's observed / abstain / imagined map next to native confidence, on a DTU
scan with an inpainted hidden region and on ETH3D building-scale scenes (relief, facade, courtyard), clean and with
a hidden region. Stands in for the heritage study of issue #6 with laser-scanned heritage-type scenes.

Usage (GPU): python scripts/make_fig_threeway.py --dtu <dtu root> --eth3d <preprocessed eth3d root> --ckpt checkpoints
Writes paper/figures/fig10_threeway.png/.pdf and results/P3-figs/threeway_rows.json (the per-panel numbers).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H
import co3d
import oia_head as O

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLORS = np.array([[46, 160, 67], [237, 178, 44], [201, 48, 44]], np.float32) / 255.0   # observed, abstain, imagined


def three_way_rgb(score, lo=0.3, hi=0.7):
    tw = O.three_way(score, lo, hi)
    return COLORS[tw]


def panel_set(wrap_native, wrap_head, data, masks, fill, seed, cfg, view=0):
    imgs = data["imgs"] if masks is None else H.apply_fill(data["imgs"], masks["hide"], fill, seed)
    out_hw = data["depths"].shape[1:]
    model_hw = tuple(data.get("model_hw", cfg["model_hw"]))
    d_n, c_n = wrap_native.predict(imgs, model_hw, out_hw)
    d_h, s_h = wrap_head.predict(imgs, model_hw, out_hw)
    row = dict(scan=str(data["scan"]), fill=fill if masks is not None else "clean")
    if masks is not None:
        U, S = masks["U"], masks["S"]
        row.update(native_auroc=float(H.auroc(c_n[S], c_n[U])), head_auroc=float(H.auroc(s_h[S], s_h[U])),
                   head_imagined_in_U=float((s_h[U] <= 0.3).mean()), head_observed_in_S=float((s_h[S] >= 0.7).mean()))
    else:
        row.update(head_observed_frac=float((s_h[data["depths"] > 0] >= 0.7).mean()),
                   head_imagined_frac=float((s_h[data["depths"] > 0] <= 0.3).mean()))
    return dict(img=imgs[view], conf=np.log(c_n[view] + 1e-6), score=s_h[view], tw=three_way_rgb(s_h[view]),
                U=None if masks is None else masks["U"][view], row=row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtu", required=True)
    ap.add_argument("--eth3d", default=None)
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints"))
    ap.add_argument("--base", default="vggt")
    ap.add_argument("--scenes", default="relief,facade,courtyard")
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "figures", "fig10_threeway"))
    a = ap.parse_args()
    os.environ.setdefault("OIA_CKPT_DIR", a.ckpt)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    native = H.make_model(a.base)
    head = H.make_model(f"{a.base}_oia")
    cfg = dict(H.DEFAULT_CFG)
    sets = []
    root = H.find_dtu_root((a.dtu,))
    d = H.load_scan(root, 3, H.select_views(10), 3)
    m = H.build_masks(d, 0, 0.18, cfg)
    sets.append(("DTU scan 3, inpainted hidden region", panel_set(native, head, d, m, "inpaint", 0, cfg)))
    if a.eth3d:
        eroot = co3d.find_co3d_root((a.eth3d,))
        ecfg = {**cfg, "dataset": "co3d"}
        for sc in a.scenes.split(","):
            if sc not in co3d.list_co3d_scenes(eroot):
                continue
            dd = co3d.load_co3d_scene(eroot, sc, H.select_views(10, total=co3d.n_frames(eroot, sc)))
            sets.append((f"ETH3D {sc}, clean", panel_set(native, head, dd, None, None, 0, ecfg)))
            try:
                mm = H.build_masks(dd, 0, 0.18, ecfg)
                sets.append((f"ETH3D {sc}, inpainted hidden region", panel_set(native, head, dd, mm, "inpaint", 0, ecfg)))
            except RuntimeError as e:
                print(f"[fig] {sc}: no sphere fits ({e})")
    n = len(sets)
    fig, ax = plt.subplots(n, 4, figsize=(14, 2.9 * n))
    ax = np.atleast_2d(ax)
    for i, (title, p) in enumerate(sets):
        ax[i, 0].imshow(p["img"]); ax[i, 0].set_title(title, fontsize=9, loc="left")
        ax[i, 1].imshow(p["conf"], cmap="magma"); ax[i, 1].set_title("native log-confidence", fontsize=9)
        ax[i, 2].imshow(p["score"], cmap="viridis", vmin=0, vmax=1); ax[i, 2].set_title("support head: 1 - P(imagined)", fontsize=9)
        ax[i, 3].imshow(p["tw"]); ax[i, 3].set_title("observed / abstain / imagined", fontsize=9)
        if p["U"] is not None:
            for j in (1, 2, 3):
                ax[i, j].contour(p["U"].astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
        for j in range(4):
            ax[i, j].axis("off")
    fig.legend(handles=[Patch(color=COLORS[0], label="observed"), Patch(color=COLORS[1], label="abstain"),
                        Patch(color=COLORS[2], label="imagined")], loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out + ".png", dpi=150); fig.savefig(a.out + ".pdf")
    rows = [p["row"] | {"panel": t} for t, p in sets]
    os.makedirs(os.path.join(ROOT, "results", "P3-figs"), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, "results", "P3-figs", "threeway_rows.json"), "w"), indent=1)
    for r in rows:
        print(r)
    print("->", a.out + ".png")


if __name__ == "__main__":
    main()
