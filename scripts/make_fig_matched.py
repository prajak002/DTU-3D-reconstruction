"""Figure of the matched-patch protocol (P11-matched): three identical patches on one scene, one imagined and two
observed in a clean view, and what each detector says about them. Runs the backbone (GPU), one scene.

  python scripts/make_fig_matched.py --dtu <root> --scan 28 --seed 0 --evidence 1 --fill blur \
      --models vggt,vggt_reproj,vggt_oiamatchselfpatch,vggt_oiamatchself --out paper/figures/fig_matched

Panels, left to right: the filled reference view with the patches outlined (red: hidden in every view, imagined;
green: hidden here but visible in a clean view, observed), the clean evidence view with the observed sphere's
surface outlined, then one panel per detector with its score on the reference view and its AUROC on this scene for
U against the clean supported pixels (the benchmark's original number) and U against the filled-but-observed pixels
(the matched number). Writes <out>.pdf and <out>.png."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H  # noqa: E402

LABEL = {"": "native confidence", "_reproj": "reprojection, own cameras", "_oiamatchselfpatch": "per-image control",
         "_oiamatchself": "matched head, label-free", "_oiamatch": "matched head, GT placement", "_oiavisself": "support head"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtu", required=True)
    ap.add_argument("--scan", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radius", type=float, default=0.18)
    ap.add_argument("--evidence", default="1")
    ap.add_argument("--fill", default="blur")
    ap.add_argument("--models", default="vggt,vggt_reproj,vggt_oiamatchselfpatch,vggt_oiamatchself")
    ap.add_argument("--out", default=os.path.join("paper", "figures", "fig_matched"))
    a = ap.parse_args()
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = H.find_dtu_root((a.dtu,))
    data = H.load_scan(root, a.scan, H.select_views(10), 3)
    cfg = dict(H.DEFAULT_CFG)
    m = H.build_masks_matched(data, a.seed, a.radius, cfg, a.evidence)
    imgs = H.apply_fill(data["imgs"], m["hide"], a.fill, a.seed)
    out_hw = data["depths"].shape[1:]
    ref, ev = 0, m["evidence_views"][0]
    h, w = out_hw
    up = lambda x: cv2.resize(x.astype(np.uint8), (imgs.shape[2], imgs.shape[1]), interpolation=cv2.INTER_NEAREST) > 0
    models = a.models.split(",")
    scores, aur = {}, {}
    for name in models:
        model = H.make_model(name)
        d, c = model.predict(imgs, tuple(cfg["model_hw"]), out_hw)
        scores[name] = c[ref]
        mt = H.compute_metrics_matched(d, c, data["depths"], m, cfg)
        aur[name] = (mt["auroc"], mt["auroc_m"])
        if hasattr(model, "free"):
            model.free()
    n = 2 + len(models)
    fig, ax = plt.subplots(1, n, figsize=(2.75 * n, 3.0))
    ax[0].imshow(imgs[ref]); ax[0].set_title(f"reference view, {a.fill} fill" + "\nred: imagined, green: observed", fontsize=9)
    ax[0].contour(up(m["U"][ref]), levels=[0.5], colors="#d62728", linewidths=1.4)
    ax[0].contour(up(m["Sm"][ref]), levels=[0.5], colors="#2ca02c", linewidths=1.4)
    ax[1].imshow(data["imgs"][ev]); ax[1].set_title(f"clean view {ev}" + "\nthe green surface is visible here", fontsize=9)
    # the observed sphere's surface in the evidence view: pixels of view ev that see the Sm pixels of the reference view
    seen_m, _, _ = H.support_matrix(data["depths"], data["K_dep"], data["E"])
    Pw = H.backproject(data["depths"][ref], data["K_dep"][ref], data["E"][ref])
    u, v, z = H.project(Pw[m["Sm"][ref]], data["K_dep"][ev], data["E"][ev])
    mk = np.zeros((h, w), bool)
    ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    mk[np.round(v[ok]).astype(int), np.round(u[ok]).astype(int)] = True
    mk = cv2.dilate(mk.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ax[1].contour(up(mk), levels=[0.5], colors="#2ca02c", linewidths=1.4, linestyles="--")
    for k, name in enumerate(models):
        s = scores[name]
        lo, hi = np.quantile(s, 0.02), np.quantile(s, 0.98)
        ax[2 + k].imshow(np.clip((s - lo) / (hi - lo + 1e-9), 0, 1), cmap="viridis")
        ax[2 + k].contour(m["U"][ref].astype(float), levels=[0.5], colors="#d62728", linewidths=1.0)
        ax[2 + k].contour(m["Sm"][ref].astype(float), levels=[0.5], colors="#2ca02c", linewidths=1.0)
        suf = name.split("_", 1)[1] if "_" in name else ""
        ax[2 + k].set_title(f"{LABEL.get('_' + suf if suf else '', name)}\nAUROC {aur[name][0]:.2f} vs clean, {aur[name][1]:.2f} matched", fontsize=8.5)
    for x in ax:
        x.axis("off")
    fig.text(0.5, 0.01, "red: hidden in every view (imagined).  green: hidden in this view, visible in a clean view (observed, filled). "
             "Same fill, size and depth; a per-image score cannot tell them apart.", ha="center", fontsize=8.5)
    plt.tight_layout(rect=(0, 0.06, 1, 0.98))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out + ".pdf"); fig.savefig(a.out + ".png", dpi=150)
    print("->", a.out, {k: tuple(round(x, 3) for x in v) for k, v in aur.items()})


if __name__ == "__main__":
    main()
