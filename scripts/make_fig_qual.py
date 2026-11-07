"""The qualitative figure of the paper (one row per dataset) from the per-pixel dumps, no model run.

  python scripts/make_fig_qual.py --dump results/P6-dump --out paper/figures/fig_qual.pdf \
      --rows "dtu:28__s0__v5:3" "co3d:bench_415_57112_110099__s0__v5:3" "eth3d:electro__s0__v5:0" "imc:sagrada_familia__s0__v5:2"

Each row: the photograph, ground-truth support (single-view red, supported green, unlabelled dark), native
confidence, reprojection consistency, the per-image control, the label-free head's observed / abstain / imagined
bands (0.7 / 0.3). Also writes a PNG next to the PDF and per-panel PNGs under <out dir>/qual_panels/ for the site."""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join("results", "P6-dump"))
    ap.add_argument("--rows", nargs="+", required=True, help="dataset:cell_key:view_index")
    ap.add_argument("--base", default="vggt")
    ap.add_argument("--out", default=os.path.join("paper", "figures", "fig_qual.pdf"))
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    labels = {"dtu": "DTU", "co3d": "CO3Dv2", "eth3d": "ETH3D", "imc": "Phototourism"}
    cols = ["photograph", "ground-truth support", "native confidence", "reprojection consistency", "per-image control", "label-free head: observed / abstain / imagined"]
    specs = [r.split(":") for r in a.rows]
    fig, ax = plt.subplots(len(specs), len(cols), figsize=(len(cols) * 2.55, len(specs) * 1.95), squeeze=False)
    pdir = os.path.join(os.path.dirname(a.out) or ".", "qual_panels"); os.makedirs(pdir, exist_ok=True)
    for i, (ds, key, row) in enumerate(specs):
        r = int(row)
        L = np.load(os.path.join(a.dump, ds, "labels", key + ".npz"))
        U, S = L["U"][r], L["S"][r]
        rgb = L["rgb"][r] if "rgb" in L.files else np.zeros((*U.shape, 3), np.uint8)
        maps = {}
        for d in (a.base, f"{a.base}_reproj", f"{a.base}_oiavispatch", f"{a.base}_oiavisself"):
            p = os.path.join(a.dump, ds, d, key + ".npz")
            maps[d] = np.load(p)["conf"][r].astype(np.float32) if os.path.exists(p) else None
        lab = np.zeros((*U.shape, 3), np.float32) + 0.15; lab[S] = (0.2, 0.65, 0.3); lab[U] = (0.85, 0.2, 0.2)
        panels = [rgb, lab]
        for d in (a.base, f"{a.base}_reproj", f"{a.base}_oiavispatch"):
            m = maps[d]
            if m is None:
                panels.append(np.zeros_like(lab)); continue
            lo, hi = np.percentile(m, 2), np.percentile(m, 98)
            panels.append(plt.get_cmap("viridis")(np.clip((m - lo) / (hi - lo + 1e-9), 0, 1))[..., :3])
        s = maps[f"{a.base}_oiavisself"]
        band = np.zeros_like(lab)
        if s is not None:
            band[s > 0.7] = (0.2, 0.65, 0.3); band[(s <= 0.7) & (s >= 0.3)] = (0.95, 0.75, 0.15); band[s < 0.3] = (0.85, 0.2, 0.2)
            band = 0.35 * rgb.astype(np.float32) / 255 + 0.65 * band
        panels.append(band)
        for j, P in enumerate(panels):
            ax[i, j].imshow(P if P.dtype == np.uint8 else np.clip(P, 0, 1)); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(cols[j], fontsize=8)
            img = Image.fromarray((P if P.dtype == np.uint8 else (np.clip(P, 0, 1) * 255).astype(np.uint8)))
            img.save(os.path.join(pdir, f"{ds}_{key}_r{r}_{j}.png"))
        fr = float(U.sum() / max((U | S).sum(), 1))
        ax[i, 0].set_ylabel(f"{labels.get(ds, ds)}\n{fr * 100:.0f}% single-view", fontsize=8)
    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight"); fig.savefig(a.out.rsplit(".", 1)[0] + ".png", dpi=110, bbox_inches="tight")
    print("->", a.out)


if __name__ == "__main__":
    main()
